"""
HexapodMjxEnv — JAX/MJX-native PhantomX hexapod RL environment.

Pure functional API: reset(rng, gait_scale=0.0) -> EnvState; step(state,
action) -> EnvState. All math runs on the GPU under MJX. Compatible with
Brax-style training (see envs.hexapod_env_jax.brax_wrapper for the Brax
adapter — added in Phase 4).

Mirrors the SubprocVecEnv `HexapodEnv` reward + termination + obs:
  * mjx physics, scaffold-mix action targets
  * tracking + survive bonus, action-rate, body-angvel, drift, speed-track
  * foot-contact shaping (sliding, excess contact, airborne, short_lifts)
  * foot-position deviation (post-step actual vs scaffold intent)
  * no-progress termination (rolling speed-along-cmd ring buffer)
  * BC target in metrics
  * gait_scale carried in EnvState (set_gait_scale helper for the
    trainer's curriculum hook to fade it between rollouts without
    recompiling the step function)

Stage is fixed to 3 (full motion cmd mask) for now.
"""

from typing import NamedTuple
import math

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from gait import controller_jax as gait_jax


# ----------------------------------------------------------------------------
# Stage 3 cmd mask + reward-track mask. Translation, yaw, height, width,
# pitch, roll, shift_x, shift_y all active. width is now tracked too (via
# actual stance width measured from foot positions, see _compute_reward);
# shifts still excluded because we don't track body-frame translation.
# ----------------------------------------------------------------------------
_REWARD_TRACK_MASK = jnp.array([1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=jnp.float32)

# Curriculum cmd masks. Index = cmd slot. 1 = sampled, 0 = clamped to zero.
# Pick one in make_env_params(cmd_mask_name=...).
#
#                          [vx, vy, wz, p, r, h, w, sx, sy]
CMD_MASKS = {
    # stage1: translation only — body stays level (drift_pen enforces),
    #         at ref_body_z height (height_track_pen enforces). Use as
    #         the simplest from-scratch curriculum starting point.
    "stage1": jnp.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32),
    # stage2: + yaw + height + width.
    "stage2": jnp.array([1, 1, 1, 0, 0, 1, 1, 0, 0], dtype=jnp.float32),
    # stage3: + pitch + roll + body shifts. (Full motion — the previous default.)
    "stage3": jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.float32),
    # paper: (vx, vy, wz) only — matches Chen et al. AMP setup. Used for the
    # paper-pure baseline (v4+). Pitch/roll/height/width are deferred to a
    # later stage; the discriminator stays cmd-invariant.
    "paper": jnp.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.float32),
    # paper_stance: paper + height + width. v6 adds user-controllable stance
    # height (cmd[5]) and stance width (cmd[6]) so the policy responds to
    # the D-pad axes. Pitch/roll/shifts still deferred. Priors must span
    # height/width so the discriminator stays cmd-invariant on those axes.
    "paper_stance": jnp.array([1, 1, 1, 0, 0, 1, 1, 0, 0], dtype=jnp.float32),
}

# v20 (2026-05-10) — height-conditional stance width envelope. Coefficients
# imported from envs/stance_envelope.py (single source of truth shared with
# scripts/demo_phases.py and tools/watch_scaffold.py). When the verified
# bounds change, edit stance_envelope.py — env, watch tests, and tuner all
# pick up the new values automatically.
from envs import stance_envelope as _stance
MAX_DW_INTERCEPT = jnp.float32(_stance.MAX_DW_INTERCEPT)
MAX_DW_SLOPE     = jnp.float32(_stance.MAX_DW_SLOPE)
MIN_DW_INTERCEPT = jnp.float32(_stance.MIN_DW_INTERCEPT)
MIN_DW_SLOPE     = jnp.float32(_stance.MIN_DW_SLOPE)

# Tibia body names in canonical leg order. Foot tip is computed as
# tibia-world + tibia-rotation @ (0.134, 0.031, 0). This matches the gym
# env's _FOOT_LOCAL_OFFSET and the Controller's calibrated FOOT_TIP_LOCAL
# (geom_pos of the foot sphere = sphere center). Using sphere bottom
# (z=-0.008) instead would be more physically correct for "ground contact
# point" but creates an 8mm systematic offset relative to the scaffold's
# intended foot position (which the Controller computes at the sphere
# center). For reward-parity with the gym env we use the sphere center
# everywhere, with FOOT_CONTACT_Z compensating in the contact threshold.
_LEG_NAMES         = ("RR", "RM", "RF", "LR", "LM", "LF")
_TIBIA_BODY_NAMES  = tuple(f"tibia_{n}" for n in _LEG_NAMES)
_FOOT_LOCAL_OFFSET = jnp.array([0.134, 0.031, 0.0], dtype=jnp.float32)

# No-progress ring buffer length — must be a JIT-compile-time constant.
NO_PROGRESS_WINDOW = 400


# ----------------------------------------------------------------------------
# Static config — NamedTuple of JAX-compatible values.
# ----------------------------------------------------------------------------
class EnvParams(NamedTuple):
    mjx_model:         mjx.Model
    gait:              gait_jax.GaitParams
    gait_neutral_pose: jnp.ndarray   # (18,) widened-stance neutral pose
    err_inv_scales:    jnp.ndarray   # (9,)  per-cmd-slot tracking error scale
    reward_track_mask: jnp.ndarray   # (9,)
    cmd_mask:          jnp.ndarray   # (9,)  stage-3 mask
    neutral_lateral_spread: jnp.ndarray  # () baseline mean(|foot_y|) at width=0
    cmd_sample_ranges: jnp.ndarray   # (9, 2) lo/hi per slot
    spawn_qpos:        jnp.ndarray   # (nq,) spawn pose
    base_body_id:      int           # body id of "base"
    tibia_body_ids:    jnp.ndarray   # (6,) int32 — foot tip = xpos[tib] + R[tib] @ offset
    foot_local_offset: jnp.ndarray   # (3,) tibia-local foot-tip offset
    # Reward weights and gates.
    # DEPRECATED weights (post-2026-05-07 paper-aligned redesign). Kept as
    # fields for backward compat with any code that constructs EnvParams
    # directly, but defaulted to 0 so they don't contribute to the reward.
    survive_bonus:     float = 0.0      # DEPRECATED — paper has no constant bonus
    action_rate_w:     float = 0.01     # KEPT — smoothness penalty (matches paper)
    angvel_w:          float = 0.0      # DEPRECATED — replaced by z_vel_w + AMP discriminator
    yaw_drift_w:       float = 0.0      # DEPRECATED — redundant with gaussian tracking
    pitch_drift_w:     float = 0.0      # DEPRECATED — redundant with gaussian tracking
    roll_drift_w:      float = 0.0      # DEPRECATED — redundant with gaussian tracking
    yaw_gate:          float = 0.05
    tilt_gate:         float = math.radians(3.0)
    speed_track_w:     float = 0.0      # DEPRECATED — redundant with gaussian tracking
    residual_scale_max: float = 0.80
    # Foot-contact shaping. Most of these are now off — action_rate and
    # angvel_pen already discourage twitchy/jerky motion, so we don't
    # need a second layer of "anti-cheat" contact penalties holding the
    # gait close to the scaffold's expected pattern.
    foot_contact_z:    float = 0.005
    sliding_w:         float = 0.0      # DEPRECATED — AMP handles contact patterns implicitly
    sliding_deadzone:  float = 0.010
    excess_contact_w:  float = 0.0      # was 0.05 — disabled. Slow walking
                                         #   gaits with 4+ supports are fine.
    airborne_w:        float = 0.0      # was 0.02 — disabled. Bounding /
                                         #   hopping gaits should be reachable.
    short_contact_w:   float = 0.0      # was 0.1 — disabled. Anti-cheat for a
                                         #   problem we're not actually seeing.
    min_contact_steps: int   = 30       # unused while short_contact_w=0
    foot_dev_w:        float = 0.0      # was 0.3 — disabled. Bot is free to
                                         #   discover any foot trajectory; the
                                         #   scaffold is a hint via obs but no
                                         #   longer enforced via reward.
    # Sharp linear height tracking — parallel to speed_track_w, gives a
    # tighter "actually hit the commanded height" gradient than the gaussian
    # tracking term alone.
    height_track_w:    float = 0.0      # DEPRECATED — replaced by z_vel_w in
                                         #   the paper-aligned reward redesign
                                         #   (2026-05-07). Kept at 0 for back-compat.

    # Paper-aligned reward (post-2026-05-07 redesign matching the SJTU AMP
    # paper's reward shape). Coefficients match Chen et al. Table I exactly
    # for the v4 baseline (2026-05-08): we previously over-weighted the
    # penalties by 200-500× which destabilized AMP training at style_weight=1.0.
    # The AMP discriminator's style reward is added separately in HexapodAMPEnv.
    z_vel_w:               float = 1.0     # paper: 1.0 × v_z^2
    body_angvel_xy_w:      float = 0.08    # paper: 0.08 × ||ω_xy||^2  (NEW v4)
    joint_torque_w:        float = 2e-6    # paper: 2e-6 × ||τ||^2
    joint_vel_limit:       float = 6.18    # AX-12A no-load max angular velocity (rad/s)
    joint_vel_limit_w:     float = 0.5     # paper: 0.5 × ||max(|q̇|-q̇_lim,0)||^2
    joint_torque_limit:    float = 1.5     # AX-12A stall torque (N·m)
    joint_torque_limit_w:  float = 0.05    # paper: 0.05 × ||max(|τ|-τ_lim,0)||^2
    limit_deadband_frac:   float = 1.0     # paper: penalty fires only at the actual limit (no deadband)
    # Foot contact force limit (NEW v5). Paper: -0.1 × ||max(|f|-f_lim, 0)||²
    # Bot mass = 1.64 kg, weight = 16 N. Tripod stance ≈ 5.4 N/foot statically.
    # 30 N gives room for ~3× dynamic loading; only fires on actual stomping.
    foot_force_limit:      float = 30.0    # N — per-foot force magnitude ceiling
    foot_force_limit_w:    float = 0.1     # paper: 0.1 coefficient
    # Scaffold yaw-trim coefficients (NEW for prior gen — defaults 0 so
    # training behavior is unchanged). When non-zero, they're added to
    # cmd[2] BEFORE the scaffold computes joint targets, but state.cmd
    # (visible to the obs / tracking reward) is unchanged. This lets us
    # generate drift-free priors without lying to the policy about cmd.
    # Tuned for our specific scaffold by tools/test_scaffold_drift.py:
    #   wz_trim_vx=0.005, wz_trim_vy_abs=-0.012 reduces forward/back/strafe
    #   drift to <1° per 5m. Used by amp/prior_data.py.
    scaffold_wz_trim_vx:     float = 0.0   # cmd[2] += -k * cmd[0]
    scaffold_wz_trim_vy_abs: float = 0.0   # cmd[2] += -k * |cmd[1]|
    # Anti-drift penalties (v10). Linear penalty on |actual - commanded|
    # for yaw and lateral. Doesn't affect intentional turning / strafing
    # since the penalty is between actual and commanded values, not |actual|.
    # v16 (2026-05-10): drift penalties zeroed. The natural scaffold bias
    # is small enough to be visually invisible (verified via watch_scaffold
    # A/B), and the linear penalties were creating perverse incentives —
    # "don't move" got less penalty than "walk with the small natural bias,"
    # which combined with weak AMP signal pushed the policy off scaffold
    # toward degenerate "low-motion" gait. Tracking gaussian still
    # captures velocity errors; we don't need a separate drift term.
    yaw_drift_w:           float = 0.0     # was 5.0 (v11); 0.5 (v10)
    vy_drift_w:            float = 0.0     # was 0.5 (v10)
    # v25 (2026-05-13): EMA-filter motion components of tracking reward to
    # close the "wobble to match instantaneous velocity" gaming exploit.
    # alpha=0.05 at 200Hz → ~20-step (100ms) time constant. Jittering
    # averages to ~zero on this window so the policy can't fake velocity
    # tracking via high-frequency motion.
    linvel_ema_alpha:      float = 0.05
    # v25: gait-phase contact penalty — uses the bot's now-available motor
    # feedback to know which legs ARE loaded. Penalizes feet that should
    # be in stance per the tripod cycle but aren't (and vice versa). Max
    # mismatch = 6 legs × 1000 steps = 6000 per episode; w=0.02 caps
    # episodic contribution at ~120 — comparable to other penalty terms.
    contact_mismatch_w:    float = 0.02
    # Domain randomization at episode reset (NEW v10). Random perturbations
    # to initial joint angles + body z so the policy sees a broader range
    # of starting states (helps generalize, learn to recover from non-ideal
    # initial conditions). Set noise to 0 to disable.
    init_joint_noise_rad:  float = 0.087   # ~5° per joint
    init_body_z_noise_m:   float = 0.010   # ±10mm height perturbation
    # Termination dynamics.
    min_z:             float = 0.06
    max_tilt_sq:       float = 0.067
    tilt_dev_limit:    float = math.radians(5.0)
    tilt_dev_grace:    int   = 400
    no_progress_grace: int   = 200
    no_progress_frac:  float = 0.20
    ref_body_z:        float = 0.13
    speed_min_frac:    float = 0.40
    speed_max_frac:    float = 0.85
    # Dynamic-cmd within-episode (v9+). When enabled, the cmd evolves over
    # the episode: every K steps a new target is sampled and the cmd
    # smoothly interpolates over T steps before holding at the new target.
    # K and T are sampled per-event from the ranges below. Disabled by
    # default — preserves v8's static-cmd training behavior.
    cmd_dynamics_enabled:  bool  = False
    cmd_event_min_steps:   int   = 150   # 3 sec @ 50Hz minimum hold
    cmd_event_max_steps:   int   = 400   # 8 sec @ 50Hz maximum hold
    cmd_transition_min:    int   = 25    # 0.5 sec ramp minimum
    cmd_transition_max:    int   = 50    # 1.0 sec ramp maximum
    # Foot-space action mode: when step_foot() is used instead of step(),
    # the policy's 18-dim action is interpreted as a (6, 3) foot residual
    # in body frame. Each foot's body-frame target = scaffold's intent +
    # action × foot_residual_scale_max. IK then converts to joint targets.
    # 0.020 m = ±20 mm per axis at action=±1; smooth coordinated noise.
    foot_residual_scale_max: float = 0.020

    # Recovery curriculum (v22+, 2026-05-11) — mid-episode push impulses
    # applied to the base body so the policy learns to recover from
    # disturbances. Disabled by default. When enabled:
    #   - Every push_interval_min..push_interval_max steps, a horizontal
    #     force impulse is applied for push_duration_steps.
    #   - Direction is random; magnitude = max_push_force_N * magnitude_scale.
    #   - magnitude_scale follows a curriculum (low early, full late) set
    #     externally by the training script between segments.
    disturbance_enabled:       bool  = False
    push_interval_min_steps:   int   = 600       # 3 sec @ 200Hz
    push_interval_max_steps:   int   = 1600      # 8 sec @ 200Hz
    push_duration_steps:       int   = 10        # ~50 ms
    max_push_force_n:          float = 8.0       # full-curriculum force in newtons
    disturbance_magnitude_scale: float = 1.0     # 0.0 = no push, 1.0 = full magnitude


class EnvState(NamedTuple):
    mjx_data:           mjx.Data
    cmd:                jnp.ndarray   # (9,) — current cmd (interpolated when in transition)
    sim_time:           jnp.ndarray   # ()
    step_count:         jnp.ndarray   # () int32
    last_action:        jnp.ndarray   # (18,)
    last_feet_body:     jnp.ndarray   # (6, 3) — scaffold's foot intent at THIS step
    prev_foot_world:    jnp.ndarray   # (6, 3) — for finite-diff foot velocity
    foot_was_in_contact: jnp.ndarray  # (6,) bool — last step's contact state
    foot_contact_steps: jnp.ndarray   # (6,) int32 — consecutive steps in contact
    progress_buf:       jnp.ndarray   # (NO_PROGRESS_WINDOW,) ring buffer
    progress_idx:       jnp.ndarray   # () int32
    progress_full:      jnp.ndarray   # () bool
    # Dynamic-cmd state (used when EnvParams.cmd_dynamics_enabled = True).
    # When disabled, these fields exist but the step logic ignores them
    # and cmd stays at its initial sampled value.
    cmd_target:           jnp.ndarray  # (9,) — where cmd is heading
    cmd_delta_per_step:   jnp.ndarray  # (9,) — precomputed (target - start) / T
    transition_remaining: jnp.ndarray  # () int32 — steps left in current ramp
    event_remaining:      jnp.ndarray  # () int32 — steps until next target sample
    # Recovery-curriculum state (v22+). Force in WORLD frame applied to base
    # body. When push_remaining > 0, the force is applied each step via
    # mjx_data.xfrc_applied[base_body_id, 0:3].
    push_force_world:     jnp.ndarray  # (3,) — currently active push force (N)
    push_remaining:       jnp.ndarray  # () int32 — steps left in current push
    next_push_in:         jnp.ndarray  # () int32 — steps until next push event
    # v25 (2026-05-13): EMA-filtered motion velocities [vx_body, vy_body, wz].
    # Used in tracking reward to defeat the "wobble to fake velocity" exploit.
    linvel_ema:           jnp.ndarray  # (3,)
    gait_scale:         jnp.ndarray   # () float32 — externally mutable curriculum
    obs:                jnp.ndarray   # (78,)
    reward:             jnp.ndarray   # ()
    done:               jnp.ndarray   # () bool
    rng:                jax.Array
    metrics:            dict


# ----------------------------------------------------------------------------
# EnvParams construction
# ----------------------------------------------------------------------------
def make_env_params(model_path: str,
                    cmd_mask_name: str = "stage3",
                    cmd_dynamics_enabled: bool = False,
                    scaffold_wz_trim_vx: float = 0.0,
                    scaffold_wz_trim_vy_abs: float = 0.0,
                    disturbance_enabled: bool = False,
                    disturbance_magnitude_scale: float = 1.0) -> EnvParams:
    """Load MJCF, build mjx model, calibrate gait — done once at startup.

    `cmd_mask_name` selects which cmd slots are sampled at episode reset.
    `cmd_dynamics_enabled` (v9+): if True, cmd evolves within an episode
        with smooth ramp transitions between resampled targets (good for
        teaching the policy to handle live controller inputs).
    "stage1" = translation only (the curriculum starting point).
    "stage3" = full motion (translation + yaw + tilt + stance + shifts).
    """
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mjx_model = mjx.put_model(mj_model)
    gait = gait_jax.build_params(model_path, dtype=jnp.float32)

    np_ctrl = gait_jax.Controller(model_path)
    gait_neutral = jnp.asarray(np_ctrl.gait_neutral_pose, dtype=jnp.float32)
    max_speed    = float(np_ctrl.MAX_SPEED)
    max_yaw_rate = float(np_ctrl.MAX_YAW_RATE)

    FULL_HEIGHT_DEV = 0.025
    FULL_WIDTH_DEV  = 0.020   # ±20mm covers the ±15mm cmd range with margin
    FULL_PITCH_DEV  = math.radians(15)
    FULL_ROLL_DEV   = math.radians(15)
    err_inv_scales = jnp.array([
        1.0 / max_speed, 1.0 / max_speed, 1.0 / max_yaw_rate,
        1.0 / FULL_PITCH_DEV, 1.0 / FULL_ROLL_DEV,
        1.0 / FULL_HEIGHT_DEV, 1.0 / FULL_WIDTH_DEV, 0.0, 0.0,
    ], dtype=jnp.float32)

    # Compute baseline mean(|foot_y|) at neutral stance (cmd=0).
    # This is the reference from which width_delta is measured. Used by
    # _compute_reward to convert current foot positions to a width error.
    _zero_cmd = np.zeros(9, dtype=np.float64)
    _, feet_neutral_body = np_ctrl.predict_with_feet(_zero_cmd, t=0.0)
    neutral_lateral_spread = jnp.float32(np.mean(np.abs(feet_neutral_body[:, 1])))

    # Stance-shape ranges (v20 2026-05-10): expanded ~4× for height and ~7×
    # for max width based on user kinematic verification on the mesh model
    # (tools/watch_scaffold.py --interactive). Width is dh-conditional —
    # see DH_TO_DW_* below for the linear envelope formula.
    cmd_sample_ranges = jnp.array([
        [-1.0, 1.0],                                              # vx (placeholder)
        [-1.0, 1.0],                                              # vy (placeholder)
        [-max_yaw_rate, max_yaw_rate],                            # wz
        [-math.radians(10), math.radians(10)],                    # pitch
        [-math.radians(15), math.radians(15)],                    # roll
        [_stance.DH_MIN, _stance.DH_MAX],                         # height (v20: was ±0.020)
        [_stance.DW_JOINT_MIN, _stance.DW_JOINT_MAX],             # width (joint range; actual sampler clips per dh)
        [-0.030, 0.030],                                          # shift_x
        [-0.030, 0.030],                                          # shift_y
    ], dtype=jnp.float32)

    nq = mj_model.nq
    spawn_qpos = jnp.zeros(nq, dtype=jnp.float32)
    spawn_qpos = spawn_qpos.at[2].set(0.18)
    spawn_qpos = spawn_qpos.at[3].set(1.0)
    spawn_qpos = spawn_qpos.at[7:25].set(gait_neutral)

    tibia_body_ids = jnp.asarray(
        [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
         for n in _TIBIA_BODY_NAMES], dtype=jnp.int32,
    )
    base_body_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base"))

    return EnvParams(
        cmd_dynamics_enabled=cmd_dynamics_enabled,
        disturbance_enabled=disturbance_enabled,
        disturbance_magnitude_scale=disturbance_magnitude_scale,
        scaffold_wz_trim_vx=scaffold_wz_trim_vx,
        scaffold_wz_trim_vy_abs=scaffold_wz_trim_vy_abs,
        mjx_model=mjx_model,
        gait=gait,
        gait_neutral_pose=gait_neutral,
        err_inv_scales=err_inv_scales,
        reward_track_mask=_REWARD_TRACK_MASK,
        cmd_mask=CMD_MASKS[cmd_mask_name],
        neutral_lateral_spread=neutral_lateral_spread,
        cmd_sample_ranges=cmd_sample_ranges,
        spawn_qpos=spawn_qpos,
        base_body_id=base_body_id,
        tibia_body_ids=tibia_body_ids,
        foot_local_offset=_FOOT_LOCAL_OFFSET,
    )


# ----------------------------------------------------------------------------
# Cmd sampling — polar translation + per-slot uniform.
# ----------------------------------------------------------------------------
def _sample_cmd(params: EnvParams, rng: jax.Array) -> jnp.ndarray:
    rngs = jax.random.split(rng, 9)
    heading = jax.random.uniform(rngs[0], (), minval=0.0, maxval=2.0 * jnp.pi)
    max_speed = 1.0 / params.err_inv_scales[0]
    speed_min = params.speed_min_frac * max_speed
    speed_max = params.speed_max_frac * max_speed
    magnitude = jax.random.uniform(rngs[1], (), minval=speed_min, maxval=speed_max)
    vx = magnitude * jnp.cos(heading)
    vy = magnitude * jnp.sin(heading)

    # Slots 2..4 (wz, pitch, roll) and 7..8 (sx, sy) sample uniformly from
    # cmd_sample_ranges. dh (slot 5) is sampled here too. dw (slot 6) is
    # sampled CONDITIONAL on dh via the linear envelope (v20+) — same RNG
    # split index so randomness stays deterministic per env-reset.
    wz    = jax.random.uniform(rngs[4], (),
                               minval=params.cmd_sample_ranges[2, 0],
                               maxval=params.cmd_sample_ranges[2, 1])
    pitch = jax.random.uniform(rngs[5], (),
                               minval=params.cmd_sample_ranges[3, 0],
                               maxval=params.cmd_sample_ranges[3, 1])
    roll  = jax.random.uniform(rngs[6], (),
                               minval=params.cmd_sample_ranges[4, 0],
                               maxval=params.cmd_sample_ranges[4, 1])
    dh    = jax.random.uniform(rngs[7], (),
                               minval=params.cmd_sample_ranges[5, 0],
                               maxval=params.cmd_sample_ranges[5, 1])
    # dh-conditional dw envelope: linear lower-bound fit verified in
    # tools/watch_scaffold.py --interactive on the mesh model.
    max_dw_at_dh = MAX_DW_INTERCEPT + MAX_DW_SLOPE * dh
    min_dw_at_dh = MIN_DW_INTERCEPT + MIN_DW_SLOPE * dh
    dw    = jax.random.uniform(rngs[8], (), minval=min_dw_at_dh, maxval=max_dw_at_dh)
    # sx, sy use spare hashes off rngs[0] / rngs[1] to avoid burning more splits.
    sx_key, sy_key = jax.random.split(rngs[0])
    sx = jax.random.uniform(sx_key, (),
                            minval=params.cmd_sample_ranges[7, 0],
                            maxval=params.cmd_sample_ranges[7, 1])
    sy = jax.random.uniform(sy_key, (),
                            minval=params.cmd_sample_ranges[8, 0],
                            maxval=params.cmd_sample_ranges[8, 1])
    cmd = jnp.array([vx, vy, wz, pitch, roll, dh, dw, sx, sy])
    return cmd * params.cmd_mask


# ----------------------------------------------------------------------------
# Kinematic helpers
# ----------------------------------------------------------------------------
def _body_frame_linvel(qpos, qvel):
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)
    c = jnp.cos(yaw); s = jnp.sin(yaw)
    vx_w, vy_w, vz_w = qvel[0], qvel[1], qvel[2]
    return jnp.array([c * vx_w + s * vy_w,
                      -s * vx_w + c * vy_w,
                      vz_w])


def _body_pitch_roll(qpos):
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    sinp  = jnp.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = -jnp.arcsin(sinp)
    sinr  = 2.0 * (qw * qx + qy * qz)
    cosr  = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll  = jnp.arctan2(sinr, cosr)
    return pitch, roll


def _foot_world_positions(params: EnvParams, mjx_data) -> jnp.ndarray:
    """(6, 3) foot-tip world positions, computed as
    `xpos[tibia] + R[tibia] @ foot_local_offset`. Avoids reading geom_xpos
    (which can trigger extra kinematics) and matches the gym env's path
    so reward semantics stay identical."""
    tibia_pos = mjx_data.xpos[params.tibia_body_ids]                       # (6, 3)
    tibia_R   = mjx_data.xmat[params.tibia_body_ids].reshape(6, 3, 3)
    return tibia_pos + jnp.einsum('lij,j->li', tibia_R, params.foot_local_offset)


def _foot_body_positions(params: EnvParams, mjx_data) -> jnp.ndarray:
    """(6, 3) foot-tip body-frame positions. Equivalent to the gym env's
    body-frame foot-target target via R_body.T @ (foot_world - body_pos)."""
    foot_world = _foot_world_positions(params, mjx_data)
    body_pos   = mjx_data.xpos[params.base_body_id]
    R_body     = mjx_data.xmat[params.base_body_id].reshape(3, 3)
    return (foot_world - body_pos) @ R_body   # (6, 3): rel @ R = R^T @ rel.T


# ----------------------------------------------------------------------------
# Foot-contact shaping — pure JAX.
# ----------------------------------------------------------------------------
def _foot_contact_stats(params: EnvParams,
                        foot_world: jnp.ndarray,
                        prev_foot_world: jnp.ndarray,
                        was_in_contact: jnp.ndarray,
                        contact_steps: jnp.ndarray,
                        dt: float):
    """Per-step contact summary + updated tracking state.

    Returns:
      n_contact         (): int32   feet currently in contact
      sliding           (): float32 sum of horizontal-foot-speed above deadzone
                                    over feet currently in contact
      short_lifts       (): int32   lift-off events this step where prior
                                    contact lasted < min_contact_steps
      new_was_in_contact (6,) bool
      new_contact_steps  (6,) int32
    """
    in_contact_now = foot_world[:, 2] < params.foot_contact_z         # (6,) bool

    # Horizontal foot velocity via finite difference.
    foot_vel    = (foot_world - prev_foot_world) / dt
    horiz_speed = jnp.hypot(foot_vel[:, 0], foot_vel[:, 1])           # (6,)
    slip        = jnp.maximum(0.0, horiz_speed - params.sliding_deadzone)
    sliding     = jnp.sum(jnp.where(in_contact_now, slip, 0.0))

    n_contact = jnp.sum(in_contact_now.astype(jnp.int32))

    # Lift-off detection: was True now False.
    lifted_off = jnp.logical_and(was_in_contact,
                                 jnp.logical_not(in_contact_now))
    is_short = jnp.logical_and(lifted_off,
                               contact_steps < params.min_contact_steps)
    short_lifts = jnp.sum(is_short.astype(jnp.int32))

    # Update contact_steps: +1 if in_contact_now, else 0.
    new_contact_steps = jnp.where(in_contact_now, contact_steps + 1, 0)

    return n_contact, sliding, short_lifts, in_contact_now, new_contact_steps


# ----------------------------------------------------------------------------
# Reward
# ----------------------------------------------------------------------------
def _compute_reward(params: EnvParams,
                    state: EnvState,
                    mjx_data,
                    action: jnp.ndarray,
                    scaffold_intent_body: jnp.ndarray):
    """Paper-aligned reward (post-2026-05-07 redesign). Six terms:
        +tracking_reward         (gaussian on cmd-vs-actual velocity)
        -action_rate_pen          (smoothness)
        -z_vel_pen                (body vertical velocity squared)
        -joint_torque_pen         (sum of joint torque squared)
        -joint_vel_limit_pen      (deadband+quadratic above 90% of motor velocity limit)
        -joint_torque_limit_pen   (deadband+quadratic above 90% of motor stall torque)

    AMP discriminator's style reward is added separately by HexapodAMPEnv.

    Foot-contact statistics are still computed because EnvState carries
    `foot_was_in_contact` / `foot_contact_steps` across steps for use in
    observations / terminations — but they no longer feed the reward.
    """
    qpos = mjx_data.qpos
    qvel = mjx_data.qvel
    body_linvel  = _body_frame_linvel(qpos, qvel)
    vx, vy       = body_linvel[0], body_linvel[1]
    body_vz      = qvel[2]                                            # world-frame vertical velocity
    wz           = qvel[5]
    pitch, roll  = _body_pitch_roll(qpos)
    height_delta = qpos[2] - params.ref_body_z

    # Compute actual stance width as the deviation of mean(|foot_y|) (in body
    # frame) from the neutral baseline. Used for explicit width tracking
    # (slot 6 of cmd) — without this signal the policy ignores width cmds.
    foot_body_now = _foot_body_positions(params, mjx_data)             # (6, 3)
    actual_width_delta = jnp.mean(jnp.abs(foot_body_now[:, 1])) - params.neutral_lateral_spread

    # v25 (2026-05-13): EMA-filter the motion components (vx, vy, wz) to make
    # tracking robust against high-frequency jitter that gamed the instantaneous
    # gaussian. Posture components (pitch, roll, height/width deltas) stay
    # instantaneous — those are body-state signals, not motion that can be jittered.
    instant_motion  = jnp.array([vx, vy, wz], dtype=jnp.float32)
    new_linvel_ema  = ((1.0 - params.linvel_ema_alpha) * state.linvel_ema
                       + params.linvel_ema_alpha * instant_motion)
    f_vx, f_vy, f_wz = new_linvel_ema[0], new_linvel_ema[1], new_linvel_ema[2]

    actual = jnp.array([f_vx, f_vy, f_wz, pitch, roll, height_delta, actual_width_delta, 0.0, 0.0])

    # 1. Task tracking — gaussian on cmd-vs-actual error (motion components
    # EMA-filtered to defeat the jitter-gaming exploit).
    err = (state.cmd - actual) * params.err_inv_scales * params.reward_track_mask
    tracking = jnp.exp(-jnp.dot(err, err))

    # 2. Action smoothness — penalize per-step change in action.
    action_delta = action - state.last_action
    action_rate  = params.action_rate_w * jnp.dot(action_delta, action_delta)

    # 3. Body vertical velocity — encourages stable height (no bouncing).
    z_vel_pen = params.z_vel_w * (body_vz ** 2)

    # 3b. Body roll/pitch angular velocity (ω_x, ω_y) — paper: -0.08 × ||ω_xy||²
    # Discourages body twist; commanded body shape comes from cmd, not motion.
    body_angvel_xy = qvel[3:5]                                           # (2,)
    body_angvel_xy_pen = params.body_angvel_xy_w * jnp.dot(body_angvel_xy, body_angvel_xy)

    # 4. Joint torque magnitude — encourages low-effort motion.
    # actuator_force has shape (nu,) = (18,) for our 18 actuators (one per joint).
    joint_torques = mjx_data.actuator_force                            # (18,)
    joint_torque_pen = params.joint_torque_w * jnp.dot(joint_torques, joint_torques)

    # 5. Joint velocity limit — deadband+quadratic above 90% of motor max (6.18 rad/s).
    joint_vels = qvel[6:24]                                            # (18,)
    vel_threshold = params.joint_vel_limit * params.limit_deadband_frac
    vel_excess = jnp.maximum(jnp.abs(joint_vels) - vel_threshold, 0.0)
    joint_vel_limit_pen = params.joint_vel_limit_w * jnp.dot(vel_excess, vel_excess)

    # 6. Joint torque limit — deadband+quadratic above 90% of motor stall (1.5 N·m).
    torque_threshold = params.joint_torque_limit * params.limit_deadband_frac
    torque_excess = jnp.maximum(jnp.abs(joint_torques) - torque_threshold, 0.0)
    joint_torque_limit_pen = params.joint_torque_limit_w * jnp.dot(torque_excess, torque_excess)

    # 7. Foot contact force limit (NEW v5). cfrc_ext[tibia_id, 3:6] is the
    # linear ground reaction force on each tibia (only contacts that body has
    # are foot↔floor since the MJCF restricts contact pairs). Per-foot force
    # magnitude penalized via deadband+quadratic above f_limit.
    tibia_forces = mjx_data.cfrc_ext[params.tibia_body_ids, 3:6]              # (6, 3)
    foot_force_mag = jnp.linalg.norm(tibia_forces, axis=1)                    # (6,)
    force_excess = jnp.maximum(foot_force_mag - params.foot_force_limit, 0.0)
    foot_force_limit_pen = params.foot_force_limit_w * jnp.dot(force_excess, force_excess)

    # 8. Anti-drift LINEAR penalties (NEW v10). The gaussian tracking is
    # forgiving for small errors by design — at 0.05 rad/s of unintended yaw,
    # the gaussian dent is ~0.001 reward. Linear penalty bites consistently
    # at any error magnitude and specifically targets drift on STRAIGHT cmds
    # while still allowing intentional turning (penalty is on (actual - cmd)).
    yaw_drift_pen = params.yaw_drift_w * jnp.abs(state.cmd[2] - wz)
    vy_drift_pen  = params.vy_drift_w  * jnp.abs(state.cmd[1] - vy)

    # Foot-contact stats still computed for state-bookkeeping (observations,
    # terminations downstream) AND now (v25) used for the gait-phase contact
    # mismatch penalty.
    foot_world = _foot_world_positions(params, mjx_data)
    dt = params.mjx_model.opt.timestep
    n_contact, _sliding, _short_lifts, in_contact_now, new_contact_steps = (
        _foot_contact_stats(params, foot_world, state.prev_foot_world,
                            state.foot_was_in_contact,
                            state.foot_contact_steps, dt)
    )

    # v25: gait-phase contact mismatch penalty.
    # Tripod alternation: A = (RR, RF, LM)=idx[0,2,4], B = (RM, LR, LF)=idx[1,3,5].
    # Phase ∈ [0, 0.5]: A in stance. Phase ∈ [0.5, 1]: B in stance.
    phase = (state.sim_time / params.gait.gait_period) % 1.0
    a_in_stance = phase < 0.5
    expected_contact = jnp.where(
        a_in_stance,
        jnp.array([True, False, True, False, True, False]),    # A stance
        jnp.array([False, True, False, True, False, True]),    # B stance
    )
    contact_mismatch = jnp.logical_xor(in_contact_now, expected_contact)
    contact_mismatch_pen = (params.contact_mismatch_w
                            * jnp.sum(contact_mismatch.astype(jnp.float32)))

    reward = (tracking
              - action_rate
              - z_vel_pen
              - body_angvel_xy_pen
              - joint_torque_pen
              - joint_vel_limit_pen
              - joint_torque_limit_pen
              - foot_force_limit_pen
              - yaw_drift_pen
              - vy_drift_pen
              - contact_mismatch_pen)

    metrics = {
        "tracking_reward":          tracking,
        "action_rate_pen":          action_rate,
        "z_vel_pen":                z_vel_pen,
        "body_angvel_xy_pen":       body_angvel_xy_pen,
        "joint_torque_pen":         joint_torque_pen,
        "joint_vel_limit_pen":      joint_vel_limit_pen,
        "joint_torque_limit_pen":   joint_torque_limit_pen,
        "foot_force_limit_pen":     foot_force_limit_pen,
        "yaw_drift_pen":            yaw_drift_pen,
        "vy_drift_pen":             vy_drift_pen,
        "contact_mismatch_pen":     contact_mismatch_pen,
        "n_contact":                n_contact.astype(jnp.float32),
        "short_lifts":              _short_lifts.astype(jnp.float32),
        "foot_dev_total":           jnp.float32(0.0),
        # Deprecated keys kept as 0.0 for log-parser back-compat.
        "drift_pen":                jnp.float32(0.0),
        "speed_track_pen":          jnp.float32(0.0),
        "height_track_pen":         jnp.float32(0.0),
        "angvel_pen":               jnp.float32(0.0),
        "sliding_pen":              jnp.float32(0.0),
        "excess_contact_pen":       jnp.float32(0.0),
        "airborne_pen":             jnp.float32(0.0),
        "short_contact_pen":        jnp.float32(0.0),
        "foot_dev_pen":             jnp.float32(0.0),
    }
    return (reward, metrics,
            in_contact_now, new_contact_steps,
            foot_world, body_linvel, new_linvel_ema)


# ----------------------------------------------------------------------------
# Termination
# ----------------------------------------------------------------------------
def _compute_done(params: EnvParams,
                  qpos: jnp.ndarray,
                  cmd: jnp.ndarray,
                  step_count: jnp.ndarray,
                  progress_full: jnp.ndarray,
                  progress_buf_mean: jnp.ndarray,
                  cmd_speed: jnp.ndarray):
    body_z = qpos[2]
    qx, qy = qpos[4], qpos[5]
    tilt_sq = qx * qx + qy * qy
    pitch, roll = _body_pitch_roll(qpos)
    pitch_dev = jnp.abs(pitch - cmd[3])
    roll_dev  = jnp.abs(roll  - cmd[4])
    in_tilt_grace = step_count <= params.tilt_dev_grace
    tilt_overshoot = jnp.logical_and(
        jnp.logical_not(in_tilt_grace),
        jnp.logical_or(pitch_dev > params.tilt_dev_limit,
                       roll_dev  > params.tilt_dev_limit),
    )
    fell = jnp.logical_or(jnp.logical_or(body_z < params.min_z,
                                         tilt_overshoot),
                          tilt_sq > params.max_tilt_sq)

    # No-progress termination: only after grace window AND ring buffer is full
    # AND non-zero speed commanded AND running average is below the threshold.
    has_cmd = cmd_speed > 1e-6
    out_of_grace = step_count > params.no_progress_grace
    no_progress = (progress_full
                   & out_of_grace
                   & has_cmd
                   & (progress_buf_mean < params.no_progress_frac * cmd_speed))
    return jnp.logical_or(fell, no_progress), fell, no_progress


# ----------------------------------------------------------------------------
# Observation
# ----------------------------------------------------------------------------
def _get_obs(params: EnvParams, mjx_data, cmd, sim_time, scaffold_hint):
    from envs.obs_layout import compose_obs
    qpos = mjx_data.qpos
    qvel = mjx_data.qvel
    phase = (sim_time / params.gait.gait_period) % 1.0
    return compose_obs(
        joint_pos       = qpos[7:25],
        joint_vel       = qvel[6:24],
        imu_quat        = qpos[3:7],
        imu_gyro        = qvel[3:6],
        imu_accel       = jnp.zeros(3, dtype=jnp.float32),
        scaffold_hint   = scaffold_hint.reshape(-1),
        phase_sc        = jnp.array([jnp.sin(2.0 * jnp.pi * phase),
                                     jnp.cos(2.0 * jnp.pi * phase)]),
        cmd             = cmd,
        body_linvel     = _body_frame_linvel(qpos, qvel),
        joint_torque    = mjx_data.qfrc_actuator[6:24],   # AX-12A "present load"
        joint_pos_error = mjx_data.ctrl[:18] - qpos[7:25],
        concat_fn       = jnp.concatenate,
    )


# ----------------------------------------------------------------------------
# reset / step / set_gait_scale
# ----------------------------------------------------------------------------
def reset(params: EnvParams,
          rng: jax.Array,
          gait_scale: float = 0.0) -> EnvState:
    rng, sample_rng, jn_rng, z_rng = jax.random.split(rng, 4)
    cmd = _sample_cmd(params, sample_rng)

    # Domain randomization: perturb spawn pose (joint angles + body z) so
    # the policy sees a broader range of starting states. Helps generalize
    # and learn to recover from non-ideal initial conditions. Set noise
    # values to 0 in EnvParams to disable.
    spawn_qpos = params.spawn_qpos
    joint_noise = jax.random.uniform(
        jn_rng, (18,),
        minval=-params.init_joint_noise_rad,
        maxval= params.init_joint_noise_rad,
    )
    spawn_qpos = spawn_qpos.at[7:25].add(joint_noise)
    z_noise = jax.random.uniform(
        z_rng, (),
        minval=-params.init_body_z_noise_m,
        maxval= params.init_body_z_noise_m,
    )
    spawn_qpos = spawn_qpos.at[2].add(z_noise)

    mjx_data = mjx.make_data(params.mjx_model)
    mjx_data = mjx_data.replace(qpos=spawn_qpos,
                                ctrl=params.gait_neutral_pose)
    mjx_data = mjx.forward(params.mjx_model, mjx_data)

    sim_time   = jnp.float32(0.0)
    step_count = jnp.int32(0)
    last_action = jnp.zeros(18, dtype=jnp.float32)

    _, feet_body = gait_jax.predict_with_feet(params.gait, cmd, sim_time)

    foot_world_init = _foot_world_positions(params, mjx_data)
    # Spawn-time contact initialization: bot's feet are planted, so set
    # was_in_contact=True and contact_steps=min so the first lift is never
    # flagged as "too short."
    foot_was_in_contact = jnp.ones((6,), dtype=jnp.bool_)
    foot_contact_steps  = jnp.full((6,), params.min_contact_steps, dtype=jnp.int32)
    progress_buf  = jnp.zeros(NO_PROGRESS_WINDOW, dtype=jnp.float32)
    progress_idx  = jnp.int32(0)
    progress_full = jnp.bool_(False)

    obs = _get_obs(params, mjx_data, cmd, sim_time, feet_body)

    zero = jnp.float32(0.0)
    metrics_zero = {
        "tracking_reward":          zero, "action_rate_pen":          zero,
        # New paper-aligned terms (added 2026-05-07; body_angvel_xy_pen 2026-05-08).
        "z_vel_pen":                zero, "joint_torque_pen":         zero,
        "joint_vel_limit_pen":      zero, "joint_torque_limit_pen":   zero,
        "body_angvel_xy_pen":       zero, "foot_force_limit_pen":     zero,
        "yaw_drift_pen":            zero, "vy_drift_pen":             zero,
        "contact_mismatch_pen":     zero,
        # Deprecated keys kept at 0 for log-parser back-compat.
        "angvel_pen":               zero, "drift_pen":                zero,
        "speed_track_pen":          zero, "height_track_pen":         zero,
        "sliding_pen":              zero,
        "excess_contact_pen":       zero, "airborne_pen":             zero,
        "short_contact_pen":        zero, "foot_dev_pen":             zero,
        "foot_dev_total":           zero,
        "n_contact":                zero, "short_lifts":              zero,
        "bc_target":                jnp.zeros(18, dtype=jnp.float32),
        "fell":                     jnp.bool_(False),
        "no_progress":              jnp.bool_(False),
    }

    # Initialize dynamic-cmd state: cmd starts as the sampled value, target
    # equals current cmd (no transition yet), event_remaining sampled in
    # [event_min, event_max] so the first transition fires after a normal
    # hold time. delta is zero (no movement during initial hold).
    rng, rng_event0 = jax.random.split(rng)
    event_remaining = jax.random.randint(
        rng_event0, (),
        minval=params.cmd_event_min_steps,
        maxval=params.cmd_event_max_steps + 1,
    )

    # Initialize recovery-curriculum state. Begin with no active push and a
    # randomized next_push_in so envs don't all get pushed at the same step.
    rng, rng_push0 = jax.random.split(rng)
    next_push_in = jax.random.randint(
        rng_push0, (),
        minval=params.push_interval_min_steps,
        maxval=params.push_interval_max_steps + 1,
    )
    return EnvState(
        mjx_data=mjx_data,
        cmd=cmd,
        sim_time=sim_time,
        step_count=step_count,
        last_action=last_action,
        last_feet_body=feet_body,
        prev_foot_world=foot_world_init,
        foot_was_in_contact=foot_was_in_contact,
        foot_contact_steps=foot_contact_steps,
        progress_buf=progress_buf,
        progress_idx=progress_idx,
        progress_full=progress_full,
        gait_scale=jnp.float32(gait_scale),
        obs=obs,
        reward=zero,
        done=jnp.bool_(False),
        rng=rng,
        metrics=metrics_zero,
        cmd_target=cmd,
        cmd_delta_per_step=jnp.zeros(9, dtype=jnp.float32),
        transition_remaining=jnp.int32(0),
        event_remaining=event_remaining,
        push_force_world=jnp.zeros(3, dtype=jnp.float32),
        push_remaining=jnp.int32(0),
        next_push_in=next_push_in,
        linvel_ema=jnp.zeros(3, dtype=jnp.float32),
    )


def _advance_dynamic_cmd(params: EnvParams, state: EnvState):
    """Advance the dynamic-cmd state machine by one step.
    Returns (new_cmd, new_cmd_target, new_delta, new_trans_remaining,
             new_event_remaining, new_rng).

    No-op (cmd unchanged) when params.cmd_dynamics_enabled is False.

    Logic:
      - If event_remaining <= 0: sample new target + transition window +
        next event interval. Compute per-step delta = (target-cmd)/T.
      - If transition_remaining > 0: cmd += delta_per_step.
      - Else: cmd holds at cmd_target.
    """
    rng = state.rng
    rng, rng_t, rng_w, rng_n = jax.random.split(rng, 4)

    # Always sample candidate values (cheap; conditional usage via where).
    target_new = _sample_cmd(params, rng_t)                         # (9,)
    W_new = jax.random.randint(rng_w, (),
                                minval=params.cmd_transition_min,
                                maxval=params.cmd_transition_max + 1)
    N_new = jax.random.randint(rng_n, (),
                                minval=params.cmd_event_min_steps,
                                maxval=params.cmd_event_max_steps + 1)
    delta_new = (target_new - state.cmd) / W_new.astype(jnp.float32)

    trigger = state.event_remaining <= 0

    # Conditionally adopt new event params.
    cmd_target = jnp.where(trigger, target_new, state.cmd_target)
    cmd_delta  = jnp.where(trigger, delta_new, state.cmd_delta_per_step)
    trans_rem  = jnp.where(trigger, W_new, state.transition_remaining)
    event_rem  = jnp.where(trigger, N_new, state.event_remaining - 1)

    # Advance cmd: if in transition, step toward target; else hold at target.
    in_trans = trans_rem > 0
    new_cmd  = jnp.where(in_trans, state.cmd + cmd_delta, cmd_target)
    trans_rem_after = jnp.where(in_trans, trans_rem - 1, trans_rem)

    # Disable everything if cmd_dynamics is off — return state.cmd unchanged.
    enabled = params.cmd_dynamics_enabled
    new_cmd  = jnp.where(enabled, new_cmd,  state.cmd)
    cmd_target = jnp.where(enabled, cmd_target, state.cmd_target)
    cmd_delta  = jnp.where(enabled, cmd_delta,  state.cmd_delta_per_step)
    trans_rem_after = jnp.where(enabled, trans_rem_after, state.transition_remaining)
    event_rem  = jnp.where(enabled, event_rem,  state.event_remaining)
    return new_cmd, cmd_target, cmd_delta, trans_rem_after, event_rem, rng


def _advance_disturbance(params: EnvParams, state: EnvState):
    """Advance the recovery-curriculum push state machine by one step.
    Returns (new_push_force, new_push_remaining, new_next_push_in, new_rng).

    Logic per step:
      - If a push is ACTIVE (push_remaining > 0): keep applying the existing
        force, decrement push_remaining.
      - Else if next_push_in > 0: no force, decrement next_push_in.
      - Else: sample a fresh push (random horizontal direction, magnitude =
        max_force * disturbance_magnitude_scale), activate it for
        push_duration_steps, then reset next_push_in.

    No-op (force=0) when params.disturbance_enabled is False.
    """
    rng = state.rng
    rng, rng_angle = jax.random.split(rng)

    # Always sample a candidate fresh push (cheap; conditional usage via where).
    angle = jax.random.uniform(rng_angle, (), minval=0.0,
                                maxval=2.0 * jnp.pi)
    magnitude = params.max_push_force_n * params.disturbance_magnitude_scale
    force_xy = jnp.stack([magnitude * jnp.cos(angle),
                          magnitude * jnp.sin(angle),
                          jnp.float32(0.0)])

    push_active = state.push_remaining > 0
    cooldown    = jnp.logical_and(~push_active, state.next_push_in > 0)
    trigger     = jnp.logical_and(~push_active, ~cooldown)

    # Branch state updates with `where`s.
    new_force = jnp.where(push_active,
                          state.push_force_world,        # keep existing
                          jnp.where(trigger,
                                    force_xy,            # new push
                                    jnp.zeros(3, dtype=jnp.float32)))
    new_push_rem = jnp.where(push_active,
                             state.push_remaining - 1,
                             jnp.where(trigger,
                                       jnp.int32(params.push_duration_steps),
                                       jnp.int32(0)))
    # Recompute next_push_in via fresh sampling on trigger.
    rng, rng_intv = jax.random.split(rng)
    intv_new = jax.random.randint(rng_intv, (),
                                  minval=params.push_interval_min_steps,
                                  maxval=params.push_interval_max_steps + 1)
    new_next_in = jnp.where(push_active,
                            state.next_push_in,
                            jnp.where(cooldown,
                                      state.next_push_in - 1,
                                      intv_new))

    # Disable entirely when disturbance is off.
    enabled = params.disturbance_enabled
    new_force    = jnp.where(enabled, new_force,    jnp.zeros(3, dtype=jnp.float32))
    new_push_rem = jnp.where(enabled, new_push_rem, jnp.int32(0))
    new_next_in  = jnp.where(enabled, new_next_in,  state.next_push_in)
    return new_force, new_push_rem, new_next_in, rng


def _trimmed_cmd_for_scaffold(params: EnvParams, cmd: jnp.ndarray) -> jnp.ndarray:
    """Add scaffold-only yaw trim to cmd[2]. Used for prior generation
    to cancel physics-induced drift WITHOUT changing the cmd visible to
    obs / tracking reward. Returns the cmd-as-passed-to-scaffold; the
    state's `cmd` field is unchanged."""
    delta = (-params.scaffold_wz_trim_vx     * cmd[0]
             -params.scaffold_wz_trim_vy_abs * jnp.abs(cmd[1]))
    return cmd.at[2].add(delta)


def step(params: EnvParams, state: EnvState, action: jnp.ndarray) -> EnvState:
    # 1. Scaffold + intended feet at THIS step's sim_time.
    scaffold_cmd = _trimmed_cmd_for_scaffold(params, state.cmd)
    scaffold_joints, feet_body = gait_jax.predict_with_feet(
        params.gait, scaffold_cmd, state.sim_time)

    # 2. Mix: target = scaffold·gs + (neutral + a·R)·(1-gs).
    policy_target = params.gait_neutral_pose + action * params.residual_scale_max
    target = (scaffold_joints * state.gait_scale
              + policy_target * (1.0 - state.gait_scale))

    # 2b. Apply recovery-curriculum push (no-op when disturbance_enabled=False).
    new_push_force, new_push_rem, new_next_in, _ = \
        _advance_disturbance(params, state)
    xfrc = state.mjx_data.xfrc_applied
    xfrc = xfrc.at[params.base_body_id, 0:3].set(new_push_force)

    mjx_data = state.mjx_data.replace(ctrl=target, xfrc_applied=xfrc)
    mjx_data = mjx.step(params.mjx_model, mjx_data)

    sim_time   = state.sim_time + params.mjx_model.opt.timestep
    step_count = state.step_count + 1

    # 3. Reward + foot tracking update.
    (reward, reward_metrics,
     in_contact_now, new_contact_steps,
     foot_world, body_linvel, new_linvel_ema) = _compute_reward(
        params, state, mjx_data, action, feet_body)

    # 4. Progress ring buffer — advance with current speed-along-cmd.
    cmd_speed = jnp.hypot(state.cmd[0], state.cmd[1])
    safe_cs   = jnp.where(cmd_speed > 1e-6, cmd_speed, 1.0)
    cmd_dir_x = jnp.where(cmd_speed > 1e-6, state.cmd[0] / safe_cs, 0.0)
    cmd_dir_y = jnp.where(cmd_speed > 1e-6, state.cmd[1] / safe_cs, 0.0)
    speed_along_cmd = body_linvel[0] * cmd_dir_x + body_linvel[1] * cmd_dir_y
    new_buf  = state.progress_buf.at[state.progress_idx].set(speed_along_cmd)
    new_idx  = (state.progress_idx + 1) % NO_PROGRESS_WINDOW
    new_full = jnp.logical_or(state.progress_full, new_idx == 0)
    buf_mean = jnp.mean(new_buf)

    # 5. Termination.
    done, fell, no_progress = _compute_done(
        params, mjx_data.qpos, state.cmd, step_count,
        new_full, buf_mean, cmd_speed)

    # 6. BC target = scaffold's joint targets re-encoded as policy action.
    bc_target = jnp.clip(
        (scaffold_joints - params.gait_neutral_pose) / params.residual_scale_max,
        -1.0, 1.0)

    obs = _get_obs(params, mjx_data, state.cmd, sim_time, feet_body)

    metrics = dict(reward_metrics)
    metrics["bc_target"]   = bc_target
    metrics["fell"]        = fell
    metrics["no_progress"] = no_progress

    # Advance dynamic-cmd state (no-op when params.cmd_dynamics_enabled=False).
    new_cmd, new_target, new_delta, new_trans, new_event, new_rng = \
        _advance_dynamic_cmd(params, state)

    return state._replace(
        mjx_data=mjx_data,
        cmd=new_cmd,
        sim_time=sim_time,
        step_count=step_count,
        last_action=action,
        last_feet_body=feet_body,
        prev_foot_world=foot_world,
        foot_was_in_contact=in_contact_now,
        foot_contact_steps=new_contact_steps,
        progress_buf=new_buf,
        progress_idx=new_idx,
        progress_full=new_full,
        obs=obs,
        reward=reward,
        done=done,
        metrics=metrics,
        rng=new_rng,
        cmd_target=new_target,
        cmd_delta_per_step=new_delta,
        transition_remaining=new_trans,
        event_remaining=new_event,
        push_force_world=new_push_force,
        push_remaining=new_push_rem,
        next_push_in=new_next_in,
        linvel_ema=new_linvel_ema,
    )


def step_foot(params: EnvParams, state: EnvState, action: jnp.ndarray) -> EnvState:
    """Foot-space variant of step(). Action is interpreted as a (6, 3)
    foot residual in body frame, added to the scaffold's intended foot
    positions; IK then converts to joint targets. The three joints in
    each leg are coordinated by the IK so a single foot perturbation
    produces a smooth physical motion (vs joint-space noise's per-joint
    independent twitching that often compounds destructively).
    """
    # 1. Scaffold + intended feet at THIS step's sim_time.
    scaffold_cmd = _trimmed_cmd_for_scaffold(params, state.cmd)
    scaffold_joints, feet_body = gait_jax.predict_with_feet(
        params.gait, scaffold_cmd, state.sim_time)

    # 2. Apply foot residual in body frame, then IK to joint targets.
    foot_residual_body = action.reshape(6, 3) * params.foot_residual_scale_max
    feet_body_target = feet_body + foot_residual_body                 # (6, 3)
    feet_coxa_target = gait_jax._body_to_coxa_local_batch(feet_body_target,
                                                           params.gait)
    policy_joint_target = gait_jax._joints_from_feet_coxa(feet_coxa_target,
                                                           params.gait)

    # 3. Mix scaffold + policy joint target in joint space (same as step()).
    target = (scaffold_joints * state.gait_scale
              + policy_joint_target * (1.0 - state.gait_scale))

    # 3b. Apply recovery-curriculum push (no-op when disturbance_enabled=False).
    new_push_force, new_push_rem, new_next_in, _ = \
        _advance_disturbance(params, state)
    xfrc = state.mjx_data.xfrc_applied
    xfrc = xfrc.at[params.base_body_id, 0:3].set(new_push_force)

    mjx_data = state.mjx_data.replace(ctrl=target, xfrc_applied=xfrc)
    mjx_data = mjx.step(params.mjx_model, mjx_data)

    sim_time   = state.sim_time + params.mjx_model.opt.timestep
    step_count = state.step_count + 1

    # 4. Reward + foot tracking (identical to step()).
    (reward, reward_metrics,
     in_contact_now, new_contact_steps,
     foot_world, body_linvel, new_linvel_ema) = _compute_reward(
        params, state, mjx_data, action, feet_body)

    # 5. Progress ring buffer (identical to step()).
    cmd_speed = jnp.hypot(state.cmd[0], state.cmd[1])
    safe_cs   = jnp.where(cmd_speed > 1e-6, cmd_speed, 1.0)
    cmd_dir_x = jnp.where(cmd_speed > 1e-6, state.cmd[0] / safe_cs, 0.0)
    cmd_dir_y = jnp.where(cmd_speed > 1e-6, state.cmd[1] / safe_cs, 0.0)
    speed_along_cmd = body_linvel[0] * cmd_dir_x + body_linvel[1] * cmd_dir_y
    new_buf  = state.progress_buf.at[state.progress_idx].set(speed_along_cmd)
    new_idx  = (state.progress_idx + 1) % NO_PROGRESS_WINDOW
    new_full = jnp.logical_or(state.progress_full, new_idx == 0)
    buf_mean = jnp.mean(new_buf)

    # 6. Termination (identical to step()).
    done, fell, no_progress = _compute_done(
        params, mjx_data.qpos, state.cmd, step_count,
        new_full, buf_mean, cmd_speed)

    # 7. BC target = zeros. In foot-space mode, action=0 already produces
    # the scaffold's behavior (since the residual sum is zero); BC's
    # supervised target is therefore "always output zero." That's a
    # degenerate target — better to skip BC pretrain entirely for
    # foot-space mode and start PPO from a fresh small-init policy.
    bc_target = jnp.zeros_like(action)

    obs = _get_obs(params, mjx_data, state.cmd, sim_time, feet_body)

    metrics = dict(reward_metrics)
    metrics["bc_target"]   = bc_target
    metrics["fell"]        = fell
    metrics["no_progress"] = no_progress

    # Advance dynamic-cmd state (no-op when params.cmd_dynamics_enabled=False).
    new_cmd, new_target, new_delta, new_trans, new_event, new_rng = \
        _advance_dynamic_cmd(params, state)

    return state._replace(
        mjx_data=mjx_data,
        cmd=new_cmd,
        sim_time=sim_time,
        step_count=step_count,
        last_action=action,
        last_feet_body=feet_body,
        prev_foot_world=foot_world,
        foot_was_in_contact=in_contact_now,
        foot_contact_steps=new_contact_steps,
        progress_buf=new_buf,
        progress_idx=new_idx,
        progress_full=new_full,
        obs=obs,
        reward=reward,
        done=done,
        metrics=metrics,
        rng=new_rng,
        cmd_target=new_target,
        cmd_delta_per_step=new_delta,
        transition_remaining=new_trans,
        event_remaining=new_event,
        push_force_world=new_push_force,
        push_remaining=new_push_rem,
        next_push_in=new_next_in,
        linvel_ema=new_linvel_ema,
    )


def set_gait_scale(state: EnvState, value) -> EnvState:
    """Returns a new EnvState with gait_scale replaced. Use this in the
    trainer's curriculum hook to fade scaffold contribution between
    rollouts without recompiling step()."""
    return state._replace(gait_scale=jnp.float32(value))


# ----------------------------------------------------------------------------
# Smoke test — verify reset/step run, single env + vmapped batch.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    MODEL_PATH = "models/phantomx_simple_mjx.xml"

    print(f"loading {MODEL_PATH} + building env params...", end=" ", flush=True)
    t0 = time.perf_counter()
    params = make_env_params(MODEL_PATH)
    print(f"done ({time.perf_counter() - t0:.2f}s)")

    jit_reset = jax.jit(reset)
    jit_step  = jax.jit(step)

    rng = jax.random.PRNGKey(0)
    print("single-env reset...", end=" ", flush=True)
    t0 = time.perf_counter()
    s = jit_reset(params, rng)
    s.obs.block_until_ready()
    print(f"done ({time.perf_counter() - t0:.2f}s)")
    print(f"  obs shape:   {s.obs.shape}")
    print(f"  cmd:         {s.cmd}")

    print("single-env 5 steps (zero action)...", end=" ", flush=True)
    t0 = time.perf_counter()
    a = jnp.zeros(18, dtype=jnp.float32)
    for _ in range(5):
        s = jit_step(params, s, a)
    s.obs.block_until_ready()
    print(f"done ({time.perf_counter() - t0:.2f}s)")
    print(f"  reward: {float(s.reward):.4f}   done: {bool(s.done)}   "
          f"n_contact: {float(s.metrics['n_contact']):.0f}   "
          f"foot_dev: {float(s.metrics['foot_dev_total'])*1000:.2f} mm   "
          f"bc_target[0]: {float(s.metrics['bc_target'][0]):+.3f}")

    BATCH = 4096
    STEPS = 200
    print(f"\nvmap throughput test (BATCH={BATCH}, STEPS={STEPS})...")

    rngs = jax.random.split(rng, BATCH)
    vreset = jax.jit(jax.vmap(reset, in_axes=(None, 0)))
    vstep  = jax.jit(jax.vmap(step, in_axes=(None, 0, 0)))

    print("  vmap reset compile...", end=" ", flush=True)
    t0 = time.perf_counter()
    vs = vreset(params, rngs)
    vs.obs.block_until_ready()
    print(f"done ({time.perf_counter() - t0:.2f}s)")

    actions = jnp.zeros((BATCH, 18), dtype=jnp.float32)
    print("  vmap step compile...", end=" ", flush=True)
    t0 = time.perf_counter()
    vs = vstep(params, vs, actions)
    vs.obs.block_until_ready()
    print(f"done ({time.perf_counter() - t0:.2f}s)")

    print(f"  measuring {STEPS} vmapped steps (Python loop)...", end=" ", flush=True)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        vs = vstep(params, vs, actions)
    vs.obs.block_until_ready()
    elapsed = time.perf_counter() - t0
    total = STEPS * BATCH
    print(f"done ({elapsed:.2f}s)")
    print(f"  PY-LOOP:   {total / elapsed:>14,.0f} samples/sec  "
          f"({elapsed * 1e3 / STEPS:.2f} ms/batch-step)")

    # Same throughput measured via lax.scan — what Brax PPO actually does.
    # Fuses the per-step dispatches into one XLA call.
    print(f"  measuring {STEPS} vmapped steps (lax.scan)...", end=" ", flush=True)
    def scan_body(carry, _):
        carry = vstep(params, carry, actions)
        return carry, carry.reward
    # Warm up + compile.
    final, _ = jax.lax.scan(scan_body, vs, jnp.arange(STEPS))
    final.obs.block_until_ready()
    t0 = time.perf_counter()
    final, _ = jax.lax.scan(scan_body, vs, jnp.arange(STEPS))
    final.obs.block_until_ready()
    elapsed = time.perf_counter() - t0
    print(f"done ({elapsed:.2f}s)")
    print(f"  LAX-SCAN:  {total / elapsed:>14,.0f} samples/sec  "
          f"({elapsed * 1e3 / STEPS:.2f} ms/batch-step)")
