"""
HexapodEnv — cross-platform PhantomX hexapod RL environment.

Unified version that works on both macOS and Linux. Live-viewer behavior
uses shared memory on both platforms: env-0 (when constructed with
`live_watch=True`) publishes qpos+qvel+sim_time every step to a named
shared-memory region (`LIVE_SHM_NAME`), and a separate `live_viewer.py`
process reads that region and renders.

Why SHM on both platforms (was Mac-only before unification): on macOS
the Cocoa viewer can only run under `mjpython`, but SubprocVecEnv
workers run as plain `python` — so the in-worker `launch_passive`
approach is unavailable. SHM decouples the viewer from the worker
process, fixing that. Linux has no such restriction (in-worker viewer
also works there), but using SHM uniformly keeps a single code path
across platforms with negligible overhead.

Per-platform launch reminders:
  * Training (any platform):     python train.py
  * Live viewer on macOS:        mjpython live_viewer.py
  * Live viewer on Linux:        python   live_viewer.py
  * Eval scripts that open the MuJoCo viewer (watch.py, watch_demo.py,
    pilot_ai.py): mjpython on macOS, python on Linux.

Thin wrapper around gait.Controller. The Controller produces a scaffold of
joint targets each step from the current cmd vector; the policy outputs an
18-dim residual that is mixed in. As gait_scale fades from 1.0 → 0.0 over
training, the scaffold contribution shrinks and the residual takes over.

Cmd vector (matches gait.Controller — physical units):
  [0] vx           m/s     body forward velocity
  [1] vy           m/s     body lateral velocity (left = +)
  [2] wz           rad/s   body yaw rate (CCW = +)
  [3] pitch        rad     body pitch target (nose up = +)
  [4] roll         rad     body roll target (left side up = + ; standard Euler;
                                              matches what _body_pitch_roll computes)
  [5] height_delta m       stance height delta (- = body raised)
  [6] width_delta  m       stance width delta (+ = wider)
  [7] shift_x      m       body shift in body +X (planted feet stay put)
  [8] shift_y      m       body shift in body +Y (planted feet stay put)

Curriculum stages — controlled by `stage` arg at construction. Each stage
unlocks a subset of cmd slots; locked slots are sampled as 0.

  Stage 1: vx, vy                                       (translation)
  Stage 2: + wz, height_delta, width_delta              (+ spin + stance)
  Stage 3: + pitch, roll, shift_x, shift_y              (+ pose, shift)
  Stage 4: same as 3 + novelty bonuses + body_linvel    (autonomous + stretch)
           dropped from obs

Reward (minimal — see project_gait_training_architecture.md):
  + tracking_exp(-||cmd-actual||²)        per active stage cmd slots
  + 0.1 * survive_bonus                   constant per step
  - 0.01 * action_delta²                  smoothness
  + stage 4: novelty bonuses              (max-speed, sustained inverted, etc)
  terminate on fall or extreme tilt
"""

import math
import os
from multiprocessing import shared_memory

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

from gait import Controller, NEUTRAL_POSE


# Shared-memory name for the env-0 → live_viewer.py state mirror. The training
# process's env-0 worker writes qpos+qvel+sim_time here every step; a separate
# `mjpython live_viewer.py` process reads it and renders in real time. This
# decouples the viewer from the worker process — necessary on macOS where the
# Cocoa viewer can only run under mjpython but SubprocVecEnv workers don't.
LIVE_SHM_NAME = "hexapod_live_state"


MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/phantomx.xml")


# Per-stage cmd-slot enable mask. 1 = sampleable / tracked, 0 = forced to zero.
# Index matches cmd vector slots [vx, vy, wz, pitch, roll, height, width, sx, sy].
STAGE_CMD_MASK = {
    # Translation-only stage 1: any direction (random heading) + variable speed
    # (polar sampling, magnitude in [0, MAX_SPEED]). The drift penalty
    # *implicitly* trains the policy to stay straight, level, and at height
    # without commanding wz/pitch/roll/height — so adding those slots later as
    # commanded skills builds on the discipline already learned here.
    1: np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
    2: np.array([1, 1, 1, 0, 0, 1, 1, 0, 0]),
    # Stage 3: full motion including body shift (shift_x, shift_y). The
    # gait controller now applies the shift overlay (task #8 done) so
    # commanding them produces a real scaffold response and provides BC/RL
    # signal. Stays off in stage 4 only if a future curriculum step wants
    # to defer it.
    3: np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
    4: np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
}

# Cmd slots whose tracking actually matters for reward. Width and shifts are
# scaffold pass-through (no direct kinematic equivalent to track); they're
# enabled in obs/cmd but don't contribute reward terms. Locomotion slots get
# dense tracking reward.
REWARD_TRACK_MASK = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0])


class HexapodEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    CMD_DIM            = 9
    ACTION_DIM         = 18

    # Policy authority — the rad/joint that an action of ±1 produces in the
    # policy_target = gait_neutral + action × RESIDUAL_SCALE_MAX formula. This
    # is CONSTANT under the new weighted-average mixing, regardless of
    # gait_scale: the policy always outputs full-authority joint targets. The
    # cross-fade between scaffold and policy is what gait_scale controls.
    RESIDUAL_SCALE_MAX = 0.80   # rad — full action authority on the policy stream
    # Old RESIDUAL_SCALE_MIN (0.10) is no longer used; under residual-add
    # mixing it suppressed policy authority during scaffold-strong, but that
    # also denied the policy any real role in training. The new architecture
    # uses BC during scaffold-strong instead.

    SURVIVE_BONUS    = 0.10
    ACTION_RATE_W    = 0.01
    ANGVEL_W         = 0.02     # damps body roll/pitch rate (wobble)

    # Translation-magnitude sampling range (fractions of MAX_SPEED).
    # Empirically, sampling all the way down to 0 produced too many "stand
    # still" episodes (policy converges on do-nothing); sampling all the way
    # to MAX_SPEED produced ragged-looking gaits at the edge of what the
    # scaffold can deliver. Squeezing the range gives the policy a cleaner
    # distribution to learn from. Episode cmds always have actual movement.
    SPEED_MIN_FRAC   = 0.40     # smallest commanded speed = 0.40 × MAX_SPEED
    SPEED_MAX_FRAC   = 0.85     # largest commanded speed  = 0.85 × MAX_SPEED

    # Drift penalties — *linear* (sharp gradient near zero) penalty on the
    # actual yaw rate / pitch / roll when the corresponding cmd is near zero.
    # Gates on cmd magnitude so it deactivates when the user actually commands
    # rotation or tilt. Squared tracking reward is too forgiving for small
    # drift (a 0.05 rad/s yaw drift squared-normalized is barely 0.02 lost
    # reward per step, accumulating to 14° unwanted rotation per episode).
    YAW_DRIFT_W       = 1.0     # weight × |actual_wz| when no yaw commanded
    PITCH_DRIFT_W     = 3.0     # weight × |actual_pitch| when no pitch commanded
    ROLL_DRIFT_W      = 3.0     # weight × |actual_roll|  when no roll commanded
    # Cmd magnitudes above which drift penalty fully deactivates. Below these,
    # the penalty linearly tapers from full → 0.
    YAW_GATE          = 0.05    # rad/s
    TILT_GATE         = math.radians(3.0)   # 3° in rad

    MIN_Z          = 0.06   # body height below this = fallen
    # qx² + qy² = sin²(θ/2). 30° absolute safety net = sin²(15°) ≈ 0.067.
    # The PRIMARY tilt-failure check is now deviation-based (TILT_DEV_LIMIT
    # below) — the bot is allowed to track high commanded pitch/roll up to
    # ±15° without false-positive termination, but if it OVERSHOOTS the
    # commanded tilt by more than 5° in any axis, that's a fall.
    MAX_TILT_SQ    = 0.067   # 30° absolute backup (only fires for extreme cases)
    # Maximum allowed deviation between actual and commanded pitch/roll.
    # 5° is the user's stated intent — "brief over 5° fine, sustained not
    # great, 10° fail." With the env's pitch sign convention fixed,
    # tracking deviation across the full ±15° cmd range stays under ~1°
    # for both pitch and roll, so 5° is a comfortable cap with margin.
    TILT_DEV_LIMIT = math.radians(5.0)
    # Grace period at episode start. Bot spawns upright (actual_pitch=0) but
    # cmd_pitch may be ±15° — the bot needs time to physically rotate into
    # the commanded tilt. Without grace, every high-tilt episode auto-fails
    # at step 0. During grace, only the absolute MAX_TILT_SQ (30°) safety
    # net is active; the per-axis deviation check is suspended.
    TILT_DEV_GRACE = 400   # 2 s @ dt=0.005 s
    REF_BODY_Z     = 0.13   # nominal standing height (used to keep height tracking sensible)

    # Linear speed-tracking penalty. The gaussian tracking reward exp(-err²) is
    # too forgiving near small errors — at cmd_speed=0.04 m/s and actual=0,
    # gaussian still gives ~0.7 reward. The policy can occupy a "stand still"
    # local optimum: low angvel, no drift_pen (gates close on non-zero cmd),
    # zero action_rate (saturated constant action), modest tracking, full
    # survive bonus. A *linear* speed-error term has a sharp gradient near
    # zero that breaks that minimum and pushes the policy to actually move.
    SPEED_TRACK_W      = 0.5    # weight on |cmd_speed - actual_speed| / MAX_SPEED

    # No-progress termination. Episode ends if the bot's avg speed along the
    # commanded heading over a rolling window stays below a fraction of the
    # commanded speed. Treats "standing still under non-zero cmd" the same
    # way as falling: terminates without survive bonus, fall_rate counts it.
    # Without this, fall_rate=0 + mean_ep_length=2000 mislead — they only
    # mean "didn't tilt past 60°," not "actually walked."
    NO_PROGRESS_WINDOW = 400    # 2.0 s window @ dt=0.005 s
    NO_PROGRESS_FRAC   = 0.20   # require ≥20% of commanded speed sustained.
                                # Was 0.30; relaxed because at stage=3 the bot
                                # is asked to walk forward AND hold up to ±15°
                                # tilt simultaneously — physical limit means
                                # forward speed under heavy tilt is reduced.
                                # speed_track_pen still strongly rewards faster
                                # walking; this only changes when termination
                                # fires on near-stationary.
    NO_PROGRESS_GRACE  = 200    # 1 s grace at episode start (acceleration window)

    # Foot-contact reward shaping. Without this, the policy "cheats" by
    # planting front legs on the ground and pushing with the rear, sliding
    # forward as if on rails. The middle legs end up doing nothing.
    #   - SLIDING_W: penalty per m/s of horizontal velocity for any foot in
    #     contact. Forces planted feet to be still while bearing weight —
    #     i.e. enforces friction-locked stance, not skating.
    #   - EXCESS_CONTACT_W: penalty per foot beyond 3 in simultaneous contact.
    #     Encourages stepping (lifting feet) rather than gliding on all 6.
    #     We do NOT penalize <3 feet contact — that would forbid emergent
    #     bounding/jumping/rolling gaits the policy might discover.
    FOOT_CONTACT_Z       = 0.005  # foot-tip world z below this = in contact
    SLIDING_W            = 2.0    # per (m/s × foot-in-contact) above deadzone
    SLIDING_DEADZONE     = 0.010  # m/s — small slip is unavoidable, don't punish
    EXCESS_CONTACT_W     = 0.05   # per foot beyond 3 in simultaneous contact
    # Airborne penalty: per-step, only when zero feet in contact. Light so a
    # brief flight phase (a hop, a bound) is cheap; sustained airborne adds up.
    AIRBORNE_W           = 0.10   # per step with n_contact == 0
    # Short-contact penalty: per lift-off event if the planted duration was
    # below MIN_CONTACT_STEPS. Catches the "rapid foot-tap" cheat where the
    # policy briefly touches each foot to game sliding/excess penalties.
    # 30 steps × 0.005 dt = 0.15 s minimum stance.
    MIN_CONTACT_STEPS    = 30
    SHORT_CONTACT_W      = 0.5    # per lift-off event below MIN_CONTACT_STEPS
    # Foot tip in tibia-local frame. See CLAUDE.md / docs/kinematics.md.
    _FOOT_LOCAL_OFFSET   = np.array([0.134, 0.031, 0.0], dtype=np.float64)
    _TIBIA_BODY_NAMES    = ("tibia_RR", "tibia_RM", "tibia_RF",
                            "tibia_LR", "tibia_LM", "tibia_LF")

    # Foot-position deviation penalty. Compares actual foot-tip body-frame
    # positions to the scaffold's intended foot-tip positions for this step
    # (cached as `_latest_feet_body`). Penalizes the policy for drifting feet
    # away from the planned stance — RL is otherwise free to re-route feet
    # to weird positions that score reward in some other way (the v3 run's
    # observed "gait deteriorating into stance creep" failure).
    # Weight is per meter per foot, summed over 6 feet. 0.05 m drift × 6
    # feet × FOOT_DEV_W = penalty/step. Soft enough to allow RL refinement
    # of the gait, hard enough to keep stance close to the scaffold's plan.
    FOOT_DEV_W           = 1.5

    # Random sampling magnitudes per stage.
    SAMPLE_RANGES = {
        # slot:  (lo, hi)
        # Speeds quote symmetric ±MAX. Yaw is signed. Pose targets are bounded angles.
        # Stance/shift use small symmetric ranges for now.
        0: (-1.0, 1.0),     # vx (× MAX_SPEED — scaled at sample time)
        1: (-1.0, 1.0),     # vy
        2: (-1.0, 1.0),     # wz (× MAX_YAW_RATE)
        # PITCH — ±10°. The body-tilt overlay can produce ±15° pitch when
        # the bot is stationary, but combined with forward walking at
        # SPEED_MIN_FRAC × MAX_SPEED+ the bot can't physically sustain
        # forward progress beyond ~±10° pitch (legs angled too steeply).
        # ±10° is the empirically-validated combined-cmd range.
        # ROLL — ±15°. Roll up to the controller's full reachable range
        # works fine even combined with full-speed forward walking.
        3: (-math.radians(10), math.radians(10)),
        4: (-math.radians(15), math.radians(15)),
        5: (-0.020, 0.020),
        6: (-0.015, 0.015),
        7: (-0.030, 0.030),
        8: (-0.030, 0.030),
    }

    def __init__(self, render_mode=None, stage=1, live_watch=False, model_path=None):
        super().__init__()
        self.render_mode = render_mode
        self.stage       = stage
        # Optional override of the MJCF file. Defaults to the project's main
        # mesh-based model (MODEL_PATH constant). Pass `model_path=...` to
        # use a different MJCF (e.g. the primitive-geom phantomx_simple.xml
        # for MJX validation runs) without permanently editing the constant.
        # Path can be absolute or relative to this env file's directory.
        if model_path is None:
            self._model_path = MODEL_PATH
        elif os.path.isabs(model_path):
            self._model_path = model_path
        else:
            # Relative paths anchor to either the env file's directory (e.g.
            # "../models/phantomx.xml") or the cwd (e.g. "models/...").
            # Pick whichever resolves to an existing file; fall back to cwd.
            env_relative = os.path.join(os.path.dirname(__file__), model_path)
            self._model_path = (env_relative if os.path.exists(env_relative)
                                else model_path)
        # Side-channel live-state mirror via shared memory. When True, this env
        # writes its qpos+qvel to a named shared memory region every step. A
        # separate `mjpython live_viewer.py` process reads it and renders. We
        # CANNOT open a viewer directly here because SubprocVecEnv workers on
        # macOS aren't mjpython and can't init the Cocoa main loop. Shared
        # memory bypasses that entirely. Only set live_watch=True for env 0.
        self.live_watch  = bool(live_watch)
        self._live_shm   = None
        self._live_buf   = None

        # Mujoco
        self._model = mujoco.MjModel.from_xml_path(self._model_path)
        self._data  = mujoco.MjData(self._model)
        self._dt    = float(self._model.opt.timestep)

        # Gait controller — the analytical scaffold.
        self._ctrl = Controller(self._model_path)

        # Per-slot tracking error scales. Normalizes errors so that a "full
        # deviation" on any slot contributes roughly 1.0 to the error norm
        # squared. Without this, slots with small physical magnitudes (e.g.,
        # height in meters) barely affect the reward while slots with larger
        # magnitudes (e.g., velocities) dominate. Inverse-multiplied, so a
        # full-scale deviation in any slot drops the gaussian reward by ~63%.
        FULL_HEIGHT_DEV = 0.025         # 25 mm height drift = "full" penalty
        FULL_PITCH_DEV  = math.radians(15)
        FULL_ROLL_DEV   = math.radians(15)
        self._err_inv_scales = np.array([
            1.0 / self._ctrl.MAX_SPEED,        # vx
            1.0 / self._ctrl.MAX_SPEED,        # vy
            1.0 / self._ctrl.MAX_YAW_RATE,     # wz
            1.0 / FULL_PITCH_DEV,              # pitch
            1.0 / FULL_ROLL_DEV,               # roll
            1.0 / FULL_HEIGHT_DEV,             # height_delta
            0.0, 0.0, 0.0,                     # width, shifts (not tracked)
        ], dtype=np.float64)

        # Cache sensor data addresses.
        self._acc_adr  = self._model.sensor("imu_acc").adr[0]
        self._gyro_adr = self._model.sensor("imu_gyro").adr[0]
        self._quat_adr = self._model.sensor("imu_quat").adr[0]

        # Cache tibia body IDs for foot-contact reward shaping.
        self._tibia_body_ids = np.array(
            [self._model.body(n).id for n in self._TIBIA_BODY_NAMES],
            dtype=np.int32,
        )
        # Cache base body id for body-frame foot-position transform.
        self._base_body_id = int(self._model.body("base").id)
        # Scratch buffer for mj_objectVelocity (avoid allocation per step).
        self._vel6_scratch = np.zeros(6, dtype=np.float64)

        # Obs layout is owned by envs/obs_layout.py — single source of truth
        # shared with hexapod_env_jax.py so layout changes can't drift apart.
        from envs.obs_layout import OBS_DIM
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(self.ACTION_DIM,), dtype=np.float32)

        # Episode-mutable state.
        self._cmd         = np.zeros(self.CMD_DIM, dtype=np.float32)
        self._sim_time    = 0.0
        self._step_count  = 0
        self._last_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        # Cached body-frame foot targets from the latest predict_with_feet()
        # call — used by _get_obs() to avoid recomputing the whole gait pipeline.
        self._latest_feet_body = np.zeros((6, 3), dtype=np.float32)
        # Rolling buffer of speed-along-cmd-direction (last NO_PROGRESS_WINDOW
        # steps) for no-progress termination. Ring buffer for O(1) updates.
        self._progress_buf  = np.zeros(self.NO_PROGRESS_WINDOW, dtype=np.float32)
        self._progress_idx  = 0
        self._progress_full = False
        # Per-foot contact-state tracking for short-contact penalty.
        # contact_steps[i]: how many consecutive steps foot i has been in
        # contact. was_in_contact[i]: whether foot i was in contact LAST step.
        # On lift-off (was=True, now=False), if contact_steps < MIN, penalize.
        self._foot_was_in_contact = np.zeros(6, dtype=bool)
        self._foot_contact_steps  = np.zeros(6, dtype=np.int32)

        # gait_scale is mutated externally by the StageManagerCallback. CRITICAL:
        # the callback must update via `env_method("set_gait_scale", value)` —
        # NOT `set_attr("gait_scale", value)`. SB3's set_attr only sets the
        # attribute on the OUTER VecEnv-side wrapper (Monitor); gym wrappers
        # don't propagate setattr to the wrapped env, so set_attr silently
        # leaves the inner HexapodEnv.gait_scale unchanged. env_method
        # forwards through Wrapper.__getattr__ to this method, which sets
        # the attribute on the actual HexapodEnv. (See set_gait_scale below.)
        self.gait_scale = 1.0

        self._viewer   = None
        self._renderer = None

        # Set up the live-state shared memory if requested. Only env 0 should
        # do this; the same name from multiple workers would collide.
        if self.live_watch:
            self._init_live_shm()

    # ------------------------------------------------------------------
    # Standard gym lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)

        self._data.qpos[2]    = 0.18           # spawn just above the ground
        self._data.qpos[3]    = 1.0            # quat w (upright)
        # Use the controller's widened gait-time neutral, NOT the hardcoded
        # NEUTRAL_POSE (which corresponds to the un-widened calibrated rest).
        # Avoids a transient on the first step when the gait pulls the legs to
        # the widened stance.
        gait_neutral = self._ctrl.gait_neutral_pose.astype(np.float32)
        self._data.qpos[7:25] = gait_neutral
        self._data.ctrl[:]    = gait_neutral
        mujoco.mj_forward(self._model, self._data)

        self._cmd         = self._sample_cmd().astype(np.float32)
        self._sim_time    = 0.0
        self._step_count  = 0
        self._last_action = np.zeros(self.ACTION_DIM, dtype=np.float32)

        # Reset the no-progress rolling buffer for this episode.
        self._progress_buf[:]  = 0.0
        self._progress_idx     = 0
        self._progress_full    = False
        # Reset per-foot contact tracking. The bot spawns with all feet on
        # the ground; initialize contact_steps to MIN_CONTACT_STEPS so the
        # FIRST lift-off after spawn is never flagged as "too short."
        self._foot_was_in_contact[:] = True
        self._foot_contact_steps[:]  = self.MIN_CONTACT_STEPS

        # Prime the feet_body cache so the first _get_obs() returns a meaningful
        # scaffold_hint (otherwise it would be the zero-init scratch buffer).
        self._latest_feet_body = self._ctrl.compute_foot_targets(
            self._cmd, self._sim_time).astype(np.float32)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # 1. Scaffold joint targets + body-frame foot positions in one pass.
        # Cache feet_body so _get_obs() doesn't re-run the gait pipeline.
        scaffold_joints, feet_body = self._ctrl.predict_with_feet(self._cmd, self._sim_time)
        scaffold_joints = scaffold_joints.astype(np.float32)
        self._latest_feet_body = feet_body.astype(np.float32)

        # 2. Mix scaffold + policy via weighted-average (NOT residual-add).
        #    target = scaffold × gait_scale + policy_target × (1 - gait_scale)
        #    where policy_target = gait_neutral + action × RESIDUAL_SCALE_MAX.
        # During scaffold-strong (gs=1.0): policy's `action` is physically
        # ignored — the scaffold drives the bot at full strength, which means
        # we can BC-supervise the policy to predict the scaffold's joint
        # trajectory without risking the scaffold's gait.
        # During autonomous (gs=0.0): pure policy at full authority — its
        # action × RESIDUAL_SCALE_MAX is the only thing driving the joints.
        # During fade: graceful crossfade, RL refines what BC learned.
        gait_neutral   = self._ctrl.gait_neutral_pose
        policy_target  = gait_neutral + action * self.RESIDUAL_SCALE_MAX
        target = (scaffold_joints * self.gait_scale
                  + policy_target * (1.0 - self.gait_scale))
        self._data.ctrl[:] = target
        # 2b. BC target — the action that, applied at gait_scale=0.0, would
        # produce the scaffold's joint targets exactly. Exposed in info for
        # supervised pretraining + (optional) auxiliary loss during RL.
        # Clipped to action space [-1, 1]; if a scaffold deviation exceeds
        # FULL residual authority, the BC target saturates rather than asks
        # for impossible actions.
        bc_target = (scaffold_joints - gait_neutral) / self.RESIDUAL_SCALE_MAX
        bc_target = np.clip(bc_target, -1.0, 1.0).astype(np.float32)
        mujoco.mj_step(self._model, self._data)
        self._sim_time   += self._dt
        self._step_count += 1

        # 3. Reward — tracking on locomotion cmd slots only.
        reward, reward_info = self._compute_reward(action)

        # 4. Update no-progress rolling buffer with current speed-along-cmd-dir.
        cmd_speed = math.hypot(float(self._cmd[0]), float(self._cmd[1]))
        if cmd_speed > 1e-6:
            cmd_dir_x = float(self._cmd[0]) / cmd_speed
            cmd_dir_y = float(self._cmd[1]) / cmd_speed
            vx_b, vy_b, _ = self._body_frame_linvel()
            speed_along_cmd = vx_b * cmd_dir_x + vy_b * cmd_dir_y
        else:
            speed_along_cmd = 0.0
        self._progress_buf[self._progress_idx] = speed_along_cmd
        self._progress_idx = (self._progress_idx + 1) % self.NO_PROGRESS_WINDOW
        if self._progress_idx == 0:
            self._progress_full = True

        # 5. Termination on fall, tilt overshoot, or no progress under non-zero cmd.
        body_z = float(self._data.qpos[2])
        qx, qy = float(self._data.qpos[4]), float(self._data.qpos[5])
        tilt_sq = qx*qx + qy*qy
        # Tilt-overshoot check: deviation from COMMANDED tilt, per-axis.
        # cmd_pitch and cmd_roll are sampled up to ±15° at stage=3, so an
        # absolute-tilt cap can't be the failure criterion — it would auto-
        # fail every high-tilt episode. Instead, fail when actual tilt exceeds
        # commanded tilt by more than TILT_DEV_LIMIT in either axis. Suspended
        # during the first TILT_DEV_GRACE steps so the bot can physically
        # rotate into the commanded tilt from its upright spawn pose.
        actual_pitch, actual_roll = self._body_pitch_roll()
        pitch_dev = abs(actual_pitch - float(self._cmd[3]))
        roll_dev  = abs(actual_roll  - float(self._cmd[4]))
        in_tilt_grace = self._step_count <= self.TILT_DEV_GRACE
        tilt_overshoot = ((not in_tilt_grace)
                          and (pitch_dev > self.TILT_DEV_LIMIT
                               or roll_dev > self.TILT_DEV_LIMIT))
        fell = ((body_z < self.MIN_Z)
                or tilt_overshoot
                or (tilt_sq > self.MAX_TILT_SQ))

        no_progress = False
        if (self._progress_full
                and self._step_count > self.NO_PROGRESS_GRACE
                and cmd_speed > 1e-6):
            avg_speed_along_cmd = float(self._progress_buf.mean())
            if avg_speed_along_cmd < self.NO_PROGRESS_FRAC * cmd_speed:
                no_progress = True

        terminated = fell or no_progress
        truncated  = False

        self._last_action = action.copy()

        if self.render_mode == "human":
            # In-process viewer (only valid for non-vec, non-worker uses such
            # as watch.py / pilot.py / single-env eval).
            self._render_human()
        if self.live_watch:
            # Out-of-process: write state to shared memory for live_viewer.py.
            self._write_live_state()

        info = {**reward_info, "body_z": body_z, "tilt_sq": float(tilt_sq),
                "no_progress": no_progress, "fell": fell,
                "bc_target": bc_target}
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Cmd sampling — stage-aware
    # ------------------------------------------------------------------
    def _sample_cmd(self):
        """Sample a cmd vector for this episode. Stage-locked slots stay zero;
        unlocked slots are drawn from their allowed range.

        Translation uses polar sampling — random heading + random magnitude in
        [SPEED_MIN_FRAC × MAX_SPEED, SPEED_MAX_FRAC × MAX_SPEED]. Tightened
        away from [0, MAX_SPEED] to give the policy a cleaner training
        distribution: no "stand still" episodes (which encourage do-nothing
        policies) and no full-MAX_SPEED episodes (which sit at the edge of
        scaffold capacity and look ragged).
        """
        mask = STAGE_CMD_MASK[self.stage]
        cmd  = np.zeros(self.CMD_DIM, dtype=np.float64)

        # Slots 0, 1 — translation, polar-sampled. Variable magnitude AND heading.
        if mask[0] and mask[1]:
            heading   = self.np_random.uniform(0.0, 2.0 * math.pi)
            magnitude = self.np_random.uniform(
                self.SPEED_MIN_FRAC * self._ctrl.MAX_SPEED,
                self.SPEED_MAX_FRAC * self._ctrl.MAX_SPEED,
            )
            cmd[0] = magnitude * math.cos(heading)
            cmd[1] = magnitude * math.sin(heading)

        # Remaining slots — independent uniform sampling.
        for slot in range(2, self.CMD_DIM):
            if mask[slot] == 0:
                continue
            lo, hi = self.SAMPLE_RANGES[slot]
            v = self.np_random.uniform(lo, hi)
            if slot == 2:
                v *= self._ctrl.MAX_YAW_RATE
            cmd[slot] = v

        return cmd

    # ------------------------------------------------------------------
    # Foot-contact analysis (for sliding / excess-contact penalties)
    # ------------------------------------------------------------------
    def _foot_contact_stats(self):
        """Update per-foot contact state and return current-step contact summary.

        Returns:
            n_contact (int): feet whose tip world-z < FOOT_CONTACT_Z this step.
            sliding (float): sum over in-contact feet of horizontal foot-tip
                speed (m/s) above SLIDING_DEADZONE. Foot-tip velocity =
                body-origin lin vel + omega × (R · local_offset).
            short_lifts (int): lift-off events this step where the prior
                contact lasted fewer than MIN_CONTACT_STEPS — i.e., the foot
                tapped the ground rather than producing a real stance.

        Side-effect: mutates `_foot_was_in_contact` and `_foot_contact_steps`.
        """
        n_contact   = 0
        sliding     = 0.0
        short_lifts = 0
        local_offset = self._FOOT_LOCAL_OFFSET
        for i, tib_id in enumerate(self._tibia_body_ids):
            R = self._data.xmat[tib_id].reshape(3, 3)
            r_world = R @ local_offset
            foot_world = self._data.xpos[tib_id] + r_world
            in_contact_now = bool(foot_world[2] < self.FOOT_CONTACT_Z)

            if in_contact_now:
                n_contact += 1
                mujoco.mj_objectVelocity(
                    self._model, self._data,
                    mujoco.mjtObj.mjOBJ_BODY, int(tib_id),
                    self._vel6_scratch, 0,   # 0 = world frame
                )
                ang_vel = self._vel6_scratch[:3]
                lin_vel = self._vel6_scratch[3:]
                foot_vel = lin_vel + np.cross(ang_vel, r_world)
                slip = math.hypot(foot_vel[0], foot_vel[1])
                sliding += max(0.0, slip - self.SLIDING_DEADZONE)
                self._foot_contact_steps[i] += 1
            else:
                # Lift-off detection: was in contact last step, not now.
                if self._foot_was_in_contact[i]:
                    if self._foot_contact_steps[i] < self.MIN_CONTACT_STEPS:
                        short_lifts += 1
                    self._foot_contact_steps[i] = 0
                # else: still airborne, contact_steps stays 0.
            self._foot_was_in_contact[i] = in_contact_now

        return n_contact, sliding, short_lifts

    def _foot_dev_total(self):
        """Sum of per-foot Euclidean distances (body frame) between actual
        foot tip and the scaffold's intended target this step.

        `self._latest_feet_body[i]` is the body-frame target position of foot
        i that the gait controller wanted on this step (set during step()
        before mj_step). The actual position is computed by transforming the
        post-step world-frame foot tip into the base-body frame.
        """
        R_body   = self._data.xmat[self._base_body_id].reshape(3, 3)
        body_pos = self._data.xpos[self._base_body_id]
        total = 0.0
        for i, tib_id in enumerate(self._tibia_body_ids):
            R_tib = self._data.xmat[tib_id].reshape(3, 3)
            foot_world = self._data.xpos[tib_id] + R_tib @ self._FOOT_LOCAL_OFFSET
            foot_body  = R_body.T @ (foot_world - body_pos)
            diff = foot_body - self._latest_feet_body[i]
            total += float(np.linalg.norm(diff))
        return total

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, action):
        # Measured kinematic state for tracking comparison.
        vx, vy, _    = self._body_frame_linvel()
        wz           = float(self._data.qvel[5])
        pitch, roll  = self._body_pitch_roll()
        height_delta = float(self._data.qpos[2]) - self.REF_BODY_Z
        actual = np.array([vx, vy, wz, pitch, roll, height_delta, 0.0, 0.0, 0.0],
                          dtype=np.float64)

        # IMPORTANT: tracking compares ALL body kinematics regardless of stage,
        # NOT just slots commanded by the current stage. Stage masks only what's
        # commanded during sampling; the reward always punishes unwanted IMU
        # deviation. Even in stage 1 (where cmd[wz]=cmd[pitch]=cmd[roll]=cmd[height]=0),
        # the bot is penalized for yawing, pitching, rolling, or bobbing —
        # because cmd is zero on those slots and any actual deviation creates
        # tracking error. This is what makes the bot learn smooth, controlled
        # locomotion instead of jerky lurching that happens to track velocity.
        # Errors are normalized per-slot (see _err_inv_scales) so each axis
        # contributes commensurately rather than being dominated by velocity.
        err = (self._cmd - actual) * self._err_inv_scales * REWARD_TRACK_MASK

        # Gaussian-shaped tracking reward, peaks at 1.0 when cmd matches actual.
        tracking = float(math.exp(-float(np.dot(err, err))))

        # Action smoothness — penalize fast changes in the policy's residual
        # between consecutive ticks. Encourages smooth, gradual joint motion.
        action_delta = action - self._last_action
        action_rate  = self.ACTION_RATE_W * float(np.dot(action_delta, action_delta))

        # Body angular-velocity damping — penalize WOBBLE that doesn't show up
        # cleanly in absolute pitch/roll tracking. Excludes yaw rate (qvel[5])
        # since that's already tracked via wz. Targets qvel[3] (roll rate) and
        # qvel[4] (pitch rate) which are unbounded oscillation modes if not
        # damped.
        body_angvel = self._data.qvel[3:5]   # roll_rate, pitch_rate (world)
        angvel_pen  = self.ANGVEL_W * float(np.dot(body_angvel, body_angvel))

        # Drift penalty — linear in |actual| on yaw rate, pitch, roll. Activates
        # only when the corresponding cmd is near zero (gates on cmd magnitude).
        # Linear shape gives a sharp gradient near zero that the squared
        # tracking term can't provide; this is what drives the policy to walk
        # straight instead of slowly drifting/turning.
        no_yaw_cmd   = max(0.0, 1.0 - abs(self._cmd[2]) / self.YAW_GATE)
        no_pitch_cmd = max(0.0, 1.0 - abs(self._cmd[3]) / self.TILT_GATE)
        no_roll_cmd  = max(0.0, 1.0 - abs(self._cmd[4]) / self.TILT_GATE)
        drift_pen = (self.YAW_DRIFT_W   * no_yaw_cmd   * abs(wz)
                   + self.PITCH_DRIFT_W * no_pitch_cmd * abs(pitch)
                   + self.ROLL_DRIFT_W  * no_roll_cmd  * abs(roll))

        # Linear speed-tracking penalty. Sharp gradient near zero error breaks
        # the "stand still" local optimum that the gaussian alone allows.
        cmd_speed    = math.hypot(float(self._cmd[0]), float(self._cmd[1]))
        actual_speed = math.hypot(vx, vy)
        speed_track_pen = (self.SPEED_TRACK_W
                           * abs(cmd_speed - actual_speed)
                           / self._ctrl.MAX_SPEED)

        # Foot-contact shaping. Discourage four cheating modes:
        # (1) Sliding: any foot in ground contact while moving horizontally.
        #     Real walking plants the foot, friction-locked. Sliding lets the
        #     bot "skate" forward without lifting its legs.
        # (2) Excess simultaneous contact: > 3 feet on the ground at once.
        #     Encourages a stepping rhythm. We do NOT penalize <3 feet contact
        #     so emergent bounding/jumping/rolling gaits stay reachable.
        # (3) Sustained airborne: zero feet down for many consecutive steps.
        #     A brief flight phase (a hop) is fine; permanent flight is not.
        # (4) Foot-tapping: lift-off after a stance shorter than
        #     MIN_CONTACT_STEPS. Otherwise the policy could game (1) and (2)
        #     by briefly touching each foot and lifting before the slip
        #     accumulates or contact-count rises.
        n_contact, sliding_speed, short_lifts = self._foot_contact_stats()
        sliding_pen        = self.SLIDING_W * sliding_speed
        excess_contact_pen = self.EXCESS_CONTACT_W * max(0, n_contact - 3)
        airborne_pen       = self.AIRBORNE_W if n_contact == 0 else 0.0
        short_contact_pen  = self.SHORT_CONTACT_W * short_lifts

        # Foot-position deviation: keep feet near the scaffold's intended
        # body-frame stance positions. Soft constraint — RL can still discover
        # better gaits, but random drift away from the scaffold's stance gets
        # corrected. Auto-gates: during scaffold-strong (gait_scale=1.0) the
        # actual foot positions ≈ scaffold positions, so dev ≈ 0.
        foot_dev_total = self._foot_dev_total()
        foot_dev_pen   = self.FOOT_DEV_W * foot_dev_total

        reward = (tracking + self.SURVIVE_BONUS
                  - action_rate - angvel_pen - drift_pen
                  - speed_track_pen - sliding_pen - excess_contact_pen
                  - airborne_pen - short_contact_pen
                  - foot_dev_pen)

        # Stage 4 stretch-goal bonuses (placeholders; tune when we get there).
        novelty = 0.0
        if self.stage == 4:
            novelty = self._novelty_bonus()
            reward += novelty

        return reward, {
            "tracking_reward":    tracking,
            "action_rate_pen":    action_rate,
            "angvel_pen":         angvel_pen,
            "drift_pen":          drift_pen,
            "speed_track_pen":    speed_track_pen,
            "sliding_pen":        sliding_pen,
            "excess_contact_pen": excess_contact_pen,
            "airborne_pen":       airborne_pen,
            "short_contact_pen":  short_contact_pen,
            "foot_dev_pen":       foot_dev_pen,
            "foot_dev_total":     foot_dev_total,
            "n_contact":          n_contact,
            "short_lifts":        short_lifts,
            "novelty_bonus":      novelty,
        }

    def _novelty_bonus(self):
        """Stage 4 only. Reward unusual achievements: high speeds, sustained
        inverted body orientation (rolling), etc. Placeholder until we tune."""
        speed = math.hypot(*self._body_frame_linvel()[:2])
        # Linear bonus for exceeding scaffold's max speed.
        speed_excess = max(0.0, speed - self._ctrl.MAX_SPEED)
        return 0.5 * speed_excess

    # ------------------------------------------------------------------
    # Helpers — kinematic readouts
    # ------------------------------------------------------------------
    def _body_frame_linvel(self):
        """World-frame body linear velocity rotated into body frame (yaw only)."""
        qw = float(self._data.qpos[3])
        qx = float(self._data.qpos[4])
        qy = float(self._data.qpos[5])
        qz = float(self._data.qpos[6])
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        c, s = math.cos(yaw), math.sin(yaw)
        vx_w = float(self._data.qvel[0])
        vy_w = float(self._data.qvel[1])
        vz_w = float(self._data.qvel[2])
        return ( c * vx_w + s * vy_w,
                -s * vx_w + c * vy_w,
                 vz_w)

    def _body_pitch_roll(self):
        """Body pitch & roll relative to world (rad).

        Pitch is negated from the standard ZYX-Euler formula so it follows the
        project's aerospace-style convention: positive pitch = nose UP. The
        raw asin() of (qw*qy - qz*qx) follows right-hand-rule math, where
        positive Y-axis rotation tilts the body's +X (forward) toward -Z
        (down) — i.e., standard math says positive pitch = nose DOWN. The
        gait controller and CLAUDE.md docs both use "nose up = +"; flipping
        the sign here makes the env's measurement match that convention so
        cmd_pitch tracking, drift_pen, and tilt-deviation termination all
        agree.
        Roll: no flip needed — for X-axis rotation, standard math and the
        project's "left side up = +" convention agree on sign.
        """
        qw = float(self._data.qpos[3])
        qx = float(self._data.qpos[4])
        qy = float(self._data.qpos[5])
        qz = float(self._data.qpos[6])
        sinp  = 2.0 * (qw * qy - qz * qx)
        sinp  = max(-1.0, min(1.0, sinp))
        pitch = -math.asin(sinp)   # negate: standard-math → aerospace convention
        sinr  = 2.0 * (qw * qx + qy * qz)
        cosr  = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll  = math.atan2(sinr, cosr)
        return pitch, roll

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self):
        """Build the policy's obs via the shared envs.obs_layout module so
        the gym env and the JAX env always emit byte-identical layouts. We
        use JAX-convention raw signals (qpos-based quat, qvel-based gyro,
        zero accel) because training happens in the JAX env — the gym env
        only exists for watch / eval / record, and must produce the same
        obs distribution the policy was trained against.
        """
        from envs.obs_layout import compose_obs
        qpos = self._data.qpos
        qvel = self._data.qvel
        scaffold_hint = self._latest_feet_body.flatten()
        phase = self._ctrl.get_phase(self._sim_time)
        body_linvel = np.asarray(self._body_frame_linvel(), dtype=np.float32)
        # Stage 4: drop privileged body_linvel. Later refinement may
        # inject IMU-only estimation noise instead.
        if self.stage == 4:
            body_linvel = np.zeros(3, dtype=np.float32)
        return compose_obs(
            joint_pos       = qpos[7:25],
            joint_vel       = qvel[6:24],
            imu_quat        = qpos[3:7],                       # JAX convention
            imu_gyro        = qvel[3:6],                       # JAX convention
            imu_accel       = np.zeros(3, dtype=np.float32),   # JAX convention
            scaffold_hint   = scaffold_hint,
            phase_sc        = np.array([math.sin(2 * math.pi * phase),
                                        math.cos(2 * math.pi * phase)],
                                       dtype=np.float32),
            cmd             = self._cmd,
            body_linvel     = body_linvel,
            concat_fn       = np.concatenate,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._model)
            self._renderer.update_scene(self._data)
            return self._renderer.render()

    def _render_human(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        self._viewer.sync()

    # ------------------------------------------------------------------
    # Live state mirror (shared memory) — for an external viewer process
    # ------------------------------------------------------------------
    # Layout: [qpos (nq) | qvel (nv) | sim_time (1)] as float64.
    def _init_live_shm(self):
        nq, nv = self._model.nq, self._model.nv
        nbytes = (nq + nv + 1) * 8
        # Clean up any stale region from a previous crashed run.
        try:
            old = shared_memory.SharedMemory(name=LIVE_SHM_NAME)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass
        try:
            self._live_shm = shared_memory.SharedMemory(
                name=LIVE_SHM_NAME, create=True, size=nbytes
            )
            self._live_buf = np.ndarray(
                (nq + nv + 1,), dtype=np.float64, buffer=self._live_shm.buf
            )
            print(f"[env-0] live state shared memory: '{LIVE_SHM_NAME}' "
                  f"({nbytes} bytes). Run `mjpython live_viewer.py` to watch.")
        except Exception as e:
            print(f"[env-0] could not create live shm: {e}")
            self._live_shm = None
            self._live_buf = None

    def _write_live_state(self):
        if self._live_buf is None:
            return
        nq, nv = self._model.nq, self._model.nv
        self._live_buf[:nq]            = self._data.qpos
        self._live_buf[nq:nq+nv]       = self._data.qvel
        self._live_buf[nq+nv]          = self._sim_time

    def _close_live_shm(self):
        if self._live_shm is not None:
            try:
                self._live_shm.close()
                self._live_shm.unlink()
            except Exception:
                pass
            self._live_shm = None
            self._live_buf = None

    # Setter for gait_scale that propagates correctly through gym wrappers.
    # See the note next to `self.gait_scale = 1.0` for why this exists.
    def set_gait_scale(self, value):
        self.gait_scale = float(value)

    def set_stage(self, value):
        self.stage = int(value)

    def close(self):
        self._close_live_shm()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
