"""
watch_demo_jax.py — JAX/Brax counterpart of watch_demo.py.

Loads a Brax-format `(normalizer, policy, value)` pickle (produced by
`pretrain_bc_jax.py` or `train_jax.py`), drives the gym `HexapodEnv`
through the same DEMO_PHASES preset cmd script, and renders via
mujoco.viewer. Lets you compare BC-only vs PPO-refined policies before
committing to fine-tuning runs.

Usage (Linux/Windows — viewer works in both):
    python watch_demo_jax.py                                       # latest
    python watch_demo_jax.py --params <path>/params.pkl            # explicit
    python watch_demo_jax.py --stochastic                          # sample actions
    python watch_demo_jax.py --gait-scale 0.0                      # pure policy

For BC eval before fine-tuning:
    python watch_demo_jax.py --params checkpoints/bc_pretrained_jax/params.pkl

For trained policy:
    python watch_demo_jax.py --params checkpoints/<run>/final/params.pkl
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# Enable persistent JAX compilation cache before any JAX op runs.
from chain_train import enable_jax_cache

enable_jax_cache()

from brax.training.acme import running_statistics
from brax.training.agents.ppo.networks import make_ppo_networks
from chain_train import ACTION_SPACE as CHAIN_ACTION_SPACE

# Pull session config from chain_train.py so watch defaults stay in
# sync with the lineage you're training. Edit BASE_NAME / ACTION_SPACE
# in chain_train.py, both scripts pick it up automatically.
from chain_train import BASE_NAME as CHAIN_BASE_NAME
from chain_train import MODEL_PATH as CHAIN_MODEL_PATH
from chain_train import RENDER_MODEL_PATH as CHAIN_RENDER_MODEL_PATH

# Shared cmd-vector demo script lives at scripts/demo_phases.py — both
# this active JAX viewer and the legacy SB3 watch_demo.py import from it.
from demo_phases import (
    DEMO_PHASES,
    DEMO_PHASES_PAPER,
    DEMO_PHASES_PAPER_STANCE,
    DEMO_PHASES_SHOWCASE,
    INTERACTIVE_TESTS,
    print_interactive_help,
)

from envs.hexapod_env import HexapodEnv

# _patch_obs_to_jax monkey-patch removed 2026-05-12 — the gym env now
# uses the shared envs/obs_layout.py module so it produces the same obs
# the JAX env did during training. No runtime override needed; any future
# obs schema change is automatically picked up by both envs.


# ============================================================================
# Brax inference loader
# ============================================================================
def _build_inference_fn(obs_size: int, act_size: int,
                        deterministic: bool,
                        log_std_override: float | None = None):
    """Build a callable obs → action using the Brax PPO policy network.

    Returns a callable with signature `infer(norm, policy, obs, rng)`.
    `rng` is ignored when deterministic=True; must be a fresh PRNGKey
    each call when deterministic=False.

    `log_std_override` (when set) replaces the policy's baked-in log_std
    output with a constant value. Useful for visualizing what bigger
    exploration noise looks like — e.g., set to -1.5 (std≈0.22, ~12°
    per-joint jitter equivalent) to see the noise pattern clearly even
    though training uses log_std=-4 which is invisible.
    """
    # Match the trained policy's architecture (v11+ uses paper-matching
    # 256/128/64; pre-v11 used Brax default 32/32/32/32).
    from scripts.pretrain_bc_jax import POLICY_HIDDEN_LAYERS
    networks = make_ppo_networks(
        observation_size=obs_size,
        action_size=act_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=POLICY_HIDDEN_LAYERS,
    )

    @jax.jit
    def _infer(normalizer_params, policy_params, obs, rng):
        out = networks.policy_network.apply(
            normalizer_params, policy_params, obs)
        mean = out[..., :act_size]
        log_std = out[..., act_size:]
        if log_std_override is not None:
            log_std = jnp.full_like(log_std, log_std_override)
        if deterministic:
            return mean
        std = jnp.exp(log_std)
        return mean + std * jax.random.normal(rng, mean.shape)
    return _infer


# ============================================================================
# Checkpoint discovery
# ============================================================================
def latest_jax_params(root="checkpoints", lineage_prefix: str | None = None):
    """Find the most-recently-modified `<root>/<lineage_prefix>*/.../params.pkl`.
    If `lineage_prefix` is None, scans all of `root` recursively.
    Returns the matching path (str) or None.
    """
    if not os.path.isdir(root):
        return None
    if lineage_prefix is None:
        pattern = os.path.join(root, "**", "params.pkl")
    else:
        pattern = os.path.join(root, f"{lineage_prefix}*", "**", "params.pkl")
    candidates = []
    for path in glob.glob(pattern, recursive=True):
        candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


# ============================================================================
# Main
# ============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Render a JAX/Brax policy through a preset cmd script.")
    p.add_argument("--params", default=None,
                   help="path to a (normalizer, policy, value) pickle "
                        "(overrides --lineage / --bc)")
    p.add_argument("--lineage", default=CHAIN_BASE_NAME,
                   help=f"lineage prefix; finds the latest "
                        f"checkpoints/<prefix>*/.../params.pkl "
                        f"(default: chain_train.BASE_NAME={CHAIN_BASE_NAME!r}).")
    p.add_argument("--bc", action="store_true",
                   help="watch the BC bootstrap checkpoint for the lineage "
                        "instead of the latest training iter. Loads "
                        "checkpoints/<lineage>_iter0_bc/final/params.pkl. "
                        "Useful for A/B comparison: 'how much did training "
                        "actually improve on BC?'.")
    p.add_argument("--iter", type=int, default=None,
                   help="watch a specific iter of the lineage. Loads "
                        "checkpoints/<lineage>_iter<N>/final/params.pkl. "
                        "Useful for rolling back to a known-good checkpoint "
                        "after a crash, or A/B comparing across the lineage.")
    p.add_argument("--model", default=CHAIN_MODEL_PATH,
                   help=f"MJCF path (default: chain_train.MODEL_PATH={CHAIN_MODEL_PATH!r})")
    p.add_argument("--stage", type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument("--gait-scale", type=float, default=0.0,
                   help="scaffold weight (0.0 = pure policy, 1.0 = full scaffold)")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions stochastically (default deterministic). "
                        "Implies the saved policy's baked-in log_std unless "
                        "--noise-std overrides it.")
    p.add_argument("--noise-std", type=float, default=None,
                   help="override the policy's log_std at inference time "
                        "(only effective with --stochastic). Examples: "
                        "-4.0 = training default (~0.4mm foot jitter, invisible), "
                        "-1.5 = ~4mm foot jitter (visible), "
                        "-1.0 = ~7mm jitter (very visible exploration).")
    p.add_argument("--amp-discriminator", type=str, default=None,
                   help="optional path to a trained AMP discriminator pickle. "
                        "When set, prints the per-step naturalness score "
                        "alongside phase transitions (rolling average). "
                        "+1 = looks like prior data; -1 = looks unnatural.")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed multiplier (1.0 = real time, 0 = uncapped)")
    p.add_argument("--action-space", type=str, default=CHAIN_ACTION_SPACE,
                   choices=["joint", "foot"],
                   help=f"must match the trained policy "
                        f"(default: chain_train.ACTION_SPACE={CHAIN_ACTION_SPACE!r}). "
                        f"foot = policy output is (6,3) foot residual; we run "
                        f"IK to convert to joint residual before feeding the "
                        f"gym env.")
    p.add_argument("--interactive", action="store_true",
                   help="enable keyboard-driven test mode in the viewer. "
                        "Press digits 1-9 in the mujoco viewer window to "
                        "switch between motion / stance / arc tests on the "
                        "fly; each test auto-cycles through related sub-cases. "
                        "Press h to reprint the help. Useful for v7+ stance "
                        "verification without re-launching.")
    p.add_argument("--demo", type=str, default="legacy",
                   choices=["legacy", "paper", "paper_stance", "showcase"],
                   help="which demo schedule to run. 'legacy' = original 17-phase "
                        "set with pitch/roll/height/width overlays (for v1-v3). "
                        "'paper' = v4-v5 schedule (vx, vy, wz) only. "
                        "'paper_stance' = v7+ stance verification. "
                        "'showcase' = v8+ full motion repertoire + stance, "
                        "designed for video recording (~95s).")
    p.add_argument("--render-model", default=CHAIN_RENDER_MODEL_PATH,
                   help=f"MJCF used ONLY for the viewer window. Inference still "
                        f"uses --model. Default (chain_train.RENDER_MODEL_PATH): "
                        f"{CHAIN_RENDER_MODEL_PATH!r}. Pass the same path as "
                        f"--model to skip the dual-render overhead.")
    # foot_residual_scale_max is sourced from the env's EnvParams below
    # (one source of truth — same value the JAX env uses at training time).
    args = p.parse_args()

    # Resolve params. Priority: --params > --iter > --bc > --lineage.
    if args.params:
        params_path = args.params
    elif args.iter is not None:
        params_path = os.path.join("checkpoints",
                                   f"{args.lineage}_iter{args.iter}",
                                   "final", "params.pkl")
        if not os.path.exists(params_path):
            print(f"No checkpoint at {params_path}")
            sys.exit(1)
    elif args.bc:
        # Load the BC bootstrap checkpoint for the current lineage.
        bc_path = os.path.join("checkpoints", f"{args.lineage}_iter0_bc",
                               "final", "params.pkl")
        if not os.path.exists(bc_path):
            # Fall back to the un-prefixed BC pickle if the chain hasn't
            # bootstrapped this lineage yet.
            fallback = os.path.join("checkpoints", "bc_pretrained_jax",
                                    "params.pkl")
            if os.path.exists(fallback):
                print(f"  no {bc_path}; using {fallback}")
                params_path = fallback
            else:
                print(f"No BC checkpoint at {bc_path} or {fallback}.")
                sys.exit(1)
        else:
            params_path = bc_path
    elif args.lineage:
        params_path = latest_jax_params(lineage_prefix=args.lineage)
        if params_path is None:
            print(f"No checkpoints found matching "
                  f"checkpoints/{args.lineage}*/.../params.pkl")
            sys.exit(1)
    else:
        params_path = latest_jax_params()
    if params_path is None or not os.path.exists(params_path):
        print("No JAX params.pkl found. Pass --params/--lineage/--bc or train first.")
        sys.exit(1)
    print(f"Loading params: {params_path}")
    with open(params_path, "rb") as f:
        loaded = pickle.load(f)
    if isinstance(loaded, tuple) and len(loaded) == 3:
        normalizer_params, policy_params, _value_params = loaded
    elif isinstance(loaded, tuple) and len(loaded) == 2:
        # Backwards-compat: (norm, policy) without value.
        normalizer_params, policy_params = loaded
    else:
        print(f"Unrecognized params format: {type(loaded)} "
              f"(expected 2- or 3-tuple)")
        sys.exit(1)

    # Detect this policy's expected obs dim from the normalizer's mean shape.
    # Backwards-compat: pre-v24 policies were trained on 78-dim obs; v24+
    # on 114-dim. If the policy expects fewer dims than the env currently
    # produces, slice the obs to the first N dims at inference time.
    # The new dims (joint_torque, joint_pos_error) are APPENDED at the end
    # of obs_layout.py's slot order, so slicing to first N == old layout.
    try:
        policy_obs_dim = int(normalizer_params.mean.shape[-1])
    except AttributeError:
        # Older normalizer pytree variants — try first leaf. (`jax` is
        # imported at module level; don't re-import here or Python will
        # treat it as a local var and shadow the outer reference.)
        leaves = jax.tree_util.tree_leaves(normalizer_params)
        policy_obs_dim = int(leaves[0].shape[-1])
    print(f"Policy expects obs dim = {policy_obs_dim}")

    # Determine whether we need a separate render model.
    # None / same path → use the env's own viewer (existing path, zero overhead).
    # Interactive mode FORCES our own viewer so we can attach key_callback —
    # the env's internal viewer doesn't expose key handling. If no --render-model
    # is given alongside --interactive, we just point the dual-render at the
    # inference model itself (zero polish but keyboard works).
    render_model_path = args.render_model
    if args.interactive and render_model_path is None:
        render_model_path = args.model
    dual_render = (render_model_path is not None and
                   (args.interactive or
                    os.path.realpath(render_model_path) != os.path.realpath(args.model)))

    import mujoco
    import mujoco.viewer as mj_viewer

    if dual_render:
        # Build env without an internal viewer; we'll drive a separate one.
        env = HexapodEnv(stage=args.stage, render_mode=None, model_path=args.model)
    else:
        # Build env (gym side has the renderer).
        env = HexapodEnv(stage=args.stage, render_mode="human", model_path=args.model)
    env.gait_scale = args.gait_scale
    # No need to monkey-patch obs anymore — both envs share
    # envs/obs_layout.py so the gym env emits the same layout the JAX env
    # produced during training.
    # Source the foot residual scale from the JAX env's params so it tracks
    # any changes to envs/hexapod_env_jax.py without needing a separate flag.
    from envs import hexapod_env_jax as hex_jax
    _jax_params = hex_jax.make_env_params(args.model)
    foot_residual_scale = float(_jax_params.foot_residual_scale_max)
    # Disable training-time terminations that fire on cmd discontinuities at
    # demo-phase boundaries. The TILT_DEV_LIMIT (5°) is what kills the bot
    # when phase 11 (pitch wobble) ends with actual_pitch≈7° and phase 12
    # (roll wobble) instantly drops cmd_pitch to 0 — the dev exceeds the
    # limit before the bot can settle. NO_PROGRESS triggers similarly when
    # cmds change mid-stride.
    env.TILT_DEV_LIMIT = math.pi              # effectively disable
    env.MAX_TILT_SQ    = 4.0                  # ~2x absolute fence
    env.MIN_Z          = 0.0                  # don't terminate on ground touch
    env.NO_PROGRESS_GRACE = 1_000_000_000     # disable no-progress check

    # Build the secondary render model if requested.
    # nq must match: both models must have the same DoF layout (18 joints +
    # 7-element freejoint = 25 qpos). If the user points at an unrelated MJCF
    # (wrong robot, wrong joint count) we catch it here rather than silently
    # writing garbage into qpos.
    render_mjmodel = None
    render_mjdata  = None
    render_viewer  = None
    if dual_render:
        render_mjmodel = mujoco.MjModel.from_xml_path(render_model_path)
        inf_nq = env._model.nq
        ren_nq = render_mjmodel.nq
        if inf_nq != ren_nq:
            print(f"ERROR: joint count mismatch between inference model "
                  f"({args.model}, nq={inf_nq}) and render model "
                  f"({render_model_path}, nq={ren_nq}). "
                  f"Make sure --render-model has the same DoF layout.")
            sys.exit(1)
        render_mjdata = mujoco.MjData(render_mjmodel)
        # Seed the render data with the inference model's initial qpos so the
        # viewer opens in the robot's standing pose rather than the XML default.
        render_mjdata.qpos[:] = env._data.qpos
        mujoco.mj_forward(render_mjmodel, render_mjdata)
        # Launch the passive viewer bound to the render model's data.
        # All subsequent syncs write inference qpos/qvel then call mj_forward
        # so the viewer always reflects the simple-model rollout state.
        # --- Interactive-mode key callback ---
        # Wrapped in a list so it's mutable from the closure.
        active_test_key = ["1"]   # default to forward walk
        active_test_t0  = [time.time()]

        def _key_callback(key: int):
            try:
                ch = chr(key).lower() if 0 <= key < 0x110000 else None
            except ValueError:
                ch = None
            if ch is None:
                return
            if ch == "h" or ch == "?":
                print_interactive_help()
                return
            if ch == "r":
                # Hard reset: re-init env to neutral standing pose. Clears
                # stuck legs / weird configurations / accumulated drift.
                env.reset()
                # Restart the active test from t=0 so the cmd schedule
                # picks up cleanly post-reset.
                active_test_t0[0] = time.time()
                print("  >>> [r] RESET bot to neutral pose")
                return
            if ch == "p":
                print(f"  bot world pos = "
                      f"({env._data.qpos[0]:+.3f}, {env._data.qpos[1]:+.3f}, "
                      f"{env._data.qpos[2]:+.3f})")
                return
            if ch in INTERACTIVE_TESTS:
                active_test_key[0] = ch
                active_test_t0[0]  = time.time()
                label, dur, _ = INTERACTIVE_TESTS[ch]
                print(f"  >>> [{ch}] {label}  (loop {dur:.0f}s)")

        if args.interactive:
            render_viewer = mj_viewer.launch_passive(
                render_mjmodel, render_mjdata, key_callback=_key_callback)
            print_interactive_help()
        else:
            render_viewer = mj_viewer.launch_passive(render_mjmodel, render_mjdata)
        print(f"Render model:  {render_model_path}  (nq={ren_nq})")

    env_obs_size = env.observation_space.shape[0]
    # Build the network with the OBS DIM THE POLICY EXPECTS — not the env's
    # current obs dim. If they differ (e.g., older policy seeing newer env),
    # we slice obs[:policy_obs_dim] each step to bridge the gap.
    obs_size = policy_obs_dim
    if env_obs_size != policy_obs_dim:
        print(f"NOTE: env obs is {env_obs_size}-dim but policy is "
              f"{policy_obs_dim}-dim; slicing obs[:{policy_obs_dim}] each step "
              f"(legacy-policy backwards-compat).")
    act_size = env.action_space.shape[0]
    infer = _build_inference_fn(obs_size, act_size,
                                deterministic=not args.stochastic,
                                log_std_override=args.noise_std)
    # Per-step RNG for stochastic inference; fresh subkey each step so
    # successive noise samples are independent (training-time behavior).
    infer_rng = jax.random.PRNGKey(0)

    demo_phases = {
        "showcase":      DEMO_PHASES_SHOWCASE,
        "paper":         DEMO_PHASES_PAPER,
        "paper_stance":  DEMO_PHASES_PAPER_STANCE,
        "legacy":        DEMO_PHASES,
    }[args.demo]
    total_dur = sum(d for _, d, _ in demo_phases)
    print(f"\nModel:       {args.model}")
    if dual_render:
        print(f"Render model: {render_model_path}")
    print(f"Stage:       {args.stage}")
    print(f"gait_scale:  {env.gait_scale:.2f}")
    print(f"Action mode: {'stochastic' if args.stochastic else 'deterministic'}")
    if args.interactive:
        print("Mode:        INTERACTIVE (keyboard-driven, default test = [1])")
    else:
        print(f"Demo:        {args.demo} schedule, {len(demo_phases)} phases, {total_dur:.0f}s per loop")
    print(f"MAX_SPEED:   {env._ctrl.MAX_SPEED:.4f} m/s")
    print(f"MAX_YAW:     {env._ctrl.MAX_YAW_RATE:.4f} rad/s\n")

    obs, _ = env.reset()
    # Settle with a small in-distribution forward walk (matches watch_demo.py
    # rationale — env's _sample_cmd never produces cmd=0 so the policy was
    # never trained on cmd=0 observations).
    settle_speed = (env.SPEED_MIN_FRAC + 0.05) * env._ctrl.MAX_SPEED
    env._cmd = np.array([settle_speed, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(200):
        obs, *_ = env.step(zero_action)

    # Optional AMP discriminator scoring
    amp_score_fn = None
    prev_amp_state = None
    amp_score_window = []      # rolling window of scores for moving average
    if args.amp_discriminator is not None:
        from amp.discriminator import Discriminator
        with open(args.amp_discriminator, "rb") as f:
            amp_disc_params = pickle.load(f)
        amp_disc_module = Discriminator()
        @jax.jit
        def amp_score_fn(transition):
            return amp_disc_module.apply(amp_disc_params, transition[None, :])[0]
        print(f"\nAMP discriminator loaded: {args.amp_discriminator}")
        # Build helper to grab the AMP state vector from the gym env.
        def _gym_amp_state():
            qpos = env._data.qpos
            qvel = env._data.qvel
            joint_pos = qpos[7:25]
            joint_vel = qvel[6:24]
            body_linvel = np.asarray(env._body_frame_linvel(), dtype=np.float32)
            body_angvel = qvel[3:6]
            body_height = qpos[2:3]
            R_body = env._data.xmat[env._base_body_id].reshape(3, 3)
            body_pos = env._data.xpos[env._base_body_id]
            foot_zs = []
            for tib_id in env._tibia_body_ids:
                R = env._data.xmat[tib_id].reshape(3, 3)
                foot_world = env._data.xpos[tib_id] + R @ env._FOOT_LOCAL_OFFSET
                foot_body = R_body.T @ (foot_world - body_pos)
                foot_zs.append(foot_body[2])
            return np.concatenate([joint_pos, joint_vel, body_linvel,
                                    body_angvel, body_height,
                                    np.asarray(foot_zs, dtype=np.float32)])
        prev_amp_state = _gym_amp_state()

    sim_t = 0.0
    last_phase_idx = -1
    wall_t0 = time.time()

    while True:
        # Exit cleanly when the user closes the viewer window (mesh viewer
        # is the user-facing window when --render-model is set).
        if render_viewer is not None and not render_viewer.is_running():
            break
        if args.interactive:
            # Interactive mode: read the active test (set by key_callback),
            # use its cmd_fn at the wall-clock time since it became active.
            test_key = active_test_key[0]
            label, loop_dur, fn = INTERACTIVE_TESTS[test_key]
            t_in_seq = (time.time() - active_test_t0[0]) % loop_dur
            cmd = fn(t_in_seq, env._ctrl)
        else:
            t_in_cycle = sim_t % total_dur
            cumulative = 0.0
            phase_idx  = 0
            for i, (_, dur, _) in enumerate(demo_phases):
                if t_in_cycle < cumulative + dur:
                    phase_idx = i
                    break
                cumulative += dur

            label, dur, fn = demo_phases[phase_idx]
            t_in_phase = t_in_cycle - cumulative
            cmd = fn(t_in_phase, env._ctrl)

            if phase_idx != last_phase_idx:
                amp_avg_str = ""
                if amp_score_fn is not None and amp_score_window:
                    amp_avg_str = f"  amp_avg={np.mean(amp_score_window):+.3f}"
                    amp_score_window.clear()
                print(f"  t={sim_t:6.1f}s  [{phase_idx+1:2d}/{len(demo_phases)}]  "
                      f"{label}  ({dur:.1f}s){amp_avg_str}")
                last_phase_idx = phase_idx

        # Override env cmd + refresh scaffold_hint cache (same rationale as
        # watch_demo.py — preserves the (cmd, scaffold_hint) pairing the
        # policy was trained on).
        env._cmd = cmd
        env._latest_feet_body = env._ctrl.compute_foot_targets(
            env._cmd, env._sim_time).astype(np.float32)
        obs = env._get_obs()

        # JAX inference: obs → action. Add batch axis for the network.
        # Split a fresh subkey per step so stochastic mode actually varies.
        infer_rng, sub = jax.random.split(infer_rng)
        # Slice to policy's expected obs dim (no-op when env_obs_size matches).
        obs_for_policy = obs[:policy_obs_dim]
        action_jax = infer(normalizer_params, policy_params,
                           jnp.asarray(obs_for_policy)[None, :], sub)[0]
        action = np.asarray(action_jax)

        # Foot-space action mode: policy outputs (6, 3) foot residual in
        # body frame. Run IK to get joint targets, then express as joint
        # residual relative to gait_neutral so the gym env (which expects
        # a joint residual scaled by RESIDUAL_SCALE_MAX) receives the
        # right thing. Mirrors what hex_jax.step_foot does inside the
        # JAX env at training time.
        if args.action_space == "foot":
            foot_residual = action.reshape(6, 3) * foot_residual_scale
            feet_body_target = env._latest_feet_body + foot_residual
            joint_target = env._ctrl.body_to_joints(feet_body_target)
            # Convert absolute joint target → joint residual the gym env wants.
            action = (joint_target - env._ctrl.gait_neutral_pose) / env.RESIDUAL_SCALE_MAX
            action = np.clip(action, -1.0, 1.0).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        sim_t += env._dt

        # Mirror inference state into the render model and tick the viewer.
        # mj_forward recomputes FK/contacts so the mesh geoms appear in the
        # correct pose. We copy nq/nv exactly — the joint-count check above
        # guarantees they align.
        if render_viewer is not None and render_viewer.is_running():
            render_mjdata.qpos[:] = env._data.qpos
            render_mjdata.qvel[:] = env._data.qvel
            mujoco.mj_forward(render_mjmodel, render_mjdata)
            render_viewer.sync()

        # Discriminator score the (prev, current, cmd) AMP transition.
        # cmd-conditional disc requires the cmd active when prev_amp_state
        # was observed — same convention as prior_data.py / training env.
        if amp_score_fn is not None:
            cur_amp = _gym_amp_state()
            from amp.discriminator import CMD_DIM_FOR_DISC
            cmd_disc = np.asarray(cmd[:CMD_DIM_FOR_DISC],
                                  dtype=np.float32)
            transition = jnp.asarray(np.concatenate(
                [prev_amp_state, cur_amp, cmd_disc]))
            score = float(amp_score_fn(transition))
            amp_score_window.append(score)
            prev_amp_state = cur_amp

        if args.speed > 0:
            target = wall_t0 + sim_t / args.speed
            lag = target - time.time()
            if lag > 0:
                time.sleep(lag)

        if terminated or truncated:
            tag = "FELL" if terminated else "TIMEOUT"
            print(f"  [{tag}] at t={sim_t:5.1f}s  in phase '{label}'  "
                  f"track={info.get('tracking_reward', 0):.2f} "
                  f"drift={info.get('drift_pen', 0):.3f}")
            obs, _ = env.reset()
            sim_t = 0.0
            last_phase_idx = -1
            wall_t0 = time.time()
            # Re-seed render data after reset so the mesh viewer snaps to the
            # correct standing pose rather than drifting to a stale state.
            if render_mjdata is not None:
                render_mjdata.qpos[:] = env._data.qpos
                render_mjdata.qvel[:] = env._data.qvel
                mujoco.mj_forward(render_mjmodel, render_mjdata)


if __name__ == "__main__":
    main()
    # Hard-exit so any lingering GLFW / pygame / JAX-async threads don't
    # keep the process alive after the viewer window is closed.
    import os
    os._exit(0)
