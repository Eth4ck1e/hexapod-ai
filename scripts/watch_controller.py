"""
scripts/watch_controller.py — drive a trained AMP policy in MuJoCo via a
Bluetooth/USB game controller (8BitDo Ultimate 2 in Xbox mode).

Usage (Windows venv):
    $env:PYTHONPATH = "."
    .venv\\Scripts\\python.exe scripts\\watch_controller.py `
        --params checkpoints\\amp_to_v7\\iter2\\final\\params.pkl `
        --action-space foot --render-model models\\phantomx.xml

Required hardware:
    A standard XInput controller (8BitDo Ultimate 2 in Xbox mode works).
    See scripts/controller_mapping.py for the button/axis convention.

Behavior:
    50 Hz control loop reads the joystick state, builds the 9-vec cmd
    from L stick + R stick + triggers + D-pad, passes it to the policy,
    and renders the bot in the viewer. D-pad up/down/L/R edge-trigger
    discrete level steps for height/width. Start toggles servo kill.

Cross-references:
    - controller_mapping.py: pure mapping logic (port to ESP32 verbatim)
    - watch_demo_jax.py:     phase-driven demo (this is its joystick sibling)
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pygame

# Make sibling scripts importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain_train import ACTION_SPACE as CHAIN_ACTION_SPACE
from chain_train import MODEL_PATH as CHAIN_MODEL_PATH
from chain_train import RENDER_MODEL_PATH as CHAIN_RENDER_MODEL_PATH
from chain_train import enable_jax_cache

enable_jax_cache()

from brax.training.acme import running_statistics
from brax.training.agents.ppo.networks import make_ppo_networks
from controller_mapping import (
    DEFAULT_CALIBRATION,
    HEIGHT_LEVELS_M,
    WIDTH_LEVELS_M,
    ControllerState,
    build_cmd,
    load_calibration,
    read_joystick_via_calibration,
)

from envs.hexapod_env import HexapodEnv

DEFAULT_CALIBRATION_PATH = (Path(__file__).resolve().parent.parent
                            / "checkpoints" / "controller_calibration.json")


def _build_inference_fn(obs_size: int, act_size: int,
                        deterministic: bool = True,
                        log_std_override: float | None = None):
    """Mirror watch_demo_jax's inference setup. Returns a JIT'd callable
    f(normalizer_params, policy_params, obs_batch, rng) → action_batch."""
    networks = make_ppo_networks(
        observation_size=obs_size,
        action_size=act_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    if deterministic:
        def infer(norm_p, pol_p, obs, rng):
            logits = networks.policy_network.apply(norm_p, pol_p, obs)
            return networks.parametric_action_distribution.mode(logits)
    else:
        def infer(norm_p, pol_p, obs, rng):
            logits = networks.policy_network.apply(norm_p, pol_p, obs)
            if log_std_override is not None:
                # Override the log_std slice (latter half of the logits).
                half = logits.shape[-1] // 2
                logits = logits.at[..., half:].set(log_std_override)
            return networks.parametric_action_distribution.sample(logits, rng)
    return jax.jit(infer)


def _detect_joystick(prefer_substring: str | None,
                     verbose: bool = True) -> pygame.joystick.Joystick:
    """Find the joystick whose name contains `prefer_substring`.
    Exits with a clear error if no match is found — caller has to either
    connect the right controller or override with --joystick-index.
    """
    pygame.init()
    pygame.joystick.init()
    n = pygame.joystick.get_count()
    if n == 0:
        print("ERROR: no joysticks detected. Plug in / pair the controller "
              "and re-launch.")
        sys.exit(1)

    if verbose:
        print(f"Detected {n} joystick(s):")
        for i in range(n):
            j = pygame.joystick.Joystick(i)
            print(f"  [{i}] {j.get_name()}  axes={j.get_numaxes()} "
                  f"buttons={j.get_numbuttons()} hats={j.get_numhats()}")

    if prefer_substring:
        wanted = prefer_substring.lower()
        for i in range(n):
            j = pygame.joystick.Joystick(i)
            if wanted in j.get_name().lower():
                if verbose:
                    print(f"  -> selected index {i} (name contains "
                          f"'{prefer_substring}')")
                return j
        # No match — return None to signal caller to fall back.
        print(f"\nNo joystick name contained '{prefer_substring}'.")
        return None

    # No substring requested — pick first device with full XInput shape.
    for i in range(n):
        j = pygame.joystick.Joystick(i)
        if j.get_numaxes() >= 6 and j.get_numbuttons() >= 8 and j.get_numhats() >= 1:
            if verbose:
                print(f"  -> selected index {i} (XInput axis/button shape)")
            return j

    print("\nNo XInput-shaped gamepad found.")
    return None


def _fallback_to_demo_viewer(args) -> None:
    """When no controller is connected, hand off to watch_demo_jax in
    interactive (keyboard) mode with the same params + render model."""
    import subprocess
    print("\n=== FALLBACK: launching keyboard-driven demo viewer ===")
    print("(controller not detected; use the keyboard in the mujoco viewer "
          "window — press 1-9 for tests, h for help, r to reset)\n")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "watch_demo_jax.py"),
        "--params", args.params,
        "--action-space", args.action_space,
        "--interactive",
    ]
    if args.render_model:
        cmd.extend(["--render-model", args.render_model])
    if args.model:
        cmd.extend(["--model", args.model])
    sys.exit(subprocess.call(cmd))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True,
                   help="path to a trained policy params.pkl. "
                        "Recommended: checkpoints/amp_to_v7/iter2/final/params.pkl")
    p.add_argument("--model", default=CHAIN_MODEL_PATH,
                   help="MJCF for inference (must match the training MJCF).")
    p.add_argument("--render-model", default=CHAIN_RENDER_MODEL_PATH,
                   help=f"MJCF used ONLY for the viewer window. Inference still "
                        f"uses --model. Default (chain_train.RENDER_MODEL_PATH): "
                        f"{CHAIN_RENDER_MODEL_PATH!r}. Pass the same path as "
                        f"--model to skip the dual-render overhead.")
    p.add_argument("--action-space", type=str, default=CHAIN_ACTION_SPACE,
                   choices=["joint", "foot"],
                   help="must match the trained policy.")
    p.add_argument("--stage", type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument("--gait-scale", type=float, default=0.0,
                   help="scaffold cushion. 0 = pure policy (default).")
    p.add_argument("--joystick-name", type=str, default="xbox",
                   help="case-insensitive substring to match the desired "
                        "controller's name (default 'xbox'). The script will "
                        "EXIT if no match is found — connect the controller "
                        "or override with --joystick-index N.")
    p.add_argument("--joystick-index", type=int, default=None,
                   help="override auto-detect; pick this pygame joystick index.")
    p.add_argument("--rate", type=float, default=50.0,
                   help="JOYSTICK read rate, Hz (default 50). The env still "
                        "steps at the model's full physics rate (200 Hz) "
                        "so the bot moves at real-time. Each joystick tick "
                        "drives ~4 env.steps with the same cmd held.")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed multiplier (1.0 = real time, 0.5 = "
                        "half-speed, 2.0 = double-speed). Matches watch_demo_jax.")
    p.add_argument("--speed-scale", type=float, default=None,
                   help="cap stick output at this fraction of MAX_SPEED. "
                        "Default 0.85 (= trained max). Set 1.0 for full "
                        "MAX_SPEED, 2.0 to push 2x MAX_SPEED (heavy OOD — "
                        "policy may track poorly).")
    p.add_argument("--yaw-scale", type=float, default=None,
                   help="same as --speed-scale but for triggers / yaw rate.")
    p.add_argument("--smoothing", type=float, default=0.10,
                   help="exponential moving average time constant (seconds) "
                        "applied to joystick axes before they reach the "
                        "policy. Filters hand tremor and stick drift. "
                        "Default 0.10s = ~100ms response. Set 0 to disable.")
    p.add_argument("--debug", action="store_true",
                   help="print joystick state + computed cmd every 5 ticks "
                        "(10 Hz). Use to diagnose mapping mismatches.")
    p.add_argument("--calibration", type=str, default=str(DEFAULT_CALIBRATION_PATH),
                   help=f"path to a calibration JSON produced by "
                        f"calibrate_controller.py. Falls back to defaults if "
                        f"the file is missing. Default: {DEFAULT_CALIBRATION_PATH}")
    args = p.parse_args()

    # --- Joystick init ---
    if args.joystick_index is not None:
        pygame.init()
        pygame.joystick.init()
        joystick = pygame.joystick.Joystick(args.joystick_index)
    else:
        joystick = _detect_joystick(args.joystick_name)
    if joystick is None:
        # Hand off to the keyboard-driven demo viewer with the same params.
        _fallback_to_demo_viewer(args)
        return  # _fallback_to_demo_viewer calls sys.exit; this is for clarity
    print(f"Using joystick: {joystick.get_name()}")

    # --- Calibration ---
    cal_path = Path(args.calibration)
    if cal_path.exists():
        calibration = load_calibration(cal_path)
        print(f"Loaded calibration: {cal_path}")
        print(f"  Calibrated for: {calibration.get('joystick_name', '?')}")
    else:
        calibration = DEFAULT_CALIBRATION
        print(f"No calibration at {cal_path}; using DEFAULTS.")
        print("  Run scripts/calibrate_controller.py to generate one.")

    # --- Override speed/yaw scale if user passed flags ---
    import controller_mapping as _cm
    if args.speed_scale is not None:
        print(f"  speed_scale OVERRIDE: {_cm.SPEED_SCALE} -> {args.speed_scale}"
              f"  ({'OOD' if args.speed_scale > 0.85 else 'in-distribution'})")
        _cm.SPEED_SCALE = args.speed_scale
    if args.yaw_scale is not None:
        print(f"  yaw_scale OVERRIDE:   {_cm.YAW_SCALE} -> {args.yaw_scale}")
        _cm.YAW_SCALE = args.yaw_scale

    # --- Load policy params ---
    print(f"Loading params: {args.params}")
    with open(args.params, "rb") as f:
        loaded = pickle.load(f)
    if isinstance(loaded, tuple) and len(loaded) == 3:
        normalizer_params, policy_params, _ = loaded
    elif isinstance(loaded, tuple) and len(loaded) == 2:
        normalizer_params, policy_params = loaded
    else:
        print(f"Unrecognized params format: {type(loaded)}")
        sys.exit(1)

    # --- Build env + (optional) dual-render viewer ---
    import mujoco
    import mujoco.viewer as mj_viewer

    render_model_path = args.render_model
    dual_render = (render_model_path is not None and
                   os.path.realpath(render_model_path) != os.path.realpath(args.model))

    if dual_render:
        env = HexapodEnv(stage=args.stage, render_mode=None, model_path=args.model)
    else:
        env = HexapodEnv(stage=args.stage, render_mode="human", model_path=args.model)
    env.gait_scale = args.gait_scale

    # Patch obs to JAX-style (matches watch_demo_jax.py rationale).
    import math
    import types
    def _get_obs(self):
        qpos = self._data.qpos
        qvel = self._data.qvel
        imu_quat  = qpos[3:7]
        imu_gyro  = qvel[3:6]
        imu_accel = np.zeros(3, dtype=np.float32)
        scaffold_hint = self._latest_feet_body.flatten()
        phase = self._ctrl.get_phase(self._sim_time)
        body_linvel = np.asarray(self._body_frame_linvel(), dtype=np.float32)
        if self.stage == 4:
            body_linvel = np.zeros(3, dtype=np.float32)
        return np.concatenate([
            qpos[7:], qvel[6:], imu_quat, imu_gyro, imu_accel,
            scaffold_hint,
            [math.sin(2*math.pi*phase), math.cos(2*math.pi*phase)],
            self._cmd, body_linvel,
        ]).astype(np.float32)
    env._get_obs = types.MethodType(_get_obs, env)

    if dual_render:
        render_mjmodel = mujoco.MjModel.from_xml_path(render_model_path)
        if render_mjmodel.nq != env._model.nq:
            print("ERROR: joint count mismatch between inference model "
                  "and render model.")
            sys.exit(1)
        render_mjdata = mujoco.MjData(render_mjmodel)
        render_mjdata.qpos[:] = env._data.qpos
        mujoco.mj_forward(render_mjmodel, render_mjdata)
        render_viewer = mj_viewer.launch_passive(render_mjmodel, render_mjdata)
    else:
        render_viewer = None

    # --- Build inference fn ---
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    infer = _build_inference_fn(obs_size, act_size, deterministic=True)
    infer_rng = jax.random.PRNGKey(0)
    foot_residual_scale = float(env._ctrl.foot_residual_scale_max
                                if hasattr(env._ctrl, "foot_residual_scale_max")
                                else 0.020)

    print(f"\nModel:        {args.model}")
    if dual_render:
        print(f"Render model: {render_model_path}")
    print(f"Action mode:  {args.action_space}")
    print(f"MAX_SPEED:    {env._ctrl.MAX_SPEED:.4f} m/s")
    print(f"MAX_YAW_RATE: {env._ctrl.MAX_YAW_RATE:.4f} rad/s")
    print(f"Height levels (mm): {[int(h*1000) for h in HEIGHT_LEVELS_M]}")
    print(f"Width  levels (mm): {[round(w*1000, 1) for w in WIDTH_LEVELS_M]}")
    print("\n=== CONTROLLER BINDINGS ===")
    print("  L stick X/Y    -> vx, vy (omnidirectional)")
    print("  L trigger      -> turn LEFT  (proportional)")
    print("  R trigger      -> turn RIGHT (proportional)")
    print("  D-pad up/down  -> step height level (raise / lower)")
    print("  D-pad L/R      -> step width level  (narrow / wide)")
    print("  Start          -> servo kill toggle (cmd zeroed when killed)")
    print("  B button       -> RESET bot to neutral pose (recover from stuck)")
    print("==========================\n")

    obs, _ = env.reset()
    # Settle: let the bot stand briefly with cmd=0 before the user takes over.
    env._cmd = np.zeros(9, dtype=np.float32)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(50):
        obs, *_ = env.step(zero_action)

    state = ControllerState()
    last_kill_state = state.killed
    tick_count = 0

    # Rolling-average buffers for the debug readout. Smooth over ~0.5 sec
    # at the env physics rate (200 Hz) so fluctuating values are readable.
    from collections import deque
    AVG_N = 100
    buf_lx, buf_ly = deque(maxlen=AVG_N), deque(maxlen=AVG_N)
    buf_lt, buf_rt = deque(maxlen=AVG_N), deque(maxlen=AVG_N)
    buf_cmd_vx, buf_cmd_vy = deque(maxlen=AVG_N), deque(maxlen=AVG_N)
    buf_cmd_wz             = deque(maxlen=AVG_N)
    buf_act_vx, buf_act_vy = deque(maxlen=AVG_N), deque(maxlen=AVG_N)

    # Two clocks:
    #   joystick_period: how often to re-read the controller (default 50 Hz)
    #   env_dt:          model's physics timestep (5ms = 200 Hz)
    # Each loop = one env.step (advances sim by env_dt). Wall pacing keeps
    # sim:wall = 1:speed, so the bot moves at real-time when speed=1.0.
    joystick_period = 1.0 / args.rate
    env_dt = float(env._dt)

    import math as _math
    # EMA smoothing constant. alpha = 1 - exp(-dt/tau).
    # alpha closer to 0 = more smoothing; closer to 1 = less.
    if args.smoothing > 0:
        smooth_alpha = 1.0 - _math.exp(-joystick_period / args.smoothing)
    else:
        smooth_alpha = 1.0   # no smoothing — pass through
    sm_lx = sm_ly = sm_rx = sm_ry = 0.0
    sm_lt = sm_rt = -1.0     # triggers rest at -1 on XInput
    smooth_initialized = False

    print("Ready. Drive the bot with the controller. Ctrl+C to exit.")
    print(f"Joystick read rate: {args.rate} Hz   "
          f"Env physics rate: {1.0/env_dt:.0f} Hz   "
          f"Speed multiplier: {args.speed}")
    print(f"Smoothing: {'OFF' if args.smoothing <= 0 else f'{args.smoothing*1000:.0f}ms time constant (alpha={smooth_alpha:.3f})'}\n")
    if args.debug:
        print("DEBUG mode: live polar readout (self-updating line, no scroll).")
        print("Format: L: (lx,ly) m=mag a=angle  triggers  hat  cmd[...]  actual[...]  track%")
        print("Angle convention: 0°=forward, +90°=left, -90°=right, ±180°=back\n")
    wall_t0 = time.time()
    sim_t   = 0.0
    last_joystick_read_t = -1.0   # forces immediate read on first iter

    # B button index for the reset action — pull from calibration if available.
    b_button_idx = calibration.get("buttons", {}).get("b", 1)
    last_b_pressed = False

    # Initial joystick read (so cmd is defined before first env.step).
    pygame.event.pump()
    (raw_lx, raw_ly, raw_rx, raw_ry, raw_lt, raw_rt,
     hat, start_pressed) = read_joystick_via_calibration(joystick, calibration)
    # Initialize smoothing state with the first read so we don't EMA up
    # from zero on launch.
    sm_lx, sm_ly = raw_lx, raw_ly
    sm_rx, sm_ry = raw_rx, raw_ry
    sm_lt, sm_rt = raw_lt, raw_rt
    lx, ly, rx, ry, lt, rt = sm_lx, sm_ly, sm_rx, sm_ry, sm_lt, sm_rt

    try:
        while True:
            # --- Re-read joystick at the configured rate (default 50 Hz) ---
            now = time.time()
            if now - last_joystick_read_t >= joystick_period:
                pygame.event.pump()
                (raw_lx, raw_ly, raw_rx, raw_ry, raw_lt, raw_rt,
                 hat, start_pressed) = read_joystick_via_calibration(joystick, calibration)
                # EMA low-pass to filter hand tremor / stick drift.
                # Sticks AND triggers are smoothed; hat / buttons are discrete.
                a = smooth_alpha
                sm_lx = a * raw_lx + (1 - a) * sm_lx
                sm_ly = a * raw_ly + (1 - a) * sm_ly
                sm_rx = a * raw_rx + (1 - a) * sm_rx
                sm_ry = a * raw_ry + (1 - a) * sm_ry
                sm_lt = a * raw_lt + (1 - a) * sm_lt
                sm_rt = a * raw_rt + (1 - a) * sm_rt
                lx, ly = sm_lx, sm_ly
                rx, ry = sm_rx, sm_ry
                lt, rt = sm_lt, sm_rt
                last_joystick_read_t = now

            # Discrete state updates (edge-detected).
            state.step_dpad(int(hat[0]), int(hat[1]))
            state.step_start(start_pressed)

            # B button: edge-trigger env reset (recover from stuck poses).
            b_pressed = bool(joystick.get_button(b_button_idx)) if b_button_idx < joystick.get_numbuttons() else False
            if b_pressed and not last_b_pressed:
                env.reset()
                print("\n  >>> B: RESET bot to neutral pose")
            last_b_pressed = b_pressed
            if state.killed != last_kill_state:
                if state.killed:
                    print("  >>> SERVO KILL  ON  (cmd zeroed)")
                else:
                    print("  >>> SERVO KILL  OFF (cmd live)")
                last_kill_state = state.killed

            cmd = build_cmd(
                state, lx, ly, rx, ry, lt, rt,
                max_speed=float(env._ctrl.MAX_SPEED),
                max_yaw_rate=float(env._ctrl.MAX_YAW_RATE),
            )

            # Push samples into rolling buffers every tick (not just on print
            # ticks) so the rolling average is over the full window.
            buf_lx.append(lx); buf_ly.append(ly)
            buf_lt.append(lt); buf_rt.append(rt)
            buf_cmd_vx.append(float(cmd[0])); buf_cmd_vy.append(float(cmd[1]))
            buf_cmd_wz.append(float(cmd[2]))
            bvx_now, bvy_now, _ = env._body_frame_linvel()
            buf_act_vx.append(float(bvx_now)); buf_act_vy.append(float(bvy_now))

            if args.debug and tick_count % 10 == 0:
                # Compute rolling means.
                def avg(buf):
                    return sum(buf) / len(buf) if buf else 0.0
                a_lx, a_ly = avg(buf_lx), avg(buf_ly)
                a_lt, a_rt = avg(buf_lt), avg(buf_rt)
                a_cvx, a_cvy = avg(buf_cmd_vx), avg(buf_cmd_vy)
                a_cwz = avg(buf_cmd_wz)
                a_avx, a_avy = avg(buf_act_vx), avg(buf_act_vy)

                a_lmag = (a_lx*a_lx + a_ly*a_ly) ** 0.5
                a_lang = _math.degrees(_math.atan2(-a_lx, -a_ly)) if a_lmag > 1e-3 else 0.0
                a_cmag = (a_cvx*a_cvx + a_cvy*a_cvy) ** 0.5
                a_amag = (a_avx*a_avx + a_avy*a_avy) ** 0.5
                track_pct = (a_amag / a_cmag * 100.0) if a_cmag > 1e-4 else 0.0
                kill = " KILLED" if state.killed else ""
                line = (
                    f"L:({a_lx:+.2f},{a_ly:+.2f}) m={a_lmag:.2f} a={a_lang:+6.1f}deg  "
                    f"LT={a_lt:+.2f} RT={a_rt:+.2f}  hat={tuple(int(h) for h in hat)}  "
                    f"cmd[vx={a_cvx:+.4f} vy={a_cvy:+.4f} mag={a_cmag:.4f}]  "
                    f"actual[vx={a_avx:+.4f} vy={a_avy:+.4f} mag={a_amag:.4f}] "
                    f"track={track_pct:5.1f}%{kill}"
                )
                print(line.ljust(200), end="\r", flush=True)
            tick_count += 1

            env._cmd = cmd
            env._latest_feet_body = env._ctrl.compute_foot_targets(
                env._cmd, env._sim_time).astype(np.float32)
            obs = env._get_obs()

            # Policy inference.
            infer_rng, sub = jax.random.split(infer_rng)
            action_jax = infer(normalizer_params, policy_params,
                               jnp.asarray(obs)[None, :], sub)[0]
            action = np.asarray(action_jax)

            # Foot-space conversion if needed.
            if args.action_space == "foot":
                foot_residual = action.reshape(6, 3) * foot_residual_scale
                feet_body_target = env._latest_feet_body + foot_residual
                joint_target = env._ctrl.body_to_joints(feet_body_target)
                action = (joint_target - env._ctrl.gait_neutral_pose) / env.RESIDUAL_SCALE_MAX
                action = np.clip(action, -1.0, 1.0).astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            sim_t += env_dt

            if render_viewer is not None and render_viewer.is_running():
                render_mjdata.qpos[:] = env._data.qpos
                render_mjdata.qvel[:] = env._data.qvel
                mujoco.mj_forward(render_mjmodel, render_mjdata)
                render_viewer.sync()

            # Pace to keep sim:wall = 1:speed (matches watch_demo_jax).
            if args.speed > 0:
                target_wall = wall_t0 + sim_t / args.speed
                lag = target_wall - time.time()
                if lag > 0:
                    time.sleep(lag)

    except KeyboardInterrupt:
        print("\n[watch_controller] interrupted, exiting.")
    finally:
        if render_viewer is not None and render_viewer.is_running():
            render_viewer.close()
        pygame.quit()


if __name__ == "__main__":
    main()
