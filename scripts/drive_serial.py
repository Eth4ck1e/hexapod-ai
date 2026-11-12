"""
scripts/drive_serial.py — tethered deployment: scaffold + policy on the host,
joint targets streamed to the mini hexapod over USB serial.

The firmware (mini_hexapod/firmware/arduino/hexapod_firmware.ino, menu `P`)
accepts lines `J a0 … a17` — 18 joint angles in radians, simulator
convention and simulator leg order (RR RM RF LR LM LF × coxa/femur/tibia) —
and converts them to servo pulses with its stored trims and signs. This
script produces those frames exactly the way the training env produced
joint targets:

    feet_t      = scaffold foot intent at time t           (Controller.predict_with_feet)
    action      = policy(obs)                              (deterministic mean)
    target      = IK(feet_t + action · foot_residual_scale)
    obs_next    = hardware obs (ctrl=target, Δctrl/dt, constant IMU slots,
                  scaffold_hint=feet_t, phase(t+dt), cmd)   (envs/obs_layout.py)

The policy loop runs at the training rate (the MJCF timestep, 200 Hz);
frames are decimated to --send-rate (50 Hz, the servo PWM rate). Without a
policy (--params omitted) it streams the pure analytical scaffold, which is
the first thing to test on the real bot.

Command sources:
    --source demo      scripts/demo_phases.py showcase schedule (hands-free)
    --source hold      constant --cmd "vx vy wz" (m/s, m/s, rad/s)
    --source gamepad   8BitDo / Xbox via pygame (scripts/controller_mapping.py)

Examples (Windows venv; HEXAPOD_ROBOT=mini_hexapod picks the mini model):
    set HEXAPOD_ROBOT=mini_hexapod
    .venv\\Scripts\\python.exe scripts\\drive_serial.py --port COM5 --dry-run
    .venv\\Scripts\\python.exe scripts\\drive_serial.py --port COM5 --source hold --cmd 0.04 0 0
    .venv\\Scripts\\python.exe scripts\\drive_serial.py --port COM5 ^
        --params checkpoints/amp_to_mini_hexapod_v1/iter10/final/params.pkl --source gamepad
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from chain_train import MODEL_PATH as CHAIN_MODEL_PATH  # noqa: E402

from envs.obs_layout import compose_obs, obs_dim  # noqa: E402
from gait.controller import Controller  # noqa: E402

LEG_ORDER = ("RR", "RM", "RF", "LR", "LM", "LF")


# ----------------------------------------------------------------------------
# Hardware observation — mirrors envs/hexapod_env_jax._get_obs ("hardware").
# ----------------------------------------------------------------------------
def hardware_obs(ctrl_now: np.ndarray, ctrl_prev: np.ndarray, dt: float,
                 scaffold_hint: np.ndarray, phase: float, cmd: np.ndarray,
                 imu_quat=None, imu_gyro=None) -> np.ndarray:
    if imu_quat is None:
        imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if imu_gyro is None:
        imu_gyro = np.zeros(3, dtype=np.float32)
    return compose_obs(
        joint_pos=np.asarray(ctrl_now, dtype=np.float32),
        joint_vel=((np.asarray(ctrl_now) - np.asarray(ctrl_prev)) / dt).astype(np.float32),
        imu_quat=np.asarray(imu_quat, dtype=np.float32),
        imu_gyro=np.asarray(imu_gyro, dtype=np.float32),
        imu_accel=np.zeros(3, dtype=np.float32),
        scaffold_hint=np.asarray(scaffold_hint, dtype=np.float32).reshape(-1),
        phase_sc=np.array([math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)], dtype=np.float32),
        cmd=np.asarray(cmd, dtype=np.float32),
        body_linvel=np.zeros(3, dtype=np.float32),
        concat_fn=np.concatenate,
        variant="hardware",
    ).astype(np.float32)


# ----------------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------------
def load_policy(params_path: str, obs_size: int, act_size: int = 18):
    """Returns infer(obs: np.ndarray) -> np.ndarray action (deterministic)."""
    import jax
    import jax.numpy as jnp
    from watch_demo_jax import _build_inference_fn
    with open(params_path, "rb") as f:
        loaded = pickle.load(f)
    normalizer_params, policy_params = loaded[0], loaded[1]
    policy_obs_dim = int(normalizer_params.mean.shape[-1])
    if policy_obs_dim != obs_size:
        raise SystemExit(f"policy expects {policy_obs_dim}-dim obs but this robot emits "
                         f"{obs_size} — wrong checkpoint / HEXAPOD_ROBOT?")
    infer_jax = _build_inference_fn(obs_size, act_size, deterministic=True)
    rng = jax.random.PRNGKey(0)

    def infer(obs: np.ndarray) -> np.ndarray:
        a = infer_jax(normalizer_params, policy_params, jnp.asarray(obs)[None, :], rng)[0]
        return np.asarray(a, dtype=np.float32)

    infer(np.zeros(obs_size, dtype=np.float32))   # compile once up front
    return infer


# ----------------------------------------------------------------------------
# Command sources
# ----------------------------------------------------------------------------
class DemoSource:
    def __init__(self, ctrl: Controller, name: str):
        from demo_phases import DEMO_PHASES_SHOWCASE
        self.phases = {"showcase": DEMO_PHASES_SHOWCASE}[name]
        self.total = sum(d for _, d, _ in self.phases)
        self.ctrl = ctrl
        self._last = -1

    def __call__(self, t: float) -> np.ndarray:
        tc = t % self.total
        acc = 0.0
        for i, (label, dur, fn) in enumerate(self.phases):
            if tc < acc + dur:
                if i != self._last:
                    self._last = i
                    print(f"  t={t:6.1f}s  phase {i+1}/{len(self.phases)}: {label}")
                return np.asarray(fn(tc - acc, self.ctrl), dtype=np.float32)
            acc += dur
        return np.zeros(9, dtype=np.float32)


class HoldSource:
    def __init__(self, vx: float, vy: float, wz: float):
        self.cmd = np.zeros(9, dtype=np.float32)
        self.cmd[:3] = (vx, vy, wz)

    def __call__(self, t: float) -> np.ndarray:
        return self.cmd


class GamepadSource:
    """8BitDo / Xbox via pygame, same mapping as scripts/watch_controller.py."""

    def __init__(self, ctrl: Controller, calibration_path: str | None, rate: float = 50.0):
        import controller_mapping as cm
        import pygame
        from watch_controller import _detect_joystick
        self.cm = cm
        self.pygame = pygame
        self.joy = _detect_joystick(None, verbose=True)
        if self.joy is None:
            raise SystemExit("no joystick")
        self.joy.init()
        cal_path = calibration_path or str(_ROOT / "checkpoints" / "controller_calibration.json")
        self.cal = cm.load_calibration(cal_path) if os.path.exists(cal_path) else dict(cm.DEFAULT_CALIBRATION)
        print(f"controller calibration: {cal_path if os.path.exists(cal_path) else 'DEFAULT_CALIBRATION'}")
        self.state = cm.ControllerState()
        self.ctrl = ctrl
        self.period = 1.0 / rate
        self._next = 0.0
        self._cmd = np.zeros(9, dtype=np.float32)

    def __call__(self, t: float) -> np.ndarray:
        now = time.time()
        if now < self._next:
            return self._cmd
        self._next = now + self.period
        self.pygame.event.pump()
        lx, ly, rx, ry, lt, rt, hat, start = self.cm.read_joystick_via_calibration(self.joy, self.cal)
        self.state.step_dpad(int(hat[0]), int(hat[1]))
        self.state.step_start(start)
        self._cmd = np.asarray(self.cm.build_cmd(self.state, lx, ly, rx, ry, lt, rt,
                                                 self.ctrl.MAX_SPEED, self.ctrl.MAX_YAW_RATE),
                               dtype=np.float32)
        return self._cmd


# ----------------------------------------------------------------------------
# Serial link
# ----------------------------------------------------------------------------
class Link:
    def __init__(self, port: str | None, baud: int, dry_run: bool):
        self.dry = dry_run or port is None
        self.ser = None
        if not self.dry:
            import serial
            self.ser = serial.Serial(port, baud, timeout=0.02)
            time.sleep(0.3)
            self.ser.reset_input_buffer()
            self.ser.write(b"P\n")                 # enter stream mode from the main menu
            time.sleep(0.3)
            self.drain(verbose=True)

    def drain(self, verbose: bool = False) -> None:
        if self.ser is None:
            return
        while self.ser.in_waiting:
            line = self.ser.readline().decode(errors="replace").strip()
            if verbose and line:
                print(f"  bot: {line}")
            elif line and line not in ("ok",):
                print(f"  bot: {line}")

    def send(self, joints: np.ndarray) -> None:
        frame = "J " + " ".join(f"{a:.4f}" for a in joints) + "\n"
        if self.ser is None:
            return
        self.ser.write(frame.encode())

    def close(self) -> None:
        if self.ser is not None:
            self.ser.write(b"Q\n")
            time.sleep(0.2)
            self.ser.close()


# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=None, help="serial port (COM5, /dev/tty.usbmodem…); omit with --dry-run")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--model", default=CHAIN_MODEL_PATH, help="MJCF → robot profile (HEXAPOD_ROBOT)")
    p.add_argument("--params", default=None, help="policy params.pkl; omit to stream the pure scaffold")
    p.add_argument("--speed-level", type=float, default=None, help="profile speed ladder entry")
    p.add_argument("--source", choices=["demo", "hold", "gamepad"], default="demo")
    p.add_argument("--demo", default="showcase")
    p.add_argument("--cmd", type=float, nargs=3, default=(0.04, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    p.add_argument("--calibration", default=None, help="controller calibration json")
    p.add_argument("--send-rate", type=float, default=50.0, help="frames/s to the bot")
    p.add_argument("--duration", type=float, default=None, help="seconds; default: until Ctrl-C")
    p.add_argument("--realtime", type=float, default=1.0, help="wall-clock speed factor (1.0 = real time)")
    p.add_argument("--dry-run", action="store_true", help="no serial; print a frame per second")
    p.add_argument("--log-csv", default=None, help="write t, cmd, joint targets per sent frame")
    args = p.parse_args()

    import mujoco
    model_path = args.model if os.path.isabs(args.model) else str(_ROOT / args.model)
    ctrl = Controller(model_path, speed_level=args.speed_level)
    prof = ctrl.profile
    dt = float(mujoco.MjModel.from_xml_path(model_path).opt.timestep)   # policy rate = training rate
    obs_size = obs_dim(prof.obs_variant)
    residual_scale = float(prof.foot_residual_scale)
    print(f"robot={prof.name}  obs={prof.obs_variant}({obs_size})  imu={prof.imu_available}  "
          f"policy dt={dt*1e3:.1f} ms  send {args.send_rate:.0f} Hz  MAX_SPEED={ctrl.MAX_SPEED:.3f} m/s")
    if prof.obs_variant != "hardware":
        print("WARNING: this profile's obs variant is not 'hardware'; the policy expects state "
              "the bot cannot measure. Use HEXAPOD_ROBOT=mini_hexapod.")

    infer = load_policy(args.params, obs_size) if args.params else None
    print("policy:", args.params or "(none — pure scaffold)")

    if args.source == "demo":
        source = DemoSource(ctrl, args.demo)
    elif args.source == "hold":
        source = HoldSource(*args.cmd)
    else:
        source = GamepadSource(ctrl, args.calibration)

    link = Link(args.port, args.baud, args.dry_run)
    csv_f = open(args.log_csv, "w") if args.log_csv else None   # noqa: SIM115 — closed in finally
    if csv_f:
        csv_f.write("t," + ",".join(f"cmd{i}" for i in range(9)) + ","
                    + ",".join(f"{n}_{j}" for n in LEG_ORDER for j in ("coxa", "femur", "tibia")) + "\n")

    # Reset state exactly like the env: neutral pose, hint at t=0, phase 0.
    neutral = ctrl.gait_neutral_pose.astype(np.float32)
    t = 0.0
    cmd = source(t)
    _, feet0 = ctrl.predict_with_feet(cmd, t)
    prev_target, target = neutral.copy(), neutral.copy()
    obs = hardware_obs(target, prev_target, dt, feet0, 0.0, cmd)

    send_every = max(1, int(round(1.0 / (dt * args.send_rate))))
    k = 0
    wall0 = time.time()
    last_print = -1.0
    link.send(neutral)
    print("streaming… Ctrl-C to stop")
    try:
        while args.duration is None or t < args.duration:
            cmd = source(t)
            scaffold_joints, feet_t = ctrl.predict_with_feet(cmd, t)
            if infer is None:
                target = np.asarray(scaffold_joints, dtype=np.float32)
            else:
                # Brax PPO's NormalTanhDistribution: deterministic action = tanh(mean),
                # which is what the env saw during training.
                action = np.tanh(infer(obs))
                feet_target = feet_t + action.reshape(6, 3) * residual_scale
                target = np.asarray(ctrl.body_to_joints(feet_target), dtype=np.float32)

            if k % send_every == 0:
                link.send(target)
                link.drain()
                if csv_f:
                    csv_f.write(f"{t:.4f}," + ",".join(f"{c:.4f}" for c in cmd) + ","
                                + ",".join(f"{a:.4f}" for a in target) + "\n")
                if args.dry_run and t - last_print >= 1.0:
                    last_print = t
                    print(f"  t={t:6.2f}  cmd=({cmd[0]:+.3f},{cmd[1]:+.3f},{cmd[2]:+.3f})  "
                          f"RR=({target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f})")

            t += dt
            obs = hardware_obs(target, prev_target, dt, feet_t, (t / ctrl.gait_period) % 1.0, cmd)
            prev_target = target
            k += 1
            # Wall-clock pacing.
            if args.realtime > 0:
                lag = wall0 + t / args.realtime - time.time()
                if lag > 0:
                    time.sleep(lag)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        link.close()
        if csv_f:
            csv_f.close()


if __name__ == "__main__":
    main()
