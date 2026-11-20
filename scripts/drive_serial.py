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
    --source console   typed commands, each move runs for a few seconds then the
                       bot idles (table-safe); --script FILE replays a list.
                       See ConsoleSource.HELP for the grammar.

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


class ConsoleSource:
    """Typed / scripted commands with a time limit per move (table-safe).

    Grammar (case-insensitive, one command per line):
        forward | back | left | right | fl | fr | bl | br   [speed 0-1] [for SEC]
        spin left | spin right                              [scale 0-1] [for SEC]
        arc left | arc right                                [speed 0-1] [for SEC]
        up MM | down MM            stance height, persistent, ramped, no time limit
        wide MM | narrow MM        stance width, persistent, ramped
        speed F                    default speed fraction for following moves
        stop                       idle now        home   idle + firmware home pose
        wait SEC                   (scripts) do nothing for SEC
        demo                       run the showcase schedule once
        help | quit

    With --script the lines run in order (a move waits for the previous one
    to finish); when stdin is not a terminal the run ends after the last line,
    otherwise the console stays open for typed commands.

    A move without `for` runs for --default-duration seconds. Between moves
    the bot idles: zero velocity, current stance, and — because training never
    commanded speeds below 40 % of max — the policy residual is faded out
    (over FADE_S) so idle is the pure scaffold. The residual fades back in at
    the start of the next move.
    """
    FADE_S = 0.5
    RAMP_MM_S = 20.0                  # stance height / width ramp rate
    HEADINGS = {"forward": 0.0, "fwd": 0.0, "f": 0.0, "back": math.pi, "backward": math.pi, "b": math.pi,
                "left": math.pi / 2, "strafe-left": math.pi / 2, "right": -math.pi / 2,
                "strafe-right": -math.pi / 2, "fl": math.pi / 4, "fr": -math.pi / 4,
                "bl": 3 * math.pi / 4, "br": -3 * math.pi / 4}
    HELP = __doc__

    def __init__(self, ctrl: Controller, default_duration: float = 5.0,
                 script: str | None = None, speed_frac: float = 0.7, interactive: bool = True):
        import queue
        import threading

        from demo_phases import DEFAULT_SPEED_FRAC, TRAINED_SPEED_MAX_FRAC, TRAINED_SPEED_MIN_FRAC

        from envs.stance_envelope import envelope_for
        self.ctrl = ctrl
        self.env = envelope_for(ctrl.profile)
        self.default_duration = float(default_duration)
        self.speed_frac = speed_frac or DEFAULT_SPEED_FRAC
        self.band = (TRAINED_SPEED_MIN_FRAC, TRAINED_SPEED_MAX_FRAC)
        self.vel = np.zeros(3, dtype=np.float32)      # active move (vx, vy, wz)
        self.move_end = -1.0                          # sim time the move stops
        self.move_label = ""
        self.dh = 0.0; self.dw = 0.0                  # current (ramped) stance
        self.dh_goal = 0.0; self.dw_goal = 0.0
        self._t_prev = 0.0
        self._gain = 0.0
        self.paused = False                           # True after `home`: stop streaming frames
        self.home_request = False
        self.done = False
        self.demo: DemoSource | None = None
        self.demo_t0 = 0.0
        self._wait_until = -1.0
        self.queue: queue.Queue[str] = queue.Queue()
        self.interactive = interactive
        with open(script) if script else open(os.devnull) as fh:
            self.script = [ln.strip() for ln in fh]
        self.script = [ln for ln in self.script if ln and not ln.startswith("#")]
        if interactive:
            threading.Thread(target=self._stdin_reader, daemon=True).start()
        print(f"console: moves default to {self.default_duration:g} s, speed {self.speed_frac:.2f}×max "
              f"({self.speed_frac * ctrl.MAX_SPEED:.3f} m/s); type 'help' for the grammar")

    def _stdin_reader(self) -> None:
        for line in sys.stdin:
            self.queue.put(line.strip())

    # -- command parsing ---------------------------------------------------
    def _parse_tail(self, toks: list[str]) -> tuple[float | None, float | None]:
        """[speed] [for SEC] / [SECs] → (speed_frac, duration)."""
        speed, dur = None, None
        i = 0
        while i < len(toks):
            tk = toks[i].lower()
            if tk == "for" and i + 1 < len(toks):
                dur = float(toks[i + 1].rstrip("s")); i += 2; continue
            if tk.endswith("s") and tk[:-1].replace(".", "", 1).isdigit():
                dur = float(tk[:-1]); i += 1; continue
            try:
                v = float(tk)
            except ValueError:
                raise ValueError(f"unexpected token {toks[i]!r}")
            if v <= 1.0:
                speed = v
            else:
                dur = v
            i += 1
        return speed, dur

    def _start_move(self, vel: np.ndarray, label: str, t: float, dur: float | None) -> None:
        self.vel = np.asarray(vel, dtype=np.float32)
        dur = self.default_duration if dur is None else dur
        self.move_end = t + dur
        self.move_label = label
        print(f"  ▶ {label} for {dur:g} s")

    def handle(self, line: str, t: float) -> None:
        toks = line.split()
        if not toks:
            return
        verb = toks[0].lower()
        try:
            if verb in ("help", "?"):
                print(self.HELP)
            elif verb in ("quit", "exit", "q"):
                self.done = True
            elif verb == "stop":
                self.move_end = t; self.demo = None
                print("  ■ idle")
            elif verb == "home":
                self.move_end = t; self.demo = None; self.paused = True; self.home_request = True
                print("  ■ home (streaming paused until the next move)")
            elif verb == "wait":
                self._wait_until = t + float(toks[1])
            elif verb == "speed":
                self.speed_frac = min(max(float(toks[1]), self.band[0]), self.band[1])
                print(f"  default speed {self.speed_frac:.2f}×max")
            elif verb in ("up", "down", "wide", "narrow"):
                mm = float(toks[1]) if len(toks) > 1 else 10.0
                if verb in ("up", "down"):
                    # Sign convention: dh NEGATIVE = body raised.
                    self.dh_goal = min(max(self.dh_goal + (-mm if verb == "up" else mm) / 1000.0,
                                           self.env.dh_min), self.env.dh_max)
                    print(f"  stance height → {-self.dh_goal * 1000:+.0f} mm (range "
                          f"{-self.env.dh_max * 1000:+.0f}..{-self.env.dh_min * 1000:+.0f})")
                else:
                    lo, hi = self.env.safe_dw_range(self.dh_goal)
                    self.dw_goal = min(max(self.dw_goal + (mm if verb == "wide" else -mm) / 1000.0, lo), hi)
                    print(f"  stance width → {self.dw_goal * 1000:+.0f} mm (range {lo * 1000:+.0f}..{hi * 1000:+.0f} at this height)")
            elif verb == "demo":
                self.demo = DemoSource(self.ctrl, "showcase"); self.demo_t0 = t
                self.move_end = t + self.demo.total
                print(f"  ▶ showcase demo, {self.demo.total:.0f} s")
            elif verb in ("spin", "arc"):
                if len(toks) < 2 or toks[1].lower() not in ("left", "right", "l", "r"):
                    raise ValueError("spin/arc need left|right")
                sign = 1.0 if toks[1].lower().startswith("l") else -1.0
                frac, dur = self._parse_tail(toks[2:])
                if verb == "spin":
                    scale = frac if frac is not None else 1.0
                    self._start_move([0.0, 0.0, sign * self.ctrl.MAX_YAW_RATE * scale], f"spin {toks[1]}", t, dur)
                else:
                    from demo_phases import _arc
                    c = _arc(self.ctrl, frac if frac is not None else self.speed_frac, 0.5, sign)
                    self._start_move(c[:3], f"arc {toks[1]}", t, dur)
            elif verb in self.HEADINGS:
                frac, dur = self._parse_tail(toks[1:])
                frac = min(max(frac if frac is not None else self.speed_frac, self.band[0]), self.band[1])
                speed = self.ctrl.MAX_SPEED * frac
                h = self.HEADINGS[verb]
                self._start_move([speed * math.cos(h), speed * math.sin(h), 0.0], f"{verb} {frac:.2f}×", t, dur)
            else:
                print(f"  ? unknown command {verb!r} — 'help' lists the grammar")
        except (ValueError, IndexError) as e:
            print(f"  ? {line!r}: {e}")
        if self.paused and (self.move_end > t or self.demo is not None):
            self.paused = False

    # -- per-tick ------------------------------------------------------------
    def _poll(self, t: float) -> None:
        while not self.queue.empty():
            self.handle(self.queue.get(), t)
        idle = self.move_end <= t and self.demo is None and t >= self._wait_until
        if idle and self.script:
            self.handle(self.script.pop(0), t)
        elif idle and not self.script and not self.interactive and self._gain <= 0.0:
            self.done = True                      # scripted, non-interactive run: finished

    def __call__(self, t: float) -> np.ndarray:
        if self.move_label and t >= self.move_end:
            print(f"  ■ {self.move_label} done, idle"); self.move_label = ""
        self._poll(t)
        dt = max(t - self._t_prev, 0.0); self._t_prev = t
        step = self.RAMP_MM_S / 1000.0 * dt
        self.dh += min(max(self.dh_goal - self.dh, -step), step)
        self.dw += min(max(self.dw_goal - self.dw, -step), step)
        if self.demo is not None:
            if t >= self.move_end:
                self.demo = None; print("  ■ demo done, idle")
            else:
                cmd = self.demo(t - self.demo_t0).copy()
                self._gain = min(1.0, self._gain + dt / self.FADE_S)
                return cmd
        moving = t < self.move_end
        self._gain = min(1.0, self._gain + dt / self.FADE_S) if moving else max(0.0, self._gain - dt / self.FADE_S)
        cmd = np.zeros(9, dtype=np.float32)
        if moving:
            cmd[:3] = self.vel
        cmd[5] = self.dh; cmd[6] = self.dw
        return cmd

    def residual_gain(self, t: float) -> float:
        return self._gain

    def take_home_request(self) -> bool:
        r = self.home_request; self.home_request = False
        return r


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
            self.ser.write(b"m\n")                 # back to the main menu whatever mode the bot is in
            time.sleep(0.3)
            self.ser.reset_input_buffer()
            self.ser.write(b"P\n")                 # enter stream mode
            # Wait for the firmware's ready line before the first frame; sending
            # earlier can truncate frame 1 ("err n=16") while the banner prints.
            t0, seen = time.time(), ""
            while time.time() - t0 < 3.0 and "frames" not in seen:
                seen += self.ser.read(self.ser.in_waiting or 1).decode(errors="replace")
            for line in seen.splitlines():
                if line.strip():
                    print(f"  bot: {line.strip()}")
            if "frames" not in seen:
                print("WARNING: bot did not confirm stream mode (firmware menu 'P')")
            time.sleep(0.1)

    def drain(self, verbose: bool = False) -> None:
        if self.ser is None:
            return
        while self.ser.in_waiting:
            line = self.ser.readline().decode(errors="replace").strip()
            if verbose and line:
                print(f"  bot: {line}")
            elif line and line not in ("ok",):
                print(f"  bot: {line}")

    def home(self) -> None:
        if self.ser is not None:
            self.ser.write(b"H\n")

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
    p.add_argument("--source", choices=["demo", "hold", "gamepad", "console"], default="demo")
    p.add_argument("--script", default=None, help="console: file of commands, one per line (# comments)")
    p.add_argument("--default-duration", type=float, default=5.0, help="console: seconds per move without 'for'")
    p.add_argument("--speed-frac", type=float, default=None, help="console: default speed fraction of MAX_SPEED")
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
    elif args.source == "console":
        source = ConsoleSource(ctrl, args.default_duration, args.script, args.speed_frac,
                               interactive=sys.stdin.isatty() or args.script is None)
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
    # The policy's obs carry the LAST SENT target (what the servos hold), not
    # the per-tick target — same zero-order hold the env applies
    # (RobotProfile.ctrl_hold_steps).
    sent, prev_sent = neutral.copy(), neutral.copy()
    obs = hardware_obs(sent, prev_sent, dt, feet0, 0.0, cmd)

    send_every = max(1, int(round(1.0 / (dt * args.send_rate))))
    if send_every != ctrl.profile.ctrl_hold_steps:
        print(f"WARNING: --send-rate gives a {send_every}-step hold but the profile was "
              f"trained with ctrl_hold_steps={ctrl.profile.ctrl_hold_steps}")
    k = 0
    wall0 = time.time()
    last_print = -1.0
    link.send(neutral)
    print("streaming… Ctrl-C to stop")
    try:
        while args.duration is None or t < args.duration:
            if getattr(source, "done", False):
                break
            cmd = source(t)
            gain = source.residual_gain(t) if hasattr(source, "residual_gain") else 1.0
            scaffold_joints, feet_t = ctrl.predict_with_feet(cmd, t)
            if infer is None or gain <= 0.0:
                target = np.asarray(scaffold_joints, dtype=np.float32)
            else:
                # Brax PPO's NormalTanhDistribution: deterministic action = tanh(mean),
                # which is what the env saw during training.
                action = np.tanh(infer(obs))
                feet_target = feet_t + action.reshape(6, 3) * (residual_scale * gain)
                target = np.asarray(ctrl.body_to_joints(feet_target), dtype=np.float32)

            if hasattr(source, "take_home_request") and source.take_home_request():
                link.home()
            paused = getattr(source, "paused", False)
            if k % send_every == 0:
                sent = target
                if not paused:
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
            obs = hardware_obs(sent, prev_sent, dt, feet_t, (t / ctrl.gait_period) % 1.0, cmd)
            prev_sent = sent
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
