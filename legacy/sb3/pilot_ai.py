"""
pilot_ai.py — keyboard teleop running through a trained PPO checkpoint.

Like pilot.py (which drives the analytical scaffold directly), but the joints
come from the policy + scaffold mix instead. Use this to drive a trained bot
in real time and feel out how it handles continuous, transitioning commands.

Usage (macOS — mjpython required for viewer):
    mjpython pilot_ai.py                      # latest checkpoint, gait_scale=0.0
    mjpython pilot_ai.py --gait-scale 0.5     # half scaffold contribution
    mjpython pilot_ai.py --gait-scale 1.0     # full scaffold (== pilot.py manual mode)
    mjpython pilot_ai.py --ckpt path/to/ckpt
    mjpython pilot_ai.py --run hexapod_mac_translation_25M
    mjpython pilot_ai.py --stochastic         # sample actions instead of deterministic

Keyboard layout (same as pilot.py — known to conflict with some mujoco viewer
hotkeys; gamepad support coming later for cleaner input):
    W/S    vx forward/back        I/K    pitch up/down
    A/D    vy left/right strafe   J/L    roll  right/left
    Q/E    yaw  CCW / CW          R/F    height up/down
                                  T/G    width  wider/narrower
    Z/X    shift X fore/aft       C/V    shift Y left/right
    SPACE  zero all      P  print cmd
    1  preset: walk forward       2  preset: spin CCW       3  zero
"""

import argparse
import glob
import math
import os
import re
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from envs.hexapod_env import HexapodEnv

from gait import NEUTRAL_POSE


# Per-key step sizes. Tuned so that holding a key (OS auto-repeat) gives a
# smooth ramp rather than a snappy jump.
STEP_VX     = 0.005
STEP_VY     = 0.005
STEP_WZ     = 0.05
STEP_PITCH  = math.radians(2.0)
STEP_ROLL   = math.radians(2.0)
STEP_HEIGHT = 0.002
STEP_WIDTH  = 0.002
STEP_SHIFT  = 0.005

# Run mode (hold Shift) — multiplies the active translation+yaw command while
# held. Mujoco's key_callback only fires on key DOWN, but holding Shift fires
# OS auto-repeat events ~30/sec; each one tops up a boost timer. When the
# user releases, the timer expires within BOOST_HOLD_TIMEOUT_S and the boost
# ends. Functionally indistinguishable from "hold to run" without needing
# key-up events.
SPEED_BOOST_MULT     = 2.0   # vx, vy, wz multiplied by this while Shift held
BOOST_HOLD_TIMEOUT_S = 0.30  # boost stays active this long after last Shift event

# GLFW key codes (mujoco viewer's key_callback receives these directly).
GLFW_KEY_LEFT_SHIFT  = 340
GLFW_KEY_RIGHT_SHIFT = 344


def make_clamps(ctrl):
    return {
        0: (-ctrl.MAX_SPEED,    ctrl.MAX_SPEED),
        1: (-ctrl.MAX_SPEED,    ctrl.MAX_SPEED),
        2: (-ctrl.MAX_YAW_RATE, ctrl.MAX_YAW_RATE),
        3: (-math.radians(20),  math.radians(20)),
        4: (-math.radians(20),  math.radians(20)),
        5: (-0.025,             0.025),
        6: (-0.020,             0.020),
        7: (-0.040,             0.040),
        8: (-0.040,             0.040),
    }


def fmt_cmd(cmd):
    return (f"vx={cmd[0]:+.3f} vy={cmd[1]:+.3f} wz={cmd[2]:+.3f}  "
            f"pitch={math.degrees(cmd[3]):+5.1f}° roll={math.degrees(cmd[4]):+5.1f}°  "
            f"h={cmd[5]*1000:+5.1f}mm w={cmd[6]*1000:+5.1f}mm  "
            f"sx={cmd[7]*1000:+5.1f}mm sy={cmd[8]*1000:+5.1f}mm")


def make_key_callback(state, ctrl):
    clamps = make_clamps(ctrl)

    def adjust(idx, delta):
        lo, hi = clamps[idx]
        state["cmd"][idx] = max(lo, min(hi, state["cmd"][idx] + delta))

    def topup_boost():
        # Each Shift press (or auto-repeat tick) tops up the boost timer.
        state["boost_until"] = time.time() + BOOST_HOLD_TIMEOUT_S

    KEY_MAP = {
        ord('W'): lambda: adjust(0, +STEP_VX),
        ord('S'): lambda: adjust(0, -STEP_VX),
        ord('A'): lambda: adjust(1, +STEP_VY),
        ord('D'): lambda: adjust(1, -STEP_VY),
        ord('Q'): lambda: adjust(2, +STEP_WZ),
        ord('E'): lambda: adjust(2, -STEP_WZ),
        ord('I'): lambda: adjust(3, +STEP_PITCH),
        ord('K'): lambda: adjust(3, -STEP_PITCH),
        ord('J'): lambda: adjust(4, +STEP_ROLL),
        ord('L'): lambda: adjust(4, -STEP_ROLL),
        ord('R'): lambda: adjust(5, -STEP_HEIGHT),
        ord('F'): lambda: adjust(5, +STEP_HEIGHT),
        ord('T'): lambda: adjust(6, +STEP_WIDTH),
        ord('G'): lambda: adjust(6, -STEP_WIDTH),
        ord('Z'): lambda: adjust(7, +STEP_SHIFT),
        ord('X'): lambda: adjust(7, -STEP_SHIFT),
        ord('C'): lambda: adjust(8, +STEP_SHIFT),
        ord('V'): lambda: adjust(8, -STEP_SHIFT),
        ord(' '): lambda: state["cmd"].fill(0.0),
        ord('P'): lambda: print(f"  CMD: {fmt_cmd(state['cmd'])}"),
        ord('1'): lambda: _preset_forward(state, ctrl),
        ord('2'): lambda: _preset_spin(state, ctrl),
        ord('3'): lambda: state["cmd"].fill(0.0),
        # Hold Shift → "run mode": vx/vy/wz multiplied by SPEED_BOOST_MULT.
        GLFW_KEY_LEFT_SHIFT:  topup_boost,
        GLFW_KEY_RIGHT_SHIFT: topup_boost,
    }

    def callback(keycode):
        fn = KEY_MAP.get(keycode)
        if fn is not None:
            fn()
            state["dirty"] = True

    return callback


def _preset_forward(state, ctrl):
    state["cmd"].fill(0.0)
    state["cmd"][0] = ctrl.MAX_SPEED


def _preset_spin(state, ctrl):
    state["cmd"].fill(0.0)
    state["cmd"][2] = ctrl.MAX_YAW_RATE


# ============================================================================
# Checkpoint discovery (matches watch.py / watch_demo.py)
# ============================================================================
def latest_run_dir(root="checkpoints"):
    if not os.path.isdir(root):
        return None
    runs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    if not runs:
        return None
    runs.sort(key=lambda d: max((os.path.getmtime(os.path.join(d, f))
                                 for f in os.listdir(d)), default=0))
    return runs[-1]


def latest_checkpoint(run_dir):
    final = os.path.join(run_dir, "final.zip")
    if os.path.exists(final) and time.time() - os.path.getmtime(final) > 3.0:
        return final[:-4]
    step_re = re.compile(r"_(\d+)_steps\.zip$")
    candidates = []
    for f in os.listdir(run_dir):
        m = step_re.search(f)
        if not m:
            continue
        path = os.path.join(run_dir, f)
        if time.time() - os.path.getmtime(path) < 3.0:
            continue
        candidates.append((int(m.group(1)), path[:-4]))
    return None if not candidates else sorted(candidates)[-1][1]


# ============================================================================
# Main
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Drive a trained PPO checkpoint via keyboard.")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--run",  default=None)
    p.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--gait-scale", type=float, default=0.0,
                   help="scaffold weight (0.0 = pure policy, 1.0 = full scaffold). Default 0.0.")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions stochastically (default deterministic)")
    args = p.parse_args()

    if args.ckpt:
        ckpt = args.ckpt
    else:
        run_dir = (os.path.join("checkpoints", args.run) if args.run
                   else latest_run_dir())
        if run_dir is None:
            print("No run directory found under checkpoints/.")
            sys.exit(1)
        ckpt = latest_checkpoint(run_dir)
        if ckpt is None:
            print(f"No complete checkpoint in {run_dir} yet.")
            sys.exit(1)

    # No render_mode — we manage the viewer manually so we can hook key_callback.
    env = HexapodEnv(stage=args.stage)
    env.gait_scale = args.gait_scale
    model = PPO.load(ckpt, env=env)

    # Settle the bot before opening the viewer. The Mac env's _sample_cmd
    # never produces cmd=0, so the policy was never trained on stand-still
    # observations — feeding cmd=0 produces out-of-distribution actions.
    # Use the smallest in-distribution cmd (slow forward walk) for settle.
    obs, _ = env.reset()
    settle_speed = (getattr(env, "SPEED_MIN_FRAC", 0.4) + 0.05) * env._ctrl.MAX_SPEED
    env._cmd = np.array([settle_speed, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(200):
        obs, *_ = env.step(zero_action)

    state = {
        "cmd":         np.zeros(9, dtype=np.float32),
        "boost_until": 0.0,    # time.time() < this → boost active
        "dirty":       False,
    }
    key_cb = make_key_callback(state, env._ctrl)

    print(f"\nCheckpoint:  {ckpt}")
    print(f"Stage:       {args.stage}")
    print(f"gait_scale:  {env.gait_scale:.2f}")
    print(f"Action mode: {'stochastic' if args.stochastic else 'deterministic'}")
    print(f"MAX_SPEED:   {env._ctrl.MAX_SPEED:.4f} m/s")
    print(f"MAX_YAW:     {env._ctrl.MAX_YAW_RATE:.4f} rad/s")
    print()
    print("  Controls:")
    print("    W/S  vx forward/back        I/K  pitch up/down")
    print("    A/D  vy left/right strafe   J/L  roll  right/left")
    print("    Q/E  yaw  CCW / CW          R/F  height up/down")
    print("                                T/G  width  wider/narrower")
    print("    Z/X  shift X fore/aft       C/V  shift Y left/right")
    print("    SPACE  zero all      P  print cmd")
    print("    1  walk forward      2  spin CCW       3  zero")
    print(f"    SHIFT (hold)  run mode — vx/vy/wz × {SPEED_BOOST_MULT:.1f}")
    print()

    # Note: env.step() in this script is called outside any vec_env wrapper, so
    # there's no pre-existing live_watch publishing — the viewer below renders
    # directly from env._data.
    log_every = int(0.5 / env._dt)   # ~2 Hz cmd printout
    step_count = 0

    with mujoco.viewer.launch_passive(env._model, env._data, key_callback=key_cb) as viewer:
        sim_t0  = env._data.time
        wall_t0 = time.time()
        while viewer.is_running():
            # Apply run-mode boost if Shift is currently held. Detected via
            # the auto-decay timer; if user is holding Shift, OS auto-repeat
            # keeps boost_until fresh — when they release, the timer expires
            # within BOOST_HOLD_TIMEOUT_S and boost ends naturally.
            is_boosting = time.time() < state["boost_until"]
            effective_cmd = state["cmd"].copy()
            if is_boosting:
                effective_cmd[0] *= SPEED_BOOST_MULT     # vx
                effective_cmd[1] *= SPEED_BOOST_MULT     # vy
                effective_cmd[2] *= SPEED_BOOST_MULT     # wz

            env._cmd = effective_cmd
            obs = env._get_obs()
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()

            step_count += 1
            if state["dirty"] or step_count % log_every == 0:
                if state["dirty"] or np.any(state["cmd"] != 0.0) or is_boosting:
                    boost_tag = "  [RUN]" if is_boosting else ""
                    print(f"  {fmt_cmd(effective_cmd)}   "
                          f"track={info.get('tracking_reward', 0):.2f}{boost_tag}")
                state["dirty"] = False

            if terminated or truncated:
                tag = "FELL" if terminated else "TIMEOUT"
                print(f"  [{tag}] resetting")
                obs, _ = env.reset()
                # Settle for a moment after reset so the viewer doesn't show a snap.
                for _ in range(100):
                    env.step(np.zeros(env.action_space.shape, dtype=np.float32))

            # Real-time pacing.
            target = wall_t0 + (env._data.time - sim_t0)
            lag = target - time.time()
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
