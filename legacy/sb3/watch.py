"""
watch.py — render a trained PPO checkpoint in the mujoco viewer.

Loads the latest checkpoint by default. Useful for visually checking how
training is progressing — leave it open while train.py is still running.
The viewer auto-reloads new checkpoints? No — Stable-Baselines3 doesn't
support hot-reload, so re-run watch.py to pick up newer checkpoints.

Usage (macOS — mjpython required for the viewer):
    mjpython watch.py                                  # latest checkpoint, stage 1
    mjpython watch.py --stage 2                        # eval at stage 2
    mjpython watch.py --gait-scale 0.5                 # override gait_scale
    mjpython watch.py --ckpt path/to/ckpt_no_zip       # explicit checkpoint (no .zip suffix)
    mjpython watch.py --run hexapod_v2_curriculum      # specific run directory
    mjpython watch.py --stochastic                     # sample actions instead of deterministic
"""
import argparse
import glob
import math
import os
import re
import sys
import time

from stable_baselines3 import PPO

from envs.hexapod_env import HexapodEnv


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
    """Pick the newest *complete* checkpoint in this directory.

    Skips files modified within the last 3 seconds — those may still be
    mid-write by the live training process.
    """
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
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def main():
    p = argparse.ArgumentParser(description="Render a trained PPO checkpoint.")
    p.add_argument("--ckpt", default=None, help="checkpoint path (without .zip)")
    p.add_argument("--run",  default=None, help="run dir under checkpoints/")
    p.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--gait-scale", type=float, default=None,
                   help="override gait_scale (0.0–1.0). Default: 1.0 for stage 1, 0.6 stage 2, 0.0 stage 3+.")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions stochastically (default deterministic)")
    args = p.parse_args()

    # Resolve checkpoint.
    if args.ckpt:
        ckpt = args.ckpt
    else:
        run_dir = (os.path.join("checkpoints", args.run) if args.run
                   else latest_run_dir())
        if run_dir is None:
            print("No run directory found under checkpoints/. Wait for training to save a checkpoint or pass --ckpt.")
            sys.exit(1)
        ckpt = latest_checkpoint(run_dir)
        if ckpt is None:
            print(f"No complete checkpoint in {run_dir} yet. First save is at ~2M steps "
                  f"(~7 min into training). Try again shortly.")
            sys.exit(1)

    env = HexapodEnv(stage=args.stage, render_mode="human")
    # Default gait_scale to a sensible value for the stage if not specified.
    default_gs = {1: 1.0, 2: 0.6, 3: 0.0, 4: 0.0}[args.stage]
    env.gait_scale = args.gait_scale if args.gait_scale is not None else default_gs

    model = PPO.load(ckpt, env=env)
    print(f"\nCheckpoint:  {ckpt}")
    print(f"Stage:       {args.stage}")
    print(f"gait_scale:  {env.gait_scale:.2f}")
    print(f"Action mode: {'stochastic' if args.stochastic else 'deterministic'}")
    print()

    obs, _ = env.reset()
    # Settle to a stable standing pose before starting evaluation. CRITICAL:
    # use the smallest in-distribution cmd (NOT zero) — Mac env never trained
    # on cmd=0 so feeding zeros here produces OOD garbage joint residuals.
    import numpy as _np
    settle_speed = (getattr(env, "SPEED_MIN_FRAC", 0.4) + 0.05) * env._ctrl.MAX_SPEED
    env._cmd = _np.array([settle_speed, 0, 0, 0, 0, 0, 0, 0, 0], dtype=_np.float32)
    _zero = _np.zeros(env.action_space.shape, dtype=_np.float32)
    for _ in range(200):
        obs, *_settle = env.step(_zero)

    ep_reward = 0.0
    ep_steps  = 0
    ep_count  = 0

    while True:
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_steps  += 1

        # Live status every ~0.5s of sim time.
        if ep_steps % 100 == 0:
            cmd = env._cmd
            print(f"  step={ep_steps:4d}  "
                  f"cmd=(vx={cmd[0]:+.3f} vy={cmd[1]:+.3f} wz={cmd[2]:+.3f}  "
                  f"pitch={math.degrees(cmd[3]):+.1f}° roll={math.degrees(cmd[4]):+.1f}°)  "
                  f"track={info.get('tracking_reward', 0):.3f}  "
                  f"reward={reward:+.3f}")

        if terminated or truncated:
            ep_count += 1
            cmd = env._cmd
            tag = "FELL" if terminated else "TIMEOUT"
            print(f"\n[episode {ep_count}] {tag}  "
                  f"steps={ep_steps}  total_reward={ep_reward:+.1f}  "
                  f"cmd=(vx={cmd[0]:+.3f} vy={cmd[1]:+.3f} wz={cmd[2]:+.3f})\n")
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_steps  = 0


if __name__ == "__main__":
    main()
