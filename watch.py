"""
Usage:
    python watch.py                                            # auto-load newest checkpoint in checkpoints/
    python watch.py checkpoints/hexapod_stage1_v3/final        # explicit path
"""
import os
import re
import sys
import glob
from stable_baselines3 import PPO

from envs.hexapod_env import HexapodEnv


def newest_checkpoint(root="checkpoints"):
    if not os.path.isdir(root):
        return None
    # Pick the run directory with the most recently modified files inside.
    runs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    if not runs:
        return None
    runs.sort(key=lambda d: max((os.path.getmtime(os.path.join(d, f))
                                 for f in os.listdir(d)), default=0))
    latest_run = runs[-1]
    # Prefer final.zip; otherwise highest-numbered step checkpoint.
    final = os.path.join(latest_run, "final.zip")
    if os.path.exists(final):
        return final[:-4]
    step_re = re.compile(r"_(\d+)_steps\.zip$")
    candidates = [(int(step_re.search(f).group(1)),
                   os.path.join(latest_run, f))
                  for f in os.listdir(latest_run) if step_re.search(f)]
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1][:-4]


if len(sys.argv) > 1:
    ckpt = sys.argv[1]
else:
    ckpt = newest_checkpoint()
    if ckpt is None:
        print("No checkpoints found under checkpoints/. Pass a path explicitly.")
        sys.exit(1)

env = HexapodEnv(stage=1, render_mode="human")
model = PPO.load(ckpt, env=env)
print(f"Loaded: {ckpt}")

obs, _ = env.reset()
ep_reward = 0.0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    ep_reward += reward

    if terminated or truncated:
        cmd = env._cmd
        print(f"cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, wz={cmd[2]:+.2f})  ep_reward={ep_reward:.1f}")
        obs, _ = env.reset()
        ep_reward = 0.0
