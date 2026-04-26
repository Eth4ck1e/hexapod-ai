"""
Benchmarks SubprocVecEnv throughput at different N_ENVS.
Run this before committing to a long training run.

Usage: python benchmark_envs.py
"""
import time
import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.vel_cmd_wrapper import VelCmdWrapper

STEPS     = 2_000   # steps per env per trial
CANDIDATES = [8, 16, 24, 32, 48, 64]


def make_env():
    e = gym.make("Ant-v5")
    return VelCmdWrapper(e)


if __name__ == "__main__":
    print(f"{'N_ENVS':>8}  {'fps':>10}  {'total_steps/s':>14}")
    print("-" * 38)

    best_fps = 0
    best_n   = 0

    for n in CANDIDATES:
        vec_env = make_vec_env(make_env, n_envs=n, vec_env_cls=SubprocVecEnv)
        obs = vec_env.reset()

        # Warmup
        for _ in range(50):
            actions = np.array([vec_env.action_space.sample() for _ in range(n)])
            vec_env.step(actions)

        # Timed run
        start = time.perf_counter()
        for _ in range(STEPS):
            actions = np.array([vec_env.action_space.sample() for _ in range(n)])
            vec_env.step(actions)
        elapsed = time.perf_counter() - start

        vec_env.close()

        total_steps = STEPS * n
        fps = total_steps / elapsed
        marker = " <-- best" if fps > best_fps else ""
        if fps > best_fps:
            best_fps = fps
            best_n   = n

        print(f"{n:>8}  {fps:>10,.0f}  {total_steps:>14,}{marker}")

    print("-" * 38)
    print(f"Recommended N_ENVS: {best_n}  ({best_fps:,.0f} fps)")
