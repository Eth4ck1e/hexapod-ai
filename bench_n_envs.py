"""
bench_n_envs.py — sweep N_ENVS to find the SubprocVecEnv sweet spot.

Spawns a SubprocVecEnv at each N_ENVS in N_ENVS_LIST, runs a short stepping
benchmark, and prints total SPS (samples per wall-second). Run this once
before launching a long training run to pick the best N_ENVS for your CPU.

No PPO overhead — measures raw env stepping + IPC.
"""

import time
import sys

import numpy as np
import gymnasium as gym
from functools import partial
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.hexapod_env import HexapodEnv


# ============================================================================
# CONFIG
# ============================================================================
N_ENVS_LIST     = [16, 20, 24, 28, 32]
WARMUP_STEPS    = 500      # discard initial steps (cache warm-up + IPC priming)
MEASURE_SECONDS = 60.0     # actual timed window per N_ENVS


# ============================================================================
# Top-level factory (must be picklable for SubprocVecEnv on Windows)
# ============================================================================
def _make_env(idx):
    env = HexapodEnv(stage=1)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=2000)
    env = Monitor(env)
    return env


def benchmark(n_envs):
    print(f"\n[N_ENVS = {n_envs}]")
    print(f"  spawning {n_envs} workers...", end=" ", flush=True)
    t0 = time.perf_counter()
    env_fns = [partial(_make_env, i) for i in range(n_envs)]
    vec = SubprocVecEnv(env_fns)
    vec.reset()
    setup_dt = time.perf_counter() - t0
    print(f"done ({setup_dt:.1f}s)")

    # Warm up.
    print(f"  warming up ({WARMUP_STEPS} steps)...", end=" ", flush=True)
    actions = np.zeros((n_envs, 18), dtype=np.float32)
    for _ in range(WARMUP_STEPS):
        vec.step(actions)
    print("done")

    # Timed run — count steps until MEASURE_SECONDS elapses.
    print(f"  measuring for {MEASURE_SECONDS:.1f}s...", end=" ", flush=True)
    step_count = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < MEASURE_SECONDS:
        vec.step(actions)
        step_count += 1
    elapsed = time.perf_counter() - t0
    total_samples = step_count * n_envs
    sps = total_samples / elapsed
    per_env_sps = sps / n_envs
    print(f"done")

    print(f"  vec steps: {step_count:,}  total samples: {total_samples:,}")
    print(f"  TOTAL SPS: {sps:>10,.0f}    per-env SPS: {per_env_sps:>7,.0f}")

    print(f"  closing...", end=" ", flush=True)
    vec.close()
    print("done")
    return {"n_envs": n_envs, "total_sps": sps, "per_env_sps": per_env_sps}


if __name__ == "__main__":
    print("=" * 70)
    print("N_ENVS benchmark — finding the SubprocVecEnv sweet spot")
    print("=" * 70)
    print(f"sweep:    {N_ENVS_LIST}")
    print(f"warmup:   {WARMUP_STEPS} steps")
    print(f"measure:  {MEASURE_SECONDS:.1f} s per setting")
    print(f"estimated total time: ~{len(N_ENVS_LIST) * (MEASURE_SECONDS + 8):.0f} s")

    results = []
    for n in N_ENVS_LIST:
        try:
            results.append(benchmark(n))
        except Exception as e:
            print(f"  [FAILED at N_ENVS={n}] {type(e).__name__}: {e}")

    # Summary.
    print("\n" + "=" * 70)
    print("SUMMARY (sorted by total SPS, highest first)")
    print("=" * 70)
    print(f"  {'N_ENVS':<8}{'total SPS':>12}{'per-env SPS':>14}")
    for r in sorted(results, key=lambda x: -x["total_sps"]):
        print(f"  {r['n_envs']:<8}{r['total_sps']:>12,.0f}{r['per_env_sps']:>14,.0f}")
    if results:
        winner = max(results, key=lambda x: x["total_sps"])
        baseline = next((r for r in results if r["n_envs"] == 16), results[0])
        gain = (winner["total_sps"] / baseline["total_sps"] - 1.0) * 100.0
        print(f"\n  -> Best N_ENVS = {winner['n_envs']}  "
              f"({gain:+.1f}% vs N_ENVS={baseline['n_envs']})")
