"""
validate_jax_vs_gym.py — Phase 5 part A: semantic parity test.

Loads the SubprocVecEnv gym `HexapodEnv` and the MJX `hexapod_env_jax`
on the same MJCF (`phantomx_simple_mjx.xml`), drives both with an
identical commanded vector and identical action sequence, and compares
per-step reward + key metrics.

If parity holds (small diffs, same termination behavior), we know the
JAX env's reward/termination semantics match the gym env. Differences
above a small float-precision threshold mean a real semantic bug.

Run inside WSL2 (need both gym deps + jax+mjx):
    PYTHONPATH=. ~/.venv-mjx/bin/python validate_jax_vs_gym.py
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

# Gym side.
from envs.hexapod_env import HexapodEnv
# JAX side.
from envs import hexapod_env_jax as hex_jax


MODEL_PATH = "models/phantomx_simple_mjx.xml"
N_STEPS    = 200          # ~1 sec at dt=0.005 — enough to land + take steps
SEED       = 0


def main() -> None:
    print("=" * 72)
    print("Phase 5A: gym (SubprocVecEnv) vs JAX (MJX) semantic parity test")
    print("=" * 72)

    # ----- Build both envs on the SAME MJCF -----
    # Note: the gym env defaults to phantomx.xml; pass model_path to override.
    print("loading gym env (HexapodEnv)...", end=" ", flush=True)
    gym_env = HexapodEnv(stage=3, model_path=MODEL_PATH)
    print("done")

    print("loading JAX env (hexapod_env_jax)...", end=" ", flush=True)
    jax_params = hex_jax.make_env_params(MODEL_PATH)
    jit_step   = jax.jit(hex_jax.step)
    print("done")

    # ----- Drive both with the same fixed cmd + zero action sequence -----
    # We bypass _sample_cmd and inject a known cmd directly on each side so
    # the two envs share an identical task. gait_scale=0 -> pure policy
    # (with zero action that means joint targets at gait_neutral).
    cmd = np.array([
        0.05, 0.0, 0.10, 0.05, -0.05, -0.005, 0.005, 0.000, 0.000
    ], dtype=np.float32)
    actions = np.zeros((N_STEPS, 18), dtype=np.float32)
    print(f"\ndriving both envs for {N_STEPS} steps with cmd={cmd.round(3)}\n")

    # ---- Gym reset + force the same cmd ----
    gym_obs, _ = gym_env.reset(seed=SEED)
    gym_env._cmd = cmd.copy()             # override sampled cmd
    gym_env.gait_scale = 0.0
    gym_rewards    = np.zeros(N_STEPS)
    gym_dones      = np.zeros(N_STEPS, dtype=bool)
    gym_n_contact  = np.zeros(N_STEPS)
    gym_foot_dev   = np.zeros(N_STEPS)
    gym_track      = np.zeros(N_STEPS)

    for i in range(N_STEPS):
        obs, r, term, trunc, info = gym_env.step(actions[i])
        gym_rewards[i]   = r
        gym_dones[i]     = term or trunc
        gym_n_contact[i] = info.get("n_contact", 0)
        gym_foot_dev[i]  = info.get("foot_dev_total", 0.0)
        gym_track[i]     = info.get("tracking_reward", 0.0)
        if term or trunc:
            break
    gym_steps_done = i + 1

    # ---- JAX reset + override cmd ----
    rng = jax.random.PRNGKey(SEED)
    state = hex_jax.reset(jax_params, rng, gait_scale=0.0)
    state = state._replace(cmd=jnp.asarray(cmd))   # override sampled cmd
    state = jax.jit(lambda s: s)(state)            # ensure on-device
    jax_rewards    = np.zeros(N_STEPS)
    jax_dones      = np.zeros(N_STEPS, dtype=bool)
    jax_n_contact  = np.zeros(N_STEPS)
    jax_foot_dev   = np.zeros(N_STEPS)
    jax_track      = np.zeros(N_STEPS)

    for i in range(N_STEPS):
        state = jit_step(jax_params, state, jnp.asarray(actions[i]))
        jax_rewards[i]   = float(state.reward)
        jax_dones[i]     = bool(state.done)
        jax_n_contact[i] = float(state.metrics["n_contact"])
        jax_foot_dev[i]  = float(state.metrics["foot_dev_total"])
        jax_track[i]     = float(state.metrics["tracking_reward"])
        if bool(state.done):
            break
    jax_steps_done = i + 1

    # ----- Diagnostics -----
    n = min(gym_steps_done, jax_steps_done)
    print(f"gym ran {gym_steps_done} steps, jax ran {jax_steps_done} steps "
          f"(comparing first {n})")
    print()

    def _stats(name, gy, jx):
        d = np.abs(gy[:n] - jx[:n])
        rel = d / (np.abs(gy[:n]) + 1e-9)
        print(f"  {name:<22} gym mean={gy[:n].mean():+.4f}  "
              f"jax mean={jx[:n].mean():+.4f}  "
              f"max_abs_diff={d.max():.4e}  "
              f"max_rel_diff={rel.max():.2%}")

    _stats("reward",            gym_rewards,    jax_rewards)
    _stats("tracking_reward",   gym_track,      jax_track)
    _stats("n_contact",         gym_n_contact,  jax_n_contact)
    _stats("foot_dev_total",    gym_foot_dev,   jax_foot_dev)

    print()
    print(f"  termination match: gym_done={gym_dones[:n].any()}  "
          f"jax_done={jax_dones[:n].any()}")

    # Per-step reward profile (first 10, last 10).
    print(f"\n  per-step reward (gym | jax | diff):")
    for i in list(range(min(5, n))) + (
            list(range(max(5, n-5), n)) if n > 10 else []):
        d = gym_rewards[i] - jax_rewards[i]
        print(f"    step {i:>3}:  {gym_rewards[i]:+8.4f}   "
              f"{jax_rewards[i]:+8.4f}   diff={d:+.4e}")

    # Pass/fail thresholds.
    max_reward_diff = float(np.max(np.abs(gym_rewards[:n] - jax_rewards[:n])))
    print()
    if max_reward_diff < 0.01:
        print(f"PARITY OK   max reward diff = {max_reward_diff:.4f}  "
              f"< 0.01 (within fp32 + numerical noise)")
        rc = 0
    elif max_reward_diff < 0.1:
        print(f"PARITY MARGINAL   max reward diff = {max_reward_diff:.4f}  "
              f"between 0.01 and 0.1 — investigate but probably tolerable")
        rc = 0
    else:
        print(f"PARITY MISMATCH   max reward diff = {max_reward_diff:.4f}  "
              f">= 0.1 — bug in reward formula or env semantics")
        rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
