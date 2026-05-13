"""
diagnose_bc_axes.py — does the BC policy actually differentiate between
cmd axes, or is it just outputting "tilt-ish" for any tilt input?

Loads the BC pickle, builds a fixed observation, and probes the
policy with cmds that vary along ONE axis at a time. If actions for
cmd[pitch]=+0.15 vs cmd[roll]=+0.15 are nearly identical, the BC
collapsed both axes into one mode — it never learned the difference.
Compares against the SCAFFOLD's bc_target for the same cmd as the
ground truth: if the policy's output diverges from the scaffold for
roll but matches for pitch, BC undertrained on the roll dimension.
"""
from __future__ import annotations

import argparse
import math
import pickle

import numpy as np
import jax, jax.numpy as jnp

from brax.training.agents.ppo.networks import make_ppo_networks
from brax.training.acme import running_statistics

from envs.hexapod_env import HexapodEnv

DEG = math.radians


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--params",
                   default="checkpoints/bc_pretrained_jax/params.pkl")
    p.add_argument("--model",
                   default="models/phantomx_simple_mjx.xml")
    args = p.parse_args()

    with open(args.params, "rb") as f:
        loaded = pickle.load(f)
    norm, pol, _val = loaded if len(loaded) == 3 else (*loaded, None)
    print(f"loaded {args.params}")

    # Match training-time obs construction exactly.
    env = HexapodEnv(stage=3, model_path=args.model)
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    # Reset, settle, freeze — so all probes share the same physical state.
    obs, _ = env.reset()
    settle_speed = (env.SPEED_MIN_FRAC + 0.05) * env._ctrl.MAX_SPEED
    env._cmd = np.array([settle_speed, 0, 0, 0, 0, 0, 0, 0, 0],
                        dtype=np.float32)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(200):
        obs, *_ = env.step(zero_action)

    # JAX-style obs builder (mirrors the BC training-time construction).
    def build_obs(cmd):
        env._cmd = cmd.astype(np.float32)
        env._latest_feet_body = env._ctrl.compute_foot_targets(
            env._cmd, env._sim_time).astype(np.float32)
        qpos = env._data.qpos
        qvel = env._data.qvel
        phase = env._ctrl.get_phase(env._sim_time)
        body_linvel = np.asarray(env._body_frame_linvel(), dtype=np.float32)
        return np.concatenate([
            qpos[7:],
            qvel[6:],
            qpos[3:7],
            qvel[3:6],
            np.zeros(3, dtype=np.float32),
            env._latest_feet_body.flatten(),
            [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)],
            env._cmd,
            body_linvel,
        ]).astype(np.float32)

    nets = make_ppo_networks(observation_size=obs_size, action_size=act_size,
                             preprocess_observations_fn=running_statistics.normalize)

    @jax.jit
    def infer(obs):
        out = nets.policy_network.apply(norm, pol, obs[None, :])[0]
        return out[:act_size]                     # action mean

    # Probe: forward walk, then sweep each cmd axis ±max one at a time.
    speed = (env.SPEED_MIN_FRAC + 0.40) * env._ctrl.MAX_SPEED   # mid-band walk
    base_cmd = np.array([speed, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    base_action = np.asarray(infer(jnp.asarray(build_obs(base_cmd))))
    print(f"\nbase_cmd = walk forward only")
    print(f"base_action[0:6]: {base_action[:6].round(3)}")

    probes = [
        ("pitch +10°",  3, +DEG(10)),
        ("pitch -10°",  3, -DEG(10)),
        ("roll  +15°",  4, +DEG(15)),
        ("roll  -15°",  4, -DEG(15)),
        ("roll  +8°",   4, +DEG(8)),
        ("roll  -8°",   4, -DEG(8)),
    ]
    print(f"\n  per-axis cmd response (action delta from base_cmd):")
    print(f"  {'axis':<14} {'|action - base|':<24} {'|scaffold_bc - base|':<26}")
    print(f"  {'-'*14} {'-'*24} {'-'*26}")
    for label, slot, val in probes:
        cmd = base_cmd.copy()
        cmd[slot] = val
        # Policy action.
        a = np.asarray(infer(jnp.asarray(build_obs(cmd))))
        # Scaffold's "true" bc_target for the same cmd: the action the
        # policy SHOULD output to reproduce the scaffold's joint targets.
        scaffold_joints = env._ctrl.predict(cmd, env._sim_time)
        bc_truth = np.clip(
            (scaffold_joints - env._ctrl.gait_neutral_pose) / env.RESIDUAL_SCALE_MAX,
            -1.0, 1.0).astype(np.float32)
        # Diffs.
        delta_pol = float(np.linalg.norm(a - base_action))
        delta_truth = float(np.linalg.norm(bc_truth - base_action))
        print(f"  {label:<14} {delta_pol:<24.4f} {delta_truth:<26.4f}")

    print("\nInterpretation:")
    print("  * If policy delta is ~0 for roll but >0 for pitch: BC didn't learn roll.")
    print("  * If both deltas are large but in similar directions: BC collapsed to a")
    print("    generic 'tilt' mode and conflated pitch with roll.")
    print("  * If policy delta tracks scaffold delta: BC learned the axis correctly.")


if __name__ == "__main__":
    main()
