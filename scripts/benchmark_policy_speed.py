"""
scripts/benchmark_policy_speed.py — measure the v7 policy's actual body
velocity at fixed forward cmd.

No controller, no watcher, no human in the loop. Loads the policy, sets
env._cmd to a target forward speed, runs the policy for N steps, reports
the mean actual body velocity. Compares trivially to commanded.

Used to settle disputes about "the bot is slower in the controller" —
this measures the bot's actual capability under the same code path
watch_controller / watch_demo_jax both use.

Usage:
    $env:PYTHONPATH = "."
    .venv\\Scripts\\python.exe scripts\\benchmark_policy_speed.py `
        --params checkpoints\\amp_to_v7\\iter2\\final\\params.pkl
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain_train import ACTION_SPACE as CHAIN_ACTION_SPACE
from chain_train import MODEL_PATH as CHAIN_MODEL_PATH
from chain_train import enable_jax_cache

enable_jax_cache()

from brax.training.acme import running_statistics
from brax.training.agents.ppo.networks import make_ppo_networks

from envs.hexapod_env import HexapodEnv


def _patched_obs(env: HexapodEnv) -> None:
    """JAX-style obs (matches watch_demo_jax / watch_controller)."""
    def _get_obs(self):
        qpos = self._data.qpos
        qvel = self._data.qvel
        scaffold_hint = self._latest_feet_body.flatten()
        phase = self._ctrl.get_phase(self._sim_time)
        body_linvel = np.asarray(self._body_frame_linvel(), dtype=np.float32)
        if self.stage == 4:
            body_linvel = np.zeros(3, dtype=np.float32)
        return np.concatenate([
            qpos[7:], qvel[6:], qpos[3:7], qvel[3:6], np.zeros(3, dtype=np.float32),
            scaffold_hint,
            [math.sin(2*math.pi*phase), math.cos(2*math.pi*phase)],
            self._cmd, body_linvel,
        ]).astype(np.float32)
    env._get_obs = types.MethodType(_get_obs, env)


def _build_inference(obs_size, act_size):
    networks = make_ppo_networks(
        observation_size=obs_size, action_size=act_size,
        preprocess_observations_fn=running_statistics.normalize)
    @jax.jit
    def infer(norm_p, pol_p, obs, rng):
        logits = networks.policy_network.apply(norm_p, pol_p, obs)
        return networks.parametric_action_distribution.mode(logits)
    return infer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--model", default=CHAIN_MODEL_PATH)
    p.add_argument("--action-space", type=str, default=CHAIN_ACTION_SPACE,
                   choices=["joint", "foot"])
    p.add_argument("--cmd-speed-frac", type=float, default=0.85,
                   help="commanded speed as fraction of MAX_SPEED (default 0.85, "
                        "matches trained max).")
    p.add_argument("--cmd-direction-deg", type=float, default=0.0,
                   help="0 = forward, 90 = left, -90 = right, ±180 = back.")
    p.add_argument("--settle-steps", type=int, default=200)
    p.add_argument("--measure-steps", type=int, default=300)
    args = p.parse_args()

    print(f"Loading params: {args.params}")
    with open(args.params, "rb") as f:
        loaded = pickle.load(f)
    if isinstance(loaded, tuple) and len(loaded) == 3:
        normalizer_params, policy_params, _ = loaded
    elif isinstance(loaded, tuple) and len(loaded) == 2:
        normalizer_params, policy_params = loaded
    else:
        raise SystemExit("Unrecognized params format.")

    env = HexapodEnv(stage=3, render_mode=None, model_path=args.model)
    env.gait_scale = 0.0
    _patched_obs(env)
    obs, _ = env.reset()

    max_speed = float(env._ctrl.MAX_SPEED)
    cmd_mag = args.cmd_speed_frac * max_speed
    theta = math.radians(args.cmd_direction_deg)
    target_vx = cmd_mag * math.cos(theta)
    target_vy = cmd_mag * math.sin(theta)

    print(f"\nMAX_SPEED:    {max_speed:.4f} m/s")
    print(f"cmd magnitude: {cmd_mag:.4f} m/s ({args.cmd_speed_frac*100:.0f}% MAX)")
    print(f"cmd direction: {args.cmd_direction_deg:+.1f}° "
          f"-> (vx={target_vx:+.4f}, vy={target_vy:+.4f})")

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    infer = _build_inference(obs_size, act_size)
    foot_residual_scale = float(env._ctrl.foot_residual_scale_max
                                if hasattr(env._ctrl, "foot_residual_scale_max")
                                else 0.020)

    cmd = np.array([target_vx, target_vy, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    env._cmd = cmd
    rng = jax.random.PRNGKey(0)

    print(f"\nSettling {args.settle_steps} steps at target cmd...")
    for k in range(args.settle_steps):
        env._latest_feet_body = env._ctrl.compute_foot_targets(
            env._cmd, env._sim_time).astype(np.float32)
        obs = env._get_obs()
        rng, sub = jax.random.split(rng)
        action_jax = infer(normalizer_params, policy_params,
                           jnp.asarray(obs)[None, :], sub)[0]
        action = np.asarray(action_jax)
        if args.action_space == "foot":
            foot_residual = action.reshape(6, 3) * foot_residual_scale
            feet_body_target = env._latest_feet_body + foot_residual
            joint_target = env._ctrl.body_to_joints(feet_body_target)
            action = (joint_target - env._ctrl.gait_neutral_pose) / env.RESIDUAL_SCALE_MAX
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
        obs, *_ = env.step(action)

    print(f"Measuring {args.measure_steps} steps...")
    vx_hist = []
    vy_hist = []
    for k in range(args.measure_steps):
        env._latest_feet_body = env._ctrl.compute_foot_targets(
            env._cmd, env._sim_time).astype(np.float32)
        obs = env._get_obs()
        rng, sub = jax.random.split(rng)
        action_jax = infer(normalizer_params, policy_params,
                           jnp.asarray(obs)[None, :], sub)[0]
        action = np.asarray(action_jax)
        if args.action_space == "foot":
            foot_residual = action.reshape(6, 3) * foot_residual_scale
            feet_body_target = env._latest_feet_body + foot_residual
            joint_target = env._ctrl.body_to_joints(feet_body_target)
            action = (joint_target - env._ctrl.gait_neutral_pose) / env.RESIDUAL_SCALE_MAX
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
        obs, *_ = env.step(action)
        bvx, bvy, _ = env._body_frame_linvel()
        vx_hist.append(float(bvx))
        vy_hist.append(float(bvy))

    actual_vx = float(np.mean(vx_hist))
    actual_vy = float(np.mean(vy_hist))
    actual_mag = (actual_vx**2 + actual_vy**2) ** 0.5
    track_pct = 100.0 * actual_mag / cmd_mag if cmd_mag > 1e-6 else 0.0

    print("\n=== RESULTS ===")
    print(f"Commanded:  vx={target_vx:+.4f}  vy={target_vy:+.4f}  mag={cmd_mag:.4f}")
    print(f"Actual avg: vx={actual_vx:+.4f}  vy={actual_vy:+.4f}  mag={actual_mag:.4f}")
    print(f"Tracking:   {track_pct:.1f}% of commanded")


if __name__ == "__main__":
    main()
