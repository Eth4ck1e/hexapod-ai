"""
eval_bc_quick.py — fast programmatic eval of a BC pickle in the gym env.

Runs N short rollouts at random cmds, reports per-episode tracking
reward, fall rate, mean episode length, and the average reward
breakdown by penalty term. Headless (no viewer), CPU-only — won't
fight a running GPU training.

Run on Windows venv:
    .venv\\Scripts\\python.exe eval_bc_quick.py
    .venv\\Scripts\\python.exe eval_bc_quick.py --params <path> --episodes 20
"""
from __future__ import annotations

import argparse
import math
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp

# Enable persistent JAX compilation cache before any JAX op runs.
from chain_train import enable_jax_cache
enable_jax_cache()

from brax.training.agents.ppo.networks import make_ppo_networks
from brax.training.acme import running_statistics

from envs.hexapod_env import HexapodEnv
from watch_demo_jax import _patch_obs_to_jax
# Pull session config defaults so eval defaults match training defaults.
from chain_train import (
    BASE_NAME    as CHAIN_BASE_NAME,
    ACTION_SPACE as CHAIN_ACTION_SPACE,
    MODEL_PATH   as CHAIN_MODEL_PATH,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", default="checkpoints/bc_pretrained_jax/params.pkl")
    p.add_argument("--model",  default=CHAIN_MODEL_PATH)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--stage", type=int, default=1,
                   help="gym env cmd-sampling stage. Use 1 for BC trained "
                        "with --cmd-mask stage1 (translation only).")
    p.add_argument("--gait-scale", type=float, default=0.0,
                   help="scaffold contribution: 0.0 = pure policy (default), "
                        "1.0 = full scaffold. Use to measure scaffold's own "
                        "score (gait_scale=1.0) vs policy contributions.")
    p.add_argument("--action-space", type=str, default=CHAIN_ACTION_SPACE,
                   choices=["joint", "foot"],
                   help=f"must match the policy's training mode "
                        f"(default chain_train.ACTION_SPACE={CHAIN_ACTION_SPACE!r}). "
                        f"foot = policy outputs (6,3) foot residual; we run IK "
                        f"to convert to joint residual before env.step().")
    p.add_argument("--amp-discriminator", type=str, default=None,
                   help="optional path to a trained AMP discriminator pickle. "
                        "When set, scores each step's transition with the "
                        "discriminator and reports mean naturalness score "
                        "(+1 = looks like prior data; -1 = policy garbage).")
    args = p.parse_args()

    print(f"loading {args.params}")
    print(f"  cmd-stage: {args.stage}, gait_scale: {args.gait_scale}, "
          f"action-space: {args.action_space}")
    with open(args.params, "rb") as f:
        norm, pol, _val = pickle.load(f)

    env = HexapodEnv(stage=args.stage, model_path=args.model)
    env.gait_scale = args.gait_scale
    _patch_obs_to_jax(env)
    # Source foot_residual_scale from the JAX env's params so it tracks any
    # env-side changes automatically (no separate flag to keep in sync).
    if args.action_space == "foot":
        from envs import hexapod_env_jax as hex_jax
        _jax_params = hex_jax.make_env_params(args.model)
        foot_residual_scale = float(_jax_params.foot_residual_scale_max)
    else:
        foot_residual_scale = None
    # Disable training-time terminations that fire on cmd discontinuities;
    # we want to see the bot's NATURAL fall rate, not artifacts from the
    # tilt-overshoot guard.
    # Keep MIN_Z and MAX_TILT_SQ active — those are real falls.
    env.TILT_DEV_LIMIT = math.pi
    env.NO_PROGRESS_GRACE = 1_000_000_000

    nets = make_ppo_networks(observation_size=env.observation_space.shape[0],
                             action_size=env.action_space.shape[0],
                             preprocess_observations_fn=running_statistics.normalize)
    @jax.jit
    def infer(obs):
        # Brax PPO's NormalTanhDistribution wraps the network mean with
        # tanh to bound actions in [-1, 1]. Mirror that here — without
        # tanh, OOD obs can push raw means to wild values that drive the
        # actuators into saturation and trigger massive action_rate_pen.
        out = nets.policy_network.apply(norm, pol, obs[None, :])[0]
        mean = out[:env.action_space.shape[0]]
        return jnp.tanh(mean)

    # Optional AMP discriminator scoring
    amp_disc_module = None
    amp_disc_params = None
    amp_score_fn = None
    if args.amp_discriminator is not None:
        from amp.discriminator import Discriminator, style_reward, STATE_DIM, TRANSITION_DIM
        with open(args.amp_discriminator, "rb") as f:
            amp_disc_params = pickle.load(f)
        amp_disc_module = Discriminator()
        @jax.jit
        def amp_score_fn(transition):
            return amp_disc_module.apply(amp_disc_params, transition[None, :])[0]
        print(f"  loaded discriminator: {args.amp_discriminator}")

    def _gym_amp_state(env):
        """Build the AMP state vector from the gym env's MjData. Must
        match amp/prior_data.py's format exactly."""
        qpos = env._data.qpos
        qvel = env._data.qvel
        joint_pos = qpos[7:25]
        joint_vel = qvel[6:24]
        body_linvel = np.asarray(env._body_frame_linvel(), dtype=np.float32)
        body_angvel = qvel[3:6]
        body_height = qpos[2:3]
        # Foot heights in body frame
        R_body = env._data.xmat[env._base_body_id].reshape(3, 3)
        body_pos = env._data.xpos[env._base_body_id]
        foot_zs = []
        for tib_id in env._tibia_body_ids:
            R = env._data.xmat[tib_id].reshape(3, 3)
            foot_world = env._data.xpos[tib_id] + R @ env._FOOT_LOCAL_OFFSET
            foot_body = R_body.T @ (foot_world - body_pos)
            foot_zs.append(foot_body[2])
        foot_heights = np.asarray(foot_zs, dtype=np.float32)
        return np.concatenate([joint_pos, joint_vel, body_linvel,
                                body_angvel, body_height, foot_heights])

    rewards   = []
    lengths   = []
    fells     = []
    track_avg = []
    amp_scores = []      # per-step discriminator scores, averaged over episode
    pen_breakdown = {}

    print(f"running {args.episodes} episodes...")
    t0 = time.perf_counter()
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        # Settle on small forward walk (matches watch_demo_jax convention).
        settle_speed = (env.SPEED_MIN_FRAC + 0.05) * env._ctrl.MAX_SPEED
        env._cmd = np.array([settle_speed, 0, 0, 0, 0, 0, 0, 0, 0],
                            dtype=np.float32)
        for _ in range(200):
            obs, *_ = env.step(np.zeros(env.action_space.shape, np.float32))

        # Then run the actual rollout. cmd is the env's _sample_cmd value.
        env._cmd = env._sample_cmd().astype(np.float32)
        ep_reward = 0.0
        ep_track  = 0.0
        ep_pens   = {}
        ep_amp_score_sum = 0.0
        steps = 0
        fell = False
        prev_amp_state = (_gym_amp_state(env) if amp_score_fn is not None
                          else None)
        for s in range(args.max_steps):
            a = np.asarray(infer(jnp.asarray(obs)))

            # Foot-space action mode: policy outputs (6, 3) foot residual in
            # body frame. Run IK to get joint targets, then express as joint
            # residual relative to gait_neutral so the gym env (which expects
            # a joint residual scaled by RESIDUAL_SCALE_MAX) receives the
            # right thing. Mirrors the conversion in watch_demo_jax.py.
            if args.action_space == "foot":
                foot_residual = a.reshape(6, 3) * foot_residual_scale
                feet_body_target = env._latest_feet_body + foot_residual
                joint_target = env._ctrl.body_to_joints(feet_body_target)
                a = (joint_target - env._ctrl.gait_neutral_pose) / env.RESIDUAL_SCALE_MAX
                a = np.clip(a, -1.0, 1.0).astype(np.float32)

            obs, r, term, trunc, info = env.step(a)
            ep_reward += r
            ep_track  += info.get("tracking_reward", 0.0)
            for k, v in info.items():
                if k.endswith("_pen") or k == "tracking_reward":
                    ep_pens[k] = ep_pens.get(k, 0.0) + float(v)

            # AMP discriminator scoring (optional). Score the (s_t, s_{t+1},
            # cmd_t) transition — cmd-conditional disc requires the cmd active
            # at s_t (env._cmd was set above before env.step).
            if amp_score_fn is not None:
                from amp.discriminator import CMD_DIM_FOR_DISC
                cur_amp = _gym_amp_state(env)
                cmd_disc = np.asarray(env._cmd[:CMD_DIM_FOR_DISC],
                                       dtype=np.float32)
                transition = jnp.asarray(np.concatenate(
                    [prev_amp_state, cur_amp, cmd_disc]))
                d_score = float(amp_score_fn(transition))
                ep_amp_score_sum += d_score
                prev_amp_state = cur_amp

            steps += 1
            if term:
                fell = True
                break
            if trunc:
                break
        rewards.append(ep_reward)
        lengths.append(steps)
        fells.append(fell)
        track_avg.append(ep_track / max(steps, 1))
        if amp_score_fn is not None:
            amp_scores.append(ep_amp_score_sum / max(steps, 1))
        for k, v in ep_pens.items():
            pen_breakdown[k] = pen_breakdown.get(k, 0.0) + v / max(steps, 1)
        cmd_speed = math.hypot(env._cmd[0], env._cmd[1])
        amp_str = (f"  amp={amp_scores[-1]:+.3f}"
                   if amp_score_fn is not None else "")
        print(f"  ep {ep:2d}: steps={steps:4d}  reward={ep_reward:+8.2f}  "
              f"track_avg={ep_track/max(steps,1):.3f}  "
              f"fell={'Y' if fell else 'n'}  "
              f"cmd_speed={cmd_speed:.3f}m/s{amp_str}")

    elapsed = time.perf_counter() - t0
    print(f"\n  ran {args.episodes} eps in {elapsed:.1f}s ({sum(lengths)} env steps)")
    print(f"  ----- summary -----")
    print(f"  mean episode reward  : {np.mean(rewards):+8.2f}  +/- {np.std(rewards):.2f}")
    print(f"  mean episode length  : {np.mean(lengths):4.0f} / {args.max_steps}  ({100*np.mean(lengths)/args.max_steps:.0f}%)")
    print(f"  fell rate            : {sum(fells)}/{args.episodes}  ({100*sum(fells)/args.episodes:.0f}%)")
    print(f"  mean per-step tracking reward: {np.mean(track_avg):.3f}  (1.0 = perfect cmd match)")
    if amp_score_fn is not None:
        print(f"  mean per-step AMP discriminator score: {np.mean(amp_scores):+.3f}  "
              f"(+1 = looks like prior, -1 = policy garbage)")
    print(f"\n  per-step penalty breakdown (mean across episodes):")
    for k in sorted(pen_breakdown.keys()):
        print(f"    {k:<22}: {pen_breakdown[k]/args.episodes:+.4f}")


if __name__ == "__main__":
    main()
