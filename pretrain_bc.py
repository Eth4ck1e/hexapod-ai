"""
pretrain_bc.py — supervised behavioral cloning of the analytical scaffold.

Teaches the PPO policy network to predict the joint-target action that, when
applied at gait_scale=0.0, would reproduce the scaffold's joint targets. After
this, the network already produces walking trajectories before any RL — RL
only refines what BC laid down.

How it works:
  1. Build the env with gait_scale=1.0 (scaffold drives the bot fully).
  2. Run the env for many steps. Each step, the env exposes:
       info["bc_target"] = clip((scaffold_target - gait_neutral) / RESIDUAL_SCALE_MAX, ±1)
       This is the action the policy SHOULD output to mimic the scaffold.
  3. Collect (obs, bc_target) pairs into a buffer.
  4. Train an MlpPolicy (same arch SB3 PPO uses) supervised: MSE between the
     policy's mean output and bc_target.
  5. Save the trained policy weights as a SB3-compatible checkpoint.

Outputs:
  checkpoints/bc_pretrained/policy.zip
        SB3 PPO checkpoint with the BC-pretrained network weights.

Then `train_mac.py --bc-init checkpoints/bc_pretrained/policy` continues with
RL fine-tuning, starting from a network that already knows how to walk.

Usage:
  python pretrain_bc.py                            # default: 500K steps, 10 epochs
  python pretrain_bc.py --steps 1000000            # collect more data
  python pretrain_bc.py --epochs 20                # train more
  python pretrain_bc.py --out checkpoints/bc_v2    # custom output path
"""

import argparse
import os
import time
from functools import partial

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.hexapod_env import HexapodEnv


def _make_env(idx, stage):
    env = HexapodEnv(stage=stage, live_watch=False)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=2000)
    env = Monitor(env)
    return env


def collect_demos(n_steps: int, n_envs: int = 8, stage: int = 1):
    """Roll out the scaffold and collect (obs, bc_target) pairs.

    Uses SubprocVecEnv with gait_scale=1.0 so the scaffold drives every step.
    The action sent into env.step is irrelevant under the new weighted-avg
    mixing (gs=1.0 → policy stream weighted to 0), so we send zeros.
    """
    print(f"[bc] collecting {n_steps:,} demonstration steps with {n_envs} envs (stage={stage})…")
    vec = SubprocVecEnv([partial(_make_env, i, stage) for i in range(n_envs)])
    vec.env_method("set_gait_scale", 1.0)

    obs = vec.reset()
    obs_dim    = vec.observation_space.shape[0]
    action_dim = vec.action_space.shape[0]
    obs_buf    = np.zeros((n_steps, obs_dim),    dtype=np.float32)
    target_buf = np.zeros((n_steps, action_dim), dtype=np.float32)

    zero_action = np.zeros((n_envs, action_dim), dtype=np.float32)
    written = 0
    t0 = time.time()
    while written < n_steps:
        next_obs, _, _, infos = vec.step(zero_action)
        # Pull bc_target from each env's info. SB3 propagates infos as a list
        # of dicts (length n_envs). Each contains "bc_target" set by HexapodEnv.
        for i in range(n_envs):
            if written >= n_steps:
                break
            obs_buf[written]    = obs[i]
            target_buf[written] = infos[i]["bc_target"]
            written += 1
        obs = next_obs
        if written % 50_000 == 0 or written == n_steps:
            elapsed = time.time() - t0
            print(f"[bc]  collected {written:,}/{n_steps:,}  "
                  f"({written/max(1, elapsed):.0f} steps/s)")
    vec.close()
    return obs_buf, target_buf


def train_bc(obs_buf, target_buf, epochs: int, batch_size: int, lr: float,
             out_path: str, stage: int = 1):
    """Build a fresh PPO model, train its policy supervised on (obs → target)."""
    print(f"[bc] training: epochs={epochs} batch={batch_size} lr={lr}")
    # We need a temp env to construct PPO with matching spaces — won't be used
    # for any rollouts.
    env = HexapodEnv(stage=stage)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        verbose=0, device="cpu",
    )
    env.close()

    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    obs_t    = torch.as_tensor(obs_buf,    dtype=torch.float32)
    target_t = torch.as_tensor(target_buf, dtype=torch.float32)
    n = obs_t.shape[0]

    for ep in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            batch_obs    = obs_t[idx]
            batch_target = target_t[idx]
            # SB3's MlpPolicy: get_distribution(obs).distribution.mean
            # gives the deterministic mean of the gaussian — what predict(determ=True) returns.
            dist = policy.get_distribution(batch_obs)
            mean_action = dist.distribution.mean   # for SquashedDiag etc, this is pre-squash mean
            loss = ((mean_action - batch_target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        print(f"[bc] epoch {ep+1:3d}/{epochs}  mean_loss={epoch_loss/max(1,n_batches):.5f}")

    # Save: PPO.save writes everything; the policy weights are inside.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save(out_path)
    print(f"[bc] saved BC-pretrained PPO to {out_path}.zip")


def evaluate_bc_quality(out_path: str, n_steps: int = 1000, stage: int = 1):
    """Quick sanity-check: load the BC-pretrained model, run it at gs=0.0
    against the scaffold's targets, and report mean per-step MSE."""
    print(f"[bc] evaluating BC quality at gs=0.0 (stage={stage})…")
    env = HexapodEnv(stage=stage)
    env.gait_scale = 0.0   # pure policy
    model = PPO.load(out_path, env=env)

    obs, _ = env.reset()
    err_sum, n = 0.0, 0
    sliding = []
    n_contacts = []
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        err_sum += float(np.mean((action - info["bc_target"]) ** 2))
        sliding.append(info["sliding_pen"])
        n_contacts.append(info["n_contact"])
        n += 1
        if term:
            print(f"[bc]  terminated early at step {n}: fell={info['fell']} "
                  f"no_progress={info['no_progress']}")
            obs, _ = env.reset()
    print(f"[bc] eval: mean BC mse against scaffold = {err_sum/n:.5f}  "
          f"(0 = perfect mimic)")
    print(f"[bc] eval: mean n_contact={np.mean(n_contacts):.2f}  "
          f"mean sliding_pen={np.mean(sliding):.4f}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pretrain PPO policy via BC on scaffold gait.")
    p.add_argument("--steps",  type=int, default=500_000, help="demo steps to collect")
    p.add_argument("--n-envs", type=int, default=8, help="parallel envs for collection")
    p.add_argument("--epochs", type=int, default=10, help="supervised training epochs")
    p.add_argument("--batch",  type=int, default=512)
    p.add_argument("--lr",     type=float, default=3e-4)
    p.add_argument("--out",    default="checkpoints/bc_pretrained/policy",
                   help="output path (without .zip)")
    p.add_argument("--stage",  type=int, default=1, choices=[1, 2, 3, 4],
                   help="curriculum stage controlling cmd-mask during demo "
                        "collection. 1 = translation only (vx,vy). 2 = + wz, "
                        "height, width. 3 = + pitch, roll, body shifts (full "
                        "scaffold motion). 4 = same as 3 but body_linvel "
                        "dropped from obs.")
    args = p.parse_args()

    obs_buf, target_buf = collect_demos(args.steps, n_envs=args.n_envs,
                                        stage=args.stage)
    train_bc(obs_buf, target_buf, epochs=args.epochs, batch_size=args.batch,
             lr=args.lr, out_path=args.out, stage=args.stage)
    evaluate_bc_quality(args.out, stage=args.stage)
