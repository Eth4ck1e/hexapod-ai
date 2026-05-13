"""
pretrain_bc_jax.py — JAX-native behavioral cloning of the gait scaffold.

Builds a Brax-PPO-compatible policy that mimics the analytical scaffold's
joint targets via supervised learning. Replaces `pretrain_bc.py` (which
produces an SB3 PPO policy) for the JAX/MJX training pipeline.

Pipeline:
  1. Vmap the JAX env at gait_scale=1.0 (scaffold-driven). With zero
     policy actions the bot is pure scaffold; the env exposes the
     "what action would reproduce the scaffold's joint targets" as
     `metrics["bc_target"]`.
  2. Roll out N_ENVS × N_STEPS demos via `lax.scan` (fast — env runs
     at GPU throughput).
  3. Build a Brax `make_ppo_networks` policy network (matches the arch
     `train_jax.py` will resume into).
  4. Train policy weights via Adam + MSE between predicted action mean
     and the recorded `bc_target`.
  5. Save in Brax's `restore_params` tuple format:
       (normalizer_params, policy_params, value_params)
     `train_jax.py` then resumes via `--restore <path>`.

Run inside WSL2:
    PYTHONPATH=. ~/.venv-mjx/bin/python pretrain_bc_jax.py
"""
from __future__ import annotations

import argparse
import functools
import math
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

# Enable persistent JAX compilation cache before any JAX op runs.
from chain_train import enable_jax_cache
enable_jax_cache()

from brax.training.agents.ppo.networks import make_ppo_networks
from brax.training.acme import running_statistics, specs

from envs.hexapod_brax_env import HexapodBraxEnv


# ============================================================================
# CONFIG
# ============================================================================
from chain_train import MODEL_PATH      # one source of truth
OUT_DIR       = Path("checkpoints") / "bc_pretrained_jax"

# Demo collection — at 564k SPS this is fast.
N_ENVS_DEMO   = 2048            # v24+ (2026-05-12): halved 4096→2048 to fit
                                # the larger 114-dim obs in 16GB VRAM.
                                # 2048×1500 still gives ~3M demos.
N_STEPS_DEMO  = 1500            # 2048 × 1500 ≈ 3M demos
DEMO_GAIT_SCALE = 1.0           # full scaffold drives the bot

# BC training.
N_EPOCHS      = 6
BATCH_SIZE    = 8192
LEARNING_RATE = 1e-3
SEED          = 0

# Log-stddev to bake into the saved policy's output bias for the log_std
# slots. Brax PPO's policy outputs concat[mean, log_std] per dim; the BC
# loss only trains the mean, so we set log_std manually here.
#   0.0 → std=1.0 (~57° per-joint jitter — destroys a working walker)
#  -3.0 → std≈0.05 (~2.3° per-joint jitter — refinement zone, recommended
#         when chaining BC → PPO so PPO doesn't immediately destabilize)
#  -3.5 → std≈0.03 (~1.4° — polish; PPO learns slowly but safely)
LOG_STD_INIT  = -4.0       # was -3.0 — std≈0.018, ~1.0° per-joint jitter.
                            # Going below -4.0 starts starving PPO of
                            # gradient signal per CLAUDE.md note, so this
                            # is the practical noise floor for refinement.


# ============================================================================
# Demo collection
# ============================================================================
def _collect_demos(env: HexapodBraxEnv, n_envs: int, n_steps: int, rng):
    """Roll out (obs, bc_target) demos using the scaffold-driven env.

    Returns:
        obs_all: (n_steps * n_envs, obs_size) float32
        bc_all:  (n_steps * n_envs, action_size) float32
    """
    print(f"  spawning {n_envs} envs + {n_steps} steps "
          f"(~{n_envs * n_steps / 1e6:.1f}M demos)...")
    rngs = jax.random.split(rng, n_envs)
    state = jax.jit(jax.vmap(env.reset))(rngs)

    zero_actions = jnp.zeros((n_envs, env.action_size), dtype=jnp.float32)

    @jax.jit
    def step_collect(state, _):
        nstate = jax.vmap(env.step)(state, zero_actions)
        # `bc_target` is exposed in metrics. The auto-reset wrapper at this
        # level might mask it; we read from pipeline_state.metrics directly.
        bc = nstate.pipeline_state.metrics["bc_target"]      # (n_envs, 18)
        return nstate, (nstate.obs, bc)

    t0 = time.perf_counter()
    state, (obs_seq, bc_seq) = jax.lax.scan(
        step_collect, state, jnp.arange(n_steps))
    obs_seq.block_until_ready()
    print(f"  collected in {time.perf_counter() - t0:.1f}s")

    obs_all = obs_seq.reshape(-1, obs_seq.shape[-1])
    bc_all  = bc_seq.reshape(-1, bc_seq.shape[-1])
    return obs_all, bc_all


# ============================================================================
# Networks + BC training
# ============================================================================
# v11+: paper-matching Actor (Chen et al. uses 256, 128, 64 for the
# low-level MLP). Brax default is (32,32,32,32). Param count ~50k —
# still trivial for ESP32-S3 (~200KB float32, ~50KB int8) given 8MB
# PSRAM. Generous capacity for recovery + edge-case behaviors and
# leaves headroom for distillation experiments later.
POLICY_HIDDEN_LAYERS = (256, 128, 64)


def _make_networks(obs_size, act_size, normalize):
    """Build Brax PPO networks. `normalize` is the running-statistics
    normalize function, matched to what train_jax.py uses.

    Policy net architecture: POLICY_HIDDEN_LAYERS. Value net stays at the
    Brax default (5 × 256) — critic is training-only, doesn't ship.
    """
    return make_ppo_networks(
        observation_size=obs_size,
        action_size=act_size,
        preprocess_observations_fn=normalize,
        policy_hidden_layer_sizes=POLICY_HIDDEN_LAYERS,
    )


def _bc_loss(policy_params, normalizer_params, networks, obs_batch, bc_batch):
    """MSE between predicted action mean and scaffold's bc_target.

    Brax policy network outputs concat[mean, log_std] in the last dim.
    BC ignores log_std — we'd just train the mean to match.
    """
    out = networks.policy_network.apply(
        normalizer_params, policy_params, obs_batch)
    # First half = action mean.
    act_size = bc_batch.shape[-1]
    mean = out[..., :act_size]
    return jnp.mean((mean - bc_batch) ** 2)


def _train_bc(networks, obs_all, bc_all, n_epochs, batch_size, lr, rng):
    n_samples = obs_all.shape[0]
    obs_size  = obs_all.shape[-1]
    act_size  = bc_all.shape[-1]

    # Init normalizer with statistics from the collected demos so the
    # policy sees obs in the normalized regime PPO will use later.
    print(f"  initializing normalizer from {n_samples:,d} demo obs...")
    norm_init = running_statistics.init_state(specs.Array((obs_size,), jnp.float32))
    normalizer_params = running_statistics.update(norm_init, obs_all)

    rng, init_rng = jax.random.split(rng)
    policy_params = networks.policy_network.init(init_rng)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(policy_params)

    @jax.jit
    def update(params, opt_state, obs, bc):
        loss, grads = jax.value_and_grad(_bc_loss)(
            params, normalizer_params, networks, obs, bc)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    n_batches = n_samples // batch_size
    print(f"  training {n_epochs} epochs × {n_batches:,d} batches "
          f"(batch={batch_size})...")
    for epoch in range(n_epochs):
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, n_samples)
        obs_shuf = obs_all[perm]
        bc_shuf  = bc_all[perm]
        epoch_loss = 0.0
        t0 = time.perf_counter()
        for b in range(n_batches):
            lo, hi = b * batch_size, (b + 1) * batch_size
            policy_params, opt_state, loss = update(
                policy_params, opt_state,
                obs_shuf[lo:hi], bc_shuf[lo:hi])
            epoch_loss += float(loss)
        avg_loss = epoch_loss / n_batches
        elapsed = time.perf_counter() - t0
        print(f"  epoch {epoch+1:>2}/{n_epochs}  loss={avg_loss:.5f}  "
              f"({elapsed:.1f}s)")

    # Override the policy's log_std output bias to LOG_STD_INIT. The Brax
    # policy network's last layer (`hidden_4`) has bias shape (2*act_size,)
    # where slots [0:act_size] are the action mean and [act_size:2*act_size]
    # are the log_std outputs. BC's MSE loss only trains the mean half, so
    # log_std stays at flax's default init (~0, std=1.0) — way too noisy
    # for chaining into PPO. We rewrite the log_std bias to LOG_STD_INIT so
    # PPO starts from a low-noise refinement regime.
    last_layer_key = max(policy_params["params"].keys(),
                         key=lambda k: int(k.split("_")[1]))
    last_bias = policy_params["params"][last_layer_key]["bias"]
    new_bias = last_bias.at[act_size:].set(jnp.float32(LOG_STD_INIT))
    new_layer = dict(policy_params["params"][last_layer_key])
    new_layer["bias"] = new_bias
    new_params = dict(policy_params["params"])
    new_params[last_layer_key] = new_layer
    policy_params = {"params": new_params}
    print(f"  set policy log_std bias to {LOG_STD_INIT} "
          f"(std≈{math.exp(LOG_STD_INIT):.3f}, ~{math.degrees(math.exp(LOG_STD_INIT)):.1f}° per-joint jitter)")

    # Init value head freshly (PPO will train it during RL phase).
    value_params = networks.value_network.init(jax.random.PRNGKey(SEED + 999))
    return normalizer_params, policy_params, value_params


# ============================================================================
# main
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs",     type=int, default=N_ENVS_DEMO)
    parser.add_argument("--n-steps",    type=int, default=N_STEPS_DEMO)
    parser.add_argument("--n-epochs",   type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--out-dir",    type=str, default=str(OUT_DIR))
    parser.add_argument("--cmd-mask",   type=str, default="stage3",
                        choices=["stage1", "stage2", "stage3", "paper", "paper_stance"],
                        help="cmd-slot curriculum stage to use during demo "
                             "collection — match this to the train_jax.py mask")
    parser.add_argument("--action-space", type=str, default="joint",
                        choices=["joint", "foot"],
                        help="must match train_jax.py. In foot-space the BC "
                             "target is zeros (scaffold drives baseline), so "
                             "BC mostly serves to initialize log_std and the "
                             "obs normalizer.")
    parser.add_argument("--wz-trim-vx", type=float, default=0.0,
                        help="scaffold yaw trim coef applied as -k * cmd[0] "
                             "before scaffold call. Cancels physics drift in "
                             "BC dataset. Tuned: 0.005.")
    parser.add_argument("--wz-trim-vy-abs", type=float, default=0.0,
                        help="scaffold yaw trim coef applied as -k * |cmd[1]|. "
                             "Tuned: -0.012.")
    args = parser.parse_args()

    print("=" * 70)
    print("JAX-MJX behavioral cloning")
    print(f"  model:       {MODEL_PATH}")
    print(f"  demos:       {args.n_envs} envs × {args.n_steps} steps")
    print(f"  epochs:      {args.n_epochs}")
    print(f"  batch_size:  {args.batch_size}")
    print(f"  out_dir:     {args.out_dir}")
    print(f"  jax devices: {jax.devices()}")
    print("=" * 70)

    print("\n1. building scaffold-driven env (gait_scale=1.0)...")
    env = HexapodBraxEnv(MODEL_PATH, gait_scale=DEMO_GAIT_SCALE,
                         cmd_mask=args.cmd_mask,
                         action_space=args.action_space,
                         scaffold_wz_trim_vx=args.wz_trim_vx,
                         scaffold_wz_trim_vy_abs=args.wz_trim_vy_abs)
    print(f"   obs={env.observation_size}, act={env.action_size}")
    print(f"   cmd_mask: {args.cmd_mask}, action_space: {args.action_space}")
    print(f"   trim_vx={args.wz_trim_vx}, trim_vy_abs={args.wz_trim_vy_abs}")

    rng = jax.random.PRNGKey(SEED)
    rng, demo_rng = jax.random.split(rng)

    print("\n2. collecting demos...")
    obs_all, bc_all = _collect_demos(env, args.n_envs, args.n_steps, demo_rng)
    print(f"   shapes: obs={obs_all.shape}, bc={bc_all.shape}")

    print("\n3. building networks + training BC...")
    # Use Brax's running_statistics.normalize to match train_jax.py's
    # normalize_observations=True path.
    networks = _make_networks(env.observation_size, env.action_size,
                              normalize=running_statistics.normalize)
    rng, train_rng = jax.random.split(rng)
    norm_p, pol_p, val_p = _train_bc(
        networks, obs_all, bc_all, args.n_epochs, args.batch_size,
        LEARNING_RATE, train_rng)

    print("\n4. saving (normalizer, policy, value) tuple...")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "params.pkl"
    with open(out_path, "wb") as f:
        pickle.dump((norm_p, pol_p, val_p), f)
    print(f"   wrote {out_path}")
    print(f"\nUse via:  python train_jax.py --restore {out_path}")


if __name__ == "__main__":
    main()
