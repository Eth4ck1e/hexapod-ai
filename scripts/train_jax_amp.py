"""
train_jax_amp.py — AMP-augmented PPO training.

Outer loop:
  1. Build HexapodAMPEnv with current discriminator params (frozen)
  2. Run a PPO segment (N env steps) — env.step augments task reward with
     style reward from D
  3. Roll out the trained policy for a fixed number of steps with the
     env's reward off, capture the (s_t, s_{t+1}) transitions
  4. Update discriminator on (prior batch, policy batch) for K steps
  5. Goto 1 until total budget exhausted

This is the alternating variant of AMP — D is updated between PPO
segments, not every PPO iteration. Less faithful to the paper but
much simpler to implement on top of stock Brax PPO.

Run inside WSL2:
    PYTHONPATH=. ~/.venv-mjx/bin/python scripts/train_jax_amp.py \\
        --restore checkpoints/<bc_lineage_iter0_bc>/final/params.pkl \\
        --priors checkpoints/amp_priors.npz \\
        --segments 5 \\
        --steps-per-segment 50000000

--- NaN diagnosis (2026-05-06) ---
Training rows in the log showed reward=+nan / style=+nan throughout, while
eval rows were finite and improving. This was a pure metrics key-name mismatch,
NOT a numerical NaN in the actual rewards.

Brax's progress_fn is called from two sources:
  A) Brax's EpisodeMetricsLogger — called every `training_metrics_steps`
     steps with keys like "episode/sum_reward" and "episode/<metric_name>"
     (EpisodeWrapper accumulates episode_metrics['sum_reward'] and each
     env.step metrics key; EpisodeMetricsLogger prefixes them "episode/").
  B) The eval path — called with "eval/episode_reward", "eval/episode_<name>".

_progress was looking for "training/episode_reward" and "training/amp_style_reward"
— keys that Brax never emits. The .get() fallback returned float("nan") every
time. Fix: use "episode/sum_reward" for training reward and
"episode/amp_style_reward" for training style. Both sources are now handled by
a single key-probe chain with correct names.
"""
from __future__ import annotations

import argparse
import functools
import pickle
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorboardX import SummaryWriter

# Enable persistent JAX compile cache before any JAX op runs.
from chain_train import (
    enable_jax_cache,
    MODEL_PATH, BASE_NAME, CMD_MASK, ACTION_SPACE,
    NUM_ENVS, DISC_BATCH,
)
enable_jax_cache()

# JAX 0.10 / Brax 0.14.2 compat shim — same as train_jax.py.
if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(x, devices):
        n = len(devices)
        def replicate(v):
            a = jnp.asarray(v)
            return jnp.broadcast_to(a, (n,) + a.shape).copy()
        return jax.tree.map(replicate, x)
    jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.ppo.networks import make_ppo_networks

from envs.hexapod_amp_env import HexapodAMPEnv, _extract_amp_state
from envs import hexapod_env_jax as hex_jax
from envs.cmd_bins import cmd_to_bin, N_BINS
from amp.discriminator import (
    Discriminator, discriminator_loss, STATE_DIM, TRANSITION_DIM,
    CMD_DIM_FOR_DISC, cmd_for_disc,
    MultiHeadDiscriminator, multihead_discriminator_loss,
)

# v11+: paper-matching Actor (Chen et al. uses 256, 128, 64 for the
# low-level MLP). Must match pretrain_bc_jax.py's POLICY_HIDDEN_LAYERS
# so warm-starts shape-match.
POLICY_HIDDEN_LAYERS = (256, 128, 64)


def _custom_network_factory(observation_size, action_size, preprocess_observations_fn):
    """Wraps make_ppo_networks with our custom policy hidden layers.
    Brax PPO's train() accepts this as `network_factory`."""
    return make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=POLICY_HIDDEN_LAYERS,
    )


# ============================================================================
# CONFIG
# ============================================================================
RUN_BASE       = f"{BASE_NAME}_amp"
# NUM_ENVS imported from chain_train (SoT). v26 restored to 4096 after
# reverting obs to 78-dim removed the v24 VRAM pressure.
EPISODE_LENGTH = 1000
# v18 (2026-05-10): dropped from 50M to 5M (and the default --segments to 20)
# so a full run is ~100M steps / ~15 min. Iteration speed matters more than
# squeezing the last few % from each run while we debug AMP+PPO dynamics.
STEPS_PER_SEG  = 5_000_000   # PPO env-steps per outer iteration
LEARNING_RATE  = 3e-4
ENTROPY_COST   = 1e-3
DISCOUNT       = 0.97
UNROLL_LENGTH  = 20
BATCH_SIZE     = 512
NUM_MINIBATCHES = 8
NUM_UPDATES_PER_BATCH = 4

STYLE_WEIGHT   = 0.5          # λ_style — weight on AMP reward in env

# Discriminator update params — v19 (2026-05-10): halved DISC_UPDATES to
# trade ~5 sec/segment for slightly weaker disc per cycle. Net win when
# segments are short and overhead dominates.
DISC_LR        = 1e-4
# DISC_BATCH imported from chain_train (SoT). v26 restored to 1024.
DISC_UPDATES   = 100          # was 200
DISC_GRAD_PEN  = 10.0
DISC_HIDDEN    = (1024, 512)

# Policy-rollout collection (for discriminator's "fake" batch each iter).
# v19: dropped 1024×200=204800 to 512×100=51200 — still 50× the 1024 disc
# batch size, so plenty of variety per disc step. Saves ~10s/segment.
POLICY_ROLLOUT_ENVS  = 512    # was 1024
POLICY_ROLLOUT_STEPS = 100    # was 200

# v19: 5 -> 2. v26+ (2026-05-15): 2 -> 1. Eval inside a segment costs
# ~15 s. At our 50M-step segments the within-segment learning curve is
# mostly noise — we only care about end-of-segment quality, which the
# orchestrator's separate EVAL_AFTER_SEGMENT pass captures with full
# per-iter granularity for the watcher's BEST-iter detection.
NUM_EVALS      = 1            # was 2 (was 5)

SEED = 0


# ============================================================================
# Progress tracker
# ============================================================================
_PROG = {"last_step": 0, "last_t": None, "tb": None, "outer_steps": 0}


# ============================================================================
# Async checkpoint save (v19+) — pickle.dump runs in a background thread so
# the main training loop doesn't block on I/O. Each save overlaps with the
# next segment's setup. Net win is small (~5s/run) but free.
# ============================================================================
def _async_pickle_save(obj, path: Path) -> threading.Thread:
    """Spawn a daemon thread that pickle-dumps `obj` to `path`. Returns the
    thread so callers can join() it before the process exits if needed."""
    def _save():
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    t = threading.Thread(target=_save, daemon=True)
    t.start()
    return t


def _progress(num_steps: int, metrics: dict, t0: float):
    now = time.perf_counter()
    if _PROG["last_t"] is None:
        inst = "       --"
    else:
        d_step = num_steps - _PROG["last_step"]
        d_t = max(now - _PROG["last_t"], 1e-6)
        inst = f"{d_step / d_t:>10,.0f}"
    _PROG["last_step"] = num_steps
    _PROG["last_t"] = now
    # Eval path: "eval/episode_reward", "eval/episode_amp_style_reward"
    # Training path (EpisodeMetricsLogger): "episode/sum_reward",
    #   "episode/amp_style_reward" (accumulated per-episode totals, not per-step)
    rew = metrics.get("eval/episode_reward",
                      metrics.get("episode/sum_reward", float("nan")))
    style = metrics.get("eval/episode_amp_style_reward",
                        metrics.get("episode/amp_style_reward", float("nan")))
    tag = "EVAL " if "eval/episode_reward" in metrics else "train"
    # Warn once if training metrics are still missing (guards against Brax API
    # changes silently poisoning future runs).
    if tag == "train" and rew != rew:   # NaN check
        import warnings
        warnings.warn(
            "train_jax_amp: training reward is NaN after key fix — "
            f"available keys: {sorted(metrics.keys())}",
            stacklevel=2,
        )
    elapsed = (now - t0) / 60
    total = _PROG["outer_steps"] + num_steps
    print(f"  [{tag}] step={total:>11,d}  reward={float(rew):+8.3f}  "
          f"style={float(style):+6.3f}  it/s={inst}  elapsed={elapsed:5.1f} min")
    if _PROG["tb"] is not None:
        for k, v in metrics.items():
            try:
                _PROG["tb"].add_scalar(k, float(v), total)
            except (TypeError, ValueError):
                continue
        _PROG["tb"].flush()


# ============================================================================
# Partition-disc training (v23+) — strict within-bin prior sampling
# ============================================================================
def _train_multihead_discriminator(disc_module, disc_params, opt_state,
                                   optimizer, prior_data, prior_bins,
                                   policy_data, policy_bins,
                                   priors_by_bin,
                                   n_updates: int, batch_size: int, rng):
    """Update the multi-head disc for n_updates steps.

    For each step:
      1. Sample policy batch uniformly (any cmds)
      2. For each policy transition, look up its bin and sample 1 prior
         from THAT bin's prior pool (strict within-bin matching)
      3. Compute multihead_discriminator_loss; backbone gets gradient from
         all samples, each head gets gradient only from its-bin samples
    """
    @jax.jit
    def step(params, opt_state, prior_b, prior_b_bins,
             policy_b, policy_b_bins):
        (loss, aux), grads = jax.value_and_grad(
            multihead_discriminator_loss, has_aux=True)(
            params, disc_module, prior_b, prior_b_bins,
            policy_b, policy_b_bins, DISC_GRAD_PEN)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    n_policy = policy_data.shape[0]
    losses = []
    last_aux = None
    for s in range(n_updates):
        rng, k_pol = jax.random.split(rng)
        # Sample policy batch
        idx_q = jax.random.choice(k_pol, n_policy, (batch_size,), replace=False)
        idx_q_np = np.asarray(idx_q)
        policy_b      = policy_data[idx_q]
        policy_b_bins = policy_bins[idx_q]
        # Strict within-bin prior sampling: for each policy sample's bin,
        # randomly pick one prior index from that bin's pool.
        policy_b_bins_np = np.asarray(policy_b_bins)
        prior_idx_np = np.empty(batch_size, dtype=np.int64)
        for i, b in enumerate(policy_b_bins_np):
            pool = priors_by_bin[int(b)]
            prior_idx_np[i] = pool[np.random.randint(len(pool))]
        prior_idx = jnp.asarray(prior_idx_np)
        prior_b      = prior_data[prior_idx]
        prior_b_bins = prior_bins[prior_idx]

        disc_params, opt_state, loss, aux = step(
            disc_params, opt_state, prior_b, prior_b_bins,
            policy_b, policy_b_bins)
        losses.append(float(loss))
        last_aux = aux
    return disc_params, opt_state, losses, last_aux


# ============================================================================
# Discriminator updates between PPO segments
# ============================================================================
def _train_discriminator(disc_module, disc_params, opt_state,
                         optimizer, prior_data, policy_data,
                         n_updates: int, batch_size: int, rng,
                         knn_kdtree=None, knn_scale=None, knn_k: int = 10):
    """Update discriminator for n_updates gradient steps.

    v18+ (2026-05-10) — cmd-matched prior sampling. Instead of sampling
    priors uniformly at random (which mixes ALL cmds together), we:
      1. Sample policy transitions uniformly → policy_b
      2. Extract their cmds (last CMD_DIM_FOR_DISC slots of each transition)
      3. K-NN query: for each policy cmd, find K priors with closest cmds
      4. Sample one of those K → matched prior batch

    The disc only ever sees (policy, prior) pairs at similar cmds. Fixes
    the failure mode where disc rewards turning-under-straight-cmd because
    the prior set contains turning motions at OTHER cmds.

    knn_kdtree: scipy.spatial.cKDTree built on per-dim-normalized prior cmds.
    knn_scale:  (CMD_DIM_FOR_DISC,) array, each prior cmd dim's half-range
                used for normalization (so all dims contribute equally).
    knn_k:      neighborhood size. K=10 default; smaller is stricter (purer
                cmd match), larger is looser (more variety per region).

    When knn_kdtree is None, falls back to the original uniform sampling.
    """

    @jax.jit
    def step(params, opt_state, prior_b, policy_b):
        (loss, aux), grads = jax.value_and_grad(discriminator_loss,
                                                 has_aux=True)(
            params, disc_module, prior_b, policy_b,
            grad_penalty_w=DISC_GRAD_PEN)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    n_prior = prior_data.shape[0]
    n_policy = policy_data.shape[0]
    cmd_start = 2 * STATE_DIM
    cmd_end   = cmd_start + CMD_DIM_FOR_DISC

    losses = []
    last_aux = None
    for s in range(n_updates):
        rng, k1, k2 = jax.random.split(rng, 3)
        idx_q = jax.random.choice(k2, n_policy, (batch_size,), replace=False)
        policy_b = policy_data[idx_q]

        if knn_kdtree is None:
            # Original uniform sampling.
            idx_p = jax.random.choice(k1, n_prior, (batch_size,), replace=False)
            prior_b = prior_data[idx_p]
        else:
            # Cmd-matched K-NN sampling.
            policy_cmds = np.asarray(policy_b[:, cmd_start:cmd_end])
            policy_cmds_norm = policy_cmds / knn_scale
            # KDTree query returns (B, K) indices, distance ignored.
            _, knn_idx = knn_kdtree.query(policy_cmds_norm, k=knn_k)
            if knn_k == 1:
                # query() returns 1-D array when k=1; reshape to (B, 1).
                knn_idx = knn_idx.reshape(-1, 1)
            # Random pick one of K per row.
            sub = np.random.randint(0, knn_k, size=batch_size)
            prior_idx_np = knn_idx[np.arange(batch_size), sub]
            prior_b = prior_data[jnp.asarray(prior_idx_np)]

        disc_params, opt_state, loss, aux = step(
            disc_params, opt_state, prior_b, policy_b)
        losses.append(float(loss))
        last_aux = aux
    return disc_params, opt_state, losses, last_aux


# ============================================================================
# Policy rollout collection (for next discriminator batch)
# ============================================================================
def _collect_policy_transitions(env: HexapodAMPEnv, make_inference_fn,
                                params, n_envs: int, n_steps: int, rng):
    """Roll out current policy under env.step (with AMP reward injected)
    and capture (s_t, s_{t+1}) AMP transitions. The transitions reflect
    what the policy actually visits — these are the "fake" examples the
    discriminator learns to distinguish from prior data."""
    inference_fn = make_inference_fn(params, deterministic=True)
    rngs = jax.random.split(rng, n_envs)

    @jax.jit
    def vreset(rngs):
        return jax.vmap(env.reset)(rngs)

    state = vreset(rngs)
    transitions = []

    @jax.jit
    def vstep(state, _):
        # Use deterministic policy actions
        action_keys = jax.random.split(jax.random.PRNGKey(0), state.obs.shape[0])
        actions, _ = jax.vmap(inference_fn)(state.obs, action_keys)
        # Capture pre-step cmd so the (s_t, s_{t+1}, cmd_t) transition
        # uses the cmd active at s_t — same convention as prior_data.py.
        cmd_pre = state.pipeline_state.cmd                         # (n_envs, 9)
        new_state = jax.vmap(env.step)(state, actions)
        transition = jnp.concatenate([
            state.info["prev_amp_state"],
            new_state.info["prev_amp_state"],   # post-step: this IS the new amp state
            jax.vmap(cmd_for_disc)(cmd_pre),
        ], axis=-1)
        # Also emit the bin index (v23+) for partition-disc training.
        bin_per_env = jax.vmap(cmd_to_bin)(cmd_pre)                # (n_envs,)
        return new_state, (transition, bin_per_env)

    state, (all_trans, all_bins) = jax.lax.scan(vstep, state, jnp.arange(n_steps))
    all_trans.block_until_ready()
    # all_trans: (n_steps, n_envs, TRANSITION_DIM) → flatten
    return (np.asarray(all_trans).reshape(-1, TRANSITION_DIM),
            np.asarray(all_bins).reshape(-1))


# ============================================================================
# main
# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--restore",  type=str, required=True,
                   help="path to BC pretrained params.pkl (policy+normalizer+value)")
    p.add_argument("--priors",   type=str, default="checkpoints/amp_priors.npz",
                   help="path to AMP prior dataset (npz with states_t, states_t1)")
    p.add_argument("--segments", type=int, default=5,
                   help="number of outer iterations (PPO+disc cycles)")
    p.add_argument("--steps-per-segment", type=int, default=STEPS_PER_SEG)
    p.add_argument("--num-envs", type=int, default=NUM_ENVS)
    p.add_argument("--cmd-mask", type=str, default=CMD_MASK,
                   choices=["stage1", "stage2", "stage3", "paper", "paper_stance"])
    p.add_argument("--action-space", type=str, default=ACTION_SPACE,
                   choices=["joint", "foot"])
    p.add_argument("--style-weight", type=float, default=STYLE_WEIGHT)
    p.add_argument("--run", type=str, default=RUN_BASE)
    p.add_argument("--restore-discriminator", type=str, default=None,
                   help="optional path to a previously-saved discriminator.pkl. "
                        "When set, the discriminator warm-starts from that "
                        "checkpoint instead of random init. Use to continue an "
                        "AMP lineage where both policy AND discriminator should "
                        "carry over (e.g., extending a finished run for more "
                        "segments).")
    p.add_argument("--cmd-dynamics", action="store_true",
                   help="enable in-episode cmd transitions (v9+). New cmd "
                        "targets are sampled every 3-8 sec sim time and the "
                        "cmd interpolates over 0.5-1 sec ramps. Trains the "
                        "policy to handle live controller inputs smoothly.")
    p.add_argument("--wz-trim-vx", type=float, default=0.0,
                   help="scaffold yaw trim coef applied as -k * cmd[0] before "
                        "scaffold call. Cancels physics-induced drift. Tuned: 0.005.")
    p.add_argument("--wz-trim-vy-abs", type=float, default=0.0,
                   help="scaffold yaw trim coef applied as -k * |cmd[1]|. Tuned: -0.012.")
    p.add_argument("--knn-k", type=int, default=10,
                   help="cmd-matched prior sampling: K nearest priors per "
                        "policy cmd. Smaller=stricter cmd match (less variety "
                        "per region). Larger=looser (risk of including wrong-cmd "
                        "priors). Cross-test on our dataset: K<=20 has 0%% leak "
                        "between 'straight' and 'barely-turning' neighborhoods. "
                        "Set to 0 to disable K-NN matching (uniform sampling).")
    p.add_argument("--recovery-curriculum", action="store_true",
                   help="enable random push impulses on the base body "
                        "(v22+). Force magnitude follows a curriculum: starts "
                        "at --recovery-start-scale on segment 1, ramps to "
                        "--recovery-end-scale at --recovery-full-segment, "
                        "stays at end-scale after.")
    p.add_argument("--recovery-start-scale", type=float, default=0.2,
                   help="magnitude scale at segment 1 (0..1)")
    p.add_argument("--recovery-end-scale",   type=float, default=1.0,
                   help="magnitude scale at full curriculum")
    p.add_argument("--recovery-full-segment", type=int, default=10,
                   help="segment at which magnitude reaches end-scale")
    p.add_argument("--partition-disc", action="store_true",
                   help="enable v23+ partition discriminator (150 bins, "
                        "6 motion x 5 height x 5 width). Requires priors npz "
                        "with bin_idx_t column. Replaces K-NN matching with "
                        "strict within-bin prior sampling.")
    p.add_argument("--profile", action="store_true",
                   help="wrap the training segment loop with jax.profiler. "
                        "Writes a TensorBoard trace to logs/profile/<run>/. "
                        "Adds ~10%% overhead, so run a short profile (1-2 "
                        "segments) and don't compare throughput numbers to "
                        "non-profiled runs. View via "
                        "`tensorboard --logdir logs/profile/<run>` and open "
                        "the Profile tab.")
    args = p.parse_args()

    print("=" * 70)
    print(f"AMP-PPO training: {args.run}")
    print(f"  segments={args.segments}  steps/seg={args.steps_per_segment:,d}  "
          f"style_w={args.style_weight}")
    print(f"  cmd_mask={args.cmd_mask}  action_space={args.action_space}")
    print(f"  prior data: {args.priors}")
    print(f"  bc init:    {args.restore}")
    print("=" * 70)

    # Load AMP prior data — cmd-conditional format requires `cmds_t` array
    # alongside `states_t` / `states_t1`. The discriminator pairs each
    # (s_t, s_t+1) transition with cmd_t (active at s_t).
    print("\n  loading prior data...")
    prior_npz = np.load(args.priors)
    if "cmds_t" not in prior_npz.files:
        raise RuntimeError(
            f"{args.priors} is the OLD prior format (no 'cmds_t' key). "
            f"Regenerate with the cmd-conditional `amp/prior_data.py` first.")
    prior_cmds_for_disc = prior_npz["cmds_t"][..., :CMD_DIM_FOR_DISC]
    prior_transitions = jnp.asarray(np.concatenate(
        [prior_npz["states_t"], prior_npz["states_t1"], prior_cmds_for_disc],
        axis=-1,
    ).astype(np.float32))
    print(f"  loaded {prior_transitions.shape[0]:,} prior transitions  "
          f"(state-pair + {CMD_DIM_FOR_DISC} cmd dims)")

    # Build KDTree for cmd-matched prior sampling (v18+, K-NN path).
    knn_kdtree = None
    knn_scale  = None
    if args.knn_k > 0 and not args.partition_disc:
        from scipy.spatial import cKDTree
        prior_cmds_np = prior_cmds_for_disc.astype(np.float32)
        knn_scale = (prior_cmds_np.max(0) - prior_cmds_np.min(0)) / 2.0
        knn_scale = np.where(knn_scale < 1e-6, 1.0, knn_scale).astype(np.float32)
        prior_cmds_norm = prior_cmds_np / knn_scale
        print(f"  building cmd KDTree (K={args.knn_k}) over normalized cmds...")
        t_kd = time.perf_counter()
        knn_kdtree = cKDTree(prior_cmds_norm)
        print(f"  KDTree built in {time.perf_counter()-t_kd:.1f}s.")

    # Partition-disc path (v23+) — build per-bin index pool.
    prior_bins = None
    priors_by_bin = None
    if args.partition_disc:
        if "bin_idx_t" not in prior_npz.files:
            raise RuntimeError(
                f"{args.priors} has no 'bin_idx_t' column. v23+ partition "
                f"disc requires priors generated with cmd_bins-aware "
                f"prior_data.py. Regenerate the priors.")
        prior_bins_np = prior_npz["bin_idx_t"].astype(np.int32)
        prior_bins = jnp.asarray(prior_bins_np)
        # Build per-bin index pool: priors_by_bin[b] is a numpy array of
        # prior indices whose bin == b. Used by _train_multihead_discriminator
        # to sample strictly within-bin.
        priors_by_bin = [np.where(prior_bins_np == b)[0]
                         for b in range(N_BINS)]
        empty = sum(1 for p in priors_by_bin if len(p) == 0)
        print(f"  partition-disc: {N_BINS} bins, {N_BINS - empty} populated, "
              f"min={min(len(p) for p in priors_by_bin if len(p)>0):,}, "
              f"median={int(np.median([len(p) for p in priors_by_bin if len(p)>0])):,}, "
              f"max={max(len(p) for p in priors_by_bin):,}")

    # Initialize discriminator (random or warm-start from checkpoint)
    print("\n  initializing discriminator...")
    rng = jax.random.PRNGKey(SEED)
    rng, init_k = jax.random.split(rng)
    if args.partition_disc:
        disc_module = MultiHeadDiscriminator(n_bins=N_BINS, hidden_sizes=DISC_HIDDEN)
    else:
        disc_module = Discriminator(hidden_sizes=DISC_HIDDEN)
    if args.restore_discriminator:
        print(f"  warm-start from {args.restore_discriminator}")
        with open(args.restore_discriminator, "rb") as f:
            disc_params = pickle.load(f)
    else:
        if args.partition_disc:
            disc_params = disc_module.init(
                init_k,
                jnp.zeros((1, TRANSITION_DIM), jnp.float32),
                jnp.zeros((1,), jnp.int32))
        else:
            disc_params = disc_module.init(init_k,
                                           jnp.zeros((1, TRANSITION_DIM), jnp.float32))
    disc_optimizer = optax.adam(DISC_LR)
    disc_opt_state = disc_optimizer.init(disc_params)

    # Load BC params (will be threaded through PPO via restore_params)
    print(f"\n  loading BC params from {args.restore}...")
    with open(args.restore, "rb") as f:
        ppo_params = pickle.load(f)
    print(f"  loaded {type(ppo_params).__name__}: "
          f"{[type(x).__name__ for x in ppo_params] if isinstance(ppo_params, tuple) else 'opaque'}")

    # TensorBoard writer
    log_dir = Path("logs") / args.run / "amp"
    log_dir.mkdir(parents=True, exist_ok=True)
    _PROG["tb"] = SummaryWriter(logdir=str(log_dir))
    print(f"  tensorboard: {log_dir}")

    t0 = time.perf_counter()

    # Optional profiler — wraps the segment loop. View via TensorBoard
    # Profile tab. Adds ~10% overhead so don't compare throughput numbers
    # to non-profiled runs. (Don't re-import jax inside main() — Python
    # would treat it as a local variable and shadow the module-level
    # import for the entire function scope, causing UnboundLocalError
    # at earlier `jax.*` calls.)
    profile_dir = None
    if args.profile:
        profile_dir = Path("logs/profile") / args.run
        profile_dir.mkdir(parents=True, exist_ok=True)
        print(f"  profiling -> {profile_dir}")
        jax.profiler.start_trace(str(profile_dir))

    for seg in range(args.segments):
        print(f"\n{'='*70}")
        print(f"OUTER SEGMENT {seg+1}/{args.segments}")
        print(f"{'='*70}")

        # Recovery-curriculum magnitude scale for this segment.
        # Linear ramp from start_scale (seg 1) to end_scale (at full_segment),
        # then constant at end_scale.
        if args.recovery_curriculum:
            seg_1based = seg + 1
            ramp_frac = min(1.0, (seg_1based - 1) / max(1, args.recovery_full_segment - 1))
            disturbance_scale = (args.recovery_start_scale +
                                  ramp_frac * (args.recovery_end_scale - args.recovery_start_scale))
            print(f"  recovery curriculum: seg {seg_1based}/{args.segments}  "
                  f"magnitude_scale={disturbance_scale:.3f}")
        else:
            disturbance_scale = 1.0   # ignored when disturbance_enabled=False

        # 1. Build env with current discriminator params
        print("  building HexapodAMPEnv with current discriminator...")
        env = HexapodAMPEnv(
            MODEL_PATH,
            discriminator_params=disc_params,
            gait_scale=0.0,
            cmd_mask=args.cmd_mask,
            action_space=args.action_space,
            style_weight=args.style_weight,
            cmd_dynamics_enabled=args.cmd_dynamics,
            scaffold_wz_trim_vx=args.wz_trim_vx,
            scaffold_wz_trim_vy_abs=args.wz_trim_vy_abs,
            disturbance_enabled=args.recovery_curriculum,
            disturbance_magnitude_scale=disturbance_scale,
            discriminator_hidden=DISC_HIDDEN,
            multihead_disc=args.partition_disc,
        )

        # 2. Run PPO segment
        print("  running PPO segment...")
        _PROG["last_step"] = 0
        _PROG["last_t"] = None
        seg_t0 = time.perf_counter()
        train_fn = functools.partial(
            ppo_train.train,
            environment        = env,
            num_timesteps      = args.steps_per_segment,
            num_envs           = args.num_envs,
            episode_length     = EPISODE_LENGTH,
            learning_rate      = LEARNING_RATE,
            entropy_cost       = ENTROPY_COST,
            discounting        = DISCOUNT,
            unroll_length      = UNROLL_LENGTH,
            network_factory    = _custom_network_factory,
            # Brax PPO requires batch_size * num_minibatches % num_envs == 0.
            # Default constants assume num_envs=4096 (512*8=4096). For other
            # num_envs (e.g., 8192 from the calibration sweep), scale batch_size
            # so the relationship `batch_size * num_minibatches == num_envs`
            # holds — matches per-env-per-minibatch sample count regardless of
            # num_envs.
            batch_size         = max(1, args.num_envs // NUM_MINIBATCHES),
            num_minibatches    = NUM_MINIBATCHES,
            num_updates_per_batch = NUM_UPDATES_PER_BATCH,
            num_evals          = NUM_EVALS,
            seed               = SEED + seg,
            normalize_observations = True,
            restore_params     = ppo_params,
            log_training_metrics   = True,
            training_metrics_steps = 100_000,
            progress_fn        = lambda step, m: _progress(step, m, t0),
        )
        make_inference_fn, ppo_params, _ = train_fn()
        _PROG["outer_steps"] += args.steps_per_segment
        print(f"  PPO segment done in {(time.perf_counter()-seg_t0)/60:.1f} min")

        # 3. Save policy checkpoint per segment (async — non-blocking).
        ckpt_dir = Path("checkpoints") / args.run / f"iter{seg+1}" / "final"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _async_pickle_save(ppo_params, ckpt_dir / "params.pkl")
        print(f"  saving policy (bg): {ckpt_dir/'params.pkl'}")

        # 4. Collect policy transitions (+ bins for v23+ partition disc)
        print(f"  collecting {POLICY_ROLLOUT_ENVS}×{POLICY_ROLLOUT_STEPS} policy transitions...")
        rng, rk = jax.random.split(rng)
        policy_transitions, policy_trans_bins = _collect_policy_transitions(
            env, make_inference_fn, ppo_params,
            POLICY_ROLLOUT_ENVS, POLICY_ROLLOUT_STEPS, rk)
        print(f"  collected: {policy_transitions.shape[0]:,}")

        # 5. Update discriminator — partition-disc or K-NN path.
        print(f"  updating discriminator for {DISC_UPDATES} steps...")
        rng, rk = jax.random.split(rng)
        if args.partition_disc:
            disc_params, disc_opt_state, losses, aux = _train_multihead_discriminator(
                disc_module, disc_params, disc_opt_state, disc_optimizer,
                prior_transitions, prior_bins,
                jnp.asarray(policy_transitions), jnp.asarray(policy_trans_bins),
                priors_by_bin,
                DISC_UPDATES, DISC_BATCH, rk)
        else:
            disc_params, disc_opt_state, losses, aux = _train_discriminator(
                disc_module, disc_params, disc_opt_state, disc_optimizer,
                prior_transitions, jnp.asarray(policy_transitions),
                DISC_UPDATES, DISC_BATCH, rk,
                knn_kdtree=knn_kdtree, knn_scale=knn_scale, knn_k=args.knn_k)
        print(f"  disc loss: {losses[0]:.3f} → {losses[-1]:.3f}  "
              f"d_prior={float(aux['d_prior_mean']):+.3f}  "
              f"d_policy={float(aux['d_policy_mean']):+.3f}")
        if _PROG["tb"] is not None:
            _PROG["tb"].add_scalar("disc/loss_final", losses[-1], _PROG["outer_steps"])
            _PROG["tb"].add_scalar("disc/d_prior_mean",
                                   float(aux["d_prior_mean"]), _PROG["outer_steps"])
            _PROG["tb"].add_scalar("disc/d_policy_mean",
                                   float(aux["d_policy_mean"]), _PROG["outer_steps"])
            _PROG["tb"].flush()

        # 6. Save discriminator checkpoint (async).
        _disc_save_thread = _async_pickle_save(
            disc_params, ckpt_dir / "discriminator.pkl")

    # Final flush — wait for last segment's disc save to complete before exit.
    try:
        _disc_save_thread.join(timeout=30)
    except NameError:
        pass   # no segments ran

    elapsed = (time.perf_counter() - t0) / 60
    print(f"\n{'='*70}")
    print(f"AMP TRAINING DONE in {elapsed:.1f} min ({args.segments} segments)")
    print(f"{'='*70}")

    if args.profile and profile_dir is not None:
        jax.profiler.stop_trace()
        print(f"\nprofile saved to {profile_dir}")
        print(f"view: .venv\\Scripts\\python.exe -m tensorboard.main --logdir {profile_dir} --port 6007")
        print("then open the Profile tab in TensorBoard.")


if __name__ == "__main__":
    main()
