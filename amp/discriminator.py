"""
amp/discriminator.py — AMP discriminator network + training pieces.

What's here:
  * `Discriminator` — flax linen module. 2-layer MLP, default [1024, 512]
    hidden units, scalar linear output. Input shape: (..., 2 * STATE_DIM)
    (concatenated (s_t, s_{t+1})).
  * `discriminator_loss` — paper eq. (1):
      L = E_prior[(D(T) - 1)²] + E_policy[(D(T) + 1)²]
        + α_gp/2 * E_prior[||∇_input D(T)||²]
    Trains D to push prior transitions toward +1 and policy transitions
    toward -1. Gradient penalty stabilizes adversarial training.
  * `style_reward` — paper eq. (2):
      r_style = max(0, 1 - 0.25 * (D(T) - 1)²)
    Bounded in [0, 1]. The PPO loss adds λ_style * r_style to its reward.

Use this module from train_jax_amp.py (built next): instantiate, init
params, alternate one PPO update with one discriminator update each
training iteration.

Self-test (`python -m amp.discriminator` from project root) trains the
discriminator to distinguish prior data from white noise — confirms
the network architecture and loss compute correctly before integrating
with PPO.
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn


STATE_DIM       = 49     # must match amp/prior_data.py per-step state vector
# Cmd-conditional discriminator (v15+): the discriminator additionally
# receives the active cmd at the time of the transition so it can learn
# "is this motion natural FOR THIS CMD?" — not just "is this motion
# natural in some abstract sense?" This fixes the failure mode where
# straight-cmd-with-turning-motion gets a high style score because
# turning motion exists somewhere in the prior dataset.
#
# v17 (2026-05-10): bumped from 3 → 9. Reasoning: the scaffold's motion
# depends on ALL 9 cmd dims (pitch tilts the body, dh changes height, sx
# shifts COG, etc.), so two prior transitions with the same (vx, vy, wz)
# but different posture cmds produce VERY different state transitions.
# At CMD_DIM_FOR_DISC=3 the disc had to lump all those into a single
# "what's natural under this motion cmd" mixture distribution — too
# permissive, the policy could find SOME prior variant matching almost
# any motion. With all 9 dims, disc fully conditions on what the
# scaffold was actually told to do.
CMD_DIM_FOR_DISC = 9
TRANSITION_DIM  = 2 * STATE_DIM + CMD_DIM_FOR_DISC


def cmd_for_disc(cmd: jnp.ndarray) -> jnp.ndarray:
    """Project the 9-D cmd vector down to the slots the discriminator sees.
    Currently first 3 = (vx, vy, wz). Centralizing this here so prior_data,
    env, and training code agree on which cmd dims go to disc."""
    return cmd[..., :CMD_DIM_FOR_DISC]


class Discriminator(nn.Module):
    """2-layer MLP discriminator over (s_t, s_{t+1}, cmd_t) transitions.

    Input layout: concat[s_t (49), s_{t+1} (49), cmd_motion (3)] = 101 dims.
    The cmd part lets D learn cmd-conditional naturalness — see the comment
    on TRANSITION_DIM above for why this matters."""
    hidden_sizes: Sequence[int] = (1024, 512)

    @nn.compact
    def __call__(self, transitions: jnp.ndarray) -> jnp.ndarray:
        """Returns scalar score per transition. Higher = "looks like prior data."
        No activation on the output — the loss + style reward handle scaling.

        Expects last-dim = TRANSITION_DIM (state-pair + cmd-motion)."""
        x = transitions
        for h in self.hidden_sizes:
            x = nn.Dense(h, kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2.0)))(x)
            x = nn.relu(x)
        # Final linear layer, scalar output.
        x = nn.Dense(1, kernel_init=nn.initializers.orthogonal(scale=0.01))(x)
        return x.squeeze(-1)


class MultiHeadDiscriminator(nn.Module):
    """v23+ partition-disc — shared backbone with N_BINS output heads.

    Architecture:
      - Backbone: same as Discriminator (2-layer MLP over transition).
      - Output: single Dense(n_bins) layer; per-sample score = output[bin_idx].

    During scoring with a single bin_idx per sample, this is mathematically
    equivalent to having `n_bins` separate Dense(1) heads sharing the same
    backbone, but compiles to a single matmul + gather (cheap on GPU).

    The bin index input is a non-trainable selector: it determines WHICH
    head's output is returned per sample, not what the backbone computes.
    """
    n_bins: int
    hidden_sizes: Sequence[int] = (1024, 512)

    @nn.compact
    def __call__(self, transitions: jnp.ndarray,
                 bin_idx: jnp.ndarray) -> jnp.ndarray:
        """Args:
          transitions: (B, TRANSITION_DIM)
          bin_idx:     (B,) int — selects which head's output to return per sample
        Returns: (B,) float score per sample.
        """
        x = transitions
        for h in self.hidden_sizes:
            x = nn.Dense(h, kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2.0)))(x)
            x = nn.relu(x)
        # One Dense head per bin, packed as a (B, n_bins) output.
        all_heads = nn.Dense(self.n_bins,
                             kernel_init=nn.initializers.orthogonal(scale=0.01))(x)
        # Gather per-sample head output: scores[i] = all_heads[i, bin_idx[i]].
        b = jnp.arange(transitions.shape[0])
        return all_heads[b, bin_idx]


def multihead_discriminator_loss(params,
                                 discriminator,
                                 prior_transitions: jnp.ndarray,
                                 prior_bins:        jnp.ndarray,
                                 policy_transitions: jnp.ndarray,
                                 policy_bins:       jnp.ndarray,
                                 grad_penalty_w: float = 10.0):
    """Multi-head variant of discriminator_loss.

    Each sample is routed through its bin's head. The shared backbone
    receives gradient from all samples; each head receives gradient only
    from samples in its bin. So bins with few same-bin samples per batch
    train more slowly than densely-populated bins — that's fine because
    every bin is well-represented in the prior dataset (28k+ samples).
    """
    d_prior  = discriminator.apply(params, prior_transitions, prior_bins)
    d_policy = discriminator.apply(params, policy_transitions, policy_bins)
    loss_prior  = jnp.mean((d_prior  - 1.0) ** 2)
    loss_policy = jnp.mean((d_policy + 1.0) ** 2)

    # Gradient penalty: same as the standard disc loss, just routing through
    # bin-aware apply. Computed on the prior batch (paper convention).
    def d_at(x_single, bin_single):
        # x_single: (TRANSITION_DIM,), bin_single: ()
        return discriminator.apply(params,
                                   x_single[None, :],
                                   bin_single[None]).sum()
    grad_per_input = jax.vmap(jax.grad(d_at))(prior_transitions, prior_bins)
    grad_norm_sq = jnp.sum(grad_per_input ** 2, axis=-1)
    grad_penalty = jnp.mean(grad_norm_sq)

    total = loss_prior + loss_policy + 0.5 * grad_penalty_w * grad_penalty
    aux = {
        "loss_prior":   loss_prior,
        "loss_policy":  loss_policy,
        "grad_penalty": grad_penalty,
        "d_prior_mean": jnp.mean(d_prior),
        "d_policy_mean": jnp.mean(d_policy),
    }
    return total, aux


def multihead_style_reward(params,
                           discriminator,
                           transitions: jnp.ndarray,
                           bin_idx: jnp.ndarray) -> jnp.ndarray:
    """Paper eq. (2) — same as style_reward but routes through bin's head."""
    d_score = discriminator.apply(params, transitions, bin_idx)
    return jnp.maximum(0.0, 1.0 - 0.25 * (d_score - 1.0) ** 2)


def discriminator_loss(params,
                       discriminator,
                       prior_transitions: jnp.ndarray,
                       policy_transitions: jnp.ndarray,
                       grad_penalty_w: float = 10.0):
    """Paper eq. (1) — least-squares GAN loss with gradient penalty.

    Args:
      params: discriminator params
      discriminator: a Discriminator instance (via .apply)
      prior_transitions:  (B, TRANSITION_DIM) batch sampled from prior dataset
      policy_transitions: (B, TRANSITION_DIM) batch from current policy rollouts
      grad_penalty_w: weight on the gradient penalty term (paper's α_gp)

    Returns:
      total_loss (scalar), aux dict with breakdown
    """
    # LS-GAN classification loss
    d_prior  = discriminator.apply(params, prior_transitions)        # (B,)
    d_policy = discriminator.apply(params, policy_transitions)
    loss_prior  = jnp.mean((d_prior  - 1.0) ** 2)
    loss_policy = jnp.mean((d_policy + 1.0) ** 2)

    # Gradient penalty on prior transitions (matches paper). Compute the
    # gradient of D w.r.t. INPUTS at each prior transition, sum-of-squares
    # of that gradient. Using vmap'd jax.grad keeps it fast.
    def d_at(x):
        return discriminator.apply(params, x).sum()
    grad_per_input = jax.vmap(jax.grad(d_at))(prior_transitions)     # (B, TRANSITION_DIM)
    grad_norm_sq   = jnp.sum(grad_per_input ** 2, axis=-1)           # (B,)
    grad_penalty   = jnp.mean(grad_norm_sq)

    total = loss_prior + loss_policy + 0.5 * grad_penalty_w * grad_penalty
    aux = {
        "loss_prior":   loss_prior,
        "loss_policy":  loss_policy,
        "grad_penalty": grad_penalty,
        "d_prior_mean": jnp.mean(d_prior),
        "d_policy_mean": jnp.mean(d_policy),
    }
    return total, aux


def style_reward(params,
                 discriminator,
                 transitions: jnp.ndarray) -> jnp.ndarray:
    """Paper eq. (2) — convert discriminator score into a style reward in
    [0, 1]. Higher = transition looks more like prior data."""
    d_score = discriminator.apply(params, transitions)               # (B,)
    return jnp.maximum(0.0, 1.0 - 0.25 * (d_score - 1.0) ** 2)


# ============================================================================
# Self-test — run as a script to verify the loss + network can learn to
# distinguish prior transitions from white noise.
# ============================================================================
def _self_test():
    import time
    import numpy as np
    from pathlib import Path

    PRIOR_PATH = Path("checkpoints/amp_priors_smoke.npz")
    if not PRIOR_PATH.exists():
        print(f"Need prior data at {PRIOR_PATH}.")
        print(f"Run: PYTHONPATH=. python amp/prior_data.py "
              f"--n-envs 256 --n-steps 100 --out {PRIOR_PATH}")
        return

    print(f"loading {PRIOR_PATH}")
    data = np.load(PRIOR_PATH)
    s_t  = data["states_t"].astype(np.float32)
    s_t1 = data["states_t1"].astype(np.float32)
    if "cmds_t" not in data.files:
        raise RuntimeError(
            f"{PRIOR_PATH} is the OLD prior format (no 'cmds_t' key). "
            f"Regenerate priors with the cmd-conditional `prior_data.py` first.")
    cmds = data["cmds_t"].astype(np.float32)[..., :CMD_DIM_FOR_DISC]
    prior = jnp.asarray(np.concatenate([s_t, s_t1, cmds], axis=-1))
    print(f"  prior transitions: {prior.shape}  (state-pair + cmd-motion)")

    # Synthetic "policy" data: white noise of the same shape, scaled to a
    # reasonable magnitude. The discriminator should easily learn to
    # distinguish "structured prior motion" from random noise.
    rng = jax.random.PRNGKey(0)
    rng, k = jax.random.split(rng)
    noise = jax.random.normal(k, prior.shape) * 0.5
    print(f"  noise transitions: {noise.shape}")

    disc = Discriminator()
    rng, init_rng = jax.random.split(rng)
    params = disc.init(init_rng, jnp.zeros((1, TRANSITION_DIM)))
    opt = optax.adam(1e-4)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, prior_b, policy_b, key):
        (loss, aux), grads = jax.value_and_grad(discriminator_loss,
                                                 has_aux=True)(
            params, disc, prior_b, policy_b)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    BATCH = 1024
    N_STEPS = 200
    n_prior = prior.shape[0]
    print(f"\ntraining for {N_STEPS} steps, batch={BATCH}")
    t0 = time.perf_counter()
    for s in range(N_STEPS):
        rng, k1, k2 = jax.random.split(rng, 3)
        prior_idx  = jax.random.choice(k1, n_prior, (BATCH,), replace=False)
        noise_idx  = jax.random.choice(k2, noise.shape[0], (BATCH,), replace=False)
        params, opt_state, loss, aux = step(
            params, opt_state, prior[prior_idx], noise[noise_idx], rng)
        if s % 20 == 0 or s == N_STEPS - 1:
            print(f"  step {s:>3d}  loss={float(loss):.3f}  "
                  f"d_prior={float(aux['d_prior_mean']):+.3f}  "
                  f"d_policy={float(aux['d_policy_mean']):+.3f}  "
                  f"grad_pen={float(aux['grad_penalty']):.3f}")
    elapsed = time.perf_counter() - t0
    print(f"\n  trained in {elapsed:.1f}s ({N_STEPS / elapsed:.0f} steps/s)")

    # Sanity check: discriminator should now confidently classify the two.
    final_prior  = float(jnp.mean(disc.apply(params, prior[:1024])))
    final_noise  = float(jnp.mean(disc.apply(params, noise[:1024])))
    print(f"\n  final discriminator scores:")
    print(f"    prior:  mean={final_prior:+.3f}  (target +1)")
    print(f"    noise:  mean={final_noise:+.3f}  (target -1)")
    if final_prior > 0.5 and final_noise < -0.5:
        print(f"  PASS — discriminator learned to distinguish prior from noise")
    else:
        print(f"  WARN — discrimination weak; investigate before integrating")


if __name__ == "__main__":
    _self_test()
