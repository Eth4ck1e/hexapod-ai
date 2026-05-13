"""
amp/prior_data.py — generate AMP prior dataset from scaffold rollouts.

Rolls out the analytical scaffold (gait_scale=1.0) across many vmapped
envs at varied cmds, captures per-step state vectors, and saves them as
state-transition pairs the discriminator consumes during AMP training.

State vector design (49 dims, no cmd info):
  joint_pos              18    qpos[7:25]
  joint_vel              18    qvel[6:24]
  body_linvel (body fr)   3    _body_frame_linvel(qpos, qvel)
  body_angvel             3    qvel[3:6]
  body_height             1    qpos[2]
  foot_heights (body fr)  6    z of foot_body_positions

Cmd-conditional priors (v15+): we ALSO record the active cmd at each
step (the full 9-D cmd vector). The discriminator uses the first 3
slots (vx, vy, wz) to learn cmd-conditional naturalness — see
amp/discriminator.py:CMD_DIM_FOR_DISC. Storing all 9 lets us change
which cmd dims D sees without regenerating the dataset.

Output format:
  npz with FOUR arrays:
    states_t     : (N_transitions, 49) — state at time t
    states_t1    : (N_transitions, 49) — state at time t+1
    cmds_t       : (N_transitions,  9) — cmd at time t (full vector)
    bin_idx_t    : (N_transitions,)    — precomputed partition-disc bin (v23+)

Run inside WSL2 (needs MJX/CUDA):
    PYTHONPATH=. ~/.venv-mjx/bin/python amp/prior_data.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Enable persistent JAX compilation cache before any JAX op runs.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from chain_train import enable_jax_cache, MODEL_PATH
enable_jax_cache()

from envs import hexapod_env_jax as hex_jax


STATE_DIM = 49


def extract_amp_state(mjx_data, foot_world_positions, body_pos, body_R):
    """Pull the AMP state vector (49 dims) from a single mjx_data + a few
    derived arrays. Returns a (49,) jnp array.

    Args:
      mjx_data: the env's mjx.Data after a step
      foot_world_positions: (6, 3) foot world positions
      body_pos: (3,) body world position
      body_R: (3, 3) body world rotation matrix
    """
    qpos = mjx_data.qpos
    qvel = mjx_data.qvel

    joint_pos = qpos[7:25]                                    # (18,)
    joint_vel = qvel[6:24]                                    # (18,)

    body_linvel_body = hex_jax._body_frame_linvel(qpos, qvel) # (3,)
    body_angvel      = qvel[3:6]                              # (3,)
    body_height      = qpos[2:3]                              # (1,)

    foot_body = (foot_world_positions - body_pos) @ body_R    # (6, 3)
    foot_heights = foot_body[:, 2]                            # (6,)

    return jnp.concatenate([
        joint_pos, joint_vel,
        body_linvel_body, body_angvel,
        body_height, foot_heights,
    ])


def collect_demos(model_path: str,
                  cmd_mask: str = "stage1",
                  n_envs: int = 4096,
                  n_steps: int = 1500,
                  seed: int = 0,
                  scaffold_wz_trim_vx: float = 0.0,
                  scaffold_wz_trim_vy_abs: float = 0.0):
    """Roll out the scaffold-driven env, capture AMP state vectors per
    step. Returns (states_t, states_t1) numpy arrays of shape
    (n_envs * (n_steps - 1), STATE_DIM)."""
    print(f"  building env params (cmd_mask={cmd_mask}, "
          f"trim_vx={scaffold_wz_trim_vx}, trim_vy_abs={scaffold_wz_trim_vy_abs})...")
    params = hex_jax.make_env_params(
        model_path, cmd_mask_name=cmd_mask,
        scaffold_wz_trim_vx=scaffold_wz_trim_vx,
        scaffold_wz_trim_vy_abs=scaffold_wz_trim_vy_abs,
    )

    print(f"  vmapping {n_envs} envs, {n_steps} steps...")
    rngs = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    # Reset all envs at gait_scale=1.0 (scaffold drives, policy is ignored).
    state = jax.jit(jax.vmap(
        lambda r: hex_jax.reset(params, r, gait_scale=1.0)))(rngs)

    # Action is irrelevant at gait_scale=1.0 — pass zeros.
    zero_action = jnp.zeros((n_envs, 18), dtype=jnp.float32)

    def step_and_extract(state, _):
        # Capture the cmd ACTIVE at this step (i.e., from `state`, the pre-step
        # carry) so it pairs with the (s_t, s_{t+1}) transition that the step
        # will produce. AMP's training signal is "given this cmd, does this
        # transition look like the scaffold's motion under that cmd?"
        cmd_t = state.cmd                                       # (n_envs, 9)
        new_state = jax.vmap(hex_jax.step, in_axes=(None, 0, 0))(
            params, state, zero_action)
        # Per-env: extract the AMP state vector after the step.
        def per_env_state(d):
            tibia_ids = params.tibia_body_ids
            tibia_pos = d.xpos[tibia_ids]
            tibia_R   = d.xmat[tibia_ids].reshape(6, 3, 3)
            offset    = params.foot_local_offset
            foot_world = tibia_pos + jnp.einsum('lij,j->li', tibia_R, offset)
            body_pos = d.xpos[params.base_body_id]
            body_R   = d.xmat[params.base_body_id].reshape(3, 3)
            return extract_amp_state(d, foot_world, body_pos, body_R)
        amp_state = jax.vmap(per_env_state)(new_state.mjx_data)
        return new_state, (amp_state, cmd_t)

    t0 = time.perf_counter()
    _, (all_states, all_cmds) = jax.jit(
        lambda s: jax.lax.scan(step_and_extract, s, jnp.arange(n_steps))
    )(state)
    all_states.block_until_ready()
    elapsed = time.perf_counter() - t0
    print(f"  rollout: {elapsed:.1f}s  ({n_envs * n_steps / elapsed / 1e3:.0f}k SPS)")

    # all_states: (n_steps, n_envs, 49). all_cmds: (n_steps, n_envs, 9).
    # Build (s_t, s_{t+1}, cmd_t) transitions:
    #   s_t   = states[0:T-1, :, :]
    #   s_t1  = states[1:T,   :, :]
    #   cmd_t = cmds[0:T-1,   :, :]   (cmd active when s_t was observed)
    all_states_np = np.asarray(all_states)
    all_cmds_np   = np.asarray(all_cmds)
    s_t   = all_states_np[:-1].reshape(-1, STATE_DIM)
    s_t1  = all_states_np[ 1:].reshape(-1, STATE_DIM)
    cmd_t = all_cmds_np[:-1].reshape(-1, all_cmds_np.shape[-1])
    return s_t, s_t1, cmd_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     type=str, default=MODEL_PATH)
    p.add_argument("--cmd-mask",  type=str, default="stage1",
                   choices=["stage1", "stage2", "stage3", "paper", "paper_stance"])
    p.add_argument("--n-envs",    type=int, default=4096)
    p.add_argument("--n-steps",   type=int, default=1500)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--out", type=str,
                   default="checkpoints/amp_priors.npz")
    p.add_argument("--wz-trim-vx", type=float, default=0.0,
                   help="scaffold yaw trim coef applied as -k * cmd[0] before "
                        "scaffold call. Cancels physics-induced drift in priors. "
                        "Tuned value for our scaffold: 0.005.")
    p.add_argument("--wz-trim-vy-abs", type=float, default=0.0,
                   help="scaffold yaw trim coef applied as -k * |cmd[1]| before "
                        "scaffold call. Tuned value for our scaffold: -0.012.")
    args = p.parse_args()

    print("=" * 70)
    print("AMP prior data generator")
    print(f"  model:      {args.model}")
    print(f"  cmd_mask:   {args.cmd_mask}")
    print(f"  n_envs:     {args.n_envs}")
    print(f"  n_steps:    {args.n_steps}")
    print(f"  ~demos:     {args.n_envs * (args.n_steps - 1) / 1e6:.1f}M transitions")
    print(f"  out:        {args.out}")
    print("=" * 70)

    s_t, s_t1, cmd_t = collect_demos(
        model_path=args.model,
        cmd_mask=args.cmd_mask,
        scaffold_wz_trim_vx=args.wz_trim_vx,
        scaffold_wz_trim_vy_abs=args.wz_trim_vy_abs,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        seed=args.seed,
    )

    # Precompute partition-disc bin (v23+) — one int per transition,
    # in [0, 149]. Saved alongside cmds_t so training doesn't need to
    # recompute on load.
    from envs.cmd_bins import cmd_to_bin
    bin_idx = np.asarray(cmd_to_bin(jnp.asarray(cmd_t))).astype(np.int32)
    print(f"  bin distribution: {np.bincount(bin_idx, minlength=150).min()} min, "
          f"{int(np.median(np.bincount(bin_idx, minlength=150)))} median, "
          f"{np.bincount(bin_idx, minlength=150).max()} max per bin")

    print(f"\n  saving {len(s_t):,} transitions to {args.out}...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, states_t=s_t, states_t1=s_t1,
                        cmds_t=cmd_t, bin_idx_t=bin_idx)
    sz_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  saved ({sz_mb:.1f} MB)")

    # Sanity report
    print(f"\n  state vector stats (mean, std across all transitions):")
    print(f"    joint_pos[0]   (RR coxa):     mean={s_t[:,0].mean():+.3f}  std={s_t[:,0].std():.3f}")
    print(f"    body_linvel_x:                mean={s_t[:,36].mean():+.3f}  std={s_t[:,36].std():.3f}")
    print(f"    body_angvel_z:                mean={s_t[:,41].mean():+.3f}  std={s_t[:,41].std():.3f}")
    print(f"    body_height:                  mean={s_t[:,42].mean():+.3f}  std={s_t[:,42].std():.3f}")
    print(f"    foot_height min:              mean={s_t[:,43:].mean():+.3f}  std={s_t[:,43:].std():.3f}")
    print(f"  cmd vector stats:")
    print(f"    vx (cmd[0]):                  mean={cmd_t[:,0].mean():+.3f}  std={cmd_t[:,0].std():.3f}")
    print(f"    vy (cmd[1]):                  mean={cmd_t[:,1].mean():+.3f}  std={cmd_t[:,1].std():.3f}")
    print(f"    wz (cmd[2]):                  mean={cmd_t[:,2].mean():+.3f}  std={cmd_t[:,2].std():.3f}")


if __name__ == "__main__":
    main()
