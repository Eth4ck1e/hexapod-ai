"""envs/cmd_bins.py — single source of truth for cmd-space partitioning.

The v23+ AMP discriminator partitions the 9-D cmd space into 150 bins
(6 motion modes × 5 height presets × 5 width thirds). Each bin gets its
own output head on the disc; priors are sampled strictly within-bin so
the disc only ever compares same-bin (s_t, s_{t+1}) pairs.

Used by:
  - amp/discriminator.py        (MultiHeadDiscriminator selects bin's head)
  - amp/prior_data.py           (precomputes bin_idx per prior)
  - envs/hexapod_amp_env.py     (selects head at style-reward time)
  - scripts/train_jax_amp.py    (per-bin disc training)

When the binning scheme changes (counts, boundaries), edit ONLY this file.

Bin layout:
  bin_idx = motion_idx * 25 + height_idx * 5 + width_idx
where motion ∈ [0,5], height ∈ [0,4], width ∈ [0,4].
"""
from __future__ import annotations

import jax.numpy as jnp
from envs.stance_envelope import HEIGHT_PRESETS, safe_dw_range

# --- Cardinalities -----------------------------------------------------
N_MOTION  = 6     # forward, backward, strafe-L, strafe-R, spin-L, spin-R
N_HEIGHT  = 5     # matches HEIGHT_PRESETS in stance_envelope.py
N_WIDTH   = 5     # fifths of safe_dw_range(dh) at the active height
N_BINS    = N_MOTION * N_HEIGHT * N_WIDTH   # = 150

MOTION_NAMES = ("forward", "backward", "strafe-L", "strafe-R",
                "spin-L",  "spin-R")

# Per-dim normalization for motion-mode classification: vx/vy ranges ~±0.30,
# wz range ~±1.62. Without normalization wz dominates the argmax.
_VX_HALF = 0.30
_VY_HALF = 0.30
_WZ_HALF = 1.62


def motion_idx(cmd: jnp.ndarray) -> jnp.ndarray:
    """Classify the cmd's (vx, vy, wz) into one of N_MOTION modes by argmax
    of the dominant normalized component. Returns int in [0, N_MOTION-1].

    Tie-breaks: vx > vy > wz; sign of dominant component picks the
    positive/negative variant. cmd=0 lands in forward (mode 0) by tiebreak —
    never sampled during training so the value is academic.
    """
    abs_vx = jnp.abs(cmd[..., 0]) / _VX_HALF
    abs_vy = jnp.abs(cmd[..., 1]) / _VY_HALF
    abs_wz = jnp.abs(cmd[..., 2]) / _WZ_HALF

    is_vx = (abs_vx >= abs_vy) & (abs_vx >= abs_wz)
    is_vy = (abs_vy >  abs_vx) & (abs_vy >= abs_wz)
    is_wz = (abs_wz >  abs_vx) & (abs_wz >  abs_vy)
    # is_wz catches the remaining "wz strictly dominant" case.

    vx_pos = cmd[..., 0] > 0
    vy_pos = cmd[..., 1] > 0
    wz_pos = cmd[..., 2] > 0

    # Encode the per-axis mode index then select.
    idx_vx = jnp.where(vx_pos, 0, 1)   # 0=forward, 1=backward
    idx_vy = jnp.where(vy_pos, 2, 3)   # 2=strafe-L, 3=strafe-R
    idx_wz = jnp.where(wz_pos, 4, 5)   # 4=spin-L, 5=spin-R

    return jnp.where(is_vx, idx_vx,
                     jnp.where(is_vy, idx_vy, idx_wz))


def height_idx(cmd: jnp.ndarray) -> jnp.ndarray:
    """Classify dh = cmd[5] into nearest of N_HEIGHT presets. Returns int."""
    dh = cmd[..., 5]
    presets = jnp.asarray(HEIGHT_PRESETS, dtype=dh.dtype)   # (N_HEIGHT,)
    return jnp.argmin(jnp.abs(dh[..., None] - presets), axis=-1)


def width_idx(cmd: jnp.ndarray) -> jnp.ndarray:
    """Classify dw = cmd[6] into one of N_WIDTH bins by dh-conditional fifth
    of the safe envelope. Returns int in [0, N_WIDTH-1]."""
    dh = cmd[..., 5]
    dw = cmd[..., 6]
    # Compute safe_dw_range at the actual dh, not the snapped preset, so
    # cmds slightly off-preset still get sensible bin boundaries.
    min_dw, max_dw = safe_dw_range(dh)
    span = max_dw - min_dw
    frac = jnp.clip((dw - min_dw) / jnp.maximum(span, 1e-9), 0.0, 0.99999)
    return (frac * N_WIDTH).astype(jnp.int32)


def cmd_to_bin(cmd: jnp.ndarray) -> jnp.ndarray:
    """Return bin index in [0, N_BINS-1] for a (..., 9)-shaped cmd."""
    m = motion_idx(cmd)
    h = height_idx(cmd)
    w = width_idx(cmd)
    return (m * (N_HEIGHT * N_WIDTH) + h * N_WIDTH + w).astype(jnp.int32)
