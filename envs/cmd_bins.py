"""envs/cmd_bins.py — single source of truth for cmd-space partitioning.

The v23+ AMP discriminator partitions the 9-D cmd space into 150 bins
(6 motion modes × 5 height presets × 5 width fifths). Each bin gets its
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

Robot-specific numbers (height presets, dw envelope, speed normalisation)
come in through a `BinSpec`. Every classifier defaults to the PhantomX spec
so existing callers / priors are unchanged; multi-robot callers build one
with `bin_spec_for(model_path_or_profile)` and pass it explicitly.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from envs.stance_envelope import PHANTOMX_ENVELOPE, StanceEnvelope, envelope_for

# --- Cardinalities -----------------------------------------------------
N_MOTION  = 6     # forward, backward, strafe-L, strafe-R, spin-L, spin-R
N_HEIGHT  = 5     # matches len(StanceEnvelope.height_presets)
N_WIDTH   = 5     # fifths of safe_dw_range(dh) at the active height
N_BINS    = N_MOTION * N_HEIGHT * N_WIDTH   # = 150

MOTION_NAMES = ("forward", "backward", "strafe-L", "strafe-R",
                "spin-L",  "spin-R")

# PhantomX per-dim normalization for motion-mode classification: vx/vy
# ranges ~±0.30, wz range ~±1.62. Without normalization wz dominates the
# argmax. Kept verbatim so PhantomX bins are identical to v23-v27.
_VX_HALF = 0.30
_VY_HALF = 0.30
_WZ_HALF = 1.62


class BinSpec(NamedTuple):
    """JIT-friendly bin boundaries for one robot."""
    height_presets: jnp.ndarray   # (N_HEIGHT,)
    dw_coeffs:      jnp.ndarray   # (4,) max_int, max_slope, min_int, min_slope
    motion_half:    jnp.ndarray   # (3,) vx, vy, wz normalisation half-ranges


def make_bin_spec(envelope: StanceEnvelope,
                  vx_half: float, vy_half: float, wz_half: float) -> BinSpec:
    presets = tuple(envelope.height_presets)
    if len(presets) != N_HEIGHT:
        raise ValueError(f"height_presets must have {N_HEIGHT} entries, got {len(presets)}")
    return BinSpec(
        height_presets=jnp.asarray(presets, dtype=jnp.float32),
        dw_coeffs=jnp.asarray(envelope.dw_coeffs, dtype=jnp.float32),
        motion_half=jnp.asarray([vx_half, vy_half, wz_half], dtype=jnp.float32),
    )


PHANTOMX_BIN_SPEC = make_bin_spec(PHANTOMX_ENVELOPE, _VX_HALF, _VY_HALF, _WZ_HALF)


def bin_spec_for(robot) -> BinSpec:
    """BinSpec for an MJCF path / profile name / RobotProfile. PhantomX
    returns the frozen legacy spec; other robots normalise motion by the
    scaffold's MAX_SPEED / MAX_YAW_RATE."""
    from gait.controller import Controller
    from gait.robot_profile import PHANTOMX, RobotProfile, get_profile, profile_for_model
    if isinstance(robot, RobotProfile):
        prof = robot
    elif isinstance(robot, str) and robot.endswith(".xml"):
        prof = profile_for_model(robot)
    else:
        prof = get_profile(robot)
    if prof is PHANTOMX:
        return PHANTOMX_BIN_SPEC
    ctrl = Controller(profile=prof)
    return make_bin_spec(envelope_for(prof),
                         float(ctrl.MAX_SPEED), float(ctrl.MAX_SPEED), float(ctrl.MAX_YAW_RATE))


def motion_idx(cmd: jnp.ndarray, spec: BinSpec = PHANTOMX_BIN_SPEC) -> jnp.ndarray:
    """Classify the cmd's (vx, vy, wz) into one of N_MOTION modes by argmax
    of the dominant normalized component. Returns int in [0, N_MOTION-1].

    Tie-breaks: vx > vy > wz; sign of dominant component picks the
    positive/negative variant. cmd=0 lands in forward (mode 0) by tiebreak —
    never sampled during training so the value is academic.
    """
    abs_vx = jnp.abs(cmd[..., 0]) / spec.motion_half[0]
    abs_vy = jnp.abs(cmd[..., 1]) / spec.motion_half[1]
    abs_wz = jnp.abs(cmd[..., 2]) / spec.motion_half[2]

    is_vx = (abs_vx >= abs_vy) & (abs_vx >= abs_wz)
    is_vy = (abs_vy >  abs_vx) & (abs_vy >= abs_wz)
    # remaining case: wz strictly dominant.

    vx_pos = cmd[..., 0] > 0
    vy_pos = cmd[..., 1] > 0
    wz_pos = cmd[..., 2] > 0

    idx_vx = jnp.where(vx_pos, 0, 1)   # 0=forward, 1=backward
    idx_vy = jnp.where(vy_pos, 2, 3)   # 2=strafe-L, 3=strafe-R
    idx_wz = jnp.where(wz_pos, 4, 5)   # 4=spin-L, 5=spin-R

    return jnp.where(is_vx, idx_vx,
                     jnp.where(is_vy, idx_vy, idx_wz))


def height_idx(cmd: jnp.ndarray, spec: BinSpec = PHANTOMX_BIN_SPEC) -> jnp.ndarray:
    """Classify dh = cmd[5] into nearest of N_HEIGHT presets. Returns int."""
    dh = cmd[..., 5]
    presets = spec.height_presets.astype(dh.dtype)                 # (N_HEIGHT,)
    return jnp.argmin(jnp.abs(dh[..., None] - presets), axis=-1)


def width_idx(cmd: jnp.ndarray, spec: BinSpec = PHANTOMX_BIN_SPEC) -> jnp.ndarray:
    """Classify dw = cmd[6] into one of N_WIDTH bins by dh-conditional fifth
    of the safe envelope. Returns int in [0, N_WIDTH-1]."""
    dh = cmd[..., 5]
    dw = cmd[..., 6]
    # Compute the envelope at the actual dh, not the snapped preset, so
    # cmds slightly off-preset still get sensible bin boundaries.
    max_int, max_slope, min_int, min_slope = (spec.dw_coeffs[0], spec.dw_coeffs[1],
                                              spec.dw_coeffs[2], spec.dw_coeffs[3])
    min_dw = min_int + min_slope * dh
    max_dw = max_int + max_slope * dh
    span = max_dw - min_dw
    frac = jnp.clip((dw - min_dw) / jnp.maximum(span, 1e-9), 0.0, 0.99999)
    return (frac * N_WIDTH).astype(jnp.int32)


def cmd_to_bin(cmd: jnp.ndarray, spec: BinSpec = PHANTOMX_BIN_SPEC) -> jnp.ndarray:
    """Return bin index in [0, N_BINS-1] for a (..., 9)-shaped cmd."""
    m = motion_idx(cmd, spec)
    h = height_idx(cmd, spec)
    w = width_idx(cmd, spec)
    return (m * (N_HEIGHT * N_WIDTH) + h * N_WIDTH + w).astype(jnp.int32)
