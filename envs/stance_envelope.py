"""envs/stance_envelope.py — single source of truth for stance cmd envelope.

When the verified bounds change (via tools/watch_scaffold.py --interactive),
edit this file ONLY. The env's cmd sampler, the watch tests, and the
interactive tuner all import from here, so everything stays in sync.

Consumers:
  - envs/hexapod_env_jax.py        (training cmd sampling)
  - scripts/demo_phases.py         (watch tests 7/8/9)
  - tools/watch_scaffold.py        (interactive stance presets)
  - tools/analyze_cmd_distances.py (cmd-space diagnostics)

Coefficients verified 2026-05-10 on the mesh model
(models/phantomx.xml) at MAX_SPEED forward + tightest arc-turn.
"""
from __future__ import annotations

# --- Stance height (cmd[5] = dh) ----------------------------------------------
# Sign convention: dh NEGATIVE = body raised (world z increases),
#                  dh POSITIVE = body lowered (squat).
DH_MIN = -0.045   # body raised 45 mm above neutral (max raised)
DH_MAX = +0.035   # body lowered 35 mm below neutral (max squat)

# 5 evenly-spaced height presets. Used by:
#   - tools/watch_scaffold.py interactive [6] preset cycle
#   - scripts/demo_phases.py height + combined-stance tests
HEIGHT_PRESETS = (-0.045, -0.025, -0.005, +0.015, +0.035)

# --- Stance width (cmd[6] = dw) — dh-conditional ------------------------------
# Linear lower-envelope fit (each line touches 2 measured anchors and
# undershoots the rest, never exceeding the verified safe envelope).
# Tiny extra margin baked into the intercept for float-precision safety.
MAX_DW_INTERCEPT = 0.0703
MAX_DW_SLOPE     = 0.8333
MIN_DW_INTERCEPT = -0.0283
MIN_DW_SLOPE     = 0.2500

# Joint-range bounds — these are the absolute extremes across all dh in
# [DH_MIN, DH_MAX]. The actual sampler clips to the dh-conditional range
# below, but cmd_sample_ranges needs a single rectangular envelope.
DW_JOINT_MIN = -0.040   # min over all dh
DW_JOINT_MAX = +0.100   # max over all dh


def safe_dw_range(dh):
    """Return (min_dw, max_dw) for the given dh. Pure arithmetic — works
    with Python floats, numpy scalars, jax scalars interchangeably."""
    return (MIN_DW_INTERCEPT + MIN_DW_SLOPE * dh,
            MAX_DW_INTERCEPT + MAX_DW_SLOPE * dh)
