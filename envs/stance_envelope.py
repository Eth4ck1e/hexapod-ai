"""envs/stance_envelope.py — stance cmd envelope, per robot profile.

The numbers live in `gait/robot_profile.py` (one `RobotProfile` per robot);
this module packages them as a `StanceEnvelope` and keeps the old PhantomX
module-level constants as aliases so legacy consumers keep working.

Consumers:
  - envs/hexapod_env_jax.py        (training cmd sampling — via EnvParams)
  - envs/cmd_bins.py               (dh presets + dh-conditional dw fifths)
  - scripts/demo_phases.py         (watch tests 7/8/9)
  - tools/watch_scaffold.py        (interactive stance presets)
  - tools/analyze_cmd_distances.py (cmd-space diagnostics)

PhantomX coefficients verified 2026-05-10 on the mesh model
(models/phantomx.xml) at MAX_SPEED forward + tightest arc-turn. Other
robots: tools/derive_stance_envelope.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from gait.robot_profile import PHANTOMX, RobotProfile, profile_for_model


@dataclass(frozen=True)
class StanceEnvelope:
    # Sign convention: dh NEGATIVE = body raised, POSITIVE = squat.
    dh_min: float
    dh_max: float
    height_presets: tuple            # evenly spaced dh presets (len = cmd_bins.N_HEIGHT)
    # Linear dh-conditional dw bounds.
    max_dw_intercept: float
    max_dw_slope: float
    min_dw_intercept: float
    min_dw_slope: float
    # Rectangular bound over all dh (cmd_sample_ranges needs one box).
    dw_joint_min: float
    dw_joint_max: float

    def safe_dw_range(self, dh):
        """(min_dw, max_dw) at this dh. Pure arithmetic — floats, numpy and
        jax scalars all work."""
        return (self.min_dw_intercept + self.min_dw_slope * dh,
                self.max_dw_intercept + self.max_dw_slope * dh)

    @property
    def dw_coeffs(self) -> tuple:
        """(max_int, max_slope, min_int, min_slope) — packs into a jnp array."""
        return (self.max_dw_intercept, self.max_dw_slope,
                self.min_dw_intercept, self.min_dw_slope)


def envelope_for(robot) -> StanceEnvelope:
    """Build the envelope for a `RobotProfile`, a profile name, or an MJCF
    path (profile inferred from the filename)."""
    if isinstance(robot, RobotProfile):
        prof = robot
    elif isinstance(robot, str) and robot.endswith(".xml"):
        prof = profile_for_model(robot)
    else:
        from gait.robot_profile import get_profile
        prof = get_profile(robot)
    return StanceEnvelope(
        dh_min=prof.dh_min, dh_max=prof.dh_max,
        height_presets=tuple(prof.height_presets),
        max_dw_intercept=prof.max_dw_intercept, max_dw_slope=prof.max_dw_slope,
        min_dw_intercept=prof.min_dw_intercept, min_dw_slope=prof.min_dw_slope,
        dw_joint_min=prof.dw_joint_min, dw_joint_max=prof.dw_joint_max,
    )


PHANTOMX_ENVELOPE = envelope_for(PHANTOMX)

# --- Legacy PhantomX aliases (module constants) ------------------------------
DH_MIN           = PHANTOMX_ENVELOPE.dh_min
DH_MAX           = PHANTOMX_ENVELOPE.dh_max
HEIGHT_PRESETS   = PHANTOMX_ENVELOPE.height_presets
MAX_DW_INTERCEPT = PHANTOMX_ENVELOPE.max_dw_intercept
MAX_DW_SLOPE     = PHANTOMX_ENVELOPE.max_dw_slope
MIN_DW_INTERCEPT = PHANTOMX_ENVELOPE.min_dw_intercept
MIN_DW_SLOPE     = PHANTOMX_ENVELOPE.min_dw_slope
DW_JOINT_MIN     = PHANTOMX_ENVELOPE.dw_joint_min
DW_JOINT_MAX     = PHANTOMX_ENVELOPE.dw_joint_max


def safe_dw_range(dh, envelope: StanceEnvelope = PHANTOMX_ENVELOPE):
    """Return (min_dw, max_dw) for the given dh. Defaults to PhantomX."""
    return envelope.safe_dw_range(dh)
