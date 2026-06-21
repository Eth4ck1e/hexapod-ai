"""
gait — hexapod gait library.

Single source of truth for the analytical gait math. Used by:
  - simple_gait.py  (demo viewer)
  - IK_gait.py      (sandbox tests)
  - envs/hexapod_env.py (training scaffold)
  - eventual ESP32-S3 firmware

Public API:
  Controller     — stateful gait controller (closed-form IK + overlays).
  LEG_NAMES      — leg index → name mapping.
  COXA_POS_BODY  — MJCF coxa joint positions in body frame.
  LEG_BODY_YAW   — per-leg body-frame yaw of coxa quat.
  LEG_PHASE      — tripod phase offsets per leg.
  NEUTRAL_POSE   — joint pose for the calibrated (un-widened) foot rest.
  COXA_LENGTH, FEMUR_LENGTH, TIBIA_LENGTH — link lengths.

See docs/kinematics.md for the IK derivation.
"""

from .controller import (
    COXA_LENGTH,
    COXA_POS_BODY,
    FEMUR_LENGTH,
    LEG_BODY_YAW,
    LEG_NAMES,
    LEG_PHASE,
    NEUTRAL_POSE,
    TIBIA_LENGTH,
    Controller,
)

__all__ = [
    "Controller",
    "LEG_NAMES",
    "LEG_BODY_YAW",
    "COXA_POS_BODY",
    "LEG_PHASE",
    "NEUTRAL_POSE",
    "COXA_LENGTH",
    "FEMUR_LENGTH",
    "TIBIA_LENGTH",
]
