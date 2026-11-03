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
  (The bare constants above are the PhantomX profile's values, kept for
   backward compatibility. Prefer ctrl.profile.<field>.)
  RobotProfile   — per-robot geometry + gait defaults (gait/robot_profile.py).
  get_profile    — look up a registered profile by name ("phantomx", ...).
  PHANTOMX / PROFILES / register_profile — the registry.

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
from .robot_profile import (
    DEFAULT_PROFILE,
    MINI_HEXAPOD,
    PHANTOMX,
    PROFILES,
    RobotProfile,
    get_profile,
    profile_for_model,
    register_profile,
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
    "RobotProfile",
    "get_profile",
    "profile_for_model",
    "register_profile",
    "PHANTOMX",
    "MINI_HEXAPOD",
    "PROFILES",
    "DEFAULT_PROFILE",
]
