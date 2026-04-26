"""
gait — hexapod gait library.

Single source of truth for the analytical gait math. Used by:
  - simple_gait.py (demo viewer)
  - envs/hexapod_env.py (training, after task D)
  - pilot.py (teleop, after task B)
  - ESP32-S3 firmware (deployment target)

Public API:
  Controller     — stateful gait controller (use this).
  LEG_NAMES      — leg index → name mapping.
  COXA_POS_BODY  — MJCF coxa joint positions in body frame.
  LEG_PHASE      — tripod phase offsets per leg.

See project_gait_training_architecture.md for design rationale.
"""

from .controller import Controller, LEG_NAMES, COXA_POS_BODY, LEG_PHASE, NEUTRAL_POSE

__all__ = ["Controller", "LEG_NAMES", "COXA_POS_BODY", "LEG_PHASE", "NEUTRAL_POSE"]
