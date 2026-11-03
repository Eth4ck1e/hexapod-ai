"""
gait.robot_profile — per-robot geometry + gait defaults for the gait library.

A RobotProfile carries everything the analytical gait needs to know about a
specific chassis that is NOT read from the MJCF at calibration time:

  * leg mount geometry        (coxa positions + body-frame yaw per leg)
  * link lengths              (coxa / femur / effective tibia)
  * IK anchor pose            (joint angles at the calibrated foot rest; used
                               to solve the formula→MJCF joint offsets)
  * per-joint sign conventions (formula → MJCF joint direction)
  * gait defaults             (period, path radius, stance widening, turn
                               radius window, calibrated speed ladder)
  * default MJCF path

Things that ARE read from the MJCF stay in Controller.calibrate(): foot-tip
location (named foot_<LEG> geom or tibia mesh), per-leg foot rest in body
frame, and the derived frame transforms.

Adding a robot = adding a profile here + an MJCF under models/. Nothing in
the gait math is robot-specific beyond what the profile carries.

    from gait.robot_profile import PHANTOMX, get_profile
    ctrl = Controller(profile=get_profile("phantomx"))
"""

from __future__ import annotations

import fnmatch
import math
import os
import warnings
from dataclasses import dataclass, field

import numpy as np

# Leg order is fixed library-wide: right rear, right middle, right front,
# left rear, left middle, left front. MJCF joint/actuator declaration order
# must match (3 joints per leg: coxa, femur, tibia).
LEG_NAMES = ["RR", "RM", "RF", "LR", "LM", "LF"]

# Tripod phase offsets per leg (group A in sync, group B 180° offset).
# Same for every hexapod we build; kept here so a profile may override it.
TRIPOD_PHASE = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])


def _signs_right_left(coxa, femur_right, tibia_right):
    """Build a (6, 3) sign matrix where left legs flip femur + tibia relative
    to right legs. This is the PhantomX pattern (left-side MJCF joint axes
    are mirrored); see docs/kinematics.md."""
    right = np.array([coxa, femur_right, tibia_right], dtype=np.float64)
    left = np.array([coxa, -femur_right, -tibia_right], dtype=np.float64)
    return np.stack([right, right, right, left, left, left])


@dataclass(frozen=True)
class RobotProfile:
    name: str
    model_path: str                       # default MJCF (project-relative)

    # --- leg mount geometry -------------------------------------------------
    leg_body_yaw: np.ndarray              # (6,)   body-frame yaw of each coxa
    coxa_pos_body: np.ndarray             # (6, 3) coxa joint position, body frame

    # --- link lengths (m) ----------------------------------------------------
    coxa_length: float                    # horizontal coxa-axis → femur-axis
    femur_length: float
    tibia_length: float                   # tibia axis → foot tip (effective)

    # --- IK anchor ------------------------------------------------------------
    # Joint angles (18,) at the calibrated foot rest. calibrate() runs FK at
    # this pose, reads where the feet land, and solves formula→MJCF offsets
    # so the closed-form IK reproduces exactly this pose. Any reachable pose
    # works; it just has to be consistent with the MJCF's joint conventions.
    neutral_pose: np.ndarray

    # --- joint sign conventions ----------------------------------------------
    # (6, 3): multiply raw formula angles (coxa, femur, tibia) per leg to get
    # the MJCF rotation direction. Determined by MJCF joint axes.
    joint_signs: np.ndarray

    # --- gait defaults -------------------------------------------------------
    gait_period: float = 0.45             # s per full tripod cycle
    path_radius: float = 0.040            # m, half stride
    path_res: int = 100
    stance_width: float = 0.015           # m outward per foot at neutral
    min_turn_radius: float = 0.350        # m — below: pure spin
    max_turn_radius: float = 0.755        # m — above: pure translate
    # Calibrated speed ladder: level → {"gait_period", "path_radius"}.
    speed_levels: dict = field(default_factory=dict)

    leg_phase: np.ndarray = field(default_factory=lambda: TRIPOD_PHASE.copy())

    # Basenames (fnmatch globs) of MJCFs that belong to this robot. Lets
    # Controller(model_path) pick the profile from the file alone, so the
    # env / training scripts only ever carry a MODEL_PATH.
    model_patterns: tuple = ()

    def __post_init__(self):
        # Normalise array fields so callers can pass lists.
        object.__setattr__(self, "leg_body_yaw", np.asarray(self.leg_body_yaw, dtype=np.float64))
        object.__setattr__(self, "coxa_pos_body", np.asarray(self.coxa_pos_body, dtype=np.float64))
        object.__setattr__(self, "neutral_pose", np.asarray(self.neutral_pose, dtype=np.float64))
        object.__setattr__(self, "joint_signs", np.asarray(self.joint_signs, dtype=np.float64))
        object.__setattr__(self, "leg_phase", np.asarray(self.leg_phase, dtype=np.float64))
        assert self.leg_body_yaw.shape == (6,), self.leg_body_yaw.shape
        assert self.coxa_pos_body.shape == (6, 3), self.coxa_pos_body.shape
        assert self.neutral_pose.shape == (18,), self.neutral_pose.shape
        assert self.joint_signs.shape == (6, 3), self.joint_signs.shape
        assert self.leg_phase.shape == (6,), self.leg_phase.shape

    @property
    def max_reach(self) -> float:
        return self.femur_length + self.tibia_length


# ============================================================================
# PhantomX MK-III (Trossen) — the main training robot
# ============================================================================
# Every value below is verbatim from the pre-profile gait/controller.py so
# the refactor is numerically identical for this robot.
PHANTOMX = RobotProfile(
    name="phantomx",
    model_path="models/phantomx_simple_mjx.xml",
    model_patterns=("phantomx*.xml",),
    leg_body_yaw=[
        -math.radians(135), -math.radians(90), -math.radians(45),
        +math.radians(135), +math.radians(90), +math.radians(45),
    ],
    coxa_pos_body=[
        [-0.12, -0.06, 0.0],   # RR
        [ 0.00, -0.10, 0.0],   # RM
        [ 0.12, -0.06, 0.0],   # RF
        [-0.12, +0.06, 0.0],   # LR
        [ 0.00, +0.10, 0.0],   # LM
        [ 0.12, +0.06, 0.0],   # LF
    ],
    coxa_length=0.052,
    femur_length=0.065,
    # Effective tibia length = tibia joint → actual foot tip. The PhantomX
    # tibia tip sits at (0.134, 0.031, 0.0) tibia-local, magnitude 0.138.
    tibia_length=0.138,
    neutral_pose=[
         0.0, -0.21818873, -0.32188801,    # RR
         0.0, -0.24918906, -0.26150113,    # RM
         0.0, -0.21818873, -0.32188801,    # RF
         0.0,  0.21818873,  0.32188801,    # LR
         0.0,  0.24918906,  0.26150113,    # LM
         0.0,  0.21818873,  0.32188801,    # LF
    ],
    # Coxa +1 always. Femur +1 right / -1 left (axis flipped). Tibia -1
    # right / +1 left (axis flipped AND the formula's bend direction is
    # opposite of MJCF positive rotation). Do NOT "simplify" by negating all
    # three on the left — that breaks the math.
    joint_signs=_signs_right_left(coxa=+1.0, femur_right=+1.0, tibia_right=-1.0),
    # 2026-05-08 (v8 prep): hardware-realistic speed. path_radius held small
    # (≤ 40 mm) — large radii produce kinematic singularities and body sway;
    # gait_period does the speed work, bounded by AX-12A joint velocity.
    # 4 × 0.040 / 0.45 = 0.356 m/s ≈ 0.9 × theoretical hardware max.
    gait_period=0.45,
    path_radius=0.040,
    path_res=100,
    stance_width=0.015,
    # Turn-radius window: MIN puts the turn center outside the body envelope
    # (at 0.25 m inner legs tried to orbit a point right next to them).
    min_turn_radius=0.350,
    max_turn_radius=0.755,
    # Calibrated under ±1.5 N·m motor torque — tools/scaffold_speed_calibration.csv
    # (49-combo sweep, 2026-05-07). Slip 2-4 mm per stance at all levels;
    # levels ≥ 2.5 begin to saturate motor torque.
    speed_levels={
        0.5: {"gait_period": 1.50, "path_radius": 0.020},   # ~0.050 m/s
        1.0: {"gait_period": 1.50, "path_radius": 0.030},   # ~0.077 m/s
        1.5: {"gait_period": 1.50, "path_radius": 0.040},   # ~0.103 m/s
        2.0: {"gait_period": 1.20, "path_radius": 0.040},   # ~0.128 m/s
        2.5: {"gait_period": 1.00, "path_radius": 0.050},   # ~0.190 m/s
        3.0: {"gait_period": 0.85, "path_radius": 0.060},   # ~0.261 m/s
        3.5: {"gait_period": 0.75, "path_radius": 0.070},   # ~0.362 m/s
    },
)


# ============================================================================
# Registry
# ============================================================================
PROFILES: dict[str, RobotProfile] = {
    PHANTOMX.name: PHANTOMX,
}

DEFAULT_PROFILE = PHANTOMX


def get_profile(name: str | None = None) -> RobotProfile:
    """Look up a profile by name. None → DEFAULT_PROFILE (PhantomX)."""
    if name is None:
        return DEFAULT_PROFILE
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(
            f"unknown robot profile {name!r}; known: {sorted(PROFILES)}"
        ) from None


def register_profile(profile: RobotProfile) -> RobotProfile:
    PROFILES[profile.name] = profile
    return profile


def profile_for_model(model_path: str) -> RobotProfile:
    """Pick the profile whose model_patterns match the MJCF basename.
    Falls back to DEFAULT_PROFILE with a warning if nothing matches."""
    base = os.path.basename(str(model_path))
    for prof in PROFILES.values():
        if any(fnmatch.fnmatch(base, pat) for pat in prof.model_patterns):
            return prof
    warnings.warn(
        f"no robot profile claims MJCF {base!r}; using {DEFAULT_PROFILE.name!r}. "
        f"Add the filename to a profile's model_patterns in gait/robot_profile.py.",
        stacklevel=2,
    )
    return DEFAULT_PROFILE
