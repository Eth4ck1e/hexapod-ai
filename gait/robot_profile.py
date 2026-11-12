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

    # --- scaffold behaviour --------------------------------------------------
    # Arc-aware turn-while-walking exists only in the numpy controller; the
    # JAX port (training) always superposes translate + spin. Robots trained
    # with the JAX env should set this False so eval/watch see the same
    # scaffold the policy was trained against. PhantomX keeps True for
    # backward compatibility with its existing checkpoints and priors.
    arc_turning: bool = True

    # --- stance envelope (cmd[5] = dh, cmd[6] = dw) --------------------------
    # dh NEGATIVE = body raised, POSITIVE = squat. dw is dh-conditional:
    #   max_dw(dh) = max_dw_intercept + max_dw_slope * dh
    #   min_dw(dh) = min_dw_intercept + min_dw_slope * dh
    # PhantomX numbers were verified by eye on the mesh model (2026-05-10);
    # other robots: tools/derive_stance_envelope.py gives a kinematic inner
    # bound. envs/stance_envelope.py and envs/cmd_bins.py consume these.
    dh_min: float = -0.045
    dh_max: float = +0.035
    height_presets: tuple = (-0.045, -0.025, -0.005, +0.015, +0.035)
    max_dw_intercept: float = 0.0703
    max_dw_slope: float = 0.8333
    min_dw_intercept: float = -0.0283
    min_dw_slope: float = 0.2500
    dw_joint_min: float = -0.040          # min over all dh (rectangular bound)
    dw_joint_max: float = +0.100          # max over all dh

    # --- sim / env parameters (PhantomX defaults = envs/hexapod_env_jax.py
    # EnvParams as of v27). make_env_params() reads these via the Controller's
    # profile, so a robot is fully described in one place.
    spawn_z: float = 0.18                 # body z at reset
    ref_body_z: float = 0.13              # neutral-stance body height reference
    min_body_z: float = 0.05              # fall threshold
    foot_radius: float = 0.008            # contact sphere radius (m)
    foot_contact_z: float = 0.005         # foot-centre z below this = in contact
    joint_vel_limit: float = 6.18         # motor no-load max (rad/s)
    joint_torque_limit: float = 1.5       # motor stall (N·m)
    foot_force_limit: float = 30.0        # per-foot force ceiling (N)
    height_dev: float = 0.025             # "full" height tracking error (m)
    width_dev: float = 0.020              # "full" width tracking error (m)
    foot_residual_scale: float = 0.020    # foot-space action authority (m per unit)
    init_body_z_noise: float = 0.010      # spawn-height DR (m)

    # --- policy observation ---------------------------------------------------
    # "full": the v27 78-dim obs (true joint state + IMU + body linvel).
    # "hardware": only what an open-loop-servo bot can measure — commanded
    #   joint angle + its finite difference, IMU quat/gyro (zeroed when
    #   imu_available=False), scaffold hint, phase, cmd. No body linvel.
    # See envs/obs_layout.py.
    obs_variant: str = "full"
    imu_available: bool = True

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
    spawn_z=0.18, ref_body_z=0.13, min_body_z=0.06, foot_radius=0.008,
    joint_vel_limit=6.18, joint_torque_limit=1.5, foot_force_limit=30.0,
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
# Mini hexapod — the small 9 g-servo chassis (second training target)
# ============================================================================
# Geometry from mini_hexapod/firmware/arduino/hexapod_firmware.ino (measured
# 2026-05-17). MJCFs are GENERATED by tools/gen_mini_hexapod_mjcf.py; keep
# the numbers here and there in sync (the generator is the source for the
# MJCF, this profile is the source for the gait). PLACEHOLDER = pending
# measurement of the real bot.
_MINI_L1 = 0.0        # PLACEHOLDER coxa-axis → femur-axis offset (firmware assumes 0)
_MINI_L2 = 0.0550
_MINI_L3 = 0.0895
_MINI_FOOT_R = 0.004  # PLACEHOLDER
MINI_HEXAPOD = RobotProfile(
    name="mini_hexapod",
    model_path="models/mini_hexapod_mjx.xml",
    model_patterns=("mini_hexapod*.xml",),
    leg_body_yaw=[
        -math.radians(135), -math.radians(90), -math.radians(45),
        +math.radians(135), +math.radians(90), +math.radians(45),
    ],
    coxa_pos_body=[
        [-0.05675, -0.0320, 0.0],   # RR
        [ 0.00000, -0.0424, 0.0],   # RM
        [ 0.05675, -0.0320, 0.0],   # RF
        [-0.05675, +0.0320, 0.0],   # LR
        [ 0.00000, +0.0424, 0.0],   # LM
        [ 0.05675, +0.0320, 0.0],   # LF
    ],
    coxa_length=_MINI_L1,
    femur_length=_MINI_L2,
    tibia_length=_MINI_L3,
    # L-pose (firmware home): femur horizontal, tibia straight down. With the
    # generated MJCF's axes (coxa +z, femur -y, tibia +y) that is exactly
    # (0, 0, +π/2) per leg, and every IK sign is +1.
    neutral_pose=[0.0, 0.0, math.pi / 2] * 6,
    joint_signs=np.ones((6, 3)),
    # Gait defaults — PLACEHOLDER, scaled from the firmware's working gait
    # (30 mm step, 1.5 s period) toward something brisker; calibrate with
    # tools/calibrate_scaffold_speed.py once the MJCF is trusted.
    gait_period=0.80,
    path_radius=0.015,                      # 30 mm stride, 15 mm lift
    path_res=100,
    stance_width=0.005,
    # Turn-radius window scaled by body size (~0.45× PhantomX).
    min_turn_radius=0.16,
    max_turn_radius=0.34,
    # Speed ladder from tools/calibrate_scaffold_speed.py on the placeholder
    # MJCF (2026-11-12, tools/scaffold_speed_calibration_mini.csv): rows with
    # <20 mm slip and peak torque below ~60% of the 0.18 N·m stall placeholder.
    # Above ~0.12 m/s the servos saturate. Re-run after measurement.
    speed_levels={
        0.5: {"gait_period": 1.60, "path_radius": 0.012},   # ~0.029 m/s
        1.0: {"gait_period": 1.20, "path_radius": 0.015},   # ~0.048 m/s
        1.5: {"gait_period": 1.00, "path_radius": 0.018},   # ~0.068 m/s
        2.0: {"gait_period": 0.80, "path_radius": 0.018},   # ~0.084 m/s
        2.5: {"gait_period": 0.65, "path_radius": 0.018},   # ~0.100 m/s
        3.0: {"gait_period": 0.65, "path_radius": 0.022},   # ~0.122 m/s (0.18 N·m peak)
    },
    spawn_z=_MINI_L3 + _MINI_FOOT_R + 0.015,
    ref_body_z=_MINI_L3 + _MINI_FOOT_R,
    min_body_z=0.03,
    foot_radius=_MINI_FOOT_R,
    # Miuzei MS18 datasheet-typical — PLACEHOLDER (see generator).
    joint_vel_limit=10.5,
    joint_torque_limit=0.18,
    foot_force_limit=8.0,                   # ~2× bodyweight-per-tripod-foot
    # Train/eval parity: the JAX controller has no arc mode.
    arc_turning=False,
    # Kinematic inner envelope from tools/derive_stance_envelope.py
    # (2026-11-12, placeholder MJCF, 3° joint margin, 90% dh shrink).
    # Re-derive after the real bot is measured.
    dh_min=-0.028, dh_max=0.032,
    height_presets=(-0.028, -0.013, 0.002, 0.017, 0.032),
    max_dw_intercept=0.0297, max_dw_slope=0.7714,
    min_dw_intercept=-0.0229, min_dw_slope=0.5890,
    dw_joint_min=-0.040, dw_joint_max=0.055,
    # Env scales ≈ PhantomX × leg-length ratio (0.65).
    foot_contact_z=_MINI_FOOT_R + 0.002,
    height_dev=0.016,
    width_dev=0.013,
    foot_residual_scale=0.012,
    init_body_z_noise=0.006,
    # Deployment reality: open-loop servos, IMU not fitted yet.
    obs_variant="hardware",
    imu_available=False,
)


# ============================================================================
# Registry
# ============================================================================
PROFILES: dict[str, RobotProfile] = {
    PHANTOMX.name: PHANTOMX,
    MINI_HEXAPOD.name: MINI_HEXAPOD,
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
