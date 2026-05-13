"""
IK_gait.py — closed-form IK + per-leg coxa-frame path tables (sandbox).

Drives ONE leg using a brand-new closed-form IK and a precomputed coxa-local
path table. Other legs hold NEUTRAL_POSE. Body is pinned (kinematic-only,
no gravity / no mj_step), so the leg motion can be verified in isolation.

Self-contained — does NOT modify gait/, envs/, or simple_gait.py.
gait.Controller is imported only to pull calibrated foot-rest positions.

Run:
    mjpython IK_gait.py        (macOS)
    python   IK_gait.py        (Linux/Windows)
"""

import math
import time
import numpy as np
import mujoco
import mujoco.viewer

from gait import Controller, NEUTRAL_POSE


# ============================================================================
# CONFIG
# ============================================================================
MODEL_PATH = "models/phantomx.xml"

MODE        = "tripod"     # "single" | "tripod" (translate) | "spin" | "stand" | "smart"
TEST_LEG    = 0            # 0:RR 1:RM 2:RF 3:LR 4:LM 5:LF (single mode only)
GAIT_PERIOD = 1.5          # seconds per full cycle
PATH_RADIUS = 0.025        # half-stride / lift height (m)
PATH_RES    = 100          # samples in the precomputed path table
DEFAULT_HEADING = 0.0      # heading at which the path table is precomputed (rad).
                           # heading=0 means the path's +X aligns with body +X.

# Default stance width offset: pushes each foot rest position outward (along
# coxa-local +X) by this amount. Widens the support polygon and lands the foot
# on the ground at a more vertical angle. Set to 0 for the calibrated neutral.
DEFAULT_STANCE_WIDTH = 0.015   # m — +15 mm outward per foot

# Heading cycle — body-frame direction in degrees (0 = forward, 90 = left strafe,
# 180 = backward, 270 = right strafe). Each heading dwells for HEADING_DWELL_SEC
# before stepping to the next. Set CYCLE_HEADINGS=False to walk in a single
# direction (HEADING_FIXED_DEG below).
CYCLE_HEADINGS    = False
HEADING_CYCLE_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
HEADING_DWELL_SEC = 4.0
HEADING_FIXED_DEG = 0.0     # used only when CYCLE_HEADINGS = False

# Spin cycle (spin mode) — list of (spin_sign, dwell_seconds).
# spin_sign: +1 = CCW around body +Z (left turn), -1 = CW (right turn).
# At MAX_YAW_RATE the body covers ~360 deg in ~20 s; 22 s gives a small buffer.
SPIN_CYCLE = [(-1, 22.0), (+1, 22.0)]   # right 360 then left 360, repeat

# Live stance modulation — applied on top of FOOT_REST_COXA every tick. Active
# in ALL modes (stand / tripod / spin / single), so you can e.g. set MODE="tripod"
# and watch the gait widen/narrow/raise/lower while walking.
#   width_delta_mm  : positive = stance wider  (foot pushed outward in coxa +X).
#   body_raise_mm   : positive = body raised   (foot pushed deeper, coxa z -=).
#
# When CYCLE_STANCE is True, the value is a smooth sine sweep:
#   - WIDTH sweep over WIDTH_PERIOD_SEC: 0 -> MAX -> 0 -> MIN -> 0
#   - RAISE sweep over RAISE_PERIOD_SEC: 0 -> MAX -> 0 -> MIN -> 0
#   - Phases run sequentially, then loop.
# MAX and MIN are SIGNED endpoints (not magnitudes). Typically MAX > 0 > MIN,
# but you can also push the sweep entirely above or below zero (e.g.
# MIN = +5 makes the sweep stay above 0). Tune up or down to find the
# mechanical limits before the IK clips or the robot falls / lifts a foot.
CYCLE_STANCE             = False
STANCE_WIDTH_MAX_MM      = 100    # widest stance (signed; + = wider)
STANCE_WIDTH_MIN_MM      = -75    # narrowest stance (signed; - = narrower than default)
STANCE_RAISE_MAX_MM      = 50     # tallest body  (signed; + = body raised)
STANCE_RAISE_MIN_MM      = -75    # lowest body   (signed; - = body lowered)
STANCE_WIDTH_PERIOD_SEC  = 6.0    # one full width sweep
STANCE_RAISE_PERIOD_SEC  = 6.0    # one full raise sweep

# Static fallbacks when CYCLE_STANCE = False.
STANCE_FIXED_WIDTH_MM = 0.0
STANCE_FIXED_RAISE_MM = 0.0

# ---------- Speed control ----------
# Speed = stride_length / period. Both factors move TOGETHER and inversely:
#   higher speed -> SHORTER period AND BIGGER path (longer stride + higher lift).
# A single SPEED_FACTOR in [-1, +1] drives both:
#   -1 = slowest  (period x (1 + PERIOD_RANGE), path x (1 - PATH_RANGE))
#    0 = nominal
#   +1 = fastest  (period x (1 - PERIOD_RANGE), path x (1 + PATH_RANGE))
# Period is the dominant lever (default ±50%); path is a smaller secondary
# adjustment (default ±15%) that fine-tunes stride length per step.
# Tune the two RANGE_PCT knobs below to expand or narrow the speed envelope.
SPEED_PATH_RANGE_PCT   = 0.15     # path size range, fraction (e.g., 0.15 = ±15%)
SPEED_PERIOD_RANGE_PCT = 0.75     # period range,    fraction (e.g., 0.50 = ±50%)

# Hard floors so PERIOD_RANGE_PCT or PATH_RANGE_PCT >= 1.0 don't produce zero
# or negative values (which would divide by zero in gait_phase or invert the
# gait direction). Tune down only if you want to *cap* maximum speed harder.
SPEED_PERIOD_MIN_SEC   = 0.10     # absolute lower bound on gait period (s)
SPEED_PATH_SCALE_MIN   = 0.05     # absolute lower bound on path scale

# Manual-mode speed cycling (ignored in smart mode — smart phases set their own).
# Continuous sine sweep: factor goes 0 -> +1 -> 0 -> -1 -> 0 over SWEEP_PERIOD.
CYCLE_SPEED            = False
SPEED_SWEEP_PERIOD_SEC = 16.0     # one full -1 <-> +1 sweep
SPEED_FIXED            = 0.0      # used when CYCLE_SPEED = False

# ---------- Body tilt (pitch/roll) overlay ----------
# Tilts the body around its own axes by rotating each foot's commanded body-
# frame position by R^-1 (so feet stay planted while the body rotates around
# them). Active in ALL modes — overlay during stand / walk / spin.
#   pitch positive = nose up
#   roll  positive = body's left side up
# Sweep behaves like the stance sweep: a pitch sweep then a roll sweep, looping.
# MAX/MIN are signed degrees.
CYCLE_TILT             = True
TILT_PITCH_MAX_DEG     = +15
TILT_PITCH_MIN_DEG     = -15
TILT_ROLL_MAX_DEG      = +15
TILT_ROLL_MIN_DEG      = -15
TILT_PITCH_PERIOD_SEC  = 6.0
TILT_ROLL_PERIOD_SEC   = 6.0
TILT_FIXED_PITCH_DEG   = 0.0      # used when CYCLE_TILT = False
TILT_FIXED_ROLL_DEG    = 0.0

# ---------- Body shift (translate body in body frame) overlay ----------
# Shifts the body by (sx, sy) in body frame while planted feet stay in
# world. Manual modes only — smart phases drive shift via the script.
CYCLE_SHIFT            = False
SHIFT_X_MAX_MM         = +30
SHIFT_X_MIN_MM         = -30
SHIFT_Y_MAX_MM         = +30
SHIFT_Y_MIN_MM         = -30
SHIFT_X_PERIOD_SEC     = 6.0
SHIFT_Y_PERIOD_SEC     = 6.0
SHIFT_FIXED_X_MM       = 0.0      # used when CYCLE_SHIFT = False
SHIFT_FIXED_Y_MM       = 0.0

# Smart test script — runs through every behavior + every overlay combination.
# Each entry: (label, behavior, duration_s, params)
#   behavior: "stand" | "walk" | "spin"
#   params: any of:
#       "heading_deg"   (walk only)  fixed body-frame heading
#       "heading_sweep" (walk only)  True -> sweep heading 0->360 over the phase
#       "spin_sign"     (spin only)  +1 (CCW) or -1 (CW)
#       "stance"        (any)        "width" | "raise"  -> overlay sweep
#       "shift"         (any)        "x" | "y" | "circle"  -> body-shift sweep
#       "tilt"          (any)        "pitch" | "roll" | "circle"  -> body-tilt sweep
# Intentionally NO walk+spin combinations — we want the policy to learn that.
SMART_TEST_SCRIPT = [
    ("STAND  (no overlay)",                  "stand", 3.0,  {}),
    ("STAND  + WIDTH  sweep",                "stand", 6.0,  {"stance": "width"}),
    ("STAND  + RAISE  sweep",                "stand", 6.0,  {"stance": "raise"}),
    ("STAND  + SHIFT_X sweep",               "stand", 6.0,  {"shift":  "x"}),
    ("STAND  + SHIFT_Y sweep",               "stand", 6.0,  {"shift":  "y"}),
    ("STAND  + SHIFT circle",                "stand", 6.0,  {"shift":  "circle"}),
    ("STAND  + PITCH sweep ±10°",            "stand", 6.0,  {"tilt":   "pitch"}),
    ("STAND  + ROLL  sweep ±15°",            "stand", 6.0,  {"tilt":   "roll"}),
    ("STAND  + PITCH+ROLL circle",           "stand", 8.0,  {"tilt":   "circle"}),
    ("WALK   forward  (heading=0)",          "walk",  5.0,  {"heading_deg":   0}),
    ("WALK   strafe left (heading=90)",      "walk",  5.0,  {"heading_deg":  90}),
    ("WALK   backward (heading=180)",        "walk",  5.0,  {"heading_deg": 180}),
    ("WALK   strafe right (heading=270)",    "walk",  5.0,  {"heading_deg": 270}),
    ("WALK   heading sweep 0 -> 360",        "walk", 16.0,  {"heading_sweep": True}),
    ("WALK   forward + WIDTH  overlay",      "walk",  6.0,  {"heading_deg": 0, "stance": "width"}),
    ("WALK   forward + RAISE  overlay",      "walk",  6.0,  {"heading_deg": 0, "stance": "raise"}),
    ("WALK   forward SLOW (speed=-1)",       "walk",  6.0,  {"heading_deg": 0, "speed_factor": -1.0}),
    ("WALK   forward FAST (speed=+1)",       "walk",  6.0,  {"heading_deg": 0, "speed_factor": +1.0}),
    ("WALK   forward + SPEED sweep",         "walk", 12.0,  {"heading_deg": 0, "speed_sweep": True}),
    ("SPIN   right (CW)  full 360",          "spin", 22.0,  {"spin_sign": -1}),
    ("SPIN   left  (CCW) full 360",          "spin", 22.0,  {"spin_sign": +1}),
    ("SPIN   right + WIDTH overlay",         "spin",  8.0,  {"spin_sign": -1, "stance": "width"}),
    ("SPIN   left  + RAISE overlay",         "spin",  8.0,  {"spin_sign": +1, "stance": "raise"}),
    ("SPIN   left  + SPEED sweep",           "spin", 12.0,  {"spin_sign": +1, "speed_sweep": True}),
    ("STAND  return to neutral",             "stand", 2.0,  {}),
]

# Tripod groups (phase offsets within s in [0, 1)):
#   Group A (offset 0.0):   RR (0), RF (2), LM (4)
#   Group B (offset 0.5):   RM (1), LR (3), LF (5)
LEG_PHASE = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])

# ============================================================================
# PER-LEG CONSTANTS (from MJCF)
# ============================================================================
LEG_NAMES = ["RR", "RM", "RF", "LR", "LM", "LF"]

# Body-frame yaw of each coxa, derived from the coxa quats in models/phantomx.xml.
LEG_BODY_YAW = np.array([
    -math.radians(135), -math.radians(90), -math.radians(45),
    +math.radians(135), +math.radians(90), +math.radians(45),
])

# Coxa joint position in body frame.
COXA_POS_BODY = np.array([
    [-0.12, -0.06, 0.0],   # RR
    [ 0.00, -0.10, 0.0],   # RM
    [ 0.12, -0.06, 0.0],   # RF
    [-0.12, +0.06, 0.0],   # LR
    [ 0.00, +0.10, 0.0],   # LM
    [ 0.12, +0.06, 0.0],   # LF
])

COXA_LENGTH  = 0.052
FEMUR_LENGTH = 0.065
# Effective tibia length = distance from tibia joint to foot tip.
# The MJCF foot tip sits at (0.134, 0.031, 0.0) in tibia-local frame, magnitude
# 0.138 — NOT 0.133 (the textbook tibia length). Using 0.133 here breaks the
# closed-form IK off the rest pose because the formula's geometry no longer
# matches the actual leg.
TIBIA_LENGTH = 0.138


# ============================================================================
# FRAME TRANSFORMS
# ============================================================================
def body_to_coxa_local(p_body, leg_idx):
    """Express body-frame point in this leg's coxa-local frame.
    coxa-local: origin at coxa joint, +X outward (along leg's body-yaw direction),
    +Y horizontal-perpendicular (CCW from +X seen from above), +Z up.
    """
    rel = p_body - COXA_POS_BODY[leg_idx]
    c, s = math.cos(LEG_BODY_YAW[leg_idx]), math.sin(LEG_BODY_YAW[leg_idx])
    return np.array([
         c * rel[0] + s * rel[1],
        -s * rel[0] + c * rel[1],
         rel[2],
    ])


def coxa_local_to_body(p_coxa, leg_idx):
    """Inverse of body_to_coxa_local."""
    c, s = math.cos(LEG_BODY_YAW[leg_idx]), math.sin(LEG_BODY_YAW[leg_idx])
    return COXA_POS_BODY[leg_idx] + np.array([
         c * p_coxa[0] - s * p_coxa[1],
         s * p_coxa[0] + c * p_coxa[1],
         p_coxa[2],
    ])


# ============================================================================
# CLOSED-FORM IK (coxa-local frame → joint angles, "raw" formula convention)
# ============================================================================
# Convention used here:
#   coxa angle: rotation around vertical, measured CCW from leg's outward +X.
#   femur angle: rotation in the leg's vertical plane, measured CCW from
#                in-plane +X (positive = femur tip raised above horizontal).
#   tibia angle: knee bend; π − interior knee angle (so 0 = straight leg,
#                positive = knee folded).
#
# These are GEOMETRIC angles, not MJCF joint angles. The MJCF joints have
# their own zero references (set by the femur/tibia body quats), so we
# anchor this formula to the MJCF at calibration time by adding a per-leg
# constant offset (see calibrate_offsets() below).
def ik_raw(fx, fy, fz):
    """Closed-form 3-link IK in geometric convention.
    Input: foot target in coxa-local frame.
    Returns: (coxa_geom, femur_geom, tibia_geom) — pre-MJCF-offset.
    """
    # Coxa: which vertical plane?
    coxa = math.atan2(fy, fx)

    # Reduce to 2D in the leg's vertical plane.
    r    = math.hypot(fx, fy)
    x_fp = r - COXA_LENGTH        # foot relative to femur joint, in-plane horizontal
    z_fp = fz                     # foot relative to femur joint, vertical

    D = math.hypot(x_fp, z_fp)

    # Clamp to reachable workspace.
    D = min(D, FEMUR_LENGTH + TIBIA_LENGTH - 1e-6)
    D = max(D, abs(FEMUR_LENGTH - TIBIA_LENGTH) + 1e-6)

    # Law of cosines.
    cos_a = (FEMUR_LENGTH**2 + D**2 - TIBIA_LENGTH**2) / (2 * FEMUR_LENGTH * D)
    cos_g = (FEMUR_LENGTH**2 + TIBIA_LENGTH**2 - D**2) / (2 * FEMUR_LENGTH * TIBIA_LENGTH)
    cos_a = max(-1.0, min(1.0, cos_a))
    cos_g = max(-1.0, min(1.0, cos_g))
    alpha = math.acos(cos_a)
    gamma = math.acos(cos_g)

    beta  = math.atan2(z_fp, x_fp)        # foot direction from femur joint

    # Elbow-up branch (knee above the femur→foot line — grasshopper config).
    femur = beta + alpha
    tibia = math.pi - gamma

    return coxa, femur, tibia


# ============================================================================
# CALIBRATION — get foot rest in coxa-local + per-leg formula→MJCF offsets
# ============================================================================
print("Calibrating against gait.Controller for foot-rest positions...")
_ctrl = Controller(MODEL_PATH)
LEG_ORIGIN_BODY = _ctrl.LEG_ORIGIN_BODY.copy()         # (6, 3) body frame

# Foot rest position in coxa-local frame. If DEFAULT_STANCE_WIDTH is non-zero,
# this is the widened rest — the only "neutral" downstream code sees.
FOOT_REST_COXA = np.array([
    body_to_coxa_local(LEG_ORIGIN_BODY[i], i) for i in range(6)
])
if DEFAULT_STANCE_WIDTH != 0.0:
    FOOT_REST_COXA[:, 0] += DEFAULT_STANCE_WIDTH

print("\nFoot rest position in coxa-local frame "
      f"(stance widened by {DEFAULT_STANCE_WIDTH*1000:+.1f} mm):")
print(f"  {'leg':<4}  {'(fx, fy, fz)':<32}")
for i, n in enumerate(LEG_NAMES):
    p = FOOT_REST_COXA[i]
    print(f"  {n}    ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")

# Body origin in each leg's coxa-local frame. This is the spin axis location.
# It's a fixed point per leg (independent of stance), so cached at boot.
BODY_ORIGIN_COXA = np.array([
    body_to_coxa_local(np.zeros(3), i) for i in range(6)
])

# Spin reference radius — the canonical path's px ∈ [-R, +R] maps to angular
# stride dθ = px / SPIN_REF_RADIUS. Use the outer-leg radius so the longest-
# traveling foot covers exactly PATH_RADIUS of arc length per max stride.
def _arc_radius(leg_idx, foot_rest):
    dx = foot_rest[0] - BODY_ORIGIN_COXA[leg_idx, 0]
    dy = foot_rest[1] - BODY_ORIGIN_COXA[leg_idx, 1]
    return math.hypot(dx, dy)

_arc_radii_default = np.array([_arc_radius(i, FOOT_REST_COXA[i]) for i in range(6)])
SPIN_REF_RADIUS    = float(_arc_radii_default[0])   # outer leg (RR) radius
MAX_YAW_RATE       = 4.0 * (PATH_RADIUS / SPIN_REF_RADIUS) / GAIT_PERIOD

print(f"\nSpin geometry (default stance):")
print(f"  {'leg':<4}  {'arc radius (m)':<16}")
for i, n in enumerate(LEG_NAMES):
    print(f"  {n}    {_arc_radii_default[i]:.4f}")
print(f"  SPIN_REF_RADIUS = {SPIN_REF_RADIUS:.4f} m  "
      f"(MAX_YAW_RATE = {MAX_YAW_RATE:.3f} rad/s "
      f"= {math.degrees(MAX_YAW_RATE):.1f} deg/s)")


# Per-joint sign convention: each joint maps from formula→MJCF differently
# depending on the MJCF axis direction.
#   coxa:  +1 for all legs (axis is "0 0 1" everywhere; CCW rotation around +Z
#          puts foot at +Y in coxa-local for both sides).
#   femur: +1 right, -1 left (right axis "0 0 -1", left "0 0 +1" → flipped).
#   tibia: -1 right, +1 left. Right tibia axis "0 0 +1" tibia-local maps to
#          coxa-local -Y; positive MJCF tibia rotates +X toward +Z (lifts the
#          tibia link). The formula's tibia_raw = π−γ INCREASES as the knee
#          bends MORE (more downward), so they're opposite → sign -1 on right.
#          Left tibia is flipped from right → sign +1.
def joint_signs(leg_idx):
    is_left = leg_idx >= 3
    return (
        +1.0,                       # coxa
        -1.0 if is_left else +1.0,  # femur
        +1.0 if is_left else -1.0,  # tibia
    )

# Anchor the formula→MJCF offset table once, using the only known anchor point:
# NEUTRAL_POSE corresponds to the calibrated (un-widened) rest position.
# This is purely internal — the rest of the file works in terms of FOOT_REST_COXA.
_anchor_rest_coxa = np.array([
    body_to_coxa_local(LEG_ORIGIN_BODY[i], i) for i in range(6)
])
LEG_OFFSETS = np.zeros((6, 3))
for i in range(6):
    raw = np.array(ik_raw(*_anchor_rest_coxa[i]))
    expected = NEUTRAL_POSE[i*3:i*3+3]
    s_c, s_f, s_t = joint_signs(i)
    LEG_OFFSETS[i, 0] = expected[0] - s_c * raw[0]
    LEG_OFFSETS[i, 1] = expected[1] - s_f * raw[1]
    LEG_OFFSETS[i, 2] = expected[2] - s_t * raw[2]

print("\nFormula -> MJCF offsets per leg (constant):")
print(f"  {'leg':<4}  {'(coxa, femur, tibia) offsets [rad]':<40}")
for i, n in enumerate(LEG_NAMES):
    o = LEG_OFFSETS[i]
    print(f"  {n}    ({o[0]:+.4f}, {o[1]:+.4f}, {o[2]:+.4f})")


def ik_mjcf(fx, fy, fz, leg_idx):
    """MJCF-compatible 3-link IK. Foot target in coxa-local frame -> joint angles."""
    coxa, femur, tibia = ik_raw(fx, fy, fz)
    s_c, s_f, s_t = joint_signs(leg_idx)
    return (
        s_c * coxa  + LEG_OFFSETS[leg_idx, 0],
        s_f * femur + LEG_OFFSETS[leg_idx, 1],
        s_t * tibia + LEG_OFFSETS[leg_idx, 2],
    )


# Derive the widened neutral joint configuration via the now-anchored IK.
# This is the static pose for FOOT_REST_COXA — used for settle and as the
# joint baseline in the gait. When DEFAULT_STANCE_WIDTH = 0 it equals NEUTRAL_POSE.
GAIT_NEUTRAL_POSE = np.zeros(18)
for i in range(6):
    GAIT_NEUTRAL_POSE[i*3:i*3+3] = ik_mjcf(*FOOT_REST_COXA[i], i)


# ============================================================================
# VALIDATION — round-trip check via mj_forward
# ============================================================================
print("\nRound-trip validation (formula -> joint angles -> mj_forward -> foot):")
print(f"  {'leg':<4}  {'rest pos (body) error [mm]':<30}")

_val_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
_val_data  = mujoco.MjData(_val_model)
_TIBIA_BID = np.array([
    mujoco.mj_name2id(_val_model, mujoco.mjtObj.mjOBJ_BODY, f"tibia_{n}")
    for n in LEG_NAMES
])
_FOOT_TIP_LOCAL = _ctrl.FOOT_TIP_LOCAL.copy()


def fk_foot_body(joints_18, leg_idx):
    """Run FK with body pinned at world origin → returns foot in body frame."""
    _val_data.qpos[:]    = 0
    _val_data.qpos[3]    = 1.0
    _val_data.qpos[7:25] = joints_18
    mujoco.mj_forward(_val_model, _val_data)
    bid  = _TIBIA_BID[leg_idx]
    pos  = _val_data.xpos[bid]
    xmat = _val_data.xmat[bid].reshape(3, 3)
    return pos + xmat @ _FOOT_TIP_LOCAL[leg_idx]


for i, n in enumerate(LEG_NAMES):
    foot_target_coxa = FOOT_REST_COXA[i]
    foot_target_body = coxa_local_to_body(foot_target_coxa, i)
    angles = GAIT_NEUTRAL_POSE.copy()
    angles[i*3:i*3+3] = ik_mjcf(*foot_target_coxa, i)
    foot_actual = fk_foot_body(angles, i)
    err_mm = 1000.0 * np.linalg.norm(foot_actual - foot_target_body)
    print(f"  {n}    {err_mm:>10.3f}")


# Extended validation: perturb the foot target along the path (body X, Z) and
# check the IK still lands the foot where commanded. If errors blow up off-rest,
# the formula's per-leg offset is hiding a deeper geometric bug.
print("\nPerturbed validation for TEST_LEG (deltas in coxa-local):")
print(f"  Probing along path-like deltas (px in coxa-local Y for RM, pz in Z).")
print(f"  {'(dx,    dy,    dz)':<28}  {'foot err (body) [mm]':<22}  {'cmd-vs-actual delta [mm]':<28}")

# For TEST_LEG: probe the actual gait path samples plus a few out-of-plane points.
test_deltas_coxa = [
    np.array([0.000,  0.000,  0.000]),     # rest
    np.array([0.000, +0.025,  0.000]),     # peak forward
    np.array([0.000, -0.025,  0.000]),     # peak rear
    np.array([0.000,  0.000, +0.025]),     # peak lift
    np.array([0.000, +0.012, +0.012]),     # mid-swing
    np.array([0.000, -0.012, +0.012]),     # mid-swing other side
]

for dc in test_deltas_coxa:
    foot_coxa = FOOT_REST_COXA[TEST_LEG] + dc
    angles = GAIT_NEUTRAL_POSE.copy()
    angles[TEST_LEG*3:TEST_LEG*3+3] = ik_mjcf(*foot_coxa, TEST_LEG)
    foot_actual_body = fk_foot_body(angles, TEST_LEG)
    foot_target_body = coxa_local_to_body(foot_coxa, TEST_LEG)
    err_mm   = 1000.0 * np.linalg.norm(foot_actual_body - foot_target_body)
    delta_mm = 1000.0 * (foot_actual_body - foot_target_body)
    dc_mm    = 1000.0 * dc
    print(f"  ({dc_mm[0]:+5.1f},{dc_mm[1]:+6.1f},{dc_mm[2]:+6.1f})    "
          f"{err_mm:>10.3f}            "
          f"({delta_mm[0]:+6.2f},{delta_mm[1]:+6.2f},{delta_mm[2]:+6.2f})")


# ============================================================================
# PER-LEG PATH TABLE (precomputed in coxa-local frame, as deltas from rest)
# ============================================================================
def canonical_path(s):
    """2D canonical path. Returns (px, pz) in path frame.
    px ∈ [-R, +R] is along the heading direction; pz ∈ [0, +R] is lift."""
    if s < 0.5:
        theta = math.pi * (1.0 - 2.0 * s)
        return PATH_RADIUS * math.cos(theta), PATH_RADIUS * math.sin(theta)
    ss = (s - 0.5) / 0.5
    return PATH_RADIUS * (1.0 - 2.0 * ss), 0.0


def precompute_leg_path(leg_idx, default_heading=0.0):
    """Precompute path samples as DELTAS from foot rest, in coxa-local frame.
    default_heading is the body-frame heading at which the path is "pre-aligned."
    At runtime, a heading change Δθ rotates these deltas by Δθ in coxa XY.
    """
    yaw = LEG_BODY_YAW[leg_idx]
    # Body forward direction at default_heading, expressed in coxa-local:
    c_h, s_h = math.cos(default_heading), math.sin(default_heading)
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    dir_x =  c_y * c_h + s_y * s_h    # coxa-local x component of body heading
    dir_y = -s_y * c_h + c_y * s_h    # coxa-local y component

    deltas = np.zeros((PATH_RES, 3))
    for n in range(PATH_RES):
        s = n / PATH_RES
        px, pz = canonical_path(s)
        deltas[n, 0] = px * dir_x
        deltas[n, 1] = px * dir_y
        deltas[n, 2] = pz
    return deltas


# Precompute path table for ALL six legs (table indexed [leg][phase][dim]).
LEG_PATH_DELTAS = np.array([
    precompute_leg_path(i, DEFAULT_HEADING) for i in range(6)
])

print(f"\nPath table precomputed for all 6 legs (shape: {LEG_PATH_DELTAS.shape})")


# ============================================================================
# DEMO — drive all 6 legs (tripod gait) or single TEST_LEG via MODE
# ============================================================================
_heading_state = {"last_idx": -1}
_spin_state    = {"last_idx": -1}
_stance_state  = {"last_idx": -1}
_speed_state   = {"last_idx": -1}
_tilt_state    = {"last_phase": None}

# Continuous gait-phase tracker. Integrates dt/period each call so phase stays
# smooth across speed changes. Assumes monotonic t in normal operation; resets
# if t goes backward (sim reset).
_phase_tracker = {"phase": 0.0, "last_t": None}


def gait_phase(t, period):
    if _phase_tracker["last_t"] is None:
        _phase_tracker["last_t"] = t
        return _phase_tracker["phase"]
    dt = t - _phase_tracker["last_t"]
    if dt < 0.0:
        _phase_tracker["phase"]  = 0.0
        _phase_tracker["last_t"] = t
        return 0.0
    _phase_tracker["last_t"] = t
    _phase_tracker["phase"]  = (_phase_tracker["phase"] + dt / period) % 1.0
    return _phase_tracker["phase"]


def speed_period(factor):
    """Period (s) for the given speed factor in [-1, +1]; hard-floored to
    SPEED_PERIOD_MIN_SEC so RANGE_PCT >= 1.0 doesn't drive period to 0."""
    return max(GAIT_PERIOD * (1.0 - factor * SPEED_PERIOD_RANGE_PCT),
               SPEED_PERIOD_MIN_SEC)


def speed_path_scale(factor):
    """Path-size multiplier for the given speed factor in [-1, +1]; hard-floored
    to SPEED_PATH_SCALE_MIN so RANGE_PCT >= 1.0 doesn't invert the gait."""
    return max(1.0 + factor * SPEED_PATH_RANGE_PCT,
               SPEED_PATH_SCALE_MIN)


def current_speed_factor(t):
    """Smooth sine sweep of speed factor in [-1, +1] (or static SPEED_FIXED)."""
    if not CYCLE_SPEED:
        return SPEED_FIXED
    factor = math.sin(2.0 * math.pi * t / SPEED_SWEEP_PERIOD_SEC)

    # Print at quarter-cycle inflection points so the terminal isn't silent.
    # Quarters: 0 (zero rising), 0.25 (peak +1), 0.5 (zero falling), 0.75 (trough -1).
    quarter_idx = int((t % SPEED_SWEEP_PERIOD_SEC) / (SPEED_SWEEP_PERIOD_SEC / 4)) % 4
    if quarter_idx != _speed_state["last_idx"]:
        labels = ["rising through 0", "peak +1.00 (FAST)", "falling through 0", "trough -1.00 (SLOW)"]
        p     = speed_period(factor)
        scale = speed_path_scale(factor)
        print(f"  t={t:5.1f}s  speed sweep -> {labels[quarter_idx]}  "
              f"factor={factor:+.2f}  period={p:.2f}s  path x{scale:.2f}")
        _speed_state["last_idx"] = quarter_idx
    return factor


def _sweep_value_mm(s, max_value, min_value):
    """Asymmetric sine sweep visiting MAX (signed) on positive half and MIN (signed)
    on negative half, passing through 0 between halves.

    Both arguments are signed targets, NOT magnitudes:
      - MAX is the value reached at peak (s = +1).  Typically positive.
      - MIN is the value reached at trough (s = -1). Typically negative,
        but can be any signed value (e.g., MIN = +5 keeps the sweep above 0).
    """
    return max_value * s if s >= 0.0 else -min_value * s


def current_stance(t):
    """Return (width_delta_m, body_raise_m) at time t.
    width_delta: + = stance wider (foot pushed outward in coxa-local +X).
    body_raise:  + = body raised  (foot pushed lower in coxa-local Z, coxa z -= raise).
    """
    if not CYCLE_STANCE:
        return STANCE_FIXED_WIDTH_MM * 1e-3, STANCE_FIXED_RAISE_MM * 1e-3

    total = STANCE_WIDTH_PERIOD_SEC + STANCE_RAISE_PERIOD_SEC
    t_in_cycle = t % total
    if t_in_cycle < STANCE_WIDTH_PERIOD_SEC:
        phase     = t_in_cycle / STANCE_WIDTH_PERIOD_SEC
        s         = math.sin(2.0 * math.pi * phase)
        width_mm  = _sweep_value_mm(s, STANCE_WIDTH_MAX_MM, STANCE_WIDTH_MIN_MM)
        raise_mm  = 0.0
        active    = "width"
    else:
        phase     = (t_in_cycle - STANCE_WIDTH_PERIOD_SEC) / STANCE_RAISE_PERIOD_SEC
        s         = math.sin(2.0 * math.pi * phase)
        width_mm  = 0.0
        raise_mm  = _sweep_value_mm(s, STANCE_RAISE_MAX_MM, STANCE_RAISE_MIN_MM)
        active    = "raise"

    if active != _stance_state.get("last_phase"):
        if active == "width":
            print(f"  t={t:5.1f}s  stance sweep -> WIDTH "
                  f"[{STANCE_WIDTH_MIN_MM:+d}, {STANCE_WIDTH_MAX_MM:+d}] mm "
                  f"({STANCE_WIDTH_PERIOD_SEC:.1f}s)")
        else:
            print(f"  t={t:5.1f}s  stance sweep -> RAISE "
                  f"[{STANCE_RAISE_MIN_MM:+d}, {STANCE_RAISE_MAX_MM:+d}] mm "
                  f"({STANCE_RAISE_PERIOD_SEC:.1f}s)")
        _stance_state["last_phase"] = active

    return width_mm * 1e-3, raise_mm * 1e-3


def effective_rest(width_m, raise_m):
    """Return per-leg (6,3) coxa-local foot rest with stance modulation applied."""
    out = FOOT_REST_COXA.copy()
    out[:, 0] += width_m       # widen in coxa-local +X
    out[:, 2] -= raise_m       # raise body = foot deeper (more negative coxa-local z)
    return out


def current_tilt(t):
    """Return (pitch_rad, roll_rad) at time t.
    Sequential sine sweeps: pitch first, then roll, then loop.
    """
    if not CYCLE_TILT:
        return math.radians(TILT_FIXED_PITCH_DEG), math.radians(TILT_FIXED_ROLL_DEG)

    total = TILT_PITCH_PERIOD_SEC + TILT_ROLL_PERIOD_SEC
    t_in_cycle = t % total
    if t_in_cycle < TILT_PITCH_PERIOD_SEC:
        phase     = t_in_cycle / TILT_PITCH_PERIOD_SEC
        s         = math.sin(2.0 * math.pi * phase)
        pitch_deg = _sweep_value_mm(s, TILT_PITCH_MAX_DEG, TILT_PITCH_MIN_DEG)
        roll_deg  = 0.0
        active    = "pitch"
    else:
        phase     = (t_in_cycle - TILT_PITCH_PERIOD_SEC) / TILT_ROLL_PERIOD_SEC
        s         = math.sin(2.0 * math.pi * phase)
        pitch_deg = 0.0
        roll_deg  = _sweep_value_mm(s, TILT_ROLL_MAX_DEG, TILT_ROLL_MIN_DEG)
        active    = "roll"

    if active != _tilt_state["last_phase"]:
        if active == "pitch":
            print(f"  t={t:5.1f}s  tilt sweep -> PITCH "
                  f"[{TILT_PITCH_MIN_DEG:+d}, {TILT_PITCH_MAX_DEG:+d}] deg "
                  f"({TILT_PITCH_PERIOD_SEC:.1f}s)")
        else:
            print(f"  t={t:5.1f}s  tilt sweep -> ROLL  "
                  f"[{TILT_ROLL_MIN_DEG:+d}, {TILT_ROLL_MAX_DEG:+d}] deg "
                  f"({TILT_ROLL_PERIOD_SEC:.1f}s)")
        _tilt_state["last_phase"] = active

    return math.radians(pitch_deg), math.radians(roll_deg)


def current_shift(t):
    """Return (sx_m, sy_m) at time t for manual modes. Sequential sine
    sweeps: shift_x first, then shift_y, then loop. Static when
    CYCLE_SHIFT is False."""
    if not CYCLE_SHIFT:
        return SHIFT_FIXED_X_MM * 0.001, SHIFT_FIXED_Y_MM * 0.001

    total = SHIFT_X_PERIOD_SEC + SHIFT_Y_PERIOD_SEC
    t_in_cycle = t % total
    if t_in_cycle < SHIFT_X_PERIOD_SEC:
        s     = math.sin(2.0 * math.pi * t_in_cycle / SHIFT_X_PERIOD_SEC)
        sx_mm = _sweep_value_mm(s, SHIFT_X_MAX_MM, SHIFT_X_MIN_MM)
        sy_mm = 0.0
    else:
        s     = math.sin(2.0 * math.pi *
                         (t_in_cycle - SHIFT_X_PERIOD_SEC) / SHIFT_Y_PERIOD_SEC)
        sx_mm = 0.0
        sy_mm = _sweep_value_mm(s, SHIFT_Y_MAX_MM, SHIFT_Y_MIN_MM)
    return sx_mm * 0.001, sy_mm * 0.001


def apply_shift(rest_coxa_array, sx, sy):
    """Translate the body by (sx, sy) in body frame while leaving each
    foot's WORLD position unchanged. From each leg's perspective the foot
    appears to move by (-sx, -sy) in body frame; expressed in coxa-local
    that's a per-leg rotation by the leg's body-frame yaw. Vectorized.
    """
    if sx == 0.0 and sy == 0.0:
        return rest_coxa_array
    out = rest_coxa_array.copy()
    c = np.cos(LEG_BODY_YAW)        # (6,)
    s = np.sin(LEG_BODY_YAW)        # (6,)
    out[:, 0] += -(c * sx + s * sy)
    out[:, 1] +=  (s * sx - c * sy)
    return out


def apply_tilt(rest_coxa_array, pitch, roll):
    """Apply body tilt (pitch around body Y, roll around body X).
    Each foot's body-frame position gets rotated by R^-1 so the feet stay
    planted in world while the body tilts. Returns new (6,3) coxa-local rest.
    """
    if pitch == 0.0 and roll == 0.0:
        return rest_coxa_array

    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll),  math.sin(roll)
    # Convention: pitch+ = nose up (+X rotates toward +Z),
    #             roll+  = body's left side up (+Y rotates toward +Z).
    # R_body_world = R_y(-pitch) @ R_x(roll). Then R_inv = R^T:
    R_inv = np.array([
        [ cp,        0.0,     sp     ],
        [-sp * sr,   cr,      cp * sr],
        [-sp * cr,  -sr,      cp * cr],
    ])

    out = np.zeros_like(rest_coxa_array)
    for i in range(6):
        rest_body   = coxa_local_to_body(rest_coxa_array[i], i)
        tilted_body = R_inv @ rest_body
        out[i]      = body_to_coxa_local(tilted_body, i)
    return out


def current_spin_sign(t):
    """Return the spin direction (+1, 0, -1) at sim time t from SPIN_CYCLE."""
    cycle_total = sum(d for _, d in SPIN_CYCLE)
    t_in_cycle  = t % cycle_total
    cumulative  = 0.0
    for idx, (sign, dur) in enumerate(SPIN_CYCLE):
        if t_in_cycle < cumulative + dur:
            if idx != _spin_state["last_idx"]:
                tag = "CCW (left)" if sign > 0 else ("CW (right)" if sign < 0 else "stop")
                print(f"  t={t:5.1f}s  spin -> {tag} for {dur:.1f}s")
                _spin_state["last_idx"] = idx
            return float(sign)
        cumulative += dur
    return 0.0


def spin_foot_target_coxa(leg_idx, foot_rest, px, pz, spin_sign):
    """On-the-fly spin foot target.
    Rotates the foot's XY position (relative to body origin in coxa-local) by
    dtheta = spin_sign * px / SPIN_REF_RADIUS around the vertical axis through
    body origin. Returns the new foot position in coxa-local.
    """
    cx, cy = BODY_ORIGIN_COXA[leg_idx, 0], BODY_ORIGIN_COXA[leg_idx, 1]
    rx     = foot_rest[0] - cx
    ry     = foot_rest[1] - cy
    dtheta = spin_sign * px / SPIN_REF_RADIUS
    c, s   = math.cos(dtheta), math.sin(dtheta)
    new_x  = cx + (c * rx - s * ry)
    new_y  = cy + (s * rx + c * ry)
    new_z  = foot_rest[2] + pz       # lift component (same for translation/spin)
    return np.array([new_x, new_y, new_z])


def current_heading(t):
    """Return the body-frame heading (rad) at sim time t.
    Either cycles through HEADING_CYCLE_DEG (with HEADING_DWELL_SEC dwell each)
    or returns the fixed HEADING_FIXED_DEG.
    """
    if not CYCLE_HEADINGS:
        return math.radians(HEADING_FIXED_DEG)
    idx = int(t // HEADING_DWELL_SEC) % len(HEADING_CYCLE_DEG)
    deg = HEADING_CYCLE_DEG[idx]
    if idx != _heading_state["last_idx"]:
        print(f"  t={t:5.1f}s  heading -> {deg:+4d} deg")
        _heading_state["last_idx"] = idx
    return math.radians(deg)


def rotate_delta_xy(delta, theta):
    """Rotate a 3D coxa-local delta by angle theta in the XY plane."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        c * delta[0] - s * delta[1],
        s * delta[0] + c * delta[1],
        delta[2],
    ])


# Per-mode joint helpers. Each takes a pre-computed effective rest (so stance
# modulation is already baked in) and returns 18 joint angles.
def _joints_stand(rest):
    angles = GAIT_NEUTRAL_POSE.copy()
    for i in range(6):
        angles[i*3:i*3+3] = ik_mjcf(*rest[i], i)
    return angles


def _joints_walk(t, rest, heading_rad, period, path_scale):
    """Tripod gait, all six legs, with the given body-frame heading."""
    angles = GAIT_NEUTRAL_POSE.copy()
    s_global = gait_phase(t, period)
    dtheta = heading_rad - DEFAULT_HEADING
    for i in range(6):
        s_i = (s_global + LEG_PHASE[i]) % 1.0
        n_idx = int(s_i * PATH_RES) % PATH_RES
        delta = rotate_delta_xy(LEG_PATH_DELTAS[i, n_idx], dtheta) * path_scale
        foot_coxa = rest[i] + delta
        angles[i*3:i*3+3] = ik_mjcf(*foot_coxa, i)
    return angles


def _joints_spin(t, rest, spin_sign, period, path_scale):
    angles = GAIT_NEUTRAL_POSE.copy()
    s_global = gait_phase(t, period)
    for i in range(6):
        s_i = (s_global + LEG_PHASE[i]) % 1.0
        px, pz = canonical_path(s_i)
        px *= path_scale
        pz *= path_scale
        foot_coxa = spin_foot_target_coxa(i, rest[i], px, pz, spin_sign)
        angles[i*3:i*3+3] = ik_mjcf(*foot_coxa, i)
    return angles


def _joints_single_leg(t, rest, period, path_scale):
    """TEST_LEG only, walking in current_heading(t); other legs hold rest pose."""
    angles = GAIT_NEUTRAL_POSE.copy()
    for i in range(6):
        angles[i*3:i*3+3] = ik_mjcf(*rest[i], i)
    s = gait_phase(t, period)
    n_idx = int(s * PATH_RES) % PATH_RES
    dtheta = (current_heading(t) if MODE == "single" else 0.0) - DEFAULT_HEADING
    delta = rotate_delta_xy(LEG_PATH_DELTAS[TEST_LEG, n_idx], dtheta) * path_scale
    foot_coxa = rest[TEST_LEG] + delta
    angles[TEST_LEG*3:TEST_LEG*3+3] = ik_mjcf(*foot_coxa, TEST_LEG)
    return angles


# Smart-mode internal state.
_smart_state = {"last_idx": -1, "phase_starts": None}


def _smart_phase_starts():
    if _smart_state["phase_starts"] is None:
        starts = [0.0]
        for _, _, dur, _ in SMART_TEST_SCRIPT:
            starts.append(starts[-1] + dur)
        _smart_state["phase_starts"] = starts
    return _smart_state["phase_starts"]


def _stance_overlay_for(t_in_phase, stance_kind):
    """Returns (width_m, raise_m). stance_kind in {None, 'width', 'raise'}."""
    if stance_kind == "width":
        s = math.sin(2.0 * math.pi * t_in_phase / STANCE_WIDTH_PERIOD_SEC)
        return _sweep_value_mm(s, STANCE_WIDTH_MAX_MM, STANCE_WIDTH_MIN_MM) * 1e-3, 0.0
    if stance_kind == "raise":
        s = math.sin(2.0 * math.pi * t_in_phase / STANCE_RAISE_PERIOD_SEC)
        return 0.0, _sweep_value_mm(s, STANCE_RAISE_MAX_MM, STANCE_RAISE_MIN_MM) * 1e-3
    return 0.0, 0.0


def smart_compute(t):
    starts = _smart_phase_starts()
    total  = starts[-1]
    t_in_cycle = t % total
    idx = 0
    for i in range(len(SMART_TEST_SCRIPT)):
        if starts[i] <= t_in_cycle < starts[i + 1]:
            idx = i
            break
    label, behavior, dur, params = SMART_TEST_SCRIPT[idx]
    t_in_phase = t_in_cycle - starts[idx]

    if idx != _smart_state["last_idx"]:
        print(f"  t={t:6.1f}s  smart {idx+1:2d}/{len(SMART_TEST_SCRIPT)}: {label}  ({dur:.1f}s)")
        _smart_state["last_idx"] = idx

    width_m, raise_m = _stance_overlay_for(t_in_phase, params.get("stance"))
    rest = effective_rest(width_m, raise_m)
    # Tilt overlay. "tilt" param: "pitch" | "roll" | "circle" -> sweeps;
    # else honor explicit "tilt_pitch_deg" / "tilt_roll_deg" if present.
    tilt_mode = params.get("tilt")
    if tilt_mode == "pitch":
        p_deg = _sweep_value_mm(math.sin(2.0 * math.pi * t_in_phase / dur), +10, -10)
        r_deg = 0.0
    elif tilt_mode == "roll":
        p_deg = 0.0
        r_deg = _sweep_value_mm(math.sin(2.0 * math.pi * t_in_phase / dur), +15, -15)
    elif tilt_mode == "circle":
        ang   = 2.0 * math.pi * t_in_phase / dur
        p_deg = 8.0 * math.cos(ang)
        r_deg = 8.0 * math.sin(ang)
    else:
        p_deg = float(params.get("tilt_pitch_deg", 0.0))
        r_deg = float(params.get("tilt_roll_deg",  0.0))
    if p_deg != 0.0 or r_deg != 0.0:
        rest = apply_tilt(rest, math.radians(p_deg), math.radians(r_deg))

    # Body-shift overlay. "shift" param: "x" | "y" | "circle" -> sweeps;
    # else honor explicit "shift_x_mm" / "shift_y_mm" if present.
    shift_mode = params.get("shift")
    if shift_mode == "x":
        sx_mm = _sweep_value_mm(math.sin(2.0 * math.pi * t_in_phase / dur), +30, -30)
        sy_mm = 0.0
    elif shift_mode == "y":
        sx_mm = 0.0
        sy_mm = _sweep_value_mm(math.sin(2.0 * math.pi * t_in_phase / dur), +30, -30)
    elif shift_mode == "circle":
        ang   = 2.0 * math.pi * t_in_phase / dur
        sx_mm = 25.0 * math.cos(ang)
        sy_mm = 25.0 * math.sin(ang)
    else:
        sx_mm = float(params.get("shift_x_mm", 0.0))
        sy_mm = float(params.get("shift_y_mm", 0.0))
    if sx_mm != 0.0 or sy_mm != 0.0:
        rest = apply_shift(rest, sx_mm * 0.001, sy_mm * 0.001)

    # Per-phase speed: explicit factor, sine sweep across the phase, or 0.
    if params.get("speed_sweep"):
        speed_factor = math.sin(2.0 * math.pi * t_in_phase / dur)
    elif "speed_factor" in params:
        speed_factor = float(params["speed_factor"])
    else:
        speed_factor = 0.0
    period     = speed_period(speed_factor)
    path_scale = speed_path_scale(speed_factor)

    if behavior == "stand":
        return _joints_stand(rest)
    if behavior == "walk":
        if params.get("heading_sweep"):
            heading_rad = 2.0 * math.pi * (t_in_phase / dur)
        else:
            heading_rad = math.radians(params.get("heading_deg", 0.0))
        return _joints_walk(t, rest, heading_rad, period, path_scale)
    if behavior == "spin":
        return _joints_spin(t, rest, params.get("spin_sign", 0), period, path_scale)
    return GAIT_NEUTRAL_POSE.copy()


def compute_joints(t):
    if MODE == "smart":
        return smart_compute(t)

    # Live stance + tilt + shift + speed modulation (manual modes).
    width_m, raise_m = current_stance(t)
    rest             = effective_rest(width_m, raise_m)
    pitch_rad, roll_rad = current_tilt(t)
    rest             = apply_tilt(rest, pitch_rad, roll_rad)
    sx_m, sy_m       = current_shift(t)
    rest             = apply_shift(rest, sx_m, sy_m)
    speed_factor     = current_speed_factor(t)
    period           = speed_period(speed_factor)
    path_scale       = speed_path_scale(speed_factor)

    if MODE == "stand":
        return _joints_stand(rest)
    if MODE == "spin":
        return _joints_spin(t, rest, current_spin_sign(t), period, path_scale)
    if MODE == "single":
        return _joints_single_leg(t, rest, period, path_scale)

    # Default: tripod walk with cyclable heading.
    heading_rad = current_heading(t)
    return _joints_walk(t, rest, heading_rad, period, path_scale)


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    # Spawn body at standing height; settle for a moment under gravity so feet
    # land cleanly before the gait starts.
    data.qpos[2]    = 0.18
    data.qpos[3]    = 1.0
    data.qpos[7:25] = GAIT_NEUTRAL_POSE
    data.ctrl[:]    = GAIT_NEUTRAL_POSE
    mujoco.mj_forward(model, data)
    for _ in range(200):
        data.ctrl[:] = GAIT_NEUTRAL_POSE
        mujoco.mj_step(model, data)

    if MODE == "single":
        descr = f"single leg = {LEG_NAMES[TEST_LEG]}"
    elif MODE == "spin":
        descr = "spin (body rotates around +Z; tripod gait)"
    elif MODE == "stand":
        descr = "stand (no gait; stance modulation only)"
    elif MODE == "smart":
        descr = "smart (full scripted exercise; all behaviors + overlays)"
    else:
        descr = "tripod translate (legs 0,2,4 in phase; legs 1,3,5 180 deg offset)"
    print(f"\nIK_gait | MODE={descr}")
    print(f"  GAIT_PERIOD = {GAIT_PERIOD}s   PATH_RADIUS = {PATH_RADIUS}m   "
          f"PATH_RES = {PATH_RES}")
    if MODE == "smart":
        total = sum(d for _, _, d, _ in SMART_TEST_SCRIPT)
        print(f"  smart script: {len(SMART_TEST_SCRIPT)} phases, total {total:.1f}s per loop")
        print(f"  stance overlay: width [{STANCE_WIDTH_MIN_MM:+d}, {STANCE_WIDTH_MAX_MM:+d}] mm  "
              f"raise [{STANCE_RAISE_MIN_MM:+d}, {STANCE_RAISE_MAX_MM:+d}] mm")
        print(f"  speed envelope: period x[{1-SPEED_PERIOD_RANGE_PCT:.2f}, {1+SPEED_PERIOD_RANGE_PCT:.2f}]  "
              f"path x[{1-SPEED_PATH_RANGE_PCT:.2f}, {1+SPEED_PATH_RANGE_PCT:.2f}]")
    elif MODE == "spin":
        print(f"  spin cycle: {SPIN_CYCLE}")
    elif MODE in ("tripod", "single"):
        if CYCLE_HEADINGS:
            print(f"  cycle headings (deg): {HEADING_CYCLE_DEG}  dwell={HEADING_DWELL_SEC}s/each")
        else:
            print(f"  fixed heading: {HEADING_FIXED_DEG:+.0f} deg")
    if MODE != "smart" and CYCLE_STANCE:
        print(f"  stance sweep:")
        print(f"     width  [{STANCE_WIDTH_MIN_MM:+d}, {STANCE_WIDTH_MAX_MM:+d}] mm over {STANCE_WIDTH_PERIOD_SEC:.1f}s")
        print(f"     raise  [{STANCE_RAISE_MIN_MM:+d}, {STANCE_RAISE_MAX_MM:+d}] mm over {STANCE_RAISE_PERIOD_SEC:.1f}s")
    if MODE != "smart" and CYCLE_TILT:
        print(f"  tilt sweep:")
        print(f"     pitch [{TILT_PITCH_MIN_DEG:+d}, {TILT_PITCH_MAX_DEG:+d}] deg over {TILT_PITCH_PERIOD_SEC:.1f}s")
        print(f"     roll  [{TILT_ROLL_MIN_DEG:+d}, {TILT_ROLL_MAX_DEG:+d}] deg over {TILT_ROLL_PERIOD_SEC:.1f}s")
    if MODE != "smart" and CYCLE_SPEED:
        period_lo = speed_period(+1.0)
        period_hi = speed_period(-1.0)
        path_lo   = speed_path_scale(-1.0)
        path_hi   = speed_path_scale(+1.0)
        speed_lo  = (path_lo / period_hi) * GAIT_PERIOD
        speed_hi  = (path_hi / period_lo) * GAIT_PERIOD
        print(f"  speed sweep -1 <-> +1 over {SPEED_SWEEP_PERIOD_SEC:.1f}s")
        print(f"     period range: [{period_lo:.2f}, {period_hi:.2f}] s")
        print(f"     path   range: [x{path_lo:.2f}, x{path_hi:.2f}]")
        print(f"     speed  range: [x{speed_lo:.2f}, x{speed_hi:.2f}] (relative to nominal)")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_t0  = data.time
        wall_t0 = time.time()
        while viewer.is_running():
            t = data.time - sim_t0
            data.ctrl[:] = compute_joints(t)
            mujoco.mj_step(model, data)
            viewer.sync()
            lag = (wall_t0 + (data.time - sim_t0)) - time.time()
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
