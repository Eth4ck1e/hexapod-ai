"""
tools/gen_mini_hexapod_mjcf.py — generate the mini hexapod MJCFs from one
dimension table.

Emits two files that must stay in sync (same kinematics, masses, actuators):

    models/mini_hexapod_mjx.xml     training — Euler, feet↔floor contacts only
    models/mini_hexapod_simple.xml  full self-collision — for
                                    tools/derive_joint_limits.py and viewing

Run after editing DIMS / SERVO / MASS below:

    python tools/gen_mini_hexapod_mjcf.py

Frame conventions (deliberately simpler than the PhantomX mesh model):

    coxa  body  at the coxa pivot, yawed by the leg's mount angle.
                local +x points radially outward, +z up. Joint axis +z.
    femur body  at (coxa_length, 0, 0). Joint axis -y, so a POSITIVE femur
                angle lifts the femur (matches the closed-form IK's raw
                femur angle, so joint sign = +1).
    tibia body  at (femur_length, 0, 0). Joint axis +y, so a POSITIVE tibia
                angle bends the foot DOWN from straight extension (matches
                the IK's raw tibia angle = π - knee interior angle, sign +1).
    foot  geom  sphere at (tibia_length, 0, 0) tibia-local.

    With all 18 joints at zero every leg sticks straight out horizontally.
    The gait anchor pose (RobotProfile.neutral_pose) is the firmware's
    L-pose: femur 0, tibia +π/2 → foot at (coxa_length + femur_length)
    horizontal from the coxa pivot, tibia_length below it.

Everything marked PLACEHOLDER is a guess pending measurement of the real
bot (user has no access until ~mid-Nov 2026). See CLAUDE.md "Mini hexapod".
"""
from __future__ import annotations

import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------------------
# Dimension table (metres). Source: mini_hexapod/firmware/arduino/
# hexapod_firmware.ino DEFAULT_* (measured 2026-05-17) unless noted.
# ----------------------------------------------------------------------------
DIMS = {
    # Coxa pivots in body frame, leg order RR, RM, RF, LR, LM, LF (library
    # order — firmware uses FR, MR, RR, FL, ML, RL; values re-ordered here).
    "coxa_pos": {
        "RR": (-0.05675, -0.0320, 0.0),
        "RM": ( 0.00000, -0.0424, 0.0),
        "RF": ( 0.05675, -0.0320, 0.0),
        "LR": (-0.05675, +0.0320, 0.0),
        "LM": ( 0.00000, +0.0424, 0.0),
        "LF": ( 0.05675, +0.0320, 0.0),
    },
    "coxa_yaw_deg": {"RR": -135, "RM": -90, "RF": -45, "LR": 135, "LM": 90, "LF": 45},
    # PLACEHOLDER: horizontal offset coxa yaw axis → femur pitch axis. The
    # firmware IK assumes 0 and the bot walked, but a real servo bracket
    # always has some offset. Measure and update; femur_length may shrink
    # by the same amount if the 55 mm was measured from the coxa axis.
    "coxa_length":  0.0,
    "femur_length": 0.0550,     # measured: coxa joint → knee
    "tibia_length": 0.0895,     # measured: knee → foot tip
    # PLACEHOLDER: chassis box half-sizes (must enclose the coxa pivots).
    "chassis_half": (0.068, 0.040, 0.012),
    # PLACEHOLDER: foot contact sphere radius (rubber tip / bare plastic?).
    "foot_radius":  0.004,
    # PLACEHOLDER: visual capsule radii.
    "coxa_r":  0.009,
    "femur_r": 0.007,
    "tibia_r": 0.005,
}

# ----------------------------------------------------------------------------
# Masses (kg). ALL PLACEHOLDER — weigh the bot. 18 × 9 g servos = 0.162 kg
# alone; chassis + XIAO + 2× PCA9685 + battery guessed at ~0.2 kg.
# Geoms carry mass so MuJoCo derives inertia from the shape; no hand-typed
# diaginertia to get wrong.
# ----------------------------------------------------------------------------
MASS = {
    "chassis": 0.200,
    "coxa":    0.012,   # coxa servo + horn
    "femur":   0.015,   # femur servo + bracket
    "tibia":   0.012,   # tibia servo + leg piece
}

# ----------------------------------------------------------------------------
# Servo model — Miuzei MS18 9 g micro servo, datasheet-typical figures.
# PLACEHOLDER until confirmed against the actual part / supply voltage:
#   stall torque  ≈ 1.8 kg·cm @ 4.8 V  = 0.176 N·m   (2.2 kg·cm @ 6 V = 0.216)
#   no-load speed ≈ 0.10 s / 60° @ 4.8 V ≈ 10.5 rad/s
# kp chosen so ~1.5° of error saturates torque (same ratio as the PhantomX
# model: 1.5 N·m / kp 60). kv keeps the PhantomX kv/kp ratio.
# ----------------------------------------------------------------------------
SERVO = {
    "stall_torque": 0.18,       # N·m
    "noload_speed": 10.5,       # rad/s
    "kp":  7.0,
    "kv":  0.25,
    # PLACEHOLDER joint ranges (rad) — a 180° hobby servo minus bracket
    # clearance. Replace with tools/derive_joint_limits.py output.
    "coxa_range":  (-1.20, 1.20),
    "femur_range": (-1.40, 1.40),
    "tibia_range": (-0.30, 2.40),     # mostly bends down from straight
    # Reflected rotor inertia / damping scaled from the AX-12A values
    # (0.01 / 0.3 / 0.01) by roughly the torque ratio.
    "armature":     0.0015,
    "damping":      0.04,
    "frictionloss": 0.005,
}

# Spawn height: body z such that L-pose feet just touch the floor, + 15 mm.
SPAWN_Z = DIMS["tibia_length"] + DIMS["foot_radius"] + 0.015

LEG_ORDER = ["RR", "RM", "RF", "LR", "LM", "LF"]


def _quat_z(deg: float) -> str:
    h = math.radians(deg) / 2.0
    return f"{math.cos(h):.6f} 0 0 {math.sin(h):.6f}"


def _leg_xml(name: str, mjx: bool) -> str:
    d = DIMS
    px, py, pz = d["coxa_pos"][name]
    L1, L2, L3 = d["coxa_length"], d["femur_length"], d["tibia_length"]
    foot_col = ' contype="2" conaffinity="1"' if mjx else ""
    # Coxa visual: a capsule needs non-zero length; use a sphere when L1 == 0.
    if L1 > 1e-6:
        coxa_geom = (f'<geom name="coxa_geom_{name}" type="capsule" '
                     f'fromto="0 0 0  {L1:.4f} 0 0" size="{d["coxa_r"]}" '
                     f'mass="{MASS["coxa"]}" rgba="0.2 0.5 0.9 1"/>')
    else:
        coxa_geom = (f'<geom name="coxa_geom_{name}" type="sphere" '
                     f'size="{d["coxa_r"]}" mass="{MASS["coxa"]}" rgba="0.2 0.5 0.9 1"/>')
    cr, fr, tr = SERVO["coxa_range"], SERVO["femur_range"], SERVO["tibia_range"]
    return f"""
      <!-- {name} -->
      <body name="coxa_{name}" pos="{px:.5f} {py:.5f} {pz:.5f}" quat="{_quat_z(d['coxa_yaw_deg'][name])}">
        <joint name="coxa_joint_{name}" axis="0 0 1" range="{cr[0]:.4f} {cr[1]:.4f}"/>
        {coxa_geom}
        <body name="femur_{name}" pos="{L1:.4f} 0 0">
          <joint name="femur_joint_{name}" axis="0 -1 0" range="{fr[0]:.4f} {fr[1]:.4f}"/>
          <geom name="femur_geom_{name}" type="capsule" fromto="0 0 0  {L2:.4f} 0 0" size="{d['femur_r']}" mass="{MASS['femur']}" rgba="0.2 0.5 0.9 1"/>
          <body name="tibia_{name}" pos="{L2:.4f} 0 0">
            <joint name="tibia_joint_{name}" axis="0 1 0" range="{tr[0]:.4f} {tr[1]:.4f}"/>
            <geom name="tibia_geom_{name}" type="capsule" fromto="0 0 0  {L3:.4f} 0 0" size="{d['tibia_r']}" mass="{MASS['tibia']}" rgba="0.2 0.5 0.9 1"/>
            <geom name="foot_{name}" type="sphere" pos="{L3:.4f} 0 0" size="{d['foot_radius']}" mass="0.001" rgba="0.1 0.1 0.1 1"{foot_col}/>
          </body>
        </body>
      </body>"""


def _actuators_xml() -> str:
    cr, fr, tr = SERVO["coxa_range"], SERVO["femur_range"], SERVO["tibia_range"]
    rows = []
    for n in LEG_ORDER:
        rows.append(f'    <position name="coxa_{n}"  joint="coxa_joint_{n}"  ctrlrange="{cr[0]:.4f} {cr[1]:.4f}"/>')
        rows.append(f'    <position name="femur_{n}" joint="femur_joint_{n}" ctrlrange="{fr[0]:.4f} {fr[1]:.4f}"/>')
        rows.append(f'    <position name="tibia_{n}" joint="tibia_joint_{n}" ctrlrange="{tr[0]:.4f} {tr[1]:.4f}"/>')
    return "\n".join(rows)


def build(mjx: bool) -> str:
    d, sv = DIMS, SERVO
    model_name = "mini_hexapod_mjx" if mjx else "mini_hexapod_simple"
    if mjx:
        option = ('<option timestep="0.005" gravity="0 0 -9.81" integrator="Euler"\n'
                  '          iterations="20" ls_iterations="10"/>')
        geom_default = ('<geom condim="3" friction="1.5 0.05 0.05"\n'
                        '          contype="0" conaffinity="0"/>   <!-- default: no collision -->')
        floor = ('<geom name="floor" type="plane" size="10 10 0.1" material="floor" condim="3"\n'
                 '          contype="1" conaffinity="2"/>')
        variant_note = ("MJX training variant: Euler integrator, solver caps, and collision\n"
                        "    pairs trimmed to feet ↔ floor only (same scheme as phantomx_simple_mjx.xml).")
    else:
        option = '<option timestep="0.005" gravity="0 0 -9.81" integrator="RK4"/>'
        geom_default = '<geom condim="3" friction="1.5 0.05 0.05"/>'
        floor = '<geom name="floor" type="plane" size="10 10 0.1" material="floor" condim="3"/>'
        variant_note = ("Full self-collision variant for tools/derive_joint_limits.py and the\n"
                        "    mujoco viewer. NOT for MJX training.")
    hx, hy, hz = d["chassis_half"]
    legs = "".join(_leg_xml(n, mjx) for n in LEG_ORDER)
    return f"""<mujoco model="{model_name}">
  <!--
    GENERATED by tools/gen_mini_hexapod_mjcf.py — edit that file, not this one.

    {variant_note}

    Geometry: femur {d['femur_length']*1e3:.1f} mm, tibia {d['tibia_length']*1e3:.1f} mm,
    coxa offset {d['coxa_length']*1e3:.1f} mm (PLACEHOLDER), corner coxas at
    (±{abs(d['coxa_pos']['RF'][0])*1e3:.2f}, ±{abs(d['coxa_pos']['RF'][1])*1e3:.1f}) mm @ ±45°,
    middle coxas at (0, ±{abs(d['coxa_pos']['RM'][1])*1e3:.1f}) mm @ ±90°.
    Servos: Miuzei MS18 9 g, stall {sv['stall_torque']} N·m, no-load {sv['noload_speed']} rad/s
    (PLACEHOLDER datasheet-typical). Masses PLACEHOLDER. See generator docstring.

    Joint axes: coxa +z, femur -y, tibia +y → all closed-form-IK signs are +1.
    All joints at 0 = legs straight out horizontally. L-pose = femur 0, tibia +π/2.
  -->

  <compiler angle="radian"/>

  {option}

  <default>
    <joint damping="{sv['damping']}" armature="{sv['armature']}" frictionloss="{sv['frictionloss']}"/>
    {geom_default}
    <position kp="{sv['kp']}" kv="{sv['kv']}" ctrllimited="true" ctrlrange="-3.14 3.14"
              forcelimited="true" forcerange="-{sv['stall_torque']} {sv['stall_torque']}"/>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.5 0.7 0.9" rgb2="1 1 1" width="512" height="512"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.4 0.4 0.4" rgb2="0.5 0.5 0.5" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="4 4" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" pos="0 0 4" dir="0 0 -1"/>
    {floor}

    <body name="base" pos="0 0 {SPAWN_Z:.4f}">
      <freejoint/>
      <geom name="chassis" type="box" size="{hx} {hy} {hz}" mass="{MASS['chassis']}" rgba="0.2 0.2 0.2 1"/>
      <site name="imu" pos="0 0 0" size="0.004" rgba="0 1 0 0.5"/>
{legs}

    </body>
  </worldbody>

  <actuator>
{_actuators_xml()}
  </actuator>

  <sensor>
    <accelerometer site="imu" name="imu_acc"/>
    <gyro          site="imu" name="imu_gyro"/>
    <framequat     objtype="site" objname="imu" name="imu_quat"/>
  </sensor>

</mujoco>
"""


def main():
    out_dir = PROJECT_ROOT / "models"
    for mjx, fname in ((True, "mini_hexapod_mjx.xml"), (False, "mini_hexapod_simple.xml")):
        path = out_dir / fname
        path.write_text(build(mjx), encoding="utf-8")
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
