"""
simple_gait.py — parallel-plane gait scaffold

Concept (per design plan):
  1. Define ONE path in the body XZ plane (Y=0), centered at origin (0,0,0).
     Path: half-circle on top + flat line on bottom.
        - SWING  (top arc):    foot lifts and arcs from (-R, 0) over (0, +R) to (+R, 0)
        - STANCE (flat line):  foot drags backward at z=0 from (+R, 0) to (-R, 0)
  2. Sample (x, z) along the path at the per-tick phase s ∈ [0, 1).
  3. For each leg, translate that (x, z) sample to a parallel plane offset by
     the leg's default foot position:  foot_body[i] = LEG_ORIGIN[i] + (x, 0, z).
  4. Convert body-frame foot targets to joint angles per leg.

No heading rotation, no per-leg phase offset yet — this file just verifies the
parallel-plane concept produces correct body-frame motion.

Run:
    mjpython simple_gait.py
"""

import math
import time
import numpy as np
import mujoco
import mujoco.viewer

from envs.hexapod_env import _ik, NEUTRAL_POSE

MODEL_PATH = "models/phantomx.xml"
LEG_NAMES  = ["RR", "RM", "RF", "LR", "LM", "LF"]
MJCF_SIGN  = np.array([+1, +1, +1, -1, -1, -1])

# IK input at NEUTRAL_POSE for each leg (matches envs/hexapod_env.py).
from envs.hexapod_env import INIT_FOOT_POS_X, INIT_FOOT_POS_Y, INIT_FOOT_POS_Z
IK_NEUTRAL_INPUT = np.array([
    [-INIT_FOOT_POS_X[i], INIT_FOOT_POS_Y[i], INIT_FOOT_POS_Z[i]]
    for i in range(6)
])

# ============================================================================
# PATH — half-circle top + flat bottom, centered at origin in body XZ plane
# ============================================================================
PATH_RADIUS = 0.025   # m — half the stride length (also the lift height)

def path(s):
    """Return (x, z) on body XZ plane, centered at origin.
    s in [0, 0.5) → SWING : arc from (-R, 0) over (0, +R) to (+R, 0)
    s in [0.5, 1) → STANCE: flat from (+R, 0) back to (-R, 0)
    """
    if s < 0.5:
        theta = math.pi * (1.0 - 2.0 * s)             # pi → 0
        return PATH_RADIUS * math.cos(theta), PATH_RADIUS * math.sin(theta)
    ss = (s - 0.5) / 0.5
    return PATH_RADIUS * (1.0 - 2.0 * ss), 0.0


def sample_path(resolution):
    """Return (N, 2) array of (x, z) samples along the path."""
    return np.array([path(i / resolution) for i in range(resolution)])


# ============================================================================
# CALIBRATION — empirically determine, for each leg:
#   LEG_ORIGIN_BODY[i]    : actual foot position in body frame at NEUTRAL
#   FOOT_TIP_LOCAL[i]     : foot tip position in tibia local frame
#   R_IK_TO_BODY[i]       : 3x3 rotation matrix from IK frame deltas to body deltas
# ============================================================================

def _set_pose(model, data, joint_angles_18):
    """Pin body at world origin identity; set the 18 leg joints."""
    data.qpos[:] = 0
    data.qpos[3] = 1.0
    data.qpos[7:25] = joint_angles_18
    mujoco.mj_forward(model, data)


def _ik_with_flip(fx, fy, fz, leg_idx):
    """Run _ik and apply the MJCF sign flip for left legs."""
    c, f, t = _ik(fx, fy, fz, leg_idx)
    if leg_idx >= 3:
        c, f, t = -c, -f, -t
    return c, f, t


def _set_leg(model, data, leg_idx, fx, fy, fz, base=NEUTRAL_POSE):
    """Set NEUTRAL on every leg except `leg_idx`, which gets IK(fx, fy, fz)."""
    joints = base.copy()
    c, f, t = _ik_with_flip(fx, fy, fz, leg_idx)
    joints[leg_idx*3:leg_idx*3+3] = (c, f, t)
    _set_pose(model, data, joints)


def calibrate():
    """One-shot calibration. Returns (LEG_ORIGIN_BODY, FOOT_TIP_LOCAL, R_IK_TO_BODY).

    Foot tip in tibia local frame is determined by finding the lowest-world-Z
    vertex of the tibia mesh at NEUTRAL_POSE. All 6 tibias share the same mesh,
    so the same tip position applies to every leg.
    """
    # Find the foot tip via lowest-Z mesh vertex at NEUTRAL_POSE (one-time per leg).
    tmp_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    tmp_data  = mujoco.MjData(tmp_model)
    _set_pose(tmp_model, tmp_data, NEUTRAL_POSE)
    mesh_id = mujoco.mj_name2id(tmp_model, mujoco.mjtObj.mjOBJ_MESH, "tibia")
    v0 = tmp_model.mesh_vertadr[mesh_id]; vn = tmp_model.mesh_vertnum[mesh_id]
    mesh_verts = tmp_model.mesh_vert[v0:v0+vn].copy()
    FOOT_TIP_LOCAL = np.zeros((6, 3))
    for i, n in enumerate(LEG_NAMES):
        # Tibia geom for this leg
        for g in range(tmp_model.ngeom):
            bid = tmp_model.geom_bodyid[g]
            if (mujoco.mj_id2name(tmp_model, mujoco.mjtObj.mjOBJ_BODY, bid) == f"tibia_{n}"
                and tmp_model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH):
                gp, gm = tmp_data.geom_xpos[g], tmp_data.geom_xmat[g].reshape(3,3)
                world_verts = (gm @ mesh_verts.T).T + gp
                low = world_verts[np.argmin(world_verts[:, 2])]
                tibia_bid = mujoco.mj_name2id(tmp_model, mujoco.mjtObj.mjOBJ_BODY, f"tibia_{n}")
                tp = tmp_data.xpos[tibia_bid]; tm = tmp_data.xmat[tibia_bid].reshape(3,3)
                FOOT_TIP_LOCAL[i] = tm.T @ (low - tp)
                break

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)

    # Body pinned at world origin, identity orientation, joints at NEUTRAL_POSE.
    _set_pose(model, data, NEUTRAL_POSE)

    LEG_ORIGIN_BODY = np.zeros((6, 3))
    for i in range(6):
        tibia_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tibia_{LEG_NAMES[i]}")
        LEG_ORIGIN_BODY[i] = data.xpos[tibia_id] + data.xmat[tibia_id].reshape(3,3) @ FOOT_TIP_LOCAL[i]

    # Step 3 — empirically derive each leg's IK→body 3x3 rotation by perturbing
    # each IK axis and reading the foot displacement (using FOOT_TIP_LOCAL).
    R_IK_TO_BODY = np.zeros((6, 3, 3))
    EPS = 1e-3

    def foot_body_for(leg_idx, fx, fy, fz):
        _set_leg(model, data, leg_idx, fx, fy, fz)
        tibia_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tibia_{LEG_NAMES[leg_idx]}")
        return data.xpos[tibia_id] + data.xmat[tibia_id].reshape(3,3) @ FOOT_TIP_LOCAL[leg_idx]

    for i in range(6):
        f0 = IK_NEUTRAL_INPUT[i]
        # Foot at neutral IK input, body pinned at origin identity (so body == world)
        p0 = foot_body_for(i, *f0)
        px = foot_body_for(i, f0[0]+EPS, f0[1],     f0[2])
        py = foot_body_for(i, f0[0],     f0[1]+EPS, f0[2])
        pz = foot_body_for(i, f0[0],     f0[1],     f0[2]+EPS)
        R_IK_TO_BODY[i, :, 0] = (px - p0) / EPS
        R_IK_TO_BODY[i, :, 1] = (py - p0) / EPS
        R_IK_TO_BODY[i, :, 2] = (pz - p0) / EPS

    return LEG_ORIGIN_BODY, FOOT_TIP_LOCAL, R_IK_TO_BODY


print("Calibrating leg frames empirically...")
LEG_ORIGIN_BODY, FOOT_TIP_LOCAL, R_IK_TO_BODY = calibrate()
R_BODY_TO_IK = np.array([np.linalg.inv(R) for R in R_IK_TO_BODY])

print("\nCalibration results:")
for i in range(6):
    print(f"  {LEG_NAMES[i]}: LEG_ORIGIN_BODY={LEG_ORIGIN_BODY[i].round(4)}")

print("\nR_IK_TO_BODY[RR] (each column = IK +x/+y/+z direction in body frame):")
print(R_IK_TO_BODY[0].round(3))
print("(For a clean leg this should be a near-rotation, columns ≈ unit length, mutually orthogonal.)")


# ============================================================================
# BODY-FRAME FOOT TARGET → JOINT ANGLES
# ============================================================================

# Scratch model + data used for FK+Jacobian-based body-frame IK.
# This bypasses envs/hexapod_env._ik (which is geometrically inconsistent with
# this MJCF — it was ported from another project's link conventions). We use
# mujoco.mj_jac on the actual MJCF kinematics, which is guaranteed correct.
_FK_MODEL  = mujoco.MjModel.from_xml_path(MODEL_PATH)
_FK_DATA   = mujoco.MjData(_FK_MODEL)
_TIBIA_BID = np.array([
    mujoco.mj_name2id(_FK_MODEL, mujoco.mjtObj.mjOBJ_BODY, f"tibia_{n}") for n in LEG_NAMES
])
# qvel slot for each leg's three joints (coxa, femur, tibia). Layout:
# qvel[0:6] is the freejoint, then 3 dof per leg in MJCF order.
_LEG_QVEL_SLOTS = [(6 + i*3, 6 + i*3 + 3) for i in range(6)]


def _fk_foot_world(joints_18, leg_idx):
    """Set joints, run FK, return (foot_world_pos, tibia_xmat)."""
    _set_pose(_FK_MODEL, _FK_DATA, joints_18)
    bid  = _TIBIA_BID[leg_idx]
    pos  = _FK_DATA.xpos[bid]
    xmat = _FK_DATA.xmat[bid].reshape(3, 3)
    return pos + xmat @ FOOT_TIP_LOCAL[leg_idx], xmat


def _foot_jacobian(leg_idx):
    """3×3 Jacobian of foot world position w.r.t. this leg's 3 joint angles.
    Caller must have already set joints + mj_forward'd _FK_DATA.
    """
    bid = _TIBIA_BID[leg_idx]
    # mj_jac wants the foot point in WORLD coords (despite docs sometimes saying body).
    # Workaround: compute the world point ourselves and pass it.
    pos = _FK_DATA.xpos[bid] + _FK_DATA.xmat[bid].reshape(3, 3) @ FOOT_TIP_LOCAL[leg_idx]
    jacp = np.zeros((3, _FK_MODEL.nv))
    jacr = np.zeros((3, _FK_MODEL.nv))
    mujoco.mj_jac(_FK_MODEL, _FK_DATA, jacp, jacr, pos, bid)
    qs = _LEG_QVEL_SLOTS[leg_idx]
    return jacp[:, qs[0]:qs[1]]


def body_to_joints(feet_body, max_iter=20, tol=1e-7, max_step=0.3):
    """(6, 3) body-frame foot targets → (18,) joint angles.

    Per-leg Jacobian iteration on the actual MJCF kinematics: compute current
    foot position via FK, the 3×3 Jacobian via mj_jac, take a damped Newton step
    in joint space, repeat. Step is clamped to max_step rad to keep the iteration
    in the linear regime. Body is pinned at world origin so world == body frame.
    """
    angles = NEUTRAL_POSE.copy()  # warm start
    for i in range(6):
        target_world = feet_body[i]   # body pinned at origin → world == body frame
        for _ in range(max_iter):
            actual, _ = _fk_foot_world(angles, i)
            residual = target_world - actual
            if np.linalg.norm(residual) < tol:
                break
            J = _foot_jacobian(i)
            try:
                step = np.linalg.solve(J, residual)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(J) @ residual
            # Clamp step to keep the linearization valid.
            sn = np.linalg.norm(step)
            if sn > max_step:
                step *= max_step / sn
            angles[i*3:i*3+3] += step
    return angles


# ============================================================================
# DEMO
# ============================================================================
# MODE options:
#   "single" : drive only TEST_LEG; others hold neutral.
#   "all"    : drive all six legs in lockstep (no phase offset).
#   "tripod" : tripod gait — legs {0,2,4} share phase 0; legs {1,3,5} are 180° offset.
MODE        = "tripod"
TEST_LEG    = 3       # used only when MODE == "single"
GAIT_PERIOD = 1.5     # seconds per full cycle

# Per-leg phase offset within s ∈ [0, 1). 0 = in sync; 0.5 = 180° out of phase.
# Tripod groups:  A = {0:RR, 2:RF, 4:LM}  ;  B = {1:RM, 3:LR, 5:LF}
LEG_PHASE = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])


def compute_targets(t):
    feet = LEG_ORIGIN_BODY.copy()
    s_global = (t / GAIT_PERIOD) % 1.0

    if MODE == "single":
        px, pz = path(s_global)
        feet[TEST_LEG] = feet[TEST_LEG] + np.array([px, 0.0, pz])
    elif MODE == "all":
        px, pz = path(s_global)
        for i in range(6):
            feet[i] = feet[i] + np.array([px, 0.0, pz])
    elif MODE == "tripod":
        for i in range(6):
            s_i = (s_global + LEG_PHASE[i]) % 1.0
            px, pz = path(s_i)
            feet[i] = feet[i] + np.array([px, 0.0, pz])

    return body_to_joints(feet)


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.18
    data.qpos[3] = 1.0
    data.qpos[7:25] = NEUTRAL_POSE
    data.ctrl[:]    = NEUTRAL_POSE
    mujoco.mj_forward(model, data)
    for _ in range(200):
        data.ctrl[:] = NEUTRAL_POSE
        mujoco.mj_step(model, data)

    if MODE == "single":
        descr = f"single leg = {LEG_NAMES[TEST_LEG]}"
    elif MODE == "tripod":
        descr = "tripod (legs 0,2,4 in phase; legs 1,3,5 180° offset)"
    else:
        descr = MODE
    print(f"\nsimple_gait | MODE={descr}  PATH_RADIUS={PATH_RADIUS}  PERIOD={GAIT_PERIOD}s")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_t0  = data.time
        wall_t0 = time.time()
        while viewer.is_running():
            t = data.time - sim_t0
            data.ctrl[:] = compute_targets(t)
            mujoco.mj_step(model, data)
            viewer.sync()
            lag = (wall_t0 + (data.time - sim_t0)) - time.time()
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
