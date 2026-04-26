"""
gait.controller — stateful hexapod gait controller.

Cmd vector (9 floats, all physical units):
  [0] vx           m/s     body-frame forward velocity
  [1] vy           m/s     body-frame lateral velocity (left = +)
  [2] wz           rad/s   body yaw rate (CCW = +)
  [3] pitch        rad     body pitch target (nose up = +)
  [4] roll         rad     body roll target (right side up = +)
  [5] height_delta m       stance height delta from neutral (- raises body)
  [6] width_delta  m       stance width delta from neutral
  [7] shift_x      m       body shift in body +X direction
  [8] shift_y      m       body shift in body +Y direction

All overlays compose: gait runs normally inside any combination of
stance/pose/shift adjustments.
"""

import math
import numpy as np
import mujoco


LEG_NAMES = ["RR", "RM", "RF", "LR", "LM", "LF"]

# Coxa joint positions in body frame (model constants from MJCF).
COXA_POS_BODY = np.array([
    [-0.12, -0.06, 0.0],   # 0: RR
    [ 0.00, -0.10, 0.0],   # 1: RM
    [ 0.12, -0.06, 0.0],   # 2: RF
    [-0.12, +0.06, 0.0],   # 3: LR
    [ 0.00, +0.10, 0.0],   # 4: LM
    [ 0.12, +0.06, 0.0],   # 5: LF
])

# Tripod phase offsets per leg. Group A (0,2,4) leads; Group B (1,3,5) lags 0.5.
LEG_PHASE = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])

# Calibration neutral pose — the joint configuration used to:
#   (1) Detect the foot tip via lowest-Z mesh vertex during calibrate().
#   (2) Define LEG_ORIGIN_BODY (feet at this pose are the rest positions).
#   (3) Warm-start the iterative IK each tick.
# Values originally derived from the legacy IK in envs/hexapod_env.py — that IK
# was geometrically wrong overall but produced a usable standing pose. Keep
# these hardcoded so this library has zero dependency on envs/.
NEUTRAL_POSE = np.array([
     0.0, -0.21818873, -0.32188801,    # RR (rear right)
     0.0, -0.24918906, -0.26150113,    # RM (middle right)
     0.0, -0.21818873, -0.32188801,    # RF (front right)
     0.0,  0.21818873,  0.32188801,    # LR (rear left)
     0.0,  0.24918906,  0.26150113,    # LM (middle left)
     0.0,  0.21818873,  0.32188801,    # LF (front left)
])


class Controller:
    """Stateful hexapod gait controller. Cmd → 18 joint targets.

    Lifecycle:
        ctrl = Controller("models/phantomx.xml")     # calibrates on construction
        ctrl.set_cmd(np.array([0.05, 0, 0, 0, 0, 0, 0, 0, 0]))   # walk forward
        joints = ctrl.step(dt)                       # advance time, get joints
        # OR
        joints = ctrl.predict(cmd, t)                # stateless eval

    Public attributes (read-only after calibrate()):
        LEG_ORIGIN_BODY        (6,3)  foot rest positions in body frame
        FOOT_TIP_LOCAL         (6,3)  foot tip in tibia local frame
        LEG_RADIAL_DIR_XY      (6,2)  horizontal coxa→foot unit vectors
        gait_period            float  seconds per full gait cycle
        path_radius            float  half-stride / lift height (m)
        spin_ref_radius        float  reference radius for angular sweep (m)
    """

    DEFAULT_GAIT_PERIOD     = 1.5
    DEFAULT_PATH_RADIUS     = 0.025
    DEFAULT_SPIN_REF_RADIUS = 0.204

    def __init__(self, model_path,
                 gait_period=None, path_radius=None, spin_ref_radius=None):
        self.model_path      = model_path
        self.gait_period     = gait_period     or self.DEFAULT_GAIT_PERIOD
        self.path_radius     = path_radius     or self.DEFAULT_PATH_RADIUS
        self.spin_ref_radius = spin_ref_radius or self.DEFAULT_SPIN_REF_RADIUS

        # Calibration outputs (set by calibrate()).
        self.LEG_ORIGIN_BODY    = None      # (6, 3)
        self.FOOT_TIP_LOCAL     = None      # (6, 3)
        self.LEG_RADIAL_DIR_XY  = None      # (6, 2)

        # FK scratch (private).
        self._fk_model       = None
        self._fk_data        = None
        self._tibia_bid      = None
        self._leg_qvel_slots = None

        # Working state.
        self._cmd = np.zeros(9, dtype=np.float64)
        self._t   = 0.0

        self.calibrate()

    # ----------------------------------------------------------------------
    # Derived constants — ranges that cmd[0:3] saturate at by default.
    # The policy can exceed these in principle (cmd is just a target); these
    # describe what the analytical scaffold can deliver at full path radius.
    # ----------------------------------------------------------------------
    @property
    def MAX_SPEED(self):
        """Max body linear speed (m/s) at full stride. ≈ 4·R/T."""
        return 4.0 * self.path_radius / self.gait_period

    @property
    def MAX_YAW_RATE(self):
        """Max body yaw rate (rad/s) at full angular stride. ≈ 4·Δθ_max/T."""
        return 4.0 * (self.path_radius / self.spin_ref_radius) / self.gait_period

    # ----------------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------------
    def calibrate(self):
        """Empirical leg-frame setup. Run once on construction (or manually
        if model_path / gait_period / path_radius changes)."""
        self._fk_model = mujoco.MjModel.from_xml_path(self.model_path)
        self._fk_data  = mujoco.MjData(self._fk_model)
        self._tibia_bid = np.array([
            mujoco.mj_name2id(self._fk_model, mujoco.mjtObj.mjOBJ_BODY, f"tibia_{n}")
            for n in LEG_NAMES
        ])
        # qvel layout: 6 freejoint slots + 18 leg joints (3 per leg, MJCF order).
        self._leg_qvel_slots = [(6 + i*3, 6 + i*3 + 3) for i in range(6)]

        # NEUTRAL_POSE: solve once for the joint angles that put each foot at
        # its empirically-detected rest body-frame position. Bootstrap by
        # using the model's natural rest pose (all joints = 0) to find the
        # foot tip, then hold those joints as our neutral.
        # Simpler: just use joints=0 as neutral. The lowest-mesh-vertex test
        # below works regardless of what neutral pose we pick — it just gives
        # us LEG_ORIGIN_BODY for that pose. We'll use joints=0.
        self._set_pose(NEUTRAL_POSE)

        # Foot tip in tibia local frame: lowest world-Z vertex of each tibia
        # mesh at the neutral pose. All 6 tibias share the same mesh.
        mesh_id    = mujoco.mj_name2id(self._fk_model, mujoco.mjtObj.mjOBJ_MESH, "tibia")
        v0         = self._fk_model.mesh_vertadr[mesh_id]
        vn         = self._fk_model.mesh_vertnum[mesh_id]
        mesh_verts = self._fk_model.mesh_vert[v0:v0+vn].copy()

        self.FOOT_TIP_LOCAL  = np.zeros((6, 3))
        self.LEG_ORIGIN_BODY = np.zeros((6, 3))
        for i, n in enumerate(LEG_NAMES):
            tibia_geom = self._find_tibia_geom(n)
            gp = self._fk_data.geom_xpos[tibia_geom]
            gm = self._fk_data.geom_xmat[tibia_geom].reshape(3, 3)
            world_verts = (gm @ mesh_verts.T).T + gp
            low = world_verts[np.argmin(world_verts[:, 2])]

            tibia_bid = self._tibia_bid[i]
            tp = self._fk_data.xpos[tibia_bid]
            tm = self._fk_data.xmat[tibia_bid].reshape(3, 3)
            self.FOOT_TIP_LOCAL[i]  = tm.T @ (low - tp)
            self.LEG_ORIGIN_BODY[i] = low                # body frame == world here

        # Per-leg horizontal unit vector pointing from coxa joint outward to foot.
        diff_xy = self.LEG_ORIGIN_BODY[:, :2] - COXA_POS_BODY[:, :2]
        norms   = np.linalg.norm(diff_xy, axis=1, keepdims=True)
        self.LEG_RADIAL_DIR_XY = diff_xy / norms

    def _find_tibia_geom(self, leg_name):
        for g in range(self._fk_model.ngeom):
            bid   = self._fk_model.geom_bodyid[g]
            bname = mujoco.mj_id2name(self._fk_model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if bname == f"tibia_{leg_name}" and self._fk_model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                return g
        raise RuntimeError(f"No mesh geom found on tibia_{leg_name}")

    def reset(self):
        """Zero phase, clear cmd."""
        self._t = 0.0
        self._cmd[:] = 0.0

    # ----------------------------------------------------------------------
    # Cmd-driven evaluation
    # ----------------------------------------------------------------------
    def set_cmd(self, cmd):
        """Update the active cmd vector (used by step())."""
        cmd = np.asarray(cmd, dtype=np.float64)
        if cmd.shape != (9,):
            raise ValueError(f"cmd must be (9,), got {cmd.shape}")
        self._cmd = cmd.copy()

    def step(self, dt):
        """Advance internal phase by dt and return 18 joint targets for the active cmd."""
        self._t += float(dt)
        return self.predict(self._cmd, self._t)

    def predict(self, cmd, t):
        """Stateless eval: 18 joint targets at sim time t for the given cmd."""
        feet = self.compute_foot_targets(cmd, t)
        return self.body_to_joints(feet)

    def compute_foot_targets(self, cmd, t):
        """Return (6,3) body-frame foot targets for cmd at time t.

        Useful directly as a policy-observation hint (the analytical scaffold's
        idea of where each foot should be), without running IK."""
        cmd = np.asarray(cmd, dtype=np.float64)
        vx, vy, wz, pitch, roll, dh, dw, sx, sy = cmd

        # 1. Calibrated origins.
        feet = self.LEG_ORIGIN_BODY.copy()

        # 2. Stance height (vertical Z offset).
        feet[:, 2] += dh

        # 3. Stance width (radial XY offset along each leg's coxa→foot direction).
        feet[:, :2] += self.LEG_RADIAL_DIR_XY * dw

        # 4. Body shift (translate all leg origins horizontally in body frame).
        feet[:, 0] += sx
        feet[:, 1] += sy

        # 5. Gait cycle: translation + spin overlay (per-leg, phase-offset for tripod).
        # Path amplitude scales with how much motion is commanded — when cmd has
        # zero motion, the gait is fully suppressed (no swing, no lift) and feet
        # rest at their adjusted origins. At partial speed, stride and lift both
        # scale proportionally.
        speed        = math.hypot(vx, vy)
        stride_scale = min(speed / self.MAX_SPEED, 1.0) if speed > 1e-9 else 0.0
        spin_scale   = max(min(wz / self.MAX_YAW_RATE, 1.0), -1.0)
        gait_active  = max(stride_scale, abs(spin_scale))

        if gait_active > 1e-9:
            s_global = self.get_phase(t)
            heading  = math.atan2(vy, vx) if speed > 1e-9 else 0.0
            for i in range(6):
                s_i = (s_global + LEG_PHASE[i]) % 1.0
                px, pz = self.path_sample(s_i)

                # Translation contribution (same direction for every leg).
                trans_dx = stride_scale * px * math.cos(heading)
                trans_dy = stride_scale * px * math.sin(heading)

                # Spin contribution (per-leg, on its own concentric circle).
                if abs(spin_scale) > 1e-9:
                    x0, y0  = feet[i, 0], feet[i, 1]
                    r_i     = math.hypot(x0, y0)
                    th_i    = math.atan2(y0, x0)
                    dtheta  = spin_scale * px / self.spin_ref_radius
                    spin_dx = r_i * math.cos(th_i + dtheta) - x0
                    spin_dy = r_i * math.sin(th_i + dtheta) - y0
                else:
                    spin_dx, spin_dy = 0.0, 0.0

                feet[i, 0] += trans_dx + spin_dx
                feet[i, 1] += trans_dy + spin_dy
                feet[i, 2] += pz * gait_active   # lift scales with motion amplitude

        # 6. Body pitch/roll. Body tilts by R_body = R_y(pitch)·R_x(roll).
        # World-fixed foot positions, expressed in the tilted body frame, are
        # rotated by R_body⁻¹. Foot z's shift across legs to physically tilt
        # the body while feet stay planted.
        if abs(pitch) > 1e-9 or abs(roll) > 1e-9:
            feet = self._apply_body_tilt(feet, pitch, roll)

        return feet

    def get_phase(self, t):
        """Gait phase s ∈ [0, 1) at time t."""
        return (t / self.gait_period) % 1.0

    def path_sample(self, s):
        """Half-circle top + flat bottom path in body XZ plane, centered at origin.

        s ∈ [0, 0.5) → SWING : arc from (-R, 0) over (0, +R) to (+R, 0)
        s ∈ [0.5, 1) → STANCE: flat from (+R, 0) back to (-R, 0)
        """
        if s < 0.5:
            theta = math.pi * (1.0 - 2.0 * s)
            return self.path_radius * math.cos(theta), self.path_radius * math.sin(theta)
        ss = (s - 0.5) / 0.5
        return self.path_radius * (1.0 - 2.0 * ss), 0.0

    # ----------------------------------------------------------------------
    # Pitch/roll (body tilt) overlay
    # ----------------------------------------------------------------------
    @staticmethod
    def _apply_body_tilt(feet, pitch, roll):
        """Rotate each foot target by R_body⁻¹ where R_body = R_y(pitch)·R_x(roll).

        For a body that should tilt by (pitch around body Y, roll around body X),
        world-fixed feet appear in the tilted body frame as their original
        positions rotated by the inverse. Same rotation applied to every leg —
        legs at +X end up lower, legs at -X end up higher (for nose-up pitch),
        etc.
        """
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll),  math.sin(roll)
        # R = R_x(-roll) · R_y(-pitch)
        # Each row of R = [R_x(-roll) row i] · R_y(-pitch)
        R = np.array([
            [cp,     0.0,   -sp     ],
            [sr*sp,  cr,    sr*cp   ],
            [cr*sp,  -sr,   cr*cp   ],
        ])
        return (R @ feet.T).T

    # ----------------------------------------------------------------------
    # IK: body-frame foot targets → joint angles (mj_jac iterative)
    # ----------------------------------------------------------------------
    def body_to_joints(self, feet_body, max_iter=20, tol=1e-7, max_step=0.3):
        """Per-leg Jacobian iteration on the actual MJCF kinematics.

        Each leg solved independently. Body pinned at world origin → world
        positions == body-frame positions. Damped Newton step in joint space,
        clamped to max_step rad to keep the linearization valid.
        """
        feet_body = np.asarray(feet_body, dtype=np.float64)
        if feet_body.shape != (6, 3):
            raise ValueError(f"feet_body must be (6,3), got {feet_body.shape}")
        angles = NEUTRAL_POSE.copy()
        for i in range(6):
            target = feet_body[i]
            for _ in range(max_iter):
                actual = self._fk_foot_world(angles, i)
                residual = target - actual
                if np.linalg.norm(residual) < tol:
                    break
                J = self._foot_jacobian(i)
                try:
                    step = np.linalg.solve(J, residual)
                except np.linalg.LinAlgError:
                    step = np.linalg.pinv(J) @ residual
                sn = np.linalg.norm(step)
                if sn > max_step:
                    step *= max_step / sn
                angles[i*3:i*3+3] += step
        return angles

    def _set_pose(self, joints_18):
        self._fk_data.qpos[:]     = 0
        self._fk_data.qpos[3]     = 1.0          # quat w (identity orientation)
        self._fk_data.qpos[7:25]  = joints_18
        mujoco.mj_forward(self._fk_model, self._fk_data)

    def _fk_foot_world(self, joints_18, leg_idx):
        self._set_pose(joints_18)
        bid  = self._tibia_bid[leg_idx]
        pos  = self._fk_data.xpos[bid]
        xmat = self._fk_data.xmat[bid].reshape(3, 3)
        return pos + xmat @ self.FOOT_TIP_LOCAL[leg_idx]

    def _foot_jacobian(self, leg_idx):
        bid = self._tibia_bid[leg_idx]
        pos = self._fk_data.xpos[bid] + self._fk_data.xmat[bid].reshape(3, 3) @ self.FOOT_TIP_LOCAL[leg_idx]
        jacp = np.zeros((3, self._fk_model.nv))
        jacr = np.zeros((3, self._fk_model.nv))
        mujoco.mj_jac(self._fk_model, self._fk_data, jacp, jacr, pos, bid)
        qs = self._leg_qvel_slots[leg_idx]
        return jacp[:, qs[0]:qs[1]]
