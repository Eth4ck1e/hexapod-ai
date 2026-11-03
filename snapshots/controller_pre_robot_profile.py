"""
gait.controller — stateful hexapod gait controller (closed-form IK + overlays).

Public API (used by simple_gait, IK_gait, envs/hexapod_env, eventual ESP32-S3
firmware):

    ctrl = Controller("models/phantomx.xml")
    ctrl.set_cmd(cmd_9d)
    joints = ctrl.step(dt)
        # OR:
    joints = ctrl.predict(cmd_9d, t)
    feet_body = ctrl.compute_foot_targets(cmd_9d, t)   # (6, 3) body frame
    s = ctrl.get_phase(t)

Cmd vector (9 floats, all physical units):
    [0] vx           m/s     body forward velocity
    [1] vy           m/s     body lateral velocity (left = +)
    [2] wz           rad/s   body yaw rate (CCW = +)
    [3] pitch        rad     body pitch target (nose up = +)
    [4] roll         rad     body roll target (left side up = +)
                              [Euler convention; matches env's _body_pitch_roll]
    [5] height_delta m       stance height delta (- = body raised)
    [6] width_delta  m       stance width delta (+ = wider)
    [7] shift_x      m       body shift in body +X (planted feet stay put)
    [8] shift_y      m       body shift in body +Y (planted feet stay put)

Pipeline per call:
    1. Stance overlay  (cmd[5], cmd[6])  → modifies coxa-local rest position.
    2. Tilt overlay    (cmd[3], cmd[4])  → rotates each foot's body-frame
       position by R⁻¹ so feet stay planted while body rotates around them.
    3. Body shift      (cmd[7], cmd[8])  → translates body in body frame;
                                           planted feet stay in world frame
                                           by shifting their coxa-local
                                           targets by the inverse delta.
    4. Translation     (cmd[0], cmd[1])  → precomputed coxa-local path delta
                                           (xy), rotated by heading, scaled
                                           by stride_scale = |v|/MAX_SPEED.
    5. Spin            (cmd[2])          → on-the-fly arc rotation of foot
                                           around body-origin-in-coxa-local,
                                           scaled by wz/MAX_YAW_RATE.
    6. Lift            (canonical path z) → scaled by max(stride_scale,
                                            |spin_scale|) so feet only swing
                                            up when there is motion to make.
    7. IK              (closed-form, 3 trig calls per leg)  → 18 joint angles.

When neither stride nor spin is commanded, the gait is fully suppressed (no
swing, no lift) and feet rest at their stance-modulated, tilted positions.
"""

import math

import mujoco
import numpy as np

# ============================================================================
# Constants used by both Controller and external callers
# ============================================================================
LEG_NAMES = ["RR", "RM", "RF", "LR", "LM", "LF"]

# Body-frame yaw of each coxa, derived from coxa quats in models/phantomx.xml.
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

# Tripod phase offsets per leg (group A in sync, group B 180° offset).
LEG_PHASE = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])

# PhantomX link lengths (from MJCF).
COXA_LENGTH  = 0.052
FEMUR_LENGTH = 0.065
# Effective tibia length = distance from tibia joint to actual foot tip.
# Foot tip sits at (0.134, 0.031, 0.0) in tibia-local frame, magnitude 0.138.
TIBIA_LENGTH = 0.138

# NEUTRAL_POSE — joint configuration matching the calibrated (un-widened) foot
# rest. Hardcoded so the library has zero dependency on envs/. Used as the
# anchor for the formula→MJCF offset calibration; the runtime gait neutral
# (with stance widening applied) is derived from this and exposed via
# Controller.gait_neutral_pose.
NEUTRAL_POSE = np.array([
     0.0, -0.21818873, -0.32188801,    # RR
     0.0, -0.24918906, -0.26150113,    # RM
     0.0, -0.21818873, -0.32188801,    # RF
     0.0,  0.21818873,  0.32188801,    # LR
     0.0,  0.24918906,  0.26150113,    # LM
     0.0,  0.21818873,  0.32188801,    # LF
])


# ============================================================================
# Closed-form IK helpers (module-level for testing + reuse)
# ============================================================================
def _joint_signs(leg_idx):
    """Per-joint sign mapping from formula convention to MJCF joint angles.
    Coxa: +1 always. Femur: +1 right, -1 left (axis flipped). Tibia: -1 right,
    +1 left (axis flipped AND formula bend direction is opposite of MJCF positive
    rotation around the tibia axis). See docs/kinematics.md for derivation.
    """
    is_left = leg_idx >= 3
    return (
        +1.0,
        -1.0 if is_left else +1.0,
        +1.0 if is_left else -1.0,
    )


def _ik_raw(fx, fy, fz):
    """Closed-form 3-link IK in geometric (pre-MJCF-offset) convention.
    Single-leg version. Input: foot target in coxa-local frame. Returns
    (coxa, femur, tibia) raw."""
    coxa = math.atan2(fy, fx)
    r    = math.hypot(fx, fy)
    x_fp = r - COXA_LENGTH
    z_fp = fz
    D    = math.hypot(x_fp, z_fp)
    D = min(D, FEMUR_LENGTH + TIBIA_LENGTH - 1e-6)
    D = max(D, abs(FEMUR_LENGTH - TIBIA_LENGTH) + 1e-6)
    cos_a = (FEMUR_LENGTH**2 + D**2 - TIBIA_LENGTH**2) / (2 * FEMUR_LENGTH * D)
    cos_g = (FEMUR_LENGTH**2 + TIBIA_LENGTH**2 - D**2) / (2 * FEMUR_LENGTH * TIBIA_LENGTH)
    cos_a = max(-1.0, min(1.0, cos_a))
    cos_g = max(-1.0, min(1.0, cos_g))
    alpha = math.acos(cos_a)
    gamma = math.acos(cos_g)
    beta  = math.atan2(z_fp, x_fp)
    # Elbow-up (knee above femur→foot line — grasshopper config).
    femur = beta + alpha
    tibia = math.pi - gamma
    return coxa, femur, tibia


def _ik_raw_batch(feet_coxa):
    """Vectorized closed-form IK over all 6 legs at once.
    Input: (6, 3) array of foot targets in each leg's coxa-local frame.
    Returns: (6, 3) array of (coxa, femur, tibia) raw angles."""
    fx = feet_coxa[:, 0]
    fy = feet_coxa[:, 1]
    fz = feet_coxa[:, 2]
    coxa = np.arctan2(fy, fx)
    r    = np.hypot(fx, fy)
    x_fp = r - COXA_LENGTH
    z_fp = fz
    D    = np.hypot(x_fp, z_fp)
    D = np.minimum(D, FEMUR_LENGTH + TIBIA_LENGTH - 1e-6)
    D = np.maximum(D, abs(FEMUR_LENGTH - TIBIA_LENGTH) + 1e-6)
    cos_a = (FEMUR_LENGTH**2 + D**2 - TIBIA_LENGTH**2) / (2 * FEMUR_LENGTH * D)
    cos_g = (FEMUR_LENGTH**2 + TIBIA_LENGTH**2 - D**2) / (2 * FEMUR_LENGTH * TIBIA_LENGTH)
    np.clip(cos_a, -1.0, 1.0, out=cos_a)
    np.clip(cos_g, -1.0, 1.0, out=cos_g)
    alpha = np.arccos(cos_a)
    gamma = np.arccos(cos_g)
    beta  = np.arctan2(z_fp, x_fp)
    femur = beta + alpha
    tibia = math.pi - gamma
    return np.stack([coxa, femur, tibia], axis=1)


# ============================================================================
# Controller
# ============================================================================
class Controller:
    """Stateful hexapod gait controller — cmd → 18 joint angles.

    Construction calibrates against the MJCF: detects foot tip via lowest-Z
    mesh vertex, derives per-leg foot rest positions, anchors the closed-form
    IK to MJCF joint conventions via NEUTRAL_POSE.

    Public read-only attributes (set by calibrate()):
        LEG_ORIGIN_BODY        (6, 3)  foot rest positions in body frame
                                       (calibrated, before stance widening)
        FOOT_TIP_LOCAL         (6, 3)  foot tip in tibia local frame
        LEG_RADIAL_DIR_XY      (6, 2)  horizontal coxa→foot unit vectors
        FOOT_REST_COXA         (6, 3)  active foot rest in each leg's coxa-local
                                       frame (= calibrated rest + default stance
                                       widening)
        BODY_ORIGIN_COXA       (6, 3)  body origin expressed in each leg's
                                       coxa-local frame (constant per leg)
        LEG_OFFSETS            (6, 3)  per-joint formula→MJCF offsets
        LEG_PATH_DELTAS        (6, N, 2)  precomputed translation path xy at
                                          heading=0
        gait_neutral_pose      (18,)   joint angles for FOOT_REST_COXA (use
                                       this for env settle / reset)
    """

    # 2026-05-08 (v8 prep): defaults tuned for hardware-realistic speed.
    #   path_radius is held SMALL (≤ 40 mm) — large radii produce kinematic
    #     singularities, asymmetric leg loading, and visible body sway.
    #   gait_period does the speed work — it's bounded only by AX-12A joint
    #     velocity (max practical ~3 rad/s under load).
    # 4 × 0.040 / 0.45 = 0.356 m/s ≈ 0.9 × theoretical hardware max (0.40 m/s).
    # See docs/MAX_SPEED_ANALYSIS.md for the servo-velocity derivation.
    DEFAULT_GAIT_PERIOD     = 0.45
    DEFAULT_PATH_RADIUS     = 0.040
    DEFAULT_PATH_RES        = 100
    DEFAULT_STANCE_WIDTH    = 0.015     # +15 mm outward per foot at neutral

    # Turning-radius operational range for arc-aware turn-while-walking. When
    # both speed and wz are non-zero, the implied turn radius R = speed / |wz|
    # is checked against this range:
    #   R < MIN_TURN_RADIUS  → fall back to pure spin (essentially in-place rotation)
    #   R > MAX_TURN_RADIUS  → fall back to pure translate (essentially straight)
    #   in range             → arc-aware path generation around the turn center
    # Bounds shifted up after visual review: at MIN=0.135m the turn center
    # was inside the body footprint and inner legs (LM/RM at ±177mm laterally)
    # were trying to orbit a point right next to them, causing legs to converge
    # near body center. New MIN=0.250m puts the turn center safely outside the
    # body envelope. MAX shifted up by the same amount to preserve range width.
    MIN_TURN_RADIUS         = 0.350    # m, ~1.3× body length (was 0.250)
    MAX_TURN_RADIUS         = 0.755    # m, ~2.8× body length (was 0.655)

    # Calibrated speed ladder. Each level maps to a (gait_period, path_radius)
    # tuple that delivers the listed actual velocity in physics simulation
    # under ±1.5 N·m motor torque. Source: tools/scaffold_speed_calibration.csv
    # (49-combo sweep run 2026-05-07). Slip is 2-4 mm per stance at all levels;
    # levels ≥ 2.5 begin to saturate motor torque.
    SPEED_LEVELS = {
        0.5: {"gait_period": 1.50, "path_radius": 0.020},   # ~0.050 m/s
        1.0: {"gait_period": 1.50, "path_radius": 0.030},   # ~0.077 m/s
        1.5: {"gait_period": 1.50, "path_radius": 0.040},   # ~0.103 m/s
        2.0: {"gait_period": 1.20, "path_radius": 0.040},   # ~0.128 m/s
        2.5: {"gait_period": 1.00, "path_radius": 0.050},   # ~0.190 m/s
        3.0: {"gait_period": 0.85, "path_radius": 0.060},   # ~0.261 m/s
        3.5: {"gait_period": 0.75, "path_radius": 0.070},   # ~0.362 m/s
    }

    def __init__(self, model_path,
                 speed_level=None,
                 gait_period=None, path_radius=None, path_res=None,
                 default_stance_width=None):
        # speed_level resolves to (gait_period, path_radius) via SPEED_LEVELS.
        # Mutually exclusive with explicit gait_period / path_radius — pick one.
        if speed_level is not None:
            if gait_period is not None or path_radius is not None:
                raise ValueError(
                    "speed_level is mutually exclusive with gait_period and "
                    "path_radius — pick one or the other, not both.")
            if speed_level not in self.SPEED_LEVELS:
                raise ValueError(
                    f"speed_level {speed_level} not in calibrated set "
                    f"{sorted(self.SPEED_LEVELS.keys())}")
            level_cfg     = self.SPEED_LEVELS[speed_level]
            gait_period   = level_cfg["gait_period"]
            path_radius   = level_cfg["path_radius"]
        self.speed_level          = speed_level   # None if explicit/default
        self.model_path           = model_path
        self.gait_period          = gait_period          or self.DEFAULT_GAIT_PERIOD
        self.path_radius          = path_radius          or self.DEFAULT_PATH_RADIUS
        self.path_res             = path_res             or self.DEFAULT_PATH_RES
        self.default_stance_width = (default_stance_width
                                     if default_stance_width is not None
                                     else self.DEFAULT_STANCE_WIDTH)

        # Cached cos/sin for body→coxa rotation (one pair per leg, both as
        # tuples for legacy single-leg code AND as (6,) numpy arrays for
        # vectorized batch operations).
        self._yaw_cs = [(math.cos(y), math.sin(y)) for y in LEG_BODY_YAW]
        self._yaw_c  = np.array([math.cos(y) for y in LEG_BODY_YAW])   # (6,)
        self._yaw_s  = np.array([math.sin(y) for y in LEG_BODY_YAW])   # (6,)

        # Per-joint sign matrix for vectorized IK→MJCF mapping. Shape (6, 3),
        # one row per leg, columns are (coxa_sign, femur_sign, tibia_sign).
        self._joint_signs_array = np.array([_joint_signs(i) for i in range(6)])

        # Working state.
        self._cmd = np.zeros(9, dtype=np.float64)
        self._t   = 0.0

        self.calibrate()
        self._precompute_paths()

    # ----------------------------------------------------------------------
    # Derived constants — what the analytical scaffold can deliver.
    # ----------------------------------------------------------------------
    @property
    def MAX_SPEED(self):
        """Max body linear speed (m/s). 4·R/T (one full ±R stride per stance,
        two stances per cycle, one cycle = T)."""
        return 4.0 * self.path_radius / self.gait_period

    @property
    def MAX_YAW_RATE(self):
        """Max body yaw rate (rad/s) at full angular stride."""
        return 4.0 * (self.path_radius / self.spin_ref_radius) / self.gait_period

    # ----------------------------------------------------------------------
    # Calibration
    # ----------------------------------------------------------------------
    def calibrate(self):
        """Compute per-leg geometric constants from the MJCF + anchor IK to it.
        Run once on construction; idempotent."""
        model = mujoco.MjModel.from_xml_path(self.model_path)
        data  = mujoco.MjData(model)
        tibia_bid = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tibia_{n}")
            for n in LEG_NAMES
        ])

        # Set joints to NEUTRAL_POSE and run FK so geom positions are valid.
        self._set_pose(model, data, NEUTRAL_POSE)

        # Foot tip detection: two paths depending on whether the model uses a
        # mesh tibia (legacy phantomx.xml) or primitive geoms (phantomx_simple.xml).
        #   * Mesh path: find lowest-world-Z vertex of the tibia mesh at
        #     NEUTRAL_POSE — empirical, picks up whatever the mesh says.
        #   * Primitive path: look up the named foot_<LEG> geom and read its
        #     `geom_pos` (tibia-local position). Direct, exact, no search.
        # The primitive path is preferred — it's cheaper and produces an
        # exact result rather than an empirical approximation.
        tibia_mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "tibia")
        use_primitive_foot = (tibia_mesh_id < 0)

        self.FOOT_TIP_LOCAL  = np.zeros((6, 3))
        self.LEG_ORIGIN_BODY = np.zeros((6, 3))

        if use_primitive_foot:
            for i, n in enumerate(LEG_NAMES):
                foot_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{n}")
                if foot_gid < 0:
                    raise RuntimeError(
                        f"Model has no tibia mesh AND no 'foot_{n}' geom. "
                        f"Add either a mesh tibia or a named primitive foot geom."
                    )
                # geom_pos is the geom origin in its parent body's local frame —
                # for our foot spheres, that's tibia-local coordinates directly.
                self.FOOT_TIP_LOCAL[i] = model.geom_pos[foot_gid].copy()
                # Body-frame foot rest = tibia world pos + rotation @ tibia-local foot tip.
                tp = data.xpos[tibia_bid[i]]
                tm = data.xmat[tibia_bid[i]].reshape(3, 3)
                self.LEG_ORIGIN_BODY[i] = tp + tm @ self.FOOT_TIP_LOCAL[i]
        else:
            # Mesh path — empirical lowest-Z vertex search.
            v0 = model.mesh_vertadr[tibia_mesh_id]
            vn = model.mesh_vertnum[tibia_mesh_id]
            mesh_verts = model.mesh_vert[v0:v0+vn].copy()
            for i, n in enumerate(LEG_NAMES):
                tg = self._find_tibia_geom(model, n)
                gp = data.geom_xpos[tg]
                gm = data.geom_xmat[tg].reshape(3, 3)
                world_verts = (gm @ mesh_verts.T).T + gp
                low = world_verts[np.argmin(world_verts[:, 2])]
                tp = data.xpos[tibia_bid[i]]
                tm = data.xmat[tibia_bid[i]].reshape(3, 3)
                self.FOOT_TIP_LOCAL[i]  = tm.T @ (low - tp)
                self.LEG_ORIGIN_BODY[i] = low                # body == world here

        # Normalize foot z to a common plane (2026-05-10): the MJCF naturally
        # produces ~3 mm spread between middle legs (-148 mm) and front/rear
        # (-145 mm) because middle coxas mount slightly lower. The walking
        # path-mapping math assumes all feet share a z plane, so we set
        # LEG_ORIGIN_BODY[:, 2] = mean across legs. The IK then resolves
        # per-leg joint angles to plant each foot at this uniform z. R/L
        # mirror symmetry is already perfect (verified to micron precision)
        # so this only equalizes front/middle/rear, not left vs right.
        self.LEG_ORIGIN_BODY[:, 2] = self.LEG_ORIGIN_BODY[:, 2].mean()

        # Per-leg horizontal coxa→foot unit vector (body frame).
        diff_xy = self.LEG_ORIGIN_BODY[:, :2] - COXA_POS_BODY[:, :2]
        norms   = np.linalg.norm(diff_xy, axis=1, keepdims=True)
        self.LEG_RADIAL_DIR_XY = diff_xy / norms

        # Foot rest in each leg's coxa-local frame (calibrated, un-widened).
        # Used internally as the IK offset anchor.
        rest_coxa_cal = np.array([
            self._body_to_coxa_local(self.LEG_ORIGIN_BODY[i], i)
            for i in range(6)
        ])

        # Active foot rest = calibrated + default stance widening.
        self.FOOT_REST_COXA = rest_coxa_cal.copy()
        if self.default_stance_width != 0.0:
            self.FOOT_REST_COXA[:, 0] += self.default_stance_width

        # Body origin in each leg's coxa-local frame (spin axis location).
        self.BODY_ORIGIN_COXA = np.array([
            self._body_to_coxa_local(np.zeros(3), i) for i in range(6)
        ])

        # Spin reference radius — distance from body origin to outer-leg foot.
        # Used to scale wz into a per-cycle angular stride that's identical
        # across legs (so body spins coherently regardless of leg-specific
        # arc radii).
        arc_radii = np.array([
            math.hypot(self.FOOT_REST_COXA[i, 0] - self.BODY_ORIGIN_COXA[i, 0],
                       self.FOOT_REST_COXA[i, 1] - self.BODY_ORIGIN_COXA[i, 1])
            for i in range(6)
        ])
        self.spin_ref_radius = float(arc_radii[0])     # outer-leg radius (RR)
        self._arc_radii      = arc_radii               # informational

        # Formula→MJCF joint-angle offsets, anchored at NEUTRAL_POSE
        # (which matches the un-widened calibrated rest).
        self.LEG_OFFSETS = np.zeros((6, 3))
        for i in range(6):
            raw = np.array(_ik_raw(*rest_coxa_cal[i]))
            expected = NEUTRAL_POSE[i*3:i*3+3]
            s_c, s_f, s_t = _joint_signs(i)
            self.LEG_OFFSETS[i, 0] = expected[0] - s_c * raw[0]
            self.LEG_OFFSETS[i, 1] = expected[1] - s_f * raw[1]
            self.LEG_OFFSETS[i, 2] = expected[2] - s_t * raw[2]

        # Gait-time neutral pose: joints that put each foot at the WIDENED
        # FOOT_REST_COXA. Equals NEUTRAL_POSE when default_stance_width == 0.
        self.gait_neutral_pose = np.zeros(18)
        for i in range(6):
            self.gait_neutral_pose[i*3:i*3+3] = self._ik_mjcf(self.FOOT_REST_COXA[i], i)

    @staticmethod
    def _set_pose(model, data, joints_18):
        data.qpos[:]    = 0.0
        data.qpos[3]    = 1.0          # quat w (identity)
        data.qpos[7:25] = joints_18
        mujoco.mj_forward(model, data)

    @staticmethod
    def _find_tibia_geom(model, leg_name):
        for g in range(model.ngeom):
            bid = model.geom_bodyid[g]
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) == f"tibia_{leg_name}"
                    and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH):
                return g
        raise RuntimeError(f"No mesh geom found on tibia_{leg_name}")

    # ----------------------------------------------------------------------
    # Path precompute
    # ----------------------------------------------------------------------
    def _precompute_paths(self):
        """Per-leg coxa-local translation path (XY only, at heading=0).
        Stored as (6, N, 2). Lift (path_z) is sampled on-the-fly because it's
        scaled by gait_active, not stride_scale."""
        self.LEG_PATH_DELTAS = np.zeros((6, self.path_res, 2))
        for i in range(6):
            c_y, s_y = self._yaw_cs[i]
            # body forward (1, 0) expressed in this leg's coxa-local:
            dir_x =  c_y
            dir_y = -s_y
            for n in range(self.path_res):
                px, _ = self._canonical_path(n / self.path_res)
                self.LEG_PATH_DELTAS[i, n, 0] = px * dir_x
                self.LEG_PATH_DELTAS[i, n, 1] = px * dir_y

    def _canonical_path(self, s):
        """Single-sample canonical path: half-circle top + flat bottom.
        Returns (px, pz)."""
        if s < 0.5:
            theta = math.pi * (1.0 - 2.0 * s)
            return self.path_radius * math.cos(theta), self.path_radius * math.sin(theta)
        ss = (s - 0.5) / 0.5
        return self.path_radius * (1.0 - 2.0 * ss), 0.0

    def _canonical_path_batch(self, s_array):
        """Vectorized canonical path. Input: (N,) phase samples in [0, 1).
        Returns (px_array, pz_array), each (N,)."""
        R = self.path_radius
        px = np.empty_like(s_array)
        pz = np.empty_like(s_array)
        swing = s_array < 0.5
        # Swing branch: theta = π·(1 - 2s); px = R·cos(θ), pz = R·sin(θ).
        theta = math.pi * (1.0 - 2.0 * s_array[swing])
        px[swing] = R * np.cos(theta)
        pz[swing] = R * np.sin(theta)
        # Stance branch: linear from (+R, 0) to (-R, 0).
        ss = (s_array[~swing] - 0.5) / 0.5
        px[~swing] = R * (1.0 - 2.0 * ss)
        pz[~swing] = 0.0
        return px, pz

    # Backward-compat alias (the old Controller exposed this name).
    def path_sample(self, s):
        return self._canonical_path(s)

    # ----------------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------------
    def reset(self):
        self._t = 0.0
        self._cmd[:] = 0.0

    def set_cmd(self, cmd):
        cmd = np.asarray(cmd, dtype=np.float64)
        if cmd.shape != (9,):
            raise ValueError(f"cmd must be (9,), got {cmd.shape}")
        self._cmd = cmd.copy()

    def step(self, dt):
        """Advance internal phase by dt and return 18 joint targets."""
        self._t += float(dt)
        return self.predict(self._cmd, self._t)

    def get_phase(self, t):
        """Gait phase s ∈ [0, 1) at time t. Stateless; period is constant."""
        return (t / self.gait_period) % 1.0

    # ----------------------------------------------------------------------
    # Cmd → joints (the hot path)
    # ----------------------------------------------------------------------
    def predict(self, cmd, t):
        """Stateless eval: cmd + sim time → 18 joint targets."""
        feet_coxa = self._compute_foot_targets_coxa(cmd, t)
        return self._joints_from_feet_coxa(feet_coxa)

    def compute_foot_targets(self, cmd, t):
        """Public wrapper returning foot targets in BODY frame, (6, 3)."""
        feet_coxa = self._compute_foot_targets_coxa(cmd, t)
        return self._feet_coxa_to_body(feet_coxa)

    def predict_with_feet(self, cmd, t):
        """One-pass evaluation: returns (joints_18, feet_body_6x3) at the same
        cost as a single predict(). Use this when the caller needs BOTH the
        joint targets AND the body-frame foot positions for the same
        (cmd, t) — e.g. the training env, which uses joints for the actuators
        and feet_body for the scaffold-hint observation.

        Replaces a separate predict() + compute_foot_targets() pair, which
        would run the full gait pipeline (stance, tilt, path, spin, lift)
        twice."""
        feet_coxa = self._compute_foot_targets_coxa(cmd, t)
        joints    = self._joints_from_feet_coxa(feet_coxa)
        feet_body = self._feet_coxa_to_body(feet_coxa)
        return joints, feet_body

    def _feet_coxa_to_body(self, feet_coxa):
        """Convert (6, 3) coxa-local foot positions to body frame (vectorized)."""
        return self._coxa_local_to_body_batch(feet_coxa)

    def body_to_joints(self, feet_body):
        """Public IK utility — body-frame foot targets → joint angles (vectorized)."""
        feet_body = np.asarray(feet_body, dtype=np.float64)
        if feet_body.shape != (6, 3):
            raise ValueError(f"feet_body must be (6,3), got {feet_body.shape}")
        feet_coxa = self._body_to_coxa_local_batch(feet_body)
        return self._joints_from_feet_coxa(feet_coxa)

    # ----------------------------------------------------------------------
    # Internal: cmd → foot targets in coxa-local
    # ----------------------------------------------------------------------
    def _compute_foot_targets_coxa(self, cmd, t):
        cmd = np.asarray(cmd, dtype=np.float64)
        vx, vy, wz, pitch, roll, dh, dw, sx, sy = cmd

        # 1. Stance overlay.
        rest = self.FOOT_REST_COXA.copy()
        rest[:, 0] += dw          # widen
        rest[:, 2] += dh          # height_delta (env: - = body raised → z more negative)

        # 2. Tilt overlay (rotate each foot's body-frame position by R⁻¹).
        if pitch != 0.0 or roll != 0.0:
            rest = self._apply_tilt(rest, pitch, roll)

        # 3. Body shift overlay. Translates the body by (sx, sy) in body
        # frame while keeping the planted feet at the same WORLD position —
        # i.e. the feet appear to move by (-sx, -sy) in body frame. In each
        # leg's coxa-local frame that body-frame delta becomes
        #   dx_coxa =  c·(-sx) + s·(-sy)  =  -(c·sx + s·sy)
        #   dy_coxa = -s·(-sx) + c·(-sy)  =   s·sx - c·sy
        # where (c, s) = (cos, sin) of the leg's body-frame yaw. Vectorized
        # below using the cached _yaw_c / _yaw_s arrays. Cheap — pure adds.
        if sx != 0.0 or sy != 0.0:
            rest[:, 0] += -(self._yaw_c * sx + self._yaw_s * sy)
            rest[:, 1] +=  (self._yaw_s * sx - self._yaw_c * sy)

        # 4. Speed scales for translation, spin, and lift.
        speed        = math.hypot(vx, vy)
        stride_scale = min(speed / self.MAX_SPEED, 1.0) if speed > 1e-9 else 0.0
        spin_scale   = max(min(wz / self.MAX_YAW_RATE, 1.0), -1.0) if abs(wz) > 1e-9 else 0.0
        gait_active  = max(stride_scale, abs(spin_scale))

        if gait_active < 1e-9:
            # No motion commanded — feet rest at the modulated rest.
            return rest

        s_global = self.get_phase(t)
        heading  = math.atan2(vy, vx) if speed > 1e-9 else 0.0
        c_h, s_h = math.cos(heading), math.sin(heading)

        # Per-leg phase indices (vectorized).
        s_i = (s_global + LEG_PHASE) % 1.0                            # (6,)
        n_idx = (s_i * self.path_res).astype(np.int64) % self.path_res # (6,)

        # Lift + canonical path samples for all 6 legs at once.
        px_arr, pz_arr = self._canonical_path_batch(s_i)              # (6,) each

        # Determine motion regime: arc-aware or superposition.
        # Arc mode requires both translation AND rotation, with an implied
        # turning radius R = speed / |wz| within the operational range.
        arc_mode = False
        if speed > 1e-9 and abs(wz) > 1e-9:
            R = speed / abs(wz)
            if self.MIN_TURN_RADIUS < R < self.MAX_TURN_RADIUS:
                arc_mode = True

        if arc_mode:
            # ARC-AWARE PATH (Iter-7 of scaffold tuning, 2026-05-07):
            # When the body is moving along a circular arc of radius R around
            # turn center C, each leg's foot in body frame rotates around C by
            # the OPPOSITE of the body's rotation (so the foot stays fixed in
            # world during stance — no slip).
            #
            # C_body = perp(heading) · R · sign(wz)
            #   where perp(h) = (-sin h, cos h) is 90° CCW from heading.
            #   For wz>0 (CCW turn), C is to the LEFT of motion direction.
            # Each leg's foot rotates around C by dtheta = wz·T/4·(px/path_radius).
            # This produces TRUE arc paths instead of the superposition of
            # linear translation + body-origin spin.
            sign_wz = math.copysign(1.0, wz)
            C_body_x = -s_h * R * sign_wz
            C_body_y =  c_h * R * sign_wz
            # Express C in each leg's coxa-local frame (broadcast same body
            # point to all 6 legs; the batch helper rotates each by leg yaw
            # and translates by leg coxa origin).
            C_body_arr = np.tile(np.array([C_body_x, C_body_y, 0.0]), (6, 1))
            C_coxa = self._body_to_coxa_local_batch(C_body_arr)        # (6, 3)
            cx = C_coxa[:, 0]
            cy = C_coxa[:, 1]
            # Angular sweep per phase. The peak |dtheta| = wz·T/4 matches the
            # body's angular advance over a quarter cycle. This is exactly
            # what the existing pure-spin code computes (see derivation in
            # docs); just rotate around C instead of BODY_ORIGIN_COXA.
            dtheta = wz * self.gait_period / 4.0 * (px_arr / self.path_radius)
            cs = np.cos(dtheta)
            ss = np.sin(dtheta)
            rx = rest[:, 0] - cx
            ry = rest[:, 1] - cy
            arc_x = (cx + (cs * rx - ss * ry)) - rest[:, 0]
            arc_y = (cy + (ss * rx + cs * ry)) - rest[:, 1]
            feet = np.empty_like(rest)
            feet[:, 0] = rest[:, 0] + arc_x
            feet[:, 1] = rest[:, 1] + arc_y
            feet[:, 2] = rest[:, 2] + pz_arr * gait_active
            return feet

        # SUPERPOSITION (existing pure-translate, pure-spin, or out-of-range
        # combined motion that falls back to superposition).
        # Translation contribution: per-leg precomputed coxa-local path xy at
        # heading=0, fancy-indexed by n_idx to pick the active phase per leg,
        # then rotated by heading, then scaled by stride_scale.
        path_xy = self.LEG_PATH_DELTAS[np.arange(6), n_idx]            # (6, 2)
        trans_x = stride_scale * (c_h * path_xy[:, 0] - s_h * path_xy[:, 1])
        trans_y = stride_scale * (s_h * path_xy[:, 0] + c_h * path_xy[:, 1])

        # Spin contribution: on-the-fly rotation around body-origin-in-coxa
        # for each leg, scaled by spin_scale * px / SPIN_REF_RADIUS.
        if abs(spin_scale) > 1e-9:
            cx = self.BODY_ORIGIN_COXA[:, 0]                          # (6,)
            cy = self.BODY_ORIGIN_COXA[:, 1]
            rx = rest[:, 0] - cx
            ry = rest[:, 1] - cy
            dtheta = spin_scale * px_arr / self.spin_ref_radius       # (6,)
            cs = np.cos(dtheta)
            ss = np.sin(dtheta)
            spin_x = (cx + (cs * rx - ss * ry)) - rest[:, 0]
            spin_y = (cy + (ss * rx + cs * ry)) - rest[:, 1]
        else:
            spin_x = 0.0
            spin_y = 0.0

        # Lift scales with gait_active (suppress lift when no motion commanded).
        lift = pz_arr * gait_active                                   # (6,)

        feet = np.empty_like(rest)
        feet[:, 0] = rest[:, 0] + trans_x + spin_x
        feet[:, 1] = rest[:, 1] + trans_y + spin_y
        feet[:, 2] = rest[:, 2] + lift
        return feet

    # ----------------------------------------------------------------------
    # Internal: closed-form IK (per-leg)
    # ----------------------------------------------------------------------
    def _joints_from_feet_coxa(self, feet_coxa):
        """Vectorized: (6, 3) coxa-local foot targets → (18,) joint angles."""
        raw = _ik_raw_batch(feet_coxa)                              # (6, 3)
        mjcf = self._joint_signs_array * raw + self.LEG_OFFSETS    # (6, 3)
        return mjcf.flatten()

    def _ik_mjcf(self, foot_coxa, leg_idx):
        """Single-leg IK → MJCF joint angles. Used at calibration time."""
        coxa_raw, femur_raw, tibia_raw = _ik_raw(foot_coxa[0], foot_coxa[1], foot_coxa[2])
        s_c, s_f, s_t = _joint_signs(leg_idx)
        return (
            s_c * coxa_raw  + self.LEG_OFFSETS[leg_idx, 0],
            s_f * femur_raw + self.LEG_OFFSETS[leg_idx, 1],
            s_t * tibia_raw + self.LEG_OFFSETS[leg_idx, 2],
        )

    # ----------------------------------------------------------------------
    # Internal: frame transforms
    # ----------------------------------------------------------------------
    def _body_to_coxa_local(self, p_body, leg_idx):
        """Single-leg body-frame point → coxa-local. (Legacy path, used by
        calibration code; runtime uses the batch version below.)"""
        rel = p_body - COXA_POS_BODY[leg_idx]
        c, s = self._yaw_cs[leg_idx]
        return np.array([
             c * rel[0] + s * rel[1],
            -s * rel[0] + c * rel[1],
             rel[2],
        ])

    def _coxa_local_to_body(self, p_coxa, leg_idx):
        """Single-leg inverse of _body_to_coxa_local."""
        c, s = self._yaw_cs[leg_idx]
        return COXA_POS_BODY[leg_idx] + np.array([
             c * p_coxa[0] - s * p_coxa[1],
             s * p_coxa[0] + c * p_coxa[1],
             p_coxa[2],
        ])

    def _body_to_coxa_local_batch(self, p_body):
        """Vectorized body-frame → coxa-local for all 6 legs.
        Input: (6, 3) body-frame points (one per leg).
        Output: (6, 3) coxa-local points."""
        rel = p_body - COXA_POS_BODY                          # (6, 3)
        out = np.empty_like(rel)
        out[:, 0] =  self._yaw_c * rel[:, 0] + self._yaw_s * rel[:, 1]
        out[:, 1] = -self._yaw_s * rel[:, 0] + self._yaw_c * rel[:, 1]
        out[:, 2] = rel[:, 2]
        return out

    def _coxa_local_to_body_batch(self, p_coxa):
        """Vectorized coxa-local → body-frame for all 6 legs.
        Input: (6, 3) coxa-local points. Output: (6, 3) body-frame points."""
        out = np.empty_like(p_coxa)
        out[:, 0] = self._yaw_c * p_coxa[:, 0] - self._yaw_s * p_coxa[:, 1]
        out[:, 1] = self._yaw_s * p_coxa[:, 0] + self._yaw_c * p_coxa[:, 1]
        out[:, 2] = p_coxa[:, 2]
        return out + COXA_POS_BODY

    # ----------------------------------------------------------------------
    # Internal: tilt overlay
    # ----------------------------------------------------------------------
    def _apply_tilt(self, rest_coxa_array, pitch, roll):
        """Vectorized tilt: rotate each foot's body-frame position by R⁻¹ where
        R = R_y(-pitch)·R_x(roll). Convention: pitch+ = nose up,
        roll+ = body's left side up (Euler).
        """
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll),  math.sin(roll)
        R_inv = np.array([
            [ cp,        0.0,     sp     ],
            [-sp * sr,   cr,      cp * sr],
            [-sp * cr,  -sr,      cp * cr],
        ])
        # All 6 legs in one round-trip:  coxa-local → body → tilt → coxa-local.
        rest_body   = self._coxa_local_to_body_batch(rest_coxa_array)   # (6, 3)
        tilted_body = rest_body @ R_inv.T                                # (6, 3)
        return self._body_to_coxa_local_batch(tilted_body)               # (6, 3)
