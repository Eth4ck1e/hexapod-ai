import math
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/phantomx.xml")

# PhantomX params from phantomX.yaml
COXA_LENGTH  = 0.052
FEMUR_LENGTH = 0.065
TIBIA_LENGTH = 0.133
CYCLE_LENGTH = 50
LEG_LIFT_HEIGHT = 0.038
MAX_SPEED    = 0.082  # m/s

INIT_COXA_ANGLE = np.deg2rad([-45, 0, 45, -45, 0, 45])
# Foot rest position in each leg's IK frame.
# X offset paired with INIT_COXA_ANGLE keeps coxa=0 at neutral; Y is outward
# distance; Z is depth below coxa joint. Widened from the ROS YAML values
# (which assumed an extra tarsus segment) so legs splay outward instead of
# folding under the body.
INIT_FOOT_POS_X = np.array([-0.13,  0.0,   0.13, -0.13,  0.0,   0.13])
INIT_FOOT_POS_Y = np.array([ 0.13,  0.18,  0.13,  0.13,  0.18,  0.13])
INIT_FOOT_POS_Z = np.array([ 0.10,  0.10,  0.10,  0.10,  0.10,  0.10])

# Tripod groups — leg order: [RR, RM, RF, LR, LM, LF]
# {1,0,1,0,1,0}: RR/RF/LM stance, RM/LR/LF swing
TRIPOD_INIT = np.array([1, 0, 1, 0, 1, 0])


def _ik(foot_x, foot_y, foot_z, leg_idx):
    """
    Foot position (local coxa frame) → (coxa, femur, tibia) joint angles.
    Ported from hexapod_ros ik.cpp.
    foot_z is positive downward (distance below coxa joint).
    """
    femur_to_tarsus = math.sqrt(foot_x**2 + foot_y**2) - COXA_LENGTH
    side_c = math.sqrt(femur_to_tarsus**2 + foot_z**2)

    cos_b = np.clip((FEMUR_LENGTH**2 - TIBIA_LENGTH**2 + side_c**2) /
                    (2 * FEMUR_LENGTH * side_c), -1, 1)
    cos_c = np.clip((FEMUR_LENGTH**2 + TIBIA_LENGTH**2 - side_c**2) /
                    (2 * FEMUR_LENGTH * TIBIA_LENGTH), -1, 1)

    angle_b = math.acos(cos_b)
    angle_c = math.acos(cos_c)
    theta   = math.atan2(femur_to_tarsus, foot_z)

    coxa  = math.atan2(foot_x, foot_y) + INIT_COXA_ANGLE[leg_idx]
    femur = (math.pi / 2) - (theta + angle_b)
    tibia = (math.pi / 2) - angle_c
    return coxa, femur, tibia


def _compute_neutral_pose():
    """Joint angles (18,) that put all feet at their INIT_FOOT positions."""
    q = np.zeros(18)
    for i in range(6):
        c, f, t = _ik(-INIT_FOOT_POS_X[i], INIT_FOOT_POS_Y[i], INIT_FOOT_POS_Z[i], i)
        # MJCF left-leg coxa shares +Z axis with right legs, but femur/tibia
        # axes are flipped — negate c, f, t to keep left/right physically
        # symmetric.
        if i >= 3:
            c = -c
            f = -f
            t = -t
        q[i * 3]     = c
        q[i * 3 + 1] = f
        q[i * 3 + 2] = t
    return q


def _tripod_gait(cmd_x, cmd_y, cycle_period, cycle_leg_number):
    """
    Returns foot offsets (6, 3) — [dx, dy, dz] per leg for this cycle step.
    Ported from hexapod_ros gait.cpp.
    """
    period_height   = math.sin(cycle_period * math.pi / CYCLE_LENGTH)
    period_distance = math.cos(cycle_period * math.pi / CYCLE_LENGTH)

    offsets = np.zeros((6, 3))
    for i in range(6):
        if cycle_leg_number[i] == 0:  # swing
            offsets[i, 0] = cmd_x * period_distance
            offsets[i, 1] = cmd_y * period_distance
            offsets[i, 2] = LEG_LIFT_HEIGHT * period_height
        else:  # stance
            pd = math.cos(cycle_period * math.pi / CYCLE_LENGTH)
            offsets[i, 0] = -cmd_x * pd
            offsets[i, 1] = -cmd_y * pd
            offsets[i, 2] = 0.0
    return offsets


NEUTRAL_POSE = _compute_neutral_pose()


class HexapodEnv(gym.Env):
    """
    PhantomX hexapod with tripod gait base controller.
    Policy outputs residual joint angles on top of the gait.

    Obs (60): joint_pos(18) + joint_vel(18) + imu_quat(4) + imu_gyro(3) +
              imu_accel(3) + body_linvel(3) + gait_phase(2) + cmd(9)
    Act (18): residual joint angle targets, normalised to [-1, 1]

    cmd vector slot map (stable across stages — earlier stages zero unused):
      [0] vx              forward velocity                (stage 1+)
      [1] vy              lateral velocity                (stage 1+, cardinal-only in stage 1)
      [2] wz              yaw rate                        (stage 3+)
      [3] pitch_offset    body pitch target vs gravity    (stage 4+)
      [4] roll_offset     body roll target vs gravity     (stage 4+)
      [5] height_offset   stance height delta             (future)
      [6] stance_width    stance width scale              (future)
      [7] shift_x         body fore/aft shift             (future)
      [8] shift_y         body left/right shift           (future)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    CMD_DIM            = 9
    RESIDUAL_SCALE_MIN = 0.1   # rad — residual range when gait scaffold is fully on
    RESIDUAL_SCALE_MAX = 0.8   # rad — residual range when gait scaffold is fully off
    CTRL_COST_W        = 0.02
    SURVIVE_BONUS  = 0.25
    MIN_Z          = 0.06  # body height below this = fallen
    MAX_TILT_SQ    = 0.5   # quat (qx²+qy²) above this = flipped (~60°)
    REF_BODY_Z     = 0.13  # nominal standing height — used for z-bob penalty
    TILT_W         = 0.05    # tilt-from-gravity penalty (level body plane)
    Z_BOB_W        = 0.5     # body z-bob penalty
    ACTION_RATE_W  = 0.02    # smoothness: penalise rapid action changes
    JOINT_VEL_W    = 0.0001  # smoothness: penalise high joint speeds
    ANGVEL_W       = 0.02    # smoothness: penalise body wobble/spin
    YAW_RATE_W     = 0.5     # penalise yaw rate deviation from cmd[2]

    def __init__(self, render_mode=None, stage=1):
        super().__init__()
        self.render_mode   = render_mode
        self.stage         = stage

        self._model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self._data  = mujoco.MjData(self._model)
        self._viewer = None
        self._renderer = None

        # Cache sensor data addresses.
        self._acc_adr  = self._model.sensor("imu_acc").adr[0]
        self._gyro_adr = self._model.sensor("imu_gyro").adr[0]
        self._quat_adr = self._model.sensor("imu_quat").adr[0]

        obs_dim = 18 + 18 + 4 + 3 + 3 + 3 + 2 + self.CMD_DIM  # = 60
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(18,), dtype=np.float32)

        self._cmd            = np.zeros(self.CMD_DIM, dtype=np.float32)
        self._cycle_period   = 0
        self._cycle_leg_num  = TRIPOD_INIT.copy()
        self._step_count     = 0
        self._last_action    = np.zeros(18, dtype=np.float32)
        # gait_scale: 1.0 = full scripted gait scaffold, 0.0 = pure policy.
        # Updated externally by a training callback for progressive fade-out.
        self.gait_scale      = 1.0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)

        # Body starting pose
        self._data.qpos[2] = 0.13           # z height (matches widened stance)
        self._data.qpos[3] = 1.0            # quaternion w (upright)

        # Neutral standing pose
        self._data.qpos[7:] = NEUTRAL_POSE

        mujoco.mj_forward(self._model, self._data)

        # Sample command (stage-aware).
        self._cmd           = self._sample_cmd().astype(np.float32)
        self._cycle_period  = 0
        self._cycle_leg_num = TRIPOD_INIT.copy()
        self._step_count    = 0
        self._last_action   = np.zeros(18, dtype=np.float32)

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _body_frame_linvel(self):
        """World-frame body linear velocity rotated into body frame (yaw only).
        Cmd is body-relative (forward/strafe), so reward and obs use this too."""
        qw = float(self._data.qpos[3])
        qx = float(self._data.qpos[4])
        qy = float(self._data.qpos[5])
        qz = float(self._data.qpos[6])
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        c, s = math.cos(yaw), math.sin(yaw)
        vx_w = float(self._data.qvel[0])
        vy_w = float(self._data.qvel[1])
        vz_w = float(self._data.qvel[2])
        return ( c * vx_w + s * vy_w,
                -s * vx_w + c * vy_w,
                 vz_w)

    # ------------------------------------------------------------------
    def _sample_cmd(self):
        """Sample cmd vector based on current curriculum stage."""
        cmd = np.zeros(self.CMD_DIM)
        if self.stage == 1:
            # Cardinal directions + stop, fixed magnitude.
            choice = np.random.randint(5)
            options = [
                ( MAX_SPEED, 0.0),
                (-MAX_SPEED, 0.0),
                (0.0,  MAX_SPEED),
                (0.0, -MAX_SPEED),
                (0.0, 0.0),
            ]
            cmd[0], cmd[1] = options[choice]
        else:
            # Higher stages — implement when reached.
            cmd[0] = np.random.uniform(-MAX_SPEED, MAX_SPEED)
            cmd[1] = np.random.uniform(-MAX_SPEED, MAX_SPEED)
        return cmd

    # ------------------------------------------------------------------
    def step(self, action):
        # 1. Gait base angles
        foot_offsets = _tripod_gait(self._cmd[0], self._cmd[1],
                                    self._cycle_period, self._cycle_leg_num)
        base_angles = np.zeros(18)
        for i in range(6):
            fx = -INIT_FOOT_POS_X[i] + foot_offsets[i, 0]
            fy =  INIT_FOOT_POS_Y[i] + foot_offsets[i, 1]
            fz =  INIT_FOOT_POS_Z[i] - foot_offsets[i, 2]  # gait z is lift, IK z is depth
            c, f, t = _ik(fx, fy, fz, i)
            base_angles[i * 3]     = c
            base_angles[i * 3 + 1] = f
            base_angles[i * 3 + 2] = t

        # 2. Combine scaled gait scaffold + policy residual.
        # As gait_scale -> 0, the gait deviation shrinks toward NEUTRAL_POSE
        # and the residual scale grows so the policy can fully control.
        residual_scale = (
            self.RESIDUAL_SCALE_MIN
            + (1.0 - self.gait_scale)
              * (self.RESIDUAL_SCALE_MAX - self.RESIDUAL_SCALE_MIN)
        )
        scaffold = NEUTRAL_POSE + (base_angles - NEUTRAL_POSE) * self.gait_scale
        target   = scaffold + action * residual_scale
        self._data.ctrl[:] = target
        mujoco.mj_step(self._model, self._data)

        # 3. Advance gait cycle
        self._cycle_period += 1
        if self._cycle_period >= CYCLE_LENGTH:
            self._cycle_period = 0
            self._cycle_leg_num = 1 - self._cycle_leg_num  # flip 0↔1

        self._step_count += 1

        # 4. Reward — tracking measured in BODY frame (cmd is body-relative).
        vx, vy, _ = self._body_frame_linvel()
        vel_err   = np.array([vx - self._cmd[0], vy - self._cmd[1]])
        tracking  = float(np.exp(-np.dot(vel_err, vel_err)))
        ctrl_cost = self.CTRL_COST_W * float(np.sum(action**2))

        # Stability — keep body level and at nominal height.
        body_z   = float(self._data.qpos[2])
        z_bob    = self.Z_BOB_W * (body_z - self.REF_BODY_Z) ** 2
        qx, qy   = float(self._data.qpos[4]), float(self._data.qpos[5])
        tilt_pen = self.TILT_W * (qx * qx + qy * qy)

        # Smoothness — discourage jolts, spasms, and drag-style locomotion.
        action_delta = action - self._last_action
        action_rate  = self.ACTION_RATE_W * float(np.dot(action_delta, action_delta))
        joint_vel    = self.JOINT_VEL_W   * float(np.dot(self._data.qvel[6:], self._data.qvel[6:]))
        angvel       = self.ANGVEL_W      * float(np.dot(self._data.qvel[3:6], self._data.qvel[3:6]))

        # Yaw tracking — keep heading stable unless cmd[2] (wz) requests rotation.
        yaw_err  = float(self._data.qvel[5] - self._cmd[2])
        yaw_pen  = self.YAW_RATE_W * yaw_err * yaw_err

        reward = (tracking + self.SURVIVE_BONUS
                  - ctrl_cost - z_bob - tilt_pen
                  - action_rate - joint_vel - angvel - yaw_pen)

        self._last_action = action.copy()

        # 5. Termination — fallen if too low OR tipped past ~60° from upright
        tilt_sq    = qx * qx + qy * qy
        terminated = (body_z < self.MIN_Z) or (tilt_sq > self.MAX_TILT_SQ)
        truncated  = False

        if self.render_mode == "human":
            self._render_human()

        return self._get_obs(), reward, terminated, truncated, {
            "x_velocity": vx, "y_velocity": vy,
            "tracking_reward": tracking, "body_z": body_z,
            "z_bob_penalty": z_bob, "tilt_penalty": tilt_pen,
            "action_rate_penalty": action_rate,
            "joint_vel_penalty": joint_vel,
            "angvel_penalty": angvel,
            "yaw_penalty": yaw_pen,
        }

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._model)
            self._renderer.update_scene(self._data)
            return self._renderer.render()

    def _render_human(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    def _get_obs(self):
        qpos = self._data.qpos
        qvel = self._data.qvel
        sd   = self._data.sensordata
        imu_quat  = sd[self._quat_adr : self._quat_adr + 4]
        imu_gyro  = sd[self._gyro_adr : self._gyro_adr + 3]
        imu_accel = sd[self._acc_adr  : self._acc_adr  + 3]
        body_linvel = np.asarray(self._body_frame_linvel(), dtype=np.float32)
        phase = self._cycle_period / CYCLE_LENGTH
        return np.concatenate([
            qpos[7:],                       # joint positions (18)
            qvel[6:],                       # joint velocities (18)
            imu_quat,                       # body quaternion from IMU (4)
            imu_gyro,                       # body angular velocity from IMU (3)
            imu_accel,                      # body linear accel from IMU (3)
            body_linvel,                    # body-frame linear velocity (3) — privileged, drop later
            [math.sin(2 * math.pi * phase),
             math.cos(2 * math.pi * phase)],# gait phase (2)
            self._cmd,                      # cmd vector (9)
        ]).astype(np.float32)
