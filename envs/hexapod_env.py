"""
HexapodEnv — gymnasium environment for PhantomX hexapod RL.

Thin wrapper around gait.Controller. The Controller produces a scaffold of
joint targets each step from the current cmd vector; the policy outputs an
18-dim residual that is mixed in. As gait_scale fades from 1.0 → 0.0 over
training, the scaffold contribution shrinks and the residual takes over.

Cmd vector (matches gait.Controller — physical units):
  [0] vx           m/s     body forward velocity
  [1] vy           m/s     body lateral velocity (left = +)
  [2] wz           rad/s   body yaw rate (CCW = +)
  [3] pitch        rad     body pitch target (nose up = +)
  [4] roll         rad     body roll target (right side up = +)
  [5] height_delta m       stance height delta (- = body raised)
  [6] width_delta  m       stance width delta
  [7] shift_x      m       body shift in body +X
  [8] shift_y      m       body shift in body +Y

Curriculum stages — controlled by `stage` arg at construction. Each stage
unlocks a subset of cmd slots; locked slots are sampled as 0.

  Stage 1: vx, vy                                       (translation)
  Stage 2: + wz, height_delta, width_delta              (+ spin + stance)
  Stage 3: + pitch, roll, shift_x, shift_y              (+ pose, shift)
  Stage 4: same as 3 + novelty bonuses + body_linvel    (autonomous + stretch)
           dropped from obs

Reward (minimal — see project_gait_training_architecture.md):
  + tracking_exp(-||cmd-actual||²)        per active stage cmd slots
  + 0.1 * survive_bonus                   constant per step
  - 0.01 * action_delta²                  smoothness
  + stage 4: novelty bonuses              (max-speed, sustained inverted, etc)
  terminate on fall or extreme tilt
"""

import math
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

from gait import Controller, NEUTRAL_POSE


MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/phantomx.xml")


# Per-stage cmd-slot enable mask. 1 = sampleable / tracked, 0 = forced to zero.
# Index matches cmd vector slots [vx, vy, wz, pitch, roll, height, width, sx, sy].
STAGE_CMD_MASK = {
    1: np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
    2: np.array([1, 1, 1, 0, 0, 1, 1, 0, 0]),
    3: np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
    4: np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
}

# Cmd slots whose tracking actually matters for reward. Width and shifts are
# scaffold pass-through (no direct kinematic equivalent to track); they're
# enabled in obs/cmd but don't contribute reward terms. Locomotion slots get
# dense tracking reward.
REWARD_TRACK_MASK = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0])


class HexapodEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    CMD_DIM            = 9
    ACTION_DIM         = 18

    # Residual range scales inversely with gait_scale: when scaffold is heavy,
    # residual is small; when scaffold is gone, residual has full authority.
    RESIDUAL_SCALE_MIN = 0.10   # rad — gait_scale = 1.0
    RESIDUAL_SCALE_MAX = 0.80   # rad — gait_scale = 0.0

    SURVIVE_BONUS    = 0.10
    ACTION_RATE_W    = 0.01
    ANGVEL_W         = 0.02     # damps body roll/pitch rate (wobble)

    MIN_Z          = 0.06   # body height below this = fallen
    MAX_TILT_SQ    = 0.50   # qx² + qy² above this = flipped (~60°)
    REF_BODY_Z     = 0.13   # nominal standing height (used to keep height tracking sensible)

    # Random sampling magnitudes per stage.
    SAMPLE_RANGES = {
        # slot:  (lo, hi)
        # Speeds quote symmetric ±MAX. Yaw is signed. Pose targets are bounded angles.
        # Stance/shift use small symmetric ranges for now.
        0: (-1.0, 1.0),     # vx (× MAX_SPEED — scaled at sample time)
        1: (-1.0, 1.0),     # vy
        2: (-1.0, 1.0),     # wz (× MAX_YAW_RATE)
        3: (-math.radians(15), math.radians(15)),
        4: (-math.radians(15), math.radians(15)),
        5: (-0.020, 0.020),
        6: (-0.015, 0.015),
        7: (-0.030, 0.030),
        8: (-0.030, 0.030),
    }

    def __init__(self, render_mode=None, stage=1):
        super().__init__()
        self.render_mode = render_mode
        self.stage       = stage

        # Mujoco
        self._model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self._data  = mujoco.MjData(self._model)
        self._dt    = float(self._model.opt.timestep)

        # Gait controller — the analytical scaffold.
        self._ctrl = Controller(MODEL_PATH)

        # Per-slot tracking error scales. Normalizes errors so that a "full
        # deviation" on any slot contributes roughly 1.0 to the error norm
        # squared. Without this, slots with small physical magnitudes (e.g.,
        # height in meters) barely affect the reward while slots with larger
        # magnitudes (e.g., velocities) dominate. Inverse-multiplied, so a
        # full-scale deviation in any slot drops the gaussian reward by ~63%.
        FULL_HEIGHT_DEV = 0.025         # 25 mm height drift = "full" penalty
        FULL_PITCH_DEV  = math.radians(15)
        FULL_ROLL_DEV   = math.radians(15)
        self._err_inv_scales = np.array([
            1.0 / self._ctrl.MAX_SPEED,        # vx
            1.0 / self._ctrl.MAX_SPEED,        # vy
            1.0 / self._ctrl.MAX_YAW_RATE,     # wz
            1.0 / FULL_PITCH_DEV,              # pitch
            1.0 / FULL_ROLL_DEV,               # roll
            1.0 / FULL_HEIGHT_DEV,             # height_delta
            0.0, 0.0, 0.0,                     # width, shifts (not tracked)
        ], dtype=np.float64)

        # Cache sensor data addresses.
        self._acc_adr  = self._model.sensor("imu_acc").adr[0]
        self._gyro_adr = self._model.sensor("imu_gyro").adr[0]
        self._quat_adr = self._model.sensor("imu_quat").adr[0]

        # Spaces. Obs layout (78 dim, see _get_obs):
        #   joint_pos(18) + joint_vel(18) + imu_quat(4) + imu_gyro(3) + imu_accel(3)
        #   + scaffold_hint(18, foot positions in body frame)
        #   + gait_phase(2) + cmd(9) + body_linvel(3)
        obs_dim = 18 + 18 + 4 + 3 + 3 + 18 + 2 + self.CMD_DIM + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(self.ACTION_DIM,), dtype=np.float32)

        # Episode-mutable state.
        self._cmd         = np.zeros(self.CMD_DIM, dtype=np.float32)
        self._sim_time    = 0.0
        self._step_count  = 0
        self._last_action = np.zeros(self.ACTION_DIM, dtype=np.float32)

        # gait_scale is mutated externally by the GaitFadeCallback.
        self.gait_scale = 1.0

        self._viewer   = None
        self._renderer = None

    # ------------------------------------------------------------------
    # Standard gym lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)

        self._data.qpos[2]    = 0.18           # spawn just above the ground
        self._data.qpos[3]    = 1.0            # quat w (upright)
        self._data.qpos[7:25] = NEUTRAL_POSE
        self._data.ctrl[:]    = NEUTRAL_POSE
        mujoco.mj_forward(self._model, self._data)

        self._cmd         = self._sample_cmd().astype(np.float32)
        self._sim_time    = 0.0
        self._step_count  = 0
        self._last_action = np.zeros(self.ACTION_DIM, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # 1. Scaffold joint targets from the analytical controller.
        scaffold_joints = self._ctrl.predict(self._cmd, self._sim_time).astype(np.float32)

        # 2. Mix scaffold + residual. As gait_scale → 0 the scaffold relaxes
        # to NEUTRAL_POSE while residual range grows.
        residual_scale = (
            self.RESIDUAL_SCALE_MIN
            + (1.0 - self.gait_scale)
              * (self.RESIDUAL_SCALE_MAX - self.RESIDUAL_SCALE_MIN)
        )
        target = (NEUTRAL_POSE
                  + (scaffold_joints - NEUTRAL_POSE) * self.gait_scale
                  + action * residual_scale)
        self._data.ctrl[:] = target
        mujoco.mj_step(self._model, self._data)
        self._sim_time   += self._dt
        self._step_count += 1

        # 3. Reward — tracking on locomotion cmd slots only.
        reward, reward_info = self._compute_reward(action)

        # 4. Termination on fall or extreme tilt.
        body_z = float(self._data.qpos[2])
        qx, qy = float(self._data.qpos[4]), float(self._data.qpos[5])
        tilt_sq = qx*qx + qy*qy
        terminated = (body_z < self.MIN_Z) or (tilt_sq > self.MAX_TILT_SQ)
        truncated  = False

        self._last_action = action.copy()

        if self.render_mode == "human":
            self._render_human()

        info = {**reward_info, "body_z": body_z, "tilt_sq": float(tilt_sq)}
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Cmd sampling — stage-aware
    # ------------------------------------------------------------------
    def _sample_cmd(self):
        """Sample a cmd vector for this episode. Stage-locked slots stay zero;
        unlocked slots are drawn from their allowed range.

        Translation magnitude is FIXED at MAX_SPEED (only the heading is random).
        Speed-magnitude variation is intentionally left out of training for now —
        the gait library currently models speed via stride_scale only, but real
        speed is mostly cadence (gait_period) with a smaller stride contribution.
        We'll re-enable variable-speed sampling once the gait library has proper
        period+scale speed modulation. See feedback_speed_control.md.
        """
        mask = STAGE_CMD_MASK[self.stage]
        cmd  = np.zeros(self.CMD_DIM, dtype=np.float64)

        # Slots 0, 1 — translation. Random heading at fixed MAX_SPEED magnitude.
        if mask[0] and mask[1]:
            heading = self.np_random.uniform(0.0, 2.0 * math.pi)
            cmd[0]  = self._ctrl.MAX_SPEED * math.cos(heading)
            cmd[1]  = self._ctrl.MAX_SPEED * math.sin(heading)

        # Remaining slots — independent uniform sampling.
        for slot in range(2, self.CMD_DIM):
            if mask[slot] == 0:
                continue
            lo, hi = self.SAMPLE_RANGES[slot]
            v = self.np_random.uniform(lo, hi)
            if slot == 2:
                v *= self._ctrl.MAX_YAW_RATE
            cmd[slot] = v

        return cmd

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, action):
        # Measured kinematic state for tracking comparison.
        vx, vy, _    = self._body_frame_linvel()
        wz           = float(self._data.qvel[5])
        pitch, roll  = self._body_pitch_roll()
        height_delta = float(self._data.qpos[2]) - self.REF_BODY_Z
        actual = np.array([vx, vy, wz, pitch, roll, height_delta, 0.0, 0.0, 0.0],
                          dtype=np.float64)

        # IMPORTANT: tracking compares ALL body kinematics regardless of stage,
        # NOT just slots commanded by the current stage. Stage masks only what's
        # commanded during sampling; the reward always punishes unwanted IMU
        # deviation. Even in stage 1 (where cmd[wz]=cmd[pitch]=cmd[roll]=cmd[height]=0),
        # the bot is penalized for yawing, pitching, rolling, or bobbing —
        # because cmd is zero on those slots and any actual deviation creates
        # tracking error. This is what makes the bot learn smooth, controlled
        # locomotion instead of jerky lurching that happens to track velocity.
        # Errors are normalized per-slot (see _err_inv_scales) so each axis
        # contributes commensurately rather than being dominated by velocity.
        err = (self._cmd - actual) * self._err_inv_scales * REWARD_TRACK_MASK

        # Gaussian-shaped tracking reward, peaks at 1.0 when cmd matches actual.
        tracking = float(math.exp(-float(np.dot(err, err))))

        # Action smoothness — penalize fast changes in the policy's residual
        # between consecutive ticks. Encourages smooth, gradual joint motion.
        action_delta = action - self._last_action
        action_rate  = self.ACTION_RATE_W * float(np.dot(action_delta, action_delta))

        # Body angular-velocity damping — penalize WOBBLE that doesn't show up
        # cleanly in absolute pitch/roll tracking. Excludes yaw rate (qvel[5])
        # since that's already tracked via wz. Targets qvel[3] (roll rate) and
        # qvel[4] (pitch rate) which are unbounded oscillation modes if not
        # damped.
        body_angvel = self._data.qvel[3:5]   # roll_rate, pitch_rate (world)
        angvel_pen  = self.ANGVEL_W * float(np.dot(body_angvel, body_angvel))

        reward = tracking + self.SURVIVE_BONUS - action_rate - angvel_pen

        # Stage 4 stretch-goal bonuses (placeholders; tune when we get there).
        novelty = 0.0
        if self.stage == 4:
            novelty = self._novelty_bonus()
            reward += novelty

        return reward, {
            "tracking_reward": tracking,
            "action_rate_pen": action_rate,
            "angvel_pen":      angvel_pen,
            "novelty_bonus":   novelty,
        }

    def _novelty_bonus(self):
        """Stage 4 only. Reward unusual achievements: high speeds, sustained
        inverted body orientation (rolling), etc. Placeholder until we tune."""
        speed = math.hypot(*self._body_frame_linvel()[:2])
        # Linear bonus for exceeding scaffold's max speed.
        speed_excess = max(0.0, speed - self._ctrl.MAX_SPEED)
        return 0.5 * speed_excess

    # ------------------------------------------------------------------
    # Helpers — kinematic readouts
    # ------------------------------------------------------------------
    def _body_frame_linvel(self):
        """World-frame body linear velocity rotated into body frame (yaw only)."""
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

    def _body_pitch_roll(self):
        """Body pitch & roll relative to world (rad). Standard q→Euler."""
        qw = float(self._data.qpos[3])
        qx = float(self._data.qpos[4])
        qy = float(self._data.qpos[5])
        qz = float(self._data.qpos[6])
        sinp  = 2.0 * (qw * qy - qz * qx)
        sinp  = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)
        sinr  = 2.0 * (qw * qx + qy * qz)
        cosr  = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll  = math.atan2(sinr, cosr)
        return pitch, roll

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self):
        qpos = self._data.qpos
        qvel = self._data.qvel
        sd   = self._data.sensordata
        imu_quat  = sd[self._quat_adr : self._quat_adr + 4]
        imu_gyro  = sd[self._gyro_adr : self._gyro_adr + 3]
        imu_accel = sd[self._acc_adr  : self._acc_adr  + 3]

        # Scaffold hint: where the analytical gait wants each foot, in body frame.
        foot_targets = self._ctrl.compute_foot_targets(self._cmd, self._sim_time)
        scaffold_hint = foot_targets.flatten()   # (6, 3) → (18,)

        phase = self._ctrl.get_phase(self._sim_time)
        body_linvel = np.asarray(self._body_frame_linvel(), dtype=np.float32)

        # Stage 4: drop privileged body_linvel. Implement as zero-out for now;
        # noise injection / IMU-only estimation is a later refinement.
        if self.stage == 4:
            body_linvel = np.zeros(3, dtype=np.float32)

        return np.concatenate([
            qpos[7:],                              # joint positions     (18)
            qvel[6:],                              # joint velocities    (18)
            imu_quat,                              # body quaternion     ( 4)
            imu_gyro,                              # body angular vel    ( 3)
            imu_accel,                             # body linear accel   ( 3)
            scaffold_hint,                         # foot-target hint    (18)
            [math.sin(2 * math.pi * phase),
             math.cos(2 * math.pi * phase)],      # gait phase           ( 2)
            self._cmd,                             # cmd vector          ( 9)
            body_linvel,                           # body linvel         ( 3)
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Render
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
