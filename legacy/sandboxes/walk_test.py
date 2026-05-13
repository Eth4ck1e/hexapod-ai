"""
Runs the scripted tripod gait with no policy (zero residuals).
Use this to validate the base controller before training.

Controls:
  W/S  — increase/decrease forward speed
  A/D  — strafe left/right
  Q/E  — yaw left/right (not yet in gait, placeholder)
  R    — reset
  ESC  — quit
"""
import math
import numpy as np
import mujoco
import mujoco.viewer

from envs.hexapod_env import (
    HexapodEnv, NEUTRAL_POSE, CYCLE_LENGTH, TRIPOD_INIT,
    _tripod_gait, _ik,
    INIT_FOOT_POS_X, INIT_FOOT_POS_Y, INIT_FOOT_POS_Z,
    MAX_SPEED,
)

model = mujoco.MjModel.from_xml_path("models/phantomx.xml")
data  = mujoco.MjData(model)

def reset():
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.13
    data.qpos[3] = 1.0
    data.qpos[7:] = NEUTRAL_POSE
    data.ctrl[:]  = NEUTRAL_POSE
    mujoco.mj_forward(model, data)

reset()

cmd        = np.array([MAX_SPEED, 0.0])
cycle      = 0
leg_num    = TRIPOD_INIT.copy()

# How many physics steps per gait step.
# At 0.005s/step, 4 steps = 0.02s per gait step → ~1 full cycle/second.
SIM_STEPS_PER_GAIT = 4

print(f"Walking test | cmd=({cmd[0]:+.3f}, {cmd[1]:+.3f}) m/s")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        offsets = _tripod_gait(cmd[0], cmd[1], cycle, leg_num)
        angles  = np.zeros(18)
        for i in range(6):
            # Match hexapod_ros ik.cpp: right legs (0..2) negate foot.y;
            # X gait offset is NOT mirrored.
            sign = -1.0 if i < 3 else 1.0
            fx = -INIT_FOOT_POS_X[i] + offsets[i, 0]
            fy =  INIT_FOOT_POS_Y[i] + sign * offsets[i, 1]
            fz =  INIT_FOOT_POS_Z[i] - offsets[i, 2]
            c, f, t = _ik(fx, fy, fz, i)
            # MJCF: all coxa axes are +Z (so +coxa rotates left/right legs
            # in opposite body-frame directions); femur/tibia axes are
            # flipped on left vs right. Negate all three for left legs.
            if i >= 3:
                c = -c
                f = -f
                t = -t
            angles[i*3]   = c
            angles[i*3+1] = f
            angles[i*3+2] = t

        data.ctrl[:] = angles

        for _ in range(SIM_STEPS_PER_GAIT):
            mujoco.mj_step(model, data)
            viewer.sync()

        cycle += 1
        if cycle >= CYCLE_LENGTH:
            cycle   = 0
            leg_num = 1 - leg_num

print("Done.")
