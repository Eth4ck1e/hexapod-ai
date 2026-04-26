"""
pilot.py — keyboard-controlled teleop for the hexapod manual gait.

Drive the bot around with your keyboard, manually testing the cmd vector.
Each key press advances the corresponding cmd slot by a small step. Hold keys
to keep advancing (OS auto-repeat handles this). SPACE zeros the cmd. ESC or
close viewer to quit.

Run:
    mjpython pilot.py        (macOS)
    python   pilot.py        (Linux/Windows)

Keyboard layout:
    Translation        W / S       vx forward / back        (m/s)
                       A / D       vy left strafe / right
    Yaw rate           Q / E       wz CCW / CW              (rad/s)
    Body pitch         I / K       pitch up / down          (rad)
    Body roll          J / L       roll right / left
    Stance height      R / F       body up / down           (m, delta)
    Stance width       T / G       wider / narrower
    Body shift X       Z / X       fore / aft
    Body shift Y       C / V       left / right
    SPACE              zero all cmd slots
    P                  print current cmd to terminal
    1/2/3              snap to preset (forward / spin / stand)
"""

import math
import time
import numpy as np
import mujoco
import mujoco.viewer

from gait import Controller, NEUTRAL_POSE


MODEL_PATH = "models/phantomx.xml"

# Per-key step sizes — small enough that auto-repeat gives a smooth feel.
STEP_VX     = 0.005                # m/s
STEP_VY     = 0.005                # m/s
STEP_WZ     = 0.05                 # rad/s
STEP_PITCH  = math.radians(2.0)    # rad
STEP_ROLL   = math.radians(2.0)
STEP_HEIGHT = 0.002                # m
STEP_WIDTH  = 0.002
STEP_SHIFT  = 0.005

# Cmd-slot clamps — keep within reachable workspace.
def make_clamps(ctrl):
    return {
        0: (-ctrl.MAX_SPEED,    ctrl.MAX_SPEED),
        1: (-ctrl.MAX_SPEED,    ctrl.MAX_SPEED),
        2: (-ctrl.MAX_YAW_RATE, ctrl.MAX_YAW_RATE),
        3: (-math.radians(20),  math.radians(20)),
        4: (-math.radians(20),  math.radians(20)),
        5: (-0.025,             0.025),
        6: (-0.020,             0.020),
        7: (-0.040,             0.040),
        8: (-0.040,             0.040),
    }


def fmt_cmd(cmd):
    return (f"vx={cmd[0]:+.3f} vy={cmd[1]:+.3f} wz={cmd[2]:+.3f}  "
            f"pitch={math.degrees(cmd[3]):+5.1f}° roll={math.degrees(cmd[4]):+5.1f}°  "
            f"h={cmd[5]*1000:+5.1f}mm w={cmd[6]*1000:+5.1f}mm  "
            f"sx={cmd[7]*1000:+5.1f}mm sy={cmd[8]*1000:+5.1f}mm")


def make_key_callback(state, ctrl):
    """Returns a key_callback closure for mujoco.viewer.launch_passive."""
    clamps = make_clamps(ctrl)

    def adjust(idx, delta):
        lo, hi = clamps[idx]
        state["cmd"][idx] = max(lo, min(hi, state["cmd"][idx] + delta))

    KEY_MAP = {
        # ord() of the key character → callable
        ord('W'): lambda: adjust(0, +STEP_VX),
        ord('S'): lambda: adjust(0, -STEP_VX),
        ord('A'): lambda: adjust(1, +STEP_VY),    # +Y = body left
        ord('D'): lambda: adjust(1, -STEP_VY),
        ord('Q'): lambda: adjust(2, +STEP_WZ),    # +wz = CCW (RH around +Z)
        ord('E'): lambda: adjust(2, -STEP_WZ),
        ord('I'): lambda: adjust(3, +STEP_PITCH), # nose up
        ord('K'): lambda: adjust(3, -STEP_PITCH),
        ord('J'): lambda: adjust(4, +STEP_ROLL),  # right side up
        ord('L'): lambda: adjust(4, -STEP_ROLL),
        ord('R'): lambda: adjust(5, -STEP_HEIGHT),# - = body up
        ord('F'): lambda: adjust(5, +STEP_HEIGHT),# + = body down
        ord('T'): lambda: adjust(6, +STEP_WIDTH), # wider
        ord('G'): lambda: adjust(6, -STEP_WIDTH), # narrower
        ord('Z'): lambda: adjust(7, +STEP_SHIFT),
        ord('X'): lambda: adjust(7, -STEP_SHIFT),
        ord('C'): lambda: adjust(8, +STEP_SHIFT),
        ord('V'): lambda: adjust(8, -STEP_SHIFT),
        ord(' '): lambda: state["cmd"].fill(0.0),
        ord('P'): lambda: print(f"  CMD: {fmt_cmd(state['cmd'])}"),
        ord('1'): lambda: _preset_forward(state, ctrl),
        ord('2'): lambda: _preset_spin(state, ctrl),
        ord('3'): lambda: state["cmd"].fill(0.0),
    }

    def callback(keycode):
        fn = KEY_MAP.get(keycode)
        if fn is not None:
            fn()
            state["dirty"] = True

    return callback


def _preset_forward(state, ctrl):
    state["cmd"].fill(0.0)
    state["cmd"][0] = ctrl.MAX_SPEED


def _preset_spin(state, ctrl):
    state["cmd"].fill(0.0)
    state["cmd"][2] = ctrl.MAX_YAW_RATE


def main():
    ctrl  = Controller(MODEL_PATH)
    state = {"cmd": np.zeros(9), "dirty": False}

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

    print("\npilot.py — keyboard teleop")
    print(f"  MAX_SPEED    = {ctrl.MAX_SPEED:.4f} m/s")
    print(f"  MAX_YAW_RATE = {ctrl.MAX_YAW_RATE:.4f} rad/s")
    print()
    print("  Controls:")
    print("    W/S  vx forward/back        I/K  pitch up/down")
    print("    A/D  vy left/right strafe   J/L  roll  right/left")
    print("    Q/E  yaw  CCW / CW          R/F  height up/down")
    print("                                T/G  width  wider/narrower")
    print("    Z/X  shift X fore/aft       C/V  shift Y left/right")
    print("    SPACE  zero all      P  print cmd")
    print("    1  preset: walk forward     2  preset: spin CCW     3  zero")
    print()

    log_every_steps = int(0.5 / model.opt.timestep)   # ~2 Hz cmd printout
    step_count = 0

    with mujoco.viewer.launch_passive(model, data,
                                      key_callback=make_key_callback(state, ctrl)) as viewer:
        sim_t0  = data.time
        wall_t0 = time.time()
        while viewer.is_running():
            t = data.time - sim_t0
            data.ctrl[:] = ctrl.predict(state["cmd"], t)
            mujoco.mj_step(model, data)
            viewer.sync()

            # Periodic terminal feedback so user sees their commanded state.
            step_count += 1
            if state["dirty"] or step_count % log_every_steps == 0:
                if state["dirty"] or np.any(state["cmd"] != 0.0):
                    print(f"  t={t:6.1f}s  {fmt_cmd(state['cmd'])}")
                state["dirty"] = False

            lag = (wall_t0 + (data.time - sim_t0)) - time.time()
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
