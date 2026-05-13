"""
demo.py — automatic showcase of all gait library overlays.

Runs through every cmd-vector capability in sequence with on-screen labels and
terminal annotations. Use this to verify each overlay visually after changes.

Run:
    mjpython demo.py        (macOS)
    python   demo.py        (Linux/Windows)

Each phase sets the cmd vector for a few seconds, then advances. The full
sequence loops; ESC or close viewer to quit.
"""

import math
import time
import numpy as np
import mujoco
import mujoco.viewer

from gait import Controller, NEUTRAL_POSE


MODEL_PATH = "models/phantomx.xml"


# ----------------------------------------------------------------------------
# Helpers — small builders for cmd vectors. Each returns a 9-vec.
# ----------------------------------------------------------------------------
def _cmd(**slots):
    """Build a cmd vector from named slots."""
    cmd = np.zeros(9)
    name_to_idx = {
        "vx": 0, "vy": 1, "wz": 2,
        "pitch": 3, "roll": 4,
        "height": 5, "width": 6,
        "shift_x": 7, "shift_y": 8,
    }
    for k, v in slots.items():
        cmd[name_to_idx[k]] = v
    return cmd


def _walk(ctrl, heading_rad, speed_frac=1.0):
    """Walk command at fraction of MAX_SPEED in given heading direction."""
    speed = ctrl.MAX_SPEED * speed_frac
    return _cmd(vx=speed * math.cos(heading_rad),
                vy=speed * math.sin(heading_rad))


def _spin(ctrl, sign, scale=1.0):
    return _cmd(wz=sign * ctrl.MAX_YAW_RATE * scale)


# ----------------------------------------------------------------------------
# Demo phases. Each entry: (duration_seconds, label, cmd_function(t_in_phase, ctrl)).
# ----------------------------------------------------------------------------
def build_sequence(ctrl):
    DEG = math.radians

    return [
        (3.0,  "Standing neutral",
            lambda t: np.zeros(9)),

        # --- Pitch ---
        (2.5,  "Pitch UP +15° (nose up)",
            lambda t: _cmd(pitch=DEG(15))),
        (2.5,  "Pitch DOWN -15° (nose down)",
            lambda t: _cmd(pitch=DEG(-15))),
        (5.0,  "Pitch sweep ±15°",
            lambda t: _cmd(pitch=DEG(15) * math.sin(2*math.pi*t/5))),

        # --- Roll ---
        (2.5,  "Roll RIGHT +15° (right side up)",
            lambda t: _cmd(roll=DEG(15))),
        (2.5,  "Roll LEFT -15°",
            lambda t: _cmd(roll=DEG(-15))),
        (5.0,  "Roll sweep ±15°",
            lambda t: _cmd(roll=DEG(15) * math.sin(2*math.pi*t/5))),

        # --- Combined pitch+roll: body traces a tilt circle ---
        (6.0,  "Pitch+roll circle (body wobble)",
            lambda t: _cmd(pitch=DEG(12) * math.cos(2*math.pi*t/6),
                           roll=DEG(12)  * math.sin(2*math.pi*t/6))),

        # --- Shift ---
        (2.5,  "Body shift FORWARD +30mm",
            lambda t: _cmd(shift_x=+0.030)),
        (2.5,  "Body shift BACKWARD -30mm",
            lambda t: _cmd(shift_x=-0.030)),
        (2.5,  "Body shift LEFT +30mm",
            lambda t: _cmd(shift_y=+0.030)),
        (2.5,  "Body shift RIGHT -30mm",
            lambda t: _cmd(shift_y=-0.030)),
        (6.0,  "Body shift CIRCLE",
            lambda t: _cmd(shift_x=0.025 * math.cos(2*math.pi*t/6),
                           shift_y=0.025 * math.sin(2*math.pi*t/6))),

        # --- Stance ---
        (5.0,  "Stance height squat / rise ±20mm",
            lambda t: _cmd(height=0.020 * math.sin(2*math.pi*t/5))),
        (5.0,  "Stance width narrow / wide ±15mm",
            lambda t: _cmd(width=0.015 * math.sin(2*math.pi*t/5))),

        # --- Translation walking ---
        (5.0,  "Walk FORWARD",     lambda t: _walk(ctrl, 0.0)),
        (5.0,  "Walk BACKWARD",    lambda t: _walk(ctrl, math.pi)),
        (5.0,  "Strafe LEFT",      lambda t: _walk(ctrl, math.pi/2)),
        (5.0,  "Strafe RIGHT",     lambda t: _walk(ctrl, -math.pi/2)),
        (10.0, "Walk in a CIRCLE (heading sweep)",
            lambda t: _walk(ctrl, 2*math.pi*t/10)),

        # --- Spin ---
        (8.0,  "Spin in place CCW",  lambda t: _spin(ctrl, +1)),
        (8.0,  "Spin in place CW",   lambda t: _spin(ctrl, -1)),

        # --- Combined locomotion + overlay ---
        (8.0,  "Walk forward + pitch wobble",
            lambda t: _walk(ctrl, 0.0) + _cmd(pitch=DEG(10)*math.sin(2*math.pi*t/4))),
        (8.0,  "Walk forward + roll wobble",
            lambda t: _walk(ctrl, 0.0) + _cmd(roll=DEG(10)*math.sin(2*math.pi*t/4))),
        (8.0,  "Walk forward at HALF SPEED + low stance",
            lambda t: _walk(ctrl, 0.0, speed_frac=0.5) + _cmd(height=+0.020)),
        (8.0,  "Spin + body shift_y wobble",
            lambda t: _spin(ctrl, +1) + _cmd(shift_y=0.020*math.sin(2*math.pi*t/4))),

        # --- Finale: everything together ---
        (15.0, "EVERYTHING — heading sweep + pitch wobble + height bob + width breath",
            lambda t: _walk(ctrl, 2*math.pi*t/15)
                      + _cmd(pitch  = DEG(8) * math.sin(2*math.pi*t/3),
                             roll   = DEG(8) * math.cos(2*math.pi*t/3),
                             height = 0.012  * math.sin(2*math.pi*t/2),
                             width  = 0.010  * math.cos(2*math.pi*t/4))),

        (3.0,  "Returning to neutral", lambda t: np.zeros(9)),
    ]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ctrl = Controller(MODEL_PATH)
    sequence = build_sequence(ctrl)
    cycle_total = sum(d for d, _, _ in sequence)

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

    print(f"\ndemo.py — {len(sequence)} phases, {cycle_total:.0f}s per full loop")
    print(f"  MAX_SPEED    = {ctrl.MAX_SPEED:.4f} m/s")
    print(f"  MAX_YAW_RATE = {ctrl.MAX_YAW_RATE:.4f} rad/s")

    last_phase_idx = -1
    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_t0  = data.time
        wall_t0 = time.time()
        while viewer.is_running():
            t = data.time - sim_t0
            t_in_cycle = t % cycle_total

            cumulative = 0.0
            for idx, (dur, label, fn) in enumerate(sequence):
                if t_in_cycle < cumulative + dur:
                    t_phase = t_in_cycle - cumulative
                    cmd     = fn(t_phase)
                    if idx != last_phase_idx:
                        print(f"  t={t:6.1f}s  [{idx+1:2d}/{len(sequence)}]  {label}")
                        last_phase_idx = idx
                    break
                cumulative += dur
            else:
                cmd = np.zeros(9)

            data.ctrl[:] = ctrl.predict(cmd, t)
            mujoco.mj_step(model, data)
            viewer.sync()
            lag = (wall_t0 + (data.time - sim_t0)) - time.time()
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
