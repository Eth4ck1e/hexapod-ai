"""
simple_gait.py — demo viewer for the gait library.

Thin wrapper around gait.Controller that runs the mujoco viewer with
configurable cycle demos for each cmd-vector slot. Used to visually verify
the analytical scaffold; not used in training.

Run:
    mjpython simple_gait.py        (macOS — required for the viewer)
    python   simple_gait.py        (Linux/Windows)

Toggle demos at the bottom of the file. All demos are driven by the same
9-dim cmd vector; the only difference is what cycles dynamically vs holds.
"""

import math
import time
import numpy as np
import mujoco
import mujoco.viewer

from gait import Controller, NEUTRAL_POSE


MODEL_PATH = "models/phantomx.xml"

# ============================================================================
# CYCLE INFRASTRUCTURE
# ============================================================================
# Each cycle helper returns the current value at sim time t, dwelling at each
# entry for some duration before stepping to the next, looping forever.
# Prints to stdout when the value changes so you can correlate motion with cmd.

def _cycle_value(t, values, dwell_s, state, label, fmt):
    """Generic cycle stepper. `state` = {"last_idx": -1}; mutates state in-place."""
    idx = int(t // dwell_s) % len(values)
    if idx != state["last_idx"]:
        print(f"  t={t:5.1f}s  {label} → {fmt(values[idx])}")
        state["last_idx"] = idx
    return values[idx]


# ----------------- HEADING CYCLE (cmd vx, vy direction) -----------------
CYCLE_HEADINGS    = False
HEADING_CYCLE_DEG = list(range(0, 360, 30))
HEADING_DWELL_SEC = 3.0
HEADING            = 0.0   # static fallback (rad)
_heading_state = {"last_idx": -1}

def current_heading(t):
    if not CYCLE_HEADINGS:
        return HEADING
    deg = _cycle_value(t, HEADING_CYCLE_DEG, HEADING_DWELL_SEC,
                       _heading_state, "heading",
                       lambda d: f"{d:+4d}°")
    return math.radians(deg)


# ----------------- STANCE HEIGHT CYCLE (cmd[5]) -----------------
CYCLE_STANCE_HEIGHTS  = False
STANCE_HEIGHT_CYCLE_M = [-0.025, -0.015, -0.005, +0.005, +0.015, +0.025,
                        +0.015, +0.005, -0.005, -0.015]
STANCE_HEIGHT_DWELL_SEC = 1.5
STANCE_HEIGHT_OFFSET    = 0.0   # static fallback (m)
_height_state = {"last_idx": -1}

def current_stance_height(t):
    if not CYCLE_STANCE_HEIGHTS:
        return STANCE_HEIGHT_OFFSET
    return _cycle_value(t, STANCE_HEIGHT_CYCLE_M, STANCE_HEIGHT_DWELL_SEC,
                        _height_state, "stance height",
                        lambda v: f"{v*1000:+5.1f} mm")


# ----------------- STANCE WIDTH CYCLE (cmd[6]) -----------------
CYCLE_STANCE_WIDTHS    = False
STANCE_WIDTH_CYCLE_M   = [-0.020, -0.010, 0.0, +0.010, +0.020,
                         +0.010, 0.0, -0.010]
STANCE_WIDTH_DWELL_SEC = 1.5
STANCE_WIDTH_OFFSET    = 0.0   # static fallback (m)
_width_state = {"last_idx": -1}

def current_stance_width(t):
    if not CYCLE_STANCE_WIDTHS:
        return STANCE_WIDTH_OFFSET
    return _cycle_value(t, STANCE_WIDTH_CYCLE_M, STANCE_WIDTH_DWELL_SEC,
                        _width_state, "stance width ",
                        lambda v: f"{v*1000:+5.1f} mm")


# ----------------- SPIN CYCLE (cmd[2] = wz) -----------------
# Each entry is (sign, duration_seconds). +1 = CCW in world, -1 = CW.
CYCLE_SPIN = False
SPIN_CYCLE = [(+1, 22.0), (-1, 25.0), (+1, 20.0), (-1, 23.5)]
SPIN_DIRECTION = +1   # static fallback
_spin_state = {"last_idx": -1}

def current_spin_sign(t):
    if not CYCLE_SPIN:
        return SPIN_DIRECTION
    cycle_total = sum(d for _, d in SPIN_CYCLE)
    t_in_cycle  = t % cycle_total
    cumulative  = 0.0
    for idx, (sign, dur) in enumerate(SPIN_CYCLE):
        if t_in_cycle < cumulative + dur:
            if idx != _spin_state["last_idx"]:
                world_dir = "CCW" if sign > 0 else "CW "
                print(f"  t={t:5.1f}s  spin → world {world_dir} for {dur:.1f}s")
                _spin_state["last_idx"] = idx
            return sign
        cumulative += dur
    return 1.0


# ----------------- PITCH/ROLL CYCLE (cmd[3], cmd[4]) -----------------
CYCLE_PITCH_ROLL = False
# Each entry is (pitch_deg, roll_deg, dwell_s)
PITCH_ROLL_CYCLE = [
    (  0.0,   0.0, 1.0),
    (+10.0,   0.0, 1.5),
    (  0.0,   0.0, 1.0),
    (-10.0,   0.0, 1.5),
    (  0.0,   0.0, 1.0),
    (  0.0, +10.0, 1.5),
    (  0.0,   0.0, 1.0),
    (  0.0, -10.0, 1.5),
]
PITCH_OFFSET = 0.0   # static fallback (rad)
ROLL_OFFSET  = 0.0
_pitchroll_state = {"last_idx": -1}

def current_pitch_roll(t):
    if not CYCLE_PITCH_ROLL:
        return PITCH_OFFSET, ROLL_OFFSET
    cycle_total = sum(e[2] for e in PITCH_ROLL_CYCLE)
    t_in_cycle  = t % cycle_total
    cumulative  = 0.0
    for idx, (pdeg, rdeg, dur) in enumerate(PITCH_ROLL_CYCLE):
        if t_in_cycle < cumulative + dur:
            if idx != _pitchroll_state["last_idx"]:
                print(f"  t={t:5.1f}s  body tilt → pitch={pdeg:+5.1f}°  roll={rdeg:+5.1f}°")
                _pitchroll_state["last_idx"] = idx
            return math.radians(pdeg), math.radians(rdeg)
        cumulative += dur
    return 0.0, 0.0


# ----------------- SHIFT CYCLE (cmd[7], cmd[8]) -----------------
CYCLE_SHIFT = False
# Each entry is (shift_x_mm, shift_y_mm, dwell_s).
SHIFT_CYCLE = [
    (  0,   0, 1.0),
    (+30,   0, 1.5),
    (  0, +30, 1.5),
    (-30,   0, 1.5),
    (  0, -30, 1.5),
]
SHIFT_X = 0.0   # static fallback (m)
SHIFT_Y = 0.0
_shift_state = {"last_idx": -1}

def current_shift(t):
    if not CYCLE_SHIFT:
        return SHIFT_X, SHIFT_Y
    cycle_total = sum(e[2] for e in SHIFT_CYCLE)
    t_in_cycle  = t % cycle_total
    cumulative  = 0.0
    for idx, (sx_mm, sy_mm, dur) in enumerate(SHIFT_CYCLE):
        if t_in_cycle < cumulative + dur:
            if idx != _shift_state["last_idx"]:
                print(f"  t={t:5.1f}s  body shift → ({sx_mm:+3d}, {sy_mm:+3d}) mm")
                _shift_state["last_idx"] = idx
            return sx_mm * 1e-3, sy_mm * 1e-3
        cumulative += dur
    return 0.0, 0.0


# ============================================================================
# DEMO CONFIG
# ============================================================================
# MODE describes which cmd-vector behavior is active. Every mode honors all
# the cycle toggles above (e.g., enable CYCLE_PITCH_ROLL alongside MODE="walk"
# to walk while bobbing the body).
#   "stand" — no movement, only stance/pose overlays
#   "walk"  — translation in current_heading() direction at MAX_SPEED
#   "spin"  — yaw rotation at sign × MAX_YAW_RATE
MODE = "stand"

GAIT_PERIOD = 1.5      # seconds per full gait cycle (passed to Controller)
PATH_RADIUS = 0.025    # half-stride / lift height (passed to Controller)


# ============================================================================
# MAIN LOOP
# ============================================================================
def build_cmd(t, controller):
    """Assemble the 9-dim cmd vector for sim time t from active cycles + MODE."""
    cmd = np.zeros(9)
    pitch, roll = current_pitch_roll(t)
    cmd[3] = pitch
    cmd[4] = roll
    cmd[5] = current_stance_height(t)
    cmd[6] = current_stance_width(t)
    sx, sy = current_shift(t)
    cmd[7] = sx
    cmd[8] = sy

    if MODE == "walk":
        h = current_heading(t)
        cmd[0] = controller.MAX_SPEED * math.cos(h)
        cmd[1] = controller.MAX_SPEED * math.sin(h)
    elif MODE == "spin":
        cmd[2] = current_spin_sign(t) * controller.MAX_YAW_RATE
    # MODE=="stand" leaves cmd[0:3] at zero
    return cmd


def main():
    ctrl = Controller(MODEL_PATH, gait_period=GAIT_PERIOD, path_radius=PATH_RADIUS)

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

    print(f"\nsimple_gait | MODE={MODE!r}  PERIOD={GAIT_PERIOD}s  PATH_RADIUS={PATH_RADIUS}")
    print(f"  MAX_SPEED    = {ctrl.MAX_SPEED:.4f} m/s")
    print(f"  MAX_YAW_RATE = {ctrl.MAX_YAW_RATE:.4f} rad/s")
    cycles_on = [n for n, on in [
        ("heading", CYCLE_HEADINGS),
        ("height",  CYCLE_STANCE_HEIGHTS),
        ("width",   CYCLE_STANCE_WIDTHS),
        ("spin",    CYCLE_SPIN),
        ("pitch_roll", CYCLE_PITCH_ROLL),
        ("shift",   CYCLE_SHIFT),
    ] if on]
    print(f"  cycles enabled: {', '.join(cycles_on) if cycles_on else 'none'}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_t0  = data.time
        wall_t0 = time.time()
        while viewer.is_running():
            t = data.time - sim_t0
            cmd = build_cmd(t, ctrl)
            data.ctrl[:] = ctrl.predict(cmd, t)
            mujoco.mj_step(model, data)
            viewer.sync()
            lag = (wall_t0 + (data.time - sim_t0)) - time.time()
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
