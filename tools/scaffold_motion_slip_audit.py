"""
scaffold_motion_slip_audit.py — measure foot slip during stance segments
across the full motion repertoire of the scaffold gait.

Calibrate_scaffold_speed.py only sweeps forward motion at varied (period,
path_radius). This script holds those at calibrated defaults and instead
sweeps the cmd vector across all motion types we care about: cardinal
translations, diagonals, pure yaw, forward + arc turns. Output is a CSV
showing per-scenario slip stats — informational, used to spot motion
types where the scaffold is sloppy enough to need attention.

Run (Windows venv, headless):
    .venv\\Scripts\\python.exe tools/scaffold_motion_slip_audit.py
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gait.controller import LEG_NAMES, Controller


def run_scenario(label: str, cmd: np.ndarray,
                 model_path: str, n_cycles: int, settle_cycles: int) -> dict:
    """Run one motion scenario under physics. Returns slip + torque stats.
    cmd is a 9-vec; only (vx, vy, wz) used here (paper cmd-mask)."""
    ctrl = Controller(model_path)

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    foot_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"foot_{n}") for n in LEG_NAMES]

    body_height = float(-ctrl.LEG_ORIGIN_BODY[:, 2].mean())
    joints0, _ = ctrl.predict_with_feet(cmd, t=0.0)
    data.qpos[0:3] = (0.0, 0.0, body_height + 0.005)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7:25] = joints0
    data.ctrl[:18] = joints0
    mujoco.mj_forward(model, data)

    sim_dt = float(model.opt.timestep)
    ctrl_rate = 50.0
    ctrl_dt = 1.0 / ctrl_rate
    sim_per_ctrl = max(1, int(round(ctrl_dt / sim_dt)))

    total_cycles = settle_cycles + n_cycles
    total_dur = total_cycles * ctrl.gait_period
    n_ctrl_steps = int(total_dur * ctrl_rate)
    settle_steps = int(settle_cycles * ctrl.gait_period * ctrl_rate)

    foot_xyz = np.zeros((n_ctrl_steps, 6, 3))
    torque_max = 0.0

    for k in range(n_ctrl_steps):
        t = k * ctrl_dt
        joints, _ = ctrl.predict_with_feet(cmd, t)
        data.ctrl[:18] = joints
        for _ in range(sim_per_ctrl):
            mujoco.mj_step(model, data)
        for li in range(6):
            foot_xyz[k, li] = data.geom_xpos[foot_gids[li]]
        torque_max = max(torque_max,
                         float(np.max(np.abs(data.actuator_force[:18]))))

    meas = slice(settle_steps, n_ctrl_steps)
    foot_meas = foot_xyz[meas]

    # Per-foot stance slip (foot sphere center sits at z ≈ 7 mm when
    # planted; threshold at 12 mm catches stance but excludes lift-off).
    per_foot_mean = []
    per_foot_max = []
    for li in range(6):
        in_stance = foot_meas[:, li, 2] < 0.012
        segs = []
        start = None
        for k in range(foot_meas.shape[0]):
            if in_stance[k] and start is None:
                start = k
            elif not in_stance[k] and start is not None:
                segs.append((start, k))
                start = None
        if start is not None:
            segs.append((start, foot_meas.shape[0]))
        segs = [s for s in segs if s[1] - s[0] >= 2]
        slips = []
        for s, e in segs:
            xy = foot_meas[s:e, li, :2]
            slips.append(1000.0 * float(np.linalg.norm(xy[-1] - xy[0])))
        per_foot_mean.append(float(np.mean(slips)) if slips else 0.0)
        per_foot_max.append(float(np.max(slips)) if slips else 0.0)

    all_slips = []
    for li in range(6):
        if per_foot_mean[li] > 0:
            all_slips.append(per_foot_mean[li])

    out = {
        "label": label,
        "vx": float(cmd[0]), "vy": float(cmd[1]), "wz": float(cmd[2]),
        "mean_slip_mm":   float(np.mean(per_foot_mean)),
        "max_slip_mm":    float(np.max(per_foot_max)),
        "peak_torque_Nm": torque_max,
    }
    for i, n in enumerate(LEG_NAMES):
        out[f"{n}_mean_mm"] = per_foot_mean[i]
    return out


def build_scenarios(ctrl: Controller) -> list[tuple[str, np.ndarray]]:
    """All scenarios we test. cmd 9-vec; only [vx, vy, wz] non-zero."""
    SP   = 0.7 * ctrl.MAX_SPEED                         # speed for translations
    SP_W = 0.6 * ctrl.MAX_SPEED                         # speed during arcs
    YAW  = 0.7 * ctrl.MAX_YAW_RATE                      # in-place yaw rate

    def cmd(vx=0.0, vy=0.0, wz=0.0):
        c = np.zeros(9, dtype=np.float64)
        c[0], c[1], c[2] = vx, vy, wz
        return c

    def arc(speed, R, sign):
        """Forward at `speed`, turning at radius R (sign = +1 left, -1 right)."""
        return cmd(vx=speed, wz=sign * speed / R)

    R_min  = ctrl.MIN_TURN_RADIUS + 0.005   # tightest controllable
    R_med  = 0.5 * (ctrl.MIN_TURN_RADIUS + ctrl.MAX_TURN_RADIUS)
    R_max  = ctrl.MAX_TURN_RADIUS - 0.005   # gentlest controllable

    return [
        # Pure translations (4 cardinals)
        ("forward",          cmd(vx=SP)),
        ("backward",         cmd(vx=-SP)),
        ("strafe_left",      cmd(vy=SP)),
        ("strafe_right",     cmd(vy=-SP)),
        # Diagonals (4)
        ("forward_left_45",  cmd(vx=SP*math.cos( math.pi/4), vy=SP*math.sin( math.pi/4))),
        ("forward_right_45", cmd(vx=SP*math.cos(-math.pi/4), vy=SP*math.sin(-math.pi/4))),
        ("backward_left_45", cmd(vx=SP*math.cos( 3*math.pi/4), vy=SP*math.sin( 3*math.pi/4))),
        ("backward_right_45",cmd(vx=SP*math.cos(-3*math.pi/4), vy=SP*math.sin(-3*math.pi/4))),
        # Pure yaw (turn in place)
        ("spin_left",        cmd(wz= YAW)),
        ("spin_right",       cmd(wz=-YAW)),
        # Arcs (forward + turn) at 3 radii each direction
        ("arc_left_tight",   arc(SP_W, R_min, +1)),
        ("arc_left_medium",  arc(SP_W, R_med, +1)),
        ("arc_left_gentle",  arc(SP_W, R_max, +1)),
        ("arc_right_tight",  arc(SP_W, R_min, -1)),
        ("arc_right_medium", arc(SP_W, R_med, -1)),
        ("arc_right_gentle", arc(SP_W, R_max, -1)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/phantomx_simple_mjx.xml")
    ap.add_argument("--cycles", type=int, default=8,
                    help="measurement cycles after settle (default 8)")
    ap.add_argument("--settle", type=int, default=2,
                    help="warm-up cycles to skip (default 2)")
    ap.add_argument("--out", default="tools/scaffold_motion_slip_audit.csv")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_path = str(project_root / args.model)
    out_path = project_root / args.out

    ctrl_for_scenarios = Controller(model_path)
    scenarios = build_scenarios(ctrl_for_scenarios)
    print(f"Sweeping {len(scenarios)} motion scenarios under physics")
    print(f"  measurement: {args.cycles} cycles after {args.settle}-cycle settle")
    print(f"  MAX_SPEED={ctrl_for_scenarios.MAX_SPEED:.4f} m/s, "
          f"MAX_YAW_RATE={ctrl_for_scenarios.MAX_YAW_RATE:.4f} rad/s, "
          f"R in [{ctrl_for_scenarios.MIN_TURN_RADIUS:.2f}, "
          f"{ctrl_for_scenarios.MAX_TURN_RADIUS:.2f}] m")
    print()
    print(f"{'scenario':>20} {'vx':>7} {'vy':>7} {'wz':>7}"
          f" {'meanSlp':>8} {'maxSlp':>8} {'peakF':>7}")
    print(f"{'':>20} {'(m/s)':>7} {'(m/s)':>7} {'(rad/s)':>7}"
          f" {'(mm)':>8} {'(mm)':>8} {'(Nm)':>7}")

    results = []
    for label, cmd in scenarios:
        r = run_scenario(label, cmd, model_path, args.cycles, args.settle)
        results.append(r)
        print(f"{r['label']:>20} {r['vx']:>+7.3f} {r['vy']:>+7.3f} {r['wz']:>+7.3f}"
              f" {r['mean_slip_mm']:>8.2f} {r['max_slip_mm']:>8.2f}"
              f" {r['peak_torque_Nm']:>7.2f}")

    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\n[audit] saved {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
