"""
calibrate_scaffold_speed.py — sweep (gait_period, path_radius) and measure
what each combo actually delivers under physics.

For each (period, path_radius) combination:
  1. Build a Controller with those params
  2. Initialize the bot in mujoco physics, settle for 1 cycle
  3. Run scaffold cmd at the combo's own MAX_SPEED (so stride is at max)
  4. Step physics for N cycles
  5. Measure: actual body velocity, per-foot slip during stance,
     peak actuator torque drawn

Output: CSV with all combos + diagnostics. The user picks a discrete
ladder of "speed levels" from the rows where slip is low and torque
margin is healthy.

Run (Windows venv, headless — no viewer):
    .venv\\Scripts\\python.exe tools/calibrate_scaffold_speed.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import mujoco

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gait.controller import Controller, LEG_NAMES


def run_one(period: float, path_radius: float,
            model_path: str, n_cycles: int, settle_cycles: int) -> dict:
    """Run one (period, path_radius) combo under physics. Returns metrics."""
    ctrl = Controller(model_path,
                      gait_period=period,
                      path_radius=path_radius)
    max_speed = ctrl.MAX_SPEED          # 4 * path_radius / period
    cmd = np.array([max_speed, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    foot_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"foot_{n}") for n in LEG_NAMES]

    # Standing initial pose. Body at neutral height; joints at scaffold t=0.
    body_height = float(-ctrl.LEG_ORIGIN_BODY[:, 2].mean())
    joints0, _ = ctrl.predict_with_feet(cmd, t=0.0)
    data.qpos[0:3] = (0.0, 0.0, body_height + 0.005)  # tiny drop for contact settle
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7:25] = joints0
    data.ctrl[:18] = joints0
    mujoco.mj_forward(model, data)

    # Physics rate (mj_step uses model.opt.timestep; typically 0.005 = 200 Hz).
    sim_dt = float(model.opt.timestep)
    ctrl_rate = 50.0
    ctrl_dt = 1.0 / ctrl_rate
    sim_per_ctrl = max(1, int(round(ctrl_dt / sim_dt)))

    # Total duration: settle + measurement.
    total_cycles = settle_cycles + n_cycles
    total_dur = total_cycles * period
    n_ctrl_steps = int(total_dur * ctrl_rate)
    settle_steps = int(settle_cycles * period * ctrl_rate)

    # Logs
    base_xy = np.zeros((n_ctrl_steps, 2))
    foot_xyz = np.zeros((n_ctrl_steps, 6, 3))
    torque_max = 0.0

    for k in range(n_ctrl_steps):
        t = k * ctrl_dt
        joints, _ = ctrl.predict_with_feet(cmd, t)
        data.ctrl[:18] = joints
        for _ in range(sim_per_ctrl):
            mujoco.mj_step(model, data)
        base_xy[k] = data.qpos[:2]
        for li in range(6):
            foot_xyz[k, li] = data.geom_xpos[foot_gids[li]]
        # Peak actuator force across all joints (proxy for motor torque demand).
        torque_max = max(torque_max,
                         float(np.max(np.abs(data.actuator_force[:18]))))

    # Measurement: only use steps after settle.
    meas = slice(settle_steps, n_ctrl_steps)
    base_meas = base_xy[meas]
    duration_meas = (n_ctrl_steps - settle_steps) * ctrl_dt
    if duration_meas <= 0 or base_meas.shape[0] < 2:
        actual_vx = 0.0
    else:
        actual_vx = float((base_meas[-1, 0] - base_meas[0, 0]) / duration_meas)

    # Per-foot stance slip. The foot geom is a sphere with ~7 mm radius,
    # so a "planted" foot's geom-center sits at z ≈ 7 mm, not 0. Threshold
    # at 12 mm captures planted feet (lifted feet are >30 mm at swing peak).
    foot_meas = foot_xyz[meas]
    slips_mm = []
    for li in range(6):
        in_stance = foot_meas[:, li, 2] < 0.012
        # Find contiguous stance segments.
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
        for s, e in segs:
            xy = foot_meas[s:e, li, :2]
            slips_mm.append(1000.0 * float(np.linalg.norm(xy[-1] - xy[0])))

    mean_slip = float(np.mean(slips_mm)) if slips_mm else 0.0
    max_slip = float(np.max(slips_mm)) if slips_mm else 0.0

    return {
        "period": period,
        "path_radius": path_radius,
        "max_speed_cmd": max_speed,
        "actual_vx": actual_vx,
        "speed_efficiency": (actual_vx / max_speed) if max_speed > 1e-9 else 0.0,
        "mean_slip_mm": mean_slip,
        "max_slip_mm": max_slip,
        "peak_torque_Nm": torque_max,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/phantomx_simple_mjx.xml")
    ap.add_argument("--periods", type=float, nargs="+",
                    default=[2.0, 1.5, 1.2, 1.0, 0.85, 0.75, 0.6])
    ap.add_argument("--path-radii", type=float, nargs="+",
                    default=[0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.070])
    ap.add_argument("--cycles", type=int, default=8,
                    help="measurement cycles AFTER settle (default 8)")
    ap.add_argument("--settle", type=int, default=2,
                    help="warm-up cycles to skip (default 2)")
    ap.add_argument("--out", default="tools/scaffold_speed_calibration.csv")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_path = str(project_root / args.model)
    out_path = project_root / args.out

    results = []
    print(f"Sweeping {len(args.periods)} × {len(args.path_radii)} = "
          f"{len(args.periods) * len(args.path_radii)} combos")
    print(f"  cmd: max_speed for each combo (= 4 × path_radius / period)")
    print(f"  measurement: {args.cycles} cycles after {args.settle}-cycle settle")
    print()
    print(f"{'period':>7} {'path_R':>7} {'max_cmd':>9} {'actual':>9} {'eff':>5}"
          f" {'meanSlp':>8} {'maxSlp':>7} {'peakF':>7}")
    print(f"{'(s)':>7} {'(m)':>7} {'(m/s)':>9} {'(m/s)':>9} {'(%)':>5}"
          f" {'(mm)':>8} {'(mm)':>7} {'(Nm)':>7}")

    for period in args.periods:
        for path_radius in args.path_radii:
            r = run_one(period, path_radius, model_path,
                        args.cycles, args.settle)
            results.append(r)
            print(f"{r['period']:>7.2f} {r['path_radius']:>7.3f} "
                  f"{r['max_speed_cmd']:>9.4f} {r['actual_vx']:>+9.4f} "
                  f"{r['speed_efficiency']*100:>4.0f}% "
                  f"{r['mean_slip_mm']:>8.1f} {r['max_slip_mm']:>7.1f} "
                  f"{r['peak_torque_Nm']:>7.2f}")

    # CSV
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n[calibrate] saved {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
