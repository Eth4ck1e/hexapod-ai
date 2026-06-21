"""
tools/test_scaffold_drift.py — measure heading drift of the SCAFFOLD
under physics when commanded to walk straight.

If the scaffold itself drifts, the AMP priors generated from it will
encode that drift as "natural motion," and the discriminator will
never push the policy against it. We need to confirm the priors are
clean before training more aggressively against drift.

Test: command (vx=0.30, vy=0, wz=0) for 30 sec of physics, measure
body yaw drift over time. Repeat for backward + a couple of speed levels.

Run (Windows venv, headless):
    .venv\\Scripts\\python.exe tools/test_scaffold_drift.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gait.controller import Controller


def quat_to_yaw(qw, qx, qy, qz):
    """Extract yaw (rotation about world +Z) from a unit quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def apply_trim(cmd: np.ndarray, wz_trim_vx: float = 0.0,
                wz_trim_vy_abs: float = 0.0) -> np.ndarray:
    """Add a yaw trim to the cmd before passing to scaffold:
        cmd[2] += -wz_trim_vx * cmd[0]    (linear with vx, sign-flips for backward)
        cmd[2] += -wz_trim_vy_abs * |cmd[1]|  (constant during strafe; doesn't flip)
    Returns a NEW array (doesn't modify caller's).
    """
    trimmed = cmd.copy()
    trimmed[2] += -wz_trim_vx * cmd[0] - wz_trim_vy_abs * abs(cmd[1])
    return trimmed


def run_drift_test(label: str, cmd: np.ndarray, model_path: str,
                   duration: float = 30.0, ctrl_rate: float = 50.0,
                   wz_trim_vx: float = 0.0, wz_trim_vy_abs: float = 0.0):
    """Drive the scaffold at `cmd` (with optional trim) under physics for
    `duration` sec. Return (final_pos, final_yaw_deg, yaw_per_meter_deg_per_m).
    """
    ctrl = Controller(model_path)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    body_height = float(-ctrl.LEG_ORIGIN_BODY[:, 2].mean())
    cmd_trimmed = apply_trim(cmd, wz_trim_vx, wz_trim_vy_abs)
    joints0, _ = ctrl.predict_with_feet(cmd_trimmed, t=0.0)
    data.qpos[0:3] = (0.0, 0.0, body_height + 0.005)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7:25] = joints0
    data.ctrl[:18] = joints0
    mujoco.mj_forward(model, data)

    sim_dt = float(model.opt.timestep)
    ctrl_dt = 1.0 / ctrl_rate
    sim_per_ctrl = max(1, int(round(ctrl_dt / sim_dt)))
    n_ctrl_steps = int(duration * ctrl_rate)

    yaws_at_t = []  # list of (t_sec, x, y, yaw_deg)
    for k in range(n_ctrl_steps):
        t = k * ctrl_dt
        cmd_trimmed = apply_trim(cmd, wz_trim_vx, wz_trim_vy_abs)
        joints, _ = ctrl.predict_with_feet(cmd_trimmed, t)
        data.ctrl[:18] = joints
        for _ in range(sim_per_ctrl):
            mujoco.mj_step(model, data)
        if k % 50 == 0:  # log every second
            yaw = np.degrees(quat_to_yaw(*data.qpos[3:7]))
            yaws_at_t.append((t, float(data.qpos[0]), float(data.qpos[1]), yaw))

    final_x, final_y = float(data.qpos[0]), float(data.qpos[1])
    final_yaw_deg = np.degrees(quat_to_yaw(*data.qpos[3:7]))
    distance = float(np.hypot(final_x, final_y))
    yaw_per_m = final_yaw_deg / max(distance, 1e-6) if distance > 0.01 else 0.0

    print(f"\n=== {label} ===")
    print(f"  cmd: vx={cmd[0]:+.3f}  vy={cmd[1]:+.3f}  wz={cmd[2]:+.3f}")
    print(f"  duration: {duration}s,  steps: {n_ctrl_steps}")
    print(f"  final position: ({final_x:+.3f}, {final_y:+.3f}) m,  distance: {distance:.3f} m")
    print(f"  final yaw:      {final_yaw_deg:+.2f} deg")
    print(f"  drift rate:     {yaw_per_m:+.2f} deg/m  ({final_yaw_deg/duration:+.2f} deg/sec)")
    if abs(final_yaw_deg) < 2.0:
        print("  -> CLEAN: < 2° total drift")
    elif abs(final_yaw_deg) < 10.0:
        print("  -> SMALL drift")
    else:
        print("  -> SIGNIFICANT drift -- scaffold contributes to policy drift")
    return yaws_at_t


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wz-trim-vx", type=float, default=0.0,
                    help="yaw trim coefficient applied as -k * vx. "
                         "Counters CCW-on-forward / CW-on-backward bias. "
                         "Try 0.005 to 0.020 to dial in.")
    ap.add_argument("--wz-trim-vy-abs", type=float, default=0.0,
                    help="yaw trim coefficient applied as -k * |vy|. "
                         "Counters consistent-direction strafe drift. "
                         "Try 0.005 to 0.020.")
    ap.add_argument("--duration", type=float, default=20.0)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_path = str(project_root / "models" / "phantomx_simple_mjx.xml")

    ctrl = Controller(model_path)
    print(f"Scaffold defaults: gait_period={ctrl.gait_period}, "
          f"path_radius={ctrl.path_radius}")
    print(f"MAX_SPEED = {ctrl.MAX_SPEED:.4f} m/s")
    print(f"MAX_YAW_RATE = {ctrl.MAX_YAW_RATE:.4f} rad/s")

    # Test scenarios.
    fwd_speed = 0.85 * ctrl.MAX_SPEED        # match trained max
    slow_speed = 0.40 * ctrl.MAX_SPEED       # min trained
    medium_speed = 0.62 * ctrl.MAX_SPEED     # mid trained

    cmds = [
        ("Forward fast (85% MAX_SPEED)",  np.array([+fwd_speed, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("Forward medium (62% MAX_SPEED)", np.array([+medium_speed, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("Forward slow (40% MAX_SPEED)",   np.array([+slow_speed, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("Backward fast",                  np.array([-fwd_speed, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("Strafe LEFT",                    np.array([0, +fwd_speed, 0, 0, 0, 0, 0, 0, 0])),
        ("Strafe RIGHT",                   np.array([0, -fwd_speed, 0, 0, 0, 0, 0, 0, 0])),
    ]

    print(f"\nTrim coefficients: wz_trim_vx={args.wz_trim_vx}, "
          f"wz_trim_vy_abs={args.wz_trim_vy_abs}")

    for label, cmd in cmds:
        run_drift_test(label, cmd, model_path, duration=args.duration,
                       wz_trim_vx=args.wz_trim_vx,
                       wz_trim_vy_abs=args.wz_trim_vy_abs)


if __name__ == "__main__":
    main()
