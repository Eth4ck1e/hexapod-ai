"""
to_rotate.py — generate omnidirectional TO trajectories from a forward template
via per-leg rotation + closed-form IK. NO solver invoked.

For each direction angle θ:
  - For each leg, the foot's body-frame motion (its OFFSET from the leg's
    radial-neutral position) is rotated by θ around the leg's vertical axis.
  - foot_rest_body[leg] stays put. Only the local oscillation rotates.
  - Joint angles are recomputed via Controller.body_to_joints().
  - Body trajectory rotates: base_new[k] = R_z(θ) @ base_forward[k]

This produces a kinematically consistent walking trajectory for any
commanded direction in seconds, without an NLP solve.

Caveats:
  - Pure translation only (yaw stays 0). Turning needs a separate generator.
  - Joint limits are NOT enforced — the script reports any out-of-bound joints
    and the caller decides what to do.
  - The forward template carries its boundary transients (spin-up/spin-down);
    rotated copies inherit them rotated.

Run:
    .venv\\Scripts\\python.exe tools/to_rotate.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mujoco

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gait.controller import Controller, LEG_NAMES


def _rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def _fk_foot_body(model, data, joints_18: np.ndarray,
                  foot_gids: list[int]) -> np.ndarray:
    """Forward kinematics: 18 joint angles → (6, 3) foot positions in body frame.
    Uses identity body pose (yaw=pitch=roll=0, base at origin) so the world
    foot positions ARE body-frame foot positions."""
    data.qpos[:3] = 0.0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:25] = joints_18
    mujoco.mj_forward(model, data)
    return np.stack([data.geom_xpos[g].copy() for g in foot_gids])


def rotate_trajectory(forward_traj: dict, ctrl: Controller,
                       model, data, foot_gids: list[int],
                       theta_rad: float) -> dict:
    """Take a forward trajectory and produce a rotated copy via per-leg
    rotation + IK. Returns a dict with the same schema as the input
    (joints, base, pitch, roll, dt)."""
    joints_fwd = np.asarray(forward_traj["joints"], dtype=np.float64)   # (N, 18)
    base_fwd   = np.asarray(forward_traj["base"],   dtype=np.float64)   # (N, 3)
    pitch_fwd  = np.asarray(forward_traj["pitch"],  dtype=np.float64)
    roll_fwd   = np.asarray(forward_traj["roll"],   dtype=np.float64)
    dt = float(forward_traj["dt"])
    N = joints_fwd.shape[0]

    # Recover body-frame foot positions per knot via FK at identity body pose.
    foot_body_fwd = np.zeros((N, 6, 3))
    for k in range(N):
        foot_body_fwd[k] = _fk_foot_body(model, data, joints_fwd[k], foot_gids)

    foot_rest = ctrl.LEG_ORIGIN_BODY.copy()                              # (6, 3)
    R = _rotz(theta_rad)

    # Per-leg rotation of the offset from neutral.
    joints_rot = np.zeros_like(joints_fwd)
    foot_body_rot = np.zeros_like(foot_body_fwd)
    for k in range(N):
        for leg in range(6):
            offset = foot_body_fwd[k, leg] - foot_rest[leg]
            offset_rot = R @ offset
            foot_body_rot[k, leg] = foot_rest[leg] + offset_rot
        joints_rot[k] = ctrl.body_to_joints(foot_body_rot[k])

    # Body trajectory rotates around world vertical (xy plane).
    base_rot = (R @ base_fwd.T).T
    # pitch/roll stay near zero in both forward and rotated frames; copy through.
    return {
        "joints": joints_rot,
        "base":   base_rot,
        "pitch":  pitch_fwd.copy(),
        "roll":   roll_fwd.copy(),
        "dt":     np.array(dt),
    }


def _check_joint_limits(joints: np.ndarray, model) -> tuple[float, list]:
    """Report joints that exceed their MJCF range. Returns (max_violation_rad,
    list of (knot, joint_idx, value, range)). max_violation_rad is 0 if all OK."""
    # Joint range in MJCF: indices 1..18 are the actuators (0 is freejoint).
    # jnt_range shape: (njnt, 2). For free joint at index 0, range is unused.
    ranges = model.jnt_range[1:]  # (18, 2)
    violations = []
    max_viol = 0.0
    for k in range(joints.shape[0]):
        for j in range(18):
            lo, hi = ranges[j]
            v = joints[k, j]
            if v < lo:
                viol = lo - v
                if viol > max_viol:
                    max_viol = viol
                violations.append((k, j, v, (lo, hi)))
            elif v > hi:
                viol = v - hi
                if viol > max_viol:
                    max_viol = viol
                violations.append((k, j, v, (lo, hi)))
    return max_viol, violations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", default=".cache/to_trajectory.npz",
                   help="Forward TO template (vx>0, vy=0)")
    p.add_argument("--model", default="models/phantomx_simple_mjx.xml")
    p.add_argument("--out-dir", default="tools/cache/to_rot",
                   help="Where to save per-direction trajectory npzs")
    p.add_argument("--n-directions", type=int, default=12,
                   help="Sample evenly around 360°")
    p.add_argument("--include-zero", action="store_true",
                   help="Also save the 0° (forward) result as a sanity check")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    template_path = project_root / args.template
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rotate] Loading template: {template_path}")
    template = dict(np.load(template_path))

    print(f"[rotate] Building Controller from {args.model}")
    ctrl = Controller(str(project_root / args.model))
    # Controller doesn't retain its MjModel; load our own for FK + joint-range checks.
    model = mujoco.MjModel.from_xml_path(str(project_root / args.model))
    data = mujoco.MjData(model)
    foot_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                   f"foot_{n}") for n in LEG_NAMES]

    angles = np.arange(args.n_directions) * (360.0 / args.n_directions)
    if not args.include_zero and 0.0 in angles:
        # 0° is the template itself; skip.
        angles = angles[angles != 0.0]

    summary = []
    for deg in angles:
        theta = np.deg2rad(float(deg))
        traj = rotate_trajectory(template, ctrl, model, data, foot_gids, theta)
        max_viol, viols = _check_joint_limits(traj["joints"], model)

        out_path = out_dir / f"dir_{int(round(deg)):03d}.npz"
        np.savez_compressed(out_path,
                             joints=traj["joints"],
                             base=traj["base"],
                             pitch=traj["pitch"],
                             roll=traj["roll"],
                             dt=traj["dt"])
        n_viols = len(viols)
        max_viol_deg = float(np.rad2deg(max_viol))
        n_total = traj["joints"].size
        flag = " " if n_viols == 0 else "!"
        rel = out_path.relative_to(project_root)
        print(f"  {flag} dir={deg:5.1f}°  OK={n_total - n_viols:>5d}/"
              f"{n_total:>5d} knot-joints  max_violation={max_viol_deg:+.2f}°  "
              f"-> {rel}")
        summary.append({"deg": deg, "max_viol_deg": max_viol_deg,
                         "n_viols": n_viols, "out": str(out_path)})

    # Summary
    print()
    print(f"[rotate] {len(angles)} directions generated.")
    n_clean = sum(1 for s in summary if s["n_viols"] == 0)
    print(f"  {n_clean}/{len(summary)} directions are joint-limit clean.")
    if n_clean < len(summary):
        worst = max(summary, key=lambda s: s["max_viol_deg"])
        print(f"  Worst violation: dir={worst['deg']:.1f}°, "
              f"{worst['max_viol_deg']:.2f}° past RoM.")


if __name__ == "__main__":
    main()
