"""
Watch a TO-derived trajectory in mujoco.viewer.

Loads the .npz produced by `tools/trajectory_opt_demo.py` and plays it
back through the same MJCF (`models/phantomx_simple_mjx.xml`) used
elsewhere in the project. Joint angles are sent directly to the
position actuators; the base body is left free (will fall under
gravity if the gait can't actually support it - that's part of what
makes this an interesting comparison vs the scaffold).

Usage (from project root):
    PYTHONPATH=. .venv\\Scripts\\python.exe tools/watch_to_trajectory.py \\
        --traj .cache/to_trajectory.npz

  (Run on Windows - mujoco.viewer needs the host's display.)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=".cache/to_trajectory.npz",
                    help="path to .npz produced by trajectory_opt_demo.py")
    ap.add_argument("--model", default="models/phantomx_simple_mjx.xml")
    ap.add_argument("--loops", type=int, default=10,
                    help="how many times to loop the playback (default 10)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed multiplier (1.0 = real-time)")
    ap.add_argument("--kinematic", action="store_true",
                    help="If set, hold the base fixed at the recorded base "
                         "pose (no physics). Otherwise let MuJoCo simulate "
                         "the base body under joint torques + gravity.")
    ap.add_argument("--play-hz", type=float, default=0.0,
                    help="Render playback at this rate via cubic-spline "
                         "interpolation between TO knots. 0 = play at "
                         "native knot rate (no interpolation).")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    traj_path = project_root / args.traj
    model_path = project_root / args.model

    if not traj_path.exists():
        raise SystemExit(f"Trajectory not found: {traj_path}\n"
                         "Run `python tools/trajectory_opt_demo.py` first.")
    data = np.load(traj_path)
    joints = data["joints"]            # (N, 18)
    base = data["base"]                # (N, 3)
    pitch = data["pitch"]              # (N,)
    roll = data["roll"]                # (N,)
    dt = float(data["dt"])
    N = joints.shape[0]
    duration = (N - 1) * dt
    print(f"[watch] Loaded trajectory: N={N} knots, dt={dt:.4f}s, "
          f"duration={duration:.2f}s")

    # Optional cubic-spline upsample. The TO solver's internal collocation
    # IS smooth between knots, but only knot values land in the npz —
    # interpolating here recovers that smoothness at the viewer.
    if args.play_hz > 0:
        from scipy.interpolate import CubicSpline
        t_knot = np.arange(N) * dt
        n_play = int(round(duration * args.play_hz)) + 1
        t_play = np.linspace(0.0, duration, n_play)
        joints = CubicSpline(t_knot, joints, axis=0)(t_play)
        base   = CubicSpline(t_knot, base,   axis=0)(t_play)
        pitch  = CubicSpline(t_knot, pitch)(t_play)
        roll   = CubicSpline(t_knot, roll)(t_play)
        dt = duration / (n_play - 1)
        N = n_play
        print(f"[watch] Cubic-spline upsampled to N={N}, dt={dt:.4f}s "
              f"({args.play_hz:.0f} Hz)")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    sim_data = mujoco.MjData(model)

    # Initial pose: base at recorded start, joints at recorded start.
    sim_data.qpos[0:3] = base[0]
    sim_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]   # identity quat
    sim_data.qpos[7:25] = joints[0]
    sim_data.ctrl[:18] = joints[0]
    mujoco.mj_forward(model, sim_data)

    period = dt / max(args.speed, 1e-6)
    with mujoco.viewer.launch_passive(model, sim_data) as viewer:
        for loop in range(args.loops):
            print(f"[watch] Loop {loop + 1}/{args.loops}")
            t_start = time.time()
            for k in range(N):
                if not viewer.is_running():
                    return
                # Push joint targets to actuators.
                sim_data.ctrl[:18] = joints[k]
                if args.kinematic:
                    # Override base pose with the recorded one; zero base velocity.
                    sim_data.qpos[0:3] = base[k]
                    # Build quaternion from pitch/roll (yaw=0). Order:
                    # R = Rz(0) * Ry(-pitch) * Rx(roll). For mujoco quat layout
                    # (w, x, y, z).
                    cp = np.cos(-pitch[k] / 2)
                    sp = np.sin(-pitch[k] / 2)
                    cr = np.cos(roll[k] / 2)
                    sr = np.sin(roll[k] / 2)
                    # qy * qx where qy = (cp,0,sp,0), qx = (cr,sr,0,0)
                    sim_data.qpos[3] = cp * cr
                    sim_data.qpos[4] = cp * sr
                    sim_data.qpos[5] = sp * cr
                    sim_data.qpos[6] = -sp * sr
                    # Write the recorded joint angles. mj_forward only runs FK;
                    # it doesn't execute actuators, so ctrl[] alone keeps the
                    # joints frozen at frame 0 in kinematic mode.
                    sim_data.qpos[7:25] = joints[k]
                    sim_data.qvel[:6] = 0.0
                    mujoco.mj_forward(model, sim_data)
                else:
                    # Full physics.
                    target = t_start + k * period
                    while time.time() < target:
                        mujoco.mj_step(model, sim_data)
                viewer.sync()
                if not args.kinematic:
                    continue  # mj_step already paced via while loop above
                # Kinematic mode: pace by sleep.
                target = t_start + k * period
                slack = target - time.time()
                if slack > 0:
                    time.sleep(slack)
            # brief pause between loops
            time.sleep(0.5)


if __name__ == "__main__":
    main()
