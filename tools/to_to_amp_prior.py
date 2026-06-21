"""
tools/to_to_amp_prior.py — Convert 8 directional TO trajectories into an
AMP prior dataset matching the schema produced by amp/prior_data.py.

Reads all 8 directional npzs (0, 45, 90, ... 315 degrees), computes the
49-dim AMP state vector at each knot via numerical differentiation, builds
(s_t, s_{t+1}) transition pairs, and saves them concatenated.

State vector layout (must match amp/prior_data.py STATE_DIM=49):
  joint_pos              18   joints[k]
  joint_vel              18   numerical diff of joints, dt-spaced
  body_linvel_body        3   world body velocity rotated into body frame
                              via R(yaw)^T (yaw-only rotation; TO trajectories
                              have yaw=0, so body frame == world frame for linvel)
  body_angvel             3   numerical diff of (roll, pitch, yaw=0), dt-spaced
  body_height             1   base[k, 2]
  foot_heights            6   foot world Z minus body Z, computed via FK

First and last knots are dropped (one-sided derivatives invalid).
Each N-knot trajectory contributes (N-2) states and (N-3) transition pairs.
For the default N=220: 218 valid states, 217 transition pairs per trajectory.

Run on Windows venv (CPU, no JAX/GPU needed):
  PYTHONPATH=. .venv\\Scripts\\python.exe tools/to_to_amp_prior.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import scipy.spatial.transform as transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gait.controller import LEG_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = str(PROJECT_ROOT / "models" / "phantomx_simple_mjx.xml")
OUT_PATH     = str(PROJECT_ROOT / "checkpoints" / "amp_priors_to_omni.npz")
STATE_DIM    = 49   # must match amp/prior_data.py / amp/discriminator.py

_FOOT_LOCAL  = np.array([0.134, 0.031, 0.0])  # tibia-local foot-tip (sphere center)

# All 8 directions in order with their source npz paths (relative to project root).
TRAJECTORY_FILES = [
    (".cache/to_trajectory.npz",        "0 deg forward"),
    (".cache/to_trajectory_45deg.npz",  "45 deg forward-left"),
    (".cache/to_trajectory_90deg.npz",  "90 deg pure-left"),
    (".cache/to_trajectory_135deg.npz", "135 deg (FB mirror of 45)"),
    (".cache/to_trajectory_180deg.npz", "180 deg (FB mirror of 0)"),
    (".cache/to_trajectory_225deg.npz", "225 deg (both mirror of 45)"),
    (".cache/to_trajectory_270deg.npz", "270 deg (LR mirror of 90)"),
    (".cache/to_trajectory_315deg.npz", "315 deg (LR mirror of 45)"),
]


def _setup_fk(model_path: str):
    model     = mujoco.MjModel.from_xml_path(model_path)
    mj_data   = mujoco.MjData(model)
    tibia_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tibia_{n}")
                 for n in LEG_NAMES]
    return model, mj_data, tibia_ids


def _foot_world(model, mj_data, tibia_ids,
                base: np.ndarray, pitch: float, roll: float,
                joints: np.ndarray) -> np.ndarray:
    """Return (6, 3) foot world positions via FK at a single knot."""
    quat_xyzw = transform.Rotation.from_euler("yx", [pitch, roll]).as_quat()
    mj_data.qpos[:3]   = base
    mj_data.qpos[3:7]  = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    mj_data.qpos[7:25] = joints
    mujoco.mj_forward(model, mj_data)
    fw = np.empty((6, 3))
    for i, bid in enumerate(tibia_ids):
        fw[i] = mj_data.xpos[bid] + mj_data.xmat[bid].reshape(3, 3) @ _FOOT_LOCAL
    return fw


def _trajectory_to_states(npz_path: str, label: str,
                           model, mj_data, tibia_ids) -> np.ndarray:
    """Build (N, 49) AMP state array for all N knots in a trajectory.

    Only interior knots [1..N-2] will be used for transitions (caller drops
    boundaries), but we keep the full-length arrays for clean indexing.
    """
    d        = np.load(npz_path)
    joints   = d["joints"]   # (N, 18)
    base     = d["base"]     # (N, 3)
    pitch    = d["pitch"]    # (N,)
    roll_arr = d["roll"]     # (N,)
    dt       = float(d["dt"])
    N        = joints.shape[0]
    print(f"  {label}: N={N}, dt={dt:.5f}s")

    # Central differences for all kinematic quantities.
    def _cen_diff(arr, axis=0):
        out = np.empty_like(arr)
        slices_prev = [slice(None)] * arr.ndim
        slices_next = [slice(None)] * arr.ndim
        slices_prev[axis] = slice(None, -2)
        slices_next[axis] = slice(2, None)
        slices_mid = [slice(None)] * arr.ndim
        slices_mid[axis] = slice(1, -1)
        out[tuple(slices_mid)] = (arr[tuple(slices_next)] - arr[tuple(slices_prev)]) / (2 * dt)
        # Boundary (one-sided) — these rows are dropped by the caller anyway.
        if axis == 0:
            out[0]  = (arr[1]  - arr[0])   / dt
            out[-1] = (arr[-1] - arr[-2])  / dt
        return out

    joint_vel      = _cen_diff(joints, axis=0)        # (N, 18)
    base_vel_world = _cen_diff(base,   axis=0)        # (N, 3)
    roll_vel       = _cen_diff(roll_arr)               # (N,)
    pitch_vel      = _cen_diff(pitch)                  # (N,)

    # TO trajectories have yaw=0 throughout; yaw-only rotation is identity,
    # so body_linvel_body == world linear velocity for these trajectories.
    # (The JAX env's _body_frame_linvel uses yaw from quat; for yaw=0 it
    # reduces to the identity rotation as well — consistent.)
    body_linvel_body = base_vel_world   # (N, 3) — yaw=0, no rotation needed

    # body_angvel order matches qvel[3:6] = [roll_rate, pitch_rate, yaw_rate].
    body_angvel = np.stack([roll_vel, pitch_vel, np.zeros(N)], axis=1)  # (N, 3)

    states = np.empty((N, STATE_DIM), dtype=np.float32)
    for k in range(N):
        fw = _foot_world(model, mj_data, tibia_ids,
                         base[k], pitch[k], roll_arr[k], joints[k])
        foot_heights_k = fw[:, 2] - base[k, 2]   # (6,)

        states[k] = np.concatenate([
            joints[k].astype(np.float32),            # [0:18]  joint_pos
            joint_vel[k].astype(np.float32),         # [18:36] joint_vel
            body_linvel_body[k].astype(np.float32),  # [36:39] body_linvel
            body_angvel[k].astype(np.float32),       # [39:42] body_angvel
            np.array([base[k, 2]], dtype=np.float32),  # [42]  body_height
            foot_heights_k.astype(np.float32),       # [43:49] foot_heights
        ])

    return states


def build_prior(out_path: str, model_path: str = MODEL_PATH) -> None:
    model, mj_data, tibia_ids = _setup_fk(model_path)

    all_st  = []
    all_st1 = []
    all_cmds = []

    for npz_rel, label in TRAJECTORY_FILES:
        npz_path = str(PROJECT_ROOT / npz_rel)
        if not Path(npz_path).exists():
            print(f"  MISSING: {npz_path} -- skipping")
            continue

        d = np.load(npz_path)
        states = _trajectory_to_states(npz_path, label, model, mj_data, tibia_ids)
        N = states.shape[0]

        # Per-trajectory cmd is constant across all knots (TO solves are
        # for a fixed (vx, vy, wz=0)). Pull from npz keys saved by to_solver.
        vx = float(d["vx_cmd"]) if "vx_cmd" in d.files else 0.0
        vy = float(d["vy_cmd"]) if "vy_cmd" in d.files else 0.0
        wz = float(d["wz_cmd"]) if "wz_cmd" in d.files else 0.0
        # Full 9-D cmd vector matching prior_data.py format. Posture slots
        # (pitch, roll, dh, dw, sx, sy) are zero — the TO trajectories were
        # solved without those degrees of freedom.
        cmd = np.array([vx, vy, wz, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        # Drop first/last knot; build consecutive pairs from interior.
        # s_t  = states[1:-2], s_t1 = states[2:-1] → (N-3) pairs.
        # cmd_t broadcasts to one row per transition (constant per trajectory).
        n_trans = N - 3
        all_st.append(states[1:-2])
        all_st1.append(states[2:-1])
        all_cmds.append(np.broadcast_to(cmd, (n_trans, 9)).copy())
        print(f"    => {n_trans} transitions  cmd=({vx:+.3f}, {vy:+.3f}, {wz:+.3f})")

    states_t  = np.concatenate(all_st,   axis=0)
    states_t1 = np.concatenate(all_st1,  axis=0)
    cmds_t    = np.concatenate(all_cmds, axis=0)

    print(f"\nTotal transitions: {states_t.shape[0]}")
    print(f"states_t  shape : {states_t.shape}")
    print(f"states_t1 shape : {states_t1.shape}")
    print(f"cmds_t    shape : {cmds_t.shape}")

    nan_count = int(np.isnan(states_t).sum() + np.isnan(states_t1).sum())
    print(f"NaN count: {nan_count}")

    height_col = states_t[:, 42]
    print(f"body_height col: mean={height_col.mean():.4f}  "
          f"min={height_col.min():.4f}  max={height_col.max():.4f}")

    linvel_x = states_t[:, 36]
    print(f"body_linvel_x : mean={linvel_x.mean():+.4f}  std={linvel_x.std():.4f}")

    assert states_t.shape[0] > 1500, f"expected >1500 transitions, got {states_t.shape[0]}"
    assert nan_count == 0, "NaN detected in output"
    assert 0.10 < height_col.mean() < 0.20, f"body_height mean looks wrong: {height_col.mean()}"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, states_t=states_t, states_t1=states_t1,
                        cmds_t=cmds_t)
    sz_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"\nSaved to {out_path} ({sz_mb:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert 8 directional TO trajectories into AMP prior npz.")
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--out",   default=OUT_PATH)
    args = p.parse_args()

    print("=" * 60)
    print("TO -> AMP prior converter")
    print(f"model : {args.model}")
    print(f"out   : {args.out}")
    print("=" * 60)
    build_prior(args.out, model_path=args.model)


if __name__ == "__main__":
    main()
