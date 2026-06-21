"""
tools/to_mirror.py — Mirror TO (trajectory optimisation) trajectories.

Generates new directional trajectories by reflecting existing ones.
Two mirror operations supported:

  fb  (front-back): x-flipped body motion + leg re-assignment RR↔RF, LR↔LF.
      Produces a trajectory that moves in the opposite x-direction.
  lr  (left-right): y-flipped body motion + leg re-assignment R↔L per slot.
      Produces a trajectory that moves in the opposite y-direction.
  both: fb then lr simultaneously (combined x+y flip + full leg permutation).

Why leg re-assignment is necessary:
  Each leg's IK operates in its coxa-local frame. After an FB flip of foot
  world positions, the flipped position of (say) the RR foot ends up nearest
  to the RF coxa — not the RR coxa — because the coxas are fixed to the body.
  Assigning the mirrored foot to the correct coxa is required to keep joint
  angles within the reachable workspace.

  Leg permutations (canonical order: RR=0 RM=1 RF=2 LR=3 LM=4 LF=5):
    FB  : [0,1,2,3,4,5] → [2,1,0,5,4,3]  (RR↔RF, LR↔LF, mid stays)
    LR  : [0,1,2,3,4,5] → [3,4,5,0,1,2]  (R→L, L→R)
    Both: [0,1,2,3,4,5] → [5,4,3,2,1,0]  (full reversal)

Mirror procedure per knot:
  1. FK: set base pose + joints → mj_forward → read foot world positions (6, 3).
  2. Apply chosen coordinate flip to foot_world and base/pitch/roll scalars.
  3. Permute the flipped foot array so index i holds the foot assigned to leg i.
  4. Convert foot_world to body frame (subtract mirrored base, ignore yaw).
  5. IK via Controller.body_to_joints → new 18 joint angles.
  6. RoM sanity-check (warn if any joint exceeds empirical limits + 0.02 rad).

CLI:
  python tools/to_mirror.py --source .cache/to_trajectory_45deg.npz \\
                             --mirror fb \\
                             --out .cache/to_trajectory_135deg.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import scipy.spatial.transform as transform

from gait.controller import LEG_NAMES, Controller

MODEL_PATH    = "models/phantomx_simple_mjx.xml"
FOOT_LOCAL    = np.array([0.134, 0.031, 0.0])   # tibia-local foot-tip (sphere center)
ROM_TOLERANCE = 0.02  # rad, ~1° — violations beyond this trigger a warning

# Leg permutation indices for each mirror operation.
# Canonical order: RR=0 RM=1 RF=2 LR=3 LM=4 LF=5
_PERM_FB   = [2, 1, 0, 5, 4, 3]  # RR↔RF, LR↔LF, middles stay (inverts x-reachability)
_PERM_LR   = [3, 4, 5, 0, 1, 2]  # R↔L same slot
_PERM_BOTH = [5, 4, 3, 2, 1, 0]  # full reversal (fb + lr combined)


def _load_joint_limits(json_path: str = "joint_limits.json") -> np.ndarray:
    """Return (18, 2) empirical [lo, hi] limits in canonical joint order
    RR, RM, RF, LR, LM, LF × (coxa, femur, tibia)."""
    with open(json_path) as f:
        data = json.load(f)

    limits = []
    for leg in LEG_NAMES:
        for jtype in ("coxa", "femur", "tibia"):
            lo, hi = data[f"{jtype}_joint_{leg}"]["empirical"]
            limits.append((lo, hi))
    return np.array(limits, dtype=np.float64)   # (18, 2)


def _set_qpos(mj_data: mujoco.MjData, base: np.ndarray,
              pitch: float, roll: float, joints: np.ndarray) -> None:
    """Load a knot into MjData. The TO stores pitch/roll in aerospace convention
    (pitch+ = nose up); scipy 'yx' Euler = Ry(pitch)·Rx(roll) matches that."""
    quat_xyzw = transform.Rotation.from_euler("yx", [pitch, roll]).as_quat()
    mj_data.qpos[:3]   = base
    mj_data.qpos[3:7]  = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # wxyz
    mj_data.qpos[7:25] = joints


def _get_foot_world(model: mujoco.MjModel, mj_data: mujoco.MjData,
                    tibia_ids: list[int]) -> np.ndarray:
    """(6, 3) foot world positions after mj_forward."""
    foot_world = np.empty((6, 3))
    for i, bid in enumerate(tibia_ids):
        foot_world[i] = mj_data.xpos[bid] + mj_data.xmat[bid].reshape(3, 3) @ FOOT_LOCAL
    return foot_world


def _check_rom(joints: np.ndarray, limits: np.ndarray, knot_idx: int) -> int:
    """Return violation count; print one line per violating joint."""
    lo = limits[:, 0] - ROM_TOLERANCE
    hi = limits[:, 1] + ROM_TOLERANCE
    viol = np.where((joints < lo) | (joints > hi))[0]
    for j in viol:
        print(f"  WARN knot {knot_idx:3d} joint {j:2d} ({LEG_NAMES[j // 3]}): "
              f"{joints[j]:.4f} outside [{lo[j]:.4f}, {hi[j]:.4f}]")
    return len(viol)


def mirror_trajectory(source_path: str, mirror_op: str, out_path: str) -> None:
    print(f"source : {source_path}")
    print(f"mirror : {mirror_op}")
    print(f"out    : {out_path}")

    data_in   = np.load(source_path)
    joints_in = data_in["joints"]   # (N, 18)
    base_in   = data_in["base"]     # (N, 3)
    pitch_in  = data_in["pitch"]    # (N,)
    roll_in   = data_in["roll"]     # (N,)
    dt        = float(data_in["dt"])
    N         = joints_in.shape[0]

    model     = mujoco.MjModel.from_xml_path(MODEL_PATH)
    mj_data   = mujoco.MjData(model)
    tibia_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tibia_{n}")
                 for n in LEG_NAMES]

    ctrl   = Controller(MODEL_PATH)
    limits = _load_joint_limits()

    # Coordinate flips applied to base/pitch/roll scalars.
    flip_x = mirror_op in ("fb", "both")
    flip_y = mirror_op in ("lr", "both")

    base_out  = base_in.copy()
    pitch_out = pitch_in.copy()
    roll_out  = roll_in.copy()
    if flip_x:
        base_out[:, 0] = -base_in[:, 0]
        pitch_out      = -pitch_in        # x-flip couples to pitch tilt
    if flip_y:
        base_out[:, 1] = -base_in[:, 1]
        roll_out       = -roll_in         # y-flip couples to roll tilt

    # Leg permutation: maps source leg index → target leg slot.
    perm = {"fb": _PERM_FB, "lr": _PERM_LR, "both": _PERM_BOTH}[mirror_op]

    joints_out     = np.empty_like(joints_in)
    total_violations = 0

    for k in range(N):
        # FK on source knot.
        _set_qpos(mj_data, base_in[k], pitch_in[k], roll_in[k], joints_in[k])
        mujoco.mj_forward(model, mj_data)
        foot_world = _get_foot_world(model, mj_data, tibia_ids)   # (6, 3)

        # Flip foot world coordinates.
        foot_flipped = foot_world.copy()
        if flip_x:
            foot_flipped[:, 0] = -foot_world[:, 0]
        if flip_y:
            foot_flipped[:, 1] = -foot_world[:, 1]

        # Permute: foot that was leg perm[i] now belongs to leg i.
        foot_permuted = foot_flipped[perm]   # (6, 3)

        # Convert flipped+permuted world positions into body frame for the
        # mirrored knot.  Body yaw is 0 throughout (TO trajectories start at
        # the world origin with identity heading), so body frame == world
        # frame minus the body translation.
        foot_body = foot_permuted - base_out[k]   # (6, 3)

        try:
            j = ctrl.body_to_joints(foot_body)
        except Exception as exc:
            print(f"  WARN knot {k}: IK failed ({exc}), carrying source joints")
            j = joints_in[k].copy()

        joints_out[k] = j
        total_violations += _check_rom(j, limits, k)

    if total_violations:
        print(f"  total RoM violations: {total_violations} (saved anyway)")
    else:
        print("  all joints within RoM")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                        joints=joints_out.astype(np.float64),
                        base=base_out.astype(np.float64),
                        pitch=pitch_out.astype(np.float64),
                        roll=roll_out.astype(np.float64),
                        dt=dt)
    print(f"  saved {N} knots to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Mirror a TO trajectory (FB, LR, or both).")
    p.add_argument("--source", required=True,
                   help="Input npz path (.cache/to_trajectory*.npz)")
    p.add_argument("--mirror", required=True, choices=["fb", "lr", "both"],
                   help="fb=front-back flip, lr=left-right flip, both=fb+lr")
    p.add_argument("--out", required=True,
                   help="Output npz path")
    args = p.parse_args()

    mirror_trajectory(args.source, args.mirror, args.out)


if __name__ == "__main__":
    main()
