"""derive_joint_limits.py — empirically discover the safe range of motion
(RoM) for every hexapod joint by collision-sweeping the simplified MJCF.

Approach
--------
For each of the 18 joints (3 per leg × 6 legs):
  1. Reset the bot to NEUTRAL_POSE (gait library's calibrated rest pose).
  2. Sweep the swept joint outward in small steps (default 0.005 rad) in
     both + and - directions while keeping every other joint fixed.
  3. At every step run `mj_forward` then `mj_collision`. Filter out the
     foot-floor contacts that are present at NEUTRAL_POSE (planted feet)
     by ignoring contacts where either geom is the floor. Anything else
     is a self-collision.
  4. Stop when self-collision is detected (record previous angle), the
     hard MJCF range limit is hit (record the limit), or a global
     ±SAFETY_BOUND fence is exceeded.

After collecting the 18 (lo, hi) tuples, group joints by the symmetry
classes the user described (outer-rear, outer-front, middle) × right/left
and average within each group while accounting for the per-leg sign
flips on femur and tibia (see gait.controller._joint_signs). Finally
apply a small inward buffer (max(5%-of-range, 2°)) on each side and
write the JSON. The companion script `apply_joint_limits.py` patches
the three MJCFs from the JSON.

Run validation with --validate to confirm no in-range angle triggers a
self-collision in the patched models.

Usage
-----
    .venv/Scripts/python.exe derive_joint_limits.py
    .venv/Scripts/python.exe derive_joint_limits.py --validate
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import mujoco
import numpy as np

# Allow importing the gait package from the repo root.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gait import NEUTRAL_POSE  # noqa: E402
from gait.controller import _joint_signs  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LEG_NAMES = ["RR", "RM", "RF", "LR", "LM", "LF"]
JOINT_TYPES = ["coxa", "femur", "tibia"]

# Sweep parameters.
STEP_RAD = 0.005          # 0.005 rad ≈ 0.29°
SAFETY_BOUND = math.pi    # absolute fence on either side of neutral

# Buffer applied inward when going from empirical → buffered.
BUFFER_FRAC = 0.05         # 5% of the empirical range
BUFFER_MIN_RAD = math.radians(2.0)  # but at least 2°

# File paths.
MODEL_DIR = ROOT / "models"
SIMPLE_MJCF = MODEL_DIR / "phantomx_simple.xml"
MESH_MJCF = MODEL_DIR / "phantomx.xml"
MJX_MJCF = MODEL_DIR / "phantomx_simple_mjx.xml"
JSON_OUT = ROOT / "joint_limits.json"


def joint_name(joint_type: str, leg: str) -> str:
    return f"{joint_type}_joint_{leg}"


# ---------------------------------------------------------------------------
# Sweeping
# ---------------------------------------------------------------------------
def _floor_geom_id(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


def _set_neutral_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Plant the bot at NEUTRAL_POSE with feet roughly on the ground.

    The free joint occupies qpos[0:7]; the 18 leg joints come right after
    in MJCF declaration order. NEUTRAL_POSE is also in that same order.
    """
    data.qpos[:] = 0.0
    # free joint: pos = (0, 0, 0.18), quat = (1, 0, 0, 0)
    data.qpos[0:3] = [0.0, 0.0, 0.18]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:25] = NEUTRAL_POSE
    data.qvel[:] = 0.0


def _has_self_collision(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_id: int,
    relevant_geoms: set | None = None,
) -> bool:
    """Run collision detection and return True if any contact pair is
    between two non-floor geoms.

    If `relevant_geoms` is given, only count a contact when BOTH geoms
    are in that set. Use this to filter out collisions involving distal
    links that move along with the swept joint as a kinematic chain
    (their collisions don't reflect the swept joint's own range).
    """
    mujoco.mj_collision(model, data)
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == floor_id or c.geom2 == floor_id:
            continue
        if relevant_geoms is not None:
            if c.geom1 not in relevant_geoms or c.geom2 not in relevant_geoms:
                continue
        return True
    return False


def _bodies_under(model: mujoco.MjModel, root_body_id: int) -> list[int]:
    """All body IDs in the kinematic subtree rooted at `root_body_id`."""
    out = [root_body_id]
    stack = [root_body_id]
    while stack:
        parent = stack.pop()
        for b in range(model.nbody):
            if int(model.body_parentid[b]) == parent:
                out.append(b)
                stack.append(b)
    return out


def _geoms_on_bodies(model: mujoco.MjModel, body_ids) -> set[int]:
    """All geom IDs whose `body_id` is in the given iterable."""
    body_set = set(int(b) for b in body_ids)
    return {g for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in body_set}


def _relevant_geoms_for_sweep(
    model: mujoco.MjModel, joint_type: str, leg: str
) -> set[int]:
    """Build the geom set whose collisions matter for sweeping
    `joint_type`_joint_`leg`. Filters out distal-chain effects.

    Coxa sweep:   chassis + this-leg coxa + ALL other coxas (the leg's
                  swing arc can hit a neighbor coxa).
    Femur sweep:  chassis + this-leg coxa + this-leg femur. (Adjacent
                  legs aren't usually a constraint on femur pitch from
                  neutral; if they ever are, widen here.)
    Tibia sweep:  chassis + this-leg coxa + femur + tibia.
    """
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    chassis_geoms = _geoms_on_bodies(model, [base_id])

    coxa_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"coxa_{leg}")
    femur_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"femur_{leg}")
    tibia_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tibia_{leg}")

    if joint_type == "coxa":
        # All coxas on every leg, plus chassis.
        all_coxa_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"coxa_{l}")
            for l in LEG_NAMES
        ]
        leg_geoms = _geoms_on_bodies(model, all_coxa_ids)
    elif joint_type == "femur":
        leg_geoms = _geoms_on_bodies(model, [coxa_id, femur_id])
    else:  # tibia
        leg_geoms = _geoms_on_bodies(model, [coxa_id, femur_id, tibia_id])

    return chassis_geoms | leg_geoms


def _sweep_one_joint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_id: int,
    qpos_adr: int,
    floor_id: int,
    direction: int,
    range_lo: float,
    range_hi: float,
    neutral_value: float,
    relevant_geoms: set | None = None,
) -> tuple[float, str]:
    """Sweep one joint in one direction. Returns (angle, stop_reason).

    direction: +1 or -1.
    Returns the last collision-free angle (which may equal the MJCF
    range limit if no collision is found before the limit).
    """
    last_safe = neutral_value
    angle = neutral_value
    while True:
        next_angle = angle + direction * STEP_RAD

        # Hit the MJCF hard range?
        if direction > 0 and next_angle > range_hi:
            return range_hi, "mjcf_limit_hi"
        if direction < 0 and next_angle < range_lo:
            return range_lo, "mjcf_limit_lo"

        # Hit the global safety fence?
        if abs(next_angle - neutral_value) > SAFETY_BOUND:
            return last_safe, "safety_fence"

        # Apply candidate angle.
        # Reset to neutral first (other joints stay at neutral).
        _set_neutral_qpos(model, data)
        data.qpos[qpos_adr] = next_angle
        mujoco.mj_forward(model, data)

        if _has_self_collision(model, data, floor_id, relevant_geoms):
            return last_safe, "self_collision"

        last_safe = next_angle
        angle = next_angle


def sweep_all_joints(model_path: Path) -> dict:
    """Return {joint_name: {empirical: [lo, hi], stop_lo, stop_hi}}."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    floor_id = _floor_geom_id(model)
    if floor_id < 0:
        raise RuntimeError("floor geom not found in model")

    results: dict = OrderedDict()

    for leg_idx, leg in enumerate(LEG_NAMES):
        for j_offset, jt in enumerate(JOINT_TYPES):
            jn = joint_name(jt, leg)
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise RuntimeError(f"joint {jn} not found")
            qpos_adr = int(model.jnt_qposadr[jid])
            range_lo, range_hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
            neutral_value = float(NEUTRAL_POSE[leg_idx * 3 + j_offset])

            relevant = _relevant_geoms_for_sweep(model, jt, leg)

            hi_angle, hi_reason = _sweep_one_joint(
                model, data, jid, qpos_adr, floor_id,
                +1, range_lo, range_hi, neutral_value, relevant,
            )
            lo_angle, lo_reason = _sweep_one_joint(
                model, data, jid, qpos_adr, floor_id,
                -1, range_lo, range_hi, neutral_value, relevant,
            )
            results[jn] = {
                "empirical": [lo_angle, hi_angle],
                "stop_reason_lo": lo_reason,
                "stop_reason_hi": hi_reason,
                "neutral": neutral_value,
                "mjcf_range": [range_lo, range_hi],
            }
    return results


# ---------------------------------------------------------------------------
# Symmetry handling
# ---------------------------------------------------------------------------
# Position groups: outer rear (RR, LR), outer front (RF, LF), middle (RM, LM).
POSITION_GROUPS = {
    "outer_rear": ["RR", "LR"],
    "outer_front": ["RF", "LF"],
    "middle":     ["RM", "LM"],
}


def _signed_to_logical(joint_type: str, leg_idx: int, value: float) -> float:
    """Map an MJCF-frame joint angle to the formula/logical frame so that
    R and L mirror partners agree on the *physical* direction.

    Femur and tibia: use the MJCF axis sign from `_joint_signs` (femur axis
    is flipped between R and L; tibia axis is also flipped *and* the
    formula bend convention is opposite of MJCF positive — see the
    docstring on _joint_signs).

    Coxa: `_joint_signs` returns +1 for both R and L, because both coxa
    joints share axis=(0,0,1) in MJCF. BUT — the legs mount on opposite
    body sides, so the *physical* swing direction of a +rotation is
    OPPOSITE between an R and L coxa: e.g. RR's +0.5 rad swings the leg
    forward (toward RM), LR's +0.5 rad swings backward (away from LM).
    To pair them as physical mirrors we therefore apply an explicit
    body-side flip on coxa: sign=-1 for left-side legs.
    """
    s_c, s_f, s_t = _joint_signs(leg_idx)
    is_left = leg_idx >= 3
    if joint_type == "coxa":
        sign = -1.0 if is_left else +1.0
    elif joint_type == "femur":
        sign = s_f
    else:  # tibia
        sign = s_t
    return sign * value


def _logical_to_signed(joint_type: str, leg_idx: int, value: float) -> float:
    """Inverse of _signed_to_logical (sign flip)."""
    return _signed_to_logical(joint_type, leg_idx, value)  # involution


def symmetrize(empirical: dict) -> dict:
    """For each (position_group, joint_type) symmetric pair, average the
    two legs in the *logical* frame and copy the symmetrized result back
    into each leg with the appropriate sign.

    Returns a new dict {joint_name: {"empirical": [...], "buffered":
    [...]}}.
    """
    out: dict = OrderedDict()

    for group_name, legs in POSITION_GROUPS.items():
        for jt in JOINT_TYPES:
            # Collect logical-frame (lo, hi) from each leg.
            logical_los, logical_his = [], []
            for leg in legs:
                jn = joint_name(jt, leg)
                lo, hi = empirical[jn]["empirical"]
                leg_idx = LEG_NAMES.index(leg)
                # Mapping MJCF→logical with a sign flip can swap lo/hi.
                a = _signed_to_logical(jt, leg_idx, lo)
                b = _signed_to_logical(jt, leg_idx, hi)
                lo_l, hi_l = (a, b) if a <= b else (b, a)
                logical_los.append(lo_l)
                logical_his.append(hi_l)

            # Average in logical frame.
            avg_lo = float(np.mean(logical_los))
            avg_hi = float(np.mean(logical_his))

            # Inward buffer.
            rng = avg_hi - avg_lo
            buf = max(rng * BUFFER_FRAC, BUFFER_MIN_RAD)
            buf_lo = avg_lo + buf
            buf_hi = avg_hi - buf
            if buf_lo > buf_hi:
                # Range collapsed. Fall back to half the range as a
                # conservative single-sided buffer.
                buf_lo = avg_lo + rng * 0.25
                buf_hi = avg_hi - rng * 0.25

            for leg in legs:
                jn = joint_name(jt, leg)
                leg_idx = LEG_NAMES.index(leg)
                # Map symmetrized logical bounds back to MJCF frame.
                a = _logical_to_signed(jt, leg_idx, avg_lo)
                b = _logical_to_signed(jt, leg_idx, avg_hi)
                emp_lo, emp_hi = (a, b) if a <= b else (b, a)
                a = _logical_to_signed(jt, leg_idx, buf_lo)
                b = _logical_to_signed(jt, leg_idx, buf_hi)
                buf_lo_mj, buf_hi_mj = (a, b) if a <= b else (b, a)
                out[jn] = {
                    "empirical_raw": empirical[jn]["empirical"],
                    "empirical": [emp_lo, emp_hi],
                    "buffered": [buf_lo_mj, buf_hi_mj],
                    "stop_reason_lo": empirical[jn]["stop_reason_lo"],
                    "stop_reason_hi": empirical[jn]["stop_reason_hi"],
                    "neutral": empirical[jn]["neutral"],
                    "group": group_name,
                }
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(model_path: Path, limits: dict, samples: int = 32) -> dict:
    """For each joint, sample N angles in [buffered_lo, buffered_hi] (incl.
    endpoints) and confirm no self-collision. Returns a {joint:bool}
    pass/fail dict and prints a summary.
    """
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    floor_id = _floor_geom_id(model)
    results: dict = OrderedDict()

    print(f"\n--- validation: {model_path.name} ---")
    print(f"{'joint':<22} {'lo':>7} {'hi':>7}   result")

    for jn, info in limits.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qpos_adr = int(model.jnt_qposadr[jid])
        lo, hi = info["buffered"]

        # Use the same per-joint relevant-geom filter as the sweep so the
        # validator measures by the same yardstick. e.g. validating coxa
        # only checks coxa-vs-chassis / coxa-vs-coxa collisions; the
        # distal femur/tibia swinging into something doesn't count as a
        # coxa-limit failure.
        jt, leg = jn.split("_joint_")          # "coxa_joint_RR" → "coxa", "RR"
        relevant = _relevant_geoms_for_sweep(model, jt, leg)

        ok = True
        bad_angle = None
        angles = np.linspace(lo, hi, samples)
        for a in angles:
            _set_neutral_qpos(model, data)
            data.qpos[qpos_adr] = float(a)
            mujoco.mj_forward(model, data)
            if _has_self_collision(model, data, floor_id, relevant):
                ok = False
                bad_angle = float(a)
                break
        results[jn] = ok
        tag = "PASS" if ok else f"FAIL@{bad_angle:+.3f}"
        print(f"{jn:<22} {lo:+7.3f} {hi:+7.3f}   {tag}")

    n_pass = sum(1 for v in results.values() if v)
    print(f"\n{n_pass}/{len(results)} joints passed validation")
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(limits: dict) -> None:
    print("\n--- per-joint empirical → buffered ranges (rad) ---")
    print(f"{'joint':<22} {'emp lo':>8} {'emp hi':>8} | {'buf lo':>8} {'buf hi':>8}   stop(lo,hi)")
    for jn, info in limits.items():
        elo, ehi = info["empirical"]
        blo, bhi = info["buffered"]
        rl, rh = info["stop_reason_lo"], info["stop_reason_hi"]
        print(
            f"{jn:<22} {elo:+8.3f} {ehi:+8.3f} | "
            f"{blo:+8.3f} {bhi:+8.3f}   {rl}, {rh}"
        )

    print("\n--- group summary (logical frame, averaged within R/L pair) ---")
    print(f"{'group':<14} {'joint':<6} {'lo':>8} {'hi':>8} {'span':>7}")
    seen = set()
    for jn, info in limits.items():
        # Pick a representative from each (group, joint_type).
        jt = jn.split("_")[0]
        key = (info["group"], jt)
        if key in seen:
            continue
        seen.add(key)
        # Use one of the legs in the group to read back the logical-frame
        # bounds.
        leg = jn.split("_")[-1]
        leg_idx = LEG_NAMES.index(leg)
        lo, hi = info["empirical"]
        a = _signed_to_logical(jt, leg_idx, lo)
        b = _signed_to_logical(jt, leg_idx, hi)
        log_lo, log_hi = (a, b) if a <= b else (b, a)
        print(
            f"{info['group']:<14} {jt:<6} {log_lo:+8.3f} {log_hi:+8.3f} "
            f"{log_hi - log_lo:7.3f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=str(SIMPLE_MJCF),
        help="MJCF to sweep against (default: phantomx_simple.xml)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="After patching, re-sweep all three MJCFs to confirm "
             "buffered ranges are collision-free.",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Run apply_joint_limits.py after writing the JSON.",
    )
    parser.add_argument(
        "--out", default=str(JSON_OUT),
        help="Where to write joint_limits.json",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    print(f"sweeping joints in {model_path.name} ...")
    raw = sweep_all_joints(model_path)
    limits = symmetrize(raw)
    print_summary(limits)

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(limits, f, indent=2)
    print(f"\nwrote {out_path}")

    if args.apply or args.validate:
        # Local import — apply script is in the project root.
        import apply_joint_limits as ap
        ap.apply_to_all(out_path, [SIMPLE_MJCF, MESH_MJCF, MJX_MJCF])

    if args.validate:
        all_ok = True
        for p in (SIMPLE_MJCF, MESH_MJCF, MJX_MJCF):
            res = validate(p, limits)
            if not all(res.values()):
                all_ok = False
        print(
            "\nALL MODELS PASS" if all_ok
            else "\nVALIDATION FAILED — check output above"
        )


if __name__ == "__main__":
    main()
