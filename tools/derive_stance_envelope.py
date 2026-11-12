"""tools/derive_stance_envelope.py — kinematic stance envelope for a robot profile.

Sweeps stance height (cmd[5] = dh) and stance width (cmd[6] = dw) and asks,
for each (dh, dw) pair, whether the analytical scaffold can execute a full
gait cycle of every basic motion (forward, strafe, spin at scaffold max)
without any joint leaving its MJCF range or the IK clamping an unreachable
foot target. The feasible region is then summarised as the same linear,
dh-conditional envelope that `envs/stance_envelope.py` consumes:

    max_dw(dh) = MAX_DW_INTERCEPT + MAX_DW_SLOPE * dh
    min_dw(dh) = MIN_DW_INTERCEPT + MIN_DW_SLOPE * dh

Both lines are shifted so they never exceed the measured feasible range at
any sampled dh (a conservative inner fit, like the hand-fitted PhantomX one).

The PhantomX envelope in `gait/robot_profile.py` was verified by eye on the
mesh model (self-collision included) and is kept as-is; this tool exists to
produce a defensible first envelope for a chassis we cannot inspect yet
(the mini hexapod) and to re-derive it after the real bot is measured.

Usage:
    PYTHONPATH=. python tools/derive_stance_envelope.py --model models/mini_hexapod_mjx.xml
    PYTHONPATH=. python tools/derive_stance_envelope.py --model models/phantomx_simple_mjx.xml
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

from gait.controller import Controller

_LEGS = ("RR", "RM", "RF", "LR", "LM", "LF")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _motion_cmds(ctrl: Controller, speed_frac: float) -> list[np.ndarray]:
    """Representative worst-case motions at the scaffold's max stride."""
    v = speed_frac * ctrl.MAX_SPEED
    w = speed_frac * ctrl.MAX_YAW_RATE
    cmds = []
    for vx, vy, wz in ((v, 0, 0), (-v, 0, 0), (0, v, 0), (0, -v, 0), (0, 0, w), (0, 0, -w)):
        c = np.zeros(9)
        c[0], c[1], c[2] = vx, vy, wz
        cmds.append(c)
    return cmds


def _feasible(ctrl: Controller, model, data, tib_ids, jnt_lo, jnt_hi, cmds,
              dh: float, dw: float, n_t: int, fk_tol: float) -> bool:
    period = ctrl.gait_period
    for base in cmds:
        cmd = base.copy()
        cmd[5], cmd[6] = dh, dw
        for k in range(n_t):
            t = period * k / n_t
            joints, feet_body = ctrl.predict_with_feet(cmd, t)
            if np.any(joints < jnt_lo) or np.any(joints > jnt_hi):
                return False
            data.qpos[:] = 0.0
            data.qpos[3] = 1.0
            data.qpos[7:25] = joints
            mujoco.mj_forward(model, data)
            for i in range(6):
                fk = data.xpos[tib_ids[i]] + data.xmat[tib_ids[i]].reshape(3, 3) @ ctrl.FOOT_TIP_LOCAL[i]
                if np.linalg.norm(fk - feet_body[i]) > fk_tol:
                    return False
    return True


def _inner_line(dh: np.ndarray, y: np.ndarray, upper: bool):
    """Least-squares line through (dh, y), then shifted so it sits on the
    safe side of every sample (<= y for an upper bound, >= y for a lower)."""
    slope, intercept = np.polyfit(dh, y, 1)
    resid = y - (intercept + slope * dh)
    intercept += resid.min() if upper else resid.max()
    return float(intercept), float(slope)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/mini_hexapod_mjx.xml")
    ap.add_argument("--dh-span", type=float, default=None,
                    help="search |dh| up to this (m). Default: 0.6 × tibia length")
    ap.add_argument("--dw-span", type=float, default=None,
                    help="search |dw| up to this (m). Default: 1.0 × tibia length")
    ap.add_argument("--dh-steps", type=int, default=25)
    ap.add_argument("--dw-steps", type=int, default=41)
    ap.add_argument("--speed-frac", type=float, default=1.0,
                    help="fraction of scaffold MAX_SPEED / MAX_YAW_RATE to test at")
    ap.add_argument("--samples-per-cycle", type=int, default=16)
    ap.add_argument("--joint-margin-deg", type=float, default=3.0,
                    help="stay this far inside every MJCF joint range")
    ap.add_argument("--fk-tol-mm", type=float, default=5.0,
                    help="IK↔FK mismatch above this = unreachable target (clamped IK errs by cm; "
                         "PhantomX honest IK error is ~2.5 mm at max stride)")
    ap.add_argument("--shrink", type=float, default=0.90,
                    help="final dh range is scaled by this for safety margin")
    args = ap.parse_args()

    model_path = args.model if Path(args.model).is_absolute() else str(_PROJECT_ROOT / args.model)
    ctrl = Controller(model_path)
    prof = ctrl.profile
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    tib_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tibia_{n}") for n in _LEGS]

    # Hinge joint ranges in qpos[7:25] order (first joint is the free joint).
    margin = math.radians(args.joint_margin_deg)
    rng_all = model.jnt_range[1:19]
    limited = model.jnt_limited[1:19].astype(bool)
    jnt_lo = np.where(limited, rng_all[:, 0] + margin, -np.inf)
    jnt_hi = np.where(limited, rng_all[:, 1] - margin, np.inf)

    L3 = prof.tibia_length
    dh_span = args.dh_span if args.dh_span is not None else 0.6 * L3
    dw_span = args.dw_span if args.dw_span is not None else 1.0 * L3
    dh_grid = np.linspace(-dh_span, dh_span, args.dh_steps)
    dw_grid = np.linspace(-dw_span, dw_span, args.dw_steps)
    cmds = _motion_cmds(ctrl, args.speed_frac)
    fk_tol = args.fk_tol_mm * 1e-3

    print(f"profile={prof.name}  model={Path(model_path).name}")
    print(f"MAX_SPEED={ctrl.MAX_SPEED:.3f} m/s  MAX_YAW_RATE={ctrl.MAX_YAW_RATE:.3f} rad/s  "
          f"period={ctrl.gait_period:.2f}s  stride R={ctrl.path_radius*1e3:.0f} mm")
    print(f"sweep dh ∈ ±{dh_span*1e3:.0f} mm ({args.dh_steps}), dw ∈ ±{dw_span*1e3:.0f} mm "
          f"({args.dw_steps}), {len(cmds)} motions × {args.samples_per_cycle} phases each")

    rows = []   # (dh, min_dw, max_dw) for feasible dh
    print(f"\n{'dh (mm)':>8} {'min dw':>8} {'max dw':>8}")
    for dh in dh_grid:
        feas = np.array([_feasible(ctrl, model, data, tib_ids, jnt_lo, jnt_hi, cmds,
                                   dh, dw, args.samples_per_cycle, fk_tol) for dw in dw_grid])
        i0 = int(np.argmin(np.abs(dw_grid)))        # dw = 0 column
        if not feas[i0]:
            print(f"{dh*1e3:8.1f} {'--':>8} {'--':>8}   (dw=0 infeasible)")
            continue
        lo = i0
        while lo - 1 >= 0 and feas[lo - 1]:
            lo -= 1
        hi = i0
        while hi + 1 < len(dw_grid) and feas[hi + 1]:
            hi += 1
        rows.append((dh, dw_grid[lo], dw_grid[hi]))
        print(f"{dh*1e3:8.1f} {dw_grid[lo]*1e3:8.1f} {dw_grid[hi]*1e3:8.1f}")

    if not rows:
        raise SystemExit("no feasible stance found — check joint ranges / gait defaults")

    rows = np.array(rows)
    # Contiguous feasible dh block around dh = 0.
    i0 = int(np.argmin(np.abs(rows[:, 0])))
    lo = i0
    while lo - 1 >= 0 and np.isclose(rows[lo - 1, 0], rows[lo, 0] - (dh_grid[1] - dh_grid[0])):
        lo -= 1
    hi = i0
    while hi + 1 < len(rows) and np.isclose(rows[hi + 1, 0], rows[hi, 0] + (dh_grid[1] - dh_grid[0])):
        hi += 1
    rows = rows[lo:hi + 1]

    dh_min = float(rows[0, 0]) * args.shrink
    dh_max = float(rows[-1, 0]) * args.shrink
    keep = (rows[:, 0] >= dh_min - 1e-9) & (rows[:, 0] <= dh_max + 1e-9)
    r = rows[keep] if keep.sum() >= 2 else rows
    min_int, min_slope = _inner_line(r[:, 0], r[:, 1], upper=False)
    max_int, max_slope = _inner_line(r[:, 0], r[:, 2], upper=True)
    presets = tuple(round(float(x), 4) for x in np.linspace(dh_min, dh_max, 5))
    dw_joint_min = min(min_int + min_slope * dh_min, min_int + min_slope * dh_max)
    dw_joint_max = max(max_int + max_slope * dh_min, max_int + max_slope * dh_max)

    print("\n# ---- suggested RobotProfile stance fields (paste into gait/robot_profile.py) ----")
    print(f"    dh_min={dh_min:.4f}, dh_max={dh_max:.4f},")
    print(f"    height_presets={presets},")
    print(f"    max_dw_intercept={max_int:.4f}, max_dw_slope={max_slope:.4f},")
    print(f"    min_dw_intercept={min_int:.4f}, min_dw_slope={min_slope:.4f},")
    print(f"    dw_joint_min={math.floor(dw_joint_min * 1e3) / 1e3:.3f}, "
          f"dw_joint_max={math.ceil(dw_joint_max * 1e3) / 1e3:.3f},")
    print("\n# envelope check at the presets:")
    for p in presets:
        print(f"   dh={p*1e3:6.1f} mm → dw ∈ [{(min_int + min_slope*p)*1e3:6.1f}, "
              f"{(max_int + max_slope*p)*1e3:6.1f}] mm")


if __name__ == "__main__":
    main()
