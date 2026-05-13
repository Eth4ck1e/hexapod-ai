"""tools/to_parallel_sweep.py — parallel TO config sweep with metric ranking.

Runs N TO configurations in parallel via multiprocessing.Pool, capturing
per-config result npz + metrics + per-config solver log. Produces a ranked
summary CSV so we can quickly see which configs produced the best gait
candidates without re-running solves serially.

Usage:
    PYTHONPATH=. .venv/Scripts/python.exe tools/to_parallel_sweep.py \
        --sweep smoothness_ladder --n-workers 6

Built-in sweeps (use --list to see all):
  smoothness_ladder    — vary w_body_accel + w_joint_jerk
  symmetry_ladder      — vary w_sym
  pose_ladder          — vary w_pose (allow more body tilt = more natural?)
  height_ladder        — vary body_height target
  swing_ladder         — vary swing_clearance
  directions_8way      — 8 omnidirectional headings at fixed speed
  mixed_explore        — hand-curated set worth visualizing

Output structure:
  tools/sweep_results/<sweep_name>/
      <config_id>/
          stdout.log     — IPOPT output for this solve
          result.npz     — full solver output (joints/base/pitch/roll/feet)
          metrics.json   — compute_metrics(sol) + config + status
      summary.csv        — all configs ranked by composite score
      ranking.txt        — top-5 readable summary
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# HSL bootstrap (no-op on non-Windows; harmless if HSL absent).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hsl_bootstrap import HSL_AVAILABLE  # noqa: F401

import numpy as np


# Default reference TO config (matches to_solver.build_and_solve_to defaults).
DEFAULTS = dict(
    vx=0.17, vy=0.0,
    duration_s=8.0, n_strides=11, knots_per_phase=10,
    body_height=0.145, swing_clearance=0.012,
    max_pitch_roll_deg=0.45, body_z_tol=0.0045,
    w_sym=1e4, w_energy=1.0, w_joint_jerk=1.0,
    w_pose=5e6, w_height=3e8, w_speed=5e2,
    w_body_accel=150.0, w_vx_track=0.0, w_stability=0.0,
    directional_tol=0.0009,
    linear_solver="mumps",
)


# ------------------------------------------------------------------------
# Sweep definitions — each is a list of (config_id, config_overrides_dict).
# Overrides are applied on top of DEFAULTS; missing keys fall through.
# ------------------------------------------------------------------------
def _sweep_smoothness_ladder():
    """Vary smoothness penalties — does heavier smoothing produce nicer gait?"""
    out = []
    for accel_mult in [0.5, 1.0, 2.0, 5.0]:
        for jerk_mult in [0.5, 1.0, 2.0]:
            cfg_id = f"accel{accel_mult:g}_jerk{jerk_mult:g}"
            out.append((cfg_id, {
                "w_body_accel": DEFAULTS["w_body_accel"] * accel_mult,
                "w_joint_jerk": DEFAULTS["w_joint_jerk"] * jerk_mult,
            }))
    return out


def _sweep_symmetry_ladder():
    out = []
    for sym_mult in [0.1, 0.3, 1.0, 3.0, 10.0]:
        cfg_id = f"sym{sym_mult:g}x"
        out.append((cfg_id, {"w_sym": DEFAULTS["w_sym"] * sym_mult}))
    return out


def _sweep_pose_ladder():
    """Lower w_pose may allow more natural body tilt; too low collapses gait."""
    out = []
    for pose_mult in [0.01, 0.1, 0.5, 1.0, 3.0]:
        cfg_id = f"pose{pose_mult:g}x"
        out.append((cfg_id, {"w_pose": DEFAULTS["w_pose"] * pose_mult}))
    return out


def _sweep_height_ladder():
    out = []
    for h in [0.110, 0.125, 0.135, 0.145, 0.155]:
        cfg_id = f"h{int(h * 1000)}mm"
        out.append((cfg_id, {"body_height": h}))
    return out


def _sweep_swing_ladder():
    out = []
    for sw in [0.006, 0.010, 0.015, 0.020, 0.030]:
        cfg_id = f"swing{int(sw * 1000)}mm"
        out.append((cfg_id, {"swing_clearance": sw}))
    return out


def _sweep_directions_8way():
    """8 omnidirectional headings at MAX_SPEED * 0.8."""
    speed = 0.17  # m/s (matches DEFAULTS.vx)
    out = []
    for deg in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = math.radians(deg)
        vx = speed * math.cos(rad)
        vy = speed * math.sin(rad)
        cfg_id = f"dir{deg:03d}"
        out.append((cfg_id, {"vx": vx, "vy": vy}))
    return out


def _sweep_mixed_explore():
    """A small hand-curated set spanning the most interesting axes."""
    return [
        ("baseline",            {}),
        ("smoother_2x",         {"w_body_accel": 300.0, "w_joint_jerk": 2.0}),
        ("looser_pose_0.1x",    {"w_pose": 5e5}),
        ("tighter_sym_3x",      {"w_sym": 3e4}),
        ("higher_swing_20mm",   {"swing_clearance": 0.020}),
        ("low_body_125mm",      {"body_height": 0.125}),
        ("speed_track_2x",      {"w_speed": 1e3}),
        ("slower_50pct",        {"vx": 0.085, "duration_s": 8.0}),
    ]


SWEEPS = {
    "smoothness_ladder": _sweep_smoothness_ladder,
    "symmetry_ladder":   _sweep_symmetry_ladder,
    "pose_ladder":       _sweep_pose_ladder,
    "height_ladder":     _sweep_height_ladder,
    "swing_ladder":      _sweep_swing_ladder,
    "directions_8way":   _sweep_directions_8way,
    "mixed_explore":     _sweep_mixed_explore,
}


# ------------------------------------------------------------------------
# Worker (runs in a child process) — solves one TO config end-to-end and
# writes its results to disk. Returns a small status dict; full solution
# stays on disk so it doesn't have to round-trip through pickle.
# ------------------------------------------------------------------------
def _run_one_config(args_tuple) -> dict:
    cfg_id, overrides, out_root_str = args_tuple
    out_root = Path(out_root_str)
    cfg_dir = out_root / cfg_id
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_path     = cfg_dir / "stdout.log"
    npz_path     = cfg_dir / "result.npz"
    metrics_path = cfg_dir / "metrics.json"

    # Merge overrides on top of defaults.
    cfg = dict(DEFAULTS)
    cfg.update(overrides)

    # Inside the child process: silence-redirect IPOPT chatter to a file so
    # parallel workers don't interleave on the parent's terminal. CasADi's
    # IPOPT prints from native code — only print_level=0 reliably mutes it,
    # so we still redirect Python-level stdout/stderr too.
    t0 = time.perf_counter()
    status = "Failed_BeforeSolve"
    err_msg = ""
    metrics: dict[str, Any] = {}
    try:
        with open(log_path, "w") as logf:
            with redirect_stdout(logf), redirect_stderr(logf):
                # Lazy import inside child so each worker pays the import cost
                # only once across its task list (Pool reuses workers).
                import to_solver as ts
                sol = ts.solve_with_warmstart(
                    vx=cfg.pop("vx"),
                    vy=cfg.pop("vy"),
                    duration_s=cfg.pop("duration_s"),
                    n_strides=cfg.pop("n_strides"),
                    knots_per_phase=cfg.pop("knots_per_phase"),
                    **cfg,
                )
                metrics = ts.compute_metrics(sol)
                np.savez_compressed(
                    npz_path,
                    joints=sol["joints"],
                    feet_body=sol["feet_body"],
                    base=sol["base"],
                    pitch=sol["pitch"],
                    roll=sol["roll"],
                    dt=sol["dt"],
                    duration_s=sol["duration_s"],
                    vx_cmd=sol["vx_cmd"],
                    vy_cmd=sol["vy_cmd"],
                )
                status = sol.get("ipopt_status", "Unknown")
                success = bool(sol.get("success", False))
    except Exception as e:
        success = False
        err_msg = f"{type(e).__name__}: {e}"
        status = "WorkerException"
    elapsed = time.perf_counter() - t0

    # Per-config metrics file (everything we know about this run).
    out = {
        "cfg_id":      cfg_id,
        "overrides":   overrides,
        "status":      status,
        "success":     success,
        "elapsed_s":   elapsed,
        "err":         err_msg,
        "metrics":     metrics,
        "result_npz":  str(npz_path) if npz_path.exists() else "",
    }
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


# ------------------------------------------------------------------------
# Ranking — composite score (lower is better) over normalized metrics.
# Failures rank below all successes.
# ------------------------------------------------------------------------
RANK_METRICS = (
    "peak_joint_vel_radps",     # tracker stress
    "peak_abs_body_ax_mps2",    # body smoothness
    "peak_abs_body_ay_mps2",
    "max_abs_pitch_deg",
    "max_abs_roll_deg",
    "body_z_dev_max_m",
    "base_x_err_max_mm",
    "base_y_err_max_mm",
)


def _rank(results: list[dict]) -> list[dict]:
    """Returns results with an added 'composite_score' field, sorted ascending."""
    successes = [r for r in results if r["success"] and r["metrics"]]
    failures  = [r for r in results if not (r["success"] and r["metrics"])]
    if not successes:
        return results

    # Z-score each metric across successful runs, sum.
    metric_arrays = {m: np.array([r["metrics"].get(m, np.nan) for r in successes])
                     for m in RANK_METRICS}
    z_scores = {m: (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)
                for m, a in metric_arrays.items()}

    for i, r in enumerate(successes):
        score = sum(z_scores[m][i] for m in RANK_METRICS)
        r["composite_score"] = float(score)

    successes.sort(key=lambda r: r["composite_score"])
    for r in failures:
        r["composite_score"] = float("inf")
    return successes + failures


def _write_summary(out_root: Path, results: list[dict]):
    csv_path = out_root / "summary.csv"
    fieldnames = [
        "rank", "cfg_id", "success", "status", "composite_score", "elapsed_s",
        *RANK_METRICS,
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, r in enumerate(results, 1):
            row = {
                "rank":            rank,
                "cfg_id":          r["cfg_id"],
                "success":         r["success"],
                "status":          r["status"],
                "composite_score": f"{r.get('composite_score', float('nan')):.3f}",
                "elapsed_s":       f"{r['elapsed_s']:.1f}",
            }
            for m in RANK_METRICS:
                v = r.get("metrics", {}).get(m)
                row[m] = f"{v:.4f}" if isinstance(v, (int, float)) else ""
            w.writerow(row)

    # Readable top-5
    txt_path = out_root / "ranking.txt"
    with open(txt_path, "w") as f:
        f.write(f"Sweep summary — {len(results)} configs\n")
        f.write("=" * 60 + "\n\n")
        for rank, r in enumerate(results[:5], 1):
            f.write(f"#{rank}  {r['cfg_id']}  score={r.get('composite_score', float('nan')):+.2f}\n")
            f.write(f"     status={r['status']}  elapsed={r['elapsed_s']:.1f}s\n")
            f.write(f"     overrides: {r['overrides']}\n")
            for m in RANK_METRICS:
                v = r.get("metrics", {}).get(m)
                if isinstance(v, (int, float)):
                    f.write(f"     {m}: {v:.4f}\n")
            f.write("\n")
    print(f"  wrote {csv_path}")
    print(f"  wrote {txt_path}")


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", choices=list(SWEEPS), default=None,
                   help="which sweep to run")
    p.add_argument("--list", action="store_true",
                   help="list available sweeps and their sizes")
    p.add_argument("--n-workers", type=int, default=6,
                   help="multiprocessing pool size (default 6, leaves 2 cores idle)")
    p.add_argument("--out-dir", default="tools/sweep_results",
                   help="output root")
    args = p.parse_args()

    if args.list or not args.sweep:
        print("Available sweeps:")
        for name, fn in SWEEPS.items():
            configs = fn()
            print(f"  {name:24s}  {len(configs):2d} configs")
            for cfg_id, ov in configs[:3]:
                print(f"      • {cfg_id:24s}  {ov}")
            if len(configs) > 3:
                print(f"      • ... +{len(configs) - 3} more")
        return

    sweep_fn = SWEEPS[args.sweep]
    configs  = sweep_fn()
    out_root = Path(args.out_dir) / args.sweep
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"TO sweep: {args.sweep}")
    print(f"  {len(configs)} configs × ~{args.n_workers} workers")
    print(f"  out: {out_root}")
    print("=" * 60)

    work = [(cid, ov, str(out_root)) for cid, ov in configs]

    t0 = time.perf_counter()
    # Spawn context — required on Windows, safe on Linux/macOS too.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.n_workers) as pool:
        results = []
        for r in pool.imap_unordered(_run_one_config, work):
            tag = "OK" if r["success"] else "FAIL"
            print(f"  [{tag}] {r['cfg_id']:24s}  status={r['status']:30s}  "
                  f"t={r['elapsed_s']:.0f}s")
            results.append(r)
    elapsed = time.perf_counter() - t0
    print(f"\nSweep done in {elapsed:.0f}s")

    print("\nRanking by composite score (lower = smoother + tighter tracking)...")
    ranked = _rank(results)
    _write_summary(out_root, ranked)

    print("\nTop-3:")
    for r in ranked[:3]:
        print(f"  {r['cfg_id']:24s}  score={r.get('composite_score', float('nan')):+.2f}")


if __name__ == "__main__":
    main()
