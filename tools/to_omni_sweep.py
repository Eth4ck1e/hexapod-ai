"""
tools/to_omni_sweep.py — Omnidirectional TO sweep for AMP prior data.

Solves the TO problem over a 12-direction × 3-speed grid:
  directions: 0, 30, 60, ..., 330 degrees (12 evenly spaced on [0°, 360°))
  speeds:     0.10, 0.17, 0.25 m/s (3 magnitudes)
  vx = speed * cos(angle_rad),  vy = speed * sin(angle_rad)

Each trajectory saved to:
  tools/cache/to_omni/dir_<deg>_speed_<mm>.npz
  (3-digit zero-padded direction, speed in mm/s — e.g., dir_090_speed_170.npz)

Total: 36 solves, ~5-10 min each → expect 3-6 hours wall time.

Partial sweep support:
  --max-direction  skip directions higher than this (degrees, 0-330)
  --max-speed-idx  skip speed indices higher than this (0=0.10, 1=0.17, 2=0.25)

Usage:
  # Full sweep (3-6 hours):
  PYTHONPATH=. python tools/to_omni_sweep.py 2>&1 | tee tools/cache/to_omni/sweep.log

  # Smoke test (2 solves: forward + lateral):
  PYTHONPATH=. python tools/to_omni_sweep.py --max-direction 90 --max-speed-idx 1
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.to_solver import compute_metrics, solve_with_warmstart

# Grid definition. Speeds confirmed practical by docs/MAX_SPEED_ANALYSIS.md:
#   0.10 m/s — low gait,    0.17 m/s — mid (signed-off),  0.25 m/s — near max
SPEEDS_MPS   = [0.10, 0.17, 0.25]
N_DIRECTIONS = 12
DIRECTIONS_DEG = [i * 30 for i in range(N_DIRECTIONS)]   # 0, 30, ..., 330

CACHE_DIR = Path(__file__).resolve().parent / "cache" / "to_omni"


def traj_path(direction_deg: int, speed_mps: float) -> Path:
    """Canonical output path for a single (direction, speed) trajectory."""
    speed_mm = round(speed_mps * 1000)
    return CACHE_DIR / f"dir_{direction_deg:03d}_speed_{speed_mm:03d}.npz"


def run_sweep(max_direction: int, max_speed_idx: int) -> list[dict]:
    """Run the sweep, skipping already-cached trajectories. Returns a list of
    per-trajectory result dicts (for the report)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    total = 0
    solved = 0
    failed = 0
    skipped = 0

    for speed_idx, speed in enumerate(SPEEDS_MPS):
        if speed_idx > max_speed_idx:
            continue
        for direction_deg in DIRECTIONS_DEG:
            if direction_deg > max_direction:
                continue

            total += 1
            out_path = traj_path(direction_deg, speed)

            if out_path.exists():
                print(f"\n[sweep] SKIP (cached): dir={direction_deg:3d}° speed={speed:.2f} m/s "
                      f"→ {out_path.name}")
                skipped += 1
                # Load and include in results for report continuity.
                try:
                    d = np.load(out_path)
                    results.append({
                        "direction_deg": direction_deg,
                        "speed_mps":     speed,
                        "success":       bool(d["success"]),
                        "ipopt_status":  str(d["ipopt_status"]),
                        "cached":        True,
                        "path":          str(out_path),
                    })
                except Exception as e:
                    print(f"  [warn] cached file unreadable: {e}")
                continue

            angle_rad = math.radians(direction_deg)
            vx = speed * math.cos(angle_rad)
            vy = speed * math.sin(angle_rad)

            print(f"\n{'='*70}")
            print(f"[sweep] dir={direction_deg:3d}°  speed={speed:.2f} m/s  "
                  f"vx={vx:+.4f}  vy={vy:+.4f}")
            print(f"        out: {out_path.name}")
            print(f"{'='*70}")

            t_wall = time.time()
            try:
                sol = solve_with_warmstart(
                    vx=vx, vy=vy,
                    duration_s=8.0,
                    n_strides=11,
                    knots_per_phase=10,
                    body_height=0.145,
                )
                m = compute_metrics(sol)
                elapsed = time.time() - t_wall

                # Save regardless of success — the debug solution is still
                # informative and won't block other sweep iterations.
                np.savez(out_path,
                         joints=sol["joints"],
                         base=sol["base"],
                         pitch=sol["pitch"],
                         roll=sol["roll"],
                         dt=np.float64(sol["dt"]),
                         vx_cmd=np.float64(vx),
                         vy_cmd=np.float64(vy),
                         duration_s=np.float64(sol["duration_s"]),
                         success=np.bool_(sol["success"]),
                         ipopt_status=np.str_(sol["ipopt_status"]))

                entry = {
                    "direction_deg":       direction_deg,
                    "speed_mps":           speed,
                    "vx_cmd":              vx,
                    "vy_cmd":              vy,
                    "success":             sol["success"],
                    "ipopt_status":        sol["ipopt_status"],
                    "elapsed_s":           elapsed,
                    "cached":              False,
                    "path":                str(out_path),
                    **m,
                }
                results.append(entry)

                status_tag = "OK" if sol["success"] else "FAIL"
                print(f"\n[sweep] {status_tag}  dir={direction_deg:3d}°  "
                      f"speed={speed:.2f}  elapsed={elapsed:.0f}s")
                print(f"        ipopt={sol['ipopt_status']}")
                print(f"        peak|q_dot|={m['peak_joint_vel_radps']:.2f}  "
                      f"peak|ax|={m['peak_abs_body_ax_mps2']:.3f}  "
                      f"pitch={m['max_abs_pitch_deg']:.3f}°  "
                      f"roll={m['max_abs_roll_deg']:.3f}°")
                print(f"        vx act/cmd={m['mean_vx_actual']:.4f}/{vx:.4f}  "
                      f"vy act/cmd={m['mean_vy_actual']:.4f}/{vy:.4f}")

                if sol["success"]:
                    solved += 1
                else:
                    failed += 1

            except Exception as e:
                elapsed = time.time() - t_wall
                print(f"\n[sweep] EXCEPTION at dir={direction_deg}° speed={speed}: {e}")
                failed += 1
                results.append({
                    "direction_deg": direction_deg,
                    "speed_mps":     speed,
                    "vx_cmd":        vx,
                    "vy_cmd":        vy,
                    "success":       False,
                    "ipopt_status":  f"Exception: {str(e)[:100]}",
                    "elapsed_s":     elapsed,
                    "cached":        False,
                    "path":          str(out_path),
                })

    print(f"\n{'='*70}")
    print(f"[sweep] DONE: total={total}  solved={solved}  "
          f"failed={failed}  skipped(cached)={skipped}")
    print(f"{'='*70}")
    return results


def print_summary(results: list[dict]):
    print("\n--- Sweep summary ---")
    header = f"{'Dir':>4}  {'Speed':>6}  {'OK':>4}  {'IPOPT status':<35}  {'|q_dot|':>7}  {'peak|ax|':>8}  {'pitch':>7}  {'wall':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        ok   = "YES" if r["success"] else "NO"
        qdot = f"{r.get('peak_joint_vel_radps', 0):.2f}"
        ax   = f"{r.get('peak_abs_body_ax_mps2', 0):.3f}"
        pit  = f"{r.get('max_abs_pitch_deg', 0):.3f}"
        wall = f"{r.get('elapsed_s', 0):.0f}s"
        tag  = "(cached)" if r.get("cached") else ""
        print(f"{r['direction_deg']:>4}°  {r['speed_mps']:>5.2f}  {ok:>4}  "
              f"{r['ipopt_status']:<35}  {qdot:>7}  {ax:>8}  {pit:>7}  "
              f"{wall:>6} {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-direction", type=int, default=330,
                    help="Skip directions > this (degrees, 0-330). "
                         "Use 90 for quick 2-solve smoke test.")
    ap.add_argument("--max-speed-idx", type=int, default=2,
                    help="Skip speed indices > this (0=0.10, 1=0.17, 2=0.25).")
    args = ap.parse_args()

    print("=" * 70)
    print("Omnidirectional TO sweep")
    print(f"  directions: {[d for d in DIRECTIONS_DEG if d <= args.max_direction]}")
    print(f"  speeds:     {[s for i, s in enumerate(SPEEDS_MPS) if i <= args.max_speed_idx]} m/s")
    print(f"  cache dir:  {CACHE_DIR}")
    print("=" * 70)

    results = run_sweep(args.max_direction, args.max_speed_idx)
    print_summary(results)


if __name__ == "__main__":
    main()
