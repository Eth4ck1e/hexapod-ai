"""tools/bench_hsl.py — benchmark MUMPS vs HSL MA27 on a representative TO solve.

Run from project root:
    PYTHONPATH=. .venv/Scripts/python.exe tools/bench_hsl.py
"""
import sys
import time
from pathlib import Path

# MUST come before importing casadi.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hsl_bootstrap import HSL_AVAILABLE

print(f"HSL bootstrap: available={HSL_AVAILABLE}")

import to_solver as ts


def bench(linear_solver: str, n_strides: int, knots_per_phase: int):
    duration = 0.7 * n_strides   # rough scaffold-period match
    t0 = time.perf_counter()
    sol = ts.build_and_solve_to(
        vx=0.17, vy=0.0,
        duration_s=duration,
        n_strides=n_strides,
        knots_per_phase=knots_per_phase,
        linear_solver=linear_solver,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, sol


def main():
    print("=" * 60)
    print("HSL bench — MUMPS vs MA27 on a representative TO problem")
    print("=" * 60)

    # Two problem sizes:
    #   small : warm-start coarse solve  (4 strides × 8 knots = ~64 knots)
    #   large : production prior solve   (11 strides × 10 knots = 220 knots)
    for size_name, n_strides, knots in [("small", 4, 8), ("large", 11, 10)]:
        print(f"\n--- {size_name}: {n_strides} strides × {knots} knots ---")
        for solver in ["mumps", "ma27"]:
            print(f"\n  [{solver}]")
            try:
                t, sol = bench(solver, n_strides, knots)
                status = sol.get("ipopt_status", "?")
                print(f"    elapsed={t:.1f}s  status={status}")
            except Exception as e:
                print(f"    FAIL: {str(e)[:200]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
