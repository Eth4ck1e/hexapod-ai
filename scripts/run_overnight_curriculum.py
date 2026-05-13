"""
run_overnight_curriculum.py — temp script for one specific overnight
experiment: tiered scaffold-cushion curriculum.

Runs N chained training segments at each gait_scale tier:

    tier 1: 5 segments at gait_scale=0.9   (heavy scaffold cushion)
    tier 2: 5 segments at gait_scale=0.75
    tier 3: 10 segments at gait_scale=0.50  (main learning phase)
    tier 4: 10 segments at gait_scale=0.25
    tier 5: 10 segments at gait_scale=0.10
    tier 6: 10 segments at gait_scale=0.00  (autonomous)
                                            ─────
    total:  50 segments × 100M steps = 5B steps
            ~5.8 hours wallclock at ~240k it/s

The policy weights persist across tier transitions; only the env's
scaffold cushion shrinks. Each tier's last iter becomes the next
tier's restore source via chain_train.py's auto-find-latest logic.

Usage:
    .venv\\Scripts\\python.exe run_overnight_curriculum.py

Aborts immediately if any tier fails (likely a config issue worth
debugging before continuing). Logs each tier's start/end timestamp to
stdout so you can scrub through the log in the morning.
"""
from __future__ import annotations

import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta

TIERS = [
    # (segments, gait_scale, label)
    (5,  0.9,  "tier 1: heavy cushion"),
    (5,  0.75, "tier 2"),
    (10, 0.5,  "tier 3: main learning phase"),
    (10, 0.25, "tier 4"),
    (10, 0.1,  "tier 5"),
    (10, 0.0,  "tier 6: autonomous"),
]

# These flags get forwarded to chain_train.py untouched. cmd_mask /
# action_space pull from chain_train.BASE_NAME / etc by default — match
# what your current lineage was trained with.
COMMON_FLAGS = []     # e.g. ["--cmd-mask", "stage1", "--action-space", "foot"]


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def run_tier(segments: int, gait_scale: float, label: str) -> int:
    print(f"\n{'='*70}")
    print(f"[{_now()}] STARTING {label}: {segments} segments @ gait_scale={gait_scale}")
    print(f"{'='*70}")

    cmd = [
        sys.executable, "scripts/chain_train.py",
        "--segments", str(segments),
        "--gait-scale", str(gait_scale),
    ] + COMMON_FLAGS

    rc = subprocess.call(cmd)
    print(f"\n[{_now()}] tier rc = {rc}")
    return rc


def main() -> None:
    total_segments = sum(s for s, _, _ in TIERS)
    est_minutes = total_segments * 12     # ~12 min/segment empirical
    eta = datetime.now() + timedelta(minutes=est_minutes)
    print(f"\n[{_now()}] OVERNIGHT CURRICULUM STARTING")
    print(f"  total segments: {total_segments}")
    print(f"  estimated wallclock: ~{est_minutes/60:.1f} hours")
    print(f"  estimated finish: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  tiers ({len(TIERS)}):")
    for s, gs, lbl in TIERS:
        print(f"    {lbl:<35s}  {s:>3d} segments @ gait_scale={gs}")

    t0 = time.time()
    for i, (segments, gait_scale, label) in enumerate(TIERS, start=1):
        rc = run_tier(segments, gait_scale, label)
        if rc != 0:
            print(f"\n[{_now()}] TIER {i} FAILED (rc={rc}). Aborting curriculum.")
            print(f"  Resume manually with: chain_train.py --segments {segments} "
                  f"--gait-scale {gait_scale}")
            sys.exit(rc)

    elapsed_h = (time.time() - t0) / 3600
    print(f"\n{'='*70}")
    print(f"[{_now()}] CURRICULUM COMPLETE in {elapsed_h:.1f} hours")
    print(f"{'='*70}")
    print(f"\nFinal policy is the latest checkpoint matching BASE_NAME's prefix.")
    print(f"Watch with:  python watch_demo_jax.py")


if __name__ == "__main__":
    main()
