#!/usr/bin/env bash
set -euo pipefail
PROJ="C:/Users/Eth4ck1e/OneDrive/Documents/Hexapod AI Project"
PY="/c/Users/Eth4ck1e/OneDrive/Documents/Hexapod AI Project/.venv/Scripts/python.exe"
LOG="$PROJ/.cache/lateral_batch.log"

echo "=== Lateral batch started $(date) ===" | tee "$LOG"

echo "" | tee -a "$LOG"
echo "=== 90deg: pure lateral ===" | tee -a "$LOG"
PYTHONPATH="$PROJ" "$PY" "$PROJ/tools/trajectory_opt_demo.py" \
  --duration 8.0 --target-x 0.0 --target-y 1.333 \
  --n-strides 22 --knots-per-phase 5 \
  --out-traj ".cache/to_trajectory_90deg.npz" \
  --out-plot "docs/papers/to_vs_scaffold_90deg.png" \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== 45deg: forward-left diagonal ===" | tee -a "$LOG"
PYTHONPATH="$PROJ" "$PY" "$PROJ/tools/trajectory_opt_demo.py" \
  --duration 8.0 --target-x 0.943 --target-y 0.943 \
  --n-strides 22 --knots-per-phase 5 \
  --out-traj ".cache/to_trajectory_45deg.npz" \
  --out-plot "docs/papers/to_vs_scaffold_45deg.png" \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== 135deg: backward-left diagonal ===" | tee -a "$LOG"
PYTHONPATH="$PROJ" "$PY" "$PROJ/tools/trajectory_opt_demo.py" \
  --duration 8.0 --target-x -0.943 --target-y 0.943 \
  --n-strides 22 --knots-per-phase 5 \
  --out-traj ".cache/to_trajectory_135deg.npz" \
  --out-plot "docs/papers/to_vs_scaffold_135deg.png" \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Lateral batch done $(date) ===" | tee -a "$LOG"
