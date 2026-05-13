"""
segment_eval.py — post-segment eval hook for chain_train.py.

Spawns eval_bc_quick.py as a Windows-venv subprocess after each training
segment, parses its stdout summary, appends a CSV row to
logs/<run>/segment_eval.csv, and appends a one-liner to the chain lineage
file. Returns a dict of parsed metrics; never raises (eval failures log a
warning so the chain keeps going).
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Project root is one level up from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Windows venv hosts eval_bc_quick.py's deps (mujoco, SB3-adjacent imports).
# The WSL venv doesn't have mujoco.viewer's native bindings on Windows.
_WINDOWS_PYTHON = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")

# Regex patterns for eval_bc_quick.py's summary block lines.
# Each pattern captures one floating-point value (possibly with sign).
_RE_MEAN_REWARD  = re.compile(r"mean episode reward\s*:\s*([+-]?\d+\.?\d*)")
_RE_STD_REWARD   = re.compile(r"mean episode reward\s*:.*\+/-\s*(\d+\.?\d*)")
_RE_MEAN_LENGTH  = re.compile(r"mean episode length\s*:\s*(\d+\.?\d*)")
_RE_FALL_RATE    = re.compile(r"fell rate\s*:\s*\d+/\d+\s*\(\s*(\d+\.?\d*)%\)")
_RE_TRACKING     = re.compile(r"mean per-step tracking reward\s*:\s*([+-]?\d+\.?\d*)")
_RE_AMP_SCORE    = re.compile(r"mean per-step AMP discriminator score\s*:\s*([+-]?\d+\.?\d*)")

_CSV_HEADER = [
    "segment_id", "mean_reward", "std_reward", "mean_episode_length",
    "fall_rate", "mean_per_step_tracking", "mean_per_step_amp_score",
    "timestamp",
]


def _parse_summary(stdout: str) -> dict:
    """Extract scalar metrics from eval_bc_quick.py's printed summary block.

    Defensive: missing lines (e.g., AMP score when discriminator not
    supplied) fall back to None rather than raising."""
    def _first(pattern: re.Pattern, text: str):
        m = pattern.search(text)
        return float(m.group(1)) if m else None

    return {
        "mean_reward":             _first(_RE_MEAN_REWARD,  stdout),
        "std_reward":              _first(_RE_STD_REWARD,   stdout),
        "mean_episode_length":     _first(_RE_MEAN_LENGTH,  stdout),
        # fall_rate stored as 0-100 float (percentage)
        "fall_rate":               _first(_RE_FALL_RATE,    stdout),
        "mean_per_step_tracking":  _first(_RE_TRACKING,     stdout),
        "mean_per_step_amp_score": _first(_RE_AMP_SCORE,    stdout),
    }


def _append_csv(csv_path: Path, segment_id: int, metrics: dict,
                timestamp: str) -> None:
    """Append one row to segment_eval.csv, writing the header on first use."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "segment_id":              segment_id,
            "mean_reward":             metrics.get("mean_reward"),
            "std_reward":              metrics.get("std_reward"),
            "mean_episode_length":     metrics.get("mean_episode_length"),
            "fall_rate":               metrics.get("fall_rate"),
            "mean_per_step_tracking":  metrics.get("mean_per_step_tracking"),
            "mean_per_step_amp_score": metrics.get("mean_per_step_amp_score"),
            "timestamp":               timestamp,
        })


def _append_lineage_line(lineage_path: Path, segment_id: int,
                         metrics: dict, timestamp: str) -> None:
    """Append a compact markdown bullet to the chain lineage file.

    Format matches the existing lineage file style (plain text, one
    entry per line). AMP score is omitted when None."""
    r   = metrics.get("mean_reward")
    s   = metrics.get("std_reward")
    fr  = metrics.get("fall_rate")
    tr  = metrics.get("mean_per_step_tracking")
    amp = metrics.get("mean_per_step_amp_score")

    reward_str = f"{r:+.2f}+-{s:.2f}" if (r is not None and s is not None) else "N/A"
    fall_str   = f"{fr:.0f}%" if fr is not None else "N/A"
    track_str  = f"{tr:.3f}"  if tr is not None else "N/A"
    amp_str    = f"  amp={amp:+.3f}" if amp is not None else ""

    line = (f"- seg {segment_id}: reward={reward_str}  "
            f"fall={fall_str}  track={track_str}{amp_str}  ts={timestamp}\n")

    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lineage_path, "a") as fh:
        fh.write(line)


def evaluate_segment(
    run_name: str,
    segment_id: int,
    params_path: str,
    *,
    action_space: str,
    stage: int = 1,
    episodes: int = 5,
    max_steps: int = 1000,
    amp_discriminator_path: str | None = None,
    log_dir: str = "logs",
) -> dict:
    """Spawn eval_bc_quick.py against a just-saved segment checkpoint.

    Uses the Windows venv (.venv\\Scripts\\python.exe) because eval_bc_quick.py
    pulls in MuJoCo native bindings that aren't available in the WSL JAX venv.

    CSV rows are appended to logs/<run_name>/segment_eval.csv.
    A one-line markdown bullet is appended to the chain lineage file at
    checkpoints/chain_lineage_<base>.txt, where <base> is the run_name
    stripped of its trailing _iterN suffix.

    Returns a dict of parsed metrics on success, or {'error': <msg>} if
    the subprocess exits non-zero (so the caller can log a warning without
    crashing the chain).
    """
    params_path = str(params_path)

    # Derive the lineage prefix by stripping _iterN or _iter0_bc suffix.
    # e.g. "mjx_stage1_foot_v1_iter12" -> "mjx_stage1_foot_v1"
    base_name = re.sub(r"_iter\d+(_bc)?$", "", run_name)
    lineage_path = CHECKPOINT_DIR / f"chain_lineage_{base_name}.txt"

    log_dir_path = PROJECT_ROOT / log_dir / run_name
    csv_path     = log_dir_path / "segment_eval.csv"

    cmd = [
        _WINDOWS_PYTHON,
        str(PROJECT_ROOT / "scripts" / "eval_bc_quick.py"),
        "--params",       params_path,
        "--episodes",     str(episodes),
        "--max-steps",    str(max_steps),
        "--stage",        str(stage),
        "--action-space", action_space,
        "--gait-scale",   "0.0",
    ]
    if amp_discriminator_path is not None:
        cmd += ["--amp-discriminator", str(amp_discriminator_path)]

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}

    print(f"\n[segment_eval] evaluating seg {segment_id} of {run_name} ...")
    print(f"  params: {params_path}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        msg = f"subprocess launch failed: {exc}"
        print(f"[segment_eval] WARNING: {msg}", file=sys.stderr)
        return {"error": msg}

    # Always echo eval output so it appears in chain_train's console log.
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        msg = f"eval_bc_quick.py exited {result.returncode}"
        print(f"[segment_eval] WARNING: {msg}", file=sys.stderr)
        if result.stderr:
            print(result.stderr[-2000:], file=sys.stderr)   # tail stderr to avoid spam
        return {"error": msg, "stderr": result.stderr[-500:]}

    metrics = _parse_summary(result.stdout)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metrics["timestamp"] = timestamp
    metrics["segment_id"] = segment_id

    _append_csv(csv_path, segment_id, metrics, timestamp)
    _append_lineage_line(lineage_path, segment_id, metrics, timestamp)

    r  = metrics.get("mean_reward")
    tr = metrics.get("mean_per_step_tracking")
    print(f"[segment_eval] seg {segment_id}: reward={r:+.2f}  "
          f"track={tr:.3f}  -> {csv_path.relative_to(PROJECT_ROOT)}")

    return metrics
