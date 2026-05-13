#!/usr/bin/env python
"""scripts/launch.py — interactive launcher for hexapod workflows.

Usage:
    .venv\\Scripts\\python.exe scripts\\launch.py watcher

Subcommands:
    watcher    Pick lineage + iter + mode and launch a watcher.
    train      (placeholder — coming next.)
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import questionary
except ImportError:
    print("questionary missing. Install:", file=sys.stderr)
    print("  .venv\\Scripts\\python.exe -m pip install questionary", file=sys.stderr)
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
LOGS_STDOUT = PROJECT_ROOT / "logs" / "stdout"


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------
def _fmt_age(mtime: float) -> str:
    sec = int(datetime.now().timestamp() - mtime)
    if sec < 60:    return f"{sec}s ago"
    if sec < 3600:  return f"{sec // 60}m ago"
    if sec < 86400: return f"{sec // 3600}h ago"
    return f"{sec // 86400}d ago"


def list_lineages(include_bc: bool = False) -> list[tuple[str, float]]:
    """Return [(name, mtime), ...] sorted newest first."""
    if not CHECKPOINTS.is_dir():
        return []
    out = []
    for d in CHECKPOINTS.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("amp_to_"):
            out.append(d)
        elif include_bc and d.name.startswith("bc_pretrained"):
            out.append(d)
    out.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [(d.name, d.stat().st_mtime) for d in out]


def list_iters(lineage: str) -> list[int]:
    """Return iter numbers (ascending) that have iter*/final/params.pkl."""
    base = CHECKPOINTS / lineage
    if not base.is_dir():
        return []
    out = []
    for d in base.iterdir():
        m = re.match(r"iter(\d+)$", d.name)
        if m and (d / "final" / "params.pkl").exists():
            out.append(int(m.group(1)))
    return sorted(out)


def find_log(lineage: str) -> Path | None:
    """Best-effort find the stdout log for this lineage."""
    if not LOGS_STDOUT.is_dir():
        return None
    candidates: list[Path] = []
    # 1. Exact / contains-lineage match
    candidates += list(LOGS_STDOUT.glob(f"*{lineage}*.log"))
    # 2. Strip amp_to_ prefix and retry
    if lineage.startswith("amp_to_"):
        suffix = lineage[len("amp_to_"):]
        candidates += list(LOGS_STDOUT.glob(f"*{suffix}*.log"))
    candidates = list(set(candidates))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def parse_iter_evals(log_path: Path | None, lineage: str) -> dict[int, float]:
    """Pair [EVAL ] lines with the subsequent 'saving policy .../iter<N>/' line."""
    if log_path is None or not log_path.exists():
        return {}
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return {}
    eval_re = re.compile(r"\[EVAL\s*\][^\n]*reward=([+\-\d\.]+)")
    save_re = re.compile(rf"saving policy[^\n]*?{re.escape(lineage)}[\\/]iter(\d+)")
    events: list[tuple[int, str, float | int]] = []
    for m in eval_re.finditer(text):
        events.append((m.start(), "eval", float(m.group(1))))
    for m in save_re.finditer(text):
        events.append((m.start(), "save", int(m.group(1))))
    events.sort(key=lambda e: e[0])
    out: dict[int, float] = {}
    pending_eval: float | None = None
    for _, kind, val in events:
        if kind == "eval":
            pending_eval = float(val)
        elif kind == "save" and pending_eval is not None:
            out[int(val)] = pending_eval
            pending_eval = None
    return out


# ---------------------------------------------------------------------------
# Subcommand: watcher
# ---------------------------------------------------------------------------
def cmd_watcher() -> None:
    # 1. Lineage
    lineages = list_lineages(include_bc=False)
    if not lineages:
        print(
            "No amp_to_* lineages found under checkpoints/. Train something first.",
            file=sys.stderr,
        )
        sys.exit(1)

    lineage_choices = []
    for i, (name, mtime) in enumerate(lineages):
        label = f"{name}  ({_fmt_age(mtime)})"
        if i == 0:
            label += "  [latest]"
        lineage_choices.append(questionary.Choice(label, value=name))
    lineage_choices.append(questionary.Choice("[enter a name manually]", value="__manual__"))

    lineage = questionary.select(
        "Select lineage:",
        choices=lineage_choices,
        default=lineage_choices[0],
    ).ask()
    if lineage is None:
        return
    if lineage == "__manual__":
        lineage = questionary.text("Lineage name:").ask()
        if not lineage:
            return

    # 2. Iter (default: best-eval, fallback: latest)
    iters = list_iters(lineage)
    if not iters:
        print(
            f"No iter*/final/params.pkl found under checkpoints/{lineage}/.",
            file=sys.stderr,
        )
        sys.exit(1)
    evals = parse_iter_evals(find_log(lineage), lineage)
    best_iter = max(evals, key=evals.get) if evals else None
    final_iter = iters[-1]
    default_iter = best_iter if best_iter is not None else final_iter

    iter_choices = []
    for itr in iters:
        bits = []
        if itr in evals:
            bits.append(f"eval={evals[itr]:+.1f}")
        if itr == best_iter:
            bits.append("BEST")
        if itr == final_iter:
            bits.append("FINAL")
        suffix = f"  ({', '.join(bits)})" if bits else ""
        iter_choices.append(questionary.Choice(f"iter{itr}{suffix}", value=itr))
    default_choice = next(c for c in iter_choices if c.value == default_iter)

    chosen_iter = questionary.select(
        "Select iter:",
        choices=iter_choices,
        default=default_choice,
    ).ask()
    if chosen_iter is None:
        return

    # 3. Mode
    mode = questionary.select(
        "Mode:",
        choices=[
            questionary.Choice("interactive  (keyboard-driven test cycles)", value="interactive"),
            questionary.Choice("controller   (gamepad / Bluetooth)", value="controller"),
            questionary.Choice("demo         (auto-cycle preset phases)", value="demo"),
        ],
        default="interactive",
    ).ask()
    if mode is None:
        return

    demo = None
    if mode == "demo":
        demo = questionary.select(
            "Demo schedule:",
            choices=["showcase", "paper_stance", "paper", "legacy"],
            default="showcase",
        ).ask()
        if demo is None:
            return

    # 4. Assemble command
    params_path = CHECKPOINTS / lineage / f"iter{chosen_iter}" / "final" / "params.pkl"
    python_exe = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

    if mode == "controller":
        script = PROJECT_ROOT / "scripts" / "watch_controller.py"
        extra: list[str] = []
    else:
        script = PROJECT_ROOT / "scripts" / "watch_demo_jax.py"
        extra = ["--demo", demo] if mode == "demo" else ["--interactive"]

    cmd = [
        str(python_exe), str(script),
        "--params", str(params_path),
        "--action-space", "foot",
        *extra,
    ]

    # 5. Confirm + run
    print()
    print("=" * 70)
    print("Launching:")
    print("  " + " ".join(_quote(c) for c in cmd))
    print("=" * 70)
    print()

    action = questionary.select(
        "Confirm:",
        choices=[
            questionary.Choice("Launch now", value="launch"),
            questionary.Choice("Print and quit (copy/paste manually)", value="print"),
            questionary.Choice("Cancel", value="cancel"),
        ],
        default="launch",
    ).ask()
    if action != "launch":
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))


def _quote(s: str) -> str:
    return f'"{s}"' if " " in s else s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive launcher for hexapod workflows."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("watcher", help="Launch a watcher (interactive / controller / demo)")
    sub.add_parser("train",   help="Launch a training run (TODO)")
    args = parser.parse_args()

    if args.cmd == "watcher":
        cmd_watcher()
    elif args.cmd == "train":
        print("Training launcher not yet implemented.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
