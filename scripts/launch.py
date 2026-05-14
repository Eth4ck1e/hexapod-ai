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


def _pad() -> None:
    """Print a few blank lines before a prompt so questionary's choice list
    has vertical room above the cursor (otherwise the top of the prompt
    gets clipped in short terminals)."""
    print("\n" * 3)


def _to_wsl_path(win_path: Path) -> str:
    """Convert C:\\foo\\bar -> /mnt/c/foo/bar."""
    s = str(win_path).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:]}"
    return s


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
# Shared picker: lineage + iter
# ---------------------------------------------------------------------------
def pick_lineage_and_iter() -> tuple[str, int] | None:
    """Interactive lineage + iter picker.

    Used by watcher, record, and eval subcommands. Returns
    (lineage_name, iter_num), or None if the user cancels at any prompt.
    Lineage list is amp_to_* dirs sorted newest-first; iter list is
    annotated with eval reward (parsed from stdout log) plus BEST/FINAL tags.
    """
    lineages = list_lineages(include_bc=False)
    if not lineages:
        print(
            "No amp_to_* lineages found under checkpoints/. Train something first.",
            file=sys.stderr,
        )
        return None

    lineage_choices = []
    for i, (name, mtime) in enumerate(lineages):
        label = f"{name}  ({_fmt_age(mtime)})"
        if i == 0:
            label += "  [latest]"
        lineage_choices.append(questionary.Choice(label, value=name))
    lineage_choices.append(questionary.Choice("[enter a name manually]", value="__manual__"))

    _pad()
    lineage = questionary.select(
        "Select lineage:",
        choices=lineage_choices,
        default=lineage_choices[0],
    ).ask()
    if lineage is None:
        return None
    if lineage == "__manual__":
        _pad()
        lineage = questionary.text("Lineage name:").ask()
        if not lineage:
            return None

    iters = list_iters(lineage)
    if not iters:
        print(
            f"No iter*/final/params.pkl found under checkpoints/{lineage}/.",
            file=sys.stderr,
        )
        return None
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

    _pad()
    chosen_iter = questionary.select(
        "Select iter:",
        choices=iter_choices,
        default=default_choice,
    ).ask()
    if chosen_iter is None:
        return None

    return lineage, chosen_iter


def _params_path_for(lineage: str, iter_n: int) -> Path:
    return CHECKPOINTS / lineage / f"iter{iter_n}" / "final" / "params.pkl"


# ---------------------------------------------------------------------------
# Subcommand: watcher
# ---------------------------------------------------------------------------
def cmd_watcher() -> None:
    picked = pick_lineage_and_iter()
    if picked is None:
        return
    lineage, chosen_iter = picked

    # Mode
    _pad()
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
        _pad()
        demo = questionary.select(
            "Demo schedule:",
            choices=["showcase", "paper_stance", "paper", "legacy"],
            default="showcase",
        ).ask()
        if demo is None:
            return

    # Assemble command
    params_path = _params_path_for(lineage, chosen_iter)
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

    _pad()
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


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------
def cmd_train() -> None:
    # 1. Restore source: BC pretrain OR resume from amp iter
    bc_dirs = sorted(
        (d for d in CHECKPOINTS.iterdir()
         if d.is_dir() and d.name.startswith("bc_pretrained_jax_")
         and (d / "params.pkl").exists()),
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    amp_lineages = list_lineages(include_bc=False)

    if not bc_dirs and not amp_lineages:
        print("No BC checkpoints or AMP lineages to restore from. "
              "Run pretrain_bc_jax.py first.", file=sys.stderr)
        sys.exit(1)

    restore_choices = []
    for i, d in enumerate(bc_dirs):
        label = f"BC: {d.name}  ({_fmt_age(d.stat().st_mtime)})"
        if i == 0:
            label += "  [latest]"
        restore_choices.append(questionary.Choice(label, value=("bc", d.name, None)))
    for name, mtime in amp_lineages[:5]:
        iters = list_iters(name)
        if iters:
            latest = iters[-1]
            label = f"Resume: {name}/iter{latest}  ({_fmt_age(mtime)})"
            restore_choices.append(questionary.Choice(label, value=("amp", name, latest)))
    restore_choices.append(questionary.Choice("[type a custom params.pkl path]", value=("custom", None, None)))

    _pad()
    restore = questionary.select(
        "Restore from:",
        choices=restore_choices,
        default=restore_choices[0],
    ).ask()
    if restore is None:
        return
    kind, name, iter_n = restore
    if kind == "bc":
        restore_path = f"checkpoints/{name}/params.pkl"
    elif kind == "amp":
        restore_path = f"checkpoints/{name}/iter{iter_n}/final/params.pkl"
    else:
        _pad()
        restore_path = questionary.text("Path to params.pkl (relative to project root):").ask()
        if not restore_path:
            return

    # 2. Priors
    prior_files = sorted(
        CHECKPOINTS.glob("amp_priors_*.npz"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not prior_files:
        print("No amp_priors_*.npz under checkpoints/. Generate via amp/prior_data.py.",
              file=sys.stderr)
        sys.exit(1)

    prior_choices = []
    for i, p in enumerate(prior_files):
        label = f"{p.name}  ({_fmt_age(p.stat().st_mtime)})"
        if i == 0:
            label += "  [latest]"
        prior_choices.append(questionary.Choice(label, value=p.name))

    _pad()
    priors_name = questionary.select(
        "Priors:",
        choices=prior_choices,
        default=prior_choices[0],
    ).ask()
    if priors_name is None:
        return
    priors_path = f"checkpoints/{priors_name}"

    # 3. Style weight
    _pad()
    style_weight = questionary.text(
        "Style weight (typical: 0.5 - 1.0):",
        default="1.0",
    ).ask()
    if not style_weight:
        return

    # 4. Length
    length_choices = [
        questionary.Choice("Quick test   (5 x 50M  =  250M, ~45 min)",   value=(5,  50_000_000)),
        questionary.Choice("Short tuning (10 x 50M =  500M, ~1.5 hr)",   value=(10, 50_000_000)),
        questionary.Choice("Full run     (20 x 50M = 1B,    ~3 hr)",     value=(20, 50_000_000)),
        questionary.Choice("[custom]", value="custom"),
    ]
    _pad()
    length = questionary.select(
        "Training length:",
        choices=length_choices,
        default=length_choices[1],
    ).ask()
    if length is None:
        return
    if length == "custom":
        _pad()
        segs_s = questionary.text("Segments:", default="10").ask()
        if not segs_s:
            return
        _pad()
        steps_s = questionary.text("Steps per segment:", default="50000000").ask()
        if not steps_s:
            return
        segments, steps_per_segment = int(segs_s), int(steps_s)
    else:
        segments, steps_per_segment = length

    # 5. Run name
    latest_amp = amp_lineages[0][0] if amp_lineages else "amp_to_v1"
    _pad()
    run_name = questionary.text(
        "Run name (will live at checkpoints/<name>/):",
        default=latest_amp,
    ).ask()
    if not run_name:
        return
    if (CHECKPOINTS / run_name).exists():
        _pad()
        overwrite = questionary.confirm(
            f"checkpoints/{run_name}/ already exists. Continue (may overwrite)?",
            default=False,
        ).ask()
        if not overwrite:
            return

    # 6. Build WSL command
    wsl_project = _to_wsl_path(PROJECT_ROOT)
    log_path = f"logs/stdout/{run_name}.log"
    inner = (
        f"PYTHONPATH=. ~/.venv-mjx/bin/python -u scripts/train_jax_amp.py "
        f"--restore {restore_path} "
        f"--priors {priors_path} "
        f"--cmd-mask paper_stance --action-space foot "
        f"--style-weight {style_weight} "
        f"--segments {segments} --steps-per-segment {steps_per_segment} "
        f"--partition-disc "
        f"--run {run_name} "
        f"2>&1 | tee {log_path}"
    )
    wsl_full = f"cd '{wsl_project}' && {inner}"

    print()
    print("=" * 70)
    print("Launching:")
    print(f"  wsl bash -lc \"{wsl_full}\"")
    print("=" * 70)
    print()

    _pad()
    action = questionary.select(
        "Confirm:",
        choices=[
            questionary.Choice("Launch (foreground - live stdout; ctrl-c aborts)", value="fg"),
            questionary.Choice("Print and quit (copy/paste manually)",            value="print"),
            questionary.Choice("Cancel",                                          value="cancel"),
        ],
        default="fg",
    ).ask()
    if action != "fg":
        return

    subprocess.run(["wsl", "bash", "-lc", wsl_full], cwd=str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Subcommand: record
# ---------------------------------------------------------------------------
def cmd_record() -> None:
    picked = pick_lineage_and_iter()
    if picked is None:
        return
    lineage, chosen_iter = picked

    _pad()
    demo = questionary.select(
        "Demo schedule:",
        choices=["showcase", "paper_stance", "paper", "legacy"],
        default="showcase",
    ).ask()
    if demo is None:
        return

    _pad()
    orbit_period_s = questionary.text(
        "Orbit period in seconds (0 = static camera):",
        default="6.0",
    ).ask()
    if orbit_period_s is None:
        return
    try:
        orbit_period = float(orbit_period_s)
    except ValueError:
        orbit_period = 0.0

    default_out = f"media/recordings/{lineage}_iter{chosen_iter}_{demo}.mp4"
    _pad()
    out_path = questionary.text(
        "Output MP4 path:",
        default=default_out,
    ).ask()
    if not out_path:
        return

    params_path = _params_path_for(lineage, chosen_iter)
    python_exe = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    script = PROJECT_ROOT / "scripts" / "record_policy.py"

    cmd = [
        str(python_exe), str(script),
        "--params", str(params_path),
        "--action-space", "foot",
        "--demo", demo,
        "--out", out_path,
    ]
    if orbit_period > 0:
        cmd.extend(["--orbit-period", str(orbit_period)])

    print()
    print("=" * 70)
    print("Recording:")
    print("  " + " ".join(_quote(c) for c in cmd))
    print("=" * 70)
    print()

    _pad()
    action = questionary.select(
        "Confirm:",
        choices=[
            questionary.Choice("Record now", value="go"),
            questionary.Choice("Print and quit (copy/paste manually)", value="print"),
            questionary.Choice("Cancel", value="cancel"),
        ],
        default="go",
    ).ask()
    if action != "go":
        return

    out_parent = Path(out_path).parent
    if not out_parent.is_absolute():
        out_parent = PROJECT_ROOT / out_parent
    out_parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Subcommand: eval
# ---------------------------------------------------------------------------
def cmd_eval() -> None:
    picked = pick_lineage_and_iter()
    if picked is None:
        return
    lineage, chosen_iter = picked

    _pad()
    episodes_s = questionary.text(
        "Episodes:",
        default="32",
    ).ask()
    if not episodes_s:
        return
    try:
        episodes = int(episodes_s)
    except ValueError:
        print(f"Invalid episode count: {episodes_s!r}", file=sys.stderr)
        return

    params_path = _params_path_for(lineage, chosen_iter)
    python_exe = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    script = PROJECT_ROOT / "scripts" / "eval_bc_quick.py"

    cmd = [
        str(python_exe), str(script),
        "--params", str(params_path),
        "--action-space", "foot",
        "--episodes", str(episodes),
    ]

    print()
    print("=" * 70)
    print("Evaluating:")
    print("  " + " ".join(_quote(c) for c in cmd))
    print("=" * 70)
    print()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Subcommand: bc (BC pretraining)
# ---------------------------------------------------------------------------
def cmd_bc() -> None:
    bc_dirs = sorted(
        (d for d in CHECKPOINTS.iterdir()
         if d.is_dir() and d.name.startswith("bc_pretrained_jax_")),
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    suggested = bc_dirs[0].name if bc_dirs else "bc_pretrained_jax_v1"

    _pad()
    out_name = questionary.text(
        "Output dir name (checkpoints/<name>/):",
        default=suggested,
    ).ask()
    if not out_name:
        return
    if (CHECKPOINTS / out_name).exists():
        _pad()
        overwrite = questionary.confirm(
            f"checkpoints/{out_name}/ already exists. Continue (will overwrite)?",
            default=False,
        ).ask()
        if not overwrite:
            return

    wsl_project = _to_wsl_path(PROJECT_ROOT)
    log_path = f"logs/stdout/{out_name}.log"
    inner = (
        f"PYTHONPATH=. ~/.venv-mjx/bin/python -u scripts/pretrain_bc_jax.py "
        f"--cmd-mask paper_stance --action-space foot "
        f"--out-dir checkpoints/{out_name} "
        f"2>&1 | tee {log_path}"
    )
    wsl_full = f"cd '{wsl_project}' && {inner}"

    print()
    print("=" * 70)
    print("BC pretrain:")
    print(f"  wsl bash -lc \"{wsl_full}\"")
    print("=" * 70)
    print()

    _pad()
    action = questionary.select(
        "Confirm:",
        choices=[
            questionary.Choice("Launch (foreground, ~10-15 min)", value="go"),
            questionary.Choice("Print and quit (copy/paste manually)", value="print"),
            questionary.Choice("Cancel", value="cancel"),
        ],
        default="go",
    ).ask()
    if action != "go":
        return

    subprocess.run(["wsl", "bash", "-lc", wsl_full], cwd=str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Subcommand: priors (regenerate AMP prior dataset)
# ---------------------------------------------------------------------------
def cmd_priors() -> None:
    existing = sorted(
        CHECKPOINTS.glob("amp_priors_*.npz"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    suggested = existing[0].name if existing else "amp_priors_v1.npz"

    _pad()
    out_name = questionary.text(
        "Output npz filename (under checkpoints/):",
        default=suggested,
    ).ask()
    if not out_name:
        return
    if not out_name.endswith(".npz"):
        out_name += ".npz"
    if (CHECKPOINTS / out_name).exists():
        _pad()
        overwrite = questionary.confirm(
            f"checkpoints/{out_name} already exists. Overwrite?",
            default=False,
        ).ask()
        if not overwrite:
            return

    _pad()
    n_steps_s = questionary.text(
        "Steps per env (~n_steps × 4096 envs = total transitions; 1500 -> ~6M):",
        default="1500",
    ).ask()
    if not n_steps_s:
        return
    try:
        n_steps = int(n_steps_s)
    except ValueError:
        print(f"Invalid step count: {n_steps_s!r}", file=sys.stderr)
        return

    wsl_project = _to_wsl_path(PROJECT_ROOT)
    log_path = f"logs/stdout/priors_{Path(out_name).stem}.log"
    inner = (
        f"PYTHONPATH=. ~/.venv-mjx/bin/python -u amp/prior_data.py "
        f"--cmd-mask paper_stance "
        f"--n-steps {n_steps} "
        f"--out checkpoints/{out_name} "
        f"2>&1 | tee {log_path}"
    )
    wsl_full = f"cd '{wsl_project}' && {inner}"

    print()
    print("=" * 70)
    print("Prior generation:")
    print(f"  wsl bash -lc \"{wsl_full}\"")
    print("=" * 70)
    print()

    _pad()
    action = questionary.select(
        "Confirm:",
        choices=[
            questionary.Choice("Launch (foreground, ~5-15 min)", value="go"),
            questionary.Choice("Print and quit (copy/paste manually)", value="print"),
            questionary.Choice("Cancel", value="cancel"),
        ],
        default="go",
    ).ask()
    if action != "go":
        return

    subprocess.run(["wsl", "bash", "-lc", wsl_full], cwd=str(PROJECT_ROOT))


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
    sub.add_parser("train",   help="Launch a training run (BC restore / AMP-PPO chain)")
    sub.add_parser("record",  help="Record an MP4 of a policy through a demo schedule")
    sub.add_parser("eval",    help="Run quantitative eval (reward / tracking / fall rate)")
    sub.add_parser("bc",      help="Run BC pretraining from the scaffold")
    sub.add_parser("priors",  help="Regenerate AMP prior dataset from the scaffold")
    args = parser.parse_args()

    if args.cmd == "watcher":
        cmd_watcher()
    elif args.cmd == "train":
        cmd_train()
    elif args.cmd == "record":
        cmd_record()
    elif args.cmd == "eval":
        cmd_eval()
    elif args.cmd == "bc":
        cmd_bc()
    elif args.cmd == "priors":
        cmd_priors()


if __name__ == "__main__":
    main()
