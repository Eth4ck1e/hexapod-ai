"""
chain_train.py — chain N PPO training segments forward from a lineage's
latest checkpoint, auto-naming each run.

Picks the most recent `<prefix>*/final/params.pkl` under `checkpoints/`,
restores from it, and runs the next training segment with a derived
run name (`<prefix>_iterN` where N counts existing matches + 1).

If no checkpoint matches `<prefix>*`, runs `pretrain_bc_jax.py` first
to bootstrap the BC starting point, then chains training as iter1.
This makes "start a fresh experiment" a single command:

    edit BASE_NAME → "mjx_stage1_foot_v2"
    python chain_train.py --segments 5 --action-space foot --cmd-mask stage1
    # auto-runs pretrain since no mjx_stage1_foot_v2* exists yet,
    # then 5 chained 100M segments named iter1..iter5

Examples:
  python chain_train.py                                    # use BASE_NAME, 1 segment
  python chain_train.py --segments 5                       # 5 segments
  python chain_train.py --prefix mjx_other_v1              # override BASE_NAME
  python chain_train.py --archive-old                      # move non-matching logs/ dirs
                                                           #   to logs_legacy/ before starting

Logs a `chain_lineage.txt` per prefix listing the resume-from / produced
checkpoints in order, so you can trace the lineage later without
spelunking through directory mtimes.

Run from PowerShell — it shells out to wsl for training itself.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ============================================================================
# Edit these when you want to start a new training session chain. The
# values are also imported by watch_demo_jax.py so its defaults stay
# in sync with what you're training.
# ============================================================================
BASE_NAME    = "mjx_stage1_foot_v1"
CMD_MASK     = "stage1"          # stage1 | stage2 | stage3
ACTION_SPACE = "foot"            # joint | foot
MODEL_PATH   = "models/phantomx_simple_mjx.xml"
# Default render model for watch scripts. Inference still uses MODEL_PATH;
# RENDER_MODEL_PATH is ONLY for the viewer window (mesh visuals).
# Set to MODEL_PATH for zero-overhead viewing.
RENDER_MODEL_PATH = "models/phantomx.xml"
# Set False to skip per-segment eval globally; also suppressible via --no-eval.
EVAL_AFTER_SEGMENT = True

# chain_train.py lives at scripts/chain_train.py after the reorg, so the
# project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR        = PROJECT_ROOT / "logs"
LOG_LEGACY_DIR = PROJECT_ROOT / "logs_legacy"
BC_OUT_DIR     = CHECKPOINT_DIR / "bc_pretrained_jax"

# Persistent JAX compilation cache — saves ~30-60s of JIT recompile per
# launch of train_jax.py / pretrain_bc_jax.py / watch_demo_jax.py. Call
# enable_jax_cache() right after `import jax`, before any JAX operation
# runs. Stale caches are auto-invalidated by JAX when traced code changes.
JAX_CACHE_DIR = str(PROJECT_ROOT / ".cache" / "jax_hexapod")


def enable_jax_cache(cache_dir: str = JAX_CACHE_DIR) -> None:
    """Turn on JAX's persistent compilation cache. Idempotent — safe to
    call multiple times. Must be called BEFORE any JAX op runs (else
    the early ops will compile uncached)."""
    import jax
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)


def find_latest_checkpoint(prefix: str) -> Path | None:
    """Find the most-recently-modified `<prefix>*/final/params.pkl` under
    checkpoints/. Returns None if no match exists."""
    candidates = []
    for run_dir in CHECKPOINT_DIR.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith(prefix):
            continue
        params = run_dir / "final" / "params.pkl"
        if params.exists():
            candidates.append((params.stat().st_mtime, params, run_dir.name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]                                      # most recent


def count_existing_runs(prefix: str) -> int:
    """How many `<prefix>*` runs exist under checkpoints/."""
    return sum(1 for d in CHECKPOINT_DIR.iterdir()
               if d.is_dir() and d.name.startswith(prefix))


def next_iter_name(prefix: str) -> str:
    """Pick a name for the next run: `<prefix>_iter<N>` with the lowest
    N >= 1 that doesn't already exist. Skips the bootstrap dir
    `<prefix>_iter0_bc` (only used as the BC restore source)."""
    existing = {d.name for d in CHECKPOINT_DIR.iterdir() if d.is_dir()}
    n = 1
    while f"{prefix}_iter{n}" in existing:
        n += 1
    return f"{prefix}_iter{n}"


def append_lineage_log(prefix: str, restore_path: Path, new_run: str,
                       steps: int, cmd_mask: str) -> None:
    log = CHECKPOINT_DIR / f"chain_lineage_{prefix}.txt"
    with open(log, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  "
                f"resume={restore_path.parent.parent.name}  "
                f"-> {new_run}  steps={steps}  cmd_mask={cmd_mask}\n")


def archive_old_logs(prefix: str) -> int:
    """Move every logs/<dir> NOT matching `<prefix>*` to logs_legacy/.
    Returns the count moved. Idempotent: re-running on an already-clean
    logs/ does nothing. Doesn't touch checkpoints (they're cheaper to
    keep around for A/B comparisons later)."""
    if not LOG_DIR.is_dir():
        return 0
    LOG_LEGACY_DIR.mkdir(exist_ok=True)
    moved = 0
    for entry in LOG_DIR.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith(prefix):
            continue
        dest = LOG_LEGACY_DIR / entry.name
        if dest.exists():
            # Avoid clobber — append a timestamp.
            dest = LOG_LEGACY_DIR / f"{entry.name}_{int(time.time())}"
        shutil.move(str(entry), str(dest))
        print(f"  archived: logs/{entry.name} -> logs_legacy/{dest.name}")
        moved += 1
    return moved


def run_pretrain(cmd_mask: str, action_space: str) -> int:
    """Invoke pretrain_bc_jax.py via WSL. Returns the exit code."""
    cmd_inner = (
        f"cd '/mnt/c/Users/Eth4ck1e/OneDrive/Documents/Hexapod AI Project' && "
        f"PYTHONPATH=. ~/.venv-mjx/bin/python scripts/pretrain_bc_jax.py "
        f"--cmd-mask {cmd_mask} "
        f"--action-space {action_space}"
    )
    if platform.system() == "Windows":
        cmd = ["wsl", "bash", "-lc", cmd_inner]
    else:
        cmd = ["bash", "-lc", cmd_inner]
    print(f"\n  exec: {cmd_inner}\n")
    return subprocess.call(cmd)


def run_one_segment(restore_path: Path, run_name: str,
                    steps: int, num_envs: int, cmd_mask: str,
                    action_space: str, gait_scale: float,
                    amp: bool = False, amp_priors: str | None = None,
                    amp_segments: int = 1) -> int:
    """Invoke train_jax.py (or train_jax_amp.py) for one segment via WSL.
    Returns the exit code.

    If amp=True, dispatches to scripts/train_jax_amp.py with the priors
    file path. The 'one segment' inside chain_train maps to a full
    train_jax_amp run with `amp_segments` outer iterations (PPO + disc
    update cycles).
    """
    rel_restore = restore_path.relative_to(PROJECT_ROOT).as_posix()

    if amp:
        if amp_priors is None:
            raise ValueError("amp=True requires amp_priors path")
        cmd_inner = (
            f"cd '/mnt/c/Users/Eth4ck1e/OneDrive/Documents/Hexapod AI Project' && "
            f"PYTHONPATH=. ~/.venv-mjx/bin/python scripts/train_jax_amp.py "
            f"--restore {rel_restore} "
            f"--priors {amp_priors} "
            f"--segments {amp_segments} "
            f"--steps-per-segment {steps // max(amp_segments, 1)} "
            f"--num-envs {num_envs} "
            f"--run {run_name} "
            f"--cmd-mask {cmd_mask} "
            f"--action-space {action_space}"
        )
    else:
        cmd_inner = (
            f"cd '/mnt/c/Users/Eth4ck1e/OneDrive/Documents/Hexapod AI Project' && "
            f"PYTHONPATH=. ~/.venv-mjx/bin/python scripts/train_jax.py "
            f"--restore {rel_restore} "
            f"--steps {steps} "
            f"--num-envs {num_envs} "
            f"--run {run_name} "
            f"--cmd-mask {cmd_mask} "
            f"--action-space {action_space} "
            f"--gait-scale {gait_scale}"
        )
    if platform.system() == "Windows":
        cmd = ["wsl", "bash", "-lc", cmd_inner]
    else:
        cmd = ["bash", "-lc", cmd_inner]
    print(f"\n  exec: {cmd_inner}\n")
    return subprocess.call(cmd)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default=BASE_NAME,
                   help=f"lineage prefix (default: BASE_NAME={BASE_NAME!r}). "
                        f"Latest <prefix>*/final/params.pkl is the restore "
                        f"source. If no checkpoint matches, BC pretrain runs "
                        f"first and the chain starts as <prefix>_iter1.")
    p.add_argument("--segments", type=int, default=1,
                   help="number of training segments to chain (default 1)")
    p.add_argument("--steps", type=int, default=200_000_000,
                   help="env steps per segment (default 200M — anchored to "
                        "MuJoCo Playground / Brax humanoid published norms; "
                        "see research/notes/final_report_legged-rl-budgets.md)")
    p.add_argument("--num-envs", type=int, default=8192,
                   help="parallel envs (default 8192 — sweet spot at 265k it/s "
                        "on RTX 5060 Ti per num_envs sweep)")
    p.add_argument("--cmd-mask", type=str, default=CMD_MASK,
                   choices=["stage1", "stage2", "stage3"],
                   help=f"cmd-slot curriculum stage (default: CMD_MASK={CMD_MASK!r})")
    p.add_argument("--action-space", type=str, default=ACTION_SPACE,
                   choices=["joint", "foot"],
                   help=f"action space (default: ACTION_SPACE={ACTION_SPACE!r}). "
                        f"joint = 18-dim joint residual; foot = (6,3) foot "
                        f"residual + IK (smoother coordinated noise)")
    p.add_argument("--archive-old", action="store_true",
                   help="before starting, move every logs/<dir> NOT matching "
                        "<prefix>* to logs_legacy/. Use when starting a fresh "
                        "session chain to keep TensorBoard uncluttered.")
    p.add_argument("--gait-scale", type=float, default=0.0,
                   help="scaffold contribution: 0.0 = pure policy (default), "
                        "0.5 = half scaffold cushion, 1.0 = full scaffold. "
                        "Use to run tiered curriculum: e.g. several segments "
                        "at 0.9, then 0.75, then 0.5, etc., to gradually wean "
                        "the policy off scaffold support.")
    p.add_argument("--amp", action="store_true",
                   help="use AMP-augmented training (train_jax_amp.py) "
                        "instead of plain PPO. Requires --amp-priors.")
    p.add_argument("--amp-priors", type=str, default="checkpoints/amp_priors.npz",
                   help="path to AMP prior dataset (npz from amp/prior_data.py). "
                        "Only used when --amp is set.")
    p.add_argument("--amp-inner-segments", type=int, default=3,
                   help="number of inner PPO+disc-update cycles per chain "
                        "segment when --amp is set. Total env steps per chain "
                        "segment = --steps; each inner segment gets "
                        "(steps / amp-inner-segments) PPO steps before a "
                        "discriminator update.")
    p.add_argument("--no-eval", action="store_true",
                   help="skip post-segment eval (overrides EVAL_AFTER_SEGMENT). "
                        "Useful when running on WSL without Windows venv access "
                        "or when eval latency is undesirable.")
    args = p.parse_args()

    print("=" * 70)
    print(f"chain_train: lineage={args.prefix}  segments={args.segments}  "
          f"steps/segment={args.steps:,d}  cmd_mask={args.cmd_mask}  "
          f"action_space={args.action_space}")
    print("=" * 70)

    if args.archive_old:
        print(f"\narchiving non-matching logs/<dir> to logs_legacy/...")
        n = archive_old_logs(args.prefix)
        print(f"  archived {n} dir(s).")

    # Bootstrap: if no checkpoint matches, run BC pretrain first.
    if find_latest_checkpoint(args.prefix) is None:
        print(f"\nno checkpoint matching checkpoints/{args.prefix}*/final/params.pkl")
        print(f"running BC pretrain to bootstrap the lineage...")
        rc = run_pretrain(args.cmd_mask, args.action_space)
        if rc != 0:
            print(f"\nERROR: pretrain failed with exit code {rc}")
            sys.exit(rc)
        # pretrain wrote checkpoints/bc_pretrained_jax/params.pkl. Copy it
        # under the lineage prefix so find_latest_checkpoint() picks it up.
        bc_src = BC_OUT_DIR / "params.pkl"
        bc_dst_dir = CHECKPOINT_DIR / f"{args.prefix}_iter0_bc" / "final"
        bc_dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(bc_src, bc_dst_dir / "params.pkl")
        print(f"  staged BC params at {bc_dst_dir.relative_to(PROJECT_ROOT)}")

    for seg in range(args.segments):
        latest = find_latest_checkpoint(args.prefix)
        if latest is None:
            print(f"ERROR: no checkpoint found matching "
                  f"checkpoints/{args.prefix}*/final/params.pkl "
                  f"(this should never happen after pretrain step)")
            sys.exit(1)
        next_name = next_iter_name(args.prefix)
        print(f"\n--- segment {seg+1}/{args.segments} ---")
        print(f"  restore from: {latest.parent.parent.name}")
        print(f"  produces:     {next_name}")

        rc = run_one_segment(latest, next_name,
                             args.steps, args.num_envs, args.cmd_mask,
                             args.action_space, args.gait_scale,
                             amp=args.amp,
                             amp_priors=args.amp_priors,
                             amp_segments=args.amp_inner_segments)
        if rc != 0:
            print(f"\nERROR: segment {seg+1} failed with exit code {rc}. "
                  f"Lineage stopped here.")
            sys.exit(rc)

        append_lineage_log(args.prefix, latest, next_name,
                           args.steps, args.cmd_mask)

        # Post-segment eval: CPU-side quality snapshot so reward trends are
        # visible per segment without manually triggering eval_bc_quick.py.
        # Runs in the Windows venv (mujoco native bindings). Failures are
        # non-fatal — the chain always continues.
        if EVAL_AFTER_SEGMENT and not args.no_eval:
            new_params = (CHECKPOINT_DIR / next_name / "final" / "params.pkl")
            if new_params.exists():
                try:
                    from scripts.segment_eval import evaluate_segment
                    # Discriminator is saved by train_jax_amp.py next to
                    # params.pkl (see train_jax_amp:310 — ckpt_dir/final/).
                    disc_path = new_params.parent / "discriminator.pkl"
                    amp_disc = (str(disc_path)
                                if args.amp and disc_path.exists()
                                else None)
                    evaluate_segment(
                        run_name=next_name,
                        segment_id=seg + 1,
                        params_path=str(new_params),
                        action_space=args.action_space,
                        amp_discriminator_path=amp_disc,
                    )
                except Exception as exc:
                    print(f"\n[chain_train] WARNING: post-segment eval failed: {exc}")
            else:
                print(f"\n[chain_train] WARNING: expected checkpoint not found at "
                      f"{new_params.relative_to(PROJECT_ROOT)} — skipping eval.")

        print(f"\n--- segment {seg+1} done ---")

    print(f"\nALL {args.segments} segment(s) done.")
    print(f"Lineage log: checkpoints/chain_lineage_{args.prefix}.txt")


if __name__ == "__main__":
    main()
