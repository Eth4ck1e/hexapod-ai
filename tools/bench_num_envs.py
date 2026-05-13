"""
bench_num_envs.py — sweep over num_envs to find optimal PPO throughput.

For each num_envs in the candidate list, this script launches a fresh
WSL subprocess running tools/_bench_num_envs_worker.py, which executes a
short Brax PPO segment and emits a single `BENCH_RESULT: {...}` JSON
line on stdout. We parse that, accumulate results, and print a
comparison table at the end.

Why subprocess isolation:
  * Each JAX/MJX run holds its GPU memory until process exit; spawning
    fresh processes guarantees clean VRAM between runs.
  * If a config OOMs, the worker exits non-zero and we move on without
    bringing down the controller (or the whole desktop session).

Smart-timer steady-state SPS is implemented in the worker: it discards
the first N progress callbacks (JIT compile / warmup) and computes
SPS over a window of the next M callbacks, then exits. The full
num_timesteps budget is never used — it's just a safety upper bound.

Usage:
    python tools/bench_num_envs.py
    python tools/bench_num_envs.py --num-envs 2048,4096,8192
    python tools/bench_num_envs.py --num-envs 16384 --timeout 600

Run from anywhere on Windows (the script shells out to WSL itself).
JSON results are saved to tools/bench_num_envs_results.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "tools" / "bench_num_envs_results.json"

DEFAULT_NUM_ENVS = [2048, 4096, 8192, 16384]


def make_batch_minibatches(num_envs: int) -> tuple[int, int]:
    """Pick (batch_size, num_minibatches) such that
       batch_size * num_minibatches % num_envs == 0
    using the project's training default of num_minibatches=8 and
    batch_size = num_envs // 8 (matches Brax PPO best-practice of one
    minibatch per parallel env after vmap)."""
    num_minibatches = 8
    if num_envs % num_minibatches != 0:
        # Fallback: try 16, 4, 2, 1 minibatches.
        for nm in (16, 4, 2, 1):
            if num_envs % nm == 0:
                num_minibatches = nm
                break
    batch_size = num_envs // num_minibatches
    return batch_size, num_minibatches


def gpu_status_line() -> str:
    """One-line VRAM snapshot via nvidia-smi (in WSL). Empty string on
    failure — used only for diagnostics, never required."""
    cmd = ("nvidia-smi --query-gpu=memory.used,memory.total "
           "--format=csv,noheader,nounits")
    if platform.system() == "Windows":
        full = ["wsl", "bash", "-lc", cmd]
    else:
        full = ["bash", "-lc", cmd]
    try:
        out = subprocess.check_output(full, text=True, timeout=10).strip()
        # e.g. "1643, 16311"
        used, total = (int(x.strip()) for x in out.split(","))
        return f"VRAM {used} / {total} MiB ({100*used/total:.0f}%)"
    except Exception:
        return ""


def run_one(num_envs: int, batch_size: int, num_minibatches: int,
            timeout_s: int, training_metrics_steps: int,
            warmup_callbacks: int, measure_callbacks: int,
            num_timesteps: int) -> dict:
    """Spawn a worker subprocess for one config; return the parsed
    BENCH_RESULT dict (with extra controller-side fields added)."""
    cmd_inner = (
        "cd '/mnt/c/Users/Eth4ck1e/OneDrive/Documents/Hexapod AI Project' && "
        "PYTHONPATH=. ~/.venv-mjx/bin/python tools/_bench_num_envs_worker.py "
        f"--num-envs {num_envs} "
        f"--batch-size {batch_size} "
        f"--num-minibatches {num_minibatches} "
        f"--training-metrics-steps {training_metrics_steps} "
        f"--warmup-callbacks {warmup_callbacks} "
        f"--measure-callbacks {measure_callbacks} "
        f"--num-timesteps {num_timesteps}"
    )
    if platform.system() == "Windows":
        full = ["wsl", "bash", "-lc", cmd_inner]
    else:
        full = ["bash", "-lc", cmd_inner]

    print(f"\n>>> RUN num_envs={num_envs}  batch={batch_size}  "
          f"minibatches={num_minibatches}  "
          f"product={batch_size*num_minibatches}")
    pre = gpu_status_line()
    if pre:
        print(f"    pre-run {pre}")
    print(f"    exec: {cmd_inner}")
    print(f"    timeout: {timeout_s}s")

    t0 = time.perf_counter()
    parsed: dict | None = None
    last_result_line: str | None = None
    timed_out = False
    rc = None

    try:
        # We stream stdout line-by-line so we can mirror the worker's
        # progress to our own stdout AND scan for BENCH_RESULT.
        proc = subprocess.Popen(
            full,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        deadline = t0 + timeout_s
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                # Mirror so the user can watch progress live.
                print(f"    [worker] {line}")
                if line.startswith("BENCH_RESULT:"):
                    last_result_line = line[len("BENCH_RESULT:"):].strip()
                if time.perf_counter() > deadline:
                    print(f"    !! timeout exceeded ({timeout_s}s) — killing worker")
                    timed_out = True
                    proc.kill()
                    break
        finally:
            try:
                rc = proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait(timeout=10)
    except Exception as e:
        print(f"    !! controller-side exception: {type(e).__name__}: {e}")
        rc = -1

    elapsed = time.perf_counter() - t0
    post = gpu_status_line()
    if post:
        print(f"    post-run {post}")

    if last_result_line is not None:
        try:
            parsed = json.loads(last_result_line)
        except json.JSONDecodeError as e:
            print(f"    !! could not parse BENCH_RESULT JSON: {e}")
            parsed = None

    if parsed is None:
        # Synthesize a failure record — we never got a clean BENCH_RESULT.
        likely_oom = rc is not None and rc != 0
        parsed = {
            "ok": False,
            "num_envs": num_envs,
            "batch_size": batch_size,
            "num_minibatches": num_minibatches,
            "error_class": "Timeout" if timed_out else "NoResult",
            "error_msg": ("worker timed out before producing a result"
                          if timed_out else
                          f"worker exited rc={rc} without BENCH_RESULT"),
            "likely_oom": bool(likely_oom and not timed_out),
        }

    parsed["timed_out"] = timed_out
    parsed["controller_wall_s"] = float(elapsed)
    parsed["worker_exit_code"] = rc
    return parsed


def print_table(results: list[dict]) -> None:
    """Print the comparison table the user wanted."""
    print("\n" + "=" * 80)
    print("NUM_ENVS SWEEP RESULTS")
    print("=" * 80)
    hdr = (
        f"{'num_envs':>10}  "
        f"{'batch':>6}  "
        f"{'mini':>5}  "
        f"{'steady it/s':>12}  "
        f"{'wnd_steps':>11}  "
        f"{'wnd_wall_s':>11}  "
        f"{'wall_s':>8}  "
        f"{'status':<10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ne = r.get("num_envs", "?")
        bs = r.get("batch_size", "?")
        nm = r.get("num_minibatches", "?")
        if r.get("ok"):
            sps = r.get("steady_sps", float("nan"))
            sps_str = f"{sps:>12,.0f}" if sps == sps else f"{'nan':>12}"
            ws = r.get("window_steps", 0)
            ww = r.get("window_wall_s", 0.0)
            tw = r.get("controller_wall_s", 0.0)
            status = "INCOMPLETE" if r.get("incomplete") else "ok"
            print(f"{ne:>10}  {bs:>6}  {nm:>5}  {sps_str}  "
                  f"{ws:>11,d}  {ww:>11,.1f}  {tw:>8,.1f}  {status:<10}")
        else:
            tw = r.get("controller_wall_s", 0.0)
            err = r.get("error_class", "FAIL")
            note = "OOM?" if r.get("likely_oom") else err
            print(f"{ne:>10}  {bs:>6}  {nm:>5}  "
                  f"{'--':>12}  {'--':>11}  {'--':>11}  "
                  f"{tw:>8,.1f}  {note:<10}")
    print("=" * 80)

    # Identify sweet spot.
    ok = [r for r in results if r.get("ok") and not r.get("incomplete")]
    if ok:
        best = max(ok, key=lambda r: r.get("steady_sps", 0.0))
        print(f"\nBest steady-state throughput: num_envs={best['num_envs']} "
              f"@ {best.get('steady_sps', 0):,.0f} it/s "
              f"(batch={best['batch_size']}, minibatches={best['num_minibatches']})")
    else:
        print("\nNo successful runs in this sweep.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=str,
                   default=",".join(str(n) for n in DEFAULT_NUM_ENVS),
                   help="comma-separated list of num_envs to test "
                        f"(default: {DEFAULT_NUM_ENVS})")
    p.add_argument("--timeout", type=int, default=300,
                   help="per-config timeout in seconds (default 300 = 5 min)")
    p.add_argument("--num-timesteps", type=int, default=200_000_000,
                   help="upper bound on env steps per worker — should be "
                        "comfortably larger than (warmup+measure) * "
                        "training_metrics_steps so the worker never "
                        "exits 'normally' before the measurement window "
                        "fills (default 200M)")
    p.add_argument("--training-metrics-steps", type=int, default=200_000,
                   help="env steps between Brax progress_fn callbacks "
                        "(default 200000)")
    p.add_argument("--warmup-callbacks", type=int, default=2,
                   help="number of progress callbacks to discard before "
                        "starting the measurement window (default 2)")
    p.add_argument("--measure-callbacks", type=int, default=4,
                   help="callbacks to use for the steady-state window "
                        "(default 4: SPS = delta_step / delta_wall over "
                        "the last 3 intervals)")
    p.add_argument("--stop-on-fail", action="store_true",
                   help="halt the sweep on the first failed config "
                        "(default behaviour after 2026-05-05 plan: "
                        "treat OOM at high num_envs as the natural stop)")
    args = p.parse_args()

    candidates = [int(x.strip()) for x in args.num_envs.split(",") if x.strip()]
    print("=" * 80)
    print(f"bench_num_envs sweep — {len(candidates)} config(s)")
    print(f"  candidates: {candidates}")
    print(f"  timeout/cfg: {args.timeout}s")
    print(f"  warmup callbacks: {args.warmup_callbacks}")
    print(f"  measure callbacks: {args.measure_callbacks}")
    print(f"  metrics_steps: {args.training_metrics_steps:,d}")
    print(f"  num_timesteps cap: {args.num_timesteps:,d}")
    print(f"  results JSON: {RESULTS_PATH}")
    print("=" * 80)

    results: list[dict] = []
    sweep_started = datetime.utcnow().isoformat() + "Z"
    sweep_t0 = time.perf_counter()

    for ne in candidates:
        bs, nm = make_batch_minibatches(ne)
        result = run_one(
            num_envs=ne,
            batch_size=bs,
            num_minibatches=nm,
            timeout_s=args.timeout,
            training_metrics_steps=args.training_metrics_steps,
            warmup_callbacks=args.warmup_callbacks,
            measure_callbacks=args.measure_callbacks,
            num_timesteps=args.num_timesteps,
        )
        results.append(result)
        # Persist incrementally — if we crash on a later config, the
        # earlier ones are still saved.
        _save(results, sweep_started, args)
        if not result.get("ok"):
            print(f"\n!! config num_envs={ne} did NOT produce a clean result.")
            if args.stop_on_fail:
                print(f"   --stop-on-fail set; aborting sweep here.")
                break
            # Heuristic: if a config OOMs/fails at a high num_envs, the
            # next-larger candidate will almost certainly OOM too. Skip
            # the rest of the sweep to protect the system.
            if ne >= 8192:
                print(f"   num_envs >= 8192 failed; skipping larger "
                      f"candidates as a safety measure.")
                break

    sweep_total = time.perf_counter() - sweep_t0
    print(f"\nsweep total wall time: {sweep_total:.1f}s "
          f"({sweep_total/60:.1f} min)")
    print_table(results)
    _save(results, sweep_started, args)
    print(f"\nresults saved: {RESULTS_PATH}")


def _save(results: list[dict], sweep_started: str, args) -> None:
    payload = {
        "sweep_started_utc": sweep_started,
        "saved_utc": datetime.utcnow().isoformat() + "Z",
        "host": platform.node(),
        "platform": platform.platform(),
        "candidates": [int(x.strip()) for x in args.num_envs.split(",")
                       if x.strip()],
        "config": {
            "timeout_s": args.timeout,
            "num_timesteps_cap": args.num_timesteps,
            "training_metrics_steps": args.training_metrics_steps,
            "warmup_callbacks": args.warmup_callbacks,
            "measure_callbacks": args.measure_callbacks,
        },
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
