"""
bench_unroll_length.py — sweep over unroll_length to find optimal PPO
throughput at the proven num_envs=8192 operating point.

For each unroll_length in the candidate list, this script launches a
fresh WSL subprocess running tools/_bench_unroll_length_worker.py, which
executes a short Brax PPO segment and emits a single
`BENCH_RESULT: {...}` JSON line on stdout. We parse that, accumulate
results, and print a comparison table at the end.

Why clone rather than parameterise bench_num_envs.py:
  The sweep dimensions differ in what's held fixed vs swept. The
  batch_size/minibatches formula changes: for num_envs sweeps we derive
  batch_size from num_envs; here batch_size and num_minibatches are
  fixed constants (1024 and 8 at num_envs=8192) and only unroll_length
  varies. The table columns and result schema also need the unroll_length
  field prominently. A clean clone avoids if-branches that would obscure
  both sweep types.

Usage:
    python tools/bench_unroll_length.py
    python tools/bench_unroll_length.py --unroll-lengths 5,10,20,40
    python tools/bench_unroll_length.py --timeout 400

Run from anywhere on Windows (the script shells out to WSL itself).
JSON results are saved to tools/bench_unroll_length_results.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "tools" / "bench_unroll_length_results.json"

# Fixed params from the num_envs sweep best result.
FIXED_NUM_ENVS       = 8192
FIXED_BATCH_SIZE     = 1024   # num_envs // num_minibatches
FIXED_NUM_MINIBATCHES = 8

DEFAULT_UNROLL_LENGTHS = [5, 10, 15, 20, 30, 40]


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
        used, total = (int(x.strip()) for x in out.split(","))
        return f"VRAM {used} / {total} MiB ({100*used/total:.0f}%)"
    except Exception:
        return ""


def run_one(unroll_length: int, timeout_s: int,
            training_metrics_steps: int,
            warmup_callbacks: int, measure_callbacks: int,
            num_timesteps: int) -> dict:
    """Spawn a worker subprocess for one config; return the parsed
    BENCH_RESULT dict (with extra controller-side fields added)."""
    cmd_inner = (
        "cd '/mnt/c/Users/Eth4ck1e/OneDrive/Documents/Hexapod AI Project' && "
        "PYTHONPATH=. ~/.venv-mjx/bin/python tools/_bench_unroll_length_worker.py "
        f"--num-envs {FIXED_NUM_ENVS} "
        f"--batch-size {FIXED_BATCH_SIZE} "
        f"--num-minibatches {FIXED_NUM_MINIBATCHES} "
        f"--unroll-length {unroll_length} "
        f"--training-metrics-steps {training_metrics_steps} "
        f"--warmup-callbacks {warmup_callbacks} "
        f"--measure-callbacks {measure_callbacks} "
        f"--num-timesteps {num_timesteps}"
    )
    if platform.system() == "Windows":
        full = ["wsl", "bash", "-lc", cmd_inner]
    else:
        full = ["bash", "-lc", cmd_inner]

    print(f"\n>>> RUN unroll_length={unroll_length}  num_envs={FIXED_NUM_ENVS}  "
          f"batch={FIXED_BATCH_SIZE}  minibatches={FIXED_NUM_MINIBATCHES}")
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
        parsed = {
            "ok": False,
            "num_envs": FIXED_NUM_ENVS,
            "batch_size": FIXED_BATCH_SIZE,
            "num_minibatches": FIXED_NUM_MINIBATCHES,
            "unroll_length": unroll_length,
            "error_class": "Timeout" if timed_out else "NoResult",
            "error_msg": ("worker timed out before producing a result"
                          if timed_out else
                          f"worker exited rc={rc} without BENCH_RESULT"),
            "likely_oom": bool(rc is not None and rc != 0 and not timed_out),
        }

    parsed["timed_out"] = timed_out
    parsed["controller_wall_s"] = float(elapsed)
    parsed["worker_exit_code"] = rc
    return parsed


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("UNROLL_LENGTH SWEEP RESULTS  (num_envs=8192, batch=1024, minibatches=8)")
    print("=" * 80)
    hdr = (
        f"{'unroll':>8}  "
        f"{'steady it/s':>12}  "
        f"{'wnd_steps':>11}  "
        f"{'wnd_wall_s':>11}  "
        f"{'wall_s':>8}  "
        f"{'status':<10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ul = r.get("unroll_length", "?")
        if r.get("ok"):
            sps = r.get("steady_sps", float("nan"))
            sps_str = f"{sps:>12,.0f}" if sps == sps else f"{'nan':>12}"
            ws = r.get("window_steps", 0)
            ww = r.get("window_wall_s", 0.0)
            tw = r.get("controller_wall_s", 0.0)
            status = "INCOMPLETE" if r.get("incomplete") else "ok"
            print(f"{ul:>8}  {sps_str}  "
                  f"{ws:>11,d}  {ww:>11,.1f}  {tw:>8,.1f}  {status:<10}")
        else:
            tw = r.get("controller_wall_s", 0.0)
            err = r.get("error_class", "FAIL")
            note = "OOM?" if r.get("likely_oom") else err
            print(f"{ul:>8}  {'--':>12}  {'--':>11}  {'--':>11}  "
                  f"{tw:>8,.1f}  {note:<10}")
    print("=" * 80)

    ok = [r for r in results if r.get("ok") and not r.get("incomplete")]
    if ok:
        best = max(ok, key=lambda r: r.get("steady_sps", 0.0))
        prev = next((r for r in ok if r.get("unroll_length") == 20), None)
        print(f"\nBest steady-state throughput: unroll_length={best['unroll_length']} "
              f"@ {best.get('steady_sps', 0):,.0f} it/s")
        if prev and prev is not best:
            delta_pct = 100 * (best["steady_sps"] - prev["steady_sps"]) / prev["steady_sps"]
            print(f"vs default unroll_length=20:  {prev.get('steady_sps', 0):,.0f} it/s "
                  f"(delta {delta_pct:+.1f}%)")
        elif prev:
            print("Default unroll_length=20 IS the best candidate.")
    else:
        print("\nNo successful runs in this sweep.")


def _save(results: list[dict], sweep_started: str, args) -> None:
    payload = {
        "sweep_started_utc": sweep_started,
        "saved_utc": datetime.utcnow().isoformat() + "Z",
        "host": platform.node(),
        "platform": platform.platform(),
        "fixed": {
            "num_envs": FIXED_NUM_ENVS,
            "batch_size": FIXED_BATCH_SIZE,
            "num_minibatches": FIXED_NUM_MINIBATCHES,
        },
        "candidates": [int(x.strip()) for x in args.unroll_lengths.split(",")
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--unroll-lengths", type=str,
                   default=",".join(str(n) for n in DEFAULT_UNROLL_LENGTHS),
                   help="comma-separated list of unroll_length values to test "
                        f"(default: {DEFAULT_UNROLL_LENGTHS})")
    p.add_argument("--timeout", type=int, default=300,
                   help="per-config timeout in seconds (default 300 = 5 min)")
    p.add_argument("--num-timesteps", type=int, default=200_000_000,
                   help="upper bound on env steps per worker — safety cap; "
                        "worker exits via progress_fn well before this")
    p.add_argument("--training-metrics-steps", type=int, default=400_000,
                   help="env steps between Brax progress_fn callbacks "
                        "(default 400000 — matches num_envs sweep setting)")
    p.add_argument("--warmup-callbacks", type=int, default=2,
                   help="progress callbacks to discard before measurement "
                        "(covers JIT compile + GPU warmup)")
    p.add_argument("--measure-callbacks", type=int, default=5,
                   help="callbacks to use for the steady-state window "
                        "(SPS = delta_steps / delta_wall over them)")
    args = p.parse_args()

    candidates = [int(x.strip()) for x in args.unroll_lengths.split(",")
                  if x.strip()]
    print("=" * 80)
    print(f"bench_unroll_length sweep — {len(candidates)} config(s)")
    print(f"  fixed:      num_envs={FIXED_NUM_ENVS}  batch={FIXED_BATCH_SIZE}  "
          f"minibatches={FIXED_NUM_MINIBATCHES}")
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

    for ul in candidates:
        result = run_one(
            unroll_length=ul,
            timeout_s=args.timeout,
            training_metrics_steps=args.training_metrics_steps,
            warmup_callbacks=args.warmup_callbacks,
            measure_callbacks=args.measure_callbacks,
            num_timesteps=args.num_timesteps,
        )
        results.append(result)
        # Persist incrementally — if a later config crashes, earlier ones survive.
        _save(results, sweep_started, args)
        if not result.get("ok"):
            print(f"\n!! unroll_length={ul} did NOT produce a clean result.")
            print(f"   unroll_length variation should never OOM at fixed "
                  f"num_envs=8192; continuing sweep.")

    sweep_total = time.perf_counter() - sweep_t0
    print(f"\nsweep total wall time: {sweep_total:.1f}s "
          f"({sweep_total/60:.1f} min)")
    print_table(results)
    _save(results, sweep_started, args)
    print(f"\nresults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
