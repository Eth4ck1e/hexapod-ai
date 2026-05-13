"""
_bench_num_envs_worker.py — single-config PPO benchmark worker.

Runs ONE Brax PPO training segment at the requested num_envs and prints
a single line `BENCH_RESULT: {...}` (JSON) to stdout when measurement
is complete, then exits.

Designed to be invoked by tools/bench_num_envs.py as a subprocess (so
each config gets a fresh JAX process and clean GPU memory). Crashes /
OOMs are caught by the parent via the subprocess exit code.

Methodology — "smart timer" steady-state SPS:

  1. Brax fires `progress_fn(step, metrics)` every TRAINING_METRICS_STEPS
     env steps (cheap rollout-time metrics, no eval overhead).
  2. The first few callbacks include JIT compile + warmup time and are
     unreliable; we SKIP them.
  3. After the warmup window, we record (step, wall-time) for each
     subsequent callback. Once we have enough samples, the steady-state
     SPS = (step_last - step_first) / (t_last - t_first) over the
     measurement window. We then sys.exit(0) — no need to finish the
     full num_timesteps budget.

Run only inside the WSL2 venv:
    PYTHONPATH=. ~/.venv-mjx/bin/python tools/_bench_num_envs_worker.py \\
        --num-envs 4096 --batch-size 512 --num-minibatches 8
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
import traceback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-minibatches", type=int, required=True)
    parser.add_argument("--num-timesteps", type=int, default=200_000_000,
                        help="upper bound on env steps; we exit early once "
                             "the measurement window is full")
    parser.add_argument("--training-metrics-steps", type=int, default=200_000,
                        help="how often Brax fires progress_fn (env steps)")
    parser.add_argument("--warmup-callbacks", type=int, default=2,
                        help="number of progress_fn calls to discard before "
                             "starting the measurement window")
    parser.add_argument("--measure-callbacks", type=int, default=4,
                        help="callbacks to use for the steady-state window "
                             "(SPS = delta_steps / delta_wall over them)")
    parser.add_argument("--cmd-mask", type=str, default="stage1")
    parser.add_argument("--action-space", type=str, default="foot")
    parser.add_argument("--model-path", type=str,
                        default="models/phantomx_simple_mjx.xml")
    parser.add_argument("--gait-scale", type=float, default=0.0)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t_script_start = time.perf_counter()

    # ---- JAX / Brax import + Brax 0.14.2 / JAX 0.10 shim ------------------
    import jax
    import jax.numpy as jnp

    # Persistent JAX cache — much shorter compile times after first run.
    # `scripts/` has no __init__.py so we can't `import scripts.chain_train`;
    # add scripts/ to sys.path and import directly.
    try:
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from chain_train import enable_jax_cache
        enable_jax_cache()
    except Exception as e:
        print(f"[bench] (warning) JAX cache disabled: {e}", flush=True)

    if not hasattr(jax, "device_put_replicated"):
        def _device_put_replicated(x, devices):
            n = len(devices)
            def replicate(v):
                a = jnp.asarray(v)
                return jnp.broadcast_to(a, (n,) + a.shape).copy()
            return jax.tree.map(replicate, x)
        jax.device_put_replicated = _device_put_replicated

    from brax.training.agents.ppo import train as ppo_train
    from envs.hexapod_brax_env import HexapodBraxEnv

    # ---- progress_fn: smart-timer steady-state SPS ------------------------
    state = {
        "calls": 0,
        "samples": [],          # list of (step, wall) AFTER warmup
        "first_call_t": None,
        "exit_done": False,
    }

    def progress_fn(num_steps: int, metrics: dict) -> None:
        now = time.perf_counter()
        state["calls"] += 1
        if state["first_call_t"] is None:
            state["first_call_t"] = now
            print(f"  [bench] callback 1 at step={num_steps:,d} "
                  f"(t+{now - t_script_start:.1f}s since process start)",
                  flush=True)
            return

        idx = state["calls"]
        if idx <= args.warmup_callbacks:
            print(f"  [bench] warmup callback {idx} at step={num_steps:,d} "
                  f"(t+{now - t_script_start:.1f}s) — discarding",
                  flush=True)
            return

        # In the measurement window.
        state["samples"].append((num_steps, now))
        n = len(state["samples"])
        if n >= 2:
            s0_step, s0_t = state["samples"][0]
            s1_step, s1_t = state["samples"][-1]
            d_step = s1_step - s0_step
            d_t = max(s1_t - s0_t, 1e-9)
            inst_sps = d_step / d_t
            print(f"  [bench] meas {n}/{args.measure_callbacks} step={num_steps:,d} "
                  f"window_sps={inst_sps:>10,.0f}",
                  flush=True)
        else:
            print(f"  [bench] meas {n}/{args.measure_callbacks} step={num_steps:,d} "
                  f"(anchor)",
                  flush=True)

        if n >= args.measure_callbacks and not state["exit_done"]:
            state["exit_done"] = True
            s0_step, s0_t = state["samples"][0]
            s1_step, s1_t = state["samples"][-1]
            d_step = s1_step - s0_step
            d_t = s1_t - s0_t
            steady_sps = d_step / max(d_t, 1e-9)
            result = {
                "ok": True,
                "num_envs": args.num_envs,
                "batch_size": args.batch_size,
                "num_minibatches": args.num_minibatches,
                "unroll_length": args.unroll_length,
                "warmup_callbacks": args.warmup_callbacks,
                "measure_callbacks": args.measure_callbacks,
                "window_steps": int(d_step),
                "window_wall_s": float(d_t),
                "steady_sps": float(steady_sps),
                "total_wall_s": float(now - t_script_start),
                "total_steps_to_finish": int(num_steps),
            }
            print("BENCH_RESULT: " + json.dumps(result), flush=True)
            # Exit hard — Brax internals don't expect early termination, so
            # raising in the callback can corrupt logs. os._exit skips any
            # atexit / cleanup that might hang.
            os._exit(0)

    # ---- Build env + train ------------------------------------------------
    print(f"[bench] config: num_envs={args.num_envs}  batch={args.batch_size}  "
          f"minibatches={args.num_minibatches}  "
          f"product={args.batch_size * args.num_minibatches}  "
          f"divides_num_envs={'YES' if (args.batch_size * args.num_minibatches) % args.num_envs == 0 else 'NO'}",
          flush=True)
    print(f"[bench] jax devices: {jax.devices()}", flush=True)

    env = HexapodBraxEnv(args.model_path,
                         gait_scale=args.gait_scale,
                         cmd_mask=args.cmd_mask,
                         action_space=args.action_space)

    train_fn = functools.partial(
        ppo_train.train,
        environment           = env,
        num_timesteps         = args.num_timesteps,
        num_envs              = args.num_envs,
        episode_length        = args.episode_length,
        learning_rate         = 3e-4,
        entropy_cost          = 1e-3,
        discounting           = 0.97,
        unroll_length         = args.unroll_length,
        batch_size            = args.batch_size,
        num_minibatches       = args.num_minibatches,
        num_updates_per_batch = 4,
        num_evals             = 2,                 # minimal; eval cost is small
        seed                  = args.seed,
        normalize_observations = True,
        log_training_metrics   = True,
        training_metrics_steps = args.training_metrics_steps,
        progress_fn            = progress_fn,
    )

    try:
        train_fn()
    except SystemExit:
        raise
    except Exception as e:
        # OOM / shape mismatch / etc — emit structured failure record.
        msg = f"{type(e).__name__}: {e}"
        # Truncate the traceback to keep stdout reasonable.
        tb = traceback.format_exc().splitlines()
        tb_tail = "\n".join(tb[-25:])
        result = {
            "ok": False,
            "num_envs": args.num_envs,
            "batch_size": args.batch_size,
            "num_minibatches": args.num_minibatches,
            "error_class": type(e).__name__,
            "error_msg": str(e)[:500],
            "traceback_tail": tb_tail,
            "total_wall_s": time.perf_counter() - t_script_start,
        }
        print("BENCH_RESULT: " + json.dumps(result), flush=True)
        os._exit(2)

    # If we got here without exiting via progress_fn, training finished
    # before the measurement window filled — flag as inconclusive.
    samples = state["samples"]
    if len(samples) >= 2:
        s0_step, s0_t = samples[0]
        s1_step, s1_t = samples[-1]
        steady_sps = (s1_step - s0_step) / max(s1_t - s0_t, 1e-9)
    else:
        steady_sps = float("nan")
    result = {
        "ok": True,
        "incomplete": True,
        "num_envs": args.num_envs,
        "batch_size": args.batch_size,
        "num_minibatches": args.num_minibatches,
        "steady_sps": float(steady_sps),
        "samples_collected": len(samples),
        "note": "training finished before measurement window filled",
    }
    print("BENCH_RESULT: " + json.dumps(result), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
