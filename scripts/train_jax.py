"""
train_jax.py — Brax PPO training for the MJX-native hexapod env.

First cut of Phase 4: minimum viable PPO loop. Single gait_scale (no
curriculum yet); no BC init yet. Both layer cleanly on top of this
once the basic loop is proven.

Run inside the WSL2 venv (jax+cuda+brax+mujoco-mjx):
    PYTHONPATH=. ~/.venv-mjx/bin/python train_jax.py

Tuned to be a quick smoke run by default — bump TOTAL_STEPS to do
real training. Inputs/outputs:
  * progress logged via brax's `progress_fn` callback to stdout
  * final params written to `checkpoints/<RUN_NAME>/params.pkl`
"""
from __future__ import annotations

import argparse
import functools
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp

# Enable persistent JAX compilation cache before any JAX op runs.
from chain_train import enable_jax_cache
from tensorboardX import SummaryWriter

enable_jax_cache()

# JAX 0.10 removed `device_put_replicated` but Brax 0.14.2 still calls it.
# Shim it back: replicate `x` along a new leading axis of size len(devices).
# Single-GPU is the only configuration we use, so this is straightforward.
if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(x, devices):
        n = len(devices)
        def replicate(v):
            a = jnp.asarray(v)               # handles Python floats/ints
            return jnp.broadcast_to(a, (n,) + a.shape).copy()
        return jax.tree.map(replicate, x)
    jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import train as ppo_train

# ============================================================================
# CONFIG
# ============================================================================
from chain_train import MODEL_PATH  # one source of truth

from envs.hexapod_brax_env import HexapodBraxEnv

RUN_NAME       = "mjx_smoke"

# Curriculum: list of (gait_scale, num_steps) per stage. Each stage builds
# a fresh env at the given gait_scale and runs PPO; later stages chain
# from the previous stage's params via Brax's `restore_params`.
#
# Set to a single-stage list (e.g. [(0.0, TOTAL_STEPS)]) for autonomous
# training without scaffold. Match the project's gym-env curriculum:
#   from-scratch:     [(1.0, 8M), (0.6, 4M), (0.0, 13M)]   (~25M)
#   BC-init refine:   [(0.0, TOTAL)]                       (single stage)
#
# CLI override: `--gait-scale X` collapses the list to a single stage of
# `--steps` env steps at gait_scale=X.
CURRICULUM = [(0.0, 5_000_000)]   # default = single autonomous stage

# Throughput knobs.
NUM_ENVS       = 4096         # parallel rollouts (vmapped on GPU)
EPISODE_LENGTH = 1000         # steps per rollout segment
LEARNING_RATE  = 3e-4
ENTROPY_COST   = 1e-3
DISCOUNT       = 0.97
UNROLL_LENGTH  = 20
# Brax requires batch_size * num_minibatches % num_envs == 0. With
# NUM_ENVS=4096 the smallest multiple is 4096 (e.g. 512 * 8 or 256 * 16).
BATCH_SIZE     = 512
NUM_MINIBATCHES = 8
NUM_UPDATES_PER_BATCH = 4
NUM_EVALS      = 5
# Brax also fires progress_fn with rollout-time metrics every
# TRAINING_METRICS_STEPS env steps (cheap, no extra eval rollouts).
# Smaller = chattier progress; pick a value that gives ~50–100 prints.
TRAINING_METRICS_STEPS = 100_000
SEED           = 0


# ============================================================================
# Progress + checkpoint
# ============================================================================
# Tracker for instantaneous (between-print) step rate. Resets per stage.
# `tb_writer` slot holds a tensorboardX SummaryWriter set per stage so
# every scalar in `metrics` lands in a per-run logdir.
_PROG_STATE = {"last_step": 0, "last_t": None, "tb_writer": None}


def _progress(num_steps: int, metrics: dict, t0: float) -> None:
    now = time.perf_counter()
    elapsed = now - t0
    cum_sps = num_steps / max(elapsed, 1e-6)

    # Instantaneous it/s — env steps between this and the previous print.
    # First call after a (re)start has last_t=None; show "—" so the JIT
    # compile time at the start doesn't get reported as the steady-state rate.
    if _PROG_STATE["last_t"] is None:
        inst_sps_str = "       --"
    else:
        d_step = num_steps - _PROG_STATE["last_step"]
        d_t    = max(now - _PROG_STATE["last_t"], 1e-6)
        inst_sps_str = f"{d_step / d_t:>10,.0f}"
    _PROG_STATE["last_step"] = num_steps
    _PROG_STATE["last_t"]    = now

    rew = metrics.get("eval/episode_reward",
                      metrics.get("training/episode_reward", float("nan")))
    loss = metrics.get("training/total_loss",
                       metrics.get("training/loss", float("nan")))
    tag = "EVAL " if "eval/episode_reward" in metrics else "train"
    print(f"  [{tag}] step={num_steps:>10,d}  "
          f"reward={float(rew):+7.3f}  loss={float(loss):+.4f}  "
          f"it/s={inst_sps_str}  cum={cum_sps:>9,.0f}  "
          f"elapsed={elapsed/60:5.1f} min")

    # Write every scalar metric to TensorBoard so eval reward, training
    # losses, KL, advantages, etc. can be plotted across the whole run.
    writer = _PROG_STATE.get("tb_writer")
    if writer is not None:
        for k, v in metrics.items():
            try:
                writer.add_scalar(k, float(v), num_steps)
            except (TypeError, ValueError):
                continue                         # skip non-scalar metrics
        writer.flush()


def _save_params(out_dir: Path, params) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "params.pkl"
    with open(path, "wb") as f:
        pickle.dump(params, f)
    print(f"saved params: {path}")


# ============================================================================
# main
# ============================================================================
def _train_one_stage(stage_idx, gait_scale, num_steps, num_envs,
                     restore_params, t0_run, run_name,
                     cmd_mask: str = "stage3",
                     action_space: str = "joint"):
    """Run a single PPO stage at the given gait_scale. Returns final params."""
    print(f"\n{'='*70}")
    print(f"STAGE {stage_idx}:  gait_scale={gait_scale}  steps={num_steps:,d}  "
          f"cmd_mask={cmd_mask}  action_space={action_space}  "
          f"resume={restore_params is not None}")
    print(f"{'='*70}")
    # Reset the per-stage instantaneous-rate tracker so the first JIT-compile
    # tick doesn't taint this stage's steady-state numbers.
    _PROG_STATE["last_step"] = 0
    _PROG_STATE["last_t"]    = None
    # Open a per-stage TensorBoard writer.
    log_dir = Path("logs") / run_name / f"stage_{stage_idx}_gs{gait_scale}"
    log_dir.mkdir(parents=True, exist_ok=True)
    if _PROG_STATE.get("tb_writer") is not None:
        _PROG_STATE["tb_writer"].close()
    _PROG_STATE["tb_writer"] = SummaryWriter(logdir=str(log_dir))
    print(f"tensorboard logdir: {log_dir}")
    env = HexapodBraxEnv(MODEL_PATH, gait_scale=gait_scale, cmd_mask=cmd_mask,
                         action_space=action_space)
    train_fn = functools.partial(
        ppo_train.train,
        environment        = env,
        num_timesteps      = num_steps,
        num_envs           = num_envs,
        episode_length     = EPISODE_LENGTH,
        learning_rate      = LEARNING_RATE,
        entropy_cost       = ENTROPY_COST,
        discounting        = DISCOUNT,
        unroll_length      = UNROLL_LENGTH,
        batch_size         = BATCH_SIZE,
        num_minibatches    = NUM_MINIBATCHES,
        num_updates_per_batch = NUM_UPDATES_PER_BATCH,
        num_evals          = NUM_EVALS,
        seed               = SEED + stage_idx,    # stage-specific seed
        normalize_observations = True,
        restore_params     = restore_params,
        log_training_metrics = True,
        training_metrics_steps = TRAINING_METRICS_STEPS,
        progress_fn        = lambda step, m: _progress(step, m, t0_run),
    )
    _make_inference_fn, params, _metrics = train_fn()

    # Per-stage checkpoint.
    stage_dir = Path("checkpoints") / run_name / f"stage_{stage_idx}_gs{gait_scale}"
    _save_params(stage_dir, params)
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None,
                        help="total env steps (collapses curriculum to one stage)")
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--gait-scale", type=float, default=None,
                        help="single-stage gait_scale override")
    parser.add_argument("--run", type=str, default=RUN_NAME)
    parser.add_argument("--multi-stage-smoke", action="store_true",
                        help="2-stage curriculum smoke (verifies restore chain)")
    parser.add_argument("--restore", type=str, default=None,
                        help="path to (normalizer, policy, value) pickle "
                             "from pretrain_bc_jax.py")
    parser.add_argument("--cmd-mask", type=str, default="stage3",
                        choices=["stage1", "stage2", "stage3"],
                        help="cmd-slot curriculum stage: stage1=translation "
                             "only, stage2=+yaw+height+width, stage3=full motion")
    parser.add_argument("--action-space", type=str, default="joint",
                        choices=["joint", "foot"],
                        help="joint = 18-dim joint residual (current default). "
                             "foot = (6,3) foot residual in body frame; IK "
                             "converts to joint targets. Foot-space gives "
                             "smoother coordinated noise.")
    args = parser.parse_args()

    # Build curriculum: file-level CURRICULUM, CLI single-stage override,
    # or the multi-stage smoke test.
    if args.multi_stage_smoke:
        curriculum = [(0.5, 40_000), (0.0, 40_000)]
    elif args.gait_scale is not None or args.steps is not None:
        gs = args.gait_scale if args.gait_scale is not None else CURRICULUM[0][0]
        st = args.steps      if args.steps is not None      else CURRICULUM[0][1]
        curriculum = [(gs, st)]
    else:
        curriculum = CURRICULUM

    print("=" * 70)
    print("JAX-MJX hexapod training")
    print(f"  run:        {args.run}")
    print(f"  model:      {MODEL_PATH}")
    print(f"  num_envs:   {args.num_envs}")
    print(f"  jax devices: {jax.devices()}")
    print(f"  curriculum: {len(curriculum)} stage(s)")
    for i, (gs, st) in enumerate(curriculum):
        print(f"    stage {i}:  gait_scale={gs}  steps={st:,d}")
    print("=" * 70)

    t0_run = time.perf_counter()
    params = None
    if args.restore is not None:
        with open(args.restore, "rb") as f:
            params = pickle.load(f)
        print(f"\nrestoring from {args.restore}")

    for stage_idx, (gait_scale, num_steps) in enumerate(curriculum):
        params = _train_one_stage(
            stage_idx, gait_scale, num_steps, args.num_envs,
            restore_params=params, t0_run=t0_run, run_name=args.run,
            cmd_mask=args.cmd_mask, action_space=args.action_space,
        )

    elapsed = time.perf_counter() - t0_run
    print(f"\nALL STAGES DONE in {elapsed/60:.1f} min total")

    final_dir = Path("checkpoints") / args.run / "final"
    _save_params(final_dir, params)
    print(f"final params: {final_dir}/params.pkl")


if __name__ == "__main__":
    main()
