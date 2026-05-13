# Project reorganization plan (2026-05-05)

Audit + proposed layout for the hexapod project, designed to:
- Group files by role rather than alphabetical accident
- Reserve dedicated space for upcoming AMP modules
- Move legacy SubprocVecEnv code into a clearly-marked subdir
- Move one-off tools out of the root
- Park papers in `docs/papers/` instead of the root
- Preserve git history via `git mv`
- Keep data dirs at root (no point moving them)

---

## Current root inventory (40 files at root)

### Active MJX training pipeline (KEEP, move to `scripts/`)
- `train_jax.py` — Brax PPO training
- `pretrain_bc_jax.py` — BC pretraining
- `chain_train.py` — chain orchestrator (also holds shared session config)
- `run_overnight_curriculum.py` — tiered curriculum one-shot
- `watch_demo_jax.py` — JAX/MJX policy viewer
- `eval_bc_quick.py` — quantitative eval
- `diagnose_bc_axes.py` — BC axis-response probe

### Legacy SB3 SubprocVecEnv stack (KEEP for reference, move to `legacy/sb3/`)
- `train.py` — SB3 PPO + SubprocVecEnv training
- `pretrain_bc.py` — SB3 BC
- `watch.py`, `watch_demo.py`, `watch_tiled.py` — SB3 viewers
- `pilot.py`, `pilot_ai.py` — keyboard teleop
- `live_viewer.py` — SHM live viewer (paired with SB3 train)
- `bench_n_envs.py` — SubprocVecEnv N_ENVS sweep

### Standalone sandboxes / demos (KEEP, move to `legacy/sandboxes/`)
- `IK_gait.py` — gait sandbox with 25-phase smart test
- `simple_gait.py` — basic gait demo viewer
- `demo.py` — overlay cycle demo
- `walk_test.py` — broken legacy script (per CLAUDE.md note)

### Tools / one-off helpers (KEEP, move to `tools/`)
- `derive_joint_limits.py` — auto-derive joint RoM via collision sweep
- `apply_joint_limits.py` — patch limits into MJCFs
- `validate_jax_vs_gym.py` — Phase 5 parity test (mostly historical now)

### Failed experiments (DELETE)
- `train_jax_apg.py` — APG attempt blocked by MJX while_loop diff issue, task #22 deleted

### Configs / docs / data (KEEP at root)
- `CLAUDE.md`, `README.md`, `LICENSE`, `NOTICE`
- `requirements.txt`, `.gitignore`
- `joint_limits.json` — data file consumed by `apply_joint_limits.py`

### Artifacts / logs (DELETE or ignore)
- `MUJOCO_LOG.TXT` — orphan log, delete

### Papers (move to `docs/papers/`)
- `Learning Natural and Robust Hexapod Locomotion.pdf`

---

## Proposed layout

```
/
├── (configs at root: CLAUDE.md, README.md, LICENSE, NOTICE, requirements.txt,
│    .gitignore, joint_limits.json)
│
├── amp/                                # NEW — AMP modules (upcoming work)
│   ├── __init__.py
│   ├── prior_data.py                   # collect (s_t, s_t+1) from scaffold rollouts
│   ├── discriminator.py                # discriminator network + losses
│   └── (training.py later — wires AMP into PPO loop)
│
├── envs/                               # unchanged
│   ├── __init__.py
│   ├── hexapod_env.py                  # gym env (SB3 + watch use it)
│   ├── hexapod_env_jax.py              # JAX/MJX env
│   └── hexapod_brax_env.py             # Brax adapter
│
├── gait/                               # unchanged
│   ├── __init__.py
│   ├── controller.py                   # numpy gait controller
│   └── controller_jax.py               # JAX gait controller
│
├── scripts/                            # NEW — entry-point scripts
│   ├── train_jax.py
│   ├── pretrain_bc_jax.py
│   ├── chain_train.py                  # ALSO holds shared session config
│   ├── run_overnight_curriculum.py
│   ├── watch_demo_jax.py
│   ├── eval_bc_quick.py
│   ├── diagnose_bc_axes.py
│   └── (train_jax_amp.py later — AMP variant of train_jax)
│
├── tools/                              # NEW — one-off helpers / dev
│   ├── derive_joint_limits.py
│   ├── apply_joint_limits.py
│   └── validate_jax_vs_gym.py
│
├── legacy/                             # NEW — kept for reference, not active
│   ├── sb3/                            # SB3 SubprocVecEnv pipeline
│   │   ├── train.py
│   │   ├── pretrain_bc.py
│   │   ├── watch.py
│   │   ├── watch_demo.py               # but DEMO_PHASES extracted (see note)
│   │   ├── watch_tiled.py
│   │   ├── pilot.py
│   │   ├── pilot_ai.py
│   │   ├── live_viewer.py
│   │   └── bench_n_envs.py
│   └── sandboxes/                      # standalone gait sandboxes
│       ├── IK_gait.py
│       ├── simple_gait.py
│       ├── demo.py
│       └── walk_test.py
│
├── docs/
│   ├── kinematics.md
│   ├── REORG_PLAN.md                   # this file (delete after exec)
│   └── papers/                         # NEW
│       └── Learning Natural and Robust Hexapod Locomotion.pdf
│
├── models/                             # unchanged
├── checkpoints/                        # unchanged
├── logs/, logs_legacy/                 # unchanged
├── snapshots/                          # unchanged
└── .cache/                             # JAX compile cache, gitignored
```

---

## Special concern: `DEMO_PHASES` shared between scripts

`watch_demo_jax.py` (active) imports `DEMO_PHASES` from `watch_demo.py` (legacy).
After the move, that import would cross the active/legacy boundary.

**Resolution:** extract `DEMO_PHASES` and the helper functions (`_cmd`, `_walk`,
`_spin`, `latest_run_dir`, `latest_checkpoint`) into a new shared module
`scripts/demo_phases.py`. Both `watch_demo_jax.py` and `legacy/sb3/watch_demo.py`
import from it.

This keeps the legacy SB3 watcher working (no behavior change) while allowing
the active watcher to import from a sibling, not from `legacy/`.

---

## Import + invocation impact

**Within `scripts/` siblings:** Should "just work" since Python adds the script's
own dir to `sys.path` when invoked directly. `from chain_train import BASE_NAME`
inside `watch_demo_jax.py` continues to work after both move to `scripts/`.

**Cross-package imports (`scripts/foo.py` → `envs/...`):** Need `PYTHONPATH=.`
when invoked from project root, same as today. Or wrap with `sys.path.insert`.

**Subprocess calls in `chain_train.py`:** Currently invokes
`python train_jax.py`. Needs to become `python scripts/train_jax.py` after
the move.

**Run commands in CLAUDE.md / README.md:** All `python train_jax.py ...`
examples become `python scripts/train_jax.py ...`. Affects ~10 commands.

---

## Step-by-step execution checklist (for next task #38)

1. Create new dirs (`amp/`, `scripts/`, `tools/`, `legacy/sb3/`,
   `legacy/sandboxes/`, `docs/papers/`)
2. Extract `DEMO_PHASES` etc. into `scripts/demo_phases.py`
3. Update `watch_demo.py` (legacy) to import from `scripts.demo_phases`
   instead of holding it inline (or keep both copies — decide based on
   import-path complexity)
4. `git mv` files into new homes per the layout above
5. Update import statements that referenced moved files (most are
   `from chain_train import ...` which still works since chain_train moved
   together with its consumers)
6. Update `chain_train.py`'s subprocess invocations to use new script paths
7. Update `chain_train.py`'s `PROJECT_ROOT` calculation if needed (it uses
   `Path(__file__).resolve().parent` — would now resolve to `scripts/`, but
   we want `PROJECT_ROOT` to be the actual project root; need to use
   `.parent.parent`)
8. Update CLAUDE.md "Files at a glance" section + any example commands
9. Update README.md if it mentions specific scripts
10. Delete `train_jax_apg.py` and `MUJOCO_LOG.TXT`
11. Smoke-test each active script:
    - `python scripts/chain_train.py --help`
    - `python scripts/watch_demo_jax.py --bc` (loads BC checkpoint, opens viewer)
    - `python scripts/eval_bc_quick.py --episodes 2`
    - `python scripts/diagnose_bc_axes.py`
12. Light cleanup pass while moving each file: stale comments, dead code
13. `git commit` with detailed message listing all moves

---

## What I'm NOT changing

- Module dirs (`envs/`, `gait/`) — already well-organized
- Data dirs (`checkpoints/`, `logs/`, `models/`, `snapshots/`) — keep at root
- Any code logic — only file locations and imports
- Reward function — separate task (#43)
- Network architecture — separate task (AMP work)

---

## Open questions for review

1. Are you OK with `legacy/sb3/` and `legacy/sandboxes/` as the split?
   Alternative: flat `legacy/` with everything inside.
2. `IK_gait.py` is a sandbox but is also our most useful gait visualizer
   (the 25-phase smart test). Move to `legacy/sandboxes/` or `tools/`?
   I leaned legacy because it's not part of the training pipeline, but it IS
   actively useful.
3. Do you want `docs/papers/` to also hold any PDFs we add later, or should
   they live elsewhere?
4. Anything currently at root I should KEEP at root for ergonomic reasons
   (e.g., scripts you run constantly)?
