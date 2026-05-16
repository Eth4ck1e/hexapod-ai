# Project context for Claude Code

This file is auto-read by Claude Code at session start. Carries the
design decisions, current state, and working-style preferences.

## Project goal

Reinforcement-learning locomotion controller for a PhantomX-style
hexapod, deployed eventually on an ESP32-S3 microcontroller. Trains a
policy via supervised learning + RL on top of a working analytical gait
scaffold; scaffold serves as both initialization (BC pretrain) AND as
the motion-prior source for AMP-guided RL refinement.

## ⚠️ Path note — directory moved 2026-05-13

The project root was relocated off OneDrive to fix sync issues. Active
paths now:

- **Windows**: `C:\Users\Eth4ck1e\Documents\Hexapod AI Project`
- **WSL mount**: `/mnt/c/Users/Eth4ck1e/Documents/Hexapod AI Project`
- **Hyperresearch CLI**: `C:\Users\Eth4ck1e\Documents\Hexapod AI Project\.venv\Scripts\hyperresearch.exe`

**Dead paths** (do not use):
- `C:\Users\Eth4ck1e\OneDrive\Documents\Hexapod AI Project`
- `/mnt/c/Users/Eth4ck1e/OneDrive/Documents/Hexapod AI Project`

WSL venv at `~/.venv-mjx/` lives in WSL home — unaffected by the move.

## Current state (2026-05-15)

**v27 walks cleanly** — paper-pure reward + smaller priors + fixed σ convention. Best + final iters both visually walk; eval reward ~+1979/2000 max.

**MJX pipeline mature and battle-tested.** PPO training runs at
**~220-240k it/s** on the RTX 5060 Ti via WSL2 (back to pre-v24 levels
after v26+ reverted obs to 78-dim). A 1B-step run (20 × 50M segments)
finishes in ~2-2.5 hr. Optimal config: `num_envs=4096`, `batch=512`,
`minibatches=8`, `unroll_length=20`. VRAM-budget knobs (`NUM_ENVS`,
`DISC_BATCH`) are SoT in `chain_train.py` — change one place to scale.

**AMP pipeline went through ~27 design iterations** v3 → v27. Major shifts:
- v8: hardware-realistic max speed (0.356 m/s, MAX_SPEED 5× faster)
- v10: initial-pose domain randomization
- v11: paper-matching network (256, 128, 64) actor
- v15: cmd-conditional discriminator (CMD_DIM_FOR_DISC=3 initially)
- v17: full 9-D cmd in discriminator
- v18: K-NN cmd-matched prior sampling (K=10), normalized distance
- v21: uniform-z foot rest (fixed 3mm middle-leg asymmetry)
- v22: recovery curriculum (push impulses) — **SHELVED**, hexapod stability profile makes pushes the wrong test
- v23: **partition discriminator** (150 bins). Strict within-bin prior sampling. Fixed cmd-blur failure mode.
- v24: motor feedback obs (joint_torque + joint_pos_error, 78 → 114 dims) — REVERTED in v26.
- v25: EMA-filtered velocity tracking + contact penalty + drift pens — REVERTED in v26.
- v26: **paper-pure realignment**. Stripped all hand-added penalties (drift, contact_mismatch, EMA filter); restored z_vel_w/body_angvel_xy_w to paper Table I values; added missing joint_accel_w; switched to per-axis exp tracking; obs back to 78-dim. Initial bug: `exp(-||err||²/σ²)` instead of `exp(-||err||²/σ)` made tracking 30× too tight at typical errors — bot didn't walk.
- v27: **σ fix** (`/σ` not `/σ²`, matching legged_gym convention) + **smaller priors** (`amp_priors_v26_small.npz`, ~1M transitions vs 6.1M, saves ~2.2 GB GPU). NOW WALKING — best + final iters both clean.

**Current best walking policy**: latest `amp_to_v27*/iter*/final/params.pkl` (use `python scripts\launch.py watcher` to auto-detect best iter). v23 iter12 still works for the legacy 78-dim obs (which v27 is back to).

**Hardware reality check.** Bot is a Trossen PhantomX MK-III using **Dynamixel AX-12A** servos (TTL half-duplex multidrop UART, daisychained). Motor specs: 6.18 rad/s no-load, **1.5 N·m stall**, 9-12 V. Computed practical max walking speed: ~0.40 m/s. MJCF `forcerange` ±1.5 N·m matches real motor envelope. Full analysis in `docs/MAX_SPEED_ANALYSIS.md`.

## Single source of truth modules (the unification pattern)

Three modules own schema/configuration that needs to stay consistent
across the env, watcher, training scripts, and tools. **When you
change anything in one of these areas, edit ONLY the owning module.**

| Module | Owns | Consumers |
|---|---|---|
| `envs/obs_layout.py` | Policy observation schema (78 dims as of v26+, was 114 in v24-v25) | JAX env, gym env, watch_demo_jax, eval_bc_quick, record_policy |
| `envs/stance_envelope.py` | Stance height + dh-conditional width envelope | JAX env cmd sampler, watch_scaffold interactive presets, demo_phases watch tests |
| `envs/cmd_bins.py` | 150-bin partition (motion × height × width) | JAX env disc routing, prior_data binning, multi-head disc, train_jax_amp |
| `scripts/chain_train.py` | Session config + VRAM-budget knobs: `BASE_NAME`, `MODEL_PATH`, `RENDER_MODEL_PATH`, `ACTION_SPACE`, `NUM_ENVS`, `DISC_BATCH` | train_jax_amp, pretrain_bc_jax, watch_demo_jax, watch_controller |

If you find yourself manually syncing the same constant or formula
between two files, **that's a signal to extract it to a SoT module**.
This pattern saved us from at least three drift bugs in v23→v25.

## Two training paths

| | **MJX (PRIMARY)** | **SubprocVecEnv (LEGACY)** |
|---|---|---|
| env | `envs/hexapod_env_jax.py` (pure-JAX) + `envs/hexapod_brax_env.py` (Brax adapter) | `envs/hexapod_env.py` (gym, numpy + MuJoCo) |
| controller | `gait/controller_jax.py` (pure-JAX) | `gait/controller.py` (numpy) |
| BC pretrain | `scripts/pretrain_bc_jax.py` | `legacy/sb3/pretrain_bc.py` |
| training | `scripts/train_jax_amp.py` (Brax PPO + AMP on GPU) | `legacy/sb3/train.py` (SB3 PPO on CPU) |
| watch | `scripts/watch_demo_jax.py` (uses gym env for inference) | `legacy/sb3/watch_demo.py` |
| MJCF | `models/phantomx_simple_mjx.xml` | `models/phantomx.xml` |
| effective throughput | ~220-240k it/s end-to-end (v26+ restored after obs revert) | ~5,400 SPS |

**Important**: the gym env is now used ONLY for watch / eval / record.
Both envs construct obs via the shared `envs/obs_layout.py` module so
the gym env produces JAX-convention obs (qpos-based quat, qvel-based
gyro, zero accel) matching what the policy saw during training.

## AMP architecture (current, v23+)

- **Discriminator**: `amp.discriminator.MultiHeadDiscriminator` — 2-layer shared backbone (1024, 512) + 150 output heads (one per cmd bin). Inference selects the active bin's head; training routes per-bin gradient. Total ~1.55M params, ~6 MB on GPU.
- **Loss**: `multihead_discriminator_loss` — LSGAN + gradient penalty on prior batch, weight 10.0. Each bin's head gets gradient only from its-bin samples; shared backbone gets gradient from all.
- **Style reward**: `multihead_style_reward` — `max(0, 1 - 0.25 * (D(transition, bin) - 1)²)`. Bounded in [0, 1]. Added to env reward via `λ_style * style_r` where `λ_style = 0.5` default.
- **Prior format**: npz with `states_t` (N, 49) + `states_t1` (N, 49) + `cmds_t` (N, 9) + `bin_idx_t` (N,) — `bin_idx_t` is precomputed by `amp/prior_data.py` via `cmd_bins.cmd_to_bin()`. **Current production prior: `checkpoints/amp_priors_v26_small.npz`** (1.0M transitions, ~7k median per bin, ~430 MB GPU). v23 priors (6.1M, 2.6 GB GPU) still on disk for fallback. Smaller priors saves ~2.2 GB VRAM with no observed quality loss — disc only consumes ~13k touches per bin per full training run.
- **AMP state dim**: 49 (joint_pos 18 + joint_vel 18 + body_linvel_body 3 + body_angvel 3 + body_height 1 + foot_heights 6). Unchanged since v3. NOT affected by obs schema changes — obs is what the policy sees, AMP state is what the disc sees.

## Reward function (current, v26+ paper-pure)

In `envs/hexapod_env_jax.py:_compute_reward`. **Paper Table I (Liu et al. 2511.03167) form.** Active terms:

- **Positive — task tracking** (two per-axis exp() terms, paper convention `exp(-||err||²/σ)` NOT `/σ²`):
  - `r_lin_track = 1.0 * exp(-||v_xy_cmd - v_xy||² / 0.15)` — linear velocity tracking, max +1.0
  - `r_ang_track = 0.5 * exp(-(ω_z_cmd - ω_z)² / 0.15)` — yaw rate tracking, max +0.5
  - **Posture cmd components (pitch, roll, dh, dw) NOT tracked here** — AMP partition disc enforces them via cmd-bin routing. Was tracked in v25-/joint-gaussian, removed in v26.
- **Penalties** (all subtracted, all paper Table I weights):
  - `action_rate_pen` (w=0.01) — smoothness
  - `z_vel_pen` (w=1.0) — body bounce
  - `body_angvel_xy_pen` (w=0.08) — body twist
  - `joint_torque_pen` (w=2e-6) — torque magnitude
  - `joint_accel_pen` (w=1.5e-7) — **NEW v26**, was missing
  - `joint_vel_limit_pen` (w=0.5) — deadband+quadratic over motor max
  - `joint_torque_limit_pen` (w=0.05) — deadband+quadratic over stall
  - `foot_force_limit_pen` (w=0.1) — foot contact forces > 30 N
- **Zeroed in v26 (kept in metrics for log-parser compat)**:
  - `yaw_drift_pen`, `vy_drift_pen`, `vx_drift_pen` (w=0; not in paper, were fighting exploits AMP should solve)
  - `contact_mismatch_pen` (w=0; partition disc handles tripod gait via bin routing)
  - EMA filter on velocity tracking (`linvel_ema_alpha = 1.0` passes through instantaneous; was 0.05 in v25)
  - Legacy contact penalties from pre-AMP era

**Style reward** is added by `HexapodAMPEnv.step` separately, NOT in `_compute_reward`. `λ_style = 1.0` (was 0.5 in v25). Style component is also bounded ~+0.5/step → max total reward per step ≈ 2.0, max per 1000-step episode ≈ +2000.

**Critical convention**: legged-RL tracking reward divides squared error by `σ`, not `σ²`. Paper Table I notation `exp(||err||₂/0.15)` looks ambiguous but the legged_gym source confirms the convention. v26.0 had this wrong (`/σ²`); v27 fixes to `/σ`.

## Cmd vector (9 floats, physical units)

```
[0] vx           m/s     body forward velocity
[1] vy           m/s     body lateral velocity (left = +)
[2] wz           rad/s   body yaw rate (CCW = +)
[3] pitch        rad     body pitch target (nose up = +)
[4] roll         rad     body roll target (LEFT side up = +)
[5] height_delta m       stance height delta (- raises body)
[6] width_delta  m       stance width delta from neutral (+ wider)
[7] shift_x      m       body shift in body +X  (currently unused, fixed 0)
[8] shift_y      m       body shift in body +Y  (currently unused, fixed 0)
```

**Stance envelope** (verified 2026-05-10, owned by `envs/stance_envelope.py`):
- `dh ∈ [-0.045, +0.035]` m. 5 preset heights at -45, -25, -5, +15, +35 mm.
- `dw` is dh-conditional. Linear envelope: `max_dw(dh) = 0.0703 + 0.8333·dh`, `min_dw(dh) = -0.0283 + 0.25·dh`. JAX env's cmd sampler clips dw per dh.

## Multi-stage training pipeline (current view)

- ✅ **Stage 0** — BC pretrain (mimics scaffold; zero foot residual = scaffold motion). Current best: `bc_pretrained_jax_v26b` (78-dim obs).
- ✅ **Stage 1** — AMP-guided PPO with partition disc + paper-pure reward. Current best: latest `amp_to_v27*` (clean walker on best + final).
- ⏳ **Stage 2** — Heavy domain randomization (#82). Motor strength, friction, payload, joint stiffness, IMU noise. Sim-to-real critical.
- ⏳ **Stage 3** — Terrain randomization (#72). Perlin heightfield + obstacles. AMP's actual value-add over scaffold becomes visible here. Will likely need asymmetric A-C (#85) + memory encoder (#86) per paper.
- ⏳ **Stage 4** — PTQ int8 quantization (#101). ~50 KB deployable model.
- ⏳ **Stage 5** — ESP32-S3 firmware (#77). Policy inference + scaffold + servo bus + BT controller.

## Active task plan (post-2026-05-15)

**In execution order. See task list (#100+) for current.**

**Now / immediate:**
1. ✅ v27 walking — paper-pure + σ fix + smaller priors (DONE)
2. ⏳ Heavy DR (#82) — motor strength, friction, payload, sensor noise

**Phase 2 (after DR settles):**
3. ⏳ Asymmetric A-C with privileged critic (#85) — paper's terrain solution
4. ⏳ Memory/temporal encoder (#86) — 5-step proprio history latent, paper's terrain solution
5. ⏳ Terrain randomization (#72) — perlin heightfield + obstacles, regen priors on terrain

**Phase 3 (deployment prep):**
6. ⏳ PTQ int8 (#101) — int8 quantize the final policy
7. ⏳ QAT fallback (#102) — only if PTQ degrades quality

**Phase 4 (hardware, when bot arrives):**
8. ⏳ AX-12A endpoint calibration (#74)
9. ⏳ ESP32 firmware (#77)

**Side projects:**
- Mini hexapod chassis (separate subdir, when measurements available)

**Backlog / nice-to-have:**
- Distillation only if PTQ insufficient (#87)
- Pedagogical 3D gait visualizer (#64)
- Learning topic: num_envs vs PPO quality (#25)

## Important non-obvious things

- **Gait IK is closed-form and vectorized**. Sub-mm round-trip accuracy validated against MJCF FK. See `docs/kinematics.md`.
- **Per-joint sign conventions**: coxa always +1; femur +1 right / -1 left; tibia -1 right / +1 left. Don't "simplify" by negating all three for left legs — breaks the math.
- **Foot tip is empirical**: tibia-local `(0.134, 0.031, 0.0)`. The simple models replicate this with a foot sphere geom at the same position.
- **Foot z is normalized to uniform plane in body frame** (v21+). `gait/controller.py:calibrate()` sets `LEG_ORIGIN_BODY[:, 2] = mean(z)` so all 6 feet sit at exactly -145.99 mm in body frame. Fixes a 3mm asymmetry from MJCF coxa mount differences.
- **Foot sphere has ~7 mm radius**: stance detection from `geom_xpos` uses threshold `z < 12 mm`.
- **Pitch sign convention**: env's `_body_pitch_roll` extracts via standard math (positive = nose DOWN), then negates to match aerospace convention.
- **Body-side coxa convention**: same MJCF axis on R and L coxas, but legs mount on opposite body sides → +rotation swings the leg in OPPOSITE physical directions.
- **MJX requires Euler integrator + trimmed contact pairs**. `phantomx_simple_mjx.xml` uses `integrator="Euler"`, `iterations="20"`, contact bitmasks restricting to feet↔floor. Gives 200k+ it/s vs ~6k with defaults. Tried lowering iterations to 15 (2026-05-15) for more throughput — solver was already converging well below the cap, no measurable gain. 20 stays.
- **Tracking reward σ convention**: `exp(-||err||²/σ)`, NOT `exp(-||err||²/σ²)`. Squared error in numerator divides by σ (not σ²). Confirmed against `legged_gym/legged_robot.py` source. v26.0 had this wrong as `/σ²`, made tracking ~30× too tight at typical errors → bot got near-zero reward → couldn't learn. v27 fixes to `/σ`. When porting any tracking-reward formula from a paper, cross-check against legged_gym source — paper notation often elides the convention.
- **XLA flag verification gotcha**: Windows JAX is CPU-only and silently accepts GPU flag names with invalid enum values. Always test new XLA flags via `wsl bash -lc "XLA_FLAGS='...' ~/.venv-mjx/bin/python -c 'import jax; print(jax.devices())'"`. Burned by `--xla_gpu_enable_command_buffer=COMMAND_BUFFER` (passed Windows test, failed on WSL). Verified working on jaxlib 0.10: `--xla_gpu_enable_latency_hiding_scheduler=true`. Verified zero gain on our workload (tested + reverted): `--xla_gpu_enable_command_buffer=...`.
- **Brax 0.14.2 + JAX 0.10 incompat**: Brax uses `jax.device_put_replicated` which JAX 0.10 removed. `scripts/train_jax_amp.py` shims it before importing brax.
- **PowerShell `>>` continuation breaks `wsl bash -lc "..."`**: paste the whole invocation as ONE physical line.
- **Always use `python -u` when piping training stdout through `tee`**: block-buffered pipes can swallow hours of training output before flushing.
- **The hovering exploit (pre-v23 era)**: with airborne/contact penalties disabled, the policy maximizes velocity-tracking by hovering. v22+ partial fix via cmd-conditional disc, v25 explicit fix via contact_mismatch_pen.
- **Reward hacking is the dominant failure mode.** Every reward shaping the policy will exploit if there's a degenerate strategy that scores well. v14/v15 hit it (high reward, no walking). v24 hit it again (better metrics, worse walking). v25's EMA + contact penalty is the structural fix.
- **Watcher auto-detects policy obs dim** from the normalizer mean shape. Pre-v24 policies (78-dim obs) work in the new 114-dim env via `obs[:78]` slicing. See `scripts/watch_demo_jax.py:main()`.

## Files at a glance (post-2026-05-13)

### Active MJX pipeline → `scripts/`

| Path | Purpose |
|------|---------|
| `scripts/train_jax_amp.py` | Brax PPO + AMP training. `--partition-disc`, `--knn-k`, `--style-weight`, `--recovery-curriculum` (shelved). |
| `scripts/pretrain_bc_jax.py` | BC pretraining. Bakes `log_std=-4.0`. Default `n_envs=2048` for VRAM fit. |
| `scripts/chain_train.py` | Chain orchestrator + shared session config (`BASE_NAME`, `CMD_MASK`, etc). |
| `scripts/watch_demo_jax.py` | Render JAX/Brax policy through preset cmd phases. Auto-detects policy obs dim, exits cleanly on viewer close. |
| `scripts/watch_controller.py` | 8BitDo / Xbox controller-driven inference. Falls back to watch_demo_jax `--interactive` if no controller. |
| `scripts/record_policy.py` | Offscreen MP4 render of policy + tracking + orbit camera. |
| `scripts/eval_bc_quick.py` | Quantitative eval — episode reward, tracking, fall rate. |
| `scripts/demo_phases.py` | Shared 17-phase preset cmd script + interactive tests 1-9. Uses `envs/stance_envelope.py` for height/width cycles. |
| `scripts/controller_mapping.py` / `calibrate_controller.py` / `test_controller.py` | Bluetooth controller wiring. |

### AMP modules → `amp/`

| Path | Purpose |
|------|---------|
| `amp/prior_data.py` | Collect (s_t, s_{t+1}, cmd_t, bin_idx_t) from scaffold rollouts. Saves to npz. |
| `amp/discriminator.py` | `Discriminator` (vanilla) + `MultiHeadDiscriminator` (v23+ partition). LSGAN loss + gradient penalty. |

### Env modules → `envs/`

| Path | Purpose |
|------|---------|
| `envs/hexapod_env_jax.py` | JAX/MJX-native env (functional reset/step). Owner of cmd sampling, reward function, episode termination. |
| `envs/hexapod_brax_env.py` | Brax `Env` adapter wrapping the JAX env. |
| `envs/hexapod_amp_env.py` | `HexapodBraxEnv` + AMP style reward via frozen disc. |
| `envs/hexapod_env.py` | Gym env. Used ONLY for watch / eval / record. Shares obs layout with JAX env. |
| **`envs/obs_layout.py`** | **SoT**: 114-dim obs schema (joint_pos, joint_vel, imu_quat, imu_gyro, imu_accel, scaffold_hint, phase_sc, cmd, body_linvel, joint_torque, joint_pos_error). |
| **`envs/stance_envelope.py`** | **SoT**: dh range + dh-conditional dw envelope. |
| **`envs/cmd_bins.py`** | **SoT**: 150-bin partition for the multi-head discriminator. |

### Gait / kinematics → `gait/`

| Path | Purpose |
|------|---------|
| `gait/controller.py` | Numpy gait controller. Normalizes foot z to uniform plane (v21+). |
| `gait/controller_jax.py` | Pure-JAX port. Calibrates via numpy `Controller` then copies constants. |

### Models → `models/`

| Path | Purpose |
|------|---------|
| `models/phantomx_simple_mjx.xml` | MJX training MJCF. Primitive geoms, Euler, feet↔floor only. |
| `models/phantomx_simple.xml` | Primitive-geom MJCF with full self-collision (RoM derivation). |
| `models/phantomx.xml` | Mesh-based MJCF (visualization, NOT training). |

### Tools → `tools/`

| Path | Purpose |
|------|---------|
| `tools/watch_scaffold.py` | Render scaffold gait directly via `Controller.predict()`. `--interactive` mode for stance tuning (keys 1-6). Uses `envs/stance_envelope.py` for height presets. |
| `tools/analyze_cmd_distances.py` | Diagnostic: K-NN structure of priors in cmd space. Was used to design the 150-bin partition. |
| `tools/to_parallel_sweep.py` | TO sweep harness — parallel solve of many cost-weight / gait-param variations. |
| `tools/to_solver.py` / `tools/trajectory_opt_demo.py` | CasADi+IPOPT TO solver. Supports `linear_solver=ma27` via `tools/_hsl_bootstrap.py`. |
| `tools/bench_hsl.py` | MUMPS vs MA27 benchmark on representative TO problem (MA27 is worse for our problem; stay on MUMPS). |
| `tools/derive_joint_limits.py` / `tools/apply_joint_limits.py` | Auto-derive joint RoM + patch into MJCFs. |
| `tools/calibrate_scaffold_speed.py` | 49-combo `(period, path_radius)` sweep under physics. |

### Legacy / standalone

- `legacy/sb3/` — old SubprocVecEnv stack (kept for Mac-side use).
- `legacy/sandboxes/` — old visualizer experiments.
- `vendor/coinhsl/` — HSL Archive (gitignored, per-user STFC license).

### Docs / data

`docs/` (kinematics, papers, MAX_SPEED analysis), `research/notes/` (hyperresearch vault), `checkpoints/` (run artifacts), `logs/` (active tensorboard + stdout subdirs), `legacy/logs/` (archived runs), `snapshots/`, `media/recordings/`, `joint_limits.json`.

## Directory placement conventions

**Where new files go.** Project root is top-level config only (CLAUDE.md, README, requirements, .gitignore, LICENSE, joint_limits.json).

| Kind | Goes in |
|---|---|
| Training script | `scripts/train_*.py` or `pretrain_*.py` |
| Watch / demo / control script | `scripts/watch_*.py`, `scripts/record_*.py`, etc. |
| One-off diagnostic / measurement tool | `tools/` |
| Env / sim modules | `envs/` |
| **Schema/config single source of truth** | `envs/<name>.py` — see SoT pattern above |
| Gait math / IK | `gait/` |
| AMP infra | `amp/` |
| MJCF, meshes, joint config | `models/` |
| Recorded videos / GIFs / screenshots | `media/recordings/` |
| Active tensorboard event files | `logs/<run_name>/` |
| **Old / failed run logs** | `legacy/logs/` (OUT of `logs/` so tensorboard ignores) |
| stdout `.log` files | `logs/stdout/` |
| Documentation | `docs/` |
| Hyperresearch notes | `research/notes/` |
| Checkpoint dirs | `checkpoints/<run_name>/iter*/final/{params,discriminator}.pkl` |
| HSL / third-party binaries (gitignored) | `vendor/` |

**Never put in root**: recorded videos, tensorboard runs, one-off test scripts, temporary `.log` files, generated artifacts.

## Working-style preferences (apply to all conversations)

**One question at a time.** When you have multiple design questions for the user, ask ONE AT A TIME and wait for the answer.

**Trust the user on validation.** When the user reports a feature "works" or "feels right," accept it and move on.

**Bite-sized teaching.** When the user wants to understand something new, present ONE focused idea per reply, define acronyms inline (e.g., "PPO = Proximal Policy Optimization"), and pause for confirmation before continuing.

**Snapshot before risky refactors.** Copy current working file into `snapshots/` before invasive changes.

**No-redundant-IK rule for env hot paths.** Both gym and JAX env step functions use `predict_with_feet(cmd, t)` which returns BOTH joint targets AND body-frame foot positions in one gait-pipeline pass.

**SoT pattern for cross-file constants/schemas.** If a constant or formula appears in more than one env / script / tool, extract it to a `envs/<name>.py` single-source-of-truth module. See the three existing ones (obs_layout, stance_envelope, cmd_bins).

**Background long training, foreground quick diagnostics.** `Bash` with `run_in_background=true` for any run > 60s. Always `tee` output to `logs/stdout/<run>.log`. Use `ScheduleWakeup` for periodic check-ins on running training.

## User's machines

### Workstation (Windows 11) — primary for MJX training

- **Hardware**: AMD Ryzen 7 7800X3D, 32 GB RAM, RTX 5060 Ti (16 GB VRAM).
- **Project path**: see "Path note" at top of this file. Project was moved off OneDrive 2026-05-13.
- **Two Python environments**:
  - **Windows venv** (`.venv\`): SB3, numpy, mujoco, JAX-CPU, Brax. Hosts watch scripts (uses mujoco.viewer). Run via `.venv\Scripts\python.exe scripts\<file>.py` with `$env:PYTHONPATH="."`.
  - **WSL venv** (`~/.venv-mjx/` inside Ubuntu WSL2): JAX-CUDA + MJX + Brax. Required for training. Invoke via `wsl bash -lc "cd '/mnt/c/<path>' && PYTHONPATH=. ~/.venv-mjx/bin/python -u scripts/<file>.py"`.
- **GitHub**: `gh` CLI installed via winget. Repo at github.com/Eth4ck1e/hexapod-ai.

### M3 Max MacBook Pro (legacy / SubprocVecEnv only)

JAX-Metal can't run MJX (Cholesky op missing). Mac runs only the legacy SB3 stack at ~9k SPS.

### M1 Max Mac Studio — third machine, less commonly used.

## AMP background (the foundational paper — see `docs/papers/`)

The paper "Learning Natural and Robust Hexapod Locomotion" (Chen et al, SJTU) trains a real PhantomX-style hexapod via PPO + AMP. Key ideas:

- **Adversarial Motion Priors** (AMP): a discriminator network learns to distinguish "real motion" (from a prior dataset) from policy output. The discriminator's score becomes a continuous "naturalness reward" that pushes the policy toward the prior's style.
- **Three-part reward**: task tracking (gaussian on cmd-vs-actual velocity) + style (AMP disc score) + paper penalties.
- **Trajectory optimization (TO)** generates the prior dataset. We use our scaffold instead — same purpose, different generation method, way more data per cmd region.
- **Networks**: asymmetric Actor-Critic + memory encoder + state estimator. Their critic gets privileged info (terrain); the actor gets only proprioception. We're starting simpler (symmetric A-C, no memory encoder yet — both in backlog).

**Our diverge from the paper**:
1. **Partition discriminator** instead of single conditional. The paper's conditional disc still permits cross-cmd blur; our 150-bin partition forces strict same-bin comparison and fixed v22d's "scores well but doesn't walk" failure mode.
2. **Motor feedback obs** (joint_torque + joint_pos_error). The paper doesn't expose these to the policy; we found they're necessary for the policy to know when feet are loaded — without them, contact-aware rewards are useless and recovery / terrain training is fundamentally limited.

## Lessons learned (universal RL principles)

- **Reward hacking is the dominant failure mode.** Every reward function the policy will exploit. v14/v15 (hovering), v22d (recovery training degraded walking), v24 (gameable velocity tracking despite better metrics). Each time the fix was structural (AMP, partition disc, EMA tracking), not penalty tuning.
- **Metrics can lie.** v22d hit +666 eval, v24 hit +740 — both visually broken. Always watch alongside reading metrics.
- **PPO at default `log_std` destroys a working BC walker.** Now baked into `pretrain_bc_jax.py`: `log_std=-4.0` (~1° per-joint jitter).
- **Reward gaussian's exponential falloff is harsh.** Total squared error of 5.0 collapses tracking from ~1.0 to ~0.007.
- **Same seed → near-identical reward trajectories across env changes.** v20, v21, v22 all followed nearly identical training curves because the seed dominates the trajectory through param space. To get meaningfully different policies, change the seed OR change something that affects the loss landscape (architecture, reward shape).
- **Spawn-state DR matters.** v10 added joint-angle ±5° and body-z ±10mm at reset; helped generalization without hurting walking.
- **AMP only earns its keep when scaffold isn't optimal.** On flat ground with no DR, BC seed = scaffold = a local optimum AMP can't improve. AMP's actual value-add appears under terrain / DR / disturbances where the scaffold breaks.

## MJX / hardware findings (historical reference)

- **CUDA MJX (5060 Ti via WSL2)**: ~377k SPS on Brax Ant. Real-world end-to-end with our hexapod env + 114-dim obs + partition disc: ~200k it/s.
- **Mac (JAX-Metal)**: blocked; Cholesky op missing. CPU MJX on M3 Max is 1.8× SPS — not worth it.
- **Memory pressure on 5060 Ti**: 16 GB VRAM is tight with 114-dim obs + 150-head disc. Constants `NUM_ENVS=2048`, `DISC_BATCH=512`, `N_ENVS_DEMO=2048`. Going larger triggers RESOURCE_EXHAUSTED in the disc gradient penalty.

## Patterns developed in workstation sessions

- **Smoke-test after env edits**: one-shot `python -c "..."` that resets and steps the env before declaring a change done.
- **Filesystem-based run monitoring**: check `checkpoints/<run>/iter*/` mtimes to confirm a run is alive.
- **Always launch long training via WSL with `python -u` and `tee`** to a stdout log; check on it with `ScheduleWakeup`.
- **`HexapodAMPEnv` constructor must be called from inside the segment loop** each iteration with the updated discriminator params — that's how the disc-updates propagate to the next PPO segment.
- **Numerical-equivalence + round-trip validation** on every IK refactor: random cmd → predict → mj_forward → measured foot. Sub-mm error means correctness preserved.

## Memory / sync

This file is the shared context. The local `.claude/memory/` directory holds machine-local memory which is more detailed but does NOT travel with the repo.

<!-- hyperresearch:start -->
## Research Base (hyperresearch)

**CLI path: `<project_root>/.venv/Scripts/hyperresearch.exe`** — substitute the actual absolute path at invocation. Was previously hardcoded to OneDrive location; now generic so the directory move doesn't break it. May not be on system PATH.

**Paths in this document are relative to your current working directory**, not to the CLI binary's location. Use `research/notes/final_report_<vault_tag>.md` when you save files.

This project uses hyperresearch as an agent-driven research knowledge base. The `research/` directory contains markdown notes collected from web sources and original research. Append `--json` to any command for structured output.

### How to do research

**Run a research session with `/hyperresearch <query>`.** This invokes the V8 16-step pipeline. The entry skill at `.claude/skills/hyperresearch/SKILL.md` is a thin ROUTER. Step procedures live in their own skills (`hyperresearch-1-decompose` through `hyperresearch-16-readability-audit`) and are loaded fresh into context via the `Skill` tool when each step runs.

Step 1 classifies the query into one of two tiers (`light` or `full`) and the pipeline scales accordingly — short bounded queries skip depth investigations / critics / patcher (~30-40 min); argumentative deep-research queries run all 16 steps with adversarial review (~1.5-2.5 hours).

**Do NOT use WebFetch for source pages** — use `hyperresearch.exe fetch` instead. The skill files explain when to fetch vs. search.

### Academic APIs before web search

For any topic with a research literature, hit academic APIs BEFORE web searches:

- **Semantic Scholar:** `https://api.semanticscholar.org/graph/v1/paper/search?query=<q>&fields=title,year,citationCount,externalIds&limit=10`
- **arXiv:** `https://export.arxiv.org/api/query?search_query=cat:cs.LG+AND+all:<q>&sortBy=relevance&max_results=25`
- **OpenAlex:** `https://api.openalex.org/works?search=<q>&sort=cited_by_count:desc&per-page=15&mailto=research@example.com`
- **PubMed:** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=<q>&retmode=json&retmax=20`

After the academic sweep, run web searches for context + at least one adversarial search.

### PDFs fetch directly

`hyperresearch.exe fetch` auto-detects PDF URLs (arXiv, NBER, SSRN, direct `.pdf`) and extracts full text via pymupdf. Raw PDFs land in `research/raw/<note-id>.pdf` and the note's frontmatter links back via `raw_file:`.

### Searching the vault

```bash
hyperresearch.exe search "query" --json                 # Full-text search
hyperresearch.exe search "query" --tag ml --json        # Filter by tag / status
hyperresearch.exe search "query" --include-body --json  # Full-body search
hyperresearch.exe note show <id> --json                 # Read one note
hyperresearch.exe note show <id1> <id2> --json          # Batch-read
hyperresearch.exe note list --json                      # List all notes
hyperresearch.exe tags --json                           # Tag vocabulary
```

### Authenticated crawling

Login-gated content needs a browser profile. Set up once via `hyperresearch.exe setup` or `crwl profiles`. Config in `.hyperresearch/config.toml` under `[web]`: `profile = "research"`, `magic = true`.

### Curate after every session

```bash
hyperresearch.exe note list --status draft -j
hyperresearch.exe note show <id> -j
hyperresearch.exe note update <id> --summary "<specific summary>" --add-tag <t> -j
hyperresearch.exe lint -j
hyperresearch.exe repair -j
hyperresearch.exe status -j
```

Lifecycle: `draft` → `review` → `evergreen` (or `stale` → `deprecated` → `archive` for outdated).

### Key conventions

- Notes live in `research/notes/` as markdown with YAML frontmatter
- Link notes with `[[note-id]]` syntax
- After editing `.md` files directly, run `hyperresearch.exe sync` to update the index
<!-- hyperresearch:end -->
