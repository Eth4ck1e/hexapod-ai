# Project context for Claude Code

This file is auto-read by Claude Code at session start. It carries the
design decisions, current state, and working-style preferences that the
local `.claude/` memory system captures, so any Claude instance opening
this repo on any machine has the context. Edit freely if anything below
becomes stale.

## Project goal

Reinforcement-learning locomotion controller for a PhantomX-style
hexapod, deployed eventually on an ESP32-S3 microcontroller. PPO trains
a residual policy on top of a working analytical gait scaffold; the
scaffold fades out across a 4-stage curriculum so the final policy is
fully autonomous and able to learn emergent behaviors (max-speed
exploration, rolling tricks) beyond what the analytical gait can do.

## Current state (2026-04-29)

Gait library uses **closed-form, vectorized IK** with full overlay
support (stance width/height, body tilt, spin). All gait math validated
sub-millimeter against MJCF FK. The `IK_gait.py` sandbox tested every
overlay independently (smart test script with 19 phases covers them
all). Training pipeline integrated and benchmarked at the workstation's
sweet spot. Long-form curriculum training runs reasonable.

**Best policy: BC v2** (`checkpoints/bc_pretrained_v2/policy.zip`,
gitignored). Trained via supervised behavioral cloning on 20M scaffold
demonstrations across stage=3 (full motion). Walks indefinitely at
clean cmds, completes ~70% of stochastic stage=3 episodes — failures
are `no_progress` at near-extreme combined cmds (real operational
limit, not a bug). RL fine-tuning attempts (v3, v6) have so far ended
worse than BC v2 because PPO's stochastic noise destabilizes the
working walker faster than it can find improvements. See "Methodology
learnings" below for the noise-vs-refinement tradeoff.

**Unified cross-platform stack (2026-04-29 — was a Mac/Linux split).**
Single `envs/hexapod_env.py` and `train.py` work on both platforms. The
viewer-mechanism difference that originally forced the split (macOS's
Cocoa viewer can only run under mjpython, but SubprocVecEnv workers
run plain `python`) is resolved by routing live-watch through shared
memory on both platforms: env-0 publishes qpos+qvel+sim_time to the
`hexapod_live_state` SHM region every step, and a separate viewer
process reads it. On macOS run that viewer with `mjpython
live_viewer.py`; on Linux just `python live_viewer.py`.

The previous Mac variant carried all the recent improvements (drift
penalty, polar-sampled translation, contact shaping, foot deviation,
no_progress termination, BC infrastructure, pitch sign fix, etc.).
The unified version is that codebase — the older Linux env/train were
deleted as obsolete. Snapshots in `snapshots/*_pre_unify.py` preserve
the pre-unification state.

All scripts (`watch.py`, `watch_demo.py`, `pilot_ai.py`, `pretrain_bc.py`)
import directly from `envs.hexapod_env` — no platform-detect needed.

## Architecture (locked decisions)

- **`gait/` package** is the single source of truth for gait math.
  `Controller` class: stateful, `predict(cmd, t) → 18 joint targets`.
  Used by `simple_gait.py` (demo viewer), `IK_gait.py` (sandbox),
  `pilot.py` (teleop), `envs/hexapod_env.py` (training), eventually
  ESP32-S3 firmware.
- **Cmd vector** (9 floats, physical units):
  ```
  [0] vx           m/s     body forward velocity
  [1] vy           m/s     body lateral velocity (left = +)
  [2] wz           rad/s   body yaw rate (CCW = +)
  [3] pitch        rad     body pitch target (nose up = +)
  [4] roll         rad     body roll target (LEFT side up = +; standard
                                              Euler — matches what
                                              env's _body_pitch_roll
                                              computes from the quat)
  [5] height_delta m       stance height delta (- raises body)
  [6] width_delta  m       stance width delta from neutral (+ wider)
  [7] shift_x      m       body shift in body +X       (RESERVED — task #8)
  [8] shift_y      m       body shift in body +Y       (RESERVED — task #8)
  ```
- **Reward (minimal)**: `tracking_exp(-||cmd-actual||²) + 0.1*survive
  - 0.01*action_delta² - 0.02*body_angvel² + (stage 4 only) novelty`.
  Errors are normalized per-slot by `_err_inv_scales` so each axis
  contributes commensurately. Tracking compares ALL body kinematics
  every step regardless of which stage the bot is in — locked cmd
  slots are zero, so any actual deviation creates real tracking error.
  This is what makes the bot learn smooth controlled locomotion vs
  jerky lurching that happens to track velocity.
- **Curriculum / training modes** (in `train.py`):
  Two supported modes — pick via `INITIAL_STAGE` and `SKIP_SCAFFOLD`
  constants at the top of `train.py`:
  - **BC-init refinement** (current default; `INITIAL_STAGE=3`,
    `SKIP_SCAFFOLD=True`). Loads BC-pretrained policy via `--bc-init`,
    runs single-stage at gait_scale=0.0 throughout (scaffold suppressed).
    All cmd slots active from step 0 (translation, yaw, height, width,
    pitch, roll). Recommended for refining a working walker. Pair with
    `--log-std-init -3.0` for refinement noise calibration.
  - **From-scratch curriculum** (`INITIAL_STAGE=1`, `SKIP_SCAFFOLD=False`).
    Original 4-stage mode that progressively unlocks cmd slots and
    fades the scaffold:
    - Stage 1: vx, vy. `gait_scale=1.0` throughout.
    - Stage 2: + wz, height, width. `gait_scale 1.0 → 0.6`.
    - Stage 3: + pitch, roll. `gait_scale 0.6 → 0.0`.
    - Stage 4: pure policy, novelty bonuses, body_linvel dropped from obs.
    - Adaptive advance: tracking ≥ 0.65 AND fall rate ≤ 0.30 AND min
      step count reached.
    Per-stage step budgets (`STAGE_MIN_STEPS`) and `gait_scale`
    envelope (`STAGE_GAIT_SCALE`) live in `train.py` and have been
    iterated several times. Current values reflect "scaffold is good
    enough that early fading is safe."
- **Default stance widening**: `Controller.DEFAULT_STANCE_WIDTH = 0.015`
  (+15 mm outward per foot). The widened rest gives a more vertical
  foot-ground contact angle and generally walks better than the
  calibrated rest. `Controller.gait_neutral_pose` exposes the joint
  pose at this widened rest; env uses this for spawn settle (not the
  hardcoded `NEUTRAL_POSE`, which corresponds to the un-widened rest
  and is just the IK calibration anchor).
- **Deployment plan**: analytical gait library ships to ESP32-S3 as a
  standalone "manual mode" AND as a safety fallback under the policy.
  Three runtime modes: manual / ai / assisted.

## Important non-obvious things

- **Gait IK is closed-form and vectorized** (was iterative `mj_jac`
  during the first integration; replaced after benchmarking exposed
  it as the dominant cost). Sub-mm round-trip accuracy validated
  against MJCF FK across the full reachable workspace. See
  `docs/kinematics.md` for the derivation. The closed-form path uses
  per-leg formula→MJCF offsets calibrated once at boot to handle the
  MJCF's non-trivial femur/tibia body quaternions and per-side axis
  flips.
- **Per-joint sign conventions** for the closed-form IK are
  unintuitive: coxa always +1; femur +1 right / -1 left; tibia
  -1 right / +1 left (femur and tibia are flipped between right and
  left, and the formula's bend convention is opposite of MJCF tibia's
  positive rotation, so right-side tibia gets -1 by itself even
  before the side flip). Don't try to "simplify" by negating all
  three for left legs — that breaks the math.
- **Foot tip is empirical, not (0, 0, -TIBIA_LENGTH).** The PhantomX
  tibia mesh has its tip at tibia-local `(0.134, 0.031, 0.0)`,
  magnitude 0.138 m. `TIBIA_LENGTH = 0.138`, NOT 0.133. Using 0.133
  in the IK formula breaks correctness off the rest pose. The +0.031
  Y component is also baked into the offset table — this is why the
  tibia formula→MJCF offset comes out to ≈ -1.346 rad (= 90° + 13°,
  the angular offset from "tibia along straight extension" to "MJCF
  tibia=0 link direction").
- **Speed-magnitude variation: polar sampling, clamped magnitude.**
  Translation cmd is sampled in polar (heading, magnitude) form: heading
  uniform in [0, 2π), magnitude in `[SPEED_MIN_FRAC, SPEED_MAX_FRAC] ×
  MAX_SPEED` (currently `[0.40, 0.85]`). The lower bound avoids the
  "stand still" failure mode (magnitude ≈ 0 episodes encourage do-
  nothing policies); the upper bound avoids the "ragged edge" mode
  (magnitude = MAX_SPEED sits at scaffold capacity, looks ugly). The
  Controller still models speed via stride-scaling only (no period
  modulation yet); the `IK_gait.py` sandbox has the period+stride
  proof-of-concept for when we eventually want a wider envelope.
- **macOS viewer requires `mjpython`**, not plain `python`, for any
  script that calls `mujoco.viewer.launch_passive`. Affects
  `simple_gait.py`, `IK_gait.py`, `pilot.py`, `pilot_ai.py`, `watch.py`,
  `watch_demo.py`, `watch_tiled.py`, `live_viewer.py`. Headless scripts
  (`train.py`, `pretrain_bc.py`, `bench_n_envs.py`) use plain `python`.
  Linux has no such restriction — every script just uses `python`.
- **`WATCH_LIVE=True`** publishes env-0's qpos+qvel+sim_time to the
  `hexapod_live_state` shared-memory region every step. Run
  `live_viewer.py` (mjpython on macOS, python on Linux) in a separate
  terminal to render. **Don't try to launch `train.py` itself with
  mjpython expecting the workers to inherit it — they don't, because
  mjpython is a shell wrapper around the underlying python interpreter,
  and `sys.executable` in the SubprocVecEnv workers points at python.**
- **`pilot.py` keyboard layout collides with mujoco viewer hotkeys.**
  Accepted as fine for testing only. Long-term: gamepad support via
  pygame; gamepad axes don't conflict.
- **Pitch sign convention** (resolved 2026-04-29). The env's
  `_body_pitch_roll` extracts pitch via standard right-hand-rule math
  (positive = nose DOWN). The cmd convention and the gait controller
  use aerospace (positive = nose UP). The two were inconsistent for
  multiple sessions, surfacing as "scaffold can't track tilt" / "BC
  policy tilts wrong way" — neither was the actual bug. Fixed by
  negating `asin()` in `_body_pitch_roll`. Roll has no analogous bug
  (X-axis rotation conventions agree).
- **`StageManagerCallback` must use `env_method`, NOT `set_attr`** to
  mutate `gait_scale` / `stage` on the VecEnv. SB3's `set_attr` only
  mutates the outer `Monitor` wrapper; gym wrappers don't propagate
  setattr to the wrapped env. We added `set_gait_scale()` /
  `set_stage()` methods on `HexapodEnv` and the callback uses
  `env_method("set_gait_scale", value)` to reach the inner env. Older
  set_attr-based code silently failed — entire training runs were at
  the initial gait_scale value despite TB recording the intended
  schedule.
- **Legacy demo scripts are now broken**: `sandbox.py`, `gait_design.py`,
  `walk_test.py` import deleted constants from the old env. They've
  been superseded by `simple_gait.py`, `IK_gait.py`, `pilot.py`. Safe
  to delete; left in repo for now as historical reference.
- **Snapshots** in `snapshots/` are historical versions of important
  files captured before risky refactors (e.g., `*_v1_pre_integration.py`
  are the pre-IK-refactor controller, env, and IK_gait state). Useful
  for revert paths.

## Files at a glance

| Path | Purpose | Platform |
|------|---------|----------|
| `gait/controller.py` | Stateful gait Controller. Closed-form vectorized IK. | both |
| `gait/__init__.py` | Public API exports (Controller, link lengths, NEUTRAL_POSE, etc.). | both |
| `simple_gait.py` | Demo viewer — toggle cycles for any cmd slot via Controller. | both |
| `IK_gait.py` | Sandbox with all overlays + smart-test script (19-phase exercise). | both |
| `pilot.py` | Keyboard teleop driving the analytical scaffold directly. | both |
| `pilot_ai.py` | Keyboard teleop driving a trained PPO checkpoint. Hold Shift = run mode (2× speed). | both |
| `envs/hexapod_env.py` | Cross-platform gym env. SHM-based live-state mirror, drift penalty, contact shaping, foot deviation, no_progress termination, BC `info["bc_target"]` exposure. | both |
| `train.py` | Cross-platform PPO training. Flags: `--resume`, `--bc-init`, `--log-std-init`. Config knobs: `INITIAL_STAGE`, `SKIP_SCAFFOLD`, `WATCH_LIVE`. | both |
| `pretrain_bc.py` | Supervised behavioral cloning of scaffold demonstrations. Produces a PPO-compatible policy that mimics the scaffold's joint targets. `--stage`, `--steps`, `--epochs`, `--out` flags. | both |
| `live_viewer.py` | Reads `hexapod_live_state` shm and renders — pair with `train.py`. macOS needs `mjpython`; Linux uses plain `python`. | both |
| `bench_n_envs.py` | SubprocVecEnv N_ENVS sweep — finds CPU sweet spot. | both |
| `watch.py` | Render trained checkpoint with random env-sampled cmds. Settles to neutral first. | both |
| `watch_demo.py` | Render trained checkpoint through a 23-phase preset cmd script. Settles first. | both |
| `watch_tiled.py` | N-up grid render of trained checkpoint. | both |
| `models/phantomx.xml` | MJCF model. Mesh path fixed to `./meshes/phantomx`. | both |
| `docs/kinematics.md` | IK derivation, leg geometry, MJCF axis conventions. | both |
| `snapshots/` | Historical versions before refactors (revert points). | both |

## Working-style preferences (apply to all conversations)

**One question at a time.** When you have multiple design questions
for the user, ask ONE AT A TIME and wait for the answer before asking
the next. Do not send numbered lists of questions. The user explicitly
stated this is "by far my most preferred method." Single-question
iteration produces better answers and faster decisions than batched
question lists.

**Trust the user on validation.** When the user reports a feature
"works perfectly" or "feels right," accept it and move on. Don't
re-litigate or run more verification than needed.

**Snapshot before risky refactors.** Before any large-scale rewrite,
copy the current working file into `snapshots/`. Already-snapshotted
versions live there as historical references.

**No-redundant-IK rule for the env hot path.** `HexapodEnv.step` uses
`Controller.predict_with_feet(cmd, t)` which returns BOTH joint targets
AND body-frame foot positions in a single gait-pipeline pass, then
caches `feet_body` for `_get_obs()` to read. Don't reintroduce
separate `predict()` + `compute_foot_targets()` calls in the same
step — that re-runs the whole gait pipeline twice for no gain.

## User's machines

The user has three machines, and which one is "primary" depends on the
session. Most day-to-day development and training happens on the M3 Max
MacBook Pro — the workstation is for separate sessions when the user is
at the Windows desktop. The M1 Max Mac Studio is a third machine, less
commonly used for training.

### M3 Max MacBook Pro (primary dev + training, current session)

- **Hardware**: Apple M3 Max (up to 16-core CPU, up to 40-core GPU,
  unified memory ~400 GB/s). Faster than the workstation for the
  hexapod RL workload by roughly 2× — the unified memory architecture
  removes the CPU↔RAM bottleneck that bounds `mj_step` throughput on
  discrete-memory systems. SubprocVecEnv runs scale better here than on
  the workstation despite the workstation's 96MB L3.
- **Project path**: `/Users/mitchelltrafford/Documents/hexapod-ai`
- **Python**: `.venv/` with Python 3.11. Activate via
  `source .venv/bin/activate`, or call `./.venv/bin/python` directly.
- **Shell**: zsh.
- **macOS specifics**: viewer needs `mjpython` for `launch_passive` (see
  per-platform-split note above).
- **GPU/MJX potential**: when the user wants to commit to a faster-
  iteration training stack, MJX (MuJoCo XLA) on JAX-Metal would run
  physics rollouts on the M3 Max's 40-core GPU. Realistic 50–100×
  wall-clock speedup over current SubprocVecEnv. Non-trivial port: the
  gait controller's IK and the env step would have to be rewritten in
  pure JAX (no Python control flow that depends on values, no NumPy).
  Status as of 2026-04-27: not started; user is iterating reward
  shaping first — don't port a moving target.

### Workstation (Windows 11 desktop)

The user's secondary training machine when they're at the Windows
desktop, with the project at
`C:\Users\Eth4ck1e\OneDrive\Documents\Hexapod AI Project`.

- **Hardware**: AMD Ryzen 7 7800X3D (8C/16T, 96 MB L3 V-Cache,
  AVX-512), 32 GB RAM, RTX 5060 Ti. The 96 MB L3 fits MuJoCo's per-env
  physics working set; outpaces the M1 Max Studio for SubprocVecEnv
  runs but is slower than the M3 Max MacBook Pro by roughly 2×.
- **GPU note**: SB3 PPO on the 5060 Ti is *slower* than CPU PPO on the
  same machine (~2,750 vs 5,000+ SPS) because the MLP policy is small
  enough that GPU dispatch overhead dominates gradient compute. Don't
  set `device="cuda"` for SB3 PPO. The 5060 Ti would only be useful if
  the user ports to MJX (physics on GPU) — not just the policy.
- **Python**: `.venv/` with Python 3.11. Activate via
  `.\.venv\Scripts\Activate.ps1`, or call `.\.venv\Scripts\python.exe`
  directly. System `python` on PATH is a different interpreter and
  lacks project deps — always use the venv.
- **Shell**: Claude defaults to bash (Git for Windows / MSYS), not
  PowerShell. Unix-style paths reach Windows drives via `/c/...`.
- **GitHub**: `gh` CLI installed via winget at
  `C:\Program Files\GitHub CLI\gh.exe`. HTTPS auth through gh; token
  in the Windows keyring. New shells pick up `gh` on PATH;
  mid-session shells need the full path. Repo is at
  github.com/Eth4ck1e/hexapod-ai.

## Training throughput on the workstation

Optimization history (single-env step rate, headless):
- Pre-refactor (legacy single-file gait/IK):  ~5,000 SPS
- Post-refactor with iterative `mj_jac` IK:     ~62 SPS  (massive regression)
- Closed-form looped IK + vendored meshes:    ~5,200 SPS (recovered)
- + `predict_with_feet` (no double pipeline): ~5,500 SPS
- + Vectorized IK + frame transforms:         ~5,375 SPS  (gait math <60 µs/call)

End-to-end env step is dominated by `mj_step` + reward construction,
not gait math, so further gait optimization has diminishing returns.

**`N_ENVS = 32` benchmarked as the sweet spot** for this CPU
(8 physical cores × SMT 2 = 16 logical, but lighter per-step compute
after the IK refactor leaves enough idle cycles to oversubscribe to
24-32 envs profitably). Run `python bench_n_envs.py` to re-confirm
if anything changes about the workload — sweeps 16/20/24/28/32 in
~5 minutes.

**`device="cpu"` for PPO**, always. SB3 warns about GPU for small MLP
policies and the warning is correct: GPU dispatch overhead dominates
the small gradient. We measured GPU at ~2,750 SPS vs CPU at ~5,000+
SPS on this same machine.

**Live observability** (in `train.py`):
- `WATCH_LIVE = True` makes env 0 publish `qpos+qvel+sim_time` every
  step to the `hexapod_live_state` shared-memory region. Run
  `live_viewer.py` in a separate terminal (`python live_viewer.py` on
  Linux/Windows; `mjpython live_viewer.py` on macOS) to render. Cost
  is one numpy copy per env-0 step — negligible, no SubprocVecEnv
  drag. Always-on safe.
- `AUTO_TB = True` launches `tensorboard --logdir LOG_DIR` as a child
  process and (optionally, via `AUTO_OPEN_BROWSER`) opens the URL in
  the default browser. Killed via `atexit` when training exits.

`logs/` and `checkpoints/` on this box also contain pre-refactor runs
(`hexapod_stage1`, `_test`, `_v3`, `_long`, plus `ant_vel_cmd`) that
reference deleted code. All gitignored; safe to delete when
convenient. The `hexapod_ros/` clone in the project root is similarly
redundant now that meshes are vendored under `models/meshes/`.

## Patterns developed in workstation sessions

- **Smoke-test after env edits**: one-shot
  `./.venv/Scripts/python.exe -c "..."` that resets and steps the env
  before declaring a change done. Catches obvious wiring breaks.
- **Filesystem-based run monitoring**: check `logs/<run>/` and
  `checkpoints/<run>/` mtimes to confirm a run is alive without
  poking the running process.
- **TensorBoard event-file reading**: when the user asks "how's
  training going?" parse the event file directly via
  `tensorboard.backend.event_processing.event_accumulator.EventAccumulator`
  rather than relying on a shareable dashboard. Pulls the same scalars
  TB shows. Note that `rollout/ep_rew_mean` only updates when episodes
  terminate; the `EPISODE_MAX_STEPS=2000` truncation in `train.py`
  keeps that signal fresh even after the policy stops falling.
- **Training is launched manually by the user** in a separate
  terminal. Claude's role is env / reward / training-script edits and
  inspecting on-disk artifacts; don't try to launch long training
  runs from the Bash tool.
- **Numerical-equivalence + round-trip validation** on every IK
  refactor: random cmd → predict → mj_forward → measured foot.
  Sub-mm error per leg means the refactor preserves correctness;
  any larger error means a sign/offset bug that won't surface until
  off-rest poses and is best caught early.

## Training tuning levers (actively iterated)

Things we keep adjusting between training runs as the policy improves.
All in `envs/hexapod_env.py` and `train.py` — the unified cross-platform
files. Mac and workstation use the same code, so changes here apply
everywhere.

### Reward shaping

- **Drift penalty** (`envs/hexapod_env.py:YAW_DRIFT_W` /
  `PITCH_DRIFT_W` / `ROLL_DRIFT_W`, gated by `YAW_GATE` / `TILT_GATE`).
  *Linear* penalty on `|actual_wz|`/`|actual_pitch|`/`|actual_roll|`
  when the corresponding cmd is near zero. Sharper gradient near zero
  than the squared tracking term — fixed the slow-drift problem where
  the bot would walk straight-ish but slowly turn or hold a tilt
  because the squared error was ~0 close to zero. Default 1.0/0.5/0.5
  weights with 0.05 rad/s and 3° gates. **If the bot looks rigid or
  refuses to move at all, these are too high — reduce by 50% and
  retry.** Drift penalty + scaffold-strong training together can
  produce a "do nothing" policy that aces every reward except actual
  movement.
- **Translation magnitude range** (`SPEED_MIN_FRAC=0.4`,
  `SPEED_MAX_FRAC=0.85`). Was [0, MAX_SPEED] originally; tightened to
  [40%, 85%] to remove "stand still" episodes (which encourage do-
  nothing) and "ragged edge" episodes (full-MAX_SPEED is at scaffold
  capacity, looks ugly). Heading is still random 0-360°.
- **Per-slot error scales** (`_err_inv_scales`). Each cmd error is
  divided by the slot's "full deviation" before squaring, so a
  velocity error and a pitch error contribute commensurately to the
  gaussian tracking reward. Without this, big-magnitude slots
  dominate and small-magnitude slots provide no signal.
- **Body angular-velocity penalty** (`ANGVEL_W=0.02`). Damps wobble in
  pitch/roll rates that don't show up cleanly in absolute pose
  tracking. If walking gait inherently produces >3° pitch oscillation,
  this fights legitimate motion — reduce it.

### Curriculum schedule

- **Stage masking is per-skill, not all-at-once.** Adding many cmd
  slots simultaneously in stage 1 (full overlays at once) produced
  worse results than the per-stage progression. Current
  `STAGE_CMD_MASK[1] = [1,1,0,0,0,0,0,0,0]` — translation only — with
  drift penalty implicitly training "stay straight + stay level +
  don't bounce." Add wz / height / width / pitch / roll one (or a
  small group) per future stage. Don't go back to all-at-once unless
  you have a plan for the "do nothing" trap.
- **Schedule constants** in `train.py`: `STAGE_MIN_STEPS[i]` controls
  per-stage fade-progress denominator; `STAGE_FADE_RANGE` controls
  when within the stage the fade happens. Recent successful pattern
  for short single-stage runs: 8M scaffold-strong / 4M fade / 13M
  autonomous = 25M total.
- **Per-stage `gait_scale` envelope**: short single-stage refinement
  runs use `STAGE_GAIT_SCALE[1] = (1.0, 0.0)` (full fade within the
  single stage); the original multi-stage curriculum uses
  `(1.0, 1.0) → (1.0, 0.6) → (0.6, 0.0) → (0.0, 0.0)` across stages
  1-4. Pick whichever matches your run's `INITIAL_STAGE` /
  `SKIP_SCAFFOLD` configuration.

### Auto-stop and resume

- **`EARLY_STOP_*`** in `train.py` — only active in autonomous phase
  (gait_scale == 0.0). Halts training when rolling tracking reward
  plateaus (no improvement > 0.005 over 3M steps, after a 3M warmup,
  with min reward 0.5). Saves wall time when the policy has converged
  before TOTAL_STEPS.
- **`python train.py --resume`** — load latest checkpoint from
  `CKPT_DIR` and continue training. Resume mode forces gait_scale=0.0
  throughout (no scaffold replay) and gives a fresh early-stop warmup
  window. Use this for quick fine-tunes after small reward / sampling
  changes — much faster than re-running from scratch. Pass an explicit
  path with `--resume <path-without-.zip>` to fine-tune from a
  specific older checkpoint.
- **`python train.py --bc-init <path>`** — initialize from a BC-
  pretrained policy. Resets `log_std` to a safer default (-2.0); for
  refinement runs override with `--log-std-init -3.0` (see Phase 2
  noise calibration below).

## Lessons learned about RL training (collected over Mac iteration)

These are failure modes we've actually hit, and what we learned from
each. Worth re-reading before launching a new training run.

- **"Do nothing" policy.** Drift penalty + many "stand still" episodes
  + scaffold doing the walking → policy converges on zero residual,
  which costs nothing in drift_pen and tracks well during scaffold-
  strong phase. Once scaffold fades, bot just stands. Mitigation:
  remove stand-still episodes from sampling distribution (clamp speed
  away from 0); reduce drift penalty weights; lengthen autonomous
  phase so policy has time to discover non-zero residuals.
- **"Ragged edge" gait.** Full MAX_SPEED commands sit at scaffold
  stride capacity — any extra residual destabilizes. Mitigation: clamp
  upper end of speed sampling (e.g., 85% MAX_SPEED) so scaffold has
  headroom; residuals can refine without fighting saturation.
- **Spawn-state matters for evaluation.** Eval scripts that don't
  settle the bot before showing it produce "looks weird from the
  start" perception. All eval scripts (`watch.py`, `watch_demo.py`,
  `pilot_ai.py`) now run a 200-step settling loop with cmd=0 before
  starting the real evaluation. Do not remove this.
- **Cmd-distribution overfit.** Policy trained on translation-only
  doesn't know what to do with wz/pitch/roll commands. `watch_demo.py`
  cycles through ALL phases including the untrained ones — bot may
  ignore those phases or wobble. Not a bug. Confirm trained skills
  work in the relevant phases; expect untrained ones to look weird.
- **Autonomous phase is where the actual learning happens.** During
  scaffold-strong, the residual barely matters because the scaffold
  output dominates. Most of the policy's "skill" is acquired during
  the fade and autonomous phases. Front-loading the fade (start fading
  at e.g. 8M instead of waiting for 40M like the original schedule)
  compresses training time substantially when the scaffold already
  walks well.
- **Progress signals to watch in TensorBoard**: `stage/avg_tracking`
  (target ≥ 0.65), `stage/fall_rate` (target ≤ 0.30),
  `stage/no_progress_rate` (truth-teller for "is bot actually walking"
  — added 2026-04-28), `stage/fail_rate` (= fell ∪ no_progress),
  `stage/mean_ep_length` (target → 2000 = max), `rollout/ep_rew_mean`
  (overall progress curve), `train/std` ("is the policy converging?"
  — should slowly decrease). Reward dipping when scaffold fades is
  fine if it recovers within the autonomous phase. Dipping and not
  recovering means scaffold was carrying everything and policy didn't
  learn.

## Multi-method training pipeline (planned 2026-04-29)

The project's training methodology evolved from "pure PPO with curriculum
fade" to a staged hybrid pipeline. Each phase uses the most-appropriate
method for its specific goal; advance only when the current axis is
"good enough."

| Phase | Method | Goal | Status |
|---|---|---|---|
| 0 | BC pretraining (`pretrain_bc.py`) | Bootstrap a working walker via supervised learning on scaffold demos | DONE — BC v2 |
| 1 | DAgger (script not yet built) | Refine BC into a flawless mimic — close covariate-shift gap | DEFERRED |
| 2 | Small-noise PPO (`train.py --bc-init … --log-std-init -3.0`) | Refine reward-coded axes (foot dev, sliding) past scaffold quality | CURRENT |
| 3 | Domain randomization (env modifications + RL or DAgger) | Sim-to-real robustness | not started |
| 4 | Distillation (script TBD) | Compress to small net for ESP32-S3 | not started |

**Phase 2 noise calibration matters a lot.** PPO's stochastic noise is
both required (gradient computation) and dangerous (destabilizes a
working walker). With a BC-initialized policy:
- `log_std=0.0` (default; std=1.0): random actions, bot falls instantly.
- `log_std=-2.0` (std≈0.135): too noisy for refinement; v6 run ended
  worse than BC v2.
- `log_std=-3.0` (std≈0.05): refinement zone, ~2.3° per-joint jitter.
  Recommended starting point for refinement runs.
- `log_std=-3.5` (std≈0.03): polish, ~1.4° per-joint jitter.
- Going below -4.0 starves PPO of gradient signal.

Always launch BC-init runs with `--log-std-init -3.0` or lower when
the goal is refinement (not from-scratch learning). The `--bc-init`
flag defaults to `log_std=-2.0` for safety, but for an already-working
walker you almost always want lower.

**DAgger sketch (when it gets built).** New script `dagger_train.py`
(~150 LOC), reuses `pretrain_bc.train_bc` and the env's
`info["bc_target"]` per step. Loop:
1. Drive bot at gait_scale=0.0 with current policy (deterministic).
2. Collect (obs, scaffold's-action-at-this-state) pairs each step.
3. Aggregate with original BC dataset.
4. Retrain policy on augmented data.
5. Repeat 3-5 times.

No noise, no exploration, no PPO — pure supervised learning on the
policy's own state distribution. Best tool for "make BC v2 a flawless
mimic." Cannot exceed scaffold quality (DAgger's ceiling).

## MJX exploration findings

### Mac investigation (2026-04-29)

User benchmarked Brax PPO on Ant to evaluate JAX-Metal speedup before
committing to a hexapod MJX port.

**Verdict on Apple Silicon GPU + MJX:** **blocked**. `jax-metal` v0.1.1
(latest, last release Oct 2024) doesn't implement the Cholesky
decomposition op (`mhlo.cholesky`). MJX's smooth-dynamics solver calls
`jax.scipy.linalg.cho_factor` on the mass matrix every step, so MJX
cannot run on Apple Silicon GPU at all today.

**CPU MJX on M3 Max:** ~9,000 steps/sec with 2048 vmap'd envs (Brax
PPO on Ant). Compared to SubprocVecEnv ~5,000 steps/sec, that's only
~1.8× speedup. Not worth porting on Mac alone.

### Workstation investigation (2026-04-30)

Set up Ubuntu 24.04 inside WSL2 on the Windows workstation, installed
`jax[cuda12] mujoco mujoco-mjx brax` in `~/.venv-mjx/`. WSL exposes
the host's NVIDIA driver via `/dev/dxg`; JAX detects the 5060 Ti as
`CudaDevice(id=0)`. Same Brax Ant benchmark (4096 vmap'd envs):

| Backend | Total throughput | Speedup vs SubprocVecEnv |
|---|---|---|
| SubprocVecEnv (current hexapod, N_ENVS=32) | ~5,400 SPS | 1.0× (baseline) |
| **CPU MJX** (Ryzen 7800X3D)                | **18,016 SPS** | 3.3× |
| **CUDA MJX** (RTX 5060 Ti via WSL2)        | **377,521 SPS** | **~70×** |

Confirms the original prediction (100k–500k SPS expected). The 7800X3D's
L3 + AVX-512 also doubles CPU MJX over the M3 Max — so even the CPU
fallback is ~3× SubprocVecEnv on this box.

**Decision: port the hexapod stack to MJX.** The 60× wall-clock speedup
is large enough to fundamentally change the iteration cycle (5-hour
training runs become ~5–10 minutes, enabling rapid reward / curriculum
experimentation). Plan in next section.

WSL benchmark script lives at `~/mjx_bench/ant_bench.py` (inside WSL,
not in the project tree); reusable for re-benchmarking after the port.

## MJX port plan (2026-04-30)

Goal: replace the SubprocVecEnv training stack with a JAX-native one
that runs on the 5060 Ti via WSL2. Target throughput: 150k–300k SPS for
the hexapod (slightly slower than Brax Ant's 378k due to 18 DOF + more
contacts vs Ant's 8 DOF). All current capability — closed-form IK,
overlays, BC infrastructure, curriculum — to be preserved.

### Phase 1 — Simplified MJCF (`models/phantomx_simple.xml`)

Replace mesh geoms with primitive geoms. **The meshes were never load-
bearing for physics** — they only provided collision shapes that we
can specify more cheaply with primitives, and visual appearance which
training doesn't care about. MJX strongly prefers primitives (mesh
collision is hard to JIT efficiently and limits parallelism).

Replacements:
- **Body**: box geom, dimensions matching the chassis bounding-box
  (~0.27 × 0.23 × 0.04 m). Mass + inertia stays as currently specified.
- **Coxa, femur**: capsule geoms along the link's length axis.
- **Tibia**: capsule geom + small sphere at the foot tip.
- **Joint axes, ranges, kp/kv, masses, inertias**: all unchanged.
- Visual meshes can stay as `contype=0 conaffinity=0` non-collision
  geoms (purely decorative) OR be removed entirely.

The closed-form IK only uses link lengths and the foot-tip position;
both are now explicit constants in the new model rather than empirical
values derived from mesh vertices. `FOOT_TIP_LOCAL` becomes
`(0, 0, -tibia_length)` exactly — cleaner than the current empirical
`(0.134, 0.031, 0.0)` from mesh vertex search.

Validation: run `IK_gait.py` against the new model — sub-mm round-trip
error should hold. The user's insight here is correct: physics cares
about geometry constraints (collision shapes, joint limits, masses),
not visual fidelity. The "meat on the wireframe" framing is the right
mental model.

### Phase 2 — Pure-JAX gait controller (`gait/controller_jax.py`)

Port `gait/controller.py` to JAX. Most of it translates near-1:1 since
the IK is already vectorized over `(6, *)` arrays:

- `numpy` → `jax.numpy`; `math.atan2` → `jnp.arctan2`; etc.
- Pre-compute all per-leg constants (yaw cos/sin, body origin in coxa
  frame, joint signs, leg path table) at controller construction; pass
  as `static_argnums` or fold into closure for `@jax.jit`.
- Replace `if/else` on cmd values with `jnp.where` or `jax.lax.cond`.
- The path-lookup `LEG_PATH_DELTAS[arange(6), n_idx]` advanced indexing
  is JAX-compatible; same logic ports directly.
- `set_cmd` / stateful `step(dt)` API replaced by purely functional
  `predict_with_feet(cmd, t)` (already exists; just needs JIT).

Validation: random cmd → both controllers → `max(abs(jax_out -
np_out)) < 1e-6`. Run on 1000 random cmds.

### Phase 3 — JAX-native env (`envs/hexapod_env_jax.py`)

Brax-style functional environment:

```python
state = env.reset(key)                  # PRNGKey → State (NamedTuple)
state = env.step(state, action)         # functional update
```

State carries qpos/qvel/sim_time/cmd/last_action/etc. as `jnp.ndarray`
fields. Reward and termination computed in pure JAX. Compatible with
Brax's PPO trainer out of the box.

The `live_watch` SHM observability path doesn't apply to MJX (training
runs entirely on GPU; there's no per-worker subprocess to publish
from). Live-watching during training will work differently: a separate
host process samples N_envs[0]'s qpos at, say, 30 Hz from device memory
via `state.qpos.block_until_ready()` + `np.asarray(...)`. Or just use
periodic checkpoint dumps + `watch.py` like before.

### Phase 4 — Training pipeline (`train_jax.py`)

Use Brax's PPO trainer (`brax.training.agents.ppo`). It's JAX-native,
runs entirely on the GPU, takes a Brax-style env. Curriculum
(`StageManagerCallback`-equivalent) needs porting to a JAX-friendly
training-step hook. BC pretraining (`pretrain_bc.py`) needs adapting
to produce Brax-compatible policy weights — this is the gnarliest
piece, may delay until after we validate the basic training loop works.

### Phase 5 — Validation

- Benchmark hexapod stepping on CUDA: target ≥150k SPS at N_envs=4096.
- Short training run (1-2M steps) on JAX, compare reward curve to a
  matched SubprocVecEnv run. Reward signals should be statistically
  equivalent (within stochastic noise).
- If CUDA throughput is good but training behavior diverges from
  SubprocVecEnv significantly, suspect numerical-precision drift from
  fp32 (vs SubprocVecEnv's fp64 by default in MuJoCo).

### Effort estimate

Phase 1 + 2: half a day each (clean ports of well-tested code).
Phase 3 + 4: 1-2 days each (more design surface, integration with
Brax's training APIs, BC re-targeting).
Phase 5: ongoing.

**Sequencing decision**: if the user wants to validate the model
simplification + JAX IK fast before committing to the env/training port,
phases 1-2 alone produce a useful intermediate state — the simplified
MJCF + JAX gait controller could be benchmarked against the existing
MJCF + SubprocVecEnv to confirm "the simple model walks the same."
That's a low-risk go/no-go gate before phase 3.

## Memory / sync

This file is the shared context. The local `.claude/memory/` directory
holds machine-local memory (auto-managed by Claude Code) which is more
detailed but does NOT travel with the repo. Major decisions captured
there are summarized above; refer to that directory for finer-grained
notes if available.
