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

## Current state (as of 2026-04-26)

Tasks #8–#13 complete. Training is running on the user's workstation.
Tasks remaining:

- #14: Stage 1 training run (in progress / monitoring)
- #15: Test trained checkpoint via pilot.py + iterate

## Architecture (locked decisions)

- **`gait/` package** is the single source of truth for gait math.
  `Controller` class: stateful, `predict(cmd, t) → 18 joint targets`.
  Used by `simple_gait.py` (demo viewer), `pilot.py` (teleop),
  `envs/hexapod_env.py` (training), eventually ESP32-S3 firmware.
- **Cmd vector** (9 floats, physical units):
  ```
  [0] vx           m/s     body forward velocity
  [1] vy           m/s     body lateral velocity (left = +)
  [2] wz           rad/s   body yaw rate (CCW = +)
  [3] pitch        rad     body pitch target (nose up = +)
  [4] roll         rad     body roll target (right side up = +)
  [5] height_delta m       stance height delta (- raises body)
  [6] width_delta  m       stance width delta from neutral
  [7] shift_x      m       body shift in body +X
  [8] shift_y      m       body shift in body +Y
  ```
- **Reward (minimal)**: `tracking_exp(-||cmd-actual||²) + 0.1*survive
  - 0.01*action_delta² - 0.02*body_angvel² + (stage 4 only) novelty`.
  Errors are normalized per-slot by `_err_inv_scales` so each axis
  contributes commensurately. Tracking compares ALL body kinematics
  every step regardless of which stage the bot is in — locked cmd
  slots are zero, so any actual deviation creates real tracking error.
  This is what makes the bot learn smooth controlled locomotion vs
  jerky lurching that happens to track velocity.
- **Curriculum** (in `train.py`):
  - Stage 1: vx, vy at fixed MAX_SPEED (random heading only). 40M steps
    nominal. `gait_scale = 1.0` throughout.
  - Stage 2: + wz, height, width. 30M steps. `gait_scale 1.0 → 0.6`.
  - Stage 3: + pitch, roll, shifts. 30M steps. `gait_scale 0.6 → 0.0`.
  - Stage 4: pure policy, novelty bonuses, body_linvel dropped from obs.
  - Adaptive advance: tracking ≥ 0.65 AND fall rate ≤ 0.30 AND min
    step count reached.
- **Deployment plan**: analytical gait library ships to ESP32-S3 as a
  standalone "manual mode" AND as a safety fallback under the policy.
  Three runtime modes: manual / ai / assisted.

## Important non-obvious things

- **`_ik` and `_tripod_gait` from the original `envs/hexapod_env.py`
  were geometrically inconsistent with the MJCF.** They are deleted.
  All gait math now lives in `gait/controller.py`, which uses
  `mj_jac` iterative IK on the actual MJCF kinematics. Foot tip in
  tibia local frame is determined empirically (lowest world-Z mesh
  vertex at NEUTRAL_POSE).
- **Speed-magnitude variation is intentionally NOT in the curriculum.**
  Translation cmd is always at full MAX_SPEED with random heading. The
  current gait library models speed only via stride scaling, which is
  wrong (real gait is mostly cadence/period modulation). When we
  revisit speed, we'll rewrite the library to do period+scale together
  (period ≈ >100% range, scale ≈ ~25% range). See
  `.claude/memory/project_speed_control_plan.md`.
- **macOS viewer requires `mjpython`**, not plain `python`, for any
  script that calls `mujoco.viewer.launch_passive`. Affects
  `simple_gait.py`, `demo.py`, `pilot.py`, `watch.py`, `watch_tiled.py`.
  Headless (`train.py`) uses plain `python`.
- **`pilot.py` keyboard layout collides with mujoco viewer hotkeys.**
  Accepted as fine for testing only. Long-term we'll add gamepad
  support via pygame; gamepad axes don't conflict.
- **Legacy demo scripts are now broken**: `sandbox.py`, `gait_design.py`,
  `walk_test.py` import deleted constants from the old env. They've
  been superseded by `simple_gait.py`, `demo.py`, `pilot.py`. Safe to
  delete; left in repo for now as historical reference.
- **Snapshots** in `snapshots/` are historical versions of
  `simple_gait.py` from before the library refactor — useful as
  reference for how each overlay was added incrementally.

## Files at a glance

| Path | Purpose |
|------|---------|
| `gait/controller.py` | Stateful gait Controller (the core math). |
| `gait/__init__.py` | Public API exports. |
| `simple_gait.py` | Demo viewer — toggle cycles for any cmd slot. |
| `demo.py` | 28-phase auto showcase of every overlay. |
| `pilot.py` | Keyboard teleop in mujoco viewer. |
| `envs/hexapod_env.py` | Gym env wrapping `Controller` for PPO. |
| `train.py` | Curriculum training entry point. |
| `watch.py` | Render trained checkpoint in viewer. |
| `models/phantomx.xml` | MJCF model. Mesh path fixed to `./meshes/phantomx`. |
| `snapshots/simple_gait_v*.py` | Historical pre-refactor versions. |

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

## Workstation environment (Windows host)

The user's primary training machine is a Windows 11 desktop with the
project at `C:\Users\Eth4ck1e\OneDrive\Documents\Hexapod AI Project`.

- **Hardware**: AMD Ryzen 7 7800X3D (8C/16T, 96 MB L3 V-Cache,
  AVX-512), 32 GB RAM, RTX 5060 Ti. The 96 MB L3 fits MuJoCo's per-env
  physics working set, which is why this box outpaces the M1 Max for
  SubprocVecEnv runs.
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

Pre-refactor benchmark (older single-file gait/IK, 16 envs via
SubprocVecEnv, `device="cpu"`): **~5,000–5,250 SPS**. Same config
with `device="cuda"` ran ~2,750 SPS — SB3 warns about this and it
holds: GPU dispatch overhead dominates the small MLP gradient. Keep
`device="cpu"` in `train.py`.

The new `mj_jac` iterative IK in `gait/controller.py` is more
expensive per step than the old closed-form math, so post-refactor
throughput will be lower. Re-benchmark via `python benchmark_envs.py`
when needed; the 200M-step curriculum's ~10–11 hr time estimate was
based on the older numbers.

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
  terminate; the new `EPISODE_MAX_STEPS=2000` truncation in `train.py`
  keeps that signal fresh even after the policy stops falling.
- **Training is launched manually by the user** in a separate
  terminal. Claude's role is env / reward / training-script edits and
  inspecting on-disk artifacts; don't try to launch long training
  runs from the Bash tool.

## Memory / sync

This file is the shared context. The local `.claude/memory/` directory
holds machine-local memory (auto-managed by Claude Code) which is more
detailed but does NOT travel with the repo. Major decisions captured
there are summarized above; refer to that directory for finer-grained
notes if available.
