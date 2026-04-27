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

## Current state

Gait library uses **closed-form, vectorized IK** with full overlay
support (stance width/height, body tilt, spin). All gait math validated
sub-millimeter against MJCF FK. The `IK_gait.py` sandbox tested every
overlay independently (smart test script with 19 phases covers them
all). Training pipeline integrated and benchmarked at the workstation's
sweet spot. Long-form curriculum training runs reasonable.

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
- **Curriculum** (in `train.py`):
  - Stage 1: vx, vy at fixed MAX_SPEED (random heading only). 40M steps
    nominal. `gait_scale = 1.0` throughout.
  - Stage 2: + wz, height, width. 30M steps. `gait_scale 1.0 → 0.6`.
  - Stage 3: + pitch, roll, shifts. 30M steps. `gait_scale 0.6 → 0.0`.
  - Stage 4: pure policy, novelty bonuses, body_linvel dropped from obs.
  - Adaptive advance: tracking ≥ 0.65 AND fall rate ≤ 0.30 AND min
    step count reached.
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
- **Speed-magnitude variation is intentionally NOT in the curriculum.**
  Translation cmd is always at full MAX_SPEED with random heading.
  The current Controller models speed via stride scaling only, NOT
  period modulation — period stays fixed at GAIT_PERIOD. The IK_gait
  sandbox proved variable-period speed control works (factor in
  [-1, +1] sweeps period and stride together for a wide envelope),
  but the env's Controller deliberately uses the simpler stride-only
  model for now to keep the policy's reward signal stable. Revisit
  when we want a wider speed range.
- **macOS viewer requires `mjpython`**, not plain `python`, for any
  script that calls `mujoco.viewer.launch_passive`. Affects
  `simple_gait.py`, `IK_gait.py`, `pilot.py`, `watch.py`,
  `watch_tiled.py`. Headless (`train.py`) uses plain `python` — the
  env-0 live-watch viewer works through SubprocVecEnv worker
  subprocesses where MuJoCo's viewer binding is fine.
- **`pilot.py` keyboard layout collides with mujoco viewer hotkeys.**
  Accepted as fine for testing only. Long-term: gamepad support via
  pygame; gamepad axes don't conflict.
- **Legacy demo scripts are now broken**: `sandbox.py`, `gait_design.py`,
  `walk_test.py` import deleted constants from the old env. They've
  been superseded by `simple_gait.py`, `IK_gait.py`, `pilot.py`. Safe
  to delete; left in repo for now as historical reference.
- **Snapshots** in `snapshots/` are historical versions of important
  files captured before risky refactors (e.g., `*_v1_pre_integration.py`
  are the pre-IK-refactor controller, env, and IK_gait state). Useful
  for revert paths.

## Files at a glance

| Path | Purpose |
|------|---------|
| `gait/controller.py` | Stateful gait Controller. Closed-form vectorized IK. |
| `gait/__init__.py` | Public API exports (Controller, link lengths, NEUTRAL_POSE, etc.). |
| `simple_gait.py` | Demo viewer — toggle cycles for any cmd slot via Controller. |
| `IK_gait.py` | Sandbox with all overlays + smart-test script (19-phase exercise). |
| `pilot.py` | Keyboard teleop in mujoco viewer. |
| `envs/hexapod_env.py` | Gym env wrapping `Controller` for PPO. |
| `train.py` | Curriculum training entry point + auto-TB + live env-0 watcher. |
| `bench_n_envs.py` | SubprocVecEnv N_ENVS sweep — finds CPU sweet spot. |
| `watch.py` | Render trained checkpoint in viewer. |
| `watch_tiled.py` | N-up grid render of trained checkpoint. |
| `models/phantomx.xml` | MJCF model. Mesh path fixed to `./meshes/phantomx`. |
| `docs/kinematics.md` | IK derivation, leg geometry, MJCF axis conventions. |
| `snapshots/` | Historical versions before refactors (revert points). |

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
- `WATCH_LIVE = True` opens a MuJoCo viewer window in env 0's worker
  subprocess. SubprocVecEnv synchronizes, so the viewer slowing env 0
  drags all 32 workers' step rate down — flip `WATCH_LIVE = False`
  for max throughput, on for spot-checking.
- `AUTO_TB = True` launches `tensorboard --logdir LOG_DIR` as a child
  process and opens the URL in the default browser. Killed via
  `atexit` when training exits.

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

## Memory / sync

This file is the shared context. The local `.claude/memory/` directory
holds machine-local memory (auto-managed by Claude Code) which is more
detailed but does NOT travel with the repo. Major decisions captured
there are summarized above; refer to that directory for finer-grained
notes if available.
