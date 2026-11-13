# Robotics final project — mini hexapod: from analytical gait to a learned controller

Working outline + demo runbook. Presentation ≈ 3 weeks from 2026-11-12 (date TBD).
Bot available again ≈ late Nov. Numbers marked `TODO` get filled after the
corresponding run; everything else is already true today.

## 1. The story in one slide

1. **Hardware I built**: 18-servo mini hexapod (XIAO ESP32-C3, 2× PCA9685,
   MS18 9 g servos, no position feedback, no IMU yet).
2. **It already walks** on hand-written kinematics (closed-form IK + tripod
   gait scaffold, firmware menu, trims/limits saved on-chip).
3. **Question**: can reinforcement learning improve on the hand-written gait,
   and can that transfer to a $60 robot with open-loop servos?
4. **Approach**: simulate the bot in MuJoCo-MJX, train PPO on top of the
   scaffold with Adversarial Motion Priors (AMP) so the policy stays
   "hexapod-like", randomize the physics so it survives my measurement errors,
   stream the learned joint targets to the robot over USB.
5. **Result**: `TODO` (sim videos + robustness table + live demo).

## 2. Suggested slide order (12–15 min)

| # | Slide | Asset |
|---|---|---|
| 1 | Title + the robot on the desk | photo / the bot itself |
| 2 | Hardware: parts, wiring, servo reality (500–2500 µs, no feedback) | photo, parts table |
| 3 | Kinematics: leg geometry, closed-form IK, tripod gait scaffold | `docs/kinematics.md` figure; firmware `walk` demo video or live |
| 4 | Why learn anything? scaffold = open loop, fixed gait, no terrain/disturbance handling | 2 bullets |
| 5 | Simulation: generated MJCF (`tools/gen_mini_hexapod_mjcf.py`), 4096 parallel robots on one GPU at ~200k steps/s | screenshot of `mini_hexapod_mjx.xml` render |
| 6 | Training recipe: scaffold priors → behaviour cloning → PPO + AMP (partition discriminator, 150 cmd bins) | pipeline diagram; `media/plots/training_curves_v27_vs_mini_v1.png` |
| 7 | What the policy sees / does: 75-dim obs (cmd, phase, scaffold intent, own commands), 18-dim foot residual action, 200 Hz | obs/action table |
| 8 | Sim result 1: it walks (PhantomX v27 and mini v1 videos) | `media/recordings/v27_iter5_showcase.mp4`, `mini_v1_iter10_showcase.mp4` |
| 9 | Honest finding: on flat ground AMP reproduces the scaffold (tracking 0.95 vs 0.96) — the learned part earns its keep only when the scaffold is wrong | eval table |
| 10 | Sim-to-real plan: calibrate real joint limits/trims (firmware `L`/`J` → importer), domain randomization over friction / servo strength / mass / damping | `tools/eval_dr_robustness.py` table (`TODO v1dr`) |
| 11 | Deployment: tethered — laptop runs scaffold + policy, streams 18 joint targets at 50 Hz over serial; 500 ms watchdog homes the legs | diagram; `scripts/drive_serial.py` |
| 12 | Live demo (or video fallback) | see runbook |
| 13 | Limitations: no IMU → feed-forward policy, can't react to pushes; placeholder masses until measured; servo torque is the ceiling (~0.12 m/s) | bullets |
| 14 | Next steps: IMU + filter, on-device int8 inference (≈60 KB MLP fits the C3), terrain randomization | bullets |

## 3. Key numbers (fill as they land)

| Item | Value | Source |
|---|---|---|
| Training throughput | ~200–230k env-steps/s, RTX 5060 Ti, 4096 envs | `logs/stdout/amp_mini_hexapod_v1.log` |
| Mini v1 run time | 68 min end-to-end (priors 1.0M transitions, BC, 10 × 50M PPO) | pipeline log |
| Mini v1 eval reward | +2222 → +2302 (iter 3) → +2298 (iter 10), max ≈ 2500 | plot |
| Tracking vs scaffold (stage 1 / 2) | 0.95 / 0.89 policy vs 0.96 / 0.93 scaffold, 0 falls | `eval_bc_quick.py` |
| Dynamics DR run (v1dr) | `TODO` eval reward, fall % at dr 0 / 1 / 1.5 vs v1 and scaffold | `tools/eval_dr_robustness.py` |
| Real-bot result | `TODO` (scaffold-only first, then policy) | demo log CSV |

## 4. Demo runbook (live)

Prep the night before:

1. Flash firmware v0.2 (`mini_hexapod/firmware/arduino/hexapod_firmware`), confirm
   trims/limits load from NVS (`p` prints them).
2. Battery charged; USB cable that carries data; know the serial port
   (`ls /dev/cu.usbmodem*` on the Mac, `COMx` on Windows).
3. Dry run without the bot:
   `PYTHONPATH=. python scripts/drive_serial.py --dry-run --source demo --params <policy>`.

On stage, in this order (each step is a fallback for the next):

1. **Firmware walk** (`w` in the menu) — proves the hardware. Zero software risk.
2. **Scaffold over serial** — `drive_serial.py --port … --source hold --cmd 0.04 0 0`
   (no `--params`). Proves the tether + kinematics convention.
3. **Policy over serial** — add `--params checkpoints/amp_to_mini_hexapod_<final>/iter10/final/params.pkl`,
   `--source gamepad` if the 8BitDo is paired, else `--source demo`.
4. If anything misbehaves: unplug USB → watchdog homes the legs within 0.5 s;
   switch to the videos.

Say out loud what the audience can and cannot see: the policy is a learned
*correction* on top of the scaffold (±12 mm per foot), so a good result looks
like "the same gait, a little smoother / more robust", not a new gait.

## 5. Timeline to presentation

| When | What |
|---|---|
| now → bot visit | DR run (`v1dr`), robustness table, videos, slides draft |
| bot visit (late Nov) | flash v0.2 → `L` limits for 18 joints → `J` export → `tools/import_mini_calibration.py` → measure coxa offset + masses → regenerate MJCF → rerun pipeline (≈1 h GPU + priors/BC) → scaffold-over-serial test → policy-over-serial test, record phone video |
| last week | final videos, rehearse runbook, charge batteries, print the fallback plan |
