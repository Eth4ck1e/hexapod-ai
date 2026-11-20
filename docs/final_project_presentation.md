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
5. **Result (sim, so far)**: the learned policy walks and tracks commands as well as
   the hand-written gait (tracking 0.95 vs 0.96), and *both* survive 2× changes in
   friction, servo strength and payload with ~0 falls. On flat ground the hand-written
   gait is already near-optimal; the learned part is insurance for where the model is
   wrong (hardware), and the real test is the live demo.

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
| 10 | Sim-to-real plan: calibrate real joint limits/trims (firmware `L`/`J` → importer), domain randomization over friction / servo strength / mass / damping — robustness table shows scaffold and policy both at 0 % falls up to 1.5× the DR range | robustness table from `media/plots/dr_robustness_mini.csv` |
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
| Dynamics DR run (v1dr) | continued v1 for 9 × 50M steps under full DR: eval +2278 → +2296 → +2289 (flat, same level as nominal) | `media/plots/training_curves_mini_v1_v1dr.png` |
| v2h (final recipe: 50 Hz servo hold + DR, from scratch) | priors 73 s, BC 2.5 min, 10 × 50M PPO in 67 min; eval +2208 → +2288 → +2276; this is the policy for the bot | `media/plots/training_curves_mini_all.png` |
| Robustness (512 paired episodes × 1000 steps, falls %) | scaffold / v1 / v1dr / v2h — dr 0: 0 / 0 / 0 / 0 · dr 1.0: 0 / 0 / 0 / 0 · dr 1.5 (friction 0.25–1.45×, servo 0.4–1.45×, chassis mass 0.7–1.9×): 0.2 / 0.2 / 0.2 / 0.0; any-termination at 1.5×: 3.5 / 3.5 / 3.1 / 3.5 % | `media/plots/dr_robustness_mini_hold.csv` |
| Model-only controller (2026-11-19) | 21-in/18-out MLP (IMU slots + gait clock + cmd → joint targets), 49,124 params, 192 KB float32; cloned from the scaffold dataset: 0 % falls, tracking 0.997/0.46 vs scaffold 0.998/0.47 at dr 0 and 1.0. PPO+AMP continuation (p5) scored +2042 best in training but fell 3–11 % in paired eval → the cloned network is what ships | `media/plots/pure_p5_checkpoint_eval.csv`, `media/recordings/mini_pure_bc_p4_showcase.mp4` |
| Real-bot result | `TODO` (scaffold-only first, then policy) | demo log CSV |

## 4. Demo runbook (live)

Prep the night before:

1. Flash firmware v0.3 onto the S3: `bash mini_hexapod/firmware/flash.sh s3 /dev/cu.usbmodem*`.
   It boots with the May calibration baked in (recovered from the C3 on 2026-11-19);
   `J` prints what the board holds, `python tools/mini_push_config.py --port … mini_hexapod/calibration/<latest>.json`
   re-pushes a saved calibration with readback check.
2. Battery charged; USB cable that carries data; know the serial port
   (`ls /dev/cu.usbmodem*` on the Mac, `COMx` on Windows).
3. Dry run without the bot:
   `PYTHONPATH=. python scripts/drive_serial.py --dry-run --source demo --params <policy>`.

On stage, in this order (each step is a fallback for the next):

1. **Firmware walk** (`w` in the menu) — proves the hardware. Zero software risk.
2. **Scaffold over serial** — `drive_serial.py --port … --source console` (no `--params`),
   then type `forward`, `left`, `spin right`, `up 10` … Each move runs 5 s and the bot
   stands still in between, so it stays on the table. Proves the tether + kinematics.
3. **Policy over serial** — same console with `HEXAPOD_ROBOT=mini_hexapod_pure` and
   `--params checkpoints/bc_pretrained_jax_mini_hexapod_pure_p7/params.pkl` (model-only;
   the residual v2h policy under `HEXAPOD_ROBOT=mini_hexapod` is the fallback); add
   `--speed-frac 0.85` (top of the trained band — the 0.70 default looks like crawling on
   the real bot); use `--script docs/demo_script.txt` for the rehearsed sequence,
   `--source gamepad` if the 8BitDo is paired. **Bench-tested 2026-11-20**: the full demo
   script ran on the model alone (forward/back/side-steps/spins/arc/height/width, 0 frame
   errors); the user judged it indistinguishable from the programmed gait. Foot lift was then
   raised 15 → 21 → 30 mm (`lift_scale` 2.0: the 9 g servos lag the commanded arc, so 21 mm
   still dragged) and the policy re-cloned as p7 (= scaffold, 0 % falls).
4. If anything misbehaves: unplug USB → watchdog homes the legs within 0.5 s;
   switch to the videos.
5. **Between demos type `rest`** (or run with `--auto-rest 10`): the bot squats and
   the servos go limp. The 9 g servos drift when they sit stalled and hot, so
   standing powered for minutes visibly sags the stance; resting fixes it.

Say out loud what the audience can and cannot see: the model-only policy was cloned
from the programmed gait's dataset, so a good result looks like *the same gait*, now
produced by a 49k-parameter network from (clock, command) alone — not a new gait.

## 5. Timeline to presentation

| When | What |
|---|---|
| now → bot visit | ✅ DR run, ✅ servo-hold run (`v2h`), ✅ robustness table, ✅ sim videos; slides draft |
| bot visit (late Nov) | flash v0.2 → `L` limits for 18 joints → `J` export → `tools/import_mini_calibration.py` → measure coxa offset + masses → regenerate MJCF → rerun pipeline (≈1 h GPU + priors/BC) → scaffold-over-serial test → policy-over-serial test, record phone video |
| last week | final videos, rehearse runbook, charge batteries, print the fallback plan |
