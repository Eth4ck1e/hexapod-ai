# Hexapod AI

Reinforcement-learning locomotion for a PhantomX-style hexapod robot. PPO trains a residual policy on top of a scripted tripod-gait scaffold; the gait fades out progressively over training so the policy takes over full control of the 18 joint targets. Eventual target hardware: ESP32-S3 controller, Dynamixel servos, BNO055-class IMU, optional VL53L5CX ToF sensors.

## Status

Stage 1 (cardinal commands at fixed magnitude) is the current training target. Subsequent stages will add continuous velocity, yaw rate, body-pose offsets, and stance modulation — see the cmd-vector slot map in `envs/hexapod_env.py`.

The scripted tripod gait in `walk_test.py` is functional only for forward motion in the current revision; it's a starting scaffold, not a finished controller. Iterate on it freely — it's also a useful sandbox for hand-coded gait experiments.

## Layout

```
.
├── train.py              PPO training entry point
├── walk_test.py          Scripted-gait validation (no policy)
├── watch.py              Render the latest checkpoint in MuJoCo viewer
├── watch_tiled.py        N-up grid of envs running the latest checkpoint
├── benchmark_envs.py     Throughput benchmark
├── envs/
│   └── hexapod_env.py    HexapodEnv (gymnasium) + IK + tripod gait
├── models/
│   ├── phantomx.xml      MuJoCo MJCF
│   ├── meshes/           Vendored STL meshes (PhantomX + razbot variants)
│   └── urdf/             Original URDF / xacro files (reference only)
├── hardware/             PCB design, BOM, mechanical drawings (placeholder)
├── docs/
│   └── kinematics.md     IK/FK math, leg geometry, MJCF axis conventions
├── requirements.txt
├── LICENSE               MIT (this project)
└── NOTICE                Third-party attributions (BSD-3 from upstream)
```

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
# PyTorch: install separately if you need a specific build (CPU/CUDA/MPS).
# CPU-only is fine — PPO MLP doesn't benefit from GPU at this size.
```

### After cloning, fix the mesh path

`models/phantomx.xml` currently references the upstream clone path. Edit the line:

```xml
<compiler angle="radian" meshdir="../hexapod_ros/hexapod_description/meshes/phantomX"/>
```

to:

```xml
<compiler angle="radian" meshdir="./meshes/phantomx"/>
```

This points at the vendored meshes in `models/meshes/phantomx/` so the project is self-contained. (One-time fix per clone, or done once and committed.)

## Workflow

### Validate the scripted gait first
```bash
python walk_test.py        # Linux/Windows
mjpython walk_test.py      # macOS — `launch_passive` requires mjpython
```
A MuJoCo viewer opens; the robot should walk forward. Edit `cmd` near the top of the file to test other directions.

> **macOS note:** Any script that opens the MuJoCo viewer (`walk_test.py`, `sandbox.py`, `gait_design.py`, `watch.py`, `watch_tiled.py`) must be launched with `mjpython` instead of `python`. `mjpython` ships with the `mujoco` pip package and is in `.venv/bin/`. Headless scripts (`train.py`, `benchmark_envs.py`) use plain `python`.

### Train
```bash
python train.py
```
- Run name, total steps, gait-fade schedule, and checkpoint cadence are all configured at the top of `train.py`.
- Outputs land in `logs/hexapod_<RUN_NAME>/` and `checkpoints/hexapod_<RUN_NAME>/`.

### Watch a trained policy
```bash
python watch.py                                            # auto-loads newest checkpoint
python watch.py checkpoints/hexapod_stage1_long/final      # explicit path
```

### Tile-view multiple instances
```bash
python watch_tiled.py            # 16-up grid by default
python watch_tiled.py <ckpt> 9   # 3x3 grid
```

### Monitor with TensorBoard
```bash
tensorboard --logdir logs/hexapod_stage1_long
```
Open http://localhost:6006. Notable scalars: `train/value_loss`, `train/explained_variance`, `train/std`, `train/clip_fraction`, `train/approx_kl`. (`rollout/ep_rew_mean` only appears once enough episodes terminate — early policies fall a lot, then fewer terminations once it learns to stay upright.)

### Benchmark throughput on this machine
```bash
python benchmark_envs.py
```

## Custom-gait development

`walk_test.py` runs the scripted base controller without any policy on top — it's the right place to prototype your own gait math. The MuJoCo viewer gives interactive control of camera and time-scale; pause with space, single-step with right-arrow.

`HexapodEnv` (in `envs/hexapod_env.py`) plugs your gait into PPO. The `_tripod_gait`, `_ik`, and `_compute_neutral_pose` functions are all module-level and can be swapped out independently.

## Hardware target

- **MCU**: ESP32-S3 (Xtensa LX7 dual-core @ 240 MHz, 8–16 MB PSRAM)
- **Actuators**: 18× Dynamixel AX/MX servos (PhantomX standard)
- **IMU**: BNO055 or equivalent — quaternion + gyro + accelerometer
- **Optional sensors**: VL53L5CX 8×8 ToF arrays, mounted around body for terrain/obstacle awareness
- **PCB & BOM**: in `hardware/` (when added)

## Credit

PhantomX URDF/STL meshes and the inverse-kinematics + tripod-gait math derive from Kevin Ochs's [hexapod_ros](https://github.com/KevinOchs/hexapod_ros), used under its 3-Clause BSD license. Full attribution in `NOTICE`.
