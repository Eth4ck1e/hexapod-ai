# Hexapod Kinematics Reference

Reference for the inverse and forward kinematics of one PhantomX leg, plus the conventions needed to use them across all six legs in the MJCF model.

Implementation lives in `envs/hexapod_env.py`. This document is the math behind it; useful when writing your own gait or body-pose controller.

## Leg geometry

Each leg has 3 revolute joints (coxa → femur → tibia) plus the foot at the tibia tip. There is no tarsus segment in this model.

| Constant | Value | Description |
|---|---|---|
| `COXA_LENGTH`  | 0.052 m  | Coxa pivot to femur pivot |
| `FEMUR_LENGTH` | 0.065 m  | Femur pivot to tibia pivot |
| `TIBIA_LENGTH` | 0.133 m  | Tibia pivot to foot tip |

Total straight-line reach from coxa joint to foot: `COXA + FEMUR + TIBIA = 0.250 m`.

## Per-leg constants

Order is `[RR, RM, RF, LR, LM, LF]` (right-rear, right-middle, right-front, left-rear, left-middle, left-front).

```
INIT_COXA_ANGLE = [-45°,  0°, +45°, -45°,  0°, +45°]
INIT_FOOT_POS_X = [-0.13, 0.0,  0.13, -0.13, 0.0,  0.13]   m
INIT_FOOT_POS_Y = [ 0.13, 0.18, 0.13,  0.13, 0.18, 0.13]   m
INIT_FOOT_POS_Z = [ 0.10, 0.10, 0.10,  0.10, 0.10, 0.10]   m
```

`INIT_COXA_ANGLE[i]` is the offset between the leg's body-frame mounting angle and the IK-frame "outward" direction. Adding it to the IK output gives the actual coxa joint angle. The values were chosen so that with the foot at `(INIT_FOOT_POS_X[i], INIT_FOOT_POS_Y[i], INIT_FOOT_POS_Z[i])` in the leg-local IK frame, the coxa joint is at zero.

`INIT_FOOT_POS_*` define the rest position of the foot in the leg's local IK frame:
- `+Y` = "outward from coxa" along the leg's natural extension direction
- `+X` = perpendicular to outward (rotates with coxa)
- `+Z` = downward (depth below the coxa joint)

## Inverse Kinematics

Given a foot target `(fx, fy, fz)` in the leg's IK frame, return joint angles `(coxa, femur, tibia)` in radians.

```
femur_to_tarsus = √(fx² + fy²) − COXA_LENGTH
side_c          = √(femur_to_tarsus² + fz²)

cos_b = (FEMUR² − TIBIA² + side_c²) / (2 · FEMUR · side_c)
cos_c = (FEMUR² + TIBIA² − side_c²) / (2 · FEMUR · TIBIA)

angle_b = acos(clip(cos_b, -1, 1))
angle_c = acos(clip(cos_c, -1, 1))
theta   = atan2(femur_to_tarsus, fz)

coxa  = atan2(fx, fy) + INIT_COXA_ANGLE[leg_idx]
femur = π/2 − (theta + angle_b)
tibia = π/2 − angle_c
```

`fz > 0` means the foot is below the coxa joint (positive depth). `fz < 0` lifts the foot above the coxa.

**Solvability**: requires `|side_c| ≤ FEMUR + TIBIA = 0.198 m`. If the target is unreachable, both `cos_b` and `cos_c` clip to ±1 and the leg straightens toward the target without reaching it. The IK does NOT raise; check reachability yourself if you need a hard error.

## Forward Kinematics

Given joint angles, return the foot position in the leg's IK frame:

```
# Working in the leg's vertical plane (after coxa rotation):
plane_distance  = COXA_LENGTH + FEMUR_LENGTH·cos(π/2 − femur)
                              + TIBIA_LENGTH·cos(π/2 − femur − tibia)
plane_depth     = FEMUR_LENGTH·sin(π/2 − femur)
                + TIBIA_LENGTH·sin(π/2 − femur − tibia)

# Apply coxa rotation (subtract INIT_COXA_ANGLE first to get IK-frame coxa):
ik_coxa = coxa − INIT_COXA_ANGLE[leg_idx]
fx = plane_distance · sin(ik_coxa)
fy = plane_distance · cos(ik_coxa)
fz = plane_depth
```

This is not in the codebase yet — derive it from the IK if you need it.

## Body-frame ↔ IK-frame conventions

The IK frame is leg-local: `+Y` always points away from the body along the leg's natural extension, and the math is identical for all six legs. Differences between legs are absorbed into `INIT_COXA_ANGLE` and the body-frame mounting (the coxa body's `quat` in `models/phantomx.xml`).

Body frame in the MJCF:
- `+X` = forward
- `+Y` = left
- `+Z` = up

Coxa body-frame yaw, derived from the coxa quaternions in `phantomx.xml`:

| Leg | Body yaw |
|---|---|
| RR | -135° |
| RM |  -90° |
| RF |  -45° |
| LR | +135° |
| LM |  +90° |
| LF |  +45° |

This is informational — the IK doesn't need it because `INIT_COXA_ANGLE` already accounts for it. You only need body yaw if you're directly mapping body-frame foot targets to IK-frame inputs (e.g., in a custom omnidirectional gait).

## MJCF axis-flip conventions

The MJCF model has an asymmetric joint convention you MUST account for in your control output:

| Joint | Right legs (RR/RM/RF) | Left legs (LR/LM/LF) |
|---|---|---|
| Coxa  | axis="0 0 1"  | axis="0 0 1"  |
| Femur | axis="0 0 -1" | axis="0 0 1"  |
| Tibia | axis="0 0 1"  | axis="0 0 -1" |

All coxa joints share `+Z`, but femur and tibia axes are flipped between left and right. The IK formula above produces angles in a consistent right-leg convention. To apply them to left legs in the MJCF you must negate **all three** outputs:

```python
c, f, t = ik(fx, fy, fz, leg_idx)
if leg_idx >= 3:           # LR, LM, LF
    c, f, t = -c, -f, -t
```

(Coxa is also negated because the body-frame symmetry of the legs means `+coxa` rotates left and right legs in opposite body-frame directions, even though the joint axis is identical.)

This same flip applies when computing `NEUTRAL_POSE` — see `_compute_neutral_pose()`.

## Tripod gait offsets

Stage-1 gait pattern: legs `[RR, RM, RF, LR, LM, LF]` start in tripod groups `[1, 0, 1, 0, 1, 0]`. Group 0 swings while group 1 pushes; groups flip every `CYCLE_LENGTH` (50) steps.

Per step, each leg gets a 3D offset added to its rest foot position:

```
phase = cycle_period · π / CYCLE_LENGTH
period_height   = sin(phase)        # 0 → 1 → 0 over one swing
period_distance = cos(phase)        # +1 → 0 → -1 over one swing

For swing leg (group 0):
  Δfoot_x = cmd_x · period_distance
  Δfoot_y = cmd_y · period_distance
  Δfoot_z = LEG_LIFT_HEIGHT · period_height       # 0.038 m max lift

For stance leg (group 1):
  Δfoot_x = -cmd_x · period_distance
  Δfoot_y = -cmd_y · period_distance
  Δfoot_z = 0
```

These offsets are in **body frame**. Apply per-leg `sign = -1.0 if leg_idx < 3 else +1.0` to `Δfoot_y` when feeding into IK (right legs negate Y; X is not mirrored). This matches the original ROS controller convention.

In the current implementation only forward (`cmd_x > 0`) produces clean motion — strafing/backward gait is not yet validated.

## Quick-reference: full pipeline (one leg)

```python
# Inputs: cmd_x, cmd_y in body frame; cycle_period and group from gait state.
foot_offset = tripod_gait(cmd_x, cmd_y, cycle_period, group_state)[leg_idx]

sign = -1.0 if leg_idx < 3 else 1.0
fx = -INIT_FOOT_POS_X[leg_idx] + foot_offset[0]
fy =  INIT_FOOT_POS_Y[leg_idx] + sign * foot_offset[1]
fz =  INIT_FOOT_POS_Z[leg_idx] - foot_offset[2]      # gait z is lift, IK z is depth

c, f, t = ik(fx, fy, fz, leg_idx)
if leg_idx >= 3:
    c, f, t = -c, -f, -t

ctrl[leg_idx*3 + 0] = c
ctrl[leg_idx*3 + 1] = f
ctrl[leg_idx*3 + 2] = t
```

That's the whole gait → joint-target pipeline used by `walk_test.py` and the residual-policy training base controller.
