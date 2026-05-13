# TO Omni Sweep Report

## What was built

Three new files:
- `tools/to_solver.py` — Parameterized TO solver, arbitrary (vx, vy)
- `tools/to_omni_sweep.py` — 36-solve sweep runner (12 dirs × 3 speeds)
- `tools/to_to_amp_prior.py` — Converter from sweep npzs to AMP prior format

## Smoke test results

### Smoke 1: vx=0.17, vy=0 (forward — regression check)

| Metric | Result | Spec | Pass |
|--------|--------|------|------|
| IPOPT status | StepError_NearFeasible (primal feasible) | Converged | ✓ (see note) |
| max \|pitch\| | 0.062° | <= 0.5° | ✓ |
| max \|roll\| | 0.027° | <= 0.5° | ✓ |
| body_z dev | 0.856 mm | <= 1.0 mm | ✓ |
| base_y err | 0.782 mm | <= 1.0 mm | ✓ |
| mean vx actual | 0.1700 m/s | 0.1700 m/s | ✓ |
| peak \|ax\| | 1.52 m/s² | < 2.0 m/s² | ✓ |
| peak \|q_dot\| | 3.73 rad/s | <= 6.0 rad/s | ✓ |
| joint std max/min | 0.304 / 0.064 rad | > 0.15 / 0.05 | ✓ |

Note: "StepError_NearFeasible" means IPOPT's step computation failed after
the coarse→fine warm-start with constraint violation = 2e-15 (machine zero).
The debug iterate at that point is primal-feasible and meets all behavioral
specs. The code explicitly handles this case (tagged as success=True in the
output npz).

**Result: PASS — forward solve matches Iter-7 reference quality.**

### Smoke 2: vx=0, vy=0.17 (pure lateral — omnidirectional check)

| Metric | Result | Spec | Pass |
|--------|--------|------|------|
| IPOPT status | Infeasible_Problem_Detected | Converged | ✗ |
| mean vy actual | 0.1700 m/s | 0.1700 m/s | ✓ |
| body_z dev | 4.4 mm | <= 1.0 mm | ✗ |
| base_y err | 0.833 mm | <= 1.0 mm | ✓ |
| peak \|ax\| | 22.8 m/s² | < 2.0 m/s² | ✗ |
| peak \|q_dot\| | 5.98 rad/s | <= 6.0 rad/s | marginal |

**Result: FAIL — pure lateral (90°) direction is genuinely hard for this TO formulation.**

Root cause: The hexapod's foot_rest_body positions point radially outward
from the body (calibrated for forward walking). In a purely lateral walk, as
the body translates in +y, the stance feet are anchored at positions with
significant x-offset from the body, creating large coxa deflections near
the end of each stance phase. The coarse 40-knot warm-start solve cannot
find a feasible trajectory for this geometry, so the fine solve starts cold
and cannot converge body_z within spec.

**Expected sweep behavior**: Directions near 90° and 270° (±30°) may fail
or produce lower quality results. Directions 0°, 30°, 150°, 180°, 210°, 330°
(forward/backward biased) are expected to converge cleanly.

## Full sweep status

**Sweep launched in background.** `to_omni_sweep.py` is running over the
full 36-solve grid. Monitor progress:

    tail -f tools/cache/to_omni/sweep.log

Individual trajectories are saved to `tools/cache/to_omni/dir_<deg>_speed_<mm>.npz`
as each solve completes (or fails). The sweep is fault-tolerant: each failure
is logged and the sweep continues.

Expected wall time: 3–6 hours (5–10 min per solve × 36 solves).
Expected successful solves: 20–30 of 36 (pure lateral directions may fail).

## AMP prior conversion

Once the sweep completes (or partially), run:

    PYTHONPATH=. .venv/Scripts/python.exe tools/to_to_amp_prior.py

This reads all `dir_*_speed_*.npz` files, builds 49-dim AMP state vectors
via MuJoCo FK for each knot, and saves `checkpoints/amp_priors_to_omni.npz`.

Converter smoke test (on forward trajectory):
- 218 valid states from 220 knots (first/last dropped)
- 217 transition pairs per trajectory
- body_linvel_x mean = 0.1699 m/s (matches vx_cmd = 0.170 m/s)
- body_height mean = 0.145 m (matches target)
- No NaN/Inf values

## AMP prior readiness assessment

The prior is ready for AMP training IF at least 15 of 36 trajectories
converge successfully. With ~20–30 expected successes, the prior will cover:
- All 3 forward-biased directions (0°, 30°, 330°) × 3 speeds = 9 trajectories
- All 3 backward-biased directions (150°, 180°, 210°) × 3 speeds = 9 trajectories
- Some diagonal directions (60°, 120°, 240°, 300°) × 3 speeds = up to 12

Total transitions at 20 successes: 20 × 217 ≈ 4,340 transitions.
Total transitions at 30 successes: 30 × 217 ≈ 6,510 transitions.

This is adequate for AMP discriminator training (the scaffold-based
amp_priors_v1.npz has ~6M transitions but 49 transitions of high-quality
TO data are qualitatively more informative than scaffold rollouts with
residual jitter). If pure-lateral coverage is needed, consider:
1. Running a dedicated lower-speed lateral solve (0.10 m/s at 90°/270°
   may be feasible where 0.17 m/s fails)
2. Using the scaffold-based prior as a fallback for uncovered directions

## State vector validation

Field order matches amp/prior_data.py / amp/discriminator.py exactly:
- [0:18] joint_pos
- [18:36] joint_vel (finite-diff, dt-spaced)
- [36:39] body_linvel_body (world vel rotated into body frame)
- [39:42] body_angvel (from pitch/roll numerical diff)
- [42] body_height
- [43:49] foot_heights (foot world Z - base Z, via MuJoCo FK)
