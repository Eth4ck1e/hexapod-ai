"""
gait.controller_jax — pure-JAX hexapod gait controller (port of controller.py).

Two-tier design:
  * `build_params(model_path)` — runs the numpy `Controller` once to do the
    MuJoCo-dependent calibration, then packs all per-leg constants into a
    `GaitParams` NamedTuple of `jnp.ndarray`s. Slow, not JIT'd, called once
    at startup.
  * `predict(params, cmd, t)` / `predict_with_feet(params, cmd, t)` — pure
    JAX. No MuJoCo. JIT-compatible, vmap-compatible. Identical math to the
    numpy controller.

Numerical parity with the numpy controller is validated sub-mm in the
`__main__` block at the bottom of this file (run as a script).
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from gait.controller import (
    COXA_LENGTH,
    FEMUR_LENGTH,
    TIBIA_LENGTH,
    Controller,
)
from gait.controller import (
    COXA_POS_BODY as _COXA_POS_BODY_NP,
)
from gait.controller import (
    LEG_PHASE as _LEG_PHASE_NP,
)


class GaitParams(NamedTuple):
    """All per-leg constants the JAX hot path needs.

    Shapes / dtypes documented inline. Everything jnp.float32 except phase
    indexing helpers which stay int."""
    yaw_c:             jnp.ndarray   # (6,)   cos(LEG_BODY_YAW)
    yaw_s:             jnp.ndarray   # (6,)   sin(LEG_BODY_YAW)
    coxa_pos_body:     jnp.ndarray   # (6, 3) coxa joint pos in body frame
    leg_phase:         jnp.ndarray   # (6,)   tripod phase offsets
    foot_rest_coxa:    jnp.ndarray   # (6, 3) widened rest in coxa-local
    body_origin_coxa:  jnp.ndarray   # (6, 3) body origin in each leg's coxa-local
    leg_path_deltas:   jnp.ndarray   # (6, N, 2) per-leg coxa-local path xy
    leg_offsets:       jnp.ndarray   # (6, 3) formula→MJCF joint offsets
    joint_signs:       jnp.ndarray   # (6, 3) per-joint sign flips
    spin_ref_radius:   jnp.ndarray   # () scalar
    max_speed:         jnp.ndarray   # () scalar
    max_yaw_rate:      jnp.ndarray   # () scalar
    gait_period:       jnp.ndarray   # () scalar
    path_radius:       jnp.ndarray   # () scalar  (R for canonical_path)
    path_res:          int           # static — used as int for shape ops


# ============================================================================
# Build constants from numpy controller (slow path, runs once)
# ============================================================================
def build_params(model_path: str,
                 gait_period: float = None,
                 path_radius: float = None,
                 path_res:    int   = None,
                 default_stance_width: float = None,
                 dtype = jnp.float32) -> GaitParams:
    """Calibrate via the numpy Controller, copy constants into jnp arrays."""
    ctrl = Controller(
        model_path,
        gait_period=gait_period,
        path_radius=path_radius,
        path_res=path_res,
        default_stance_width=default_stance_width,
    )
    return GaitParams(
        yaw_c            = jnp.asarray(ctrl._yaw_c,            dtype=dtype),
        yaw_s            = jnp.asarray(ctrl._yaw_s,            dtype=dtype),
        coxa_pos_body    = jnp.asarray(_COXA_POS_BODY_NP,      dtype=dtype),
        leg_phase        = jnp.asarray(_LEG_PHASE_NP,          dtype=dtype),
        foot_rest_coxa   = jnp.asarray(ctrl.FOOT_REST_COXA,    dtype=dtype),
        body_origin_coxa = jnp.asarray(ctrl.BODY_ORIGIN_COXA,  dtype=dtype),
        leg_path_deltas  = jnp.asarray(ctrl.LEG_PATH_DELTAS,   dtype=dtype),
        leg_offsets      = jnp.asarray(ctrl.LEG_OFFSETS,       dtype=dtype),
        joint_signs      = jnp.asarray(ctrl._joint_signs_array, dtype=dtype),
        spin_ref_radius  = jnp.asarray(ctrl.spin_ref_radius,   dtype=dtype),
        max_speed        = jnp.asarray(ctrl.MAX_SPEED,         dtype=dtype),
        max_yaw_rate     = jnp.asarray(ctrl.MAX_YAW_RATE,      dtype=dtype),
        gait_period      = jnp.asarray(ctrl.gait_period,       dtype=dtype),
        path_radius      = jnp.asarray(ctrl.path_radius,       dtype=dtype),
        path_res         = int(ctrl.path_res),
    )


# ============================================================================
# Hot path — pure JAX
# ============================================================================
def _canonical_path_batch(s, R):
    """Vectorized canonical path. s: (6,), R: scalar. Returns (px, pz) each (6,)."""
    # Swing branch (s < 0.5): theta = π·(1 - 2s); px = R·cos(θ), pz = R·sin(θ).
    theta = jnp.pi * (1.0 - 2.0 * s)
    swing_px = R * jnp.cos(theta)
    swing_pz = R * jnp.sin(theta)
    # Stance branch: linear from (+R, 0) to (-R, 0).
    ss = (s - 0.5) / 0.5
    stance_px = R * (1.0 - 2.0 * ss)
    stance_pz = jnp.zeros_like(s)
    is_swing = s < 0.5
    px = jnp.where(is_swing, swing_px, stance_px)
    pz = jnp.where(is_swing, swing_pz, stance_pz)
    return px, pz


def _coxa_local_to_body_batch(p_coxa, params):
    """(6, 3) coxa-local → (6, 3) body-frame, vectorized."""
    out_x = params.yaw_c * p_coxa[:, 0] - params.yaw_s * p_coxa[:, 1]
    out_y = params.yaw_s * p_coxa[:, 0] + params.yaw_c * p_coxa[:, 1]
    out_z = p_coxa[:, 2]
    return jnp.stack([out_x, out_y, out_z], axis=1) + params.coxa_pos_body


def _body_to_coxa_local_batch(p_body, params):
    """(6, 3) body-frame → (6, 3) coxa-local, vectorized."""
    rel = p_body - params.coxa_pos_body
    out_x =  params.yaw_c * rel[:, 0] + params.yaw_s * rel[:, 1]
    out_y = -params.yaw_s * rel[:, 0] + params.yaw_c * rel[:, 1]
    out_z = rel[:, 2]
    return jnp.stack([out_x, out_y, out_z], axis=1)


def _apply_tilt(rest_coxa, pitch, roll, params):
    """Always applied (no Python branch). Identity rotation when pitch=roll=0."""
    cp = jnp.cos(pitch); sp = jnp.sin(pitch)
    cr = jnp.cos(roll);  sr = jnp.sin(roll)
    R_inv = jnp.array([
        [ cp,        0.0,     sp     ],
        [-sp * sr,   cr,      cp * sr],
        [-sp * cr,  -sr,      cp * cr],
    ])
    rest_body   = _coxa_local_to_body_batch(rest_coxa, params)
    tilted_body = rest_body @ R_inv.T
    return _body_to_coxa_local_batch(tilted_body, params)


def _ik_raw_batch(feet_coxa):
    """Vectorized closed-form IK over all 6 legs. (6, 3) coxa-local → (6, 3) raw."""
    fx = feet_coxa[:, 0]
    fy = feet_coxa[:, 1]
    fz = feet_coxa[:, 2]
    coxa = jnp.arctan2(fy, fx)
    r    = jnp.hypot(fx, fy)
    x_fp = r - COXA_LENGTH
    z_fp = fz
    D    = jnp.hypot(x_fp, z_fp)
    D    = jnp.minimum(D, FEMUR_LENGTH + TIBIA_LENGTH - 1e-6)
    D    = jnp.maximum(D, abs(FEMUR_LENGTH - TIBIA_LENGTH) + 1e-6)
    cos_a = (FEMUR_LENGTH**2 + D**2 - TIBIA_LENGTH**2) / (2.0 * FEMUR_LENGTH * D)
    cos_g = (FEMUR_LENGTH**2 + TIBIA_LENGTH**2 - D**2) / (2.0 * FEMUR_LENGTH * TIBIA_LENGTH)
    cos_a = jnp.clip(cos_a, -1.0, 1.0)
    cos_g = jnp.clip(cos_g, -1.0, 1.0)
    alpha = jnp.arccos(cos_a)
    gamma = jnp.arccos(cos_g)
    beta  = jnp.arctan2(z_fp, x_fp)
    femur = beta + alpha
    tibia = jnp.pi - gamma
    return jnp.stack([coxa, femur, tibia], axis=1)


def _compute_foot_targets_coxa(params, cmd, t):
    """Pure-JAX port of Controller._compute_foot_targets_coxa."""
    vx, vy, wz, pitch, roll, dh, dw, sx, sy = (cmd[i] for i in range(9))

    # 1. Stance overlay: widen + height offset.
    rest = params.foot_rest_coxa + jnp.stack([
        jnp.full((6,), dw),
        jnp.zeros((6,)),
        jnp.full((6,), dh),
    ], axis=1)

    # 2. Tilt overlay (unconditional — identity when pitch=roll=0).
    rest = _apply_tilt(rest, pitch, roll, params)

    # 3. Body shift overlay (always applied — collapses to no-op when
    # sx=sy=0). See gait.controller._compute_foot_targets_coxa for the
    # body-frame → coxa-local derivation.
    shift_dx = -(params.yaw_c * sx + params.yaw_s * sy)
    shift_dy =  (params.yaw_s * sx - params.yaw_c * sy)
    rest = rest + jnp.stack([shift_dx, shift_dy, jnp.zeros((6,))], axis=1)

    # 4. Speed scales — use jnp.where (no Python branches on traced values).
    eps   = 1e-9
    speed = jnp.hypot(vx, vy)
    stride_scale = jnp.where(speed > eps,
                             jnp.minimum(speed / params.max_speed, 1.0),
                             0.0)
    spin_scale   = jnp.where(jnp.abs(wz) > eps,
                             jnp.clip(wz / params.max_yaw_rate, -1.0, 1.0),
                             0.0)
    gait_active  = jnp.maximum(stride_scale, jnp.abs(spin_scale))

    # Phase + heading.
    s_global = (t / params.gait_period) % 1.0
    heading  = jnp.where(speed > eps, jnp.arctan2(vy, vx), 0.0)
    c_h = jnp.cos(heading); s_h = jnp.sin(heading)

    # Per-leg phase indices.
    s_i   = (s_global + params.leg_phase) % 1.0                       # (6,)
    n_idx = (s_i * params.path_res).astype(jnp.int32) % params.path_res  # (6,)

    # Translation: gather per-leg path xy at phase index, rotate by heading.
    path_xy = params.leg_path_deltas[jnp.arange(6), n_idx]            # (6, 2)
    trans_x = stride_scale * (c_h * path_xy[:, 0] - s_h * path_xy[:, 1])
    trans_y = stride_scale * (s_h * path_xy[:, 0] + c_h * path_xy[:, 1])

    # Lift + spin reference path samples.
    px_arr, pz_arr = _canonical_path_batch(s_i, params.path_radius)

    # Spin contribution: rotate (rest - body_origin) around body origin by
    # dtheta = spin_scale * px / spin_ref_radius. Always applied; spin_x/y
    # collapse to ~0 when spin_scale ≈ 0.
    cx = params.body_origin_coxa[:, 0]
    cy = params.body_origin_coxa[:, 1]
    rx = rest[:, 0] - cx
    ry = rest[:, 1] - cy
    dtheta = spin_scale * px_arr / params.spin_ref_radius
    cs = jnp.cos(dtheta)
    ss = jnp.sin(dtheta)
    spin_x = (cx + (cs * rx - ss * ry)) - rest[:, 0]
    spin_y = (cy + (ss * rx + cs * ry)) - rest[:, 1]

    lift = pz_arr * gait_active

    # Combine. When gait_active ≈ 0, trans/spin/lift all collapse to ~0 and
    # we recover `rest` to within float precision.
    feet_x = rest[:, 0] + trans_x + spin_x
    feet_y = rest[:, 1] + trans_y + spin_y
    feet_z = rest[:, 2] + lift
    return jnp.stack([feet_x, feet_y, feet_z], axis=1)


def _joints_from_feet_coxa(feet_coxa, params):
    raw  = _ik_raw_batch(feet_coxa)
    mjcf = params.joint_signs * raw + params.leg_offsets
    return mjcf.reshape(-1)


def predict(params: GaitParams, cmd: jnp.ndarray, t) -> jnp.ndarray:
    """cmd (9,) + sim time → 18 joint targets."""
    feet_coxa = _compute_foot_targets_coxa(params, cmd, t)
    return _joints_from_feet_coxa(feet_coxa, params)


def predict_with_feet(params: GaitParams, cmd: jnp.ndarray, t) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Same cost as predict(): returns (joints_18, feet_body_6x3)."""
    feet_coxa = _compute_foot_targets_coxa(params, cmd, t)
    joints    = _joints_from_feet_coxa(feet_coxa, params)
    feet_body = _coxa_local_to_body_batch(feet_coxa, params)
    return joints, feet_body


# ============================================================================
# Parity test (run this file as a script)
# ============================================================================
if __name__ == "__main__":
    # Enable x64 for the parity test so JAX matches numpy's float64 precision.
    # In production training we run float32 — see the throughput section below
    # for the actually-relevant performance numbers.
    jax.config.update("jax_enable_x64", True)

    import time

    import numpy as np

    MODEL_PATH = "models/phantomx_simple.xml"

    print(f"building params from {MODEL_PATH}...", end=" ", flush=True)
    t0 = time.perf_counter()
    params = build_params(MODEL_PATH, dtype=jnp.float64)
    print(f"done ({time.perf_counter() - t0:.2f}s)")

    np_ctrl = Controller(MODEL_PATH)

    # JIT both functions.
    jit_predict = jax.jit(predict)
    jit_predict_with_feet = jax.jit(predict_with_feet)

    # Warm up.
    cmd0 = jnp.zeros(9)
    _ = jit_predict(params, cmd0, 0.0).block_until_ready()
    j0_jax, f0_jax = jit_predict_with_feet(params, cmd0, 0.0)
    _ = j0_jax.block_until_ready(); _ = f0_jax.block_until_ready()

    # Parity sweep.
    rng = np.random.default_rng(42)
    N = 1000
    max_joint_err = 0.0
    max_feet_err  = 0.0
    for k in range(N):
        # Random cmd within plausible bounds.
        cmd_np = np.array([
            rng.uniform(-np_ctrl.MAX_SPEED, np_ctrl.MAX_SPEED),     # vx
            rng.uniform(-np_ctrl.MAX_SPEED, np_ctrl.MAX_SPEED),     # vy
            rng.uniform(-np_ctrl.MAX_YAW_RATE, np_ctrl.MAX_YAW_RATE), # wz
            rng.uniform(-0.15, 0.15),                                # pitch
            rng.uniform(-0.15, 0.15),                                # roll
            rng.uniform(-0.02, 0.02),                                # dh
            rng.uniform(-0.01, 0.02),                                # dw
            0.0, 0.0,                                                # reserved
        ])
        t  = float(rng.uniform(0.0, 5.0))

        j_np, f_np = np_ctrl.predict_with_feet(cmd_np, t)
        cmd_jax = jnp.asarray(cmd_np)
        j_jax, f_jax = jit_predict_with_feet(params, cmd_jax, t)
        j_jax = np.asarray(j_jax)
        f_jax = np.asarray(f_jax)

        max_joint_err = max(max_joint_err, float(np.max(np.abs(j_jax - j_np))))
        max_feet_err  = max(max_feet_err,  float(np.max(np.abs(f_jax - f_np))))

    print(f"parity over {N} random cmds:")
    print(f"  max joint angle error: {max_joint_err:.3e} rad  ({math.degrees(max_joint_err)*3600:.1f} arcsec)")
    print(f"  max foot pos error:    {max_feet_err *1000:.3e} mm")

    # Thresholds tuned for fp64. fp32 noise is ~1e-5 rad / 1e-7 m which is
    # already well below anything the policy or env cares about.
    if max_joint_err < 1e-9 and max_feet_err < 1e-10:
        print("PARITY OK (fp64 — bit-equivalent to numpy)")
    elif max_joint_err < 1e-4 and max_feet_err < 1e-5:
        print("PARITY OK (within numerical-precision noise)")
    else:
        print("PARITY MISMATCH — investigate before proceeding")

    # Throughput check (single-cmd JIT'd predict).
    n_iters = 10000
    cmd_t = jnp.array([0.1, 0.0, 0.2, 0.05, 0.0, 0.0, 0.005, 0.0, 0.0])
    _ = jit_predict(params, cmd_t, 0.0).block_until_ready()
    t0 = time.perf_counter()
    for k in range(n_iters):
        out = jit_predict(params, cmd_t, k * 0.005)
    out.block_until_ready()
    dt = time.perf_counter() - t0
    print(f"\nJIT'd single predict: {n_iters / dt:,.0f} calls/sec  ({dt*1e6/n_iters:.1f} µs/call)")

    # Throughput check (vmap over 4096 envs — the MJX target batch size).
    BATCH = 4096
    vmap_predict = jax.jit(jax.vmap(predict, in_axes=(None, 0, 0)))
    cmds_batch = jnp.broadcast_to(cmd_t, (BATCH, 9))
    ts_batch   = jnp.linspace(0.0, 5.0, BATCH)
    _ = vmap_predict(params, cmds_batch, ts_batch).block_until_ready()
    t0 = time.perf_counter()
    for k in range(100):
        out = vmap_predict(params, cmds_batch, ts_batch + k * 0.005)
    out.block_until_ready()
    dt = time.perf_counter() - t0
    total = 100 * BATCH
    print(f"vmap'd predict (BATCH={BATCH}): {total / dt:,.0f} samples/sec  ({dt*1e3/100:.2f} ms/batch)")
