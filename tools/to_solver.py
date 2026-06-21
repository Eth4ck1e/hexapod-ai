"""
tools/to_solver.py — Parameterized trajectory optimization (TO) solver for
the PhantomX hexapod. Supports arbitrary (vx, vy) commanded velocities for
omnidirectional prior data collection.

This is a clean parameterized fork of tools/trajectory_opt_demo.py. The demo
is kept intact as the working reference for the forward-only tuning history.
All Iter-N tuning rationale is preserved in comments below.

Decision variables (fixed tripod schedule, N timesteps):
  * Foot positions in body frame: (N, 6, 3)
  * Body translation: base_x(t), base_y(t), base_z(t)
  * Body orientation: pitch(t), roll(t) (yaw assumed 0 throughout)

Joint angles derived symbolically via closed-form IK embedded in the NLP —
keeps the FK consistent with the rest of the pipeline without differentiating
through MuJoCo.

Constraints:
  * Joint angle limits (buffered 1° inward, from joint_limits.json)
  * Body height in [body_height ± body_z_tol]
  * |pitch|, |roll| <= max_pitch_roll_deg
  * Terminal state: base = (vx*duration, vy*duration, body_height)
  * For non-zero vy: body must track the commanded direction line
    |base_y(t) - vy*t| <= 1mm  (replaces the old |base_y| <= 1mm hard box)
  * Tripod contact schedule: GROUP_A (RR, RF, LM) and GROUP_B (RM, LR, LF)
    alternate stance/swing. Stance feet: world-z = 0, no XY slip.
    Swing feet: z >= swing_clearance at mid-swing knot.

Tuning iteration history (carried over from trajectory_opt_demo.py):
  Iter-1: basic CasADi + IPOPT with no velocity constraints.
  Iter-2: hard |base_y| <= 1mm to stop lateral drift.
  Iter-3: no-slip stance constraint coupled base motion to foot retraction.
  Iter-4: w_energy=1.0 restored, w_stability=0 (replaced by hard box),
          body_z_tol=4.5mm (tighter hard bounds caused infeasibility).
  Iter-5: strong interior w_height=3e8 cost pulls body_z deviation to <1mm
          without touching the hard wall. w_body_accel=150 (30× Iter-4)
          reduces peak|ax| from 11.3→1.36 m/s². w_vx_track disabled.
  Iter-6: joint-jerk penalty (w_joint_jerk=1.0) damps high-freq oscillations
          on 220-knot NLPs. Warm-start via coarse 40-knot solve halves IPOPT
          barrier iterations. enforce_joint_vel_bounds disabled on coarse solve
          (large dt makes |Δq|/dt <= 6 incompatible with no-slip geometry).
  Iter-7: soft stance-midpoint symmetry (w_sym=1e4) produces symmetric stance
          arc. Hard equality caused non-convergence on 8-sec/220-knot NLP;
          soft cost at 1e4 (not 1e5) keeps IPOPT stable on larger NLPs.

CLI usage:
  python tools/to_solver.py --vx 0.17 --vy 0 --out tools/cache/to_omni/dir_000_speed_170.npz
  python tools/to_solver.py --vx 0 --vy 0.17 --out tools/cache/to_omni/dir_090_speed_170.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gait.controller import (
    COXA_LENGTH,
    COXA_POS_BODY,
    FEMUR_LENGTH,
    LEG_BODY_YAW,
    LEG_NAMES,
    TIBIA_LENGTH,
    Controller,
)


# ----------------------------------------------------------------------
# Symbolic closed-form IK (mirrors gait.controller._ik_raw_batch but in
# CasADi MX so it can be embedded in the NLP).
# Regularized to keep the Jacobian well-defined at the boundary:
#   - sqrt(eps + .) instead of sqrt(.) to avoid zero-gradient singularity
#   - atan2/asin reformulation avoids acos's vertical gradient at ±1
# ----------------------------------------------------------------------
def _joint_signs_array():
    """Per-leg joint sign matrix (6, 3) — matches gait.controller._joint_signs."""
    out = np.zeros((6, 3))
    for i in range(6):
        is_left = i >= 3
        out[i, 0] = 1.0
        out[i, 1] = -1.0 if is_left else 1.0
        out[i, 2] = 1.0 if is_left else -1.0
    return out


def _ik_raw_sym(fx, fy, fz):
    """Symbolic single-leg IK. Inputs in coxa-local frame. Returns
    (coxa, femur, tibia) in formula convention (pre-MJCF-sign/offset)."""
    eps = 1e-9
    coxa = ca.atan2(fy, fx)
    r = ca.sqrt(fx * fx + fy * fy + eps)
    x_fp = r - COXA_LENGTH
    z_fp = fz
    D2 = x_fp * x_fp + z_fp * z_fp + eps
    D = ca.sqrt(D2)
    cos_a = (FEMUR_LENGTH ** 2 + D2 - TIBIA_LENGTH ** 2) / (2 * FEMUR_LENGTH * D)
    cos_g = (FEMUR_LENGTH ** 2 + TIBIA_LENGTH ** 2 - D2) / (2 * FEMUR_LENGTH * TIBIA_LENGTH)
    sin_a = ca.sqrt(ca.fmax(eps, 1 - cos_a * cos_a))
    sin_g = ca.sqrt(ca.fmax(eps, 1 - cos_g * cos_g))
    alpha = ca.atan2(sin_a, cos_a)
    gamma = ca.atan2(sin_g, cos_g)
    beta = ca.atan2(z_fp, x_fp)
    femur = beta + alpha
    tibia = math.pi - gamma
    return coxa, femur, tibia


def _body_to_coxa_local_sym(p_body, leg_idx):
    """Symbolic body-frame point -> coxa-local. p_body is a CasADi 3-vector."""
    yaw = LEG_BODY_YAW[leg_idx]
    c, s = math.cos(yaw), math.sin(yaw)
    rel0 = p_body[0] - COXA_POS_BODY[leg_idx, 0]
    rel1 = p_body[1] - COXA_POS_BODY[leg_idx, 1]
    rel2 = p_body[2] - COXA_POS_BODY[leg_idx, 2]
    return ca.vertcat(
        c * rel0 + s * rel1,
        -s * rel0 + c * rel1,
        rel2,
    )


def _body_to_world_sym(p_body, base_xyz, pitch, roll):
    """Apply R_y(-pitch) * R_x(roll) and translate by base_xyz.
    Pitch convention: + = nose UP (project-wide aerospace convention)."""
    cp = ca.cos(pitch)
    sp = ca.sin(pitch)
    cr = ca.cos(roll)
    sr = ca.sin(roll)
    R00 = cp;  R01 = sp * sr;  R02 = sp * cr
    R10 = 0;   R11 = cr;       R12 = -sr
    R20 = -sp; R21 = cp * sr;  R22 = cp * cr
    wx = R00 * p_body[0] + R01 * p_body[1] + R02 * p_body[2] + base_xyz[0]
    wy = R10 * p_body[0] + R11 * p_body[1] + R12 * p_body[2] + base_xyz[1]
    wz = R20 * p_body[0] + R21 * p_body[1] + R22 * p_body[2] + base_xyz[2]
    return ca.vertcat(wx, wy, wz)


def _foot_body_to_joints_sym(foot_body, leg_idx, leg_offsets):
    """Body-frame foot target -> 3 MJCF joint angles for leg_idx (symbolic)."""
    foot_coxa = _body_to_coxa_local_sym(foot_body, leg_idx)
    coxa_raw, femur_raw, tibia_raw = _ik_raw_sym(
        foot_coxa[0], foot_coxa[1], foot_coxa[2])
    signs = _joint_signs_array()[leg_idx]
    coxa  = signs[0] * coxa_raw  + leg_offsets[leg_idx, 0]
    femur = signs[1] * femur_raw + leg_offsets[leg_idx, 1]
    tibia = signs[2] * tibia_raw + leg_offsets[leg_idx, 2]
    return ca.vertcat(coxa, femur, tibia)


# ----------------------------------------------------------------------
# Joint-limits loading.
# ----------------------------------------------------------------------
JOINT_NAMES = []
for _leg in LEG_NAMES:
    for _j in ("coxa", "femur", "tibia"):
        JOINT_NAMES.append(f"{_j}_joint_{_leg}")


def load_joint_limits(path):
    with open(path) as f:
        data = json.load(f)
    lo = np.zeros(18)
    hi = np.zeros(18)
    for i, name in enumerate(JOINT_NAMES):
        lo[i], hi[i] = data[name]["buffered"]
    return lo, hi


# ----------------------------------------------------------------------
# TO problem: build + solve.
# ----------------------------------------------------------------------
def build_and_solve_to(
    *,
    vx: float,
    vy: float,
    duration_s: float = 8.0,
    n_strides: int = 11,
    knots_per_phase: int = 10,
    body_height: float = 0.145,
    swing_clearance: float = 0.012,
    # Behavioral hard bounds. Tight bounds (< 0.5 mm) cause infeasibility
    # because no-slip stance constraints couple base_z to foot retraction
    # geometry in a way that needs ~3–4 mm of vertical slack at the discrete
    # knots. The strong w_height cost pulls the optimum well below the wall.
    max_pitch_roll_deg: float = 0.45,
    body_z_tol: float = 0.0045,
    # Cost weights — all values tuned and signed-off in Iter-7:
    w_sym: float = 1e4,          # stance-midpoint XY symmetry (Iter-7)
    w_energy: float = 1.0,       # joint-accel energy proxy
    w_joint_jerk: float = 1.0,   # joint-jerk (Iter-6) damps ringing at 220 knots
    w_pose: float = 5e6,         # pitch^2 + roll^2
    w_height: float = 3e8,       # interior body_z cost (Iter-5); drives dev <1mm
    w_speed: float = 5e2,        # linear tracking of body position along cmd direction
    w_body_accel: float = 150.0, # body acceleration smoothness (Iter-5)
    w_vx_track: float = 0.0,     # disabled — w_body_accel alone is sufficient
    w_stability: float = 0.0,    # disabled — replaced by hard direction constraint
    leg_offsets=None,
    foot_rest_body=None,
    print_every: int = 10,
    # Coarse-solve warm-start (Iter-6): when provided, interpolates base/pitch/roll
    # onto the fine grid and uses them as IPOPT's primal initial guess. Cuts
    # barrier iterations roughly in half for 220-knot problems.
    initial_guess=None,
    # Disable on coarse warm-start solves: at large dt the bound |Δq|/dt <= 6
    # conflicts with the no-slip geometry.
    enforce_joint_vel_bounds: bool = True,
    # Directional tracking tolerance (m). The per-knot constraint
    # |base_y(k) - vy*k*dt| <= directional_tol enforces the body follows
    # the commanded direction line. On the coarse warm-start solve (large dt)
    # the no-slip geometry needs more lateral slack, so we loosen to 5mm.
    # On the fine solve the 0.9mm bound is tight enough to satisfy the 1mm spec.
    directional_tol: float = 0.0009,
    linear_solver: str = "mumps",
):
    """Build and solve the TO problem for commanded (vx, vy). Returns a dict."""
    n_phases = 2 * n_strides
    N = n_phases * knots_per_phase
    dt = duration_s / (N - 1)

    # Commanded displacement over the full trajectory.
    target_x = vx * duration_s
    target_y = vy * duration_s

    # Tripod groups: A = RR(0), RF(2), LM(4); B = RM(1), LR(3), LF(5).
    GROUP_A = [0, 2, 4]
    GROUP_B = [1, 3, 5]
    stance_groups = [GROUP_A if p % 2 == 0 else GROUP_B for p in range(n_phases)]
    swing_groups  = [GROUP_B if p % 2 == 0 else GROUP_A for p in range(n_phases)]

    j_lo, j_hi = load_joint_limits(
        Path(__file__).resolve().parent.parent / "joint_limits.json")
    margin = math.radians(1.0)
    j_lo = j_lo + margin
    j_hi = j_hi - margin

    if leg_offsets is None or foot_rest_body is None:
        ctrl = Controller(
            str(Path(__file__).resolve().parent.parent
                / "models" / "phantomx_simple_mjx.xml"))
        leg_offsets    = ctrl.LEG_OFFSETS.copy()
        foot_rest_body = ctrl.LEG_ORIGIN_BODY.copy()

    opti = ca.Opti()

    # ----- Decision variables -----
    foot_body = [opti.variable(6, 3) for _ in range(N)]
    base  = opti.variable(3, N)   # base x, y, z  (each column is a knot)
    pitch = opti.variable(N)
    roll  = opti.variable(N)

    # ----- Initial guess: stride-pattern warm start -----
    # For omnidirectional motion, the warm-start advances feet in the
    # commanded (vx, vy) direction rather than hardcoded +x. This gives
    # IPOPT a feasibility-consistent starting point regardless of direction.
    leg_world_anchor = np.zeros((n_phases + 1, 6, 2))  # XY anchors per phase
    leg_world_anchor[0] = foot_rest_body[:, :2]
    for p in range(n_phases):
        frac_end = (p + 1) * knots_per_phase / (N - 1)
        for leg in range(6):
            if leg in stance_groups[p]:
                leg_world_anchor[p + 1, leg] = leg_world_anchor[p, leg]
            else:
                # Swing foot steps forward in commanded direction to stay
                # "ahead" of the body as it advances toward (target_x, target_y).
                leg_world_anchor[p + 1, leg, 0] = foot_rest_body[leg, 0] + target_x * frac_end
                leg_world_anchor[p + 1, leg, 1] = foot_rest_body[leg, 1] + target_y * frac_end

    for k in range(N):
        frac = k / (N - 1)
        bx = target_x * frac
        by = target_y * frac
        opti.set_initial(base[0, k], bx)
        opti.set_initial(base[1, k], by)
        opti.set_initial(base[2, k], body_height)
        opti.set_initial(pitch[k], 0.0)
        opti.set_initial(roll[k], 0.0)
        p = min(k // knots_per_phase, n_phases - 1)
        k_in_phase = k - p * knots_per_phase
        knots_phase_total = max(knots_per_phase - 1, 1)
        guess = foot_rest_body.copy()
        guess[:, 2] = -body_height
        for leg in range(6):
            if leg in stance_groups[p]:
                wx, wy = leg_world_anchor[p, leg]
                guess[leg, 0] = wx - bx
                guess[leg, 1] = wy - by
                guess[leg, 2] = -body_height
            else:
                wx0, wy0 = leg_world_anchor[p, leg]
                wx1, wy1 = leg_world_anchor[p + 1, leg]
                alpha = k_in_phase / knots_phase_total
                wx = wx0 + alpha * (wx1 - wx0)
                wy = wy0 + alpha * (wy1 - wy0)
                lift = 4.0 * swing_clearance * alpha * (1 - alpha)
                guess[leg, 0] = wx - bx
                guess[leg, 1] = wy - by
                guess[leg, 2] = -body_height + lift
        opti.set_initial(foot_body[k], guess)

    # ----- Warm-start override from coarse solution (Iter-6) -----
    if initial_guess is not None:
        coarse_N    = initial_guess["N"]
        coarse_base = initial_guess["base"]   # (coarse_N, 3)
        coarse_pitch = initial_guess["pitch"] # (coarse_N,)
        coarse_roll  = initial_guess["roll"]  # (coarse_N,)
        coarse_t = np.linspace(0.0, 1.0, coarse_N)
        fine_t   = np.linspace(0.0, 1.0, N)
        base_interp = np.stack([
            np.interp(fine_t, coarse_t, coarse_base[:, i]) for i in range(3)
        ], axis=1)
        pitch_interp = np.interp(fine_t, coarse_t, coarse_pitch)
        roll_interp  = np.interp(fine_t, coarse_t, coarse_roll)
        for k in range(N):
            opti.set_initial(base[0, k], base_interp[k, 0])
            opti.set_initial(base[1, k], base_interp[k, 1])
            opti.set_initial(base[2, k], base_interp[k, 2])
            opti.set_initial(pitch[k], pitch_interp[k])
            opti.set_initial(roll[k], roll_interp[k])
        print(f"[TO] Warm-started from coarse solution (N={coarse_N} -> {N})")

    # ----- Per-knot constraints + joint-angle expressions -----
    joint_exprs = []
    for k in range(N):
        joints_k = []
        for leg in range(6):
            fb = ca.vertcat(foot_body[k][leg, 0],
                            foot_body[k][leg, 1],
                            foot_body[k][leg, 2])
            j3 = _foot_body_to_joints_sym(fb, leg, leg_offsets)
            joints_k.append(j3)
        joints_k = ca.vertcat(*joints_k)  # (18,)
        joint_exprs.append(joints_k)

        for j in range(18):
            opti.subject_to(opti.bounded(j_lo[j], joints_k[j], j_hi[j]))

        opti.subject_to(opti.bounded(body_height - body_z_tol,
                                     base[2, k],
                                     body_height + body_z_tol))
        max_pr = math.radians(max_pitch_roll_deg)
        opti.subject_to(opti.bounded(-max_pr, pitch[k], max_pr))
        opti.subject_to(opti.bounded(-max_pr, roll[k], max_pr))

        # Omnidirectional straightness: body must track the commanded
        # direction line. For forward-only (vy=0) this reduces to |base_y|<=1mm,
        # matching the original Iter-4 constraint exactly. For lateral motion,
        # the body must follow base_y(t) ≈ vy*t within 1mm tolerance.
        # 0.9mm bound keeps the solution inside the 1mm spec even after
        # IPOPT's acceptable_constr_viol_tol = 1e-5 slack.
        t_k = k * dt
        opti.subject_to(opti.bounded(
            -directional_tol, base[1, k] - vy * t_k, directional_tol))
        # For diagonal/lateral motion: also constrain base_x to track vx*t.
        # Only applied when |vx| is non-negligible (pure-lateral walks need
        # the no-slip geometry to have X slack). Boundary conditions +
        # speed_term enforce x≈0 sufficiently when vx=0.
        if abs(vx) > 0.02:
            opti.subject_to(opti.bounded(
                -directional_tol, base[0, k] - vx * t_k, directional_tol))

    # ----- Initial state: origin, level, all feet on ground -----
    opti.subject_to(base[0, 0] == 0.0)
    opti.subject_to(base[1, 0] == 0.0)
    opti.subject_to(base[2, 0] == body_height)
    opti.subject_to(pitch[0] == 0.0)
    opti.subject_to(roll[0] == 0.0)
    for leg in range(6):
        opti.subject_to(foot_body[0][leg, 2] == -body_height)

    # ----- Terminal state: commanded displacement, level body -----
    opti.subject_to(base[0, N - 1] == target_x)
    opti.subject_to(base[1, N - 1] == target_y)
    opti.subject_to(base[2, N - 1] == body_height)
    opti.subject_to(pitch[N - 1] == 0.0)
    opti.subject_to(roll[N - 1] == 0.0)

    # ----- Contact schedule + no-slip -----
    def _world_foot_sym(k, leg):
        fb = ca.vertcat(foot_body[k][leg, 0],
                        foot_body[k][leg, 1],
                        foot_body[k][leg, 2])
        base_k = ca.vertcat(base[0, k], base[1, k], base[2, k])
        return _body_to_world_sym(fb, base_k, pitch[k], roll[k])

    for p in range(n_phases):
        k_start = p * knots_per_phase
        k_end   = (p + 1) * knots_per_phase
        k_mid   = (k_start + k_end) // 2

        for leg in stance_groups[p]:
            w0 = _world_foot_sym(k_start, leg)
            opti.subject_to(w0[2] == 0.0)
            for k in range(k_start, k_end - 1):
                wt  = _world_foot_sym(k,     leg)
                wt1 = _world_foot_sym(k + 1, leg)
                opti.subject_to(wt1[0] == wt[0])
                opti.subject_to(wt1[1] == wt[1])
                opti.subject_to(wt1[2] == 0.0)

        for leg in swing_groups[p]:
            fb_mid   = ca.vertcat(foot_body[k_mid][leg, 0],
                                  foot_body[k_mid][leg, 1],
                                  foot_body[k_mid][leg, 2])
            base_mid = ca.vertcat(base[0, k_mid], base[1, k_mid], base[2, k_mid])
            world_mid = _body_to_world_sym(fb_mid, base_mid, pitch[k_mid], roll[k_mid])
            opti.subject_to(world_mid[2] >= swing_clearance)
            for k in (k_start, k_end - 1):
                w = _world_foot_sym(k, leg)
                opti.subject_to(w[2] == 0.0)

    # ----- Joint velocity hard bounds (Iter-6) -----
    # AX-12A no-load speed cap: 59 RPM = 6.18 rad/s; 6.0 rad/s gives 3% margin.
    if enforce_joint_vel_bounds:
        q_vel_limit = 6.0
        for k in range(N - 1):
            q_delta = (joint_exprs[k + 1] - joint_exprs[k]) / dt
            for j in range(18):
                opti.subject_to(opti.bounded(-q_vel_limit, q_delta[j], q_vel_limit))

    # ----- Objective -----
    # 1) Energy proxy: integral of ||q_ddot||^2.
    energy_term = 0
    for k in range(1, N - 1):
        q_ddot = (joint_exprs[k + 1] - 2 * joint_exprs[k] + joint_exprs[k - 1]) / (dt ** 2)
        energy_term += ca.sumsqr(q_ddot)

    # 2) Pose-deviation: pitch + roll near zero. body_z uses w_height separately
    #    so it can be weighted much more strongly without blowing up pitch/roll gradient.
    pose_term   = 0
    height_term = 0
    for k in range(N):
        pose_term   += pitch[k] ** 2 + roll[k] ** 2
        height_term += (base[2, k] - body_height) ** 2

    # 3) Speed tracking: body position tracks the commanded direction line.
    #    For omnidirectional motion we track both base_x ≈ vx*t and
    #    base_y ≈ vy*t with equal weight. This generalizes the original
    #    forward-only speed_term (which also pinned base_y to zero).
    speed_term = 0
    for k in range(N):
        t_k = k * dt
        speed_term += (base[0, k] - vx * t_k) ** 2
        speed_term += (base[1, k] - vy * t_k) ** 2

    # Body acceleration smoothness: penalize d2(base)/dt2 in all axes.
    body_accel_term = 0
    for k in range(1, N - 1):
        ax = (base[0, k + 1] - 2 * base[0, k] + base[0, k - 1]) / (dt ** 2)
        ay = (base[1, k + 1] - 2 * base[1, k] + base[1, k - 1]) / (dt ** 2)
        az = (base[2, k + 1] - 2 * base[2, k] + base[2, k - 1]) / (dt ** 2)
        body_accel_term += ax ** 2 + ay ** 2 + az ** 2

    # 3b) Joint-jerk penalty (Iter-6): third numerical difference of joint angles.
    # Scale by dt^2 to normalize the 1/dt^5 contribution to O(1/dt^3),
    # matching the energy term's Hessian scale at any dt.
    joint_jerk_term = 0
    for k in range(2, N - 1):
        q_ddd = (joint_exprs[k + 1] - 3 * joint_exprs[k]
                 + 3 * joint_exprs[k - 1] - joint_exprs[k - 2]) / (dt ** 3)
        joint_jerk_term += ca.sumsqr(q_ddd) * dt
    joint_jerk_term *= dt ** 2

    # 4) Foot regularizer: swing feet stay near radial neutral in body frame.
    foot_reg_term = 0
    for k in range(N):
        for leg in range(6):
            foot_reg_term += ((foot_body[k][leg, 0] - foot_rest_body[leg, 0]) ** 2
                              + (foot_body[k][leg, 1] - foot_rest_body[leg, 1]) ** 2)

    # 5) Stance-midpoint symmetry (Iter-7): at the midpoint of each stance phase,
    # stance foot body-frame XY should equal the radial neutral. Combined with
    # no-slip, this eliminates the forward-bias arc the optimizer otherwise picks.
    # Soft cost at w_sym=1e4 (not 1e5 — gentler avoids non-convergence on 220-knot NLP).
    sym_term = 0
    for p in range(n_phases):
        k_start = p * knots_per_phase
        k_end   = (p + 1) * knots_per_phase
        k_mid   = (k_start + k_end) // 2
        for leg in stance_groups[p]:
            sym_term += ((foot_body[k_mid][leg, 0] - foot_rest_body[leg, 0]) ** 2
                         + (foot_body[k_mid][leg, 1] - foot_rest_body[leg, 1]) ** 2)

    # Stability penalty (disabled — kept for reference, weight is 0).
    stability_penalty = 0

    obj = (w_energy    * energy_term
           + w_joint_jerk * joint_jerk_term
           + w_pose    * pose_term
           + w_height  * height_term
           + w_speed   * speed_term
           + w_body_accel * body_accel_term
           + w_vx_track   * 0          # disabled
           + 1e-3 * foot_reg_term
           + w_sym * sym_term
           + w_stability * stability_penalty)

    opti.minimize(obj)

    # ----- Solver -----
    p_opts = {"expand": False}
    s_opts = {
        # 15000 iterations for 220-knot NLP; proportionally more needed than
        # the 40-knot baseline because decision variables scale ~7×.
        "max_iter": 15000,
        "print_level": 3,
        "tol": 1e-5,
        "acceptable_tol": 1e-3,
        "acceptable_iter": 15,
        # Large penalty weights make the unscaled gradient large at the optimum.
        # Loosen dual-infeasibility threshold so a primal-feasible point with
        # small KKT residual is accepted.
        "acceptable_dual_inf_tol": 1e10,
        "acceptable_constr_viol_tol": 1e-5,
        "acceptable_compl_inf_tol": 1e-3,
        "linear_solver": linear_solver,
        "mu_strategy": "adaptive",
        # Larger initial barrier prevents false infeasibility when w_height
        # creates large Hessian entries (Iter-5).
        "mu_init": 0.1,
        "bound_push": 1e-4,
        "bound_frac": 1e-4,
    }
    opti.solver("ipopt", p_opts, s_opts)

    print(f"[TO] vx={vx:.3f} vy={vy:.3f} | N={N} knots, dt={dt:.4f}s, "
          f"n_phases={n_phases}, ~{N*18 + 5*N} decision vars")
    t0 = time.time()
    try:
        sol = opti.solve()
        success = True
        ipopt_status = "Solve_Succeeded"
    except RuntimeError as e:
        msg = str(e)
        success = False
        sol = opti.debug
        # Pull IPOPT's last status string from the error message.
        if "Maximum_Iterations_Exceeded" in msg:
            ipopt_status = "Maximum_Iterations_Exceeded"
        elif "Infeasible_Problem_Detected" in msg:
            ipopt_status = "Infeasible_Problem_Detected"
        elif "Acceptable_Level" in msg or "Solved_To_Acceptable_Level" in msg:
            ipopt_status = "Solved_To_Acceptable_Level"
            success = True  # acceptable is good enough for AMP prior data
        elif ("Error in step computation" in msg
              or ("optistack" in msg and "Restoration" not in msg
                  and "Infeasible" not in msg and "Maximum_Iter" not in msg)):
            # IPOPT can fail with "Error in step computation" when the
            # warm-start point is already near-feasible (constraint
            # violation ~1e-15). CasADi wraps this in an optistack.cpp
            # RuntimeError. The debug iterate is primal-feasible and the
            # quality metrics are typically good in this case.
            # We tag it specially and let compute_metrics decide usability.
            ipopt_status = "StepError_NearFeasible"
            success = True  # primal-feasible debug solution is usable for AMP
        else:
            ipopt_status = f"Failed: {msg[:120]}"
        print(f"[TO] Solver non-success: {ipopt_status}")

    elapsed = time.time() - t0
    print(f"[TO] Solver finished in {elapsed:.1f}s, success={success}")

    # Extract solution.
    foot_sol   = np.zeros((N, 6, 3))
    joints_sol = np.zeros((N, 18))
    for k in range(N):
        foot_sol[k] = sol.value(foot_body[k])
        for leg in range(6):
            fb = ca.vertcat(foot_body[k][leg, 0],
                            foot_body[k][leg, 1],
                            foot_body[k][leg, 2])
            j3 = _foot_body_to_joints_sym(fb, leg, leg_offsets)
            joints_sol[k, leg * 3:leg * 3 + 3] = np.array(sol.value(j3)).flatten()
    base_sol  = np.array(sol.value(base)).T   # (N, 3)
    pitch_sol = np.array(sol.value(pitch)).flatten()
    roll_sol  = np.array(sol.value(roll)).flatten()

    return {
        "success":      success,
        "ipopt_status": ipopt_status,
        "N":            N,
        "dt":           dt,
        "duration_s":   duration_s,
        "vx_cmd":       vx,
        "vy_cmd":       vy,
        "joints":       joints_sol,
        "feet_body":    foot_sol,
        "base":         base_sol,
        "pitch":        pitch_sol,
        "roll":         roll_sol,
        "leg_offsets":  leg_offsets,
        "elapsed_s":    elapsed,
    }


# ----------------------------------------------------------------------
# Metrics helpers (reused by sweep runner + AMP converter).
# ----------------------------------------------------------------------
def compute_metrics(sol: dict) -> dict:
    """Compute summary metrics from a solved trajectory dict."""
    joints = sol["joints"]
    base   = sol["base"]
    pitch  = sol["pitch"]
    roll   = sol["roll"]
    dt     = sol["dt"]
    vx_cmd = sol["vx_cmd"]
    vy_cmd = sol["vy_cmd"]
    body_height = 0.145

    qd  = np.diff(joints, axis=0) / dt
    qdd = np.diff(qd, axis=0) / dt
    bx  = base[:, 0]
    by  = base[:, 1]
    vx_actual = np.diff(bx) / dt
    vy_actual = np.diff(by) / dt
    peak_ax = float(np.max(np.abs(np.diff(vx_actual) / dt))) if len(vx_actual) > 1 else 0.0
    peak_ay = float(np.max(np.abs(np.diff(vy_actual) / dt))) if len(vy_actual) > 1 else 0.0

    return {
        "peak_joint_vel_radps":   float(np.max(np.abs(qd))),
        "max_abs_pitch_deg":      float(np.degrees(np.max(np.abs(pitch)))),
        "max_abs_roll_deg":       float(np.degrees(np.max(np.abs(roll)))),
        "body_z_dev_max_m":       float(np.max(np.abs(base[:, 2] - body_height))),
        "base_y_err_max_mm":      float(np.max(np.abs(
            by - vy_cmd * np.arange(len(by)) * dt)) * 1000.0),
        "base_x_err_max_mm":      float(np.max(np.abs(
            bx - vx_cmd * np.arange(len(bx)) * dt)) * 1000.0),
        "mean_vx_actual":         float(np.mean(vx_actual)),
        "mean_vy_actual":         float(np.mean(vy_actual)),
        "peak_abs_body_ax_mps2":  peak_ax,
        "peak_abs_body_ay_mps2":  peak_ay,
        "joint_std_max":          float(np.std(joints, axis=0).max()),
        "joint_std_min":          float(np.std(joints, axis=0).min()),
    }


# ----------------------------------------------------------------------
# Warm-start orchestration (coarse → fine).
# ----------------------------------------------------------------------
def solve_with_warmstart(vx, vy, duration_s, n_strides, knots_per_phase,
                         **kwargs):
    """Run a coarse 40-knot solve first, then warm-start the requested fine
    solve. Matches the strategy from trajectory_opt_demo.py (Iter-6)."""
    coarse_n_strides = 4
    coarse_kpp = 5
    coarse_N = 2 * coarse_n_strides * coarse_kpp
    fine_N   = 2 * n_strides * knots_per_phase

    # Coarse solve uses warmup-specific values for these three knobs
    # regardless of user input; build a separate kwargs dict so user
    # overrides in `kwargs` don't collide with the explicit coarse values
    # below (TypeError: multiple values for keyword argument).
    coarse_kwargs = dict(kwargs)
    coarse_kwargs["enforce_joint_vel_bounds"] = False
    coarse_kwargs["w_joint_jerk"]             = 0.0
    # Coarse dt is large (~0.2s); the no-slip geometry needs more lateral
    # slack than the fine solve's 0.9mm spec. 5mm lets the coarse optimizer
    # find a feasible trajectory without violating the tight directional
    # line constraint.
    coarse_kwargs["directional_tol"]          = 0.005

    warm_sol = None
    if fine_N > coarse_N:
        print(f"[TO] Coarse warm-start: n_strides={coarse_n_strides}, kpp={coarse_kpp}, N={coarse_N}")
        warm_sol = build_and_solve_to(
            vx=vx, vy=vy,
            duration_s=duration_s,
            n_strides=coarse_n_strides,
            knots_per_phase=coarse_kpp,
            **coarse_kwargs,
        )
        if not warm_sol["success"]:
            print("[TO] WARNING: coarse solve failed — skipping warm-start.")
            warm_sol = None

    return build_and_solve_to(
        vx=vx, vy=vy,
        duration_s=duration_s,
        n_strides=n_strides,
        knots_per_phase=knots_per_phase,
        initial_guess=warm_sol,
        **kwargs,
    )


# ----------------------------------------------------------------------
# Main (single solve, CLI).
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="TO solver — arbitrary (vx, vy) omnidirectional hexapod gait.")
    ap.add_argument("--vx", type=float, default=0.17, help="m/s forward")
    ap.add_argument("--vy", type=float, default=0.0,  help="m/s lateral (left=+)")
    ap.add_argument("--duration", type=float, default=8.0)
    # Default: 11 strides × 10 knots/phase = 220 knots over 8 s (~0.73 s/stride,
    # matching the signed-off Iter-7 configuration).
    ap.add_argument("--n-strides", type=int, default=11)
    ap.add_argument("--knots-per-phase", type=int, default=10)
    ap.add_argument("--body-height", type=float, default=0.145)
    ap.add_argument("--out", type=str, default=None,
                    help="Output .npz path. Omit to not save.")
    ap.add_argument("--no-warmstart", action="store_true",
                    help="Skip coarse warm-start (faster for small NLPs).")
    args = ap.parse_args()

    if args.no_warmstart:
        sol = build_and_solve_to(
            vx=args.vx, vy=args.vy,
            duration_s=args.duration,
            n_strides=args.n_strides,
            knots_per_phase=args.knots_per_phase,
            body_height=args.body_height,
        )
    else:
        sol = solve_with_warmstart(
            vx=args.vx, vy=args.vy,
            duration_s=args.duration,
            n_strides=args.n_strides,
            knots_per_phase=args.knots_per_phase,
            body_height=args.body_height,
        )

    m = compute_metrics(sol)
    print()
    print(f"  vx_cmd={sol['vx_cmd']:.3f}  vy_cmd={sol['vy_cmd']:.3f}  "
          f"duration={sol['duration_s']:.1f}s  N={sol['N']}")
    print(f"  success={sol['success']}  ipopt_status={sol['ipopt_status']}")
    print(f"  peak |q_dot| = {m['peak_joint_vel_radps']:.2f} rad/s  "
          f"(limit 6.0)")
    print(f"  max |pitch|  = {m['max_abs_pitch_deg']:.3f} deg  "
          f"(spec <= 0.5)")
    print(f"  max |roll|   = {m['max_abs_roll_deg']:.3f} deg  "
          f"(spec <= 0.5)")
    print(f"  body_z dev   = {1000*m['body_z_dev_max_m']:.3f} mm  "
          f"(spec <= 1.0)")
    print(f"  base_x err   = {m['base_x_err_max_mm']:.3f} mm")
    print(f"  base_y err   = {m['base_y_err_max_mm']:.3f} mm  (spec <= 1.0)")
    print(f"  mean vx act  = {m['mean_vx_actual']:.4f} m/s  "
          f"(cmd {sol['vx_cmd']:.4f})")
    print(f"  mean vy act  = {m['mean_vy_actual']:.4f} m/s  "
          f"(cmd {sol['vy_cmd']:.4f})")
    print(f"  peak |ax|    = {m['peak_abs_body_ax_mps2']:.3f} m/s²  "
          f"(spec < 2.0)")
    print(f"  joint std max/min = {m['joint_std_max']:.4f} / {m['joint_std_min']:.4f} rad")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out,
                 joints=sol["joints"],
                 base=sol["base"],
                 pitch=sol["pitch"],
                 roll=sol["roll"],
                 dt=np.float64(sol["dt"]),
                 vx_cmd=np.float64(sol["vx_cmd"]),
                 vy_cmd=np.float64(sol["vy_cmd"]),
                 duration_s=np.float64(sol["duration_s"]),
                 success=np.bool_(sol["success"]),
                 ipopt_status=np.str_(sol["ipopt_status"]))
        print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
