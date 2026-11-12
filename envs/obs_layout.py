"""envs/obs_layout.py — single source of truth for the policy's observation.

Both the JAX env (envs/hexapod_env_jax.py) and the gym env (envs/hexapod_env.py)
compose their observations through this module. When the obs schema changes,
edit ONLY this file — both envs (and downstream tools like watch_demo_jax.py)
automatically pick up the change.

Both envs share the same FORMULAS for each obs slot; they only differ in
the underlying array library (jnp vs np). compose_obs() takes the components
pre-computed by the caller plus a concatenate function — this avoids
duplicating the layout knowledge across env files.

Two layouts ("variants"), selected by `RobotProfile.obs_variant`:

  "full"     — v27 PhantomX obs, 78 dims. joint_pos/vel are the TRUE joint
               state (qpos/qvel), plus IMU and privileged body linvel.
  "hardware" — what an open-loop-servo robot can actually measure, 75 dims.
               joint_pos = COMMANDED joint angle (ctrl), joint_vel = its
               finite difference; IMU quat/gyro (constant identity/zero when
               the robot has no IMU); no body_linvel. Same slot order
               otherwise so firmware can share one packer.

Obs layout (slot → dim) — order matters; both envs must emit this exact order:
"""
from collections import OrderedDict

# Order matters — both envs concat in this exact sequence.
OBS_SLOT_DIMS: "OrderedDict[str, int]" = OrderedDict([
    ("joint_pos",       18),     # qpos[7:25]           ("hardware": ctrl)
    ("joint_vel",       18),     # qvel[6:24]           ("hardware": Δctrl/dt)
    ("imu_quat",         4),     # qpos[3:7]
    ("imu_gyro",         3),     # qvel[3:6] (world-frame angvel)
    ("imu_accel",        3),     # zeros at training time (placeholder)
    ("scaffold_hint",   18),     # flatten of (6, 3) intended foot positions
    ("phase_sc",         2),     # [sin(2π·phase), cos(2π·phase)]
    ("cmd",              9),     # 9-D cmd vector
    ("body_linvel",      3),     # body-frame linear velocity ("full" only)
    # v26 (2026-05-13): motor feedback slots (joint_torque, joint_pos_error)
    # removed. Paper Liu et al. 2511.03167 achieves rough-terrain locomotion
    # with only proprioception + cmd + prev action + short-term memory
    # latent; motor torque/position-error are not in their actor obs.
    # v24's addition correlated with the walking-quality regression and
    # is reverted here as part of the paper-pure realignment.
])

OBS_VARIANTS = {
    "full":     tuple(OBS_SLOT_DIMS.keys()),
    "hardware": tuple(k for k in OBS_SLOT_DIMS if k != "body_linvel"),
}


def obs_slots(variant: str = "full") -> tuple:
    try:
        return OBS_VARIANTS[variant]
    except KeyError:
        raise ValueError(f"unknown obs variant {variant!r}; "
                         f"choose from {sorted(OBS_VARIANTS)}") from None


def obs_dim(variant: str = "full") -> int:
    return sum(OBS_SLOT_DIMS[k] for k in obs_slots(variant))


OBS_DIM = obs_dim("full")   # 78 — legacy alias (PhantomX / v27)


def compose_obs(*, joint_pos, joint_vel, imu_quat, imu_gyro, imu_accel,
                scaffold_hint, phase_sc, cmd, body_linvel,
                concat_fn, variant: str = "full"):
    """Concatenate per-slot signals into the policy obs vector.

    Args:
        Per-slot arrays (each must be shape (slot_dim,) — caller handles
        slicing/reshaping from raw env data). For the "hardware" variant the
        caller passes the COMMANDED joint angle / its finite difference as
        joint_pos / joint_vel; body_linvel is ignored.
        concat_fn: np.concatenate or jnp.concatenate — selected by caller
                   based on which array library it's using.
        variant: key of OBS_VARIANTS.

    Returns: concatenated obs of shape (obs_dim(variant),).
    """
    parts = {
        "joint_pos": joint_pos, "joint_vel": joint_vel,
        "imu_quat": imu_quat, "imu_gyro": imu_gyro, "imu_accel": imu_accel,
        "scaffold_hint": scaffold_hint,
        "phase_sc": phase_sc,
        "cmd": cmd,
        "body_linvel": body_linvel,
    }
    return concat_fn([parts[k] for k in obs_slots(variant)])
