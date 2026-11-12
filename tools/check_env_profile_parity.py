"""tools/check_env_profile_parity.py — verify the profile-driven env refactor.

1. PhantomX bit-equivalence: the live `envs/hexapod_env_jax.py` must produce
   the same reset obs, step obs and rewards as the pre-refactor snapshot
   (`snapshots/hexapod_env_jax_pre_profile_env.py`) for identical RNG keys,
   and `envs.cmd_bins.cmd_to_bin` must agree with the snapshot binning.
2. Mini hexapod smoke: params build, reset obs is 75-dim ("hardware"
   variant), the scaffold-driven bot survives a few hundred steps, and the
   Brax adapter reports the right observation size.

Needs MJX (run in the WSL venv):
    PYTHONPATH=. ~/.venv-mjx/bin/python tools/check_env_profile_parity.py
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from envs import cmd_bins as new_bins  # noqa: E402
from envs import hexapod_env_jax as new_env  # noqa: E402

PHANTOMX = str(_ROOT / "models" / "phantomx_simple_mjx.xml")
MINI     = str(_ROOT / "models" / "mini_hexapod_mjx.xml")


def _load_snapshot(name: str):
    path = _ROOT / "snapshots" / f"{name}_pre_profile_env.py"
    spec = importlib.util.spec_from_file_location(f"snap_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rollout(mod, params, key, n_steps: int, gait_scale: float):
    state = mod.reset(params, key, gait_scale=gait_scale)
    step = jax.jit(mod.step_foot)
    zero = jnp.zeros(18, dtype=jnp.float32)
    obs, rew = [np.asarray(state.obs)], []
    for _ in range(n_steps):
        state = step(params, state, zero)
        obs.append(np.asarray(state.obs))
        rew.append(float(state.reward))
    return np.stack(obs), np.asarray(rew), state


def check_phantomx_parity() -> None:
    print("== PhantomX parity vs snapshot ==")
    old_env = _load_snapshot("hexapod_env_jax")
    old_bins = _load_snapshot("cmd_bins")

    p_new = new_env.make_env_params(PHANTOMX)
    p_old = old_env.make_env_params(PHANTOMX)
    for name in ("spawn_qpos", "cmd_sample_ranges", "err_inv_scales", "foot_local_offset"):
        a, b = np.asarray(getattr(p_new, name)), np.asarray(getattr(p_old, name))
        assert np.array_equal(a, b), f"{name} differs:\n{a}\n{b}"
    for name in ("min_z", "ref_body_z", "foot_contact_z", "joint_vel_limit",
                 "joint_torque_limit", "foot_force_limit", "foot_residual_scale_max",
                 "init_body_z_noise_m"):
        assert getattr(p_new, name) == getattr(p_old, name), name
    assert p_new.obs_cfg.variant == "full" and p_new.obs_cfg.imu_available
    # EnvParams must survive a trip through jit as an ARGUMENT (tools do this).
    jax.jit(lambda prm, x: x + prm.ref_body_z)(p_new, jnp.float32(0.0))
    print("  EnvParams scalars/arrays identical")

    key = jax.random.PRNGKey(7)
    t0 = time.time()
    obs_n, rew_n, _ = _rollout(new_env, p_new, key, 60, gait_scale=1.0)
    obs_n2, rew_n2, _ = _rollout(new_env, p_new, key, 60, gait_scale=1.0)
    obs_o, rew_o, _ = _rollout(old_env, p_old, key, 60, gait_scale=1.0)
    # The GPU MJX step is not bitwise deterministic run-to-run (~1e-5 after a
    # few dozen steps), so compare old-vs-new against that noise floor. The
    # reset obs and the first steps must still match exactly.
    noise = max(float(np.abs(obs_n - obs_n2).max()), float(np.abs(rew_n - rew_n2).max()))
    d_obs = float(np.abs(obs_n - obs_o).max())
    d_rew = float(np.abs(rew_n - rew_o).max())
    d_early = float(np.abs(obs_n[:6] - obs_o[:6]).max())
    print(f"  60-step scaffold rollout: max|Δobs|={d_obs:.2e}  max|Δreward|={d_rew:.2e}  "
          f"(GPU run-to-run noise {noise:.2e}; first 5 steps Δ={d_early:.1e})  ({time.time()-t0:.0f}s)")
    assert d_early == 0.0, "PhantomX env diverges from the snapshot immediately"
    assert max(d_obs, d_rew) <= max(10.0 * noise, 1e-6), "PhantomX env differs beyond GPU noise"

    rng = np.random.default_rng(0)
    cmds = rng.uniform([-0.4, -0.4, -2.0, -0.3, -0.3, -0.05, -0.05, -0.03, -0.03],
                       [0.4, 0.4, 2.0, 0.3, 0.3, 0.04, 0.10, 0.03, 0.03], size=(20000, 9)).astype(np.float32)
    b_new = np.asarray(new_bins.cmd_to_bin(jnp.asarray(cmds)))
    b_old = np.asarray(old_bins.cmd_to_bin(jnp.asarray(cmds)))
    assert np.array_equal(b_new, b_old), f"cmd_to_bin differs on {np.sum(b_new != b_old)} cmds"
    print(f"  cmd_to_bin identical on {len(cmds)} random cmds")


def check_mini_smoke() -> None:
    print("== Mini hexapod smoke ==")
    from envs.obs_layout import obs_dim
    p = new_env.make_env_params(MINI)
    print(f"  obs_variant={p.obs_cfg.variant} imu={p.obs_cfg.imu_available} spawn_z={float(p.spawn_qpos[2]):.4f} "
          f"ref_z={p.ref_body_z:.4f} min_z={p.min_z:.3f} foot_offset={np.asarray(p.foot_local_offset)}")
    print(f"  dh range={np.asarray(p.cmd_sample_ranges[5])}, dw box={np.asarray(p.cmd_sample_ranges[6])}, "
          f"dw_envelope={np.asarray(p.dw_envelope)}")
    key = jax.random.PRNGKey(3)
    state = new_env.reset(p, key, gait_scale=1.0)
    assert state.obs.shape == (obs_dim("hardware"),), state.obs.shape
    # hardware obs: joint_pos slot == ctrl, imu quat slot is identity, no linvel slot
    o = np.asarray(state.obs)
    assert np.allclose(o[:18], np.asarray(state.mjx_data.ctrl)), "joint_pos should be the commanded angle"
    assert np.allclose(o[36:40], [1, 0, 0, 0]) and np.allclose(o[40:43], 0.0), "IMU slots should be constant"
    print(f"  reset obs dim {o.shape[0]} OK (hardware layout, IMU constant)")

    t0 = time.time()
    n = 400
    obs, rew, final = _rollout(new_env, p, key, n, gait_scale=1.0)
    body_z = float(final.mjx_data.qpos[2])
    xy = np.asarray(final.mjx_data.qpos[:2])
    print(f"  {n} scaffold steps: done={bool(final.done)} body_z={body_z:.4f} xy={xy.round(3)} "
          f"mean reward={rew.mean():.3f} cmd={np.asarray(final.cmd).round(3)} ({time.time()-t0:.0f}s)")
    assert not bool(final.done), "mini fell / terminated under scaffold control"
    assert np.isfinite(obs).all() and np.isfinite(rew).all()

    bins = new_bins.bin_spec_for(MINI)
    print(f"  mini BinSpec: presets={np.asarray(bins.height_presets)} motion_half={np.asarray(bins.motion_half)}")
    rng = np.random.default_rng(1)
    cmds = rng.uniform([-0.08, -0.08, -0.6, 0, 0, -0.028, -0.04, 0, 0],
                       [0.08, 0.08, 0.6, 0, 0, 0.032, 0.055, 0, 0], size=(20000, 9)).astype(np.float32)
    b = np.asarray(new_bins.cmd_to_bin(jnp.asarray(cmds), bins))
    print(f"  mini bins populated: {len(np.unique(b))}/{new_bins.N_BINS}")

    from envs.hexapod_brax_env import HexapodBraxEnv
    benv = HexapodBraxEnv(MINI, gait_scale=0.0, action_space="foot")
    assert benv.observation_size == obs_dim("hardware"), benv.observation_size
    print(f"  Brax adapter observation_size={benv.observation_size}")


if __name__ == "__main__":
    check_phantomx_parity()
    check_mini_smoke()
    print("ALL CHECKS PASSED")
