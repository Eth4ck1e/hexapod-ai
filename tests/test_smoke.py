"""Quick smoke test for CI: verify all critical imports + MuJoCo model loading.

Run with:  python -m pytest tests/test_smoke.py -v
"""

import importlib
import os
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH before first-party imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pytest
from envs.cmd_bins import cmd_to_bin
from envs.obs_layout import OBS_DIM
from envs.stance_envelope import safe_dw_range
from gait import NEUTRAL_POSE, Controller


def test_import_gait():
    """gait package and Controller class."""
    assert hasattr(Controller, "predict")


def test_import_envs_obs_layout():
    """envs.obs_layout — single source of truth for obs schema."""
    assert OBS_DIM > 10


def test_import_envs_stance_envelope():
    """envs.stance_envelope — stance height/width envelope."""
    lo, hi = safe_dw_range(-0.025)
    assert lo < hi


def test_import_envs_cmd_bins():
    """envs.cmd_bins — 150-bin partition for multi-head disc."""
    idx = cmd_to_bin(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert 0 <= idx < 150


def test_import_amp_discriminator():
    """amp.discriminator — multi-head discriminator."""
    from amp.discriminator import MultiHeadDiscriminator

    assert hasattr(MultiHeadDiscriminator, "__call__")


mjx_available = pytest.mark.skipif(
    not importlib.util.find_spec("mujoco.mjx"),
    reason="mujoco.mjx not available on this platform",
)


@mjx_available
def test_import_amp_prior_data():
    """amp.prior_data — prior collection."""
    from amp.prior_data import collect_demos

    assert callable(collect_demos)


@pytest.fixture(scope="module")
def mujoco_available():
    """Verify MuJoCo can import and we have a usable GL backend on CI."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    return mujoco


def test_mujoco_import(mujoco_available):
    """MuJoCo library imports successfully."""
    assert hasattr(mujoco_available, "MjModel")


def test_mujoco_model_load(mujoco_available):
    """Load the main MJCF model and verify geometry."""
    model_path = str(_PROJECT_ROOT / "models" / "phantomx.xml")
    assert os.path.exists(model_path), f"MJCF not found: {model_path}"
    model = mujoco_available.MjModel.from_xml_path(model_path)
    assert model.nq > 0
    assert model.nv > 0
    assert model.nu == 18, f"expected 18 actuators, got {model.nu}"


def test_mujoco_step(mujoco_available):
    """Step the physics forward one tick — validates solver + geometry."""
    model_path = str(_PROJECT_ROOT / "models" / "phantomx.xml")
    model = mujoco_available.MjModel.from_xml_path(model_path)
    data = mujoco_available.MjData(model)
    mujoco_available.mj_step(model, data)
    assert data.qpos[2] >= 0.0, f"body z went negative: {data.qpos[2]}"


def test_controller_init():
    """Controller initialises with the production MJCF."""
    model_path = str(_PROJECT_ROOT / "models" / "phantomx.xml")
    ctrl = Controller(model_path)
    assert ctrl.MAX_SPEED > 0
    assert ctrl.MAX_YAW_RATE > 0


def test_controller_predict():
    """Controller.predict produces correct joint shape."""
    model_path = str(_PROJECT_ROOT / "models" / "phantomx.xml")
    ctrl = Controller(model_path)
    cmd = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    joints = ctrl.predict(cmd, 0.0)
    assert joints.shape == (18,), f"expected (18,), got {joints.shape}"
    neutral = ctrl.gait_neutral_pose
    deviation = np.max(np.abs(joints - neutral))
    assert deviation < 1.5, f"unrealistic joint deviation: {deviation:.3f} rad"


def test_controller_neutral_pose():
    """gait.NEUTRAL_POSE constant is accessible."""
    assert len(NEUTRAL_POSE) == 18


# ---------------------------------------------------------------------------
# Robot profiles (gait/robot_profile.py)
# ---------------------------------------------------------------------------
def test_profile_registry():
    """PhantomX is registered, is the default, and resolves by name."""
    from gait import DEFAULT_PROFILE, PHANTOMX, get_profile

    assert get_profile("phantomx") is PHANTOMX
    assert get_profile(None) is DEFAULT_PROFILE
    with pytest.raises(KeyError):
        get_profile("no_such_robot")


def test_profile_inferred_from_model_path():
    """Controller(model_path) picks the profile from the MJCF filename."""
    from gait import profile_for_model

    assert profile_for_model("models/phantomx.xml").name == "phantomx"
    assert profile_for_model("/abs/dir/phantomx_simple_mjx.xml").name == "phantomx"
    with pytest.warns(UserWarning):
        assert profile_for_model("models/unknown_bot.xml").name == "phantomx"


def test_controller_default_and_explicit_profile_agree():
    """Controller() (no args), Controller(path) and Controller(profile=...)
    must produce identical gait output for the same MJCF."""
    model_path = str(_PROJECT_ROOT / "models" / "phantomx_simple_mjx.xml")
    a = Controller()
    b = Controller(model_path)
    c = Controller(model_path, profile="phantomx")
    assert a.profile.name == b.profile.name == c.profile.name == "phantomx"
    cmd = np.array([0.1, 0.05, 0.3, 0.05, -0.05, -0.01, 0.01, 0.0, 0.0])
    ja, jb, jc = (x.predict(cmd, 1.234) for x in (a, b, c))
    np.testing.assert_array_equal(ja, jb)
    np.testing.assert_array_equal(jb, jc)
    # Bare module constants still mirror the PhantomX profile.
    from gait import COXA_LENGTH, FEMUR_LENGTH, TIBIA_LENGTH

    lengths = (a.profile.coxa_length, a.profile.femur_length, a.profile.tibia_length)
    assert lengths == (COXA_LENGTH, FEMUR_LENGTH, TIBIA_LENGTH)


# ---------------------------------------------------------------------------
# Mini hexapod (second robot) — generated MJCFs + profile
# ---------------------------------------------------------------------------
MINI_MODELS = ("mini_hexapod_mjx.xml", "mini_hexapod_simple.xml")


@pytest.mark.parametrize("fname", MINI_MODELS)
def test_mini_hexapod_model_loads(mujoco_available, fname):
    model = mujoco_available.MjModel.from_xml_path(str(_PROJECT_ROOT / "models" / fname))
    assert model.nu == 18
    assert model.nq == 25


@pytest.mark.parametrize("fname", MINI_MODELS)
def test_mini_hexapod_profile_and_ik_roundtrip(mujoco_available, fname):
    """Controller infers the mini profile from the filename and its IK
    round-trips through MJCF FK to sub-millimetre accuracy."""
    model_path = str(_PROJECT_ROOT / "models" / fname)
    ctrl = Controller(model_path)
    assert ctrl.profile.name == "mini_hexapod"
    # All feet rest on one plane, tibia_length below the body.
    np.testing.assert_allclose(ctrl.LEG_ORIGIN_BODY[:, 2], -ctrl.profile.tibia_length, atol=1e-9)
    model = mujoco_available.MjModel.from_xml_path(model_path)
    data = mujoco_available.MjData(model)
    tib = [mujoco_available.mj_name2id(model, mujoco_available.mjtObj.mjOBJ_BODY, f"tibia_{n}")
           for n in ("RR", "RM", "RF", "LR", "LM", "LF")]
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        feet = ctrl.LEG_ORIGIN_BODY + rng.uniform([-0.02, -0.02, -0.015], [0.02, 0.02, 0.03], size=(6, 3))
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0
        data.qpos[7:25] = ctrl.body_to_joints(feet)
        mujoco_available.mj_forward(model, data)
        for i in range(6):
            fk = data.xpos[tib[i]] + data.xmat[tib[i]].reshape(3, 3) @ ctrl.FOOT_TIP_LOCAL[i]
            worst = max(worst, float(np.linalg.norm(fk - feet[i])))
    assert worst < 1e-4, f"IK round-trip error {worst * 1e3:.3f} mm"


def test_mini_hexapod_generator_is_in_sync():
    """Regenerating the MJCFs must reproduce the committed files exactly."""
    sys.path.insert(0, str(_PROJECT_ROOT / "tools"))
    import gen_mini_hexapod_mjcf as gen

    for mjx, fname in ((True, MINI_MODELS[0]), (False, MINI_MODELS[1])):
        on_disk = (_PROJECT_ROOT / "models" / fname).read_text(encoding="utf-8")
        assert gen.build(mjx) == on_disk, f"{fname} is stale — run tools/gen_mini_hexapod_mjcf.py"


# ---------------------------------------------------------------------------
# Profile-driven env plumbing (2026-11-12): obs variants, stance envelope,
# cmd bins and the gym env all follow the robot profile.
# ---------------------------------------------------------------------------
def test_obs_variants():
    from envs.obs_layout import OBS_DIM, compose_obs, obs_dim
    assert obs_dim("full") == OBS_DIM == 78
    assert obs_dim("hardware") == 75
    parts = dict(joint_pos=np.zeros(18), joint_vel=np.zeros(18), imu_quat=np.zeros(4),
                 imu_gyro=np.zeros(3), imu_accel=np.zeros(3), scaffold_hint=np.zeros(18),
                 phase_sc=np.zeros(2), cmd=np.zeros(9), body_linvel=np.ones(3))
    assert compose_obs(**parts, concat_fn=np.concatenate, variant="full").shape == (78,)
    hw = compose_obs(**parts, concat_fn=np.concatenate, variant="hardware")
    assert hw.shape == (75,) and not np.any(hw == 1.0), "hardware obs must drop body_linvel"


def test_stance_envelope_per_profile():
    from envs.stance_envelope import PHANTOMX_ENVELOPE, envelope_for, safe_dw_range
    from gait import MINI_HEXAPOD, PHANTOMX
    assert envelope_for(PHANTOMX) == PHANTOMX_ENVELOPE
    assert safe_dw_range(-0.025) == PHANTOMX_ENVELOPE.safe_dw_range(-0.025)
    mini = envelope_for(MINI_HEXAPOD)
    assert envelope_for("models/mini_hexapod_mjx.xml") == mini
    assert mini.dh_min < 0.0 < mini.dh_max
    assert len(mini.height_presets) == 5
    for dh in mini.height_presets:
        lo, hi = mini.safe_dw_range(dh)
        assert lo < 0.0 < hi, (dh, lo, hi)
        assert mini.dw_joint_min <= lo and hi <= mini.dw_joint_max + 1e-9


def test_cmd_bins_per_profile():
    import jax.numpy as jnp
    from envs.cmd_bins import N_BINS, PHANTOMX_BIN_SPEC, bin_spec_for, cmd_to_bin
    assert bin_spec_for("models/phantomx_simple_mjx.xml") is PHANTOMX_BIN_SPEC
    spec = bin_spec_for("models/mini_hexapod_mjx.xml")
    rng = np.random.default_rng(0)
    cmds = rng.uniform([-0.08, -0.08, -0.6, 0, 0, -0.028, -0.04, 0, 0],
                       [0.08, 0.08, 0.6, 0, 0, 0.032, 0.055, 0, 0], size=(5000, 9)).astype(np.float32)
    bins = np.asarray(cmd_to_bin(jnp.asarray(cmds), spec))
    assert bins.min() >= 0 and bins.max() < N_BINS
    assert len(np.unique(bins)) > 0.8 * N_BINS, "mini cmd range should populate most bins"
    # Default spec is PhantomX — identical to passing it explicitly.
    assert np.array_equal(np.asarray(cmd_to_bin(jnp.asarray(cmds))),
                          np.asarray(cmd_to_bin(jnp.asarray(cmds), PHANTOMX_BIN_SPEC)))


@pytest.mark.parametrize("fname,variant,dim", [
    ("mini_hexapod_mjx.xml", "hardware", 75),
    ("phantomx_simple_mjx.xml", "full", 78),
])
def test_gym_env_follows_profile(mujoco_available, fname, variant, dim):
    """Gym env (watch/eval/record path) reads spawn/termination/obs layout from the profile."""
    pytest.importorskip("gymnasium")   # not installed in the WSL training venv
    from envs.hexapod_env import HexapodEnv
    env = HexapodEnv(model_path=str(_PROJECT_ROOT / "models" / fname), stage=3)
    assert env.profile.obs_variant == variant
    assert env.observation_space.shape == (dim,)
    assert env.profile.min_body_z == env.MIN_Z and env.profile.ref_body_z == env.REF_BODY_Z
    env.gait_scale = 1.0
    obs, _ = env.reset(seed=0)
    assert obs.shape == (dim,)
    for _ in range(50):
        obs, _, term, trunc, _ = env.step(np.zeros(18, dtype=np.float32))
        assert not term and not trunc
    if variant == "hardware":
        np.testing.assert_allclose(obs[:18], env._data.ctrl, atol=1e-6)
        np.testing.assert_allclose(obs[36:40], [1, 0, 0, 0])


@mjx_available
def test_dynamics_domain_randomization(mujoco_available):
    """dr_scale=0 keeps the nominal model object (bit-exact legacy graph);
    dr_scale=1 samples per-episode scales inside DR_RANGES_FULL and the
    per-env model carries them consistently (mass, subtree mass, kp, stall)."""
    import jax
    import jax.numpy as jnp
    from envs import hexapod_env_jax as hx

    model = str(_PROJECT_ROOT / "models" / "mini_hexapod_mjx.xml")
    p0 = hx.make_env_params(model, cmd_mask_name="paper_stance")
    assert not p0.dyn_cfg.dr_enabled
    assert np.array_equal(np.asarray(p0.dr_ranges), np.ones((hx.N_DR, 2), np.float32))
    s0 = hx.reset(p0, jax.random.PRNGKey(1))
    assert np.array_equal(np.asarray(s0.dr_scales), np.ones(hx.N_DR, np.float32))
    assert hx._randomized_model(p0, s0.dr_scales) is p0.mjx_model

    p1 = hx.make_env_params(model, cmd_mask_name="paper_stance", dr_scale=1.0)
    assert p1.dyn_cfg.dr_enabled
    # cmd / spawn-noise streams are unchanged by DR (key folded off separately).
    s1 = hx.reset(p1, jax.random.PRNGKey(1))
    assert np.array_equal(np.asarray(s1.cmd), np.asarray(s0.cmd))
    sc = np.asarray(s1.dr_scales)
    assert np.all(sc >= hx.DR_RANGES_FULL[:, 0]) and np.all(sc <= hx.DR_RANGES_FULL[:, 1])
    m = hx._randomized_model(p1, s1.dr_scales)
    nom = p1.mjx_model
    base = p1.base_body_id
    s_fric, s_motor, s_base, s_leg, s_damp = sc
    np.testing.assert_allclose(m.body_mass[base], nom.body_mass[base] * s_base, rtol=1e-6)
    np.testing.assert_allclose(m.body_mass[base + 1], nom.body_mass[base + 1] * s_leg, rtol=1e-6)
    # subtree mass of the base must equal the sum of the scaled body masses
    np.testing.assert_allclose(m.body_subtreemass[base], np.sum(np.asarray(m.body_mass)), rtol=1e-5)
    np.testing.assert_allclose(m.actuator_gainprm[:, 0], nom.actuator_gainprm[:, 0] * s_motor, rtol=1e-6)
    np.testing.assert_allclose(m.actuator_biasprm[:, 1], -m.actuator_gainprm[:, 0], rtol=1e-6)
    np.testing.assert_allclose(m.actuator_forcerange, nom.actuator_forcerange * s_motor, rtol=1e-6)
    np.testing.assert_allclose(m.geom_friction[:, 0], nom.geom_friction[:, 0] * s_fric, rtol=1e-6)
    np.testing.assert_allclose(m.dof_damping, nom.dof_damping * s_damp, rtol=1e-6)
    # Randomized env steps stay finite under vmap (two different episodes).
    step = jax.jit(jax.vmap(lambda s, a: hx.step_foot(p1, s, a)))
    st = jax.vmap(lambda k: hx.reset(p1, k, gait_scale=1.0))(jax.random.split(jax.random.PRNGKey(2), 2))
    assert not np.array_equal(np.asarray(st.dr_scales[0]), np.asarray(st.dr_scales[1]))
    for _ in range(20):
        st = step(st, jnp.zeros((2, 18), jnp.float32))
    assert np.isfinite(np.asarray(st.obs)).all() and not np.asarray(st.done).any()


@mjx_available
def test_servo_command_hold(mujoco_available):
    """Mini latches a new servo target every ctrl_hold_steps (50 Hz PWM frame);
    PhantomX keeps the legacy every-step update. Hardware obs report the
    latched command."""
    import jax
    import jax.numpy as jnp
    from envs import hexapod_env_jax as hx
    from gait.robot_profile import MINI_HEXAPOD, PHANTOMX

    assert PHANTOMX.ctrl_hold_steps == 1 and MINI_HEXAPOD.ctrl_hold_steps == 4
    p = hx.make_env_params(str(_PROJECT_ROOT / "models" / "mini_hexapod_mjx.xml"),
                           cmd_mask_name="paper_stance")
    assert p.dyn_cfg.ctrl_hold_steps == 4
    st = hx.reset(p, jax.random.PRNGKey(0), gait_scale=1.0)
    step = jax.jit(lambda s: hx.step_foot(p, s, jnp.zeros(18, jnp.float32)))
    ctrls = []
    for _ in range(9):
        st = step(st)
        ctrls.append(np.asarray(st.mjx_data.ctrl))
        np.testing.assert_allclose(np.asarray(st.obs[:18]), ctrls[-1], atol=1e-6)
    # steps 1-3 hold the target latched at step 0, step 4 latches a new one, ...
    assert np.array_equal(ctrls[0], ctrls[1]) and np.array_equal(ctrls[1], ctrls[3])
    assert not np.array_equal(ctrls[3], ctrls[4])
    assert np.array_equal(ctrls[4], ctrls[7]) and not np.array_equal(ctrls[7], ctrls[8])
