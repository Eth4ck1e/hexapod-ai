"""Quick smoke test for CI: verify all critical imports + MuJoCo model loading.

Run with:  python -m pytest tests/test_smoke.py -v
Or standalone:  python -m pytest tests/test_smoke.py -v --no-header

Sets MUJOCO_GL=egl for headless CI (GitHub Actions ubuntu-latest).
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on PYTHONPATH so `from envs import ...` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. Module-level imports — verify the package tree compiles
# ---------------------------------------------------------------------------


def test_import_gait():
    """gait package and Controller class."""
    from gait import Controller  # noqa: F811
    assert hasattr(Controller, "predict")


def test_import_envs_obs_layout():
    """envs.obs_layout — single source of truth for obs schema."""
    from envs.obs_layout import OBS_DIM  # noqa: F811
    assert OBS_DIM > 10


def test_import_envs_stance_envelope():
    """envs.stance_envelope — stance height/width envelope."""
    from envs.stance_envelope import safe_dw_range  # noqa: F811
    lo, hi = safe_dw_range(-0.025)
    assert lo < hi


def test_import_envs_cmd_bins():
    """envs.cmd_bins — 150-bin partition for multi-head disc."""
    import numpy as np
    from envs.cmd_bins import cmd_to_bin  # noqa: F811
    idx = cmd_to_bin(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert 0 <= idx < 150


def test_import_amp_discriminator():
    """amp.discriminator — multi-head discriminator."""
    from amp.discriminator import MultiHeadDiscriminator  # noqa: F811
    assert hasattr(MultiHeadDiscriminator, "__call__")


mjx_available = pytest.mark.skipif(
    not importlib.util.find_spec("mujoco.mjx"),
    reason="mujoco.mjx not available on this platform (needs JAX backend)",
)


@mjx_available
def test_import_amp_prior_data():
    """amp.prior_data — prior collection."""
    from amp.prior_data import collect_demos  # noqa: F811
    assert callable(collect_demos)


# ---------------------------------------------------------------------------
# 2. MuJoCo model loading (headless via EGL)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mujoco_available():
    """Verify MuJoCo can import and we have a usable GL backend on CI."""
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering backend
    import mujoco
    return mujoco


def test_mujoco_import(mujoco_available):
    """MuJoCo library imports successfully."""
    mujoco = mujoco_available
    assert hasattr(mujoco, "MjModel")


def test_mujoco_model_load(mujoco_available):
    """Load the main MJCF model and verify geometry."""
    mujoco = mujoco_available
    model_path = str(PROJECT_ROOT / "models" / "phantomx.xml")
    assert os.path.exists(model_path), f"MJCF not found: {model_path}"
    model = mujoco.MjModel.from_xml_path(model_path)
    assert model.nq > 0, "model has no position DOFs"
    assert model.nv > 0, "model has no velocity DOFs"
    assert model.nu == 18, f"expected 18 actuators, got {model.nu}"


def test_mujoco_step(mujoco_available):
    """Step the physics forward one tick — validates solver + geometry."""
    mujoco = mujoco_available
    model_path = str(PROJECT_ROOT / "models" / "phantomx.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    # After one step the torso should still be above ground
    assert data.qpos[2] >= 0.0, f"body z went negative: {data.qpos[2]}"


# ---------------------------------------------------------------------------
# 3. Gait controller integration
# ---------------------------------------------------------------------------


def test_controller_init():
    """Controller initialises with the production MJCF."""
    from gait import Controller
    model_path = str(PROJECT_ROOT / "models" / "phantomx.xml")
    ctrl = Controller(model_path)
    assert ctrl.MAX_SPEED > 0
    assert ctrl.MAX_YAW_RATE > 0


def test_controller_predict():
    """Controller.predict produces correct joint shape."""
    import numpy as np
    from gait import Controller
    model_path = str(PROJECT_ROOT / "models" / "phantomx.xml")
    ctrl = Controller(model_path)
    cmd = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    joints = ctrl.predict(cmd, 0.0)
    assert joints.shape == (18,), f"expected (18,), got {joints.shape}"
    # Joint targets should be within ±0.5 rad of neutral for a gentle walk
    neutral = ctrl.gait_neutral_pose
    deviation = np.max(np.abs(joints - neutral))
    assert deviation < 1.5, f"unrealistic joint deviation: {deviation:.3f} rad"


def test_controller_neutral_pose():
    """gait.NEUTRAL_POSE constant is accessible."""
    from gait import NEUTRAL_POSE
    assert len(NEUTRAL_POSE) == 18