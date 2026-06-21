"""Quick smoke test for CI: verify all critical imports + MuJoCo model loading.

Run with:  python -m pytest tests/test_smoke.py -v
"""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from envs.cmd_bins import cmd_to_bin
from envs.obs_layout import OBS_DIM
from envs.stance_envelope import safe_dw_range
from gait import Controller, NEUTRAL_POSE

# Ensure project root is on PYTHONPATH so `from envs import ...` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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
    model_path = str(PROJECT_ROOT / "models" / "phantomx.xml")
    assert os.path.exists(model_path), f"MJCF not found: {model_path}"
    model = mujoco_available.MjModel.from_xml_path(model_path)
    assert model.nq > 0, "model has no position DOFs"
    assert model.nv > 0, "model has no velocity DOFs"
    assert model.nu == 18, f"expected 18 actuators, got {model.nu}"


def test_mujoco_step(mujoco_available):
    """Step the physics forward one tick — validates solver + geometry."""
    model_path = str(PROJECT_ROOT / "models" / "phantomx.xml")
    model = mujoco_available.MjModel.from_xml_path(model_path)
    data = mujoco_available.MjData(model)
    mujoco_available.mj_step(model, data)
    assert data.qpos[2] >= 0.0, f"body z went negative: {data.qpos[2]}"


def test_controller_init():
    """Controller initialises with the production MJCF."""
    model_path = str(PROJECT_ROOT / "models" / "phantomx.xml")
    ctrl = Controller(model_path)
    assert ctrl.MAX_SPEED > 0
    assert ctrl.MAX_YAW_RATE > 0


def test_controller_predict():
    """Controller.predict produces correct joint shape."""
    model_path = str(PROJECT_ROOT / "models" / "phantomx.xml")
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
