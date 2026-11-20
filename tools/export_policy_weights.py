"""tools/export_policy_weights.py — turn a Brax policy checkpoint into portable files.

    python tools/export_policy_weights.py checkpoints/<run>/params.pkl --out deliverables/model
        [--obs-variant hardware_pure] [--action-space joint]

Writes:
  policy.npz          float32 arrays: obs_mean, obs_std, W0,b0 … Wn,bn (dense layers, swish between)
  policy.json         metadata: obs layout (slot names + dims), action convention, hidden sizes,
                      joint order, residual scale, sim timestep, servo hold, provenance
  policy_weights.h    C header with the same numbers (for the on-board port)
  reference_infer.py  pure-numpy inference that reproduces Brax's deterministic action exactly;
                      the exporter checks it against the Brax network on random inputs before writing.

Action convention (ACTION_SPACE "joint"): joint_target = neutral + tanh(mean) * residual_scale_max,
18 joints in simulator leg order RR RM RF LR LM LF × (coxa, femur, tibia), radians, MJCF
convention (L-pose = 0, 0, +pi/2). The normalizer and network come straight from Brax: obs is
normalized as (obs - mean) / max(std, 1e-6) ... see the generated reference for the exact form.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

REFERENCE = '''"""Pure-numpy inference for the exported hexapod policy (no JAX / Brax needed).

    import numpy as np, reference_infer as ri
    net = ri.load("policy.npz", "policy.json")
    joints = ri.joint_targets(net, obs)        # obs: float32 vector in the documented layout
"""
import json
import numpy as np


def load(npz_path, json_path):
    w = np.load(npz_path)
    meta = json.load(open(json_path))
    layers = [(w[f"W{i}"], w[f"b{i}"]) for i in range(meta["n_layers"])]
    return {"obs_mean": w["obs_mean"], "obs_std": w["obs_std"], "layers": layers, "meta": meta}


def swish(x):
    return x / (1.0 + np.exp(-x))


def policy_mean(net, obs):
    """Raw network mean output (before tanh)."""
    x = (np.asarray(obs, dtype=np.float32) - net["obs_mean"]) / net["obs_std"]
    layers = net["layers"]
    for W, b in layers[:-1]:
        x = swish(x @ W + b)
    W, b = layers[-1]
    out = x @ W + b
    return out[: net["meta"]["action_dim"]]


def action(net, obs):
    """Deterministic Brax action = tanh(mean), in [-1, 1]."""
    return np.tanh(policy_mean(net, obs))


def joint_targets(net, obs):
    """Joint angles in radians (simulator convention + leg order), ACTION_SPACE joint."""
    m = net["meta"]
    return np.asarray(m["neutral_pose_rad"], dtype=np.float32) + action(net, obs) * m["residual_scale_max"]
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=None, help="MJCF (default: chain_train MODEL_PATH)")
    ap.add_argument("--note", default="", help="free-text provenance note")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import chain_train
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo.networks import make_ppo_networks
    from envs.obs_layout import OBS_SLOT_DIMS, active_variant, obs_dim, obs_slots
    from gait.controller import Controller
    from envs.hexapod_env_jax import EnvParams
    from pretrain_bc_jax import POLICY_HIDDEN_LAYERS
    import mujoco

    model_path = args.model or chain_train.MODEL_PATH
    ctrl = Controller(str(_ROOT / model_path) if not Path(model_path).is_absolute() else model_path)
    prof = ctrl.profile
    variant = active_variant(prof.obs_variant)
    n_obs = obs_dim(variant)

    with open(args.params, "rb") as f:
        loaded = pickle.load(f)
    norm, pol = loaded[0], loaded[1]
    assert int(norm.mean.shape[-1]) == n_obs, f"checkpoint obs {norm.mean.shape[-1]} != session obs {n_obs}"

    # Flatten dense layers in order.
    params = pol["params"]
    names = sorted(params.keys(), key=lambda k: int(k.split("_")[-1]))
    Ws = [np.asarray(params[n]["kernel"], dtype=np.float32) for n in names]
    bs = [np.asarray(params[n]["bias"], dtype=np.float32) for n in names]
    hidden = [int(W.shape[1]) for W in Ws[:-1]]
    assert tuple(hidden) == tuple(POLICY_HIDDEN_LAYERS), (hidden, POLICY_HIDDEN_LAYERS)
    obs_mean = np.asarray(norm.mean, dtype=np.float32)
    obs_std = np.asarray(norm.std, dtype=np.float32)

    # Brax reference for verification.
    nets = make_ppo_networks(n_obs, 18, preprocess_observations_fn=running_statistics.normalize,
                             policy_hidden_layer_sizes=POLICY_HIDDEN_LAYERS)
    rng = np.random.default_rng(0)
    test_obs = (obs_mean + obs_std * rng.standard_normal((256, n_obs))).astype(np.float32)
    ref = np.asarray(jnp.tanh(nets.policy_network.apply(norm, pol, jnp.asarray(test_obs))[..., :18]))

    # Numpy reproduction (same math as the generated reference module).
    def swish(x):
        return x / (1.0 + np.exp(-x))
    x = (test_obs - obs_mean) / obs_std
    for W, b in zip(Ws[:-1], bs[:-1]):
        x = swish(x @ W + b)
    out = np.tanh((x @ Ws[-1] + bs[-1])[:, :18])
    err = float(np.abs(out - ref).max())
    print(f"numpy vs Brax max |Δaction| over 256 random obs: {err:.2e}")
    assert err < 1e-4, "exported math does not reproduce Brax — refusing to write"

    dt = float(mujoco.MjModel.from_xml_path(str(_ROOT / model_path)).opt.timestep)
    slots = [(k, OBS_SLOT_DIMS[k]) for k in obs_slots(variant)]
    meta = {
        "source_checkpoint": str(args.params),
        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": args.note,
        "robot": prof.name,
        "obs_variant": variant,
        "obs_dim": n_obs,
        "obs_layout": [{"name": k, "dim": d} for k, d in slots],
        "obs_notes": {
            "imu_quat": "w,x,y,z body orientation; constant [1,0,0,0] when no IMU",
            "imu_gyro": "rad/s body rates; zeros when no IMU",
            "imu_accel": "always zeros (placeholder)",
            "phase_sc": "[sin(2*pi*phase), cos(2*pi*phase)], phase = (t / gait_period) mod 1",
            "cmd": "vx m/s, vy m/s, wz rad/s, pitch rad, roll rad, height_delta m (+ = lower), width_delta m, shift_x, shift_y",
        },
        "normalization": "(obs - obs_mean) / obs_std, then dense layers with swish, last layer linear; action = tanh(out[:18])",
        "hidden_layers": hidden,
        "n_layers": len(Ws),
        "action_dim": 18,
        "action_space": chain_train.ACTION_SPACE,
        "residual_scale_max": float(EnvParams._field_defaults["residual_scale_max"]),
        "neutral_pose_rad": [float(v) for v in ctrl.gait_neutral_pose],
        "joint_order": [f"{leg}_{j}" for leg in ("RR", "RM", "RF", "LR", "LM", "LF") for j in ("coxa", "femur", "tibia")],
        "joint_convention": "MJCF/sim: coxa +z, femur -y, tibia +y; L-pose = (0, 0, +pi/2); firmware maps to its own leg order and applies trims/signs",
        "policy_rate_hz": round(1.0 / dt),
        "servo_command_hold_steps": int(prof.ctrl_hold_steps),
        "gait_period_s": float(ctrl.gait_period),
        "max_speed_m_s": float(ctrl.MAX_SPEED),
        "max_yaw_rate_rad_s": float(ctrl.MAX_YAW_RATE),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    arrays = {"obs_mean": obs_mean, "obs_std": obs_std}
    for i, (W, b) in enumerate(zip(Ws, bs)):
        arrays[f"W{i}"] = W
        arrays[f"b{i}"] = b
    np.savez(args.out / "policy.npz", **arrays)
    (args.out / "policy.json").write_text(json.dumps(meta, indent=2) + "\n")
    (args.out / "reference_infer.py").write_text(REFERENCE)

    # C header for the firmware port.
    def carr(name, a):
        flat = np.asarray(a, dtype=np.float32).ravel()
        body = ",\n  ".join(", ".join(f"{v:.8e}f" for v in flat[i:i + 8]) for i in range(0, len(flat), 8))
        return f"static const float {name}[{len(flat)}] = {{\n  {body}\n}};\n"
    h = ["// Generated by tools/export_policy_weights.py — do not edit.",
         f"// {meta['robot']} policy, obs {n_obs} ({variant}), hidden {hidden}, action {chain_train.ACTION_SPACE}.",
         f"// source: {args.params}", "#pragma once",
         f"#define POLICY_OBS_DIM {n_obs}", "#define POLICY_ACT_DIM 18", f"#define POLICY_N_LAYERS {len(Ws)}",
         f"#define POLICY_RESIDUAL_SCALE {meta['residual_scale_max']:.8e}f",
         f"#define POLICY_GAIT_PERIOD_S {meta['gait_period_s']:.8e}f",
         "static const int POLICY_LAYER_SIZES[] = {" + ", ".join(str(n_obs) if i == 0 else str(Ws[i-1].shape[1]) for i in range(len(Ws)+1)) + "};",
         carr("POLICY_OBS_MEAN", obs_mean), carr("POLICY_OBS_STD", obs_std),
         carr("POLICY_NEUTRAL_POSE", meta["neutral_pose_rad"])]
    for i, (W, b) in enumerate(zip(Ws, bs)):
        h.append(carr(f"POLICY_W{i}", W))   # row-major [in][out]
        h.append(carr(f"POLICY_B{i}", b))
    (args.out / "policy_weights.h").write_text("\n".join(h))
    n_params = sum(W.size + b.size for W, b in zip(Ws, bs))
    print(f"wrote {args.out}/policy.npz, policy.json, policy_weights.h, reference_infer.py  "
          f"({n_params:,} parameters, {n_params * 4 / 1024:.0f} KB float32)")

    # Final check: the generated reference module reproduces the same numbers.
    sys.path.insert(0, str(args.out))
    import importlib
    ri = importlib.import_module("reference_infer")
    net = ri.load(str(args.out / "policy.npz"), str(args.out / "policy.json"))
    err2 = max(float(np.abs(ri.action(net, o) - r).max()) for o, r in zip(test_obs[:32], ref[:32]))
    print(f"reference_infer.py vs Brax: {err2:.2e}")
    assert err2 < 1e-4


if __name__ == "__main__":
    main()
