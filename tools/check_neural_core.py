"""tools/check_neural_core.py — compile the firmware's neural_core.h on the host and
compare it with the Python reference (numpy inference + the gait scaffold's standing pose).

    HEXAPOD_ROBOT=mini_hexapod_pure PYTHONPATH=. python tools/check_neural_core.py \
        [--params checkpoints/.../params.pkl]   # default: the checkpoint named in policy_weights.h

Policy: 512 random (phase, cmd) draws inside the trained ranges -> max |Δjoint|.
Stance: a grid over the (dh, dw) envelope -> max |Δjoint| vs Controller.predict at zero velocity.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
FW = _ROOT / "mini_hexapod/firmware/arduino/hexapod_firmware"
TEST_C = _ROOT / "mini_hexapod/firmware/host_test/neural_core_test.c"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=None)
    ap.add_argument("--n", type=int, default=512)
    args = ap.parse_args()

    header = (FW / "policy_weights.h").read_text()
    src = args.params or re.search(r"// source: (\S+)", header).group(1)
    pitch_max = float(re.search(r"POLICY_PITCH_MAX (\S+)f", header).group(1))
    roll_max = float(re.search(r"POLICY_ROLL_MAX (\S+)f", header).group(1))
    print(f"policy_weights.h built from {src}  (tilt ±{np.degrees(pitch_max):.0f}°/±{np.degrees(roll_max):.0f}°)")

    import chain_train
    from drive_serial import hardware_obs, load_policy

    from envs.hexapod_env_jax import EnvParams
    from envs.stance_envelope import envelope_for
    from gait.controller import Controller
    ctrl = Controller(str(_ROOT / chain_train.MODEL_PATH))
    env = envelope_for(ctrl.profile)
    infer = load_policy(str(_ROOT / src) if not Path(src).is_absolute() else src, 21)
    scale = float(EnvParams._field_defaults["residual_scale_max"])
    neutral = ctrl.gait_neutral_pose.astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        exe = Path(td) / "nc_test"
        subprocess.run(["cc", "-O2", "-std=c99", "-o", str(exe), str(TEST_C), "-lm"], check=True)

        rng = np.random.default_rng(0)
        lines, ref = [], []
        for _ in range(args.n):
            phase = rng.uniform(0, 1)
            cmd = np.zeros(9, np.float32)
            speed = ctrl.MAX_SPEED * rng.uniform(0.4, 0.85)
            h = rng.uniform(-np.pi, np.pi)
            kind = rng.integers(3)
            if kind == 0:   cmd[0], cmd[1] = speed * np.cos(h), speed * np.sin(h)
            elif kind == 1: cmd[2] = rng.choice([-1, 1]) * ctrl.MAX_YAW_RATE * rng.uniform(0.4, 1.0)
            else:           cmd[0] = speed; cmd[2] = rng.choice([-1, 1]) * speed / rng.uniform(ctrl.MIN_TURN_RADIUS, ctrl.MAX_TURN_RADIUS)
            cmd[5] = rng.uniform(env.dh_min, env.dh_max)
            cmd[6] = rng.uniform(*env.safe_dw_range(cmd[5]))
            if pitch_max > 0:
                cmd[3] = rng.uniform(-pitch_max, pitch_max); cmd[4] = rng.uniform(-roll_max, roll_max)
            obs = hardware_obs(neutral, neutral, 0.005, np.zeros(18), phase, cmd, variant="hardware_pure")
            ref.append(neutral + np.tanh(infer(obs)) * scale)
            lines.append("P " + " ".join(f"{v:.8f}" for v in [phase, *cmd]))
        n_pol = len(lines)
        for dh in np.linspace(env.dh_min, env.dh_max, 9):
            for dw in np.linspace(*env.safe_dw_range(dh), 7):
                c = np.zeros(9); c[5], c[6] = dh, dw
                ref.append(ctrl.predict_with_feet(c, 0.0)[0])
                lines.append(f"S {dh:.8f} {dw:.8f}")
        for pitch, roll in ((0.17, 0.0), (0.0, -0.26), (0.1, 0.12), (-0.15, 0.2)):
            for dh in (env.dh_min, 0.0, env.dh_max):
                c = np.zeros(9); c[3], c[4], c[5] = pitch, roll, dh
                ref.append(ctrl.predict_with_feet(c, 0.0)[0])
                lines.append(f"T {dh:.8f} 0 {pitch:.8f} {roll:.8f}")
        out = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout
        got = np.array([[float(v) for v in ln.split()] for ln in out.strip().splitlines()])
    ref = np.array(ref)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    d = np.abs(got - ref)
    print(f"policy: {n_pol} draws, max |Δjoint| {d[:n_pol].max():.2e} rad ({np.degrees(d[:n_pol].max()):.4f}°)")
    print(f"stance: {len(ref) - n_pol} (dh, dw, pitch, roll) points, max |Δjoint| {d[n_pol:].max():.2e} rad")
    ok = d[:n_pol].max() < 2e-4 and d[n_pol:].max() < 1e-5
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
