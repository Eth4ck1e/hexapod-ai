"""tools/package_submission.py — assemble the class hand-in: clean firmware + model + docs.

    python tools/package_submission.py --policy checkpoints/<run>/params.pkl --out deliverables/mini_hexapod_submission
    python tools/package_submission.py --out ... --no-model       # firmware/docs only (model later)

Produces a self-contained folder (and a .zip next to it):
  firmware/hexapod_firmware/hexapod_firmware.ino   the unified firmware, comments scrubbed of
                                                   repo-internal paths/tool names, version-stamped
  firmware/flash.sh                                compile + flash helper (arduino-cli)
  firmware/README.md                               wiring, build, menu, calibration, stream protocol
  calibration/<latest>.json                        the robot's trims / signs / limits
  model/policy.npz, policy.json, policy_weights.h, reference_infer.py, README.md
  videos/, plots/                                  showcase MP4s + training/robustness figures
  README.md                                        top-level index + how to run the demo
Everything is derived from the repo at packaging time, so rerun after any change.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
FW_SRC = _ROOT / "mini_hexapod" / "firmware" / "arduino" / "hexapod_firmware" / "hexapod_firmware.ino"
FLASH_SRC = _ROOT / "mini_hexapod" / "firmware" / "flash.sh"
CAL_DIR = _ROOT / "mini_hexapod" / "calibration"

# Comment fragments that only make sense inside this repository. Each (pattern, replacement)
# is applied to comment text only; code lines are never touched.
SCRUB = [
    (r"\(?tools/mini_push_config\.py\)?", "(host upload tool)"),
    (r"tools/mini_pull_config\.py", "the host pull tool"),
    (r"scripts/drive_serial\.py", "the host driver"),
    (r"tools/import_mini_calibration\.py", "the simulator importer"),
    (r"mini_hexapod/calibration/[\w.-]+\.json", "the calibration JSON shipped with this package"),
    (r"gait/robot_profile\.py", "the simulator's robot profile"),
    (r"envs/[\w./-]+", "the simulator"),
    (r"\s*\(p1[–-]p3[^)]*\)", ""),
]


def scrub_firmware(text: str, version_note: str) -> str:
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            for pat, rep in SCRUB:
                line = re.sub(pat, rep, line)
        elif "//" in line:                      # trailing comment on a code line
            code, _, comment = line.partition("//")
            for pat, rep in SCRUB:
                comment = re.sub(pat, rep, comment)
            line = code + "//" + comment
        out.append(line)
    header = (f"// Packaged for submission {time.strftime('%Y-%m-%d')} — {version_note}\n"
              "// Build: Arduino IDE / arduino-cli, board \"XIAO_ESP32S3\" (or XIAO_ESP32C3),\n"
              "// library \"Adafruit PWM Servo Driver\". See README.md.\n")
    return header + "\n".join(out) + "\n"


FW_README = """# Mini hexapod firmware

Unified firmware for the 18-servo mini hexapod: Seeed XIAO ESP32-S3 (or ESP32-C3) driving two
PCA9685 PWM boards (right legs 0x40, left legs 0x41), 18 × 9 g servos at 50 Hz.

## Build and flash

Arduino IDE: board *XIAO_ESP32S3*, USB mode *Hardware CDC and JTAG*, library *Adafruit PWM Servo
Driver*. Or from a shell with arduino-cli:

    bash flash.sh s3 /dev/cu.usbmodemXXXX      # compile + flash (c3 for the ESP32-C3)

Flashing never erases the saved calibration (NVS). A blank board starts from the calibration
baked into the sketch (`DEFAULT_TRIM_US`, `DEFAULT_SIGN`), which is this robot's.

## Wiring

| XIAO pad | Signal |
|---|---|
| D4 | I2C SDA to both PCA9685 boards |
| D5 | I2C SCL |
| GND | common ground with the servo supply |
| 3V3 / 5V | PCA9685 logic VCC |

Servo power (5–6 V, several amps) goes to the PCA9685 V+ terminals, never through the XIAO.

## Serial menu (115200 baud)

| Key | Mode |
|---|---|
| G | geometry (leg lengths, coxa positions) |
| T | per-joint trim at the L-pose (a/d 1°, q/e 5°, s save, n skip) |
| S | per-joint sign discovery (sweeps each joint, asks which way it moved) |
| L | joint travel limits (jog to each stop, [ and ] mark, s save) |
| I | inverse-kinematics foot demo |
| W | walking controller (w/s/a/d/q/e moves, [ ] step length, , . step height, < > period, S save) |
| P | policy stream: host sends `J a0 … a17` (18 joint angles, radians) at 50 Hz; `H` home, `Z` relax, `Q` quit. If no frame arrives for 500 ms the legs home. |
| U | upload a saved calibration from the host (lines `T leg joint us`, `S leg joint ±1`, `R leg joint lo hi`, `G key value`, `E` ends) |
| J | export the full configuration as JSON |
| X | export as C++ constants |
| C | I2C scan (both drivers should answer) |
| Z | relax: all servo pulses off (lower the body first) |
| R | wipe NVS (asks for confirmation) |

Joint-angle convention for `P`/`J`: simulator order RR RM RF LR LM LF × (coxa, femur, tibia);
the L-pose (coxa straight out, femur level, tibia vertical) is (0, 0, +π/2) rad. The firmware maps to
its own leg order (FR MR RR FL ML RL) and applies the stored trims and signs.

## Calibration workflow

1. `T` trims so every joint sits at the L-pose, 2. `S` signs, 3. `L` limits, 4. `J` export and keep
the JSON (one is shipped under `../calibration/`).

## Safety

Servos stall and heat up when the robot stands powered for minutes; use `Z` (or the host's `rest`)
between runs. Pulling USB while streaming homes the legs within half a second.
"""

MODEL_README = """# Trained walking policy (model-only controller)

`policy.npz` + `policy.json` hold a 21-input, 18-output multilayer perceptron (hidden 256, 128, 64,
swish activations) trained in simulation: behaviour cloning from the analytical gait's dataset,
then PPO with Adversarial Motion Priors. `reference_infer.py` is a dependency-free numpy
implementation that reproduces the training framework's deterministic action to 1e-6;
`policy_weights.h` is the same network as C arrays for the on-board port.

Inputs (see `policy.json: obs_layout`): IMU quaternion + gyro + accel slots (constants when no IMU
is fitted), gait clock as sin/cos of phase (phase = t / gait_period mod 1), and the 9-value command
(vx, vy, wz, pitch, roll, height delta, width delta, shift x, shift y). Output: 18 joint targets,
`neutral_pose + tanh(mean) * residual_scale_max` radians, simulator joint order and convention.

Run it against the robot with the host driver (`scripts/drive_serial.py --params …` in the
repository) or port `reference_infer.py` to C using `policy_weights.h`.
"""

TOP_README = """# Mini hexapod — learned locomotion controller (class submission)

Contents

- `firmware/` — ESP32 firmware (menu-driven calibration, analytical walking controller, serial policy
  stream) + flash helper + README.
- `calibration/` — this robot's joint trims, signs and travel limits (JSON exported by the firmware).
- `model/` — the trained policy (weights, metadata, reference inference, C header) + README.
- `videos/` — simulation showcase videos. `plots/` — training curves and robustness tables.
- `docs/` — presentation notes and demo runbook.

Quick demo (tethered): flash the firmware, power the servos, then from the repository
`HEXAPOD_ROBOT=mini_hexapod_pure python scripts/drive_serial.py --port <port> --source console
--params model/policy.pkl` and type `forward`, `spin left`, `up 10`, `rest` …

Full source, training pipeline and history: [repository link]
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", type=Path, default=None, help="Brax params.pkl to export as the model")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--version-note", default="firmware v0.3")
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip")
    args = ap.parse_args()

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "firmware" / "hexapod_firmware").mkdir(parents=True)
    (out / "calibration").mkdir()
    (out / "videos").mkdir()
    (out / "plots").mkdir()
    (out / "docs").mkdir()

    # Firmware
    (out / "firmware" / "hexapod_firmware" / "hexapod_firmware.ino").write_text(
        scrub_firmware(FW_SRC.read_text(), args.version_note))
    flash = FLASH_SRC.read_text().replace('SKETCH="$HERE/arduino/hexapod_firmware"', 'SKETCH="$HERE/hexapod_firmware"')
    (out / "firmware" / "flash.sh").write_text(flash)
    (out / "firmware" / "README.md").write_text(FW_README)

    # Calibration: the newest JSON that has trims (prefer *_trimmed / readback over raw pulls).
    cals = sorted(p for p in CAL_DIR.glob("*.json") if "nvs" not in p.name)
    if cals:
        latest = cals[-1]
        shutil.copy(latest, out / "calibration" / latest.name)

    # Model
    if not args.no_model:
        if not args.policy:
            sys.exit("--policy required unless --no-model")
        subprocess.run([sys.executable, str(_ROOT / "tools" / "export_policy_weights.py"), str(args.policy),
                        "--out", str(out / "model"), "--note", "class submission export"], check=True, cwd=_ROOT)
        shutil.copy(args.policy, out / "model" / "policy.pkl")
        (out / "model" / "README.md").write_text(MODEL_README)

    # Media + docs
    for v in sorted((_ROOT / "media" / "recordings").glob("*showcase*.mp4")):
        shutil.copy(v, out / "videos" / v.name)
    for p in (_ROOT / "media" / "plots").glob("*"):
        shutil.copy(p, out / "plots" / p.name)
    for d in ("final_project_presentation.md", "demo_script.txt", "kinematics.md"):
        src = _ROOT / "docs" / d
        if src.exists():
            shutil.copy(src, out / "docs" / d)
    (out / "README.md").write_text(TOP_README)

    # Sanity: no repo-internal paths left in the shipped firmware comments.
    leftovers = [ln for ln in (out / "firmware" / "hexapod_firmware" / "hexapod_firmware.ino").read_text().splitlines()
                 if re.search(r"tools/|scripts/|envs/|gait/|checkpoints/", ln)]
    if leftovers:
        print("WARNING: repo paths still present:\n  " + "\n  ".join(leftovers[:10]))

    for pyc in out.rglob("__pycache__"):
        shutil.rmtree(pyc)
    if args.zip:
        zpath = out.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(out.parent))
        print(f"zip: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    n = sum(1 for f in out.rglob("*") if f.is_file())
    print(f"packaged {n} files into {out}")


if __name__ == "__main__":
    main()
