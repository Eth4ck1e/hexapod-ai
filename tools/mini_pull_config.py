"""tools/mini_pull_config.py — pull the mini hexapod's saved calibration over USB.

Works with BOTH firmware generations:
  * v0.1 (May 2026, on the original XIAO ESP32-C3): menu `X` prints the
    configuration as C++ constants (geometry, per-joint trim µs, signs, gait).
    No travel limits existed yet.
  * v0.2+ : menu `J` prints one JSON object between BEGIN_JSON / END_JSON.

  * stage sketches (no export at all, e.g. 05_walking_controller): dump the
    NVS partition read-only and convert it —
        esptool --chip esp32c3 --port PORT read_flash 0x9000 0x5000 nvs.bin
        python tools/nvs_dump.py nvs.bin --json nvs.json
        python tools/mini_pull_config.py --nvs-json nvs.json --out <date>.json
    (this is how the original C3's calibration was recovered on 2026-11-19).

Either way the result is written in the v0.2 JSON schema (the one
`tools/import_mini_calibration.py` and `tools/mini_push_config.py` read), plus
the raw serial capture next to it for the record:

    python tools/mini_pull_config.py --port /dev/cu.usbmodem101 \
        --out mini_hexapod/calibration/2026-11-19_c3.json

Servo power may stay off — the export modes move nothing.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import serial

FW_LEGS = ("FR", "MR", "RR", "FL", "ML", "RL")
# Fixed in the firmware source (both versions): board address + PCA9685 channels.
FW_BOARD = {"FR": 64, "MR": 64, "RR": 64, "FL": 65, "ML": 65, "RL": 65}
FW_CHANNELS = {"FR": [0, 1, 2], "MR": [3, 4, 5], "RR": [6, 7, 8],
               "FL": [0, 1, 2], "ML": [3, 4, 5], "RL": [6, 7, 8]}
SERVO = {"min_us": 500, "max_us": 2500, "us_per_deg": 2000.0 / 180.0, "pwm_hz": 50}


def read_until(ser: serial.Serial, stop: str, timeout: float) -> str:
    buf = ""
    t0 = time.time()
    while time.time() - t0 < timeout:
        chunk = ser.read(ser.in_waiting or 1).decode(errors="replace")
        if chunk:
            buf += chunk
            if stop in buf:
                break
    return buf


def banner_version(text: str) -> str | None:
    m = re.search(r"mini_hexapod unified firmware\s+v([\d.]+)", text)
    return m.group(1) if m else None


# --- v0.1 C++ export parser --------------------------------------------------
def _float(text: str, name: str) -> float:
    m = re.search(rf"constexpr float {name}\s*=\s*([-\d.]+)f?;", text)
    if not m:
        raise ValueError(f"{name} not found in export")
    return float(m.group(1))


def _array6(text: str, name: str) -> list[float]:
    m = re.search(rf"const float {name}\[6\]\s*=\s*\{{(.*?)\}};", text, re.S)
    if not m:
        raise ValueError(f"{name} not found in export")
    vals = re.findall(r"^\s*([-\d.]+)f?,", m.group(1), re.M)
    if len(vals) != 6:
        raise ValueError(f"{name}: expected 6 values, got {len(vals)}")
    return [float(v) for v in vals]


def _table6x3(text: str, decl: str, conv) -> dict[str, list]:
    m = re.search(rf"{re.escape(decl)}\s*=\s*\{{(.*?)\n\}};", text, re.S)
    if not m:
        raise ValueError(f"{decl} not found in export")
    rows = re.findall(r"/\*\s*(\w+)\s*\*/\s*\{([^}]*)\}", m.group(1))
    out = {name: [conv(x.strip()) for x in body.split(",")] for name, body in rows}
    missing = [n for n in FW_LEGS if n not in out]
    if missing:
        raise ValueError(f"{decl}: legs missing {missing}")
    return out


def parse_cpp_export(text: str, fw_version: str | None) -> dict:
    L2, L3 = _float(text, "L2"), _float(text, "L3")
    r_home, d_home = _float(text, "R_HOME"), _float(text, "D_HOME")
    cx, cy, cz = _array6(text, "COXA_X"), _array6(text, "COXA_Y"), _array6(text, "COXA_Z")
    axis = _array6(text, "AXIS_RAD")
    trims = _table6x3(text, "constexpr uint16_t TRIM_US[6][3]", int)
    signs = _table6x3(text, "constexpr int8_t SIGN[6][3]", int)
    step_len = _float(text, "STEP_LENGTH")
    step_ht = _float(text, "STEP_HEIGHT")
    m = re.search(r"STEP_PERIOD_MS\s*=\s*(\d+);", text)
    period = int(m.group(1)) if m else None
    # Home kinematic angles, same formula as firmware computeHomeKinAngles():
    # the L-pose rest point at horizontal reach R_HOME and depth D_HOME.
    home = home_kin_angles(L2, L3, r_home, d_home)
    legs = []
    for i, name in enumerate(FW_LEGS):
        legs.append({
            "name": name, "right": name.endswith("R"),
            "board": FW_BOARD[name], "channels": FW_CHANNELS[name],
            "coxa_mm": [cx[i], cy[i], cz[i]], "axis_rad": axis[i],
            "trim_us": trims[name], "sign": signs[name],
            "range_lo_us": [0, 0, 0], "range_hi_us": [0, 0, 0],
            "range_kin_rad": [None, None, None],
        })
    return {
        "firmware": f"mini_hexapod unified v{fw_version or '0.1'} (C++ export, converted)",
        "servo": SERVO,
        "geometry_mm": {"L2": L2, "L3": L3, "R_HOME": r_home, "D_HOME": d_home},
        "home_kin_angle_rad": home,
        "gait": {"step_length_mm": step_len, "step_height_mm": step_ht, "step_period_ms": period},
        "joints": ["COXA", "FEMUR", "KNEE"],
        "legs": legs,
    }


def home_kin_angles(L2: float, L3: float, r: float, d: float) -> list[float | None]:
    """Planar 2-link IK of the rest foot point (r out, d down) → [coxa, femur, knee]."""
    try:
        h = math.hypot(r, d)
        knee_inner = math.acos(max(-1.0, min(1.0, (L2 * L2 + L3 * L3 - h * h) / (2 * L2 * L3))))
        femur = math.atan2(-d, r) + math.acos(max(-1.0, min(1.0, (L2 * L2 + h * h - L3 * L3) / (2 * L2 * h))))
        return [0.0, round(femur, 5), round(math.pi - knee_inner, 5)]
    except ValueError:
        return [0.0, None, None]


def from_nvs_dump(nvs: dict, note: str) -> dict:
    """tools/nvs_dump.py JSON (hexa_trim / hexa_sign namespaces) → calibration schema.
    Geometry is the firmware's compile-time default (identical in every sketch so far)."""
    L2, L3 = 55.0, 89.5
    coxa_x = [+56.75, 0.0, -56.75, +56.75, 0.0, -56.75]
    coxa_y = [-32.0, -42.4, -32.0, +32.0, +42.4, +32.0]
    axis_deg = [-45.0, -90.0, -135.0, 45.0, 90.0, 135.0]
    legs = []
    for i, name in enumerate(FW_LEGS):
        legs.append({
            "name": name, "right": name.endswith("R"),
            "board": FW_BOARD[name], "channels": FW_CHANNELS[name],
            "coxa_mm": [coxa_x[i], coxa_y[i], 0.0], "axis_rad": math.radians(axis_deg[i]),
            "trim_us": [int(nvs["hexa_trim"][f"l{i}_j{j}"]) for j in range(3)],
            "sign": [int(nvs["hexa_sign"][f"l{i}_j{j}"]) for j in range(3)],
            "range_lo_us": [int(nvs.get("hexa_range", {}).get(f"lo_{i}_{j}", 0)) for j in range(3)],
            "range_hi_us": [int(nvs.get("hexa_range", {}).get(f"hi_{i}_{j}", 0)) for j in range(3)],
            "range_kin_rad": [None, None, None],
        })
    gait = nvs.get("hexa_gait")
    return {
        "firmware": note,
        "servo": SERVO,
        "geometry_mm": {"L2": L2, "L3": L3, "R_HOME": L2, "D_HOME": L3},
        "home_kin_angle_rad": home_kin_angles(L2, L3, L2, L3),
        "gait": ({"step_length_mm": gait.get("step_len"), "step_height_mm": gait.get("step_ht"),
                  "step_period_ms": gait.get("step_per")} if gait else None),
        "joints": ["COXA", "FEMUR", "KNEE"],
        "legs": legs,
    }


def parse_json_export(text: str) -> dict:
    m = re.search(r"BEGIN_JSON\s*(\{.*\})\s*END_JSON", text, re.S)
    if not m:
        raise ValueError("no BEGIN_JSON … END_JSON block")
    return json.loads(m.group(1))


# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="serial port of a board running the unified firmware")
    ap.add_argument("--nvs-json", type=Path, default=None,
                    help="instead of --port: tools/nvs_dump.py output of a raw NVS partition "
                         "(boards running a stage sketch with no export command)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", type=Path, required=True, help="JSON file to write (raw capture goes next to it)")
    ap.add_argument("--mode", choices=["auto", "cpp", "json"], default="auto",
                    help="which export menu to use; auto = J if the banner says v0.2+, else X")
    args = ap.parse_args()

    if args.nvs_json:
        with open(args.nvs_json) as fh:
            nvs = json.load(fh)
        data = from_nvs_dump(nvs,
                             f"NVS dump {args.nvs_json.name} (stage sketch; geometry = sketch constants)")
        data["pulled"] = {"source": str(args.nvs_json), "when": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "nvs"}
        args.out.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {args.out} from {args.nvs_json}")
        return
    if not args.port:
        ap.error("--port or --nvs-json required")
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    time.sleep(1.5)                       # native-USB boards may reset on open; let the banner land
    boot = ser.read(ser.in_waiting or 0).decode(errors="replace")
    version = banner_version(boot)
    if version is None:
        # No reset happened: ask the main menu to identify itself.
        ser.write(b"m\n"); time.sleep(0.3)
        ser.write(b"?\n")
        menu = read_until(ser, "reprint this menu", 3.0)
        boot += menu
        version = "0.2+" if ("J  export" in menu or "P  policy" in menu) else ("0.1" if "MAIN MENU" in menu else None)
    print(f"firmware banner/menu → version {version or 'unknown'}")

    mode = args.mode
    if mode == "auto":
        mode = "json" if (version or "").startswith(("0.2", "0.3", "0.4", "1")) else "cpp"
    ser.reset_input_buffer()
    ser.write(b"m\n"); time.sleep(0.2); ser.reset_input_buffer()
    ser.write(b"J\n" if mode == "json" else b"X\n")
    text = read_until(ser, "Press any key for main menu", 8.0)
    ser.write(b"\n")                      # dismiss "press any key"
    time.sleep(0.2)
    ser.close()

    raw_path = args.out.with_suffix(".raw.txt")
    raw_path.write_text(boot + "\n" + text)
    if "Press any key" not in text:
        print(text[-1500:])
        sys.exit(f"export did not complete; raw capture saved to {raw_path}")

    data = parse_json_export(text) if mode == "json" else parse_cpp_export(text, version)
    data["pulled"] = {"port": args.port, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": mode}
    args.out.write_text(json.dumps(data, indent=2) + "\n")

    g = data["geometry_mm"]
    print(f"wrote {args.out}  (raw: {raw_path})")
    print(f"  L2={g['L2']} L3={g['L3']} R_HOME={g['R_HOME']} D_HOME={g['D_HOME']}  gait={data['gait']}")
    print(f"  {'leg':>3} {'coxa_mm':>22} {'axis°':>7}  trim_us (coxa femur knee)   sign")
    for leg in data["legs"]:
        cx, cy, cz = leg["coxa_mm"]
        print(f"  {leg['name']:>3} ({cx:7.2f},{cy:7.2f},{cz:6.2f}) {math.degrees(leg['axis_rad']):7.1f}  "
              f"{leg['trim_us'][0]:5d} {leg['trim_us'][1]:5d} {leg['trim_us'][2]:5d}        "
              f"{leg['sign'][0]:+d} {leg['sign'][1]:+d} {leg['sign'][2]:+d}")
    n_trim_default = sum(1 for leg in data["legs"] for us in leg["trim_us"] if us == 1500)
    if n_trim_default == 18:
        print("WARNING: every trim is 1500 µs — this board has no saved calibration.")


if __name__ == "__main__":
    main()
