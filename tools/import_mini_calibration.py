"""tools/import_mini_calibration.py — firmware JSON export → simulator numbers.

The mini-hexapod firmware's `J` menu prints one JSON object between
BEGIN_JSON / END_JSON (geometry, per-joint trims, signs and the travel limits
measured in `L` range mode). Save that block to a file and run:

    python tools/import_mini_calibration.py mini_hexapod/calibration/2026-11-25.json

It prints:
  * per-leg joint ranges in the simulator's angle convention (rad),
  * the conservative per-joint-type range (intersection over legs, minus a
    margin) to paste into tools/gen_mini_hexapod_mjcf.py SERVO ranges,
  * geometry cross-checks against gait/robot_profile.py MINI_HEXAPOD.

Firmware kinematic angles coincide with the MJCF joint angles at the L-pose
(coxa 0, femur 0, tibia +π/2), so no extra conversion is needed; see the
`range_kin_rad` comment in hexapod_firmware.ino. Firmware leg order is
FR MR RR FL ML RL; the simulator's is RR RM RF LR LM LF — mapped below.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FW_TO_SIM = {"FR": "RF", "MR": "RM", "RR": "RR", "FL": "LF", "ML": "LM", "RL": "LR"}
SIM_ORDER = ("RR", "RM", "RF", "LR", "LM", "LF")
JOINT_KEYS = ("coxa_range", "femur_range", "tibia_range")


def load_export(path: Path) -> dict:
    text = path.read_text()
    m = re.search(r"BEGIN_JSON\s*(\{.*\})\s*END_JSON", text, re.S)
    return json.loads(m.group(1) if m else text)


def kin_rad(leg: dict, j: int, us: int, us_per_deg: float, home: list[float]) -> float:
    deg = (us - leg["trim_us"][j]) / us_per_deg
    return home[j] + leg["sign"][j] * math.radians(deg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_file", type=Path)
    ap.add_argument("--margin-deg", type=float, default=3.0,
                    help="shrink each measured stop by this much for the sim range")
    args = ap.parse_args()

    d = load_export(args.json_file)
    us_per_deg = float(d["servo"]["us_per_deg"])
    home = [float(x) for x in d["home_kin_angle_rad"]]
    legs = {FW_TO_SIM[leg["name"]]: leg for leg in d["legs"]}

    print(f"firmware: {d.get('firmware')}  L2={d['geometry_mm']['L2']} mm  L3={d['geometry_mm']['L3']} mm")
    print(f"home kinematic angles (rad): {[round(h, 4) for h in home]}")
    print("\nPer-leg travel (simulator rad), sim leg order:")
    print(f"{'leg':>4} {'coxa':>18} {'femur':>18} {'tibia':>18}")
    per_joint_lo = [[], [], []]
    per_joint_hi = [[], [], []]
    missing = []
    for name in SIM_ORDER:
        leg = legs[name]
        cells = []
        for j in range(3):
            lo_us, hi_us = leg["range_lo_us"][j], leg["range_hi_us"][j]
            if not lo_us or not hi_us:
                cells.append(f"{'(uncal)':>18}")
                missing.append(f"{name}/{j}")
                continue
            a, b = sorted((kin_rad(leg, j, lo_us, us_per_deg, home),
                           kin_rad(leg, j, hi_us, us_per_deg, home)))
            per_joint_lo[j].append(a)
            per_joint_hi[j].append(b)
            cells.append(f"{a:8.3f} .. {b:7.3f}")
        print(f"{name:>4} " + " ".join(cells))

    marg = math.radians(args.margin_deg)
    print(f"\n# tools/gen_mini_hexapod_mjcf.py SERVO ranges (intersection over legs, {args.margin_deg:g}° margin):")
    for j, key in enumerate(JOINT_KEYS):
        if not per_joint_lo[j]:
            print(f"    \"{key}\": None,   # no legs calibrated")
            continue
        lo = max(per_joint_lo[j]) + marg
        hi = min(per_joint_hi[j]) - marg
        print(f"    \"{key}\": ({lo:.3f}, {hi:.3f}),   # {math.degrees(lo):.0f}° .. {math.degrees(hi):.0f}°")
    if missing:
        print(f"\nWARNING: {len(missing)} joints uncalibrated: {', '.join(missing)}")

    try:
        from gait.robot_profile import MINI_HEXAPOD as P
        print("\nGeometry cross-check vs gait/robot_profile.py MINI_HEXAPOD:")
        print(f"  femur  firmware {d['geometry_mm']['L2']:.2f} mm   profile {P.femur_length*1e3:.2f} mm")
        print(f"  tibia  firmware {d['geometry_mm']['L3']:.2f} mm   profile {P.tibia_length*1e3:.2f} mm")
        for name in SIM_ORDER:
            leg = legs[name]
            i = SIM_ORDER.index(name)
            fw = leg["coxa_mm"]
            pr = P.coxa_pos_body[i] * 1e3
            flag = "" if max(abs(fw[0] - pr[0]), abs(fw[1] - pr[1])) < 0.5 else "   <-- differs"
            print(f"  {name} coxa firmware ({fw[0]:7.2f}, {fw[1]:7.2f})  profile ({pr[0]:7.2f}, {pr[1]:7.2f}){flag}")
    except Exception as e:  # profile import is a convenience only
        print(f"(profile cross-check skipped: {e})")


if __name__ == "__main__":
    main()
