"""tools/mini_push_config.py — push a saved calibration into a mini-hexapod board.

Sends the JSON written by tools/mini_pull_config.py (or the firmware's own `J`
export) to firmware ≥ v0.3 through menu `U` (one setting per line, each
acknowledged), then runs `J` and diffs the board's export against the file.

    python tools/mini_push_config.py --port /dev/cu.usbmodem101 \
        mini_hexapod/calibration/2026-11-19_c3.json
    python tools/mini_push_config.py --dry-run mini_hexapod/calibration/2026-11-19_c3.json

--only trims,signs,ranges,geometry,gait   choose what to send (default: all present)
Servo power may stay off — nothing moves.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

GEOM_KEYS = {"L2": "l2", "L3": "l3", "R_HOME": "r_home", "D_HOME": "d_home"}


def lines_for(cal: dict, only: set[str]) -> list[str]:
    out: list[str] = []
    if "geometry" in only and cal.get("geometry_mm"):
        for K, k in GEOM_KEYS.items():
            out.append(f"G {k} {cal['geometry_mm'][K]:.3f}")
        for i, leg in enumerate(cal["legs"]):
            cx, cy, cz = leg["coxa_mm"]
            out += [f"G cx{i} {cx:.3f}", f"G cy{i} {cy:.3f}", f"G cz{i} {cz:.3f}", f"G aa{i} {leg['axis_rad']:.6f}"]
    for leg in cal["legs"]:
        name = leg["name"]
        if "trims" in only:
            out += [f"T {name} {j} {leg['trim_us'][j]}" for j in range(3)]
        if "signs" in only:
            out += [f"S {name} {j} {leg['sign'][j]:+d}" for j in range(3)]
        if "ranges" in only and any(leg.get("range_lo_us", [0, 0, 0])):
            out += [f"R {name} {j} {leg['range_lo_us'][j]} {leg['range_hi_us'][j]}" for j in range(3)]
    if "gait" in only and cal.get("gait"):
        g = cal["gait"]
        if all(g.get(k) is not None for k in ("step_length_mm", "step_height_mm", "step_period_ms")):
            out.append(f"W {g['step_length_mm']} {g['step_height_mm']} {g['step_period_ms']}")
    return out


def read_until(ser, stop: str, timeout: float) -> str:
    buf, t0 = "", time.time()
    while time.time() - t0 < timeout:
        chunk = ser.read(ser.in_waiting or 1).decode(errors="replace")
        if chunk:
            buf += chunk
            if stop in buf:
                break
    return buf


def export_json(ser) -> dict:
    ser.reset_input_buffer()
    ser.write(b"J\n")
    text = read_until(ser, "END_JSON", 8.0)
    ser.write(b"\n")
    m = re.search(r"BEGIN_JSON\s*(\{.*\})\s*END_JSON", text, re.S)
    if not m:
        sys.exit("no JSON export from the board (firmware < v0.2?)\n" + text[-800:])
    return json.loads(m.group(1))


def diff(sent: dict, got: dict, only: set[str]) -> list[str]:
    bad = []
    got_legs = {leg["name"]: leg for leg in got["legs"]}
    for leg in sent["legs"]:
        g = got_legs[leg["name"]]
        for key, flag in (("trim_us", "trims"), ("sign", "signs")):
            if flag in only and list(leg[key]) != list(g[key]):
                bad.append(f"{leg['name']} {key}: sent {leg[key]} board {g[key]}")
        if "ranges" in only and any(leg.get("range_lo_us", [0, 0, 0])):
            for key in ("range_lo_us", "range_hi_us"):
                if list(leg[key]) != list(g[key]):
                    bad.append(f"{leg['name']} {key}: sent {leg[key]} board {g[key]}")
    if "geometry" in only and sent.get("geometry_mm"):
        for K in GEOM_KEYS:
            if abs(float(sent["geometry_mm"][K]) - float(got["geometry_mm"][K])) > 0.01:
                bad.append(f"geometry {K}: sent {sent['geometry_mm'][K]} board {got['geometry_mm'][K]}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_file", type=Path)
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--only", default="trims,signs,ranges,geometry,gait")
    ap.add_argument("--dry-run", action="store_true", help="print the lines, send nothing")
    args = ap.parse_args()

    with open(args.json_file) as fh:
        cal = json.load(fh)
    only = set(args.only.split(","))
    lines = lines_for(cal, only)
    print(f"{len(lines)} settings from {args.json_file}  ({cal.get('firmware')})")
    if args.dry_run or not args.port:
        print("\n".join(lines))
        if not args.dry_run:
            sys.exit("--port required to send")
        return

    import serial
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    time.sleep(1.5)
    boot = ser.read(ser.in_waiting or 0).decode(errors="replace")
    m = re.search(r"unified firmware\s+v([\d.]+)", boot)
    if m:
        print("board booted:", m.group(0))
    ser.write(b"m\n"); time.sleep(0.3); ser.reset_input_buffer()
    ser.write(b"U\n")
    hello = read_until(ser, "ready", 3.0)
    if "ready" not in hello:
        sys.exit("board did not enter upload mode (firmware < v0.3?)\n" + hello[-600:])

    errors = 0
    for ln in lines:
        ser.write((ln + "\n").encode())
        reply = read_until(ser, "\n", 2.0).strip()
        if not reply.startswith("ok"):
            errors += 1
            print(f"  {ln!r:32} -> {reply!r}")
    ser.write(b"E\n")
    end = read_until(ser, "]", 3.0)
    print([ln for ln in end.splitlines() if "ok end" in ln][-1] if "ok end" in end else end.strip()[-200:])
    time.sleep(0.3)

    got = export_json(ser)
    ser.close()
    bad = diff(cal, got, only)
    out = args.json_file.with_name(args.json_file.stem + "_readback.json")
    out.write_text(json.dumps(got, indent=2) + "\n")
    print(f"readback saved to {out}")
    if errors or bad:
        print(f"FAIL: {errors} rejected lines, {len(bad)} mismatches")
        print("\n".join(bad))
        sys.exit(1)
    print(f"PASS: board matches {args.json_file.name} ({', '.join(sorted(only))})")


if __name__ == "__main__":
    main()
