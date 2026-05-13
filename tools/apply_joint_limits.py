"""apply_joint_limits.py — read joint_limits.json and patch the
<joint range="..."> entries (and matching <position ctrlrange="...">
actuator entries) in all three MJCFs.

Idempotent: rerunning produces the same XML. Backs up each MJCF to
<name>.bak.xml before the first overwrite (won't overwrite an existing
backup).

Usage
-----
    .venv/Scripts/python.exe apply_joint_limits.py
    .venv/Scripts/python.exe apply_joint_limits.py --json other.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DEFAULT_JSON = ROOT / "joint_limits.json"
DEFAULT_MJCFS = [
    MODEL_DIR / "phantomx_simple.xml",
    MODEL_DIR / "phantomx.xml",
    MODEL_DIR / "phantomx_simple_mjx.xml",
]

# Map "coxa_joint_RR" → actuator name "coxa_RR".
def _actuator_name(joint_name: str) -> str:
    parts = joint_name.split("_")
    # joint_name looks like "<type>_joint_<LEG>"
    return f"{parts[0]}_{parts[2]}"


def _format_range(lo: float, hi: float) -> str:
    """Format with 4 decimal places (rad). Strip needless trailing zeros
    only if doing so still looks tidy."""
    return f"{lo:.4f} {hi:.4f}"


def patch_xml_text(xml: str, limits: dict) -> str:
    """Update both <joint name="..." range="..."> attributes on joint
    elements and <position joint="..." ctrlrange="..."> attributes on
    actuator elements.
    """
    new_xml = xml
    for jn, info in limits.items():
        lo, hi = info["buffered"]
        rng_str = _format_range(lo, hi)

        # 1) Joint range. Match <joint name="<jn>" ... range="x y" ... />
        # The element may have other attributes between name= and range=.
        joint_pat = re.compile(
            r'(<joint\b[^>]*\bname="' + re.escape(jn) +
            r'"[^>]*\brange=")[^"]*(")',
            re.DOTALL,
        )
        new_xml, n1 = joint_pat.subn(r"\g<1>" + rng_str + r"\g<2>", new_xml)

        # 2) Actuator ctrlrange. Match by joint= attribute (more robust
        # across files than name=). Both phantomx_simple_mjx.xml and the
        # others use joint="<jn>" on their <position> actuators.
        act_pat = re.compile(
            r'(<position\b[^>]*\bjoint="' + re.escape(jn) +
            r'"[^>]*\bctrlrange=")[^"]*(")',
            re.DOTALL,
        )
        new_xml, n2 = act_pat.subn(r"\g<1>" + rng_str + r"\g<2>", new_xml)

        if n1 == 0:
            # Joint not present in this file (shouldn't happen for our
            # three MJCFs but stays informative if it does).
            print(f"  [warn] joint {jn} not patched")

    return new_xml


def patch_file(mjcf_path: Path, limits: dict) -> bool:
    """Patch a single MJCF in-place. Creates a .bak.xml on first run."""
    txt = mjcf_path.read_text(encoding="utf-8")
    new_txt = patch_xml_text(txt, limits)
    if new_txt == txt:
        print(f"  {mjcf_path.name}: no changes")
        return False

    # Backup once.
    bak = mjcf_path.with_suffix(".bak.xml")
    if not bak.exists():
        shutil.copyfile(mjcf_path, bak)
        print(f"  backup: {bak.name}")
    mjcf_path.write_text(new_txt, encoding="utf-8")
    print(f"  patched: {mjcf_path.name}")
    return True


def apply_to_all(json_path: Path, mjcfs: list[Path] | None = None) -> None:
    if mjcfs is None:
        mjcfs = DEFAULT_MJCFS
    with open(json_path) as f:
        limits = json.load(f)
    print(f"applying limits from {json_path.name} → {len(mjcfs)} MJCFs")
    for p in mjcfs:
        patch_file(p, limits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument(
        "--mjcfs", nargs="*", default=[str(p) for p in DEFAULT_MJCFS],
        help="MJCFs to patch (defaults to the three in models/).",
    )
    args = parser.parse_args()
    apply_to_all(Path(args.json), [Path(p) for p in args.mjcfs])


if __name__ == "__main__":
    main()
