"""
viz/decimate_meshes.py — one-time decimation of PhantomX STL meshes.

The original meshes have ~50,000 faces each (~250k total across the bot).
Manim renders polygons individually on CPU, so the full meshes are too
heavy. Decimate to ~1500 faces each — still looks like the real bot at
1080p, but renders in reasonable time.

Run once:
    .venv\\Scripts\\python.exe viz/decimate_meshes.py

Outputs to viz/meshes_decimated/*.stl. Subsequent renders read from
there. Re-run only if you want to change the decimation target.
"""
from __future__ import annotations

from pathlib import Path

import trimesh


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "models" / "meshes" / "phantomx"
OUT_DIR = PROJECT_ROOT / "viz" / "meshes_decimated"
PART_NAMES = ["body", "coxa", "femur", "tibia"]
TARGET_FACES = {
    "body":  2000,   # body is most visually prominent
    "coxa":  600,
    "femur": 800,
    "tibia": 1200,   # tibia has detail and is long
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Decimating PhantomX meshes:")
    print(f"  src: {SRC_DIR}")
    print(f"  out: {OUT_DIR}")
    print()

    total_in = 0
    total_out = 0
    for name in PART_NAMES:
        src = SRC_DIR / f"{name}.stl"
        out = OUT_DIR / f"{name}.stl"
        target = TARGET_FACES[name]

        m = trimesh.load(src)
        n_before = len(m.faces)

        # Run quadric decimation via fast_simplification (trimesh wraps it).
        # trimesh.simplify_quadric_decimation signature is
        # (percent=None, face_count=None, aggression=None) — we pass face_count.
        m_dec = m.simplify_quadric_decimation(face_count=target)
        n_after = len(m_dec.faces)

        m_dec.export(str(out))

        total_in += n_before
        total_out += n_after
        ratio = 100.0 * n_after / n_before
        print(f"  {name:>6}: {n_before:>6,} -> {n_after:>5,} faces "
              f"({ratio:5.1f}%)  -> {out.name}")

    print()
    print(f"  TOTAL : {total_in:>6,} -> {total_out:>5,} faces "
          f"({100.0 * total_out / total_in:5.1f}%)")


if __name__ == "__main__":
    main()
