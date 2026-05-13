"""Quick render to verify the STL → Polyhedron pipeline works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bot_mesh import make_hexapod_mesh

from manim import ThreeDScene, config, BLACK, DEGREES, FadeIn

config.background_color = BLACK


class MeshTest(ThreeDScene):
    def construct(self):
        # OpenGL camera doesn't support focal_distance; use phi/theta only.
        self.set_camera_orientation(
            phi=70 * DEGREES, theta=-60 * DEGREES,
        )
        bot = make_hexapod_mesh(scale=6.0)
        self.play(FadeIn(bot, run_time=1.5))
        self.wait(2.0)
