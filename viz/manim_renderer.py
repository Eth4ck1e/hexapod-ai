"""
viz/manim_renderer.py — Manim renderer that consumes storyboard data.

Takes a list of Shot objects and renders them as Manim ThreeDScene
animations. Uses the primitive bot from bot_primitives.py — fast to
iterate, low fidelity. The Blender renderer (later) will consume the
same Shot list to produce mesh-quality output.

Render commands:
    .venv\\Scripts\\manim.exe -ql viz/manim_renderer.py Section1
        # 480p draft for fast iteration

    .venv\\Scripts\\manim.exe -qh viz/manim_renderer.py Section1
        # 1080p

The scene class name maps to the storyboard SECTION_<n> dictionary key.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from manim import (
    ThreeDScene, NumberPlane, Line3D, Text, config,
    BLACK, BLUE_E, BLUE_A, RED_E, GREEN_E,
    DEGREES, ORIGIN, UP, RIGHT, OUT,
    FadeIn, Create, rate_functions,
)

from bot_primitives import make_hexapod
from storyboard import (
    SECTION_1, ALL_SECTIONS,
    Shot, CameraState, BotPose, Overlay,
)


config.background_color = BLACK
SCENE_SCALE = 6.0


# ============================================================================
# Overlay handlers — one per overlay.type. Each returns (mobjects, animation)
# tuples that the renderer will play at the right time.
# ============================================================================
def _build_body_axes() -> list:
    axis_len = 0.18 * SCENE_SCALE
    return [
        Line3D(start=ORIGIN, end=RIGHT * axis_len, color=RED_E,   thickness=0.025),
        Line3D(start=ORIGIN, end=UP    * axis_len, color=GREEN_E, thickness=0.025),
        Line3D(start=ORIGIN, end=OUT   * axis_len, color=BLUE_E,  thickness=0.025),
    ]


def _build_axis_labels() -> list:
    axis_len = 0.18 * SCENE_SCALE
    x_label = Text("X (forward)", font_size=28, color=RED_E)
    y_label = Text("Y (left)",    font_size=28, color=GREEN_E)
    z_label = Text("Z (up)",      font_size=28, color=BLUE_E)
    x_label.rotate(90 * DEGREES, axis=RIGHT).move_to(RIGHT * (axis_len + 0.5) + OUT * 0.3)
    y_label.rotate(90 * DEGREES, axis=RIGHT).move_to(UP    * (axis_len + 0.5) + OUT * 0.3)
    z_label.rotate(90 * DEGREES, axis=RIGHT).move_to(OUT   * (axis_len + 0.5) + RIGHT * 0.3)
    return [x_label, y_label, z_label]


def _build_grid(cfg: dict) -> list:
    return [NumberPlane(
        x_range=cfg.get("x_range", (-8, 8, 1)),
        y_range=cfg.get("y_range", (-8, 8, 1)),
        background_line_style={
            "stroke_color":   BLUE_A,
            "stroke_opacity": 0.25,
            "stroke_width":   1,
        },
        faded_line_ratio=2,
    )]


_OVERLAY_BUILDERS = {
    "body_axes":   lambda cfg: _build_body_axes(),
    "axis_labels": lambda cfg: _build_axis_labels(),
    "grid":        _build_grid,
}


# ============================================================================
# Renderer base class — consumes a list[Shot] and plays them back.
# ============================================================================
class StoryboardScene(ThreeDScene):
    """Subclass and set `section_name` to the SECTION_<n> key in
    ALL_SECTIONS. The construct() method walks the shot list and
    plays each according to its camera/bot/overlay config."""
    section_name: str = ""

    def construct(self):
        if not self.section_name:
            raise ValueError(
                "Subclass StoryboardScene and set `section_name` to a key "
                "from storyboard.ALL_SECTIONS (e.g., 'section_1')."
            )
        shots = ALL_SECTIONS[self.section_name]

        # Bot lives across all shots (won't disappear between cuts).
        bot = make_hexapod(scale=SCENE_SCALE)

        # Initial camera + bot.
        first = shots[0]
        self.set_camera_orientation(
            phi=first.camera_start.phi   * DEGREES,
            theta=first.camera_start.theta * DEGREES,
            distance=first.camera_start.distance,
        )
        # Cold open: bot fades in during the first shot's first beat.
        self.play(FadeIn(bot, run_time=1.2))

        for i, shot in enumerate(shots):
            print(f"[manim_renderer] shot {i+1}/{len(shots)}: {shot.label}")
            self._play_shot(shot, is_first=(i == 0))

    def _play_shot(self, shot: Shot, is_first: bool):
        """Play one shot: optional camera move + queued overlay events."""
        # Camera animation if camera_end is set.
        camera_anim_time = 0.0
        if shot.camera_end is not None:
            ease_fn = {
                "smooth":            rate_functions.smooth,
                "linear":            rate_functions.linear,
                "ease_in_out_sine":  rate_functions.ease_in_out_sine,
            }.get(shot.camera_ease, rate_functions.smooth)
            # For the cold-open shot, the FadeIn took 1.2s already; spend
            # the remainder on the camera move.
            cam_time = max(0.5, shot.duration - (1.2 if is_first else 0.0))
            self.move_camera(
                phi=shot.camera_end.phi   * DEGREES,
                theta=shot.camera_end.theta * DEGREES,
                distance=shot.camera_end.distance,
                run_time=cam_time,
                rate_func=ease_fn,
            )
            camera_anim_time = cam_time

        # Overlay events. We sort by appear_t and play them in order,
        # waiting between events as needed.
        elapsed_in_shot = 1.2 if is_first else 0.0  # account for cold-open FadeIn
        elapsed_in_shot += camera_anim_time
        for ov in sorted(shot.overlays, key=lambda x: x.appear_t):
            wait_for = ov.appear_t - elapsed_in_shot
            if wait_for > 0.001:
                self.wait(wait_for)
                elapsed_in_shot += wait_for

            mobjects = _OVERLAY_BUILDERS[ov.type](ov.config)
            # Use Create for line-like things (axes), FadeIn for the rest.
            if ov.type == "body_axes":
                self.play(*[Create(m) for m in mobjects], run_time=ov.duration)
            else:
                self.play(*[FadeIn(m) for m in mobjects], run_time=ov.duration)
            elapsed_in_shot += ov.duration

        # Hold for the remainder of the shot.
        remaining = shot.duration - elapsed_in_shot
        if remaining > 0.001:
            self.wait(remaining)


# ============================================================================
# Concrete scene classes — one per section. Manim CLI invokes by class name.
# ============================================================================
class Section1(StoryboardScene):
    section_name = "section_1"
