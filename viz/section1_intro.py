"""
viz/section1_intro.py — Section 1: introduce the bot in its body frame.

Scenes (build order):
    Section1Shot1_ColdOpen     — bot rests on dark floor, slow camera push
    Section1Shot2_BodyFrame    — origin axes appear with X/Y/Z labels
    Section1Shot3_Grid         — 3D grid overlay centered on body origin
    Section1                   — full sequence (shots 1-3 chained)

Render commands:
    .venv\\Scripts\\manim.exe -ql viz/section1_intro.py Section1
        # quick preview (480p, fast render)

    .venv\\Scripts\\manim.exe -qh viz/section1_intro.py Section1
        # 1080p, slower

    .venv\\Scripts\\manim.exe -pql viz/section1_intro.py Section1
        # render + auto-play

Outputs land in media/videos/section1_intro/<quality>/Section1.mp4
"""
from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene, Scene,
    ThreeDAxes, NumberPlane, Line3D, Dot3D, Text, Tex,
    config,
    BLACK, WHITE, BLUE_E, BLUE_A, RED_E, GREEN_E, YELLOW_E, GREY, GREY_D,
    DEGREES, PI, ORIGIN, UP, DOWN, LEFT, RIGHT, IN, OUT,
    FadeIn, FadeOut, Create, Write,
    rate_functions,
)

# Pure-black background, like the user requested.
config.background_color = BLACK

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bot_primitives import make_hexapod, foot_origins_world


SCENE_SCALE = 6.0   # bot real-scale (meters) is multiplied by this for screen presence


class Section1Shot1_ColdOpen(ThreeDScene):
    """Shot 1: bot at rest on a dark floor. Slow camera push-in.

    No annotations yet — pure 'here is the subject' beat. ~3 seconds.
    """

    def construct(self):
        # Camera starts wide and far, pushes in to a 3/4 hero angle.
        self.set_camera_orientation(
            phi=80 * DEGREES, theta=-90 * DEGREES, distance=22, focal_distance=40,
        )

        bot = make_hexapod(scale=SCENE_SCALE)
        bot.shift(np.array([0, 0, 0]))

        # Fade the bot in from the dark.
        self.play(FadeIn(bot, run_time=1.2))

        # Camera push-in to 3/4 angle while orbiting slightly.
        self.move_camera(
            phi=70 * DEGREES, theta=-60 * DEGREES, distance=14,
            run_time=2.5, rate_func=rate_functions.smooth,
        )

        self.wait(0.5)


class Section1Shot2_BodyFrame(ThreeDScene):
    """Shot 2: continuing from Shot 1. Body-frame axes appear at the bot's
    origin with X/Y/Z labels. Audience now sees where 'forward', 'left',
    and 'up' are defined.
    """

    def construct(self):
        self.set_camera_orientation(
            phi=70 * DEGREES, theta=-60 * DEGREES, distance=14, focal_distance=40,
        )

        bot = make_hexapod(scale=SCENE_SCALE)
        self.add(bot)

        # Body-frame axes — short, sit at the bot's origin.
        axis_len = 0.18 * SCENE_SCALE
        x_axis = Line3D(start=ORIGIN, end=RIGHT * axis_len, color=RED_E, thickness=0.025)
        y_axis = Line3D(start=ORIGIN, end=UP * axis_len, color=GREEN_E, thickness=0.025)
        z_axis = Line3D(start=ORIGIN, end=OUT * axis_len, color=BLUE_E, thickness=0.025)

        # 3D-positioned text labels at the axis tips.
        x_label = Text("X (forward)", font_size=28, color=RED_E)
        y_label = Text("Y (left)",    font_size=28, color=GREEN_E)
        z_label = Text("Z (up)",      font_size=28, color=BLUE_E)
        x_label.rotate(90 * DEGREES, axis=RIGHT).move_to(RIGHT * (axis_len + 0.5) + OUT * 0.3)
        y_label.rotate(90 * DEGREES, axis=RIGHT).move_to(UP    * (axis_len + 0.5) + OUT * 0.3)
        z_label.rotate(90 * DEGREES, axis=RIGHT).move_to(OUT   * (axis_len + 0.5) + RIGHT * 0.3)

        self.play(
            Create(x_axis), Create(y_axis), Create(z_axis),
            run_time=1.0,
        )
        self.play(
            FadeIn(x_label), FadeIn(y_label), FadeIn(z_label),
            run_time=0.8,
        )
        self.wait(1.0)


class Section1Shot3_Grid(ThreeDScene):
    """Shot 3: 3D grid plane appears at the body origin's z-level (the
    walking plane), giving the audience spatial reference. Grid is subtle
    — light blue, low opacity — so it doesn't fight the bot for attention.
    """

    def construct(self):
        self.set_camera_orientation(
            phi=70 * DEGREES, theta=-60 * DEGREES, distance=14, focal_distance=40,
        )

        bot = make_hexapod(scale=SCENE_SCALE)
        self.add(bot)

        # Grid in the XY plane at z=0 (body origin level). Manim's NumberPlane
        # is 2D but rendered into 3D space at z=0 by default.
        grid = NumberPlane(
            x_range=(-8, 8, 1),
            y_range=(-8, 8, 1),
            background_line_style={
                "stroke_color": BLUE_A,
                "stroke_opacity": 0.25,
                "stroke_width": 1,
            },
            faded_line_ratio=2,
        )
        # Push grid slightly down so it sits at the floor plane (the bot's
        # feet, which are below the body origin).
        grid.shift(IN * 0.0)

        self.play(FadeIn(grid, run_time=1.5, rate_func=rate_functions.ease_in_out_sine))
        self.wait(1.2)


class Section1(ThreeDScene):
    """Full Section 1: shots 1-3 chained into one continuous beat.

    Camera flow:
      0.0-3.7s  : cold open + push-in (Shot 1)
      3.7-5.5s  : body-frame axes appear  (Shot 2)
      5.5-8.2s  : grid fades in           (Shot 3)
    """

    def construct(self):
        # ----- Shot 1: cold open -----
        self.set_camera_orientation(
            phi=80 * DEGREES, theta=-90 * DEGREES, distance=22, focal_distance=40,
        )

        bot = make_hexapod(scale=SCENE_SCALE)
        self.play(FadeIn(bot, run_time=1.2))
        self.move_camera(
            phi=70 * DEGREES, theta=-60 * DEGREES, distance=14,
            run_time=2.5, rate_func=rate_functions.smooth,
        )

        # ----- Shot 2: body-frame axes -----
        axis_len = 0.18 * SCENE_SCALE
        x_axis = Line3D(start=ORIGIN, end=RIGHT * axis_len, color=RED_E, thickness=0.025)
        y_axis = Line3D(start=ORIGIN, end=UP    * axis_len, color=GREEN_E, thickness=0.025)
        z_axis = Line3D(start=ORIGIN, end=OUT   * axis_len, color=BLUE_E, thickness=0.025)

        x_label = Text("X (forward)", font_size=28, color=RED_E)
        y_label = Text("Y (left)",    font_size=28, color=GREEN_E)
        z_label = Text("Z (up)",      font_size=28, color=BLUE_E)
        x_label.rotate(90 * DEGREES, axis=RIGHT).move_to(RIGHT * (axis_len + 0.5) + OUT * 0.3)
        y_label.rotate(90 * DEGREES, axis=RIGHT).move_to(UP    * (axis_len + 0.5) + OUT * 0.3)
        z_label.rotate(90 * DEGREES, axis=RIGHT).move_to(OUT   * (axis_len + 0.5) + RIGHT * 0.3)

        self.play(Create(x_axis), Create(y_axis), Create(z_axis), run_time=1.0)
        self.play(FadeIn(x_label), FadeIn(y_label), FadeIn(z_label), run_time=0.8)
        self.wait(0.8)

        # ----- Shot 3: 3D grid -----
        grid = NumberPlane(
            x_range=(-8, 8, 1),
            y_range=(-8, 8, 1),
            background_line_style={
                "stroke_color": BLUE_A,
                "stroke_opacity": 0.25,
                "stroke_width": 1,
            },
            faded_line_ratio=2,
        )
        self.play(FadeIn(grid, run_time=1.5, rate_func=rate_functions.ease_in_out_sine))
        self.wait(1.5)
