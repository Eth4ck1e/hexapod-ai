"""
viz/storyboard.py — single source of truth for the gait visualizer.

Defines the presentation as PURE DATA: a list of shots, each with a
camera state, a bot pose, and overlay events. Pure-Python dataclasses,
no rendering imports — both the Manim renderer (fast iteration with
primitive bot) and the Blender renderer (final mesh-quality output)
consume this same description.

When you want to change the presentation flow, edit storyboard.py.
Both renderers pick up the changes automatically. Authoring is once,
output is twice.

Convention:
- All times are in seconds.
- Angles are in DEGREES in this file (renderers convert as needed).
- Camera 'distance' is in scene-space units (~14 fits nicely for the
  current 6× bot scale).
- A shot has either a fixed camera or a (start → end) interpolation.
- bot_pose is a label that maps to a pose-spec function in pose_specs.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------------
# Camera state
# ----------------------------------------------------------------------------
@dataclass
class CameraState:
    phi:      float          # vertical angle (degrees, 0 = top-down, 90 = side)
    theta:    float          # horizontal angle (degrees, -90 = looking +Y)
    distance: float          # camera distance from focus (scene units)
    focus:    tuple[float, float, float] = (0.0, 0.0, 0.0)


# ----------------------------------------------------------------------------
# Bot pose (named pose or explicit cmd+t)
# ----------------------------------------------------------------------------
@dataclass
class BotPose:
    label: str = "neutral"   # "neutral" | "gait" | <custom>
    cmd:   tuple[float, ...] = (0.0,) * 9   # used when label="gait"
    t:     float = 0.0                       # gait time used when label="gait"


# ----------------------------------------------------------------------------
# Overlay events — things that appear/animate during a shot
# ----------------------------------------------------------------------------
@dataclass
class Overlay:
    type:      str            # "body_axes" | "axis_labels" | "grid" | ...
    appear_t:  float = 0.0    # offset within the shot (seconds)
    duration:  float = 1.0    # how long the appear animation takes
    config:    dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Shot — one continuous beat of the presentation
# ----------------------------------------------------------------------------
@dataclass
class Shot:
    label:         str
    duration:      float
    camera_start:  CameraState
    camera_end:    Optional[CameraState] = None    # None = static camera
    camera_ease:   str = "smooth"                  # "smooth" | "linear" | "ease_in_out_sine"
    bot_pose:      BotPose = field(default_factory=BotPose)
    overlays:      list[Overlay] = field(default_factory=list)
    narration:     str = ""                        # what the speaker says (for reference)


# ----------------------------------------------------------------------------
# Section 1: introduce the bot in its body frame
# ----------------------------------------------------------------------------
SECTION_1: list[Shot] = [
    Shot(
        label="1.1_cold_open",
        duration=3.7,
        camera_start=CameraState(phi=80, theta=-90, distance=22),
        camera_end=  CameraState(phi=70, theta=-60, distance=14),
        camera_ease="smooth",
        bot_pose=BotPose(label="neutral"),
        overlays=[],
        narration="The PhantomX hexapod, our subject for this walkthrough. "
                  "Six legs, eighteen servos, designed for stable locomotion "
                  "on uneven terrain. Today we'll trace how its gait was "
                  "built up from first principles.",
    ),
    Shot(
        label="1.2_body_frame",
        duration=1.8,
        camera_start=CameraState(phi=70, theta=-60, distance=14),
        bot_pose=BotPose(label="neutral"),
        overlays=[
            Overlay(type="body_axes",   appear_t=0.0, duration=1.0),
            Overlay(type="axis_labels", appear_t=1.0, duration=0.8),
        ],
        narration="Everything starts with the body frame: X forward, Y to "
                  "the left, Z up. All gait math is expressed in this frame.",
    ),
    Shot(
        label="1.3_grid",
        duration=2.7,
        camera_start=CameraState(phi=70, theta=-60, distance=14),
        bot_pose=BotPose(label="neutral"),
        overlays=[
            Overlay(type="grid", appear_t=0.0, duration=1.5,
                    config={"x_range": (-8, 8, 1), "y_range": (-8, 8, 1)}),
        ],
        narration="A grid in the walking plane gives us a stable spatial "
                  "reference to anchor everything that follows.",
    ),
]


# Future: SECTION_2 (neutral pose definition), SECTION_3 (base path), etc.

ALL_SECTIONS: dict[str, list[Shot]] = {
    "section_1": SECTION_1,
}


def section_total_duration(section: list[Shot]) -> float:
    return sum(s.duration for s in section)
