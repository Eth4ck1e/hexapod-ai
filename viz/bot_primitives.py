"""
viz/bot_primitives.py — stylized PhantomX hexapod built from Manim 3D
primitives, dimensioned from the actual gait controller's geometry.

Used by the educational Manim scenes in viz/. We do NOT load the MJCF
or render via mujoco here — Manim wants its own scene graph, and a
stylized primitive-built bot reads better in an educational visual
than a photo-real mesh anyway.

The geometry constants come from gait/controller.py so the visual
matches what the policy actually sees during training.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable so we can pull constants from gait.controller.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from manim import (
    VGroup, Prism, Cube, Cylinder, Sphere, Dot3D,
    BLUE_E, GREY_B, GREY_D, GOLD_E, WHITE,
)

# Reach into the gait controller for canonical bot dimensions.
from gait.controller import Controller as _Controller, LEG_NAMES

# Build a one-off Controller against the simple MJX model to read constants.
# The Controller doesn't run physics; instantiation is cheap.
_ctrl = _Controller(str(_PROJECT_ROOT / "models" / "phantomx_simple_mjx.xml"))

LEG_ORIGIN_BODY = np.asarray(_ctrl.LEG_ORIGIN_BODY, dtype=np.float64)   # (6,3)
GAIT_NEUTRAL    = np.asarray(_ctrl.gait_neutral_pose, dtype=np.float64) # (18,)
DEFAULT_HEIGHT  = float(-LEG_ORIGIN_BODY[:, 2].mean())
COXA_LEN        = float(getattr(_ctrl, "COXA_LENGTH", 0.054))
FEMUR_LEN       = float(getattr(_ctrl, "FEMUR_LENGTH", 0.066))
TIBIA_LEN       = float(getattr(_ctrl, "TIBIA_LENGTH", 0.138))


def make_body(scale: float = 6.0) -> VGroup:
    """Stylized chassis. Manim units are in 'scene units' — we scale up
    the bot's real meter-scale geometry by `scale` for visibility.

    Uses Prism for non-cubic dimensions (Cube.scale(vector) silently
    produces degenerate geometry in Manim 0.20.x — confirmed bug).
    """
    body_l = 0.18 * scale
    body_w = 0.12 * scale
    body_h = 0.04 * scale
    body = Prism(dimensions=[body_l, body_w, body_h])
    body.set_color(GREY_D).set_fill(GREY_D, opacity=0.95)
    return VGroup(body)


def make_leg(origin_body: np.ndarray,
             gait_neutral_3: np.ndarray,
             scale: float = 6.0,
             color=GREY_B) -> VGroup:
    """One stylized leg: short coxa cylinder + femur + tibia.

    origin_body: (3,) leg coxa origin in body frame
    gait_neutral_3: (3,) [coxa, femur, tibia] joint angles at neutral
    """
    origin = origin_body * scale
    coxa_a, femur_a, tibia_a = gait_neutral_3

    # We'll just draw the leg as 3 thin cylinders. For a stylized teaching
    # visual we don't need the full per-joint axis math — neutral pose
    # extends radially from the body origin.
    radial = origin / max(np.linalg.norm(origin[:2]), 1e-6)
    radial[2] = 0.0
    radial = radial / max(np.linalg.norm(radial), 1e-6)

    # Coxa segment (horizontal, radially outward)
    coxa_end = origin + radial * COXA_LEN * scale
    coxa = _segment(origin, coxa_end, color=color, radius=0.012 * scale)

    # Femur segment (downward + outward — approximate with neutral pose)
    femur_end = coxa_end + np.array([radial[0] * FEMUR_LEN * scale * 0.5,
                                      radial[1] * FEMUR_LEN * scale * 0.5,
                                      -FEMUR_LEN * scale * 0.6])
    femur = _segment(coxa_end, femur_end, color=color, radius=0.011 * scale)

    # Tibia segment (mostly downward)
    tibia_end = femur_end + np.array([radial[0] * TIBIA_LEN * scale * 0.2,
                                       radial[1] * TIBIA_LEN * scale * 0.2,
                                       -TIBIA_LEN * scale * 0.85])
    tibia = _segment(femur_end, tibia_end, color=color, radius=0.010 * scale)

    foot = Sphere(radius=0.014 * scale).move_to(tibia_end)
    foot.set_color(GOLD_E).set_fill(GOLD_E, opacity=1.0)

    return VGroup(coxa, femur, tibia, foot)


def _segment(p0: np.ndarray, p1: np.ndarray, color, radius: float) -> Cylinder:
    """Cylinder from p0 to p1 with given radius. Manim's Cylinder defaults
    to z-axis-oriented; we orient by computing a direction vector."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    delta = p1 - p0
    height = float(np.linalg.norm(delta))
    if height < 1e-9:
        height = 1e-9
    direction = delta / height
    cyl = Cylinder(radius=radius, height=height,
                   direction=direction, resolution=(8, 16))
    cyl.move_to((p0 + p1) / 2)
    cyl.set_color(color).set_fill(color, opacity=0.85)
    return cyl


def make_hexapod(scale: float = 6.0,
                 body_color=GREY_D,
                 leg_color=GREY_B) -> VGroup:
    """Full stylized PhantomX. Returns one VGroup that you can rotate /
    translate / fade in as a whole."""
    body = make_body(scale=scale)
    legs = VGroup()
    for i, name in enumerate(LEG_NAMES):
        origin = LEG_ORIGIN_BODY[i]
        gait3 = GAIT_NEUTRAL[i*3:(i+1)*3]
        legs.add(make_leg(origin, gait3, scale=scale, color=leg_color))
    return VGroup(body, legs)


def foot_origins_world(scale: float = 6.0,
                       height: float | None = None) -> np.ndarray:
    """Return the 6 foot tip positions in scene-space coords (after bot
    is rendered with its body at world origin and resting at default
    stance height). Used to drop visual markers (path circles, dots)
    at each foot's nominal touchdown location.

    Each foot in body frame at neutral stance is at:
      (origin_xy + radial_xy * (COXA + FEMUR_horiz_extent + TIBIA_horiz_extent),
       -DEFAULT_HEIGHT)

    For section 1 we just use the analytical Controller's neutral foot
    positions if available; otherwise approximate from leg origins.
    """
    pts = []
    for i in range(6):
        origin = LEG_ORIGIN_BODY[i].copy()
        radial_xy = origin[:2] / max(np.linalg.norm(origin[:2]), 1e-6)
        # Approximate horizontal extent of each leg at neutral pose.
        horiz = (COXA_LEN
                 + FEMUR_LEN * 0.5
                 + TIBIA_LEN * 0.2)
        foot_xy = origin[:2] + radial_xy * horiz
        foot_z  = -DEFAULT_HEIGHT
        pts.append([foot_xy[0], foot_xy[1], foot_z])
    return np.asarray(pts) * scale
