"""
demo_phases.py — shared cmd-vector demo script + helpers used by both the
active JAX/MJX viewer (watch_demo_jax.py) and the legacy SB3 viewer
(legacy/sb3/watch_demo.py).

Lives under scripts/ so the active viewer's import doesn't cross the
active/legacy boundary. Legacy viewer imports from here too.

What's here:
  * `_cmd(**slots)`   — build a 9-vec cmd from named slot kwargs
  * `_walk(ctrl, heading_rad, speed_frac=None)` — translation cmd factory
  * `_spin(ctrl, sign, scale=1.0)` — yaw cmd factory
  * `DEMO_PHASES`    — the 17-phase preset script the viewers cycle through
  * `latest_run_dir(root)` / `latest_checkpoint(run_dir)` — SB3-era
    checkpoint discovery helpers (legacy uses these; the JAX viewer has
    its own --lineage / --iter helpers)
"""
import glob
import math
import os
import re
import time

import numpy as np

DEG = math.radians


def _cmd(**slots):
    """Build a 9-vec cmd from named slots."""
    cmd = np.zeros(9, dtype=np.float32)
    NAMES = dict(vx=0, vy=1, wz=2, pitch=3, roll=4,
                 height=5, width=6, shift_x=7, shift_y=8)
    for k, v in slots.items():
        cmd[NAMES[k]] = v
    return cmd


# In-distribution range for the env's translation cmd. The policy was
# trained on speed_frac ∈ [SPEED_MIN_FRAC, SPEED_MAX_FRAC] = [0.40, 0.85].
# Cmds outside this band are OOD and produce garbage residuals.
TRAINED_SPEED_MIN_FRAC = 0.40
TRAINED_SPEED_MAX_FRAC = 0.85
DEFAULT_SPEED_FRAC     = 0.70   # mid-range default for hand-crafted phases


def _walk(ctrl, heading_rad, speed_frac=None):
    """Walk command at heading_rad (rad, 0 = body +X). speed_frac defaults to
    DEFAULT_SPEED_FRAC and is clamped to the trained-distribution band so the
    policy gets observations it actually saw at training time."""
    if speed_frac is None:
        speed_frac = DEFAULT_SPEED_FRAC
    speed_frac = max(TRAINED_SPEED_MIN_FRAC,
                     min(speed_frac, TRAINED_SPEED_MAX_FRAC))
    speed = ctrl.MAX_SPEED * speed_frac
    return _cmd(vx=speed * math.cos(heading_rad),
                vy=speed * math.sin(heading_rad))


def _spin(ctrl, sign, scale=1.0):
    return _cmd(wz=sign * ctrl.MAX_YAW_RATE * scale)


DEMO_PHASES = [
    # (label, duration_seconds, cmd_factory(t_in_phase, ctrl) -> 9-vec)
    # All WALK phases run at DEFAULT_SPEED_FRAC (0.70) — mid-range of the
    # trained [0.40, 0.85] band. Cmds stay in-distribution.
    ("WALK   forward (heading=0)",           5.0,
        lambda t, c: _walk(c, 0)),
    ("WALK   strafe left  (heading=+90)",    5.0,
        lambda t, c: _walk(c, math.pi/2)),
    ("WALK   backward (heading=180)",        5.0,
        lambda t, c: _walk(c, math.pi)),
    ("WALK   strafe right (heading=-90)",    5.0,
        lambda t, c: _walk(c, -math.pi/2)),
    ("WALK   heading sweep 0 to 360 deg",   16.0,
        lambda t, c: _walk(c, 2*math.pi*t/16)),
    ("WALK   slow (40% of MAX_SPEED)",       6.0,
        lambda t, c: _walk(c, 0, speed_frac=TRAINED_SPEED_MIN_FRAC)),
    ("WALK   fast (85% of MAX_SPEED)",       6.0,
        lambda t, c: _walk(c, 0, speed_frac=TRAINED_SPEED_MAX_FRAC)),
    ("WALK   forward speed sweep (40→85%)", 12.0,
        lambda t, c: _walk(c, 0, speed_frac=(
            TRAINED_SPEED_MIN_FRAC
            + 0.5 * (TRAINED_SPEED_MAX_FRAC - TRAINED_SPEED_MIN_FRAC)
              * (1 + math.sin(2*math.pi*t/12))))),
    ("WALK   forward + width overlay",       6.0,
        lambda t, c: _walk(c, 0) + _cmd(width=0.012 * math.sin(2*math.pi*t/3))),
    ("WALK   forward + height overlay",      6.0,
        lambda t, c: _walk(c, 0) + _cmd(height=0.015 * math.sin(2*math.pi*t/3))),
    ("WALK   forward + pitch wobble ±8°",    8.0,
        lambda t, c: _walk(c, 0) + _cmd(pitch=DEG(8) * math.sin(2*math.pi*t/3))),
    ("WALK   forward + roll wobble ±8°",     8.0,
        lambda t, c: _walk(c, 0) + _cmd(roll =DEG(8) * math.sin(2*math.pi*t/3))),
    # Diagnostic: static (held) tilt is easier for the policy to track than
    # a 3-Hz wobble. If the policy follows pitch+ but ignores roll+ in
    # these phases, BC didn't generalize across the roll axis.
    ("WALK   forward + STATIC pitch +8°",    4.0,
        lambda t, c: _walk(c, 0) + _cmd(pitch=DEG(+8))),
    ("WALK   forward + STATIC pitch -8°",    4.0,
        lambda t, c: _walk(c, 0) + _cmd(pitch=DEG(-8))),
    ("WALK   forward + STATIC roll  +8°",    4.0,
        lambda t, c: _walk(c, 0) + _cmd(roll =DEG(+8))),
    ("WALK   forward + STATIC roll  -8°",    4.0,
        lambda t, c: _walk(c, 0) + _cmd(roll =DEG(-8))),
    ("WALK   forward + pitch+roll circle ±8°", 8.0,
        lambda t, c: _walk(c, 0) + _cmd(pitch=DEG(8) * math.cos(2*math.pi*t/8),
                                        roll =DEG(8) * math.sin(2*math.pi*t/8))),
]


# ============================================================================
# Paper-aligned demo phases (v4+) — exercises (vx, vy, wz) only.
#
# v4 trains under cmd_mask="paper" which zeros pitch/roll/height/width/shifts.
# This demo exercises the full motion repertoire under that mask: pure
# translation in any direction, pure yaw, and combined motion (arc-aware
# turn-while-walking).
# ============================================================================
def _arc(ctrl, speed_frac, R_frac, sign_wz):
    """Walk forward at speed × speed_frac while turning at radius R such that
    R = MIN_TURN_RADIUS + R_frac × (MAX_TURN_RADIUS - MIN_TURN_RADIUS).
    R_frac=0 → tightest controllable turn; R_frac=1 → gentlest.
    sign_wz = +1 (left/CCW) or -1 (right/CW)."""
    speed_frac = max(TRAINED_SPEED_MIN_FRAC,
                     min(speed_frac, TRAINED_SPEED_MAX_FRAC))
    speed = ctrl.MAX_SPEED * speed_frac
    R = ctrl.MIN_TURN_RADIUS + R_frac * (ctrl.MAX_TURN_RADIUS - ctrl.MIN_TURN_RADIUS)
    wz = sign_wz * speed / R
    return _cmd(vx=speed, wz=wz)


# ============================================================================
# Interactive test sequences — keyboard-triggered in the viewer.
#
# Each entry is keyed by a single character (the key the user presses).
# Value tuple: (label, loop_duration_sec, cmd_fn)
#   cmd_fn(t_in_loop, ctrl) -> 9-vec cmd
# When the user presses the key, that sequence becomes active and loops
# indefinitely with t_in_loop wrapping. Press a different key to switch.
# ============================================================================
# Stance envelope constants come from the single source of truth in
# envs/stance_envelope.py — same numbers used by the env's cmd sampler
# during training, so watch tests automatically reflect any envelope
# change without manual sync.
from envs.stance_envelope import HEIGHT_PRESETS, safe_dw_range


def _height_cycle(t, c):
    """20s cycle: 4s at each of the 5 height presets, dw=0 throughout."""
    phase = int(t / 4.0) % len(HEIGHT_PRESETS)
    h = HEIGHT_PRESETS[phase]
    return _walk(c, 0) + _cmd(height=h)


def _width_cycle(t, c):
    """20s cycle at neutral height (dh=0): step through dw extremes.
    Min/max widths come from the dh-conditional envelope at dh=0."""
    min_dw, max_dw = safe_dw_range(0.0)
    phase = int(t / 4.0) % 5
    w = [0.0, max_dw, 0.0, min_dw, 0.0][phase]
    return _walk(c, 0) + _cmd(width=w)


def _stance_combined(t, c):
    """40s cycle: walk forward through corners — at each of the 5 height
    presets, alternate widest-allowed and narrowest-allowed for that height
    (4s per corner). Demonstrates the full dh-conditional envelope."""
    phase = int(t / 4.0) % (2 * len(HEIGHT_PRESETS))
    h = HEIGHT_PRESETS[phase // 2]
    min_dw, max_dw = safe_dw_range(h)
    w = max_dw if (phase % 2 == 0) else min_dw
    return _walk(c, 0) + _cmd(height=h, width=w)


def _arc_left(t, c):
    speed = 0.6 * c.MAX_SPEED
    R = 0.5 * (c.MIN_TURN_RADIUS + c.MAX_TURN_RADIUS)
    return _cmd(vx=speed, wz=+speed/R)


def _arc_right(t, c):
    speed = 0.6 * c.MAX_SPEED
    R = 0.5 * (c.MIN_TURN_RADIUS + c.MAX_TURN_RADIUS)
    return _cmd(vx=speed, wz=-speed/R)


def _diagonal_cycle(t, c):
    """8s cycle: forward-left → forward-right → back-right → back-left."""
    phase = int(t / 2.0) % 4
    angle = [math.pi/4, -math.pi/4, -3*math.pi/4, 3*math.pi/4][phase]
    return _walk(c, angle)


def _fwd_back_cycle(t, c):
    """8s loop: 4s forward, 4s backward."""
    return _walk(c, 0 if (int(t / 4.0) % 2 == 0) else math.pi)


def _strafe_cycle(t, c):
    """8s loop: 4s left, 4s right."""
    return _walk(c, math.pi/2 if (int(t / 4.0) % 2 == 0) else -math.pi/2)


def _spin_cycle(t, c):
    """8s loop: 4s spin left, 4s spin right."""
    sign = +1 if (int(t / 4.0) % 2 == 0) else -1
    return _spin(c, sign, 0.7)


def _arc_cycle(t, c):
    """8s loop: 4s arc left, 4s arc right (medium radius)."""
    speed = 0.6 * c.MAX_SPEED
    R = 0.5 * (c.MIN_TURN_RADIUS + c.MAX_TURN_RADIUS)
    sign = +1 if (int(t / 4.0) % 2 == 0) else -1
    return _cmd(vx=speed, wz=sign * speed / R)


INTERACTIVE_TESTS: dict = {
    # 9 tests, all bound to 1-9. Each auto-cycles through related sub-cases
    # so you don't need to switch keys to compare directions.
    "1": ("Forward / backward (alternates 4s)",       8.0,  _fwd_back_cycle),
    "2": ("Strafe left / right (alternates 4s)",      8.0,  _strafe_cycle),
    "3": ("Spin in place left / right (alternates)",  8.0,  _spin_cycle),
    "4": ("Arc turn left / right (alternates)",       8.0,  _arc_cycle),
    "5": ("Diagonal cycle (FL/FR/BR/BL, 2s each)",    8.0,  _diagonal_cycle),
    "6": ("Heading sweep 0..360 (omnidirectional)",   12.0,
        lambda t, c: _walk(c, 2*math.pi*t/12.0)),
    "7": ("Height cycle (-20 / 0 / +20 mm, 4s)",      16.0, _height_cycle),
    "8": ("Width cycle (+15 / 0 / -15 mm, 4s)",       16.0, _width_cycle),
    "9": ("Combined stance corners (h+w, 4s each)",   24.0, _stance_combined),
}


def print_interactive_help():
    print("\n=== INTERACTIVE TESTS — press a key in the viewer window ===")
    for k, (label, dur, _) in INTERACTIVE_TESTS.items():
        print(f"  [{k}]  {label:<40} (loop {dur:.0f}s)")
    print("  [r]  RESET bot to neutral pose (clears stuck legs)")
    print("  [p]  Print bot world position (debug)")
    print("  [h]  Print this help")
    print("=============================================================\n")


# Showcase demo for v8+ — designed for video recording / presentations.
# Covers the full motion repertoire with stance variation mixed in,
# rather than 12 forward-walks in a row. ~95s total.
DEMO_PHASES_SHOWCASE = [
    # MOTION DIRECTIONS — what the bot can do.
    ("WALK   forward",                                4.0,
        lambda t, c: _walk(c, 0)),
    ("WALK   backward",                               4.0,
        lambda t, c: _walk(c, math.pi)),
    ("WALK   strafe left",                            4.0,
        lambda t, c: _walk(c, math.pi/2)),
    ("WALK   strafe right",                           4.0,
        lambda t, c: _walk(c, -math.pi/2)),
    ("WALK   diagonal forward-left",                  3.5,
        lambda t, c: _walk(c, math.pi/4)),
    ("WALK   diagonal forward-right",                 3.5,
        lambda t, c: _walk(c, -math.pi/4)),
    ("WALK   diagonal backward-left",                 3.5,
        lambda t, c: _walk(c, 3*math.pi/4)),
    ("WALK   diagonal backward-right",                3.5,
        lambda t, c: _walk(c, -3*math.pi/4)),
    # PURE YAW — turn in place.
    ("SPIN   in place LEFT",                          4.0,
        lambda t, c: _spin(c, +1, 0.7)),
    ("SPIN   in place RIGHT",                         4.0,
        lambda t, c: _spin(c, -1, 0.7)),
    # ARC TURNS — combined translation + yaw.
    ("ARC    forward + left turn (tight)",            5.0,
        lambda t, c: _cmd(vx=0.6*c.MAX_SPEED, wz=+0.6*c.MAX_SPEED/c.MIN_TURN_RADIUS)),
    ("ARC    forward + right turn (tight)",           5.0,
        lambda t, c: _cmd(vx=0.6*c.MAX_SPEED, wz=-0.6*c.MAX_SPEED/c.MIN_TURN_RADIUS)),
    ("ARC    forward + left turn (gentle)",           5.0,
        lambda t, c: _cmd(vx=0.6*c.MAX_SPEED, wz=+0.6*c.MAX_SPEED/c.MAX_TURN_RADIUS)),
    # SMOOTH OMNIDIRECTIONAL — heading sweep while walking.
    ("WALK   heading sweep 0..360",                  10.0,
        lambda t, c: _walk(c, 2*math.pi*t/10)),
    # STANCE — bot held forward at constant velocity, stance changes.
    ("WALK   forward + body RAISED  (-20mm)",         4.0,
        lambda t, c: _walk(c, 0) + _cmd(height=-0.020)),
    ("WALK   forward + body LOWERED (+20mm)",         4.0,
        lambda t, c: _walk(c, 0) + _cmd(height=+0.020)),
    ("WALK   forward + stance WIDE  (+15mm/foot)",    4.0,
        lambda t, c: _walk(c, 0) + _cmd(width=+0.015)),
    ("WALK   forward + stance NARROW(-15mm/foot)",    4.0,
        lambda t, c: _walk(c, 0) + _cmd(width=-0.015)),
    # SMOOTH STANCE RAMPS — show continuous response.
    ("WALK   forward, height ramp -20 -> +20 mm",     7.0,
        lambda t, c: _walk(c, 0) + _cmd(height=-0.020 + 0.040 * (t/7.0))),
    ("WALK   forward, width ramp +15 -> -15 mm",      7.0,
        lambda t, c: _walk(c, 0) + _cmd(width=+0.015 - 0.030 * (t/7.0))),
    # FINAL — return to neutral.
    ("WALK   forward, neutral (cooldown)",            3.0,
        lambda t, c: _walk(c, 0)),
]


DEMO_PHASES_PAPER_STANCE = [
    # v6+/v7+ schedule: exercises (vx, vy, wz) AND (height, width).
    # Tests whether the policy actually responds to D-pad-style stance cmds.
    # Speed held steady at DEFAULT_SPEED_FRAC; cmd[5] (height) and cmd[6]
    # (width) overlay the underlying walk.
    ("WALK   forward, neutral stance",                4.0,
        lambda t, c: _walk(c, 0)),
    ("WALK   forward, body RAISED  (-20mm)",          5.0,
        lambda t, c: _walk(c, 0) + _cmd(height=-0.020)),
    ("WALK   forward, neutral",                       3.0,
        lambda t, c: _walk(c, 0)),
    ("WALK   forward, body LOWERED (+20mm)",          5.0,
        lambda t, c: _walk(c, 0) + _cmd(height=+0.020)),
    ("WALK   forward, neutral",                       3.0,
        lambda t, c: _walk(c, 0)),
    ("WALK   forward, stance WIDE  (+15mm/foot)",     5.0,
        lambda t, c: _walk(c, 0) + _cmd(width=+0.015)),
    ("WALK   forward, neutral",                       3.0,
        lambda t, c: _walk(c, 0)),
    ("WALK   forward, stance NARROW(-15mm/foot)",     5.0,
        lambda t, c: _walk(c, 0) + _cmd(width=-0.015)),
    ("WALK   forward, neutral",                       3.0,
        lambda t, c: _walk(c, 0)),
    # Combined corner cases
    ("WALK   forward + RAISED + WIDE",                5.0,
        lambda t, c: _walk(c, 0) + _cmd(height=-0.020, width=+0.015)),
    ("WALK   forward + LOWERED + NARROW",             5.0,
        lambda t, c: _walk(c, 0) + _cmd(height=+0.020, width=-0.015)),
    ("WALK   forward, neutral",                       3.0,
        lambda t, c: _walk(c, 0)),
    # Smooth height ramp
    ("WALK   forward, height ramp -20 -> +20 mm",    8.0,
        lambda t, c: _walk(c, 0) + _cmd(height=-0.020 + 0.040 * (t/8.0))),
    # Smooth width ramp
    ("WALK   forward, width ramp +15 -> -15 mm",     8.0,
        lambda t, c: _walk(c, 0) + _cmd(width=+0.015 - 0.030 * (t/8.0))),
    # Sanity: also re-cover the basic motions to confirm no regression
    ("WALK   strafe left  (heading=+90)",             4.0,
        lambda t, c: _walk(c, math.pi/2)),
    ("WALK   strafe right (heading=-90)",             4.0,
        lambda t, c: _walk(c, -math.pi/2)),
    ("SPIN   left in place",                          4.0,
        lambda t, c: _spin(c, +1, scale=0.7)),
    ("SPIN   right in place",                         4.0,
        lambda t, c: _spin(c, -1, scale=0.7)),
]


DEMO_PHASES_PAPER = [
    # Pure stand (sanity — policy should hold position)
    ("STAND  cmd=0",                                 3.0,
        lambda t, c: _cmd()),
    # Pure translation in 4 cardinal directions
    ("WALK   forward (heading=0)",                   5.0,
        lambda t, c: _walk(c, 0)),
    ("WALK   strafe left (heading=+90)",             5.0,
        lambda t, c: _walk(c, math.pi/2)),
    ("WALK   backward (heading=180)",                5.0,
        lambda t, c: _walk(c, math.pi)),
    ("WALK   strafe right (heading=-90)",            5.0,
        lambda t, c: _walk(c, -math.pi/2)),
    # Diagonals
    ("WALK   forward-left (45°)",                    4.0,
        lambda t, c: _walk(c, math.pi/4)),
    ("WALK   forward-right (-45°)",                  4.0,
        lambda t, c: _walk(c, -math.pi/4)),
    # Smooth omnidirectional sweep
    ("WALK   heading sweep 0 to 360 deg",            12.0,
        lambda t, c: _walk(c, 2*math.pi*t/12)),
    # Speed range
    ("WALK   slow forward (40%)",                    4.0,
        lambda t, c: _walk(c, 0, speed_frac=TRAINED_SPEED_MIN_FRAC)),
    ("WALK   fast forward (85%)",                    4.0,
        lambda t, c: _walk(c, 0, speed_frac=TRAINED_SPEED_MAX_FRAC)),
    # Pure yaw (turn in place)
    ("SPIN   left in place (+wz)",                   5.0,
        lambda t, c: _spin(c, +1, scale=0.7)),
    ("SPIN   right in place (-wz)",                  5.0,
        lambda t, c: _spin(c, -1, scale=0.7)),
    # Arc-aware turn-while-walking (combined vx + wz)
    ("ARC    forward + gentle left turn",            6.0,
        lambda t, c: _arc(c, speed_frac=0.6, R_frac=0.85, sign_wz=+1)),
    ("ARC    forward + medium left turn",            6.0,
        lambda t, c: _arc(c, speed_frac=0.6, R_frac=0.40, sign_wz=+1)),
    ("ARC    forward + tight left turn",             6.0,
        lambda t, c: _arc(c, speed_frac=0.6, R_frac=0.05, sign_wz=+1)),
    ("ARC    forward + tight right turn",            6.0,
        lambda t, c: _arc(c, speed_frac=0.6, R_frac=0.05, sign_wz=-1)),
    ("ARC    forward + medium right turn",           6.0,
        lambda t, c: _arc(c, speed_frac=0.6, R_frac=0.40, sign_wz=-1)),
    ("ARC    forward + gentle right turn",           6.0,
        lambda t, c: _arc(c, speed_frac=0.6, R_frac=0.85, sign_wz=-1)),
]


# ============================================================================
# Checkpoint discovery (legacy SB3-era)
# ============================================================================
def latest_run_dir(root="checkpoints"):
    if not os.path.isdir(root):
        return None
    runs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    if not runs:
        return None
    runs.sort(key=lambda d: max((os.path.getmtime(os.path.join(d, f))
                                 for f in os.listdir(d)), default=0))
    return runs[-1]


def latest_checkpoint(run_dir):
    """Newest *complete* SB3 checkpoint (skips files modified in last 3s)."""
    final = os.path.join(run_dir, "final.zip")
    if os.path.exists(final) and time.time() - os.path.getmtime(final) > 3.0:
        return final[:-4]
    step_re = re.compile(r"_(\d+)_steps\.zip$")
    candidates = []
    for f in os.listdir(run_dir):
        m = step_re.search(f)
        if not m:
            continue
        path = os.path.join(run_dir, f)
        if time.time() - os.path.getmtime(path) < 3.0:
            continue
        candidates.append((int(m.group(1)), path[:-4]))
    return None if not candidates else sorted(candidates)[-1][1]
