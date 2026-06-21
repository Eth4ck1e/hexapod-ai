"""
scripts/controller_mapping.py — pure-function mapping from raw controller
inputs to the 9-vec cmd vector.

The same mapping is intended to ship to the ESP32-S3 firmware (translated
to C/C++) so the bot responds to the controller identically in sim and
on hardware. Keeping this isolated from the sim viewer means we can test
the convention here, then port without ambiguity.

Convention (per user spec, 2026-05-08):
  L stick X/Y  -> (vy, vx)            omnidirectional translation
  R stick X/Y  -> (pitch, roll)       NOT YET TRAINED — slot kept for future
  L trigger    -> wz +ve              turn LEFT (CCW)  proportional to pressure
  R trigger    -> wz -ve              turn RIGHT (CW)  proportional to pressure
  D-pad up     -> step height_idx -1  body raised (cmd[5] more negative)
  D-pad down   -> step height_idx +1  body lowered
  D-pad left   -> step width_idx -1   stance narrower
  D-pad right  -> step width_idx +1   stance wider
  Start        -> servo kill toggle   (cmd zeroed when killed)
  A/B/X/Y      -> mode switching      unassigned for now

Discrete D-pad levels: 5 each for height + width. Pressing the D-pad
edge-triggers a step; holding the D-pad does NOT continuously cycle.
"""
from __future__ import annotations

import numpy as np

# --- Discrete level tables (match envs/hexapod_env_jax.py cmd_sample_ranges) ---
# Negative cmd[5] = body raised; positive = lowered.
# Negative cmd[6] = stance narrower; positive = wider.
HEIGHT_LEVELS_M = [-0.020, -0.010, 0.0, +0.010, +0.020]   # 5 levels (mm-rounded)
WIDTH_LEVELS_M  = [-0.015, -0.0075, 0.0, +0.0075, +0.015] # 5 levels
DEFAULT_HEIGHT_IDX = 2   # neutral
DEFAULT_WIDTH_IDX  = 2

# --- Stick deadband + scale ---
STICK_DEADBAND   = 0.10    # raw stick value below this is treated as zero
TRIGGER_DEADBAND = 0.05    # trigger value below this is treated as zero
SPEED_SCALE      = 0.85    # full stick = trained MAX_SPEED. Going above 0.85
                           # is OOD — policy was trained on cmds in [0.40, 0.85]
                           # × MAX_SPEED so anything above 0.85 actually walks
                           # SLOWER (poor tracking out of distribution).
YAW_SCALE        = 0.85    # same for triggers vs MAX_YAW_RATE


def _apply_deadband(v: float, threshold: float) -> float:
    """Squash values inside the deadband to 0; rescale outside so output
    spans full ±1 from threshold to 1."""
    if abs(v) < threshold:
        return 0.0
    sign = 1.0 if v > 0 else -1.0
    rescaled = (abs(v) - threshold) / (1.0 - threshold)
    return sign * rescaled


class ControllerState:
    """Mutable state that the firmware (or sim) carries between input ticks.
    Tracks discrete D-pad levels and the kill toggle.
    """
    def __init__(self):
        self.height_idx: int = DEFAULT_HEIGHT_IDX
        self.width_idx:  int = DEFAULT_WIDTH_IDX
        self.killed:     bool = False
        # Edge-detection state — only step on D-pad transitions, not while held.
        self._last_dpad_x: int = 0
        self._last_dpad_y: int = 0
        self._last_start:  bool = False

    def step_dpad(self, dpad_x: int, dpad_y: int) -> None:
        """Edge-detect D-pad inputs (-1/0/+1 per axis) → increment levels.
        D-pad UP raises body → height_idx decreases (more negative cmd[5]).
        D-pad RIGHT widens stance → width_idx increases.
        """
        # Vertical: D-pad up = +1 (pygame convention) → raise body
        if dpad_y == 1 and self._last_dpad_y != 1:
            self.height_idx = max(0, self.height_idx - 1)
        elif dpad_y == -1 and self._last_dpad_y != -1:
            self.height_idx = min(len(HEIGHT_LEVELS_M) - 1, self.height_idx + 1)
        # Horizontal: D-pad right = +1 → wider stance
        if dpad_x == 1 and self._last_dpad_x != 1:
            self.width_idx = min(len(WIDTH_LEVELS_M) - 1, self.width_idx + 1)
        elif dpad_x == -1 and self._last_dpad_x != -1:
            self.width_idx = max(0, self.width_idx - 1)
        self._last_dpad_x = dpad_x
        self._last_dpad_y = dpad_y

    def step_start(self, start_pressed: bool) -> None:
        """Edge-detect Start button → toggle servo-kill state."""
        if start_pressed and not self._last_start:
            self.killed = not self.killed
        self._last_start = start_pressed

    @property
    def height_delta(self) -> float:
        return HEIGHT_LEVELS_M[self.height_idx]

    @property
    def width_delta(self) -> float:
        return WIDTH_LEVELS_M[self.width_idx]


def _polar_stick(stick_x: float, stick_y: float,
                  deadband: float = STICK_DEADBAND) -> tuple[float, float]:
    """Convert a (lx, ly) stick reading into polar (magnitude, direction)
    where direction is a unit vector (norm_x, norm_y). This is the right
    way to treat sticks: the user pushes in some direction with some
    magnitude. Per-axis deadband is wrong because diagonal pushes give
    sqrt(2)·max instead of max, and slightly-off-axis pushes get clipped
    on the off-axis component.

    Returns:
      (mag, norm_x, norm_y)
      mag is in [0, 1] — radial deadband applied, clamped to 1.
      (norm_x, norm_y) is a unit vector pointing in the stick's direction.
      If mag is 0 (deadband), (norm_x, norm_y) = (0, 0).
    """
    mag_raw = (stick_x * stick_x + stick_y * stick_y) ** 0.5
    if mag_raw < deadband:
        return 0.0, 0.0, 0.0
    # Direction unit vector (always defined since mag_raw > deadband > 0).
    nx = stick_x / mag_raw
    ny = stick_y / mag_raw
    # Rescale magnitude so the deadband region doesn't waste stick range:
    #   mag_raw = deadband -> mag_eff = 0
    #   mag_raw = 1.0       -> mag_eff = 1
    #   mag_raw > 1.0       -> mag_eff = 1 (clamped — corner deflection)
    mag_eff = (min(1.0, mag_raw) - deadband) / (1.0 - deadband)
    mag_eff = max(0.0, min(1.0, mag_eff))
    return mag_eff, nx, ny


def build_cmd(state: ControllerState,
              lstick_x: float, lstick_y: float,
              rstick_x: float, rstick_y: float,
              ltrigger: float, rtrigger: float,
              max_speed: float, max_yaw_rate: float) -> np.ndarray:
    """Read raw stick / trigger values + discrete D-pad state → 9-vec cmd.

    Sticks are treated as POLAR (magnitude + direction), not per-axis.
    A stick pushed at any angle with full deflection produces full
    SPEED_SCALE × MAX_SPEED magnitude in that direction. This matches
    how training cmds were sampled (magnitude × (cos θ, sin θ)).

    Triggers are in pygame's natural ranges (-1..+1 or 0..1, depending
    on driver). Rescaled to 0..1 for proportional pressure mapping.
    """
    # L stick → (vx, vy) via polar
    l_mag, l_nx, l_ny = _polar_stick(lstick_x, lstick_y, STICK_DEADBAND)

    # Triggers: pygame may report -1..+1 OR 0..1. Normalize to 0..1.
    lt = ltrigger
    rt = rtrigger
    if lt < 0.0: lt = (lt + 1.0) / 2.0
    if rt < 0.0: rt = (rt + 1.0) / 2.0
    lt = _apply_deadband(lt, TRIGGER_DEADBAND)
    rt = _apply_deadband(rt, TRIGGER_DEADBAND)
    lt = max(0.0, min(1.0, lt))
    rt = max(0.0, min(1.0, rt))

    # Map polar stick → body-frame velocities.
    # Body convention: vx +ve = forward, vy +ve = LEFT.
    # Pygame stick: X +ve = RIGHT, Y +ve = DOWN (inverted from intuition).
    # Direction: stick UP/forward (ny=-1) maps to body +X (vx > 0).
    # So vx = -ny · mag, vy = -nx · mag.
    body_speed = max_speed * SPEED_SCALE * l_mag
    vx = -l_ny * body_speed
    vy = -l_nx * body_speed

    # Yaw rate: LT pressed → wz +ve (turn left/CCW). RT pressed → wz -ve.
    wz = (lt - rt) * max_yaw_rate * YAW_SCALE

    # Pitch / roll: not trained yet; populate but env's cmd_mask will zero them.
    pitch = 0.0   # placeholder for R stick X (when re-enabled)
    roll  = 0.0   # placeholder for R stick Y (when re-enabled)

    cmd = np.zeros(9, dtype=np.float32)
    if not state.killed:
        cmd[0] = vx
        cmd[1] = vy
        cmd[2] = wz
        cmd[3] = pitch
        cmd[4] = roll
        cmd[5] = state.height_delta
        cmd[6] = state.width_delta
    return cmd


# ----------------------------------------------------------------------------
# Default pygame index conventions for XInput-mode controllers. These are
# fallbacks if no calibration JSON is present. In practice every controller
# differs slightly so the calibrate_controller.py script is the source of
# truth for production use.
# ----------------------------------------------------------------------------
DEFAULT_CALIBRATION = {
    "joystick_name": "<defaults>",
    "axes": {
        "lstick_x": {"index": 0, "sign": 1.0},
        "lstick_y": {"index": 1, "sign": 1.0},
        "rstick_x": {"index": 2, "sign": 1.0},
        "rstick_y": {"index": 3, "sign": 1.0},
        "ltrigger": {"index": 4, "sign": 1.0},
        "rtrigger": {"index": 5, "sign": 1.0},
    },
    "buttons": {
        "a": 0, "b": 1, "x": 2, "y": 3,
        "back": 6, "start": 7,
    },
    "hats": {"dpad": 0},
}


def load_calibration(path) -> dict:
    """Load a calibration JSON from disk. Returns the parsed dict.
    Raises FileNotFoundError if the file is missing — callers should
    catch that and fall back to DEFAULT_CALIBRATION.
    """
    import json
    from pathlib import Path
    path = Path(path)
    with open(path, "r") as f:
        cal = json.load(f)
    # Light validation
    for key in ("axes", "buttons"):
        if key not in cal:
            raise ValueError(f"calibration missing '{key}' section")
    return cal


def read_joystick_via_calibration(joystick, cal: dict):
    """Pull (lx, ly, rx, ry, lt, rt, hat_xy, start_pressed) from the joystick
    using the indices + signs in `cal`. Returns a tuple of all values.
    Handles the case where D-pad is buttons (cal has dpad_up/down/left/right
    entries instead of hats.dpad).
    """
    a = cal.get("axes", {})
    b = cal.get("buttons", {})
    h = cal.get("hats", {})

    def get_axis(slot, default=0.0):
        e = a.get(slot)
        if e is None: return default
        return joystick.get_axis(e["index"]) * e.get("sign", 1.0)

    def get_btn(slot, default=False):
        i = b.get(slot)
        if i is None or i < 0: return default
        return bool(joystick.get_button(i))

    lx = get_axis("lstick_x")
    ly = get_axis("lstick_y")
    rx = get_axis("rstick_x")
    ry = get_axis("rstick_y")
    # Triggers: prefer analog axis, fall back to digital button (0 or 1).
    if "ltrigger" in a:
        lt = get_axis("ltrigger")
    elif "ltrigger" in b:
        lt = 1.0 if get_btn("ltrigger") else 0.0
    else:
        lt = 0.0
    if "rtrigger" in a:
        rt = get_axis("rtrigger")
    elif "rtrigger" in b:
        rt = 1.0 if get_btn("rtrigger") else 0.0
    else:
        rt = 0.0

    # D-pad: hat first, button-fallback second.
    hat_idx = h.get("dpad")
    if hat_idx is not None and joystick.get_numhats() > hat_idx:
        hat_xy = joystick.get_hat(hat_idx)
    else:
        # Synthesize hat from 4 D-pad buttons.
        hx = (1 if get_btn("dpad_right") else 0) - (1 if get_btn("dpad_left") else 0)
        hy = (1 if get_btn("dpad_up")    else 0) - (1 if get_btn("dpad_down") else 0)
        hat_xy = (hx, hy)

    start_pressed = get_btn("start")
    return lx, ly, rx, ry, lt, rt, hat_xy, start_pressed
