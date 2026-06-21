"""
scripts/calibrate_controller.py — interactive controller calibration.

Walks the user through pressing each control on their gamepad, records
which axis/button/hat each one corresponds to, and writes the result
to checkpoints/controller_calibration.json. watch_controller.py loads
this file at startup if present.

Run:
    $env:PYTHONPATH = "."
    .venv\\Scripts\\python.exe scripts\\calibrate_controller.py

The script auto-picks the first non-keypad/non-spacemouse joystick.
Override with --joystick-index N if needed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pygame

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "checkpoints" / "controller_calibration.json"


def _select_joystick(index: int | None) -> pygame.joystick.Joystick:
    pygame.init()
    pygame.joystick.init()
    n = pygame.joystick.get_count()
    if n == 0:
        print("ERROR: no joysticks detected. Connect the controller and re-run.")
        sys.exit(1)
    print(f"\nDetected {n} joystick(s):")
    for i in range(n):
        j = pygame.joystick.Joystick(i)
        print(f"  [{i}] {j.get_name()}  axes={j.get_numaxes()} "
              f"buttons={j.get_numbuttons()} hats={j.get_numhats()}")
    if index is not None:
        if index >= n:
            print(f"ERROR: --joystick-index {index} out of range")
            sys.exit(1)
        return pygame.joystick.Joystick(index)
    SKIP = ["azeron", "spacemouse", "3dconnexion", "keypad"]
    for i in range(n):
        j = pygame.joystick.Joystick(i)
        if any(s in j.get_name().lower() for s in SKIP):
            continue
        if j.get_numaxes() >= 4:
            print(f"\nAuto-selected: [{i}] {j.get_name()}")
            return j
    print(f"\nFalling back to: [0] {pygame.joystick.Joystick(0).get_name()}")
    return pygame.joystick.Joystick(0)


def _sample_axes(joystick: pygame.joystick.Joystick,
                 duration: float = 1.5) -> tuple[list[float], list[float]]:
    """Sample axes for `duration` seconds and return (min, max) per axis."""
    n = joystick.get_numaxes()
    mins  = [+1e9] * n
    maxs  = [-1e9] * n
    t0 = time.time()
    while time.time() - t0 < duration:
        pygame.event.pump()
        for i in range(n):
            v = joystick.get_axis(i)
            if v < mins[i]: mins[i] = v
            if v > maxs[i]: maxs[i] = v
        time.sleep(0.02)
    return mins, maxs


def _detect_axis_pushed_to(joystick: pygame.joystick.Joystick,
                           sign: int,
                           prompt: str,
                           min_delta: float = 0.3,
                           allow_button_fallback: bool = False) -> tuple:
    """Ask user to push a stick/trigger in a direction, sample axes, return
    (axis_index, sign) of the axis that moved most in that direction.

    With `allow_button_fallback=True`, also samples buttons during the push
    and returns ('button', button_index) if a button press registered AND
    no axis swing met threshold. Returns (axis_index, sign) for axis match.

    Prints all per-axis deltas so the user can see what's being sampled.
    """
    print(f"\n{prompt}")
    input("  Press Enter when ready, then push for ~1.5s...")

    # Sample a brief baseline (with control at rest).
    rest_mins, rest_maxs = _sample_axes(joystick, duration=0.3)
    rest = [(rest_mins[i] + rest_maxs[i]) / 2 for i in range(len(rest_mins))]

    # Capture initial button states for fallback detection.
    pygame.event.pump()
    n_buttons = joystick.get_numbuttons()
    rest_buttons = [joystick.get_button(i) for i in range(n_buttons)]

    # Now sample with the user pushing — also tracking buttons.
    print("  ... sampling 1.5s — push NOW ...")
    n_axes = joystick.get_numaxes()
    mins  = [+1e9] * n_axes
    maxs  = [-1e9] * n_axes
    button_pressed = [False] * n_buttons
    t0 = time.time()
    while time.time() - t0 < 1.5:
        pygame.event.pump()
        for i in range(n_axes):
            v = joystick.get_axis(i)
            if v < mins[i]: mins[i] = v
            if v > maxs[i]: maxs[i] = v
        for i in range(n_buttons):
            if joystick.get_button(i):
                button_pressed[i] = True
        time.sleep(0.02)

    # Compute axis deltas.
    deltas = []
    for i in range(n_axes):
        if sign > 0:
            d = maxs[i] - rest[i]
        else:
            d = rest[i] - mins[i]
        deltas.append((d, i))
    deltas.sort(reverse=True)

    # Print the top 3 axes for visibility.
    print("  Top 3 axis deltas (in requested direction):")
    for d, i in deltas[:3]:
        print(f"    axis {i:>2}  delta {d:+.3f}  rest={rest[i]:+.3f}  "
              f"min={mins[i]:+.3f}  max={maxs[i]:+.3f}")

    best_d, best_i = deltas[0]

    # If a button was pressed during the sampling AND axis swing is weak,
    # the control is probably button-mapped (e.g., DInput-mode triggers).
    new_buttons = [i for i in range(n_buttons)
                   if button_pressed[i] and not rest_buttons[i]]
    if allow_button_fallback and best_d < min_delta and new_buttons:
        print(f"  Axis swing weak ({best_d:.2f}). But buttons {new_buttons} "
              f"became pressed during the push — this control might be "
              f"button-mapped, not analog.")
        confirm = input(f"  Use button {new_buttons[0]} as digital trigger? [Y/n] ").strip().lower()
        if confirm != "n":
            print(f"  -> button {new_buttons[0]} (digital trigger fallback)")
            return ("button", new_buttons[0])

    if best_d < min_delta:
        print(f"  WARNING: largest swing {best_d:.2f} below threshold {min_delta}.")
        print("  Try pushing harder. Or check the controller is in the right mode.")
        choice = input(f"  [r]etry, [a]ccept axis {best_i} anyway, or [s]kip: ").strip().lower()
        if choice == "a":
            print(f"  -> axis {best_i} (forced)")
            return best_i, float(sign)
        if choice == "s":
            print("  -> SKIPPED")
            return -1, 0.0
        return _detect_axis_pushed_to(joystick, sign, prompt, min_delta,
                                       allow_button_fallback)
    print(f"  -> axis {best_i} (swing {best_d:.2f})")
    return best_i, float(sign)


def _detect_button_press(joystick: pygame.joystick.Joystick,
                         prompt: str) -> int:
    """Ask user to press a button. Return its index."""
    n = joystick.get_numbuttons()
    print(f"\n{prompt}")
    input("  Press Enter when ready, then press the button once...")
    print("  ... watching for press ...")
    t0 = time.time()
    while time.time() - t0 < 5.0:
        pygame.event.pump()
        for i in range(n):
            if joystick.get_button(i):
                # Wait for release so we don't sample it for the next button.
                while joystick.get_button(i):
                    pygame.event.pump()
                    time.sleep(0.02)
                print(f"  -> button {i}")
                return i
        time.sleep(0.02)
    print("  TIMEOUT: no button press detected. Skipping.")
    return -1


def _detect_hat_direction(joystick: pygame.joystick.Joystick,
                          target: tuple[int, int],
                          prompt: str) -> int:
    """Ask user to press D-pad in a specific direction. Return hat index
    that produced the target tuple. Returns -1 if D-pad is buttons-only."""
    n = joystick.get_numhats()
    if n == 0:
        return -1
    print(f"\n{prompt}")
    input("  Press Enter when ready, then press the D-pad in that direction...")
    print("  ... watching for D-pad ...")
    t0 = time.time()
    while time.time() - t0 < 5.0:
        pygame.event.pump()
        for i in range(n):
            v = joystick.get_hat(i)
            if v == target:
                # Wait for release.
                while joystick.get_hat(i) != (0, 0):
                    pygame.event.pump()
                    time.sleep(0.02)
                print(f"  -> hat {i}")
                return i
        time.sleep(0.02)
    print("  TIMEOUT: no D-pad input detected. (Maybe it's button-mapped?)")
    return -1


def _live_monitor(joystick: pygame.joystick.Joystick) -> None:
    """Show ALL axes + buttons + hats in real time. User moves controls to
    discover indices manually. Exit with Ctrl+C."""
    n_axes    = joystick.get_numaxes()
    n_buttons = joystick.get_numbuttons()
    n_hats    = joystick.get_numhats()

    print("\n=== LIVE MONITOR — move every control on your gamepad ===")
    print(f"  axes={n_axes}  buttons={n_buttons}  hats={n_hats}")
    print("  Press Ctrl+C when done watching.\n")

    try:
        while True:
            pygame.event.pump()
            ax_str = " ".join(f"{i}={joystick.get_axis(i):+.2f}"
                              for i in range(n_axes))
            btns_pressed = [i for i in range(n_buttons)
                            if joystick.get_button(i)]
            hat_str = " ".join(f"h{i}={joystick.get_hat(i)}"
                               for i in range(n_hats))
            btn_str = "btn=" + (",".join(map(str, btns_pressed))
                                if btns_pressed else "-")
            print(f"  AXES[{ax_str}]  {btn_str}  {hat_str}", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\nDone monitoring.")


def _manual_calibration(joystick: pygame.joystick.Joystick) -> dict:
    """Skip auto-detection — user types in the indices for each control."""
    print("\n=== MANUAL CALIBRATION — type the axis/button/hat index ===")
    print("Use --monitor first to figure out which index is which.\n")
    cal: dict = {
        "joystick_name": joystick.get_name(),
        "axes": {}, "buttons": {}, "hats": {},
    }

    def ask_axis(slot, prompt):
        s = input(f"  {prompt} [axis index, or blank to skip]: ").strip()
        if not s:
            return
        try:
            idx = int(s)
        except ValueError:
            print("    invalid, skipping")
            return
        # Determine sign by asking which direction is "positive."
        sign_str = input("    sign? [+1 or -1, default +1]: ").strip()
        sign = -1.0 if sign_str.startswith("-") else 1.0
        cal["axes"][slot] = {"index": idx, "sign": sign}

    def ask_btn(slot, prompt):
        s = input(f"  {prompt} [button index, or blank to skip]: ").strip()
        if s:
            try:
                cal["buttons"][slot] = int(s)
            except ValueError:
                pass

    def ask_hat(slot, prompt):
        s = input(f"  {prompt} [hat index, or blank if D-pad is buttons]: ").strip()
        if s:
            try:
                cal["hats"][slot] = int(s)
            except ValueError:
                pass

    print("\n--- STICKS ---")
    ask_axis("lstick_x", "L stick X (right = +)")
    ask_axis("lstick_y", "L stick Y (down/back = +)")
    ask_axis("rstick_x", "R stick X (right = +)")
    ask_axis("rstick_y", "R stick Y (down/back = +)")
    print("\n--- TRIGGERS (use AXIS if analog, BUTTON if digital) ---")
    s = input("  L trigger is [a]xis or [b]utton? ").strip().lower()
    if s.startswith("b"):
        ask_btn("ltrigger", "L trigger button index")
    else:
        ask_axis("ltrigger", "L trigger axis (pressed = +)")
    s = input("  R trigger is [a]xis or [b]utton? ").strip().lower()
    if s.startswith("b"):
        ask_btn("rtrigger", "R trigger button index")
    else:
        ask_axis("rtrigger", "R trigger axis (pressed = +)")
    print("\n--- D-PAD ---")
    ask_hat("dpad", "D-pad hat index (usually 0)")
    if "dpad" not in cal["hats"]:
        ask_btn("dpad_up",    "D-pad UP button")
        ask_btn("dpad_down",  "D-pad DOWN button")
        ask_btn("dpad_left",  "D-pad LEFT button")
        ask_btn("dpad_right", "D-pad RIGHT button")
    print("\n--- ACTION BUTTONS ---")
    ask_btn("a",     "A")
    ask_btn("b",     "B")
    ask_btn("x",     "X")
    ask_btn("y",     "Y")
    ask_btn("start", "START (servo-kill toggle)")
    return cal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--joystick-index", type=int, default=None,
                   help="override auto-detection.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                   help=f"output JSON path (default: {DEFAULT_OUTPUT}).")
    p.add_argument("--monitor", action="store_true",
                   help="just show live axis/button/hat values; no calibration. "
                        "Use this to figure out which indices map to which "
                        "physical control on your gamepad.")
    p.add_argument("--manual", action="store_true",
                   help="skip auto-detection — type indices directly. "
                        "Useful when auto-detection mis-identifies controls. "
                        "Run with --monitor first to discover indices.")
    args = p.parse_args()

    joystick = _select_joystick(args.joystick_index)

    if args.monitor:
        _live_monitor(joystick)
        pygame.quit()
        return

    if args.manual:
        cal = _manual_calibration(joystick)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(cal, f, indent=2)
        print(f"\nSaved manual calibration to: {args.out}")
        pygame.quit()
        return

    print(f"\nCalibrating: {joystick.get_name()}")
    print("=" * 60)
    print("This walk-through will record which axis/button/hat each")
    print("control on your gamepad maps to. ~2 minutes total.")
    print("Use --monitor to see live values, or --manual to type indices.")
    print("=" * 60)

    cal: dict = {
        "joystick_name": joystick.get_name(),
        "axes":    {},
        "buttons": {},
        "hats":    {},
    }

    # ---- Sticks (push in a clear direction so we know index + sign) ----
    print("\n--- LEFT STICK ---")
    idx, sign = _detect_axis_pushed_to(
        joystick, sign=+1,
        prompt="Push the LEFT stick fully to the RIGHT (and hold for 1s).")
    cal["axes"]["lstick_x"] = {"index": idx, "sign": sign}

    idx, sign = _detect_axis_pushed_to(
        joystick, sign=+1,
        prompt="Push the LEFT stick fully BACKWARD/DOWN (toward you, hold for 1s).")
    cal["axes"]["lstick_y"] = {"index": idx, "sign": sign}

    print("\n--- RIGHT STICK ---")
    idx, sign = _detect_axis_pushed_to(
        joystick, sign=+1,
        prompt="Push the RIGHT stick fully to the RIGHT (and hold for 1s).")
    cal["axes"]["rstick_x"] = {"index": idx, "sign": sign}

    idx, sign = _detect_axis_pushed_to(
        joystick, sign=+1,
        prompt="Push the RIGHT stick fully BACKWARD/DOWN (toward you, hold for 1s).")
    cal["axes"]["rstick_y"] = {"index": idx, "sign": sign}

    # ---- Triggers (analog OR digital fallback) ----
    print("\n--- TRIGGERS ---")
    print("Note: triggers are sometimes button-mapped instead of analog axes")
    print("on certain controllers / driver modes. The script will detect which.")

    res = _detect_axis_pushed_to(
        joystick, sign=+1,
        prompt="Press the LEFT trigger (LT) fully (hold for 1.5s).",
        allow_button_fallback=True)
    if isinstance(res, tuple) and res[0] == "button":
        cal["buttons"]["ltrigger"] = res[1]
    else:
        cal["axes"]["ltrigger"] = {"index": res[0], "sign": res[1]}

    res = _detect_axis_pushed_to(
        joystick, sign=+1,
        prompt="Press the RIGHT trigger (RT) fully (hold for 1.5s).",
        allow_button_fallback=True)
    if isinstance(res, tuple) and res[0] == "button":
        cal["buttons"]["rtrigger"] = res[1]
    else:
        cal["axes"]["rtrigger"] = {"index": res[0], "sign": res[1]}

    # ---- D-pad (try hat first, fall back to buttons) ----
    print("\n--- D-PAD ---")
    hat_up = _detect_hat_direction(joystick, (0, 1),
                                    "Press D-pad UP.")
    if hat_up >= 0:
        cal["hats"]["dpad"] = hat_up
    else:
        # D-pad shows up as buttons. Detect each direction.
        cal["buttons"]["dpad_up"]    = _detect_button_press(joystick, "Press D-pad UP.")
        cal["buttons"]["dpad_down"]  = _detect_button_press(joystick, "Press D-pad DOWN.")
        cal["buttons"]["dpad_left"]  = _detect_button_press(joystick, "Press D-pad LEFT.")
        cal["buttons"]["dpad_right"] = _detect_button_press(joystick, "Press D-pad RIGHT.")

    # ---- Action buttons ----
    print("\n--- ACTION BUTTONS ---")
    cal["buttons"]["a"]     = _detect_button_press(joystick, "Press A (or whichever you want as 'A').")
    cal["buttons"]["b"]     = _detect_button_press(joystick, "Press B.")
    cal["buttons"]["x"]     = _detect_button_press(joystick, "Press X.")
    cal["buttons"]["y"]     = _detect_button_press(joystick, "Press Y.")
    cal["buttons"]["start"] = _detect_button_press(joystick, "Press START (the small button used as servo-kill toggle).")

    # ---- Save ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cal, f, indent=2)
    print("\n" + "=" * 60)
    print(f"Saved calibration to: {args.out}")
    print("=" * 60)
    print("Now run scripts/watch_controller.py — it will auto-load this file.")
    pygame.quit()


if __name__ == "__main__":
    main()
