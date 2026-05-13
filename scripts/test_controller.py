"""
scripts/test_controller.py — diagnostic tool for the joystick layer.

Lists every connected joystick, then polls joystick 0 (or --index N) at
20 Hz and prints any axis / button / hat changes with timestamps.

Use this BEFORE watch_controller.py if pygame can't find the 8BitDo,
or if buttons/axes seem to map to the wrong slots. Common causes:
  * Controller in wrong mode (Switch / DInput / Xbox — try the X
    toggle on the back of the Ultimate 2)
  * 2.4G dongle not seated / Bluetooth not paired
  * USB cable charging-only (some cables don't pass data)

Usage:
    $env:PYTHONPATH = "."
    .venv\\Scripts\\python.exe scripts\\test_controller.py

    # Pick a different joystick if multiple are connected:
    .venv\\Scripts\\python.exe scripts\\test_controller.py --index 2
"""
from __future__ import annotations

import argparse
import time

import pygame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=None,
                   help="joystick index to monitor (default: auto-pick "
                        "first non-Azeron, non-SpaceMouse device)")
    p.add_argument("--rate", type=float, default=20.0,
                   help="poll rate (Hz, default 20)")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="axis change must exceed this to print "
                        "(events mode only, default 0.05)")
    p.add_argument("--rescan", action="store_true",
                   help="re-init pygame.joystick once before polling.")
    p.add_argument("--polar", action="store_true",
                   help="live polar readout: shows magnitude + angle for both "
                        "sticks on a single self-updating line. Best for "
                        "verifying stick math (mag should be ~1 at full "
                        "deflection in any direction; angle should be the "
                        "body-frame heading). The default mode prints change "
                        "events instead — use that for button discovery.")
    args = p.parse_args()

    pygame.init()
    pygame.joystick.init()
    if args.rescan:
        pygame.joystick.quit()
        pygame.joystick.init()

    n = pygame.joystick.get_count()
    print(f"\n=== {n} joystick(s) detected ===")
    if n == 0:
        print("None found. Make sure:")
        print("  * 8BitDo Ultimate 2 is in XBox mode (X toggle on back)")
        print("  * Either dongle is plugged into USB, OR Bluetooth is paired")
        print("  * Controller is powered ON (Home button)")
        print("  * No 'unknown device' warnings in Windows Device Manager")
        return

    for i in range(n):
        j = pygame.joystick.Joystick(i)
        guid = j.get_guid() if hasattr(j, "get_guid") else "(no guid)"
        try:
            ptype = j.get_power_level() if hasattr(j, "get_power_level") else "?"
        except Exception:
            ptype = "?"
        print(f"  [{i}] {j.get_name()}")
        print(f"       guid={guid}")
        print(f"       axes={j.get_numaxes()} buttons={j.get_numbuttons()} "
              f"hats={j.get_numhats()} balls={j.get_numballs()}  "
              f"power={ptype}")

    # Pick which to monitor.
    if args.index is not None:
        if args.index >= n:
            print(f"\nERROR: --index {args.index} but only {n} connected.")
            return
        idx = args.index
    else:
        # Auto-pick: skip Azeron/SpaceMouse if present.
        skip_substrings = ["azeron", "spacemouse", "3dconnexion"]
        idx = 0
        for i in range(n):
            name = pygame.joystick.Joystick(i).get_name().lower()
            if not any(s in name for s in skip_substrings):
                idx = i
                break
        else:
            idx = 0

    j = pygame.joystick.Joystick(idx)
    print(f"\n=== Monitoring joystick [{idx}]: {j.get_name()} ===")

    n_axes    = j.get_numaxes()
    n_buttons = j.get_numbuttons()
    n_hats    = j.get_numhats()

    # ---- Live polar mode: one self-updating line, no scrolling ----
    if args.polar:
        import math
        print("Format:")
        print("  L: (lx, ly) mag=X.XX ang=±YYY°  R: (rx, ry) ...  LT=X.XX RT=X.XX  hat btns")
        print("  angle convention: 0°=forward, +90°=left, -90°=right, ±180°=back")
        print("Move sticks. Ctrl+C to exit.\n")
        period = 1.0 / args.rate
        try:
            while True:
                pygame.event.pump()
                # L stick: pygame axes 0/1
                lx = j.get_axis(0) if n_axes > 0 else 0.0
                ly = j.get_axis(1) if n_axes > 1 else 0.0
                l_mag = (lx*lx + ly*ly) ** 0.5
                # Body-frame heading (vx = -ly, vy = -lx convention).
                l_ang = math.degrees(math.atan2(-lx, -ly)) if l_mag > 1e-3 else 0.0
                # R stick: pygame axes 2/3
                rx = j.get_axis(2) if n_axes > 2 else 0.0
                ry = j.get_axis(3) if n_axes > 3 else 0.0
                r_mag = (rx*rx + ry*ry) ** 0.5
                r_ang = math.degrees(math.atan2(-rx, -ry)) if r_mag > 1e-3 else 0.0
                # Triggers (axes 4/5).
                lt = j.get_axis(4) if n_axes > 4 else 0.0
                rt = j.get_axis(5) if n_axes > 5 else 0.0
                # Hat (D-pad).
                hat = j.get_hat(0) if n_hats > 0 else (0, 0)
                # Pressed buttons.
                btns = [i for i in range(n_buttons) if j.get_button(i)]
                btn_str = ",".join(map(str, btns)) if btns else "-"

                line = (
                    f"L:({lx:+.2f},{ly:+.2f}) m={l_mag:.2f} a={l_ang:+7.1f}deg  "
                    f"R:({rx:+.2f},{ry:+.2f}) m={r_mag:.2f} a={r_ang:+7.1f}deg  "
                    f"LT={lt:+.2f} RT={rt:+.2f}  hat={hat}  btns=[{btn_str}]"
                )
                # Pad to overwrite any longer previous line.
                print(line.ljust(140), end="\r", flush=True)
                time.sleep(period)
        except KeyboardInterrupt:
            print("\n\nDone.")
        finally:
            pygame.quit()
        return

    print(f"Move sticks / press buttons / press D-pad. Ctrl+C to exit.\n")

    # Snapshot initial state so we only print changes.
    last_axes    = [j.get_axis(i)   for i in range(n_axes)]
    last_buttons = [j.get_button(i) for i in range(n_buttons)]
    last_hats    = [j.get_hat(i)    for i in range(n_hats)]

    period = 1.0 / args.rate
    t0 = time.time()
    try:
        while True:
            tick_start = time.time()
            pygame.event.pump()

            elapsed = time.time() - t0

            # Axes: print only if any axis crossed the threshold.
            for i in range(n_axes):
                v = j.get_axis(i)
                if abs(v - last_axes[i]) > args.threshold:
                    print(f"  [{elapsed:6.2f}s]  AXIS {i:>2}  = {v:+.3f}  "
                          f"(was {last_axes[i]:+.3f})")
                    last_axes[i] = v

            # Buttons: print on press / release.
            for i in range(n_buttons):
                v = j.get_button(i)
                if v != last_buttons[i]:
                    state = "PRESSED " if v else "released"
                    print(f"  [{elapsed:6.2f}s]  BUTTON {i:>2} {state}")
                    last_buttons[i] = v

            # Hats: print on change.
            for i in range(n_hats):
                v = j.get_hat(i)
                if v != last_hats[i]:
                    print(f"  [{elapsed:6.2f}s]  HAT {i}     = {v}")
                    last_hats[i] = v

            sleep_for = period - (time.time() - tick_start)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
