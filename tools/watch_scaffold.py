"""
watch_scaffold.py — render the analytical scaffold gait directly in mujoco.viewer.

No policy, no training, no saved trajectory — calls Controller.predict(cmd, t)
in real time and pushes joint angles to a passive viewer for inspection.
Useful for tuning the scaffold's stance width / step amplitude / body bob
parameters and seeing the effect immediately.

Usage examples (Windows venv):

    .venv\\Scripts\\python.exe tools/watch_scaffold.py
        # Defaults: forward at 0.167 m/s (matches TO target)

    .venv\\Scripts\\python.exe tools/watch_scaffold.py --vy 0.167
        # Pure lateral left

    .venv\\Scripts\\python.exe tools/watch_scaffold.py --wz 0.5
        # Pure turn-in-place at 0.5 rad/s

    .venv\\Scripts\\python.exe tools/watch_scaffold.py --vx 0.118 --vy 0.118 --wz 0.3
        # Combined: 45° forward-left while turning
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gait.controller import Controller


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/phantomx_simple_mjx.xml")
    # 9-component cmd vector: [vx, vy, wz, pitch, roll, dh, dw, sx, sy]
    ap.add_argument("--vx", type=float, default=0.167,
                    help="forward velocity, m/s (default 0.167 = TO target)")
    ap.add_argument("--vy", type=float, default=0.0,
                    help="lateral velocity (left = +), m/s")
    ap.add_argument("--wz", type=float, default=0.0,
                    help="yaw rate (CCW = +), rad/s")
    ap.add_argument("--pitch", type=float, default=0.0, help="body pitch, rad")
    ap.add_argument("--roll", type=float, default=0.0, help="body roll, rad")
    ap.add_argument("--dh", type=float, default=0.0,
                    help="height delta from neutral, m (- = body raised)")
    ap.add_argument("--dw", type=float, default=0.0,
                    help="stance width delta from neutral, m (+ = wider)")
    ap.add_argument("--sx", type=float, default=0.0, help="body shift +X, m")
    ap.add_argument("--sy", type=float, default=0.0, help="body shift +Y, m")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="how long to run, seconds (default 60)")
    ap.add_argument("--rate", type=float, default=50.0,
                    help="control rate, Hz (default 50 = matches env)")
    ap.add_argument("--kinematic", action="store_true", default=True,
                    help="hold base at scaffold's commanded body pose "
                         "(default; gravity-free visualization)")
    ap.add_argument("--physics", dest="kinematic", action="store_false",
                    help="run full mujoco physics — base falls under gravity "
                         "if the gait can't support it")
    ap.add_argument("--sweep", action="store_true",
                    help="run a hardcoded turn-radius sweep instead of a static "
                         "cmd: forward + tight right + medium + gentle + "
                         "straight + gentle left + medium + tight left + forward. "
                         "Uses --vx for the forward speed; ignores --wz.")
    ap.add_argument("--stance-sweep", action="store_true",
                    help="cycle through height_delta and width_delta values "
                         "while walking forward at --vx. Demonstrates the "
                         "scaffold's response to D-pad-style stance commands. "
                         "Bounds match env: height ±20mm, width ±15mm.")
    ap.add_argument("--wz-trim-vx", type=float, default=0.0,
                    help="yaw trim coef applied as wz += -k * vx before scaffold "
                         "call. Same value used during prior gen / BC / training. "
                         "Tuned for current scaffold: 0.005.")
    ap.add_argument("--wz-trim-vy-abs", type=float, default=0.0,
                    help="yaw trim coef applied as wz += -k * |vy| before scaffold "
                         "call. Tuned: -0.012.")
    ap.add_argument("--interactive", action="store_true",
                    help="enable keyboard-driven stance + motion exploration. "
                         "Keys (in viewer window): "
                         "[1] dh-=5mm raise body | [2] dh+=5mm lower body | "
                         "[3] dw-=5mm narrower | [4] dw+=5mm wider | "
                         "[5] cycle motion mode (in-place / fwd-max / spin-max / arc-tight) | "
                         "[r] reset cmd to neutral | [h] reprint help | [q] quit. "
                         "Each change prints current cmd to console.")
    ap.add_argument("--stance-step", type=float, default=0.005,
                    help="step size for interactive dh/dw changes, m (default 5mm)")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="visualization timescale: 1.0 = real time, "
                         "0.5 = half speed (legs cycle 2x slower), "
                         "2.0 = double speed. Cmd magnitudes unchanged — "
                         "only wall-clock pacing slows. Useful with "
                         "--interactive to study fast gait patterns.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / args.model

    # Sweep schedule: (duration_sec, wz_rad/s). vx held at args.vx throughout.
    # wz values chosen to land within Controller.MIN/MAX_TURN_RADIUS (0.350-0.755m
    # at vx=0.103). wz_max ~ 0.29 (R near MIN), wz_min ~ 0.14 (R near MAX).
    sweep_phases = [
        (5.0,  0.0),     # straight forward
        (8.0, -0.29),    # tight right (R ~ 0.36m, just above MIN)
        (8.0, -0.21),    # medium right (R ~ 0.49m)
        (8.0, -0.14),    # gentle right (R ~ 0.74m, near MAX)
        (5.0,  0.0),     # straight (recover)
        (8.0, +0.14),    # gentle left
        (8.0, +0.21),    # medium left
        (8.0, +0.29),    # tight left
        (5.0,  0.0),     # straight (return to neutral)
    ]
    sweep_total = sum(p[0] for p in sweep_phases)

    # Stance-sweep schedule: cycle through (height_delta, width_delta) values.
    # Bounds match envs/hexapod_env_jax.py:cmd_sample_ranges.
    # Each phase is (duration_sec, dh, dw). vx held at args.vx throughout.
    stance_phases = [
        (4.0,  0.000,  0.000),   # neutral
        (4.0, -0.020,  0.000),   # body raised by 20 mm (max)
        (4.0,  0.000,  0.000),   # back to neutral
        (4.0, +0.020,  0.000),   # body lowered by 20 mm (max)
        (4.0,  0.000,  0.000),   # back to neutral
        (4.0,  0.000, +0.015),   # stance widened by 15 mm/foot (max)
        (4.0,  0.000,  0.000),   # back to neutral
        (4.0,  0.000, -0.015),   # stance narrowed by 15 mm/foot (max)
        (4.0,  0.000,  0.000),   # back to neutral
        (4.0, -0.020, +0.015),   # raised + wide (corner case)
        (4.0, +0.020, -0.015),   # lowered + narrow (opposite corner)
        (4.0,  0.000,  0.000),   # back to neutral
    ]
    stance_total = sum(p[0] for p in stance_phases)

    def cmd_at_time(t: float) -> np.ndarray:
        """Return the 9-component cmd at time t. For sweep modes, walk through
        the schedule with 0.5s linear ramps between phase endpoints to avoid
        abrupt discontinuities."""
        # Interactive mode overrides everything: vx/wz from current motion mode,
        # dh/dw from key-driven inter_state.
        if args.interactive:
            _, vx, wz = INTER_MOTION_MODES[inter_state["motion_idx"]]
            return np.array([vx, 0, wz, 0, 0,
                             inter_state["dh"], inter_state["dw"],
                             0, 0], dtype=np.float64)
        if args.stance_sweep:
            ramp = 0.5
            accum = 0.0
            prev_dh, prev_dw = stance_phases[0][1], stance_phases[0][2]
            for dur, dh, dw in stance_phases:
                if t < accum + dur:
                    local = t - accum
                    if local < ramp and accum > 0:
                        alpha = local / ramp
                        use_dh = (1 - alpha) * prev_dh + alpha * dh
                        use_dw = (1 - alpha) * prev_dw + alpha * dw
                    else:
                        use_dh, use_dw = dh, dw
                    return np.array([args.vx, args.vy, args.wz,
                                     args.pitch, args.roll, use_dh, use_dw,
                                     args.sx, args.sy], dtype=np.float64)
                accum += dur
                prev_dh, prev_dw = dh, dw
            return np.array([args.vx, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        if not args.sweep:
            return np.array([args.vx, args.vy, args.wz,
                             args.pitch, args.roll, args.dh, args.dw,
                             args.sx, args.sy], dtype=np.float64)
        ramp = 0.5
        accum = 0.0
        prev_wz = sweep_phases[0][1]
        for dur, wz in sweep_phases:
            if t < accum + dur:
                # Ramp in for the first `ramp` seconds of each phase.
                local = t - accum
                if local < ramp and accum > 0:
                    alpha = local / ramp
                    use_wz = (1 - alpha) * prev_wz + alpha * wz
                else:
                    use_wz = wz
                return np.array([args.vx, args.vy, use_wz,
                                 args.pitch, args.roll, args.dh, args.dw,
                                 args.sx, args.sy], dtype=np.float64)
            accum += dur
            prev_wz = wz
        return np.array([args.vx, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)

    if args.stance_sweep:
        if args.duration < stance_total:
            args.duration = stance_total
        print(f"[scaffold] stance-sweep mode: vx={args.vx} m/s, total "
              f"{stance_total:.0f}s across {len(stance_phases)} phases")
    elif args.sweep:
        # Override duration to span the full sweep.
        if args.duration < sweep_total:
            args.duration = sweep_total
        print(f"[scaffold] sweep mode: vx={args.vx} m/s, total {sweep_total:.0f}s "
              f"across {len(sweep_phases)} phases")
    elif args.interactive:
        # Interactive mode prints its own state via _print_inter() below; the
        # cmd_at_time path here would dereference INTER_MOTION_MODES which
        # isn't defined yet (set up after Controller(...) is built).
        print("[scaffold] interactive mode (see help below)")
    else:
        cmd0 = cmd_at_time(0.0)
        print(f"[scaffold] cmd = {cmd0}")
    print(f"[scaffold] kinematic={args.kinematic}, rate={args.rate} Hz, "
          f"duration={args.duration}s")

    ctrl = Controller(str(model_path))
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Interactive mode: holds mutable cmd state + motion mode index.
    # Modified by key_callback below; read by cmd_at_time during the loop.
    INTER_MOTION_MODES = [
        ("in-place",     0.000,  0.0),                      # vx, wz — kinematic shows gait cycling
        ("forward-max",  ctrl.MAX_SPEED * 0.9,  0.0),       # ~90% of physical max
        ("spin-max",     0.0,                  ctrl.MAX_YAW_RATE * 0.9),
        ("arc-tight",    ctrl.MAX_SPEED * 0.6, ctrl.MAX_YAW_RATE * 0.7),  # combined → smallest practical radius
    ]
    # Height presets imported from envs/stance_envelope.py — single source
    # of truth; if the verified bounds change, edit that file and the
    # interactive tuner automatically picks up the new presets.
    from envs.stance_envelope import HEIGHT_PRESETS
    inter_state = {
        "dh": 0.0, "dw": 0.0,
        "motion_idx": 0,
        "height_preset_idx": -1,   # -1 = "no preset active"
    }

    def _print_inter():
        name, vx, wz = INTER_MOTION_MODES[inter_state["motion_idx"]]
        dh = inter_state["dh"]; dw = inter_state["dw"]
        body_z_now = body_height_default - dh
        stance_w_delta = dw * 1000.0
        preset_tag = ""
        if inter_state["height_preset_idx"] >= 0:
            preset_tag = f" [preset {inter_state['height_preset_idx']+1}/{len(HEIGHT_PRESETS)}]"
        print(f"[interactive] motion={name:11s}  vx={vx:+.3f}  wz={wz:+.3f}  "
              f"dh={dh:+.3f} (body_z={body_z_now*1000:.1f} mm){preset_tag}  "
              f"dw={dw:+.4f} ({stance_w_delta:+.1f} mm/foot)")

    def _print_help():
        print("\n=== INTERACTIVE STANCE/MOTION TUNER ===")
        print(f"  [1] dh -= {args.stance_step*1000:.0f}mm  (raise body — body_z increases)")
        print(f"  [2] dh += {args.stance_step*1000:.0f}mm  (lower body)")
        print(f"  [3] dw -= {args.stance_step*1000:.0f}mm  (narrower stance)")
        print(f"  [4] dw += {args.stance_step*1000:.0f}mm  (wider stance)")
        print( "  [5] cycle motion mode")
        print(f"  [6] cycle height preset ({len(HEIGHT_PRESETS)} options)")
        ps = "  ".join(f"{i+1}:{int(-(body_height_default-h)*1000)}mm" if False
                       else f"{i+1}:dh{h:+.3f}"
                       for i, h in enumerate(HEIGHT_PRESETS))
        print(f"        presets: {ps}")
        print( "  [r] reset dh/dw/mode to zero")
        print( "  [h] reprint this help")
        print( "  [q] quit (or close viewer window)")
        print("=" * 40)

    def apply_trim(cmd):
        """Same trim applied during prior gen / BC / AMP training. Adjusts
        cmd[2] (wz) only; doesn't touch the cmd the policy sees in obs."""
        if args.wz_trim_vx == 0.0 and args.wz_trim_vy_abs == 0.0:
            return cmd
        out = cmd.copy()
        out[2] += -args.wz_trim_vx * cmd[0] - args.wz_trim_vy_abs * abs(cmd[1])
        return out

    # Initial pose: neutral joints, base at scaffold's commanded standing pose.
    cmd_init = apply_trim(cmd_at_time(0.0))
    joints0, feet0 = ctrl.predict_with_feet(cmd_init, t=0.0)
    # Stance height = -mean(LEG_ORIGIN_BODY z), since the foot rests are at
    # body z = -stance_height when the body sits at world z = stance_height.
    # Sign convention: cmd[5] (dh) NEGATIVE means body RAISED, so world z
    # increases as dh decreases → body_z = default - dh.
    body_height_default = float(-ctrl.LEG_ORIGIN_BODY[:, 2].mean())
    body_height = body_height_default - cmd_init[5]
    data.qpos[0:3] = (0.0, 0.0, body_height)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7:25] = joints0
    data.ctrl[:18] = joints0
    mujoco.mj_forward(model, data)

    period = 1.0 / args.rate
    n_steps = int(args.duration * args.rate)
    t_start = time.time()

    # Body-frame state for step-by-step Euler integration of the world body
    # pose. Closed-form arc integration only works for constant cmd; sweep
    # mode varies wz over time, so step integration is required.
    body_world_x = 0.0
    body_world_y = 0.0
    body_yaw = 0.0
    last_phase_idx = -1

    # Build key callback only in interactive mode; mujoco.viewer accepts
    # a `key_callback(keycode)` kwarg invoked on each key press.
    key_cb = None
    if args.interactive:
        _print_help()
        _print_inter()
        def key_cb(keycode):
            ch = chr(keycode) if 0 <= keycode < 256 else ""
            if ch == "1":
                inter_state["dh"] -= args.stance_step
                inter_state["height_preset_idx"] = -1   # manual change → leave preset
            elif ch == "2":
                inter_state["dh"] += args.stance_step
                inter_state["height_preset_idx"] = -1
            elif ch == "3":
                inter_state["dw"] -= args.stance_step
            elif ch == "4":
                inter_state["dw"] += args.stance_step
            elif ch == "5":
                inter_state["motion_idx"] = (inter_state["motion_idx"] + 1) % len(INTER_MOTION_MODES)
            elif ch == "6":
                # Cycle to next height preset, snap dh to that value.
                next_idx = (inter_state["height_preset_idx"] + 1) % len(HEIGHT_PRESETS)
                inter_state["height_preset_idx"] = next_idx
                inter_state["dh"] = HEIGHT_PRESETS[next_idx]
                inter_state["dw"] = 0.0   # reset width on each new height test
            elif ch.lower() == "r":
                inter_state["dh"] = 0.0
                inter_state["dw"] = 0.0
                inter_state["motion_idx"] = 0
                inter_state["height_preset_idx"] = -1
            elif ch.lower() == "h":
                _print_help()
                return
            else:
                return
            _print_inter()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        for k in range(n_steps):
            if not viewer.is_running():
                break
            t = k * period
            cmd = apply_trim(cmd_at_time(t))
            joints, _ = ctrl.predict_with_feet(cmd, t)

            # In sweep mode, print phase transitions to stdout for context.
            if args.sweep:
                accum = 0.0
                phase_idx = 0
                for i, (dur, wz_p) in enumerate(sweep_phases):
                    if t < accum + dur:
                        phase_idx = i
                        break
                    accum += dur
                if phase_idx != last_phase_idx:
                    last_phase_idx = phase_idx
                    dur, wz_p = sweep_phases[phase_idx]
                    R_str = (f"R={args.vx/abs(wz_p):.2f}m" if abs(wz_p) > 1e-6
                             else "straight")
                    print(f"  [{t:5.1f}s] phase {phase_idx + 1}/{len(sweep_phases)}: "
                          f"vx={args.vx} wz={wz_p:+.2f}  {R_str}")
            elif args.stance_sweep:
                accum = 0.0
                phase_idx = 0
                for i, (dur, _dh, _dw) in enumerate(stance_phases):
                    if t < accum + dur:
                        phase_idx = i
                        break
                    accum += dur
                if phase_idx != last_phase_idx:
                    last_phase_idx = phase_idx
                    dur, dh, dw = stance_phases[phase_idx]
                    print(f"  [{t:5.1f}s] phase {phase_idx + 1}/{len(stance_phases)}: "
                          f"dh={dh*1000:+5.1f}mm  dw={dw*1000:+5.1f}mm")

            if args.kinematic:
                # Interactive mode: pin body world pose to origin so the bot
                # walks "in place" — gait cycles visibly under the user's
                # eye, but the bot doesn't drift / rotate out of view across
                # motion-mode changes.
                if args.interactive:
                    body_height = body_height_default - float(cmd[5])
                    data.qpos[0:3] = (0.0, 0.0, body_height)
                    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
                else:
                    # Step-integrate body world pose. Velocity (vx, vy) is in
                    # body frame; rotate by current yaw to get world-frame velocity.
                    vx_now, vy_now, wz_now = float(cmd[0]), float(cmd[1]), float(cmd[2])
                    cy_yaw, sy_yaw = np.cos(body_yaw), np.sin(body_yaw)
                    body_world_x += period * (cy_yaw * vx_now - sy_yaw * vy_now)
                    body_world_y += period * (sy_yaw * vx_now + cy_yaw * vy_now)
                    body_yaw     += period * wz_now
                    # Body world z follows commanded height: dh negative = raised.
                    body_height = body_height_default - float(cmd[5])

                    data.qpos[0:3] = (body_world_x, body_world_y, body_height)
                    data.qpos[3] = np.cos(body_yaw / 2)
                    data.qpos[4] = 0.0
                    data.qpos[5] = 0.0
                    data.qpos[6] = np.sin(body_yaw / 2)
                data.qpos[7:25] = joints
                data.qvel[:6] = 0.0
                mujoco.mj_forward(model, data)
            else:
                data.ctrl[:18] = joints
                target = t_start + t / args.time_scale
                while time.time() < target:
                    mujoco.mj_step(model, data)

            viewer.sync()

            # Pace by sleep when in kinematic mode. time_scale stretches the
            # wall-clock target for each sim step: t/time_scale; values <1
            # play back slower (legs visibly slower despite same cmd magnitude).
            if args.kinematic:
                target = t_start + t / args.time_scale
                slack = target - time.time()
                if slack > 0:
                    time.sleep(slack)

    print(f"[scaffold] done.")


if __name__ == "__main__":
    main()
