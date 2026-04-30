"""
watch_demo.py — eval a trained PPO checkpoint by driving it through a preset
cmd-vector script. Mirrors IK_gait.py's SMART_TEST_SCRIPT but uses the trained
policy to produce joint motions instead of the analytical gait alone.

Useful for:
  - Watching a trained policy respond to specific commands (vs random ones).
  - A/B comparison against IK_gait.py: same cmd sequence, different controller.
  - Visualizing how the policy handles each cmd slot in isolation.

Usage (macOS — mjpython required for viewer):
    mjpython watch_demo.py                            # latest ckpt, gait_scale=0.0
    mjpython watch_demo.py --gait-scale 1.0           # full scaffold (analytical)
    mjpython watch_demo.py --gait-scale 0.5           # half scaffold
    mjpython watch_demo.py --ckpt path/to/ckpt        # explicit checkpoint (no .zip)
    mjpython watch_demo.py --run hexapod_v2_curriculum
    mjpython watch_demo.py --stochastic               # sample actions (default deterministic)
"""

import argparse
import glob
import math
import os
import re
import sys
import time

import numpy as np
from stable_baselines3 import PPO

# Use the Mac env on macOS (matches the training env so the policy sees the
# same obs distribution it was trained on), Linux env elsewhere.
# CRITICAL: this MUST match the env used during training. The two envs have
# the same observation_space dim (78) so PPO.load() doesn't complain about
# mismatch, but the obs *values* differ in subtle ways (scaffold_hint
# caching, reward shaping, sample_cmd ranges). Loading a Mac-trained policy
# into the Linux env produces "cursed" out-of-distribution behavior.
if sys.platform == "darwin":
    from envs.hexapod_env_mac import HexapodEnv
else:
    from envs.hexapod_env import HexapodEnv


# ============================================================================
# DEMO SCRIPT — cmd-vector phases that exercise each overlay
# ============================================================================
# Each entry: (label, dur_seconds, cmd_function(t_in_phase, ctrl) -> 9-dim cmd)
# Mirrors IK_gait.py's SMART_TEST_SCRIPT structure but for the env's cmd vec.

DEG = math.radians


def _cmd(**slots):
    """Build a 9-vec cmd from named slots."""
    cmd = np.zeros(9, dtype=np.float32)
    NAMES = dict(vx=0, vy=1, wz=2, pitch=3, roll=4,
                 height=5, width=6, shift_x=7, shift_y=8)
    for k, v in slots.items():
        cmd[NAMES[k]] = v
    return cmd


# In-distribution range for the Mac env's translation cmd. The policy was
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
    # Clamp to trained range — anything outside is OOD for the policy.
    speed_frac = max(TRAINED_SPEED_MIN_FRAC,
                     min(speed_frac, TRAINED_SPEED_MAX_FRAC))
    speed = ctrl.MAX_SPEED * speed_frac
    return _cmd(vx=speed * math.cos(heading_rad),
                vy=speed * math.sin(heading_rad))


def _spin(ctrl, sign, scale=1.0):
    return _cmd(wz=sign * ctrl.MAX_YAW_RATE * scale)


DEMO_PHASES = [
    # (label, duration_seconds, cmd_factory(t_in_phase, ctrl) -> 9-vec)
    # NOTE: STAND phases (cmd=0 translation) are commented out because the
    # Mac env's tightened sampling [40%, 85%] of MAX_SPEED never produces
    # cmd=0, so the policy was never trained on stand-still observations.
    # Eval phases that command zero produce out-of-distribution garbage.
    # Re-enable these once training includes cmd=0 episodes.
    # ("STAND  neutral",                       3.0,
    #     lambda t, c: _cmd()),
    # ("STAND  + width sweep",                 6.0,
    #     lambda t, c: _cmd(width=0.015 * math.sin(2*math.pi*t/6))),
    # ("STAND  + height sweep",                6.0,
    #     lambda t, c: _cmd(height=0.018 * math.sin(2*math.pi*t/6))),
    # ("STAND  + pitch sweep ±10°",            6.0,
    #     lambda t, c: _cmd(pitch=DEG(10) * math.sin(2*math.pi*t/6))),
    # ("STAND  + roll sweep ±10°",             6.0,
    #     lambda t, c: _cmd(roll=DEG(10) * math.sin(2*math.pi*t/6))),
    # ("STAND  + pitch+roll circle",           8.0,
    #     lambda t, c: _cmd(pitch=DEG(8) * math.cos(2*math.pi*t/8),
    #                       roll =DEG(8) * math.sin(2*math.pi*t/8))),

    # All WALK phases below run at DEFAULT_SPEED_FRAC (0.70) — mid-range of
    # the trained [0.40, 0.85] band. Cmds stay in-distribution.
    ("WALK   forward (heading=0)",           5.0,
        lambda t, c: _walk(c, 0)),
    ("WALK   strafe left  (heading=+90)",    5.0,
        lambda t, c: _walk(c, math.pi/2)),
    ("WALK   backward (heading=180)",        5.0,
        lambda t, c: _walk(c, math.pi)),
    ("WALK   strafe right (heading=-90)",    5.0,
        lambda t, c: _walk(c, -math.pi/2)),
    ("WALK   heading sweep 0 → 360°",       16.0,
        lambda t, c: _walk(c, 2*math.pi*t/16)),
    ("WALK   slow (40% of MAX_SPEED)",       6.0,
        lambda t, c: _walk(c, 0, speed_frac=TRAINED_SPEED_MIN_FRAC)),
    ("WALK   fast (85% of MAX_SPEED)",       6.0,
        lambda t, c: _walk(c, 0, speed_frac=TRAINED_SPEED_MAX_FRAC)),
    # Speed sweep within the trained band only. Maps sin([-1,+1]) → [0.40, 0.85].
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

    # SPIN phases (cmd[wz] != 0, cmd[vx] = cmd[vy] = 0) — also out of
    # distribution for translation-only training. Comment back in after
    # spin training is added in a future curriculum stage.
    # ("SPIN   CW   (full)",                  15.0,
    #     lambda t, c: _spin(c, -1)),
    # ("SPIN   CCW  (full)",                  15.0,
    #     lambda t, c: _spin(c, +1)),
    # ("SPIN   CCW  + width overlay",          8.0,
    #     lambda t, c: _spin(c, +1) + _cmd(width=0.012 * math.sin(2*math.pi*t/4))),
    # ("SPIN   CCW  + height overlay",         8.0,
    #     lambda t, c: _spin(c, +1) + _cmd(height=0.015 * math.sin(2*math.pi*t/4))),
    # ("SPIN   CCW  speed sweep",             12.0,
    #     lambda t, c: _spin(c, +1, scale=0.5*(1 + math.sin(2*math.pi*t/12)))),

    # No "return to neutral" — would re-trigger the cmd=0 OOD problem.
]


# ============================================================================
# Checkpoint discovery (matches watch.py)
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
    """Newest *complete* checkpoint (skips files modified in last 3s)."""
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


# ============================================================================
# Main
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Render a trained checkpoint through a preset cmd script.")
    p.add_argument("--ckpt", default=None, help="checkpoint path (without .zip)")
    p.add_argument("--run",  default=None, help="run dir under checkpoints/")
    p.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--gait-scale", type=float, default=None,
                   help="scaffold weight (0.0 = pure policy, 1.0 = full scaffold). "
                        "Default matches watch.py: 1.0 for stage 1, 0.6 stage 2, 0.0 stage 3+.")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions stochastically (default deterministic)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed multiplier (1.0 = real time, 2.0 = 2× faster, "
                        "0.5 = half speed, 0 = uncapped)")
    args = p.parse_args()

    # Resolve checkpoint.
    if args.ckpt:
        ckpt = args.ckpt
    else:
        run_dir = (os.path.join("checkpoints", args.run) if args.run
                   else latest_run_dir())
        if run_dir is None:
            print("No run directory found under checkpoints/. Wait for training to save or pass --ckpt.")
            sys.exit(1)
        ckpt = latest_checkpoint(run_dir)
        if ckpt is None:
            print(f"No complete checkpoint in {run_dir} yet (first save ~2M steps in).")
            sys.exit(1)

    env = HexapodEnv(stage=args.stage, render_mode="human")
    # Match watch.py's per-stage defaults so `mjpython watch_demo.py` with no
    # extra args shows scaffold-driven walking (same as `mjpython watch.py`).
    # Override with --gait-scale 0.0 to test pure policy.
    default_gs = {1: 1.0, 2: 0.6, 3: 0.0, 4: 0.0}[args.stage]
    env.gait_scale = args.gait_scale if args.gait_scale is not None else default_gs
    model = PPO.load(ckpt, env=env)

    total_dur = sum(d for _, d, _ in DEMO_PHASES)
    print(f"\nCheckpoint:  {ckpt}")
    print(f"Stage:       {args.stage}")
    print(f"gait_scale:  {env.gait_scale:.2f}")
    print(f"Action mode: {'stochastic' if args.stochastic else 'deterministic'}")
    print(f"Demo:        {len(DEMO_PHASES)} phases, {total_dur:.0f}s per loop")
    print(f"MAX_SPEED:   {env._ctrl.MAX_SPEED:.4f} m/s")
    print(f"MAX_YAW:     {env._ctrl.MAX_YAW_RATE:.4f} rad/s")
    print()

    obs, _ = env.reset()
    # Settle the bot before starting the demo. CRITICAL: the Mac env's
    # _sample_cmd never produces cmd=0, so the policy was NEVER trained on
    # cmd=0 observations. Feeding it zeros here produces out-of-distribution
    # garbage actions that topple the bot. Use the smallest in-distribution
    # cmd instead — a small forward walk that the policy has actually seen.
    settle_speed = (env.SPEED_MIN_FRAC + 0.05) * env._ctrl.MAX_SPEED
    env._cmd = np.array([settle_speed, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(200):
        obs, *_ = env.step(zero_action)

    sim_t = 0.0
    last_phase_idx = -1
    wall_t0 = time.time()

    while True:
        # Determine current phase from sim_t.
        t_in_cycle = sim_t % total_dur
        cumulative = 0.0
        phase_idx  = 0
        for i, (_, dur, _) in enumerate(DEMO_PHASES):
            if t_in_cycle < cumulative + dur:
                phase_idx = i
                break
            cumulative += dur

        label, dur, fn = DEMO_PHASES[phase_idx]
        t_in_phase = t_in_cycle - cumulative
        cmd = fn(t_in_phase, env._ctrl)

        if phase_idx != last_phase_idx:
            print(f"  t={sim_t:6.1f}s  [{phase_idx+1:2d}/{len(DEMO_PHASES)}]  {label}  ({dur:.1f}s)")
            last_phase_idx = phase_idx

        # Override env's cmd, then refresh BOTH the cmd slot AND the cached
        # scaffold_hint feet positions. _get_obs reads scaffold_hint from
        # self._latest_feet_body, which the env only updates inside step() via
        # predict_with_feet(_cmd, ...). Without this refresh, the policy sees
        # an obs where cmd is the new phase command but scaffold_hint is foot
        # targets from the PREVIOUS cmd — a (cmd, scaffold_hint) combination
        # the policy never saw during training. One bad action from that OOD
        # frame can topple the bot at every phase boundary.
        env._cmd = cmd
        env._latest_feet_body = env._ctrl.compute_foot_targets(
            env._cmd, env._sim_time).astype(np.float32)
        obs = env._get_obs()

        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, reward, terminated, truncated, info = env.step(action)
        sim_t += env._dt

        # Real-time pacing — throttle to wall clock so motion plays at natural
        # speed. --speed > 1.0 plays faster, < 1.0 slower, 0 uncapped.
        if args.speed > 0:
            target = wall_t0 + sim_t / args.speed
            lag = target - time.time()
            if lag > 0:
                time.sleep(lag)

        if terminated or truncated:
            tag = "FELL" if terminated else "TIMEOUT"
            print(f"  [{tag}] at t={sim_t:5.1f}s  in phase '{label}'  "
                  f"track={info.get('tracking_reward', 0):.2f} "
                  f"drift={info.get('drift_pen', 0):.3f}")
            obs, _ = env.reset()
            # Reset sim_t too so the demo restarts from phase 1 — easier to
            # diagnose what happened than picking up mid-script.
            sim_t = 0.0
            last_phase_idx = -1
            wall_t0 = time.time()


if __name__ == "__main__":
    main()
