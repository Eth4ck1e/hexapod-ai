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

from envs.hexapod_env import HexapodEnv

# Shared cmd-vector demo script lives at scripts/demo_phases.py so the
# active JAX viewer can also import from it without crossing the
# legacy boundary. We re-export the same names here.
from scripts.demo_phases import (
    DEMO_PHASES, _cmd, _walk, _spin,
    DEG, TRAINED_SPEED_MIN_FRAC, TRAINED_SPEED_MAX_FRAC, DEFAULT_SPEED_FRAC,
    latest_run_dir, latest_checkpoint,
)


# DEMO_PHASES, _cmd, _walk, _spin, latest_run_dir, latest_checkpoint
# all imported above from scripts.demo_phases — no inline copies here.


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
