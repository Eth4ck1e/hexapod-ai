"""
scripts/record_policy.py — record an MP4 of a trained policy with the
camera tracking the bot.

Renders offline via mujoco.Renderer (no live viewer window). Camera follows
the bot's body so the bot never walks off-frame. Drives the env through one
of the demo schedules (paper / paper_stance / interactive sequence).

Examples (Windows venv):
    $env:PYTHONPATH = "."

    # Default: paper_stance demo, 1080p, mesh model in viewer:
    .venv\\Scripts\\python.exe scripts\\record_policy.py `
        --params checkpoints\\amp_to_v8\\iter5\\final\\params.pkl `
        --action-space foot --demo paper_stance --out v8_demo.mp4

    # Custom resolution + duration, simple model:
    .venv\\Scripts\\python.exe scripts\\record_policy.py `
        --params checkpoints\\amp_to_v8\\iter5\\final\\params.pkl `
        --action-space foot --demo paper --out v8_paper.mp4 `
        --width 1920 --height 1080 --duration 60

The output MP4 plays at real-time (50 fps to match the env's 200 Hz physics
with 4× downsample = 50 frames/sec sim-time which equals real-time).
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np
import mujoco
import imageio
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain_train import enable_jax_cache, ACTION_SPACE as CHAIN_ACTION_SPACE, MODEL_PATH as CHAIN_MODEL_PATH
enable_jax_cache()

from brax.training.agents.ppo.networks import make_ppo_networks
from brax.training.acme import running_statistics

from envs.hexapod_env import HexapodEnv
from demo_phases import (
    DEMO_PHASES, DEMO_PHASES_PAPER, DEMO_PHASES_PAPER_STANCE,
    DEMO_PHASES_SHOWCASE,
)


def _patched_obs(env: HexapodEnv) -> None:
    """JAX-style obs (matches watch_demo_jax / watch_controller)."""
    def _get_obs(self):
        qpos = self._data.qpos
        qvel = self._data.qvel
        scaffold_hint = self._latest_feet_body.flatten()
        phase = self._ctrl.get_phase(self._sim_time)
        body_linvel = np.asarray(self._body_frame_linvel(), dtype=np.float32)
        if self.stage == 4:
            body_linvel = np.zeros(3, dtype=np.float32)
        return np.concatenate([
            qpos[7:], qvel[6:], qpos[3:7], qvel[3:6], np.zeros(3, dtype=np.float32),
            scaffold_hint,
            [math.sin(2*math.pi*phase), math.cos(2*math.pi*phase)],
            self._cmd, body_linvel,
        ]).astype(np.float32)
    env._get_obs = types.MethodType(_get_obs, env)


def _build_inference(obs_size, act_size, deterministic=True):
    # Match the trained policy's architecture (v11+ uses paper-matching
    # 256/128/64; pre-v11 used Brax default 32/32/32/32).
    from scripts.pretrain_bc_jax import POLICY_HIDDEN_LAYERS
    networks = make_ppo_networks(
        observation_size=obs_size, action_size=act_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=POLICY_HIDDEN_LAYERS)
    if deterministic:
        @jax.jit
        def infer(norm_p, pol_p, obs, rng):
            logits = networks.policy_network.apply(norm_p, pol_p, obs)
            return networks.parametric_action_distribution.mode(logits)
    else:
        @jax.jit
        def infer(norm_p, pol_p, obs, rng):
            logits = networks.policy_network.apply(norm_p, pol_p, obs)
            return networks.parametric_action_distribution.sample(logits, rng)
    return infer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--action-space", type=str, default=CHAIN_ACTION_SPACE,
                   choices=["joint", "foot"])
    p.add_argument("--model", default=CHAIN_MODEL_PATH,
                   help="MJCF for inference / physics. Default: simple model.")
    p.add_argument("--render-model", default="models/phantomx.xml",
                   help="MJCF rendered into the video. Default: mesh model "
                        "for photo-realistic output. Set same as --model if "
                        "you want the simple primitive look.")
    p.add_argument("--demo", type=str, default="showcase",
                   choices=["showcase", "paper", "paper_stance", "legacy"],
                   help="'showcase' (v8+ default): full motion repertoire + "
                        "stance changes, ~95s, designed for videos. "
                        "Other options match watch_demo_jax.")
    p.add_argument("--out", required=True, help="output MP4 path")
    p.add_argument("--width",    type=int, default=1280)
    p.add_argument("--height",   type=int, default=720)
    p.add_argument("--fps",      type=int, default=50,
                   help="output video framerate (default 50)")
    p.add_argument("--duration", type=float, default=None,
                   help="seconds to record. Default: one full demo loop.")
    p.add_argument("--track-distance", type=float, default=1.5,
                   help="camera distance from bot (default 1.5m).")
    p.add_argument("--track-elevation", type=float, default=-25.0,
                   help="camera elevation angle in degrees, negative = "
                        "looking down (default -25°).")
    p.add_argument("--track-azimuth", type=float, default=135.0,
                   help="starting camera azimuth in degrees (default 135° = "
                        "behind-and-right of bot). With --orbit-period, this "
                        "is just the starting angle.")
    p.add_argument("--orbit-period", type=float, default=0.0,
                   help="seconds for one full camera orbit around the bot. "
                        "0 = no orbit (default). 60 = slow orbit (one full "
                        "revolution per minute). Negative for opposite "
                        "direction. Camera always tracks the bot's body, "
                        "orbit is in addition to that.")
    args = p.parse_args()

    # --- Load policy ---
    print(f"Loading params: {args.params}")
    with open(args.params, "rb") as f:
        loaded = pickle.load(f)
    if isinstance(loaded, tuple) and len(loaded) == 3:
        normalizer_params, policy_params, _ = loaded
    elif isinstance(loaded, tuple) and len(loaded) == 2:
        normalizer_params, policy_params = loaded
    else:
        raise SystemExit("Unrecognized params format.")

    # --- Env (no live viewer) ---
    env = HexapodEnv(stage=3, render_mode=None, model_path=args.model)
    env.gait_scale = 0.0
    _patched_obs(env)
    obs, _ = env.reset()

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    infer = _build_inference(obs_size, act_size, deterministic=True)
    rng = jax.random.PRNGKey(0)
    foot_residual_scale = float(env._ctrl.foot_residual_scale_max
                                if hasattr(env._ctrl, "foot_residual_scale_max")
                                else 0.020)

    # Settle.
    settle_speed = (env.SPEED_MIN_FRAC + 0.05) * env._ctrl.MAX_SPEED
    env._cmd = np.array([settle_speed, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(200):
        env._latest_feet_body = env._ctrl.compute_foot_targets(
            env._cmd, env._sim_time).astype(np.float32)
        obs, *_ = env.step(zero_action)

    # --- Render model (separate so we can use the mesh) ---
    render_mjmodel = mujoco.MjModel.from_xml_path(args.render_model)
    if render_mjmodel.nq != env._model.nq:
        raise SystemExit(
            f"render-model nq={render_mjmodel.nq} != model nq={env._model.nq}")
    # Bump the offscreen framebuffer size so mujoco.Renderer can render at
    # the requested resolution. Default offwidth/offheight is 640x480.
    render_mjmodel.vis.global_.offwidth  = args.width
    render_mjmodel.vis.global_.offheight = args.height
    render_mjdata = mujoco.MjData(render_mjmodel)
    render_mjdata.qpos[:] = env._data.qpos
    mujoco.mj_forward(render_mjmodel, render_mjdata)

    base_body_id = mujoco.mj_name2id(render_mjmodel, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_body_id < 0:
        # Fall back to body 1 (skip world body 0).
        base_body_id = 1
    print(f"Tracking body id: {base_body_id} "
          f"({mujoco.mj_id2name(render_mjmodel, mujoco.mjtObj.mjOBJ_BODY, base_body_id)})")

    # --- Renderer + camera setup ---
    renderer = mujoco.Renderer(render_mjmodel, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance  = args.track_distance
    camera.elevation = args.track_elevation
    camera.azimuth   = args.track_azimuth
    # lookat is updated each frame to track the bot.

    # --- Demo schedule ---
    demo_phases = {
        "showcase":      DEMO_PHASES_SHOWCASE,
        "paper":         DEMO_PHASES_PAPER,
        "paper_stance":  DEMO_PHASES_PAPER_STANCE,
        "legacy":        DEMO_PHASES,
    }[args.demo]
    total_loop_dur = sum(d for _, d, _ in demo_phases)
    duration = args.duration if args.duration is not None else total_loop_dur
    n_frames = int(duration * args.fps)
    # Frames-per-step: the env steps at 200Hz; we want 50 fps output by default.
    # So we render every 4th env.step into the video.
    env_steps_per_frame = max(1, int(round((1.0 / args.fps) / env._dt)))

    print(f"\nDuration:    {duration:.1f}s ({n_frames} frames @ {args.fps} fps)")
    print(f"Resolution:  {args.width}x{args.height}")
    print(f"Demo:        {args.demo} ({len(demo_phases)} phases, "
          f"loop = {total_loop_dur:.0f}s)")
    print(f"Output:      {args.out}\n")

    # --- ffmpeg writer ---
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=1)

    sim_t = 0.0
    last_phase_idx = -1
    t_render_start = time.time()
    try:
        for frame_idx in range(n_frames):
            # Multiple env steps per video frame.
            for _ in range(env_steps_per_frame):
                # Resolve cmd from demo schedule.
                t_in_cycle = sim_t % total_loop_dur
                cumulative = 0.0
                phase_idx = 0
                for i, (_, dur, _) in enumerate(demo_phases):
                    if t_in_cycle < cumulative + dur:
                        phase_idx = i
                        break
                    cumulative += dur
                label, dur, fn = demo_phases[phase_idx]
                t_in_phase = t_in_cycle - cumulative
                cmd = fn(t_in_phase, env._ctrl)
                if phase_idx != last_phase_idx:
                    last_phase_idx = phase_idx
                    print(f"  t={sim_t:6.1f}s  phase {phase_idx+1}/{len(demo_phases)}: "
                          f"{label}")

                env._cmd = cmd
                env._latest_feet_body = env._ctrl.compute_foot_targets(
                    env._cmd, env._sim_time).astype(np.float32)
                obs = env._get_obs()

                rng, sub = jax.random.split(rng)
                action_jax = infer(normalizer_params, policy_params,
                                   jnp.asarray(obs)[None, :], sub)[0]
                action = np.asarray(action_jax)
                if args.action_space == "foot":
                    foot_residual = action.reshape(6, 3) * foot_residual_scale
                    feet_body_target = env._latest_feet_body + foot_residual
                    joint_target = env._ctrl.body_to_joints(feet_body_target)
                    action = (joint_target - env._ctrl.gait_neutral_pose) / env.RESIDUAL_SCALE_MAX
                    action = np.clip(action, -1.0, 1.0).astype(np.float32)

                obs, *_ = env.step(action)
                sim_t += env._dt

            # Mirror state into the render model.
            render_mjdata.qpos[:] = env._data.qpos
            render_mjdata.qvel[:] = env._data.qvel
            mujoco.mj_forward(render_mjmodel, render_mjdata)

            # Update tracking camera lookat → bot's body position.
            camera.lookat[:] = render_mjdata.xpos[base_body_id]
            # Optional orbit: advance azimuth over wall-time of the video.
            if args.orbit_period != 0.0:
                video_t = frame_idx / args.fps
                degrees_per_sec = 360.0 / args.orbit_period
                camera.azimuth = args.track_azimuth + degrees_per_sec * video_t

            renderer.update_scene(render_mjdata, camera=camera)
            frame = renderer.render()
            writer.append_data(frame)

            if frame_idx % args.fps == 0:
                wall_elapsed = time.time() - t_render_start
                progress = (frame_idx + 1) / n_frames * 100
                print(f"  rendering... {progress:5.1f}%  "
                      f"({wall_elapsed:.1f}s elapsed)")
    finally:
        writer.close()

    wall_total = time.time() - t_render_start
    sz_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\nDone. {n_frames} frames rendered in {wall_total:.1f}s.")
    print(f"Output: {args.out}  ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
