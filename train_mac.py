"""
train_mac.py — PPO training entry point, macOS variant.

Identical to train.py except:
  * Imports from envs.hexapod_env_mac (which writes env-0 state to a shared
    memory region instead of trying to launch_passive() inside the worker).
  * WATCH_LIVE controls the shm publishing — view it with `mjpython live_viewer.py`.
  * Default config tuned for short Mac-side experimental runs (variable —
    edit RUN_NAME / TOTAL_STEPS / schedule constants below as needed).

For Linux workstations use train.py + envs.hexapod_env (in-worker viewer
works fine there). The two trees share gait/, simple_gait.py, demo.py,
pilot.py, watch.py, watch_demo.py — only env + train differ per platform.
"""

import argparse
import atexit
import glob
import os
import re
import subprocess
import sys
import time
import webbrowser
from collections import deque
from functools import partial

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from envs.hexapod_env_mac import HexapodEnv


# ============================================================================
# CONFIG
# ============================================================================
RUN_NAME    = "mac_v6_full_motion_refine"
LOG_DIR     = f"logs/hexapod_{RUN_NAME}"
CKPT_DIR    = f"checkpoints/hexapod_{RUN_NAME}"
N_ENVS      = 32
# v6: full-motion (stage=3) RL refinement of BC-v2 policy. The constant-
# failure issue from earlier v5 attempts was diagnosed as BC's log_std=1.0
# at init causing unbounded stochastic noise; fixed by --bc-init now
# resetting log_std=-2.0 (std≈0.135). All commanded motion is enabled
# (vx, vy, wz, pitch, roll, height, width); shifts remain disabled (no
# programmed gait). Pitch sampling kept narrow (±5°) due to the gait-
# controller pitch sign bug — see env's SAMPLE_RANGES comment.
# 50M is a refinement budget; early-stop will halt sooner if reward
# plateaus.
TOTAL_STEPS = 50_000_000

# Curriculum start point. INITIAL_STAGE=3 enables all 9 cmd slots (with
# shift slots 7,8 zeroed by the stage-3 mask in env). Matches what BC v2
# was trained on. SKIP_SCAFFOLD=True forces gait_scale=0.0 throughout
# because the BC walker already produces walking trajectories — no need
# for the scaffold-fade curriculum.
INITIAL_STAGE   = 3
SKIP_SCAFFOLD   = True

# Live observability — both default ON.
#   WATCH_LIVE: env-0 publishes its qpos+qvel to shared memory every step.
#               Run `mjpython live_viewer.py` in another terminal to watch
#               training in real time. Negligible throughput cost (just one
#               numpy copy per env-0 step). Cross-platform: works on macOS
#               where the broken approach of opening the viewer inside the
#               worker process can't.
#   AUTO_TB:    auto-launches `tensorboard --logdir LOG_DIR` as a subprocess
#               and opens it in the default browser when training begins.
WATCH_LIVE = True     # publishes env-0 state to shm; mjpython live_viewer.py renders
AUTO_TB    = True
TB_PORT    = 6006
# Open the TB URL in the default browser when training starts. Default False
# now — user typically already has the TB tab open in their browser, so popping
# a new one each run is just a nuisance. Flip to True for first-time setup.
AUTO_OPEN_BROWSER = False

# Episode max length (steps). With dt=0.005s, 2000 steps = 10s of sim time.
# Episodes that hit this without falling are considered "successful walks."
EPISODE_MAX_STEPS = 2000

# Per-stage minimum step count. For this single-stage Mac run we set min just
# above TOTAL_STEPS so the auto-advance gate is never reached.
STAGE_MIN_STEPS = {
    1:   3_000_000,
    2:   3_000_000,
    3:  51_000_000,  # > TOTAL_STEPS (50M) so v6 stage 3 never advances to 4
                     # (stage 4 drops body_linvel from obs which would mismatch BC training)
    4:           0,
}

# Per-stage gait_scale schedule: (start, end). Fades linearly between
# STAGE_FADE_RANGE fractions of the stage's nominal duration (== STAGE_MIN_STEPS).
# For this run: 0–10M scaffold-full (gait_scale=1.0), 10M–70M linear fade to 0.0,
# 70M–150M autonomous (gait_scale=0.0). Fractions of the 151M nominal:
#    10/151 ≈ 0.066  fade start
#    70/151 ≈ 0.464  fade end
# 60M-step fade is 2× the previous run — gives the policy more time to absorb
# each marginal scaffold reduction without falling into cheating local optima.
STAGE_GAIT_SCALE = {
    1: (1.0, 0.0),   # full scaffold → no scaffold within stage 1
    2: (1.0, 0.6),
    3: (0.6, 0.0),
    4: (0.0, 0.0),
}
STAGE_FADE_RANGE = (10.0/151.0, 70.0/151.0)   # fade 10M → 70M absolute steps

# Stage advancement gates — both must pass simultaneously over recent episodes.
# Loose thresholds (per design discussion) so we move on quickly; later stages
# keep refining what stage 1 started.
ADVANCE_TRACKING_THRESHOLD = 0.65   # avg per-step tracking reward (max 1.0)
ADVANCE_FALL_RATE_MAX      = 0.30   # fraction of episodes that ended in fall
ADVANCE_WINDOW             = 200    # rolling window of last N episodes

# Episode length below this counts as a "fall." Above counts as a success.
# At dt=0.005s × 500 steps = 2.5s — enough time to evaluate basic stability.
FALL_LENGTH_THRESHOLD = 500

# Early stopping — halt training automatically when the rolling tracking
# reward stops improving in the autonomous phase. Only active once gait_scale
# has fully faded to 0.0; until then, training proceeds regardless. Prevents
# wasted compute past convergence and is safer than just trusting TOTAL_STEPS.
EARLY_STOP_ENABLED        = True
EARLY_STOP_WARMUP_STEPS   = 3_000_000  # autonomous warmup before plateau check
EARLY_STOP_PATIENCE_STEPS = 3_000_000  # no improvement for this long → stop
EARLY_STOP_DELTA          = 0.005      # min tracking-reward improvement to count
EARLY_STOP_MIN_REWARD     = 0.50       # must reach this baseline before stop allowed
                                        # (prevents stopping while still learning poorly)


# ============================================================================
# CALLBACKS
# ============================================================================
class StageManagerCallback(BaseCallback):
    """Owns curriculum stage progression + within-stage gait_scale fade.

    Reads tracking_reward and episode-length info from each terminated episode,
    maintains a rolling window, and advances the env's stage attribute when
    gates pass. Logs current stage + fade weight to TensorBoard.
    """

    def __init__(self, n_envs, initial_stage=1, skip_scaffold=False,
                 is_resume=False, verbose=0):
        super().__init__(verbose)
        self.n_envs           = n_envs
        self.current_stage    = initial_stage
        self.stage_start_step = 0
        # If True: gait_scale=0.0 from step 0, no fade. Used when the policy
        # is already a working walker (BC pretrained or previously trained).
        self.skip_scaffold    = skip_scaffold
        self.episode_tracking    = deque(maxlen=ADVANCE_WINDOW)
        self.episode_lengths     = deque(maxlen=ADVANCE_WINDOW)
        self.episode_no_progress = deque(maxlen=ADVANCE_WINDOW)
        self.episode_fell        = deque(maxlen=ADVANCE_WINDOW)
        # Per-env scratch — accumulates current episode's stats.
        self._track_sums  = None
        self._step_counts = None
        # Early-stopping state.
        self._autonomous_start  = None
        self._best_reward       = -float("inf")
        self._best_reward_step  = 0
        # Resume mode: skip scaffold-fade (already past it), gait_scale=0.0
        # throughout this invocation, fresh early-stop warmup window from
        # the resume point.
        self.is_resume = is_resume

    def _on_training_start(self):
        self._track_sums  = np.zeros(self.n_envs, dtype=np.float64)
        self._step_counts = np.zeros(self.n_envs, dtype=np.int64)
        # CRITICAL: must use env_method (not set_attr) — see the note in
        # HexapodEnv.__init__ next to `self.gait_scale = 1.0`. set_attr only
        # sets the attribute on the outer Monitor wrapper; the inner
        # HexapodEnv never sees the update.
        self.training_env.env_method("set_stage", self.current_stage)
        if self.is_resume or self.skip_scaffold:
            # Skip scaffold/fade entirely. Used either when resuming a
            # post-fade checkpoint or when starting fresh from BC pretrained
            # weights that already produce walking (no need to bootstrap via
            # scaffold).
            self._autonomous_start = self.num_timesteps
            self.training_env.env_method("set_gait_scale", 0.0)
            mode = "RESUME" if self.is_resume else "SKIP-SCAFFOLD"
            print(f"[stage] {mode} at step {self.num_timesteps:,} (stage "
                  f"{self.current_stage}) — gait_scale held at 0.0; fresh "
                  f"early-stop warmup window of {EARLY_STOP_WARMUP_STEPS:,} steps")
        else:
            gs = STAGE_GAIT_SCALE[self.current_stage][0]
            self.training_env.env_method("set_gait_scale", gs)
            print(f"[stage] starting stage {self.current_stage} at step 0  "
                  f"(gait_scale={gs:.2f})")

    def _on_step(self):
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        # Update per-env episode accumulators.
        for i in range(self.n_envs):
            info = infos[i]
            self._track_sums[i]  += info.get("tracking_reward", 0.0)
            self._step_counts[i] += 1
            if dones[i]:
                avg_track = self._track_sums[i] / max(1, self._step_counts[i])
                self.episode_tracking.append(float(avg_track))
                self.episode_lengths.append(int(self._step_counts[i]))
                # Termination cause: info["no_progress"] / info["fell"] are
                # set inside HexapodEnv.step. The wrapped Monitor's terminal
                # info propagates through SB3's VecEnv as info["terminal_observation"]
                # alongside the raw info dict; the keys are still readable.
                self.episode_no_progress.append(bool(info.get("no_progress", False)))
                self.episode_fell.append(bool(info.get("fell", False)))
                self._track_sums[i]  = 0.0
                self._step_counts[i] = 0

        # Compute and apply gait_scale for current stage. Must use env_method
        # (set_attr would only mutate the Monitor wrapper, not the inner env).
        gait_scale = self._current_gait_scale()
        self.training_env.env_method("set_gait_scale", gait_scale)

        # Periodically log progress and check advance gate.
        if self.num_timesteps % 50_000 == 0:
            self._log_progress(gait_scale)
        if self.current_stage < 4:
            self._maybe_advance_stage()

        # Early stopping — only active once we've fully entered the autonomous
        # phase (gait_scale == 0.0). Returns False to halt training.
        if EARLY_STOP_ENABLED and not self._check_early_stop(gait_scale):
            return False

        return True

    def _check_early_stop(self, gait_scale):
        """Return False (halt training) if reward has plateaued in the autonomous
        phase past the patience window. Otherwise True (keep training)."""
        # Enter autonomous tracking once gait_scale first hits 0.
        if self._autonomous_start is None:
            if gait_scale <= 1e-9:
                self._autonomous_start = self.num_timesteps
                print(f"[early-stop] autonomous phase started at step "
                      f"{self.num_timesteps:,}; plateau check active after warmup "
                      f"of {EARLY_STOP_WARMUP_STEPS:,} steps")
            return True

        steps_since_auto = self.num_timesteps - self._autonomous_start
        if steps_since_auto < EARLY_STOP_WARMUP_STEPS:
            return True
        if len(self.episode_tracking) < ADVANCE_WINDOW:
            return True

        cur_mean = float(np.mean(self.episode_tracking))
        # Track best-seen rolling mean.
        if cur_mean > self._best_reward + EARLY_STOP_DELTA:
            self._best_reward      = cur_mean
            self._best_reward_step = self.num_timesteps
        # Don't stop until baseline performance is reached.
        if cur_mean < EARLY_STOP_MIN_REWARD:
            return True
        # Plateau check.
        steps_since_best = self.num_timesteps - self._best_reward_step
        if steps_since_best > EARLY_STOP_PATIENCE_STEPS:
            print(f"\n[early-stop] tracking reward plateaued at {cur_mean:.3f} "
                  f"(best {self._best_reward:.3f} at step {self._best_reward_step:,}; "
                  f"no improvement for {steps_since_best:,} steps).")
            print(f"[early-stop] halting training at step {self.num_timesteps:,}.")
            return False

        return True

    def _current_gait_scale(self):
        if self.is_resume or self.skip_scaffold:
            return 0.0   # always pure-policy autonomous; no fade
        nominal = STAGE_MIN_STEPS[self.current_stage]
        if nominal <= 0:
            return STAGE_GAIT_SCALE[self.current_stage][1]
        progress = (self.num_timesteps - self.stage_start_step) / nominal
        progress = max(0.0, min(progress, 1.0))
        gs_start, gs_end = STAGE_GAIT_SCALE[self.current_stage]
        f_start, f_end   = STAGE_FADE_RANGE
        if progress < f_start:
            return gs_start
        if progress > f_end:
            return gs_end
        t = (progress - f_start) / (f_end - f_start)
        return gs_start + t * (gs_end - gs_start)

    def _failure_rate(self):
        """Fraction of recent episodes that failed (fell OR no-progress)."""
        if not self.episode_fell:
            return 0.0
        fail_flags = [1.0 if (f or np) else 0.0
                      for f, np in zip(self.episode_fell, self.episode_no_progress)]
        return float(np.mean(fail_flags))

    def _maybe_advance_stage(self):
        if len(self.episode_tracking) < ADVANCE_WINDOW:
            return
        if (self.num_timesteps - self.stage_start_step) < STAGE_MIN_STEPS[self.current_stage]:
            return
        avg_track = float(np.mean(self.episode_tracking))
        fail_rate = self._failure_rate()
        # Use combined fail_rate (fell + no_progress) — was just length-based,
        # which counted standing-still-then-truncated as a successful episode.
        if (avg_track >= ADVANCE_TRACKING_THRESHOLD
                and fail_rate <= ADVANCE_FALL_RATE_MAX):
            self._advance_stage(avg_track, fail_rate)

    def _advance_stage(self, avg_track, fail_rate):
        prev = self.current_stage
        self.current_stage   += 1
        self.stage_start_step = self.num_timesteps
        self.episode_tracking.clear()
        self.episode_lengths.clear()
        self.episode_no_progress.clear()
        self.episode_fell.clear()
        self.training_env.env_method("set_stage", self.current_stage)
        self.training_env.env_method("set_gait_scale", STAGE_GAIT_SCALE[self.current_stage][0])
        print(f"[stage] ADVANCED {prev} → {self.current_stage} at step "
              f"{self.num_timesteps:,}  (avg_track={avg_track:.3f}, "
              f"fail_rate={fail_rate:.2%})")

    def _log_progress(self, gait_scale):
        avg_track = float(np.mean(self.episode_tracking)) if self.episode_tracking else 0.0
        if self.episode_lengths:
            mean_len   = float(np.mean(self.episode_lengths))
            fell_rate  = (float(np.mean(self.episode_fell))
                          if self.episode_fell else 0.0)
            stuck_rate = (float(np.mean(self.episode_no_progress))
                          if self.episode_no_progress else 0.0)
            fail_rate  = self._failure_rate()
        else:
            mean_len = fell_rate = stuck_rate = fail_rate = 0.0
        self.logger.record("stage/current",        self.current_stage)
        self.logger.record("stage/gait_scale",     gait_scale)
        self.logger.record("stage/avg_tracking",   avg_track)
        self.logger.record("stage/fall_rate",      fell_rate)
        self.logger.record("stage/no_progress_rate", stuck_rate)
        self.logger.record("stage/fail_rate",      fail_rate)
        self.logger.record("stage/mean_ep_length", mean_len)


# ============================================================================
# ENV FACTORY (per-index — env 0 optionally renders for live watching)
# ============================================================================
def _make_env(idx, watch_live):
    """Top-level factory so it pickles cleanly for SubprocVecEnv on Windows.
    Env 0 gets `live_watch=True` (side-channel viewer; doesn't change
    render_mode so SB3's VecEnv assertion stays happy).
    Stage = INITIAL_STAGE so the env's _sample_cmd uses the right cmd mask
    from step 0 (translation only at stage=1, full motion at stage=3)."""
    kwargs = {"stage": INITIAL_STAGE, "live_watch": (idx == 0 and watch_live)}
    env = HexapodEnv(**kwargs)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=EPISODE_MAX_STEPS)
    env = Monitor(env)
    return env


# ============================================================================
# TENSORBOARD AUTO-LAUNCH
# ============================================================================
def launch_tensorboard(logdir, port=6006, open_browser=True):
    """Start TensorBoard as a child subprocess, register cleanup on exit,
    and open it in the default browser. Returns the Popen handle."""
    cmd = [sys.executable, "-m", "tensorboard.main",
           "--logdir", logdir, "--port", str(port), "--bind_all"]
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    atexit.register(lambda: proc.terminate())
    print(f"[tensorboard] launched at http://localhost:{port}  (pid {proc.pid})")
    if open_browser:
        # Brief wait so the server is bound before the browser hits it.
        time.sleep(2.5)
        webbrowser.open(f"http://localhost:{port}")
    return proc


# ============================================================================
# MAIN
# ============================================================================
def _latest_ckpt_in(run_dir):
    """Newest *complete* checkpoint in run_dir (skips files modified <3s ago)."""
    if not os.path.isdir(run_dir):
        return None
    final = os.path.join(run_dir, "final.zip")
    if os.path.exists(final) and time.time() - os.path.getmtime(final) > 3.0:
        return final[:-4]
    step_re = re.compile(r"_(\d+)_steps\.zip$")
    cands = []
    for f in os.listdir(run_dir):
        m = step_re.search(f)
        if not m:
            continue
        path = os.path.join(run_dir, f)
        if time.time() - os.path.getmtime(path) < 3.0:
            continue
        cands.append((int(m.group(1)), path[:-4]))
    return None if not cands else sorted(cands)[-1][1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training (Mac variant).")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume from checkpoint. With value: explicit path "
                             "(without .zip). Without: latest in current CKPT_DIR. "
                             "Resume mode skips the scaffold-fade and trains pure-"
                             "policy (gait_scale=0.0) for TOTAL_STEPS more steps.")
    parser.add_argument("--bc-init", default=None,
                        help="Path (without .zip) to a BC-pretrained PPO checkpoint "
                             "produced by pretrain_bc.py. Loads the policy weights as "
                             "an initialization for fresh RL training (NOT a resume — "
                             "num_timesteps starts at 0, full curriculum runs).")
    parser.add_argument("--log-std-init", type=float, default=None,
                        help="Override the policy's log_std parameter at training "
                             "start. Controls exploration noise. Lower values = "
                             "less noise = more refinement-like training. "
                             "Suggested values: -2.0 (std≈0.135, BC-init default), "
                             "-3.0 (std≈0.05, refinement), -4.0 (std≈0.018, polish). "
                             "Applied AFTER --bc-init / --resume weight load.")
    args = parser.parse_args()

    # Resolve resume checkpoint.
    resume_path = None
    if args.resume == "auto":
        resume_path = _latest_ckpt_in(CKPT_DIR)
        if resume_path is None:
            print(f"[resume] no checkpoint found in {CKPT_DIR}; starting fresh.")
        else:
            print(f"[resume] latest in {CKPT_DIR}: {resume_path}")
    elif args.resume:
        resume_path = args.resume
        print(f"[resume] using explicit checkpoint: {resume_path}")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    if AUTO_TB:
        launch_tensorboard(LOG_DIR, port=TB_PORT, open_browser=AUTO_OPEN_BROWSER)

    env_fns = [partial(_make_env, i, WATCH_LIVE) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    if WATCH_LIVE:
        print(f"[viewer] env 0 will open a MuJoCo viewer window when training starts.")

    if resume_path:
        # Load existing policy + value network from checkpoint and continue
        # training. SB3's PPO.load preserves optimizer state; learn(reset_num_
        # timesteps=False) keeps num_timesteps continuous so checkpoint files
        # are named correctly (steps continue from where they left off).
        model = PPO.load(resume_path, env=vec_env,
                         tensorboard_log=LOG_DIR, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device="cpu",
        )
        # BC-init: load weights from a behavioral-cloning pretrained checkpoint.
        # Treats this as a fresh RL run (num_timesteps=0, full curriculum) but
        # with a network that already produces walking trajectories. Loading
        # only the policy state_dict avoids inheriting the BC stage's optimizer
        # state, which would interfere with PPO's clip-range schedule.
        if args.bc_init:
            print(f"[bc-init] loading pretrained policy weights from {args.bc_init}.zip")
            bc_model = PPO.load(args.bc_init, device="cpu")
            model.policy.load_state_dict(bc_model.policy.state_dict())
            # CRITICAL: BC pretraining only trains the policy mean (MSE loss
            # on dist.distribution.mean). The log_std parameter stays at PPO's
            # default initialization (0.0 → std=1.0), so stochastic rollouts
            # would sample actions with noise as large as the entire action
            # range. We default to -2.0 (std≈0.135) which is the largest noise
            # that doesn't destabilize the BC-learned gait. For refinement
            # with even less disturbance, override via --log-std-init -3.0
            # or lower.
            import torch, math
            default_bc_init_log_std = -2.0
            with torch.no_grad():
                model.policy.log_std.data.fill_(default_bc_init_log_std)
            del bc_model
            print(f"[bc-init] policy initialized from BC pretraining; "
                  f"log_std reset to {default_bc_init_log_std} "
                  f"(std≈{math.exp(default_bc_init_log_std):.3f}) for stable rollouts.")

    # Final log_std override — applies after --bc-init's default OR --resume's
    # inherited value OR a fresh PPO's init. Gives an explicit knob for
    # tuning exploration noise without editing source.
    if args.log_std_init is not None:
        import torch, math
        with torch.no_grad():
            model.policy.log_std.data.fill_(float(args.log_std_init))
        print(f"[log_std] forced to {args.log_std_init} "
              f"(std≈{math.exp(args.log_std_init):.4f}) via --log-std-init flag")

    callbacks = CallbackList([
        StageManagerCallback(n_envs=N_ENVS,
                             initial_stage=INITIAL_STAGE,
                             skip_scaffold=SKIP_SCAFFOLD,
                             is_resume=bool(resume_path),
                             verbose=1),
        CheckpointCallback(
            save_freq=2_000_000 // N_ENVS,
            save_path=CKPT_DIR,
            name_prefix=f"hexapod_{RUN_NAME}",
        ),
    ])

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callbacks,
        reset_num_timesteps=(resume_path is None),
        progress_bar=True,
    )

    model.save(os.path.join(CKPT_DIR, "final"))
    print(f"\nDone. Model saved to {CKPT_DIR}")
    vec_env.close()
