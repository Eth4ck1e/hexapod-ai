"""
PPO training entry point — curriculum + adaptive scaffold fade.

Trains a residual policy on top of the analytical gait scaffold (gait.Controller)
across four curriculum stages. Within each stage, gait_scale fades from the
stage's starting weight toward the next stage's weight; once the policy hits
loose tracking + fall-rate gates AND the stage's minimum step count, the env
auto-advances to the next stage.

Stages:
  1: translation (cmd vx, vy)                                — gait_scale 1.0 → 1.0
  2: + yaw, height, width                                    — gait_scale 1.0 → 0.6
  3: + pitch, roll, body shifts                              — gait_scale 0.6 → 0.0
  4: scaffold gone, novelty bonuses, body_linvel obs dropped — gait_scale 0.0

See project_gait_training_architecture.md for the full design rationale.
"""

import os
from collections import deque

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from envs.hexapod_env import HexapodEnv


# ============================================================================
# CONFIG
# ============================================================================
RUN_NAME    = "v2_curriculum"
LOG_DIR     = f"logs/hexapod_{RUN_NAME}"
CKPT_DIR    = f"checkpoints/hexapod_{RUN_NAME}"
N_ENVS      = 16
TOTAL_STEPS = 200_000_000           # ~50M nominal per stage × 4

# Episode max length (steps). With dt=0.005s, 2000 steps = 10s of sim time.
# Episodes that hit this without falling are considered "successful walks."
EPISODE_MAX_STEPS = 2000

# Per-stage minimum step count before checking the advancement gates.
# Set to 0 to advance immediately as soon as gates pass.
STAGE_MIN_STEPS = {
    1: 40_000_000,   # 40M to learn full-360° translation from scratch
    2: 30_000_000,   # 30M to add yaw + stance
    3: 30_000_000,   # 30M to add pose + shifts
    4:           0,  # stage 4 is open-ended (no auto-advance, just train forever)
}

# Per-stage gait_scale schedule: (start, end). Fades linearly between
# STAGE_FADE_RANGE fractions of the stage's nominal duration.
STAGE_GAIT_SCALE = {
    1: (1.0, 1.0),   # hold scaffold strong — give policy a reliable teacher
    2: (1.0, 0.6),
    3: (0.6, 0.0),
    4: (0.0, 0.0),
}
STAGE_FADE_RANGE = (0.3, 0.9)   # fade between 30% and 90% of stage duration

# Stage advancement gates — both must pass simultaneously over recent episodes.
# Loose thresholds (per design discussion) so we move on quickly; later stages
# keep refining what stage 1 started.
ADVANCE_TRACKING_THRESHOLD = 0.65   # avg per-step tracking reward (max 1.0)
ADVANCE_FALL_RATE_MAX      = 0.30   # fraction of episodes that ended in fall
ADVANCE_WINDOW             = 200    # rolling window of last N episodes

# Episode length below this counts as a "fall." Above counts as a success.
# At dt=0.005s × 500 steps = 2.5s — enough time to evaluate basic stability.
FALL_LENGTH_THRESHOLD = 500


# ============================================================================
# CALLBACKS
# ============================================================================
class StageManagerCallback(BaseCallback):
    """Owns curriculum stage progression + within-stage gait_scale fade.

    Reads tracking_reward and episode-length info from each terminated episode,
    maintains a rolling window, and advances the env's stage attribute when
    gates pass. Logs current stage + fade weight to TensorBoard.
    """

    def __init__(self, n_envs, verbose=0):
        super().__init__(verbose)
        self.n_envs           = n_envs
        self.current_stage    = 1
        self.stage_start_step = 0
        self.episode_tracking = deque(maxlen=ADVANCE_WINDOW)
        self.episode_lengths  = deque(maxlen=ADVANCE_WINDOW)
        # Per-env scratch — accumulates current episode's stats.
        self._track_sums  = None
        self._step_counts = None

    def _on_training_start(self):
        self._track_sums  = np.zeros(self.n_envs, dtype=np.float64)
        self._step_counts = np.zeros(self.n_envs, dtype=np.int64)
        self.training_env.set_attr("stage", self.current_stage)
        self.training_env.set_attr("gait_scale", STAGE_GAIT_SCALE[self.current_stage][0])
        print(f"[stage] starting stage {self.current_stage} at step 0  "
              f"(gait_scale={STAGE_GAIT_SCALE[self.current_stage][0]:.2f})")

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
                self._track_sums[i]  = 0.0
                self._step_counts[i] = 0

        # Compute and apply gait_scale for current stage.
        gait_scale = self._current_gait_scale()
        self.training_env.set_attr("gait_scale", gait_scale)

        # Periodically log progress and check advance gate.
        if self.num_timesteps % 50_000 == 0:
            self._log_progress(gait_scale)
        if self.current_stage < 4:
            self._maybe_advance_stage()

        return True

    def _current_gait_scale(self):
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

    def _maybe_advance_stage(self):
        if len(self.episode_tracking) < ADVANCE_WINDOW:
            return
        if (self.num_timesteps - self.stage_start_step) < STAGE_MIN_STEPS[self.current_stage]:
            return
        avg_track = float(np.mean(self.episode_tracking))
        fall_rate = float(np.mean([1.0 if l < FALL_LENGTH_THRESHOLD else 0.0
                                   for l in self.episode_lengths]))
        if (avg_track >= ADVANCE_TRACKING_THRESHOLD
                and fall_rate <= ADVANCE_FALL_RATE_MAX):
            self._advance_stage(avg_track, fall_rate)

    def _advance_stage(self, avg_track, fall_rate):
        prev = self.current_stage
        self.current_stage   += 1
        self.stage_start_step = self.num_timesteps
        self.episode_tracking.clear()
        self.episode_lengths.clear()
        self.training_env.set_attr("stage", self.current_stage)
        self.training_env.set_attr("gait_scale", STAGE_GAIT_SCALE[self.current_stage][0])
        print(f"[stage] ADVANCED {prev} → {self.current_stage} at step "
              f"{self.num_timesteps:,}  (avg_track={avg_track:.3f}, "
              f"fall_rate={fall_rate:.2%})")

    def _log_progress(self, gait_scale):
        avg_track = float(np.mean(self.episode_tracking)) if self.episode_tracking else 0.0
        if self.episode_lengths:
            fall_rate = float(np.mean([1.0 if l < FALL_LENGTH_THRESHOLD else 0.0
                                       for l in self.episode_lengths]))
            mean_len = float(np.mean(self.episode_lengths))
        else:
            fall_rate = 0.0
            mean_len  = 0.0
        self.logger.record("stage/current",       self.current_stage)
        self.logger.record("stage/gait_scale",    gait_scale)
        self.logger.record("stage/avg_tracking",  avg_track)
        self.logger.record("stage/fall_rate",     fall_rate)
        self.logger.record("stage/mean_ep_length", mean_len)


# ============================================================================
# ENV FACTORY
# ============================================================================
def make_env():
    env = HexapodEnv(stage=1)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=EPISODE_MAX_STEPS)
    return env


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    vec_env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

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

    callbacks = CallbackList([
        StageManagerCallback(n_envs=N_ENVS, verbose=1),
        CheckpointCallback(
            save_freq=2_000_000 // N_ENVS,
            save_path=CKPT_DIR,
            name_prefix=f"hexapod_{RUN_NAME}",
        ),
    ])

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(os.path.join(CKPT_DIR, "final"))
    print(f"\nDone. Model saved to {CKPT_DIR}")
    vec_env.close()
