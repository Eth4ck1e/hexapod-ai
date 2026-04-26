import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from envs.hexapod_env import HexapodEnv

STAGE = 1
RUN_NAME = "stage1_long"        # 24h run on top of v4 (body-frame cmd) reward fn
LOG_DIR = f"logs/hexapod_{RUN_NAME}"
CKPT_DIR = f"checkpoints/hexapod_{RUN_NAME}"
N_ENVS = 16
TOTAL_STEPS = 400_000_000       # ~22 hrs on CPU at ~5k SPS (2 hr margin)

# Gait fade schedule (fractions of TOTAL_STEPS):
GAIT_FADE_START = 0.25          # hold gait_scale=1.0 until 25% through training
GAIT_FADE_END   = 0.75          # gait_scale=0.0 from 75% onwards


class GaitFadeCallback(BaseCallback):
    """Linearly fade env.gait_scale from 1.0 -> 0.0 across the middle of training."""
    def __init__(self, total_steps, fade_start_frac, fade_end_frac, verbose=0):
        super().__init__(verbose)
        self.total_steps     = total_steps
        self.fade_start_frac = fade_start_frac
        self.fade_end_frac   = fade_end_frac
        self._last_logged    = None

    def _scale_for_progress(self, progress):
        if progress < self.fade_start_frac:
            return 1.0
        if progress >= self.fade_end_frac:
            return 0.0
        span = self.fade_end_frac - self.fade_start_frac
        return 1.0 - (progress - self.fade_start_frac) / span

    def _on_rollout_start(self):
        progress = self.num_timesteps / self.total_steps
        scale    = self._scale_for_progress(progress)
        self.training_env.set_attr("gait_scale", scale)
        # Log on every 5% milestone for terminal visibility.
        milestone = round(progress * 20) / 20
        if milestone != self._last_logged:
            self._last_logged = milestone
            print(f"[gait_fade] progress={progress*100:5.1f}%  gait_scale={scale:.3f}")

    def _on_step(self):
        return True


def make_env():
    return HexapodEnv(stage=STAGE)


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

    checkpoint_cb = CheckpointCallback(
        save_freq=2_000_000 // N_ENVS,    # one checkpoint every ~2M total steps
        save_path=CKPT_DIR,
        name_prefix=f"hexapod_{RUN_NAME}",
    )
    gait_fade_cb = GaitFadeCallback(
        total_steps=TOTAL_STEPS,
        fade_start_frac=GAIT_FADE_START,
        fade_end_frac=GAIT_FADE_END,
        verbose=1,
    )

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=CallbackList([checkpoint_cb, gait_fade_cb]),
        progress_bar=True,
    )

    model.save(os.path.join(CKPT_DIR, "final"))
    print("Done. Model saved to", CKPT_DIR)
    vec_env.close()
