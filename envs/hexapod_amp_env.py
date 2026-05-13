"""
HexapodAMPEnv — Brax env wrapper that augments the task reward with
an AMP (Adversarial Motion Priors) style reward.

The discriminator is held as FROZEN params at env construction time.
Every step computes the AMP state vector for the new mjx_data, builds
a (s_t, s_t+1) transition with the prev state stored in `state.info`,
and adds `style_w * style_reward(D(transition))` to the env's reward.

Discriminator UPDATES happen between PPO training segments by the
training script — it instantiates a new HexapodAMPEnv with updated
params, paying one JIT recompile but nothing per step.

This is the "alternating" AMP training scheme: PPO runs many steps
with frozen D, then we update D, then PPO again with new D. Less
frequent D updates than the paper's per-iteration scheme but much
simpler to implement on top of stock brax PPO.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from brax.envs.base import State

from envs.hexapod_brax_env import (
    HexapodBraxEnv,
    _initial_brax_state,
    _scalar_metrics,
)
from envs import hexapod_env_jax as hex_jax
from envs.cmd_bins import cmd_to_bin, N_BINS
from amp.discriminator import (
    Discriminator, MultiHeadDiscriminator,
    style_reward, multihead_style_reward,
    STATE_DIM, cmd_for_disc,
)


def _extract_amp_state(params: hex_jax.EnvParams,
                       mjx_data) -> jnp.ndarray:
    """Per-step AMP state vector (49 dims). Must match amp/prior_data.py:
        joint_pos (18) + joint_vel (18) + body_linvel_body (3)
        + body_angvel (3) + body_height (1) + foot_heights (6)
    """
    qpos = mjx_data.qpos
    qvel = mjx_data.qvel

    joint_pos = qpos[7:25]
    joint_vel = qvel[6:24]
    body_linvel_body = hex_jax._body_frame_linvel(qpos, qvel)
    body_angvel = qvel[3:6]
    body_height = qpos[2:3]

    tibia_ids = params.tibia_body_ids
    tibia_pos = mjx_data.xpos[tibia_ids]
    tibia_R   = mjx_data.xmat[tibia_ids].reshape(6, 3, 3)
    foot_world = tibia_pos + jnp.einsum('lij,j->li', tibia_R,
                                        params.foot_local_offset)
    body_pos = mjx_data.xpos[params.base_body_id]
    body_R   = mjx_data.xmat[params.base_body_id].reshape(3, 3)
    foot_body_z = ((foot_world - body_pos) @ body_R)[:, 2]

    return jnp.concatenate([
        joint_pos, joint_vel,
        body_linvel_body, body_angvel,
        body_height, foot_body_z,
    ])


class HexapodAMPEnv(HexapodBraxEnv):
    """HexapodBraxEnv + AMP style reward injection.

    Constructor args (in addition to the base):
      discriminator_params: flax params for the AMP discriminator (frozen)
      style_weight: λ_style — multiplier on the style reward when adding to
                    the task reward. Paper uses ~0.5 as a starting point.
      discriminator_hidden: must match the params' shape (default
                            matches `Discriminator()`'s default).
    """

    def __init__(self,
                 model_path: str,
                 discriminator_params,
                 *,
                 gait_scale: float = 0.0,
                 cmd_mask: str = "stage3",
                 action_space: str = "joint",
                 style_weight: float = 0.5,
                 discriminator_hidden=(1024, 512),
                 cmd_dynamics_enabled: bool = False,
                 scaffold_wz_trim_vx: float = 0.0,
                 scaffold_wz_trim_vy_abs: float = 0.0,
                 disturbance_enabled: bool = False,
                 disturbance_magnitude_scale: float = 1.0,
                 multihead_disc: bool = False):
        super().__init__(model_path, gait_scale=gait_scale,
                         cmd_mask=cmd_mask, action_space=action_space,
                         cmd_dynamics_enabled=cmd_dynamics_enabled,
                         scaffold_wz_trim_vx=scaffold_wz_trim_vx,
                         scaffold_wz_trim_vy_abs=scaffold_wz_trim_vy_abs,
                         disturbance_enabled=disturbance_enabled,
                         disturbance_magnitude_scale=disturbance_magnitude_scale)
        self._disc_params = discriminator_params
        self._multihead = bool(multihead_disc)
        if multihead_disc:
            self._disc_module = MultiHeadDiscriminator(
                n_bins=N_BINS, hidden_sizes=discriminator_hidden)
        else:
            self._disc_module = Discriminator(hidden_sizes=discriminator_hidden)
        self._style_w = float(style_weight)

    def reset(self, rng: jax.Array) -> State:
        es = hex_jax.reset(self._params, rng, gait_scale=self._gait_scale)
        state = _initial_brax_state(es)
        # Seed prev_amp_state with the freshly-reset state so the first
        # step's transition is well-formed.
        amp_state = _extract_amp_state(self._params, es.mjx_data)
        info = dict(state.info)
        info["prev_amp_state"] = amp_state
        # jax.lax.scan in Brax's action_repeat wrapper requires the metrics
        # pytree to be identical between reset and step. Seed amp_style_reward
        # so step()'s addition matches the input carry.
        metrics = dict(state.metrics)
        metrics["amp_style_reward"] = jnp.zeros((), dtype=jnp.float32)
        return state.replace(info=info, metrics=metrics)

    def step(self, state: State, action: jax.Array) -> State:
        es: hex_jax.EnvState = state.pipeline_state
        # Capture the cmd ACTIVE for this step (i.e., the cmd in the pre-step
        # state). The disc input pairs the (s_t, s_{t+1}) state transition
        # with the cmd that was active when s_t was observed — same
        # convention as amp/prior_data.py.
        cmd_pre = es.cmd
        es = self._step_fn(self._params, es, action)

        # Compute AMP state on post-step mjx_data, build (s_t, s_{t+1}, cmd_t)
        # transition, score it with the (frozen) discriminator, and add
        # the resulting style reward to the env reward.
        new_amp = _extract_amp_state(self._params, es.mjx_data)
        prev_amp = state.info["prev_amp_state"]
        transition = jnp.concatenate([
            prev_amp, new_amp, cmd_for_disc(cmd_pre)
        ])                                                        # (101,)
        # style_reward expects a batched input — wrap, take [0].
        if self._multihead:
            bin_idx = cmd_to_bin(cmd_pre)[None]   # (1,)
            style_r = multihead_style_reward(self._disc_params,
                                              self._disc_module,
                                              transition[None, :],
                                              bin_idx)[0]
        else:
            style_r = style_reward(self._disc_params,
                                   self._disc_module,
                                   transition[None, :])[0]
        augmented_reward = es.reward + self._style_w * style_r

        new_metrics = {**state.metrics, **_scalar_metrics(es)}
        # Expose the style reward in metrics for TB tracking.
        new_metrics["amp_style_reward"] = style_r

        new_info = dict(state.info)
        new_info["prev_amp_state"] = new_amp

        return state.replace(
            pipeline_state=es,
            obs=es.obs,
            reward=augmented_reward,
            done=es.done.astype(jnp.float32),
            metrics=new_metrics,
            info=new_info,
        )

    # Read-only properties for inspection
    @property
    def style_weight(self) -> float:
        return self._style_w

    @property
    def discriminator_params(self):
        return self._disc_params
