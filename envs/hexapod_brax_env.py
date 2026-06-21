"""
HexapodBraxEnv — Brax-compatible wrapper around envs.hexapod_env_jax.

Maps our pure-functional `(reset, step) -> EnvState` to Brax's
`brax.envs.base.Env` API. The full `EnvState` is parked in
`State.pipeline_state`; `obs` / `reward` / `done` / `metrics` mirror
fields out of it for Brax's consumers (PPO, eval, etc).

`gait_scale` is set at construction time and baked into the env. To run
a curriculum that fades scaffold contribution, build a fresh env per
stage and call ppo_train with `restore_params=...` chaining.
"""


import jax
import jax.numpy as jnp
from brax.envs.base import Env, State

from envs import hexapod_env_jax as hex_jax


def _scalar_metrics(env_state: hex_jax.EnvState) -> dict:
    """Brax aggregates episode metrics as float32 across the eval window;
    bool scalars cause a carry-dtype mismatch in lax.scan. Cast to float32."""
    out = {}
    for k, v in env_state.metrics.items():
        if v.ndim != 0:
            continue                           # skip bc_target (18,) etc.
        out[k] = v.astype(jnp.float32) if v.dtype == jnp.bool_ else v
    return out


def _initial_brax_state(env_state: hex_jax.EnvState) -> State:
    """Build the State Brax sees on reset — empty info / metrics-from-env."""
    return State(
        pipeline_state=env_state,
        obs=env_state.obs,
        reward=env_state.reward,
        done=env_state.done.astype(jnp.float32),
        metrics=_scalar_metrics(env_state),
        info={},
    )


class HexapodBraxEnv(Env):
    """Brax `Env` wrapping our MJX hexapod stack."""

    def __init__(self, model_path: str, gait_scale: float = 0.0,
                 cmd_mask: str = "stage3",
                 action_space: str = "joint",
                 cmd_dynamics_enabled: bool = False,
                 scaffold_wz_trim_vx: float = 0.0,
                 scaffold_wz_trim_vy_abs: float = 0.0,
                 disturbance_enabled: bool = False,
                 disturbance_magnitude_scale: float = 1.0):
        self._params = hex_jax.make_env_params(
            model_path,
            cmd_mask_name=cmd_mask,
            cmd_dynamics_enabled=cmd_dynamics_enabled,
            scaffold_wz_trim_vx=scaffold_wz_trim_vx,
            scaffold_wz_trim_vy_abs=scaffold_wz_trim_vy_abs,
            disturbance_enabled=disturbance_enabled,
            disturbance_magnitude_scale=disturbance_magnitude_scale,
        )
        self._gait_scale = float(gait_scale)
        self._cmd_mask_name = cmd_mask
        if action_space not in ("joint", "foot"):
            raise ValueError(f"action_space must be 'joint' or 'foot', "
                             f"got {action_space!r}")
        self._action_space = action_space
        # Pick the matching step function from the JAX env module.
        self._step_fn = (hex_jax.step if action_space == "joint"
                         else hex_jax.step_foot)
        self._action_size = 18
        # Probe the obs size by doing a synthetic reset on host (cheap; no
        # JIT). Avoids hardcoding 78 in two places.
        probe = hex_jax.reset(self._params, jax.random.PRNGKey(0),
                              gait_scale=self._gait_scale)
        self._obs_size = int(probe.obs.shape[-1])

    # -- Required Brax Env API ---------------------------------------------
    def reset(self, rng: jax.Array) -> State:
        es = hex_jax.reset(self._params, rng, gait_scale=self._gait_scale)
        return _initial_brax_state(es)

    def step(self, state: State, action: jax.Array) -> State:
        es: hex_jax.EnvState = state.pipeline_state
        es = self._step_fn(self._params, es, action)
        # `state.replace()` preserves info + any wrapper-added metrics (e.g.
        # Brax's training wrappers add 'reward' to metrics; episode wrapper
        # adds 'steps', 'truncation', etc. to info). Update only fields we
        # actually compute. Metric keys we DO compute overwrite their
        # wrapper-added defaults; new ones are added.
        new_metrics = {**state.metrics, **_scalar_metrics(es)}
        return state.replace(
            pipeline_state=es,
            obs=es.obs,
            reward=es.reward,
            done=es.done.astype(jnp.float32),
            metrics=new_metrics,
        )

    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def backend(self) -> str:
        return "mjx"

    # -- Convenience for evaluation / replay --------------------------------
    @property
    def params(self) -> hex_jax.EnvParams:
        return self._params

    @property
    def gait_scale(self) -> float:
        return self._gait_scale
