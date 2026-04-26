import gymnasium as gym
import numpy as np


class VelCmdWrapper(gym.Wrapper):
    """
    Wraps Ant-v5 (or similar) to add a 2D velocity command to the observation
    and replaces the reward with velocity tracking.

    Obs: [original_obs, vx_cmd, vy_cmd]
    Reward: exp(-||v_actual - v_cmd||^2) + survival + ctrl_cost
    """

    def __init__(self, env, vmin=-1.0, vmax=1.0):
        super().__init__(env)
        self.vmin = vmin
        self.vmax = vmax
        self._cmd = np.zeros(2, dtype=np.float32)

        low = np.concatenate([env.observation_space.low, [vmin, vmin]]).astype(np.float32)
        high = np.concatenate([env.observation_space.high, [vmax, vmax]]).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._cmd = np.random.uniform(self.vmin, self.vmax, size=2).astype(np.float32)
        return self._augment(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        vx = info.get("x_velocity", 0.0)
        vy = info.get("y_velocity", 0.0)
        vel_err = np.array([vx - self._cmd[0], vy - self._cmd[1]])
        tracking = np.exp(-np.dot(vel_err, vel_err))  # 1.0 when perfect, ~0 when far off

        survival = info.get("reward_survive", 0.25)
        ctrl_cost = info.get("reward_ctrl", 0.0)  # already negative in Ant-v5

        reward = tracking + survival + ctrl_cost

        return self._augment(obs), float(reward), terminated, truncated, info

    def _augment(self, obs):
        return np.concatenate([obs, self._cmd]).astype(np.float32)
