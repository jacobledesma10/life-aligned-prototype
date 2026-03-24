import numpy as np
import gymnasium as gym
from gymnasium import spaces

from action.soil_env import SoilRegenerationEnv
from perception.reservoir_encoder import ReservoirEncoder
from integration.state_memory import StateMemory


class ReservoirEnvWrapper(gym.Wrapper):
    """Wraps SoilRegenerationEnv so PPO trains on 100D reservoir context states.

    Applies the same encoder → memory pipeline as main.py, ensuring that the
    observation space seen during training is identical to inference.
    """

    OBS_DIM = 100

    def __init__(self, env=None):
        if env is None:
            env = SoilRegenerationEnv()
        super().__init__(env)
        self.encoder = ReservoirEncoder(input_dim=4)  # seed=42 by default
        self.memory = StateMemory()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        # action_space inherited unchanged: Discrete(4)

    def _reset_pipeline(self):
        """Zero encoder recurrent state and clear memory history."""
        self.encoder.state = np.zeros(self.OBS_DIM)
        self.memory.history = []

    def _encode(self, raw_obs: np.ndarray) -> np.ndarray:
        neural_state = self.encoder.step(raw_obs)
        self.memory.update(neural_state)
        return self.memory.get_context().astype(np.float32)

    def reset(self, seed=None, options=None):
        raw_obs, info = self.env.reset(seed=seed, options=options)
        self._reset_pipeline()
        # Prime memory with one entry so get_context() never divides by zero
        return self._encode(raw_obs), info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._encode(raw_obs), reward, terminated, truncated, info
