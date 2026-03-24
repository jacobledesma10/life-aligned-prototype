import os
import numpy as np
from stable_baselines3 import PPO

from action.reservoir_env_wrapper import ReservoirEnvWrapper


class RLPolicy:
    def __init__(self, model_path="soil_regen_agent_100d"):
        path = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
        if os.path.isfile(path):
            self.model = PPO.load(model_path)
        else:
            env = ReservoirEnvWrapper()
            self.model = PPO("MlpPolicy", env, verbose=0)

    def propose_action(self, state_context):
        state = np.asarray(state_context, dtype=np.float32).flatten()
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)
