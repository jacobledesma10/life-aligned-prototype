import os
import numpy as np
from stable_baselines3 import PPO

from action.soil_env import SoilRegenerationEnv


class RLPolicy:
    def __init__(self, model_path="soil_regen_agent"):
        path = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
        if os.path.isfile(path):
            self.model = PPO.load(model_path)
        else:
            env = SoilRegenerationEnv()
            self.model = PPO("MlpPolicy", env, verbose=0)

    def propose_action(self, state_context):
        state = np.asarray(state_context, dtype=np.float32).flatten()
        # PPO was trained on 4-dim env; context may be reservoir (e.g. 100-dim) — use first 4
        if state.size >= 4:
            state = state[:4]
        else:
            state = np.resize(state, 4)
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)
