import sys
from pathlib import Path

# Allow running from project root: python3 src/action/train_rl.py
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from stable_baselines3 import PPO
from action.soil_env import SoilRegenerationEnv

env = SoilRegenerationEnv()
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=50000)
model.save("soil_regen_agent")
