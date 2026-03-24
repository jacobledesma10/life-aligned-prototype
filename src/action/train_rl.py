import sys
from pathlib import Path

# Allow running from project root: python3 src/action/train_rl.py
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from stable_baselines3 import PPO
from action.reservoir_env_wrapper import ReservoirEnvWrapper

# soil_regen_agent.zip (4D baseline) is preserved as a rollback reference.
# This script trains the 100D reservoir-context model.
env = ReservoirEnvWrapper()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("soil_regen_agent_100d")
