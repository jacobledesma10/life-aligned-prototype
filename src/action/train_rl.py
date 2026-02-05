from stable_baselines3 import PPO
from soil_env import SoilRegenerationEnv

env = SoilRegenerationEnv()
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=50000)
model.save("soil_regen_agent")
