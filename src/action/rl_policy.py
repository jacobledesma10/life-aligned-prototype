from stable_baselines3 import PPO

class RLPolicy:
    def __init__(self, model_path="soil_regen_agent"):
        self.model = PPO.load(model_path)

    def propose_action(self, state_context):
        action, _ = self.model.predict(state_context, deterministic=True)
        return int(action)
