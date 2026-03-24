import gymnasium as gym
import numpy as np

class SoilRegenerationEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Observation: [moisture, pH, nitrogen, temp]
        self.observation_space = gym.spaces.Box(
            low=np.array([-2, 4, 0, -10]),
            high=np.array([2, 9, 1, 40]),
            dtype=np.float32
        )

        # Actions:
        #   0 = No Action
        #   1 = Irrigate          (moisture += 0.1)
        #   2 = Rest              (nitrogen += 0.02, small recovery)
        #   3 = Intervene         (moisture -= 0.1, risky)
        #   4 = Fertilize         (nitrogen += 0.08, meaningful boost)
        #   5 = Adjust pH         (ph nudged toward 6.5 by 0.05)
        self.action_space = gym.spaces.Discrete(6)

        self.state = None

    def reset(self, seed=None, options=None):
        self.state = np.array([0.0, 6.5, 0.3, 20.0])
        return self.state, {}

    def step(self, action):
        moisture, ph, nitrogen, temp = self.state

        # Simulate effect of actions
        if action == 1:
            moisture += 0.1
        elif action == 2:
            nitrogen += 0.02
        elif action == 3:
            moisture -= 0.1  # risky intervention
        elif action == 4:
            nitrogen += 0.08  # meaningful fertilization
        elif action == 5:
            ph += np.sign(6.5 - ph) * 0.05  # lime or sulfur toward pH 6.5

        # Natural drift
        moisture += np.random.normal(0, 0.02)
        ph += np.random.normal(0, 0.01)

        self.state = np.array([moisture, ph, nitrogen, temp])

        # Life-aligned reward
        reward = self._life_reward(self.state)

        terminated = False
        truncated = False
        return self.state, reward, terminated, truncated, {}

    def _life_reward(self, state):
        moisture, ph, nitrogen, temp = state

        diversity_term  = -abs(ph - 6.5)
        resilience_term = -abs(moisture)
        regen_term      = nitrogen
        thermal_term    = -abs(temp - 18.0) / 10.0  # optimal ~18°C

        return diversity_term + resilience_term + regen_term + thermal_term


def life_reward(state) -> float:
    """Module-level life-aligned reward. state: [moisture, ph, nitrogen, temp]."""
    moisture, ph, nitrogen, temp = state
    return float(
        -abs(ph - 6.5)
        + -abs(moisture)
        + nitrogen
        + -abs(temp - 18.0) / 10.0
    )
