import numpy as np

class ReservoirEncoder:
    def __init__(self, input_dim, reservoir_size=100, seed=42):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((reservoir_size, input_dim)) * 0.1
        self.state = np.zeros(reservoir_size)

    def step(self, x):
        self.state = np.tanh(self.W @ x + self.state)
        return self.state
