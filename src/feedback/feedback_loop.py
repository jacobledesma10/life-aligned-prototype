import numpy as np


class FeedbackLoop:
    """Buffers real (state, action, next_state) transitions and periodically
    fine-tunes the WorldModel so it learns actual action effects over time."""

    def __init__(self, world_model, update_every: int = 50):
        self.world_model = world_model
        self.update_every = update_every
        self.buffer = []

    def record(self, state: np.ndarray, action: int, next_state: np.ndarray) -> None:
        """Record a real observed transition. Triggers an update when buffer is full."""
        self.buffer.append((state.copy(), action, next_state.copy()))
        if len(self.buffer) >= self.update_every:
            self._update()

    def _update(self) -> None:
        n = len(self.buffer)
        print(f"[FeedbackLoop] Updating world model on {n} real observations")
        self.world_model.fine_tune(self.buffer, epochs=20)
        self.buffer.clear()
