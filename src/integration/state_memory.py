class StateMemory:
    def __init__(self):
        self.history = []

    def update(self, state_vector):
        self.history.append(state_vector)
        if len(self.history) > 100:
            self.history.pop(0)

    def get_context(self):
        return sum(self.history) / len(self.history)
