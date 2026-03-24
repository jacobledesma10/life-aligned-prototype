import numpy as np
import torch
import torch.nn as nn


class _DynamicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),  # 4D state + 6D one-hot action
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.net(x)


class WorldModel:
    """Learned dynamics model: f(state, action) -> next_state.

    State: 4D sensor vector [moisture, ph, nitrogen, temperature]
    Action: int 0-5, one-hot encoded before concatenation (10D total input)

    Inputs and outputs are normalized per-feature using statistics computed
    from training data so that temperature (~20°C) does not dominate the MSE.
    """

    SENSOR_COLS = ["soil_moisture", "soil_ph", "nitrogen", "temperature"]

    def __init__(self):
        self.net = _DynamicsNet()
        self.trained = False
        self.state_mean: np.ndarray | None = None
        self.state_std:  np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        if self.state_mean is None:
            return state.astype(np.float32)
        return ((state - self.state_mean) / self.state_std).astype(np.float32)

    def _denormalize(self, state: np.ndarray) -> np.ndarray:
        if self.state_mean is None:
            return state
        return state * self.state_std + self.state_mean

    def _encode(self, state: np.ndarray, action: int) -> torch.Tensor:
        one_hot = np.zeros(6, dtype=np.float32)
        one_hot[action] = 1.0
        x = np.concatenate([self._normalize(state), one_hot])
        return torch.from_numpy(x)

    def _fit_scaler(self, transitions: list) -> None:
        """Compute per-feature mean/std from all states in the transition list."""
        all_states = np.array(
            [s for s, _, _ in transitions] + [ns for _, _, ns in transitions],
            dtype=np.float32,
        )
        self.state_mean = all_states.mean(axis=0)
        self.state_std  = all_states.std(axis=0) + 1e-8

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_data(self, df, epochs: int = 200, lr: float = 1e-3) -> None:
        """Train on consecutive CSV rows as (state, action=0, next_state) pairs."""
        transitions = []
        for i in range(len(df) - 1):
            s  = df.iloc[i][self.SENSOR_COLS].values.astype(np.float32)
            ns = df.iloc[i + 1][self.SENSOR_COLS].values.astype(np.float32)
            transitions.append((s, 0, ns))
        self.train_on_transitions(transitions, epochs=epochs, lr=lr)

    def train_on_transitions(self, transitions: list, epochs: int = 200, lr: float = 1e-3) -> None:
        """Train from scratch on a pre-built list of (state, action, next_state) tuples.

        Supports any mix of actions — use this instead of train_on_data() when
        labeled multi-action transitions are available.
        """
        self._fit_scaler(transitions)

        inputs  = torch.stack([self._encode(s, a) for s, a, _ in transitions])
        targets_raw = np.array([ns for _, _, ns in transitions], dtype=np.float32)
        targets = torch.tensor((targets_raw - self.state_mean) / self.state_std)

        split = int(len(inputs) * 0.8)
        X_train, X_val = inputs[:split], inputs[split:]
        y_train, y_val = targets[:split], targets[split:]

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.net.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            loss = loss_fn(self.net(X_train), y_train)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                self.net.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self.net(X_val), y_val).item()
                self.net.train()
                print(f"  Epoch {epoch:3d} | train_mse={loss.item():.5f} | val_mse={val_loss:.5f}")

        self.net.eval()
        self.trained = True

    def fine_tune(self, transitions: list, epochs: int = 20, lr: float = 1e-3) -> None:
        """Fine-tune on real (state, action, next_state) observations."""
        if not transitions:
            return
        inputs = torch.stack([self._encode(s, a) for s, a, _ in transitions])
        targets_raw = np.array([ns for _, _, ns in transitions], dtype=np.float32)
        if self.state_mean is not None:
            targets_raw = (targets_raw - self.state_mean) / self.state_std
        targets = torch.tensor(targets_raw)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.net.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.net(inputs), targets)
            loss.backward()
            optimizer.step()
        self.net.eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, state: np.ndarray, action: int) -> np.ndarray:
        """Predict next sensor state given current state and action."""
        x = self._encode(state, action).unsqueeze(0)
        with torch.no_grad():
            out = self.net(x).squeeze(0).numpy()
        return self._denormalize(out)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        payload = {"net": self.net.state_dict()}
        if self.state_mean is not None:
            payload["state_mean"] = torch.tensor(self.state_mean)
            payload["state_std"]  = torch.tensor(self.state_std)
        torch.save(payload, path)

    def load(self, path: str) -> None:
        data = torch.load(path, weights_only=True)
        if isinstance(data, dict) and "net" in data:
            self.net.load_state_dict(data["net"])
            if "state_mean" in data:
                self.state_mean = data["state_mean"].numpy()
                self.state_std  = data["state_std"].numpy()
        else:
            # Legacy checkpoint — plain state dict, no scaler
            self.net.load_state_dict(data)
        self.net.eval()
        self.trained = True
