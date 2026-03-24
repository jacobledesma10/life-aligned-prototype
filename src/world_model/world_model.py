import numpy as np
import torch
import torch.nn as nn


class _DynamicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),  # 4D state + 6D one-hot action
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        return self.net(x)


class WorldModel:
    """Learned dynamics model: f(state, action) -> next_state.

    State: 4D sensor vector [moisture, ph, nitrogen, temperature]
    Action: int 0-5, one-hot encoded before concatenation (10D total input)
    """

    SENSOR_COLS = ["soil_moisture", "soil_ph", "nitrogen", "temperature"]

    def __init__(self):
        self.net = _DynamicsNet()
        self.trained = False

    def _encode(self, state: np.ndarray, action: int) -> torch.Tensor:
        one_hot = np.zeros(6, dtype=np.float32)
        one_hot[action] = 1.0
        x = np.concatenate([state.astype(np.float32), one_hot])
        return torch.from_numpy(x)

    def train_on_data(self, df, epochs: int = 200, lr: float = 1e-3) -> None:
        """Train on consecutive CSV rows as (state, action=0, next_state) pairs."""
        states, next_states = [], []
        for i in range(len(df) - 1):
            s = df.iloc[i][self.SENSOR_COLS].values.astype(np.float32)
            ns = df.iloc[i + 1][self.SENSOR_COLS].values.astype(np.float32)
            states.append(s)
            next_states.append(ns)

        # All transitions use action=0 (natural dynamics — no interventions recorded)
        inputs = torch.stack([self._encode(s, 0) for s in states])
        targets = torch.tensor(np.array(next_states))

        split = int(len(inputs) * 0.8)
        X_train, X_val = inputs[:split], inputs[split:]
        y_train, y_val = targets[:split], targets[split:]

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.net.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            pred = self.net(X_train)
            loss = loss_fn(pred, y_train)
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
        targets = torch.tensor(np.array([ns for _, _, ns in transitions], dtype=np.float32))

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.net.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.net(inputs), targets)
            loss.backward()
            optimizer.step()
        self.net.eval()

    def predict(self, state: np.ndarray, action: int) -> np.ndarray:
        """Predict next sensor state given current state and action."""
        x = self._encode(state, action).unsqueeze(0)
        with torch.no_grad():
            out = self.net(x)
        return out.squeeze(0).numpy()

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, weights_only=True))
        self.net.eval()
        self.trained = True
