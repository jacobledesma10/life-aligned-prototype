"""Standalone script to train the WorldModel on real CSV sensor data.

Builds a labeled multi-action training set by combining:
  - Real natural transitions from CSV: (state_i, action=0, state_{i+1})
  - Synthetic labeled transitions:     (state_i, action=a, apply_action(state_i, a))
    for each of the 5 intervention actions (1-5)

This ensures the world model learns the expected effect of every action,
not just natural environmental drift.

Usage (from project root):
    python3 src/world_model/train_world_model.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.load_soil_data import load_soil_data
from world_model.world_model import WorldModel

SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "world_model.pt")


def _apply_action(state: np.ndarray, action: int) -> np.ndarray:
    """Deterministic action effect matching SoilRegenerationEnv.step() (no random drift)."""
    m, ph, n, t = state
    if action == 1:   m  += 0.1                         # irrigate
    elif action == 2: n  += 0.02                         # rest
    elif action == 3: m  -= 0.1                          # intervene
    elif action == 4: n  += 0.08                         # fertilize
    elif action == 5: ph += np.sign(6.5 - ph) * 0.05    # adjust pH
    return np.array([m, ph, n, t], dtype=np.float32)


def build_transitions(df) -> list:
    """Build (state, action, next_state) triples for all 6 actions."""
    transitions = []
    cols = WorldModel.SENSOR_COLS
    for i in range(len(df) - 1):
        s  = df.iloc[i][cols].values.astype(np.float32)
        ns = df.iloc[i + 1][cols].values.astype(np.float32)
        transitions.append((s, 0, ns))                        # real natural dynamics
        for a in range(1, 6):
            transitions.append((s, a, _apply_action(s, a)))   # synthetic labeled
    return transitions


def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print("Loading soil data...")
    df = load_soil_data()
    natural = len(df) - 1
    print(f"  {len(df)} rows loaded — {natural} natural transitions")

    transitions = build_transitions(df)
    print(f"  {len(transitions)} total transitions after action augmentation "
          f"({natural} real + {len(transitions) - natural} synthetic)")

    model = WorldModel()
    print("\nTraining world model on all 6 actions (200 epochs)...")
    model.train_on_transitions(transitions, epochs=200)

    model.save(SAVE_PATH)
    print(f"\nModel saved to {SAVE_PATH}")

    # Sanity check: irrigate on a dry state should raise moisture and score higher
    print("\nSanity check — dry state (moisture=-0.5):")
    from action.soil_env import life_reward
    dry = np.array([-0.5, 6.0, 0.3, 18.0], dtype=np.float32)
    labels = {0:"no action", 1:"irrigate", 2:"rest", 3:"intervene", 4:"fertilize", 5:"adjust pH"}
    for a in range(6):
        pred  = model.predict(dry, a)
        score = life_reward(pred)
        print(f"  a{a} {labels[a]:12s}: moisture={pred[0]:+.3f}  score={score:.4f}")


if __name__ == "__main__":
    main()
