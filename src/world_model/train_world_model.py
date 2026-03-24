"""Standalone script to train the WorldModel on real CSV sensor data.

Usage (from project root):
    python3 src/world_model/train_world_model.py
"""

import os
import sys

# Allow imports from src/ when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.load_soil_data import load_soil_data
from world_model.world_model import WorldModel

SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "world_model.pt")


def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print("Loading soil data...")
    df = load_soil_data()
    print(f"  {len(df)} rows loaded — {len(df) - 1} transitions available for training")

    model = WorldModel()
    print("\nTraining world model (200 epochs)...")
    model.train_on_data(df, epochs=200)

    model.save(SAVE_PATH)
    print(f"\nModel saved to {SAVE_PATH}")

    # Quick sanity check
    import numpy as np
    sample = df.iloc[0][WorldModel.SENSOR_COLS].values.astype("float32")
    pred_no_action = model.predict(sample, action=0)
    pred_irrigate  = model.predict(sample, action=1)
    print(f"\nSanity check on first row: {dict(zip(WorldModel.SENSOR_COLS, sample.round(3)))}")
    print(f"  Predicted next (action=0): {pred_no_action.round(4)}")
    print(f"  Predicted next (action=1): {pred_irrigate.round(4)}")


if __name__ == "__main__":
    main()
