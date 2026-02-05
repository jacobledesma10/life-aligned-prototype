import os
import pandas as pd
import numpy as np

# Output path relative to this script: data/raw/ (sibling of mock/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_out_dir = os.path.join(_script_dir, "..", "raw")
os.makedirs(_out_dir, exist_ok=True)
out_path = os.path.join(_out_dir, "soil_data.csv")

timestamps = pd.date_range("2025-01-01", periods=300, freq="h")
data = {
    "timestamp": timestamps,
    "soil_moisture": np.sin(np.linspace(0, 10, 300)) + np.random.normal(0, 0.1, 300),
    "soil_ph": 6 + np.random.normal(0, 0.05, 300),
    "nitrogen": np.random.normal(0.3, 0.05, 300),
    "temperature": 15 + 5*np.sin(np.linspace(0, 5, 300)),
}

pd.DataFrame(data).to_csv(out_path, index=False)
