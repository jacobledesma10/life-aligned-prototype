import pandas as pd
import numpy as np

timestamps = pd.date_range("2025-01-01", periods=300, freq="H")
data = {
    "timestamp": timestamps,
    "soil_moisture": np.sin(np.linspace(0, 10, 300)) + np.random.normal(0, 0.1, 300),
    "soil_ph": 6 + np.random.normal(0, 0.05, 300),
    "nitrogen": np.random.normal(0.3, 0.05, 300),
    "temperature": 15 + 5*np.sin(np.linspace(0, 5, 300))
}

pd.DataFrame(data).to_csv("data/raw/soil_data.csv", index=False)
