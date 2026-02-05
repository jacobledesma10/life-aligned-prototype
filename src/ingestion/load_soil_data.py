import pandas as pd

def load_soil_data(path="data/raw/soil_data.csv"):
    df = pd.read_csv(path)
    #df['timestamp'] = pd.to_datetime(df['timestamp'])
    #df = df.sort_values("timestamp")
    return df
