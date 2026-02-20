import pandas as pd


def load_stops(file):
    df = pd.read_csv(file)

    required = {"lat", "lon"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain lat and lon columns")

    if "stop_name" not in df.columns:
        df["stop_name"] = [f"Stop_{i}" for i in range(len(df))]

    return df[["stop_name", "lat", "lon"]]
