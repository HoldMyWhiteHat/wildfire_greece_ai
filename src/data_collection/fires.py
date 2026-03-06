# src/data_collection/fires.py
import pandas as pd
import numpy as np
from pathlib import Path


WEATHER_FILE = Path("data/processed/weather_daily.csv")
OUT_FILE = Path("data/processed/fire_labels1.csv")

GREECE_BOUNDS = {
    "lat_min": 34.5,
    "lat_max": 42.0,
    "lon_min": 19.0,
    "lon_max": 29.5
}

def create_fire_labels():
    df = pd.read_csv(WEATHER_FILE, parse_dates=["time"])

    
    # Normalize components (Greece-wide)
    
    df["heat"] = (df["t2m"] - df["t2m"].min()) / (df["t2m"].max() - df["t2m"].min())
    df["wind"] = (df["wind_speed"] - df["wind_speed"].min()) / (
        df["wind_speed"].max() - df["wind_speed"].min()
    )

    # Dryness = inverse precipitation
    df["dryness"] = 1.0 - (
        (df["tp"] - df["tp"].min()) / (df["tp"].max() - df["tp"].min())
    )

    
    # Fire danger score
    
    df["fire_score"] = (
        0.5 * df["heat"] +
        0.3 * df["wind"] +
        0.2 * df["dryness"]
    )

   
    # Top X% = fire event
    
    threshold = df["fire_score"].quantile(0.95)
    df["fire_risk"] = (df["fire_score"] >= threshold).astype(int)

    out = df[["time", "latitude", "longitude", "fire_risk"]]
    out.to_csv(OUT_FILE, index=False)

    print("Fire labels created")
    print(df["fire_risk"].value_counts(normalize=True))


if __name__ == "__main__":
    create_fire_labels()