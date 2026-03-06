# src/data_collection/vegetation.py
import pandas as pd
from pathlib import Path

WEATHER_FILE = Path("data/processed/weather_daily.csv")
OUT_FILE = Path("data/processed/vegetation_features.csv")

def compute_vegetation_features():
    df = pd.read_csv(WEATHER_FILE)

    # Simple drought proxy
    df["dryness_index"] = (
        df["t2m"] / (df["tp"] + 1)
    )

    df["fuel_risk"] = df["dryness_index"].rank(pct=True)

    out = df[[
        "time", "latitude", "longitude",
        "dryness_index", "fuel_risk"
    ]]

    out.to_csv(OUT_FILE, index=False)
    print(f"[OK] Vegetation features saved → {OUT_FILE}")

if __name__ == "__main__":
    compute_vegetation_features()