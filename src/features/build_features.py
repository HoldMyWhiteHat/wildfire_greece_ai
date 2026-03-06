import pandas as pd
from pathlib import Path

WEATHER_PATH = "data/processed/weather_daily.csv"
VEG_PATH = "data/processed/vegetation_features.csv"
FIRE_PATH = "data/processed/fire_labels1.csv"

OUTPUT_PATH = "data/processed/features_daily1.csv"


def build_features():
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["time"])
    weather = weather.rename(columns={"time": "date"})
    weather["date"] = weather["date"].dt.date

    fire = pd.read_csv(FIRE_PATH, parse_dates=["time"])
    fire = fire.rename(columns={"time": "date"})
    fire["date"] = fire["date"].dt.date

    vegetation = pd.read_csv(VEG_PATH)

    # weather + fire (must align perfectly)
    df = weather.merge(
        fire,
        on=["date", "latitude", "longitude"],
        how="left"
    )

    # add vegetation (static features)
    df = df.merge(
        vegetation,
        on=["latitude", "longitude"],
        how="left"
    )
    df["fire_risk"] = df["fire_risk"].fillna(0).astype(int)

    #  handle missing vegetation (safe defaults)
    veg_cols = vegetation.columns.drop(["latitude", "longitude"])
    df[veg_cols] = df[veg_cols].fillna(0)

    # sort for time-series models
    df = df.sort_values(["latitude", "longitude", "date"])

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Feature table saved → {OUTPUT_PATH}")
    print("Shape:", df.shape)
    print(df.head())

if __name__ == "__main__":
    build_features()