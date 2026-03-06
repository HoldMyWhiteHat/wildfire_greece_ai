# src/data/weather.py
import xarray as xr
import pandas as pd
from pathlib import Path

ERA5_DIR = Path("data/era5_extracted")
INSTANT_FILE = ERA5_DIR / "data_stream-oper_stepType-instant.nc"
ACCUM_FILE = ERA5_DIR / "data_stream-oper_stepType-accum.nc"
OUT_FILE = Path("data/processed/weather_daily.csv")

GREECE_BOUNDS = {
    "lat_min": 34.5,
    "lat_max": 42.5,
    "lon_min": 19.0,
    "lon_max": 29.5,
}

def load_era5():
    ds_inst = xr.open_dataset(INSTANT_FILE, engine="netcdf4")
    ds_acc = xr.open_dataset(ACCUM_FILE, engine="netcdf4")

    
    inst_vars = ds_inst[["t2m", "u10", "v10", "d2m"]]
    acc_vars = ds_acc[["tp"]]

    ds = xr.merge([inst_vars, acc_vars])

    # IMPORTANT: Greece spatial cut
    ds = ds.sel(
        latitude=slice(GREECE_BOUNDS["lat_max"], GREECE_BOUNDS["lat_min"]),
        longitude=slice(GREECE_BOUNDS["lon_min"], GREECE_BOUNDS["lon_max"])
    )

    return ds


def preprocess_weather():
    ds = load_era5()

    # Convert to dataframe
    df = ds.to_dataframe().reset_index()

    # Unit conversions
    df["t2m"] = df["t2m"] - 273.15  # K → °C
    df["wind_speed"] = (df["u10"]**2 + df["v10"]**2) ** 0.5
    df["tp"] = df["tp"] * 1000  # m → mm

   
   
    if "valid_time" in df.columns:
        df = df.rename(columns={"valid_time": "time"})
   

    # Daily aggregation
    daily = (
        df.groupby(["time", "latitude", "longitude"])
        .agg({
            "t2m": "mean",
            "wind_speed": "mean",
            "d2m": "mean",
            "tp": "sum"
        })
        .reset_index()
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_FILE, index=False)

    print(f"[OK] Weather data saved → {OUT_FILE}")

if __name__ == "__main__":
    preprocess_weather()