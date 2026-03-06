import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# Paths

PRED_PATH = "src/models/predictions_next_day.csv"
GREECE_PATH = "data/geo/gr.json"
OUT_PATH = "src/visualization/predictions_greece_only.csv"


# Load data

df = pd.read_csv(PRED_PATH)

greece = gpd.read_file(GREECE_PATH)
greece = greece.to_crs("EPSG:4326")

points = gpd.GeoDataFrame(
    df,
    geometry=[
        Point(lon, lat)
        for lat, lon in zip(df.latitude, df.longitude)
    ],
    crs="EPSG:4326"
)


# Clip to Greece land

points = gpd.sjoin(
    points,
    greece,
    predicate="within",
    how="inner"
)
r = points["fire_risk_tomorrow"]
points["risk_vis"] = (r - r.min()) / (r.max() - r.min() + 1e-8)

points.drop(columns=["index_right"], inplace=True)

points.to_csv(OUT_PATH, index=False)
print(f"Saved {len(points)} land points to {OUT_PATH}")