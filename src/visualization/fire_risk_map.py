import folium
import pandas as pd
from folium.plugins import HeatMap

print("fire_risk_map.py started")

CSV_PATH = "src/visualization/predictions_greece_only.csv"

df = pd.read_csv(CSV_PATH)

print("Rows loaded:", len(df))

if len(df) == 0:
    raise ValueError("CSV is empty — map cannot be created")

m = folium.Map(
    location=[38.5, 23.5],
    zoom_start=6,
    tiles="cartodbpositron"
)

heat_data = [
    [row.latitude, row.longitude, row.risk_vis]
    for _, row in df.iterrows()
]

HeatMap(
    heat_data,
    radius=18,
    blur=25,
    min_opacity=0.4
).add_to(m)

OUT_PATH = "src/visualization/fire_risk_next_day_map.html"
m.save(OUT_PATH)

print(f"Map saved to {OUT_PATH}")