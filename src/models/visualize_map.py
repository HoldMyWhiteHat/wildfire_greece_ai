import pandas as pd
import folium
from folium.plugins import HeatMap


# Load predictions

df = pd.read_csv("src/models/predictions_next_day.csv")

# Safety
df = df.dropna(subset=["latitude", "longitude", "fire_risk_tomorrow"])


# Create base map (Greece-centered)

m = folium.Map(
    location=[38.0, 23.7],  # Greece center-ish
    zoom_start=6,
    tiles="cartodbpositron"
)


# Heatmap layer

heat_data = [
    [row.latitude, row.longitude, row.fire_risk_tomorrow]
    for _, row in df.iterrows()
]

HeatMap(
    heat_data,
    radius=18,
    blur=12,
    min_opacity=0.3,
    max_zoom=8
).add_to(m)


# Save

m.save("src/models/fire_risk_next_day_map.html")
print("Map saved as fire_risk_next_day_map.html")