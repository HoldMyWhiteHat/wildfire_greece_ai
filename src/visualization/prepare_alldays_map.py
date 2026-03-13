import pandas as pd
import folium
import geopandas as gpd
from shapely.geometry import Point
#from folium.plugins import HeatMap
from pathlib import Path


# Paths

PREDICTIONS_DIR = Path("src/visualization/predictions")
GREECE_PATH = "data/geo/gr.json"
OUTPUT_DIR = Path("src/visualization/maps")
OUTPUT_DIR.mkdir(exist_ok=True)

def risk_color_class(c):
    return [
        "#2ECC71",  # very low
        "#F1C40F",  # low
        "#E67E22",  # medium
        "#E74C3C",  # high
        "#8E44AD",  # Extreme
    ][c]


# Process each CSV independently

for csv_path in sorted(PREDICTIONS_DIR.glob("*.csv")):

    print(f"Processing {csv_path.name}")

    df = pd.read_csv(csv_path)
    #f = pd.read_csv(PREDICTIONS_DIR)

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
   
    if points.empty:
        print("No Greece land points — skipping")
        continue
    points.drop(columns=["index_right"], inplace=True)

    r = points["fire_risk"]

    points["risk_vis"] = (r - r.min()) / (r.max() - r.min() + 1e-8)

    points["risk_class"] = pd.qcut(
    points["fire_risk"],
    q=5,
    labels=False,
    duplicates="drop"
)

    m = folium.Map(
        location=[38.5, 23.5],
        zoom_start=6,
        tiles="cartodbpositron"
    )

    for _, row in points.iterrows():
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=5,
            color=risk_color_class(int(row.risk_class)),
            fill=True,
            fill_color=risk_color_class(int(row.risk_class)),
            fill_opacity=0.9,
            weight=0,
            tooltip=(
                f"Fire risk: {row.fire_risk:.3f}<br>"
                f"Daily class: {int(row.risk_class) + 1}/5"
            )
        ).add_to(m)
    
    #points.to_csv(OUTPUT_DIR, index=False)
    #print(f"Saved {len(points)} land points to {OUTPUT_DIR}")
    legend_html = """
    <div style="
        position: fixed;
        bottom: 40px;
        left: 40px;
        width: 230px;
        z-index:9999;
        background-color: white;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 0 12px rgba(0,0,0,0.2);
        font-size: 13px;
    ">
    <b>Fire Risk (Daily Relative)</b><br>
    <small>Quantile-based classification</small><br><br>

    <i style="background:#2ECC71;width:12px;height:12px;display:inline-block;"></i>
    Very Low (0–20%)<br>

    <i style="background:#F1C40F;width:12px;height:12px;display:inline-block;"></i>
    Low (20–40%)<br>

    <i style="background:#E67E22;width:12px;height:12px;display:inline-block;"></i>
    Moderate (40–60%)<br>

    <i style="background:#E74C3C;width:12px;height:12px;display:inline-block;"></i>
    High (60–80%)<br>

    <i style="background:#8E44AD;width:12px;height:12px;display:inline-block;"></i>
    Extreme (80–100%)
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

    output = OUTPUT_DIR / f"fire_risk_{csv_path.stem}.html"
    m.save(output)
    print(f"Saved → {output}")

print("All Greece-only maps generated")