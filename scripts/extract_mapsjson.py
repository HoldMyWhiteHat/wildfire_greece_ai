from pathlib import Path
import json
import re

MAPS_DIR = Path("src/visualization/maps")
OUT_FILE = Path("frontend/maps.json")

dates = []

for f in MAPS_DIR.glob("fire_risk_*.html"):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
    if m:
        dates.append(m.group(1))

dates.sort()

OUT_FILE.parent.mkdir(exist_ok=True)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dates, f, indent=2)

print(f"Indexed {len(dates)} maps")