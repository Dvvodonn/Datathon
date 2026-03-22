"""
Download official Spanish municipal boundary polygons from IGN OGC API.

Source: Instituto Geográfico Nacional — INSPIRE Administrative Units
        https://api-features.ign.es/collections/administrativeunit/items

Returns 8,294 municipalities as MultiPolygon geometries (EPSG:4326).
Key fields: nameunit (name), nationalcode (INE code), nationallevel (admin level).

Output: data/raw/spain_municipal_boundaries.geojson
"""

import os
import json
import requests

BASE = "https://api-features.ign.es/collections/administrativeunit/items"
OUT  = os.path.join(os.path.dirname(__file__), "spain_municipal_boundaries.geojson")

PAGE = 5000
all_features = []
offset = 0

while True:
    print(f"  Fetching offset {offset}…")
    r = requests.get(BASE, params={"f": "json", "limit": PAGE, "offset": offset}, timeout=120)
    r.raise_for_status()
    d = r.json()

    batch = d.get("features", [])
    all_features.extend(batch)
    print(f"  {len(all_features):,} / {d.get('numberMatched', '?')} features")

    if len(batch) < PAGE:
        break
    offset += PAGE

fc = {"type": "FeatureCollection", "features": all_features}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(fc, f, ensure_ascii=False)

print(f"\nSaved {len(all_features):,} municipalities → {OUT}")


if __name__ == "__main__":
    pass
