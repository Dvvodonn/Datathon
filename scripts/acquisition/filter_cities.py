import json

with open("spain_cities_full.geojson", encoding="utf-8") as f:
    fc = json.load(f)

filtered = [feat for feat in fc["features"]
            if (feat["properties"].get("population") or 0) > 10_000]

out = {"type": "FeatureCollection", "features": filtered}

with open("filtered_cities.geojson", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print(f"Kept {len(filtered)} / {len(fc['features'])} cities (population > 10,000)")
