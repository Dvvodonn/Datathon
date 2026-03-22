"""
Filter EV charging stations that fall within a city boundary.

Cities are point geometries, so "within a city" is defined as within a
population-scaled buffer around each city centre:

    radius_m = 1500 + 12 * sqrt(population)

This gives roughly:
    10,000 pop  →  ~2.7 km
    100,000 pop →  ~5.3 km
    1,000,000 pop → ~13.5 km

Output: chargers_in_cities.csv
"""

import os
import math
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

HERE = os.path.dirname(__file__)
CITIES_GEOJSON = os.path.join(HERE, "..", "..", "download_cities", "filtered_cities.geojson")
CHARGERS_CSV   = os.path.join(HERE, "..", "raw", "chargers", "sites.csv")
OUT_CSV        = os.path.join(HERE, "chargers_in_cities.csv")

# Projected CRS for metric buffers (UTM zone 30N — Spain)
PROJ_CRS = "EPSG:25830"


def city_radius(population):
    return 1500 + 12 * math.sqrt(population)


def main():
    # ── Load cities ───────────────────────────────────────────────────────────
    print("Loading cities…")
    with open(CITIES_GEOJSON, encoding="utf-8") as f:
        fc = json.load(f)

    city_rows = []
    for feat in fc["features"]:
        lon, lat = feat["geometry"]["coordinates"]
        pop = feat["properties"].get("population") or 0
        city_rows.append({
            "city_name": feat["properties"].get("name", ""),
            "population": pop,
            "geometry": Point(lon, lat),
        })

    cities_gdf = gpd.GeoDataFrame(city_rows, crs="EPSG:4326").to_crs(PROJ_CRS)
    cities_gdf["buffer"] = cities_gdf.apply(
        lambda r: r.geometry.buffer(city_radius(r.population)), axis=1
    )
    cities_buf = cities_gdf.set_geometry("buffer")[["city_name", "population", "buffer"]]
    cities_buf = cities_buf.set_crs(PROJ_CRS)
    print(f"  {len(cities_buf):,} city buffers built")

    # ── Load chargers ─────────────────────────────────────────────────────────
    print("Loading chargers…")
    df = pd.read_csv(CHARGERS_CSV, usecols=["latitude", "longitude"]).dropna()
    df = df.reset_index(drop=True)
    chargers_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(PROJ_CRS)
    print(f"  {len(chargers_gdf):,} charger sites loaded")

    # ── Spatial join: chargers inside any city buffer ─────────────────────────
    print("Joining…")
    joined = gpd.sjoin(
        chargers_gdf,
        cities_buf,
        how="inner",
        predicate="within",
    )

    # Drop duplicates (charger inside multiple overlapping city buffers)
    joined = joined[~joined.index.duplicated(keep="first")]

    result = joined[["latitude", "longitude", "city_name", "population"]].copy()
    result = result.sort_values(["city_name", "latitude"]).reset_index(drop=True)

    result.to_csv(OUT_CSV, index=False)
    print(f"\nChargers inside cities: {len(result):,} / {len(chargers_gdf):,}")
    print(f"Saved → {OUT_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    top = (result.groupby("city_name")
                 .size()
                 .sort_values(ascending=False)
                 .head(10))
    print("\nTop 10 cities by charger count:")
    for city, count in top.items():
        print(f"  {city:<25} {count:>4}")


if __name__ == "__main__":
    main()
