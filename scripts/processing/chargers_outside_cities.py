"""
Filter EV charging stations that fall OUTSIDE any Spanish municipality
with population > 10,000.

Uses exact municipal polygon boundaries from IGN.
City polygons are identified spatially: whichever IGN polygon contains
a city point (pop > 10,000) is a city polygon — no name matching needed.

Output: data/processed/chargers_outside_cities.csv
"""

import os
import pandas as pd
import geopandas as gpd

HERE           = os.path.dirname(__file__)
BOUNDARIES     = os.path.join(HERE, "..", "raw", "spain_municipal_boundaries.geojson")
CITIES_GEOJSON = os.path.join(HERE, "..", "..", "download_cities", "filtered_cities.geojson")
CHARGERS_CSV   = os.path.join(HERE, "..", "raw", "chargers", "sites.csv")
OUT_CSV        = os.path.join(HERE, "chargers_outside_cities.csv")

PROJ_CRS = "EPSG:25830"


def main():
    # ── Load municipal polygons ───────────────────────────────────────────────
    print("Loading municipal boundaries…")
    munis = gpd.read_file(BOUNDARIES)
    munis = munis[munis["nationallevelname"] == "Municipio"].copy()
    munis = munis.to_crs(PROJ_CRS)
    print(f"  {len(munis):,} municipality polygons")

    # ── Load city points (pop > 10,000) ──────────────────────────────────────
    print("Loading city points…")
    cities = gpd.read_file(CITIES_GEOJSON).to_crs(PROJ_CRS)
    print(f"  {len(cities):,} city points")

    # ── Spatially identify which polygons contain a city point ───────────────
    # Join city points to polygons — any polygon that contains at least one
    # city point is a "city polygon". No name matching involved.
    city_poly_join = gpd.sjoin(
        cities[["geometry"]],
        munis[["nameunit", "geometry"]],
        how="inner",
        predicate="within",
    )
    city_muni_idx = set(city_poly_join["index_right"])
    city_munis = munis.loc[list(city_muni_idx)].copy()
    print(f"  {len(city_munis):,} city polygons identified spatially")

    city_munis = city_munis.to_crs(PROJ_CRS)

    # ── Load chargers ─────────────────────────────────────────────────────────
    print("Loading chargers…")
    df = pd.read_csv(CHARGERS_CSV, usecols=["latitude", "longitude"]).dropna()
    df = df.reset_index(drop=True)
    chargers = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(PROJ_CRS)
    print(f"  {len(chargers):,} charger sites loaded")

    # ── Spatial join: find chargers INSIDE any city polygon ───────────────────
    print("Running spatial join…")
    inside = gpd.sjoin(
        chargers,
        city_munis[["nameunit", "geometry"]],
        how="inner",
        predicate="within",
    )
    inside_idx = set(inside.index)

    # Outside = all chargers not matched
    outside = chargers[~chargers.index.isin(inside_idx)].copy()
    outside_df = outside[["latitude", "longitude"]].reset_index(drop=True)

    outside_df.to_csv(OUT_CSV, index=False)

    print(f"\nChargers outside cities: {len(outside_df):,} / {len(chargers):,}")
    print(f"Chargers inside cities:  {len(inside_idx):,} / {len(chargers):,}")
    print(f"Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
