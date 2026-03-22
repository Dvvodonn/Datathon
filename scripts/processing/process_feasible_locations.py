"""
Filter feasible EV charger locations (gas stations) to municipalities
with population > 10,000.

Sources:
  data/raw/feasible_locations/EstacionesDeServicio.csv  — 12,215 gas stations
  download_cities/filtered_cities.geojson               — 974 city points (pop > 10k)
  data/raw/spain_municipal_boundaries.geojson           — 8,294 municipality polygons

Method:
  1. Spatial join: find which municipality polygon contains each city point
     → gives us the set of municipality polygons with pop > 10k
  2. Convert gas station CSV → GeoDataFrame (lon/lat from CoordenadaXDec/YDec)
  3. Spatial join: keep gas stations that fall inside a high-pop municipality

Output:
  data/processed/feasible_locations/feasible_locations.gpkg  — filtered GeoPackage
  data/processed/feasible_locations/feasible_locations.csv   — flat CSV
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ── paths ─────────────────────────────────────────────────────────────────────
RAW_STATIONS  = Path("data/raw/feasible_locations/EstacionesDeServicio.csv")
CITIES_GJ     = Path("download_cities/filtered_cities.geojson")
MUNIS_GJ      = Path("data/raw/spain_municipal_boundaries.geojson")
OUT_DIR       = Path("data/processed/feasible_locations")
OUT_GPKG      = OUT_DIR / "feasible_locations.gpkg"
OUT_CSV       = OUT_DIR / "feasible_locations.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Load municipal boundaries ──────────────────────────────────────────────
print("Loading municipal boundaries …")
munis = gpd.read_file(MUNIS_GJ)
# Keep only municipalities (4th order), not provinces or regions
munis = munis[munis["nationallevelname"] == "Municipio"].copy()
print(f"  {len(munis):,} municipality polygons (Municipio level only)")

# ── 2. Find municipalities that contain a city with pop > 10k ─────────────────
print("Spatial-joining cities → municipality polygons …")
cities = gpd.read_file(CITIES_GJ)   # Points, pop > 10k already filtered
print(f"  {len(cities):,} cities with population > 10,000")

# Project to metric CRS for accurate distance matching
# sjoin_nearest in case a city centroid falls just outside its polygon
METRIC_CRS = "EPSG:25830"
cities_m = cities.to_crs(METRIC_CRS)
munis_m  = munis[["nameunit", "nationalcode", "geometry"]].to_crs(METRIC_CRS)

joined = gpd.sjoin_nearest(
    cities_m,
    munis_m,
    how="left",
    max_distance=5000,   # 5km tolerance for centroid mismatches
)

# Keep unique municipality codes that matched
pop_muni_codes = set(joined["nationalcode"].dropna().unique())
high_pop_munis = munis[munis["nationalcode"].isin(pop_muni_codes)].copy()
print(f"  {len(high_pop_munis):,} municipality polygons with pop > 10k")

# Save the municipality polygons themselves
OUT_MUNIS_GPKG = OUT_DIR / "municipalities_pop10k.gpkg"
OUT_MUNIS_CSV  = OUT_DIR / "municipalities_pop10k.csv"

# ── 3. Load gas stations ──────────────────────────────────────────────────────
print("Loading gas stations …")
df = pd.read_csv(RAW_STATIONS, sep="|", encoding="latin1", on_bad_lines="skip")
print(f"  {len(df):,} stations loaded")

# Drop rows without coordinates
df = df.dropna(subset=["CoordenadaXDec", "CoordenadaYDec"])
print(f"  {len(df):,} stations with valid coordinates")

# Build GeoDataFrame
stations = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["CoordenadaXDec"], df["CoordenadaYDec"]),
    crs="EPSG:4326",
)

# ── 4. Spatial filter: keep stations inside high-pop municipalities ────────────
print("Filtering stations to high-pop municipalities …")
# Find stations INSIDE high-pop municipalities, then keep the rest
inside = gpd.sjoin(
    stations,
    high_pop_munis[["geometry"]],
    how="inner",
    predicate="within",
).drop_duplicates(subset=["Id"])

inside_ids = set(inside["Id"])
filtered = stations[~stations["Id"].isin(inside_ids)].copy()

print(f"  {len(filtered):,} stations OUTSIDE municipalities with pop > 10k  (interurban)")
print(f"  (excluded {len(inside_ids):,} stations inside high-pop municipalities)")

# ── 5. Save ───────────────────────────────────────────────────────────────────
print(f"Saving municipality polygons → {OUT_MUNIS_GPKG}")
high_pop_munis.to_file(str(OUT_MUNIS_GPKG), driver="GPKG", layer="municipalities_pop10k")
high_pop_munis.drop(columns=["geometry"]).to_csv(str(OUT_MUNIS_CSV), index=False)

print(f"Saving → {OUT_GPKG}")
filtered.to_file(str(OUT_GPKG), driver="GPKG", layer="feasible_locations")

print(f"Saving → {OUT_CSV}")
csv_df = filtered.drop(columns=["geometry"])
csv_df.to_csv(str(OUT_CSV), index=False, encoding="utf-8")

print()
print("=== Summary ===")
print(f"  Input stations          : {len(stations):,}")
print(f"  High-pop munis (>10k)   : {len(high_pop_munis):,}")
print(f"  Inside high-pop (excl.) : {len(inside_ids):,}")
print(f"  Interurban stations     : {len(filtered):,}")
print(f"  Columns            : {list(filtered.columns)}")
print()
print(f"  GPKG  → {OUT_GPKG}  ({OUT_GPKG.stat().st_size/1e6:.1f} MB)")
print(f"  CSV   → {OUT_CSV}  ({OUT_CSV.stat().st_size/1e3:.0f} kB)")
