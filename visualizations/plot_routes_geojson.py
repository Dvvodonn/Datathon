"""
plot_routes_geojson.py — visualize city-pair route usage on actual OSM road geometry
=====================================================================================

How the merges work
-------------------
1. Load abstract route edges (routes_250_alpha45_edges.csv):
   Each row is a graph edge (lon_a, lat_a) → (lon_b, lat_b) with a usage_count
   representing how many city-pair optimal paths traverse that edge.

2. Load OSM road geometry (spain_primary_interurban_arteries.gpkg):
   Actual road LineString geometries in WGS-84.

3. Build LineString per abstract edge:
   Each abstract edge becomes a 2-point LineString from its endpoint coords.

4. Corridor buffer (8 km):
   Each abstract LineString is projected to EPSG:3857 (metres), buffered by 8,000m,
   then reprojected back to WGS-84.  This creates a ~16km-wide corridor polygon
   aligned with the abstract edge direction.

5. Spatial join (sjoin with predicate "intersects"):
   Every OSM segment whose geometry intersects the corridor polygon inherits the
   corridor's usage_count.  A single OSM segment can match multiple corridors;
   we keep the maximum usage_count across all matches.

6. Plot:
   - OSM segments coloured white→orange→red by normalised usage (plasma colormap).
   - Existing chargers: blue dots.
   - Newly built chargers (x_built == 1): green dots.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import LineString

EDGES_CSV   = "models/routes_250_alpha45_edges.csv"
NODES_CSV   = "data_main/nodes.csv"
RESULTS_CSV = "models/results_nodes_250_alpha45.csv"
ROADS_GPKG  = "data/intermediate/road_routes/spain_primary_interurban_arteries.gpkg"
OUTPUT_PNG  = "visualizations/routes_geojson_250_alpha45.png"

BUFFER_M    = 8_000   # corridor half-width in metres
CRS_M       = "EPSG:3857"
CRS_DEG     = "EPSG:4326"

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
edges_df   = pd.read_csv(EDGES_CSV)
nodes_df   = pd.read_csv(NODES_CSV)
results_df = pd.read_csv(RESULTS_CSV)
roads_gdf  = gpd.read_file(ROADS_GPKG).to_crs(CRS_DEG)

print(f"  {len(edges_df):,} abstract edges  |  {len(roads_gdf):,} OSM segments")

# ── build corridor GeoDataFrame ───────────────────────────────────────────────
print("Building edge corridors...")
geoms = [
    LineString([(row.lon_a, row.lat_a), (row.lon_b, row.lat_b)])
    for row in edges_df.itertuples(index=False)
]
corridors_gdf = gpd.GeoDataFrame(
    {"usage_count": edges_df["usage_count"].values},
    geometry=geoms,
    crs=CRS_DEG
)
corridors_gdf = corridors_gdf.to_crs(CRS_M)
corridors_gdf["geometry"] = corridors_gdf.geometry.buffer(BUFFER_M)
corridors_gdf = corridors_gdf.to_crs(CRS_DEG)

# ── spatial join ──────────────────────────────────────────────────────────────
print("Spatial joining OSM segments to corridors...")
roads_idx = roads_gdf[["geometry"]].copy().reset_index().rename(columns={"index": "road_idx"})
joined = gpd.sjoin(roads_idx, corridors_gdf, how="left", predicate="intersects")

# Keep max usage per OSM segment
usage_per_seg = (
    joined.groupby("road_idx")["usage_count"].max().reset_index()
)
roads_gdf = roads_gdf.merge(usage_per_seg, left_index=True, right_on="road_idx", how="left")
roads_gdf["usage_count"] = roads_gdf["usage_count"].fillna(0)

n_colored = (roads_gdf["usage_count"] > 0).sum()
print(f"  {n_colored:,} / {len(roads_gdf):,} OSM segments matched")

# ── charger categories ────────────────────────────────────────────────────────
existing = nodes_df[nodes_df.is_existing_charger == 1]
# results_nodes_250_alpha45.csv contains only the built (feasible) nodes
built    = results_df[results_df.is_feasible_location == 1]

# ── plot ──────────────────────────────────────────────────────────────────────
print("Plotting...")
fig, ax = plt.subplots(figsize=(18, 14), facecolor="black")
ax.set_facecolor("black")

# Roads — unused in dim grey, used coloured by usage
unused = roads_gdf[roads_gdf["usage_count"] == 0]
used   = roads_gdf[roads_gdf["usage_count"] >  0].copy()

unused.plot(ax=ax, color="#1a1a1a", linewidth=0.2, zorder=1)

if len(used) > 0:
    norm      = mcolors.LogNorm(vmin=1, vmax=used["usage_count"].max())
    cmap      = plt.cm.plasma
    used["color"] = used["usage_count"].apply(lambda v: cmap(norm(v)))
    for _, row in used.iterrows():
        ax.plot(*row.geometry.xy, color=row["color"], linewidth=0.4, zorder=2)

# Charger nodes
ax.scatter(existing.lon, existing.lat,
           c="#4488ff", s=14, alpha=0.9, zorder=4, linewidths=0,
           label=f"Existing chargers ({len(existing)})")
ax.scatter(built.lon, built.lat,
           c="#00ee66", s=22, alpha=1.0, zorder=5,
           linewidths=0.5, edgecolors="white",
           label=f"Built chargers ({len(built)})")

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, orientation="vertical")
cbar.set_label("Route usage count", color="white", fontsize=10)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

ax.set_xlim(-9.5, 4.5)
ax.set_ylim(35.8, 44.0)
ax.set_aspect("equal")
ax.axis("off")
ax.legend(loc="lower right", framealpha=0.3, facecolor="black",
          edgecolor="white", labelcolor="white", fontsize=11, markerscale=1.5)
ax.set_title("Spain EV Charging — Route Usage on OSM Roads (α=45, 250km range)",
             color="white", fontsize=14, pad=12)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight", facecolor="black")
print(f"Saved → {OUTPUT_PNG}")
