"""
visualize_network.py
====================
Map visualization of the fastest_feasible_edges network.
- Edges sampled and colored by travel time
- Nodes colored by type (city / feasible location / charger)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import contextily as ctx
from shapely.geometry import LineString, Point

# ── config ────────────────────────────────────────────────────────────────────
EDGES_CSV   = "data_main/edges.csv"
NODES_CSV   = "data_main/nodes.csv"
OUT_PNG     = "outputs/maps/network_visualization.png"
N_EDGES     = 40_000      # edges to sample for drawing
EPSG_PROJ   = 3857        # Web Mercator (contextily default)

print("Loading data …")
edges = pd.read_csv(EDGES_CSV)
nodes = pd.read_csv(NODES_CSV)

# ── sample edges in distance buckets so all ranges are represented ────────────
bins = [0, 50, 100, 150, 200, 300, 500]
edges["dist_bin"] = pd.cut(edges["distance_km"], bins=bins)
per_bin = N_EDGES // (len(bins) - 1)
sampled = (
    edges.groupby("dist_bin", observed=True)
    .apply(lambda g: g.sample(min(len(g), per_bin), random_state=42))
    .reset_index(drop=True)
)
print(f"Sampled {len(sampled):,} edges across distance buckets")

# ── build GeoDataFrames ───────────────────────────────────────────────────────
print("Building geometries …")

# Edges as LineStrings
edge_geoms = [
    LineString([(row.lon_a, row.lat_a), (row.lon_b, row.lat_b)])
    for row in sampled.itertuples()
]
gdf_edges = gpd.GeoDataFrame(sampled, geometry=edge_geoms, crs="EPSG:4326").to_crs(EPSG_PROJ)

# Nodes
gdf_nodes = gpd.GeoDataFrame(
    nodes,
    geometry=gpd.points_from_xy(nodes.lon, nodes.lat),
    crs="EPSG:4326",
).to_crs(EPSG_PROJ)

# ── plot ──────────────────────────────────────────────────────────────────────
print("Plotting …")
fig, ax = plt.subplots(figsize=(20, 18), facecolor="#0d1117")
ax.set_facecolor("#0d1117")

# Edges — colored by travel time, thin + transparent
cmap = plt.cm.plasma
norm = mcolors.Normalize(
    vmin=sampled["estimated_time_min"].quantile(0.05),
    vmax=sampled["estimated_time_min"].quantile(0.95),
)

# Draw in batches by color for speed
print("  drawing edges …")
n_batches = 50
batch_size = len(gdf_edges) // n_batches
for b in range(n_batches + 1):
    batch = gdf_edges.iloc[b * batch_size : (b + 1) * batch_size]
    if len(batch) == 0:
        continue
    colors = [cmap(norm(t)) for t in batch["estimated_time_min"]]
    batch.plot(ax=ax, color=colors, linewidth=0.3, alpha=0.35)

# Nodes — layered by type (chargers on top)
print("  drawing nodes …")
cities    = gdf_nodes[gdf_nodes.is_city == 1]
feasible  = gdf_nodes[gdf_nodes.is_feasible_location == 1]
chargers  = gdf_nodes[gdf_nodes.is_existing_charger == 1]

feasible.plot(ax=ax, color="#00ff99", markersize=2.5, alpha=0.6, label="Feasible location")
cities.plot(ax=ax,   color="#4fc3f7", markersize=6,   alpha=0.9, label="City (clustered)")
chargers.plot(ax=ax, color="#ff6b6b", markersize=3,   alpha=0.8, label="EV charger")

# Basemap
print("  adding basemap …")
try:
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter, zoom=6, alpha=0.4)
except Exception as e:
    print(f"  basemap failed ({e}), skipping")

# Colorbar for travel time
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.6)
cbar.set_label("Travel time (min)", color="white", fontsize=13)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

# Legend
legend = ax.legend(
    loc="lower left", fontsize=11,
    facecolor="#1a1f2e", edgecolor="white", labelcolor="white",
    markerscale=2, framealpha=0.8,
)

ax.set_title(
    "Spain EV Charging Network — Fastest-Path Edges",
    fontsize=18, color="white", pad=14, fontweight="bold",
)
ax.set_axis_off()

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUT_PNG}")
plt.show()
