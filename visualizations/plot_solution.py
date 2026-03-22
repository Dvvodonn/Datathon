"""
plot_solution.py — visualize road network + charger solution
  Roads              : white lines (from spatial geometry)
  Existing chargers  : blue dots
  Feasible unselected: red dots
  Selected (built)   : green dots
"""

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

ROADS_PATH   = "data/intermediate/road_routes/spain_primary_interurban_arteries.gpkg"
NODES_PATH   = "data_main/nodes.csv"
RESULTS_PATH = "models/results_nodes.csv"
OUTPUT_PATH  = "visualizations/solution_map.png"

print("Loading data …")
roads_gdf  = gpd.read_file(ROADS_PATH)
nodes_df   = pd.read_csv(NODES_PATH)
results_df = pd.read_csv(RESULTS_PATH)
print(f"  {len(roads_gdf):,} road segments")

# ── charger categories ─────────────────────────────────────────────────────────
existing  = nodes_df[nodes_df.is_existing_charger == 1]
selected  = results_df[results_df.x_built == 1]
selected_coords = set(zip(selected.lon.round(6), selected.lat.round(6)))
feasible_all = nodes_df[nodes_df.is_feasible_location == 1]
unselected = feasible_all[
    ~feasible_all.apply(lambda r: (round(r.lon, 6), round(r.lat, 6)) in selected_coords, axis=1)
]

print(f"  Existing chargers  : {len(existing)}")
print(f"  Selected (built)   : {len(selected)}")
print(f"  Feasible unselected: {len(unselected)}")

# ── plot ───────────────────────────────────────────────────────────────────────
print("Plotting …")
fig, ax = plt.subplots(figsize=(18, 14), facecolor="black")
ax.set_facecolor("black")

roads_gdf.plot(ax=ax, color="white", linewidth=0.3, alpha=0.6, zorder=1)

ax.scatter(unselected.lon, unselected.lat, c="#ff4444", s=8,  alpha=0.7, zorder=2,
           linewidths=0, label=f"Feasible unselected ({len(unselected)})")
ax.scatter(existing.lon,   existing.lat,   c="#4488ff", s=12, alpha=0.9, zorder=3,
           linewidths=0, label=f"Existing chargers ({len(existing)})")
ax.scatter(selected.lon,   selected.lat,   c="#00ee66", s=20, alpha=1.0, zorder=4,
           linewidths=0.5, edgecolors="white",
           label=f"Selected / built ({len(selected)})")

# Clip to mainland Spain (exclude Canary Islands)
ax.set_xlim(-9.5, 4.5)
ax.set_ylim(35.8, 44.0)
ax.set_aspect("equal")
ax.axis("off")

ax.legend(loc="lower right", framealpha=0.3, facecolor="black",
          edgecolor="white", labelcolor="white", fontsize=11, markerscale=1.5)

ax.set_title("Spain EV Charging Network — Optimised Solution",
             color="white", fontsize=16, pad=12)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="black")
print(f"Saved → {OUTPUT_PATH}")
