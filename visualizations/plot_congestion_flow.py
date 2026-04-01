"""
plot_congestion_flow.py
=======================
Congestion-model solution map in the style of solution_map_250_alpha45.png,
with roads coloured by calibrated AADT flow (road_edges_flow.csv).

  Roads              : dark→white→yellow→red  (log AADT scale)
  Feasible unselected: red dots
  Existing chargers  : blue dots
  Built stations     : coloured by power tier  (50=yellow  150=orange  350=red)
                       size ∝ charger count c_built
  Saturated station  : magenta X   (W_q = 480 min penalty)

Output: visualizations/solution_map_congestion_flow.png
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import geopandas as gpd

RESULTS_PATH = "congestion/outputs/results_congestion.csv"
NODES_PATH   = "data_main/nodes.csv"
ROADS_GPKG   = "data/raw/road_network/spain_interurban_edges.gpkg"
OUT_PNG      = "visualizations/solution_map_congestion_flow.png"

XLIM = (-9.5, 4.5)
YLIM = (35.8, 44.0)

# Power tier colours (light green → deep green gradient across 6 tiers)
P_COLORS = {
    50:  "#ccffcc",   # pale green
    100: "#88ff88",   # light green
    150: "#44dd44",   # medium green
    200: "#22bb22",   # green
    250: "#119911",   # dark green
    350: "#005500",   # deep forest green
}

print("Loading data …")
results = pd.read_csv(RESULTS_PATH)
nodes   = pd.read_csv(NODES_PATH)
flow    = pd.read_csv("data_main/road_edges_flow.csv", usecols=["effective_aadt"])

print("Loading road network …")
roads = gpd.read_file(ROADS_GPKG).to_crs("EPSG:4326")
roads["effective_aadt"] = flow["effective_aadt"].values
roads = roads.sort_values("effective_aadt").reset_index(drop=True)

# ── Node categories ───────────────────────────────────────────────────────────
built    = results[results["x_built"] == 1].copy()
existing = nodes[nodes["is_existing_charger"] == 1].copy()
feasible = nodes[nodes["is_feasible_location"] == 1].copy()
built_coords = set(zip(built["lon"].round(6), built["lat"].round(6)))
unselected = feasible[
    ~feasible.apply(
        lambda r: (round(r["lon"], 6), round(r["lat"], 6)) in built_coords, axis=1
    )
]
saturated    = built[built["wq_minutes"] >= 400]
normal_built = built[built["wq_minutes"] < 400]

print(f"  Built: {len(built)}  |  Unselected: {len(unselected):,}  |  "
      f"Existing: {len(existing):,}")
if "p_built_kw" in built.columns:
    print(f"  Power dist: {dict(built['p_built_kw'].value_counts().sort_index())}")

# ── Colormap ──────────────────────────────────────────────────────────────────
cmap = mcolors.LinearSegmentedColormap.from_list(
    "flow", ["#1a1a1a", "#555555", "#ffffff", "#ffee44", "#ff8800", "#cc0000"]
)
vmin = max(roads["effective_aadt"].quantile(0.05), 1)
vmax = roads["effective_aadt"].quantile(0.99)
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

# ── Plot ──────────────────────────────────────────────────────────────────────
print("Plotting …")
fig, ax = plt.subplots(figsize=(18, 14), facecolor="black")
ax.set_facecolor("black")
ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
ax.set_aspect("equal"); ax.axis("off")

print("  Drawing roads …")
roads.plot(ax=ax, column="effective_aadt", cmap=cmap, norm=norm,
           linewidth=0.25, alpha=0.75, zorder=1)

# Unselected feasible — very dim
ax.scatter(unselected["lon"], unselected["lat"],
           c="#551111", s=3, alpha=0.25, zorder=2, linewidths=0,
           label=f"Feasible unselected ({len(unselected):,})")

# Existing chargers — dim
ax.scatter(existing["lon"], existing["lat"],
           c="#223366", s=4, alpha=0.30, zorder=3, linewidths=0,
           label=f"Existing chargers ({len(existing):,})")

# Built — coloured by power tier, sized by charger count
size_map = {1: 80, 2: 130, 3: 180, 4: 240, 5: 300, 6: 360, 7: 420, 8: 480, 9: 540, 10: 600}
p_col = "p_built_kw" if "p_built_kw" in normal_built.columns else None

for p_val, color, label in [
    (50,  P_COLORS[50],  "50 kW"),
    (100, P_COLORS[100], "100 kW"),
    (150, P_COLORS[150], "150 kW"),
    (200, P_COLORS[200], "200 kW"),
    (250, P_COLORS[250], "250 kW"),
    (350, P_COLORS[350], "350 kW"),
]:
    if p_col is not None:
        sub = normal_built[normal_built[p_col] == p_val]
    else:
        sub = normal_built  # fallback: all together

    if len(sub) == 0:
        continue

    sizes = sub["c_built"].map(size_map).fillna(30)
    ax.scatter(sub["lon"], sub["lat"],
               c=color, s=sizes, alpha=1.0, zorder=5,
               linewidths=0.6, edgecolors="white",
               label=f"{label} tier  ({len(sub)} stations)")

    if p_col is None:
        break   # only one group if no p_built_kw

# Saturated
if len(saturated):
    ax.scatter(saturated["lon"], saturated["lat"],
               c="#ff00ff", s=120, alpha=1.0, zorder=6,
               linewidths=0.8, edgecolors="white", marker="X",
               label=f"Saturated (W_q=480 min, {len(saturated)})")

# ── Colorbar ──────────────────────────────────────────────────────────────────
sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.016, pad=0.01, aspect=35)
cbar.set_label("Effective AADT — vehicles/day", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
cbar.outline.set_edgecolor("white")
tick_vals = [t for t in [1000, 5000, 10000, 25000, 50000, 100000] if vmin <= t <= vmax]
cbar.set_ticks(tick_vals)
cbar.set_ticklabels([f"{t:,}" for t in tick_vals])

# ── Size legend (charger count) ───────────────────────────────────────────────
size_handles = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#888888",
           markersize=np.sqrt(size_map.get(c, 30)), label=f"{c} charger{'s' if c>1 else ''}")
    for c in [1, 2, 3, 4, 6]
]
size_legend = ax.legend(handles=size_handles, loc="lower left",
                        title="Charger count", title_fontsize=9,
                        framealpha=0.3, facecolor="black", edgecolor="white",
                        labelcolor="white", fontsize=9)
ax.add_artist(size_legend)

# ── Main legend ───────────────────────────────────────────────────────────────
ax.legend(loc="lower right", framealpha=0.3,
          facecolor="black", edgecolor="white",
          labelcolor="white", fontsize=10, markerscale=1.5)

n_c = int(built["c_built"].sum()) if len(built) else 0
ax.set_title(
    f"Spain EV Charging Network — Congestion-Optimal Solution  "
    f"({len(built)} stations, {n_c} chargers)  |  Roads coloured by AADT",
    color="white", fontsize=14, pad=12
)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="black")
print(f"Saved → {OUT_PNG}")
