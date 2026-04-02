"""
visualizations/plot_comparison_sr.py
======================================
Side-by-side comparison of fixed vs variable stop rate solutions.

Left  : Fixed stop rate (5% flat)    — results_congestion.csv
Right : Variable stop rate (corridor-aware logistic) — results_congestion_variable_sr.csv

Both panels: black background, roads coloured by AADT, green station dots.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import geopandas as gpd

RESULTS_FIXED    = "congestion/outputs/results_congestion.csv"
RESULTS_VARIABLE = "congestion/outputs/results_congestion_variable_sr.csv"
NODES_PATH       = "data_main/nodes.csv"
ROADS_GPKG       = "data/raw/road_network/spain_interurban_edges.gpkg"
OUT_PNG          = "visualizations/solution_map_comparison_sr.png"

XLIM = (-9.5, 4.5)
YLIM = (35.8, 44.0)

P_COLORS = {
    50:  "#ccffcc",
    100: "#88ff88",
    150: "#44dd44",
    200: "#22bb22",
    250: "#119911",
    350: "#005500",
}
SIZE_MAP = {1: 80, 2: 130, 3: 180, 4: 240, 5: 300,
            6: 360, 7: 420, 8: 480, 9: 540, 10: 600}

# ── Load shared data ──────────────────────────────────────────────────────────
print("Loading data …")
nodes   = pd.read_csv(NODES_PATH)
flow    = pd.read_csv("data_main/road_edges_flow.csv", usecols=["effective_aadt"])

print("Loading road network …")
roads = gpd.read_file(ROADS_GPKG).to_crs("EPSG:4326")
roads["effective_aadt"] = flow["effective_aadt"].values
roads = roads.sort_values("effective_aadt").reset_index(drop=True)

existing  = nodes[nodes["is_existing_charger"] == 1].copy()
feasible  = nodes[nodes["is_feasible_location"] == 1].copy()

# ── Road colormap ─────────────────────────────────────────────────────────────
cmap = mcolors.LinearSegmentedColormap.from_list(
    "flow", ["#1a1a1a", "#555555", "#ffffff", "#ffee44", "#ff8800", "#cc0000"]
)
vmin = max(roads["effective_aadt"].quantile(0.05), 1)
vmax = roads["effective_aadt"].quantile(0.99)
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)


def _prep_solution(results_path):
    results = pd.read_csv(results_path)
    built   = results[results["x_built"] == 1].copy()
    built_coords = set(zip(built["lon"].round(6), built["lat"].round(6)))
    unselected = feasible[
        ~feasible.apply(
            lambda r: (round(r["lon"], 6), round(r["lat"], 6)) in built_coords, axis=1
        )
    ]
    saturated    = built[built["wq_minutes"] >= 400]
    normal_built = built[built["wq_minutes"] < 400]
    return built, unselected, saturated, normal_built


def _draw_panel(ax, title, built, unselected, saturated, normal_built, show_cbar=False):
    ax.set_facecolor("black")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, color="white", fontsize=12, pad=8)

    roads.plot(ax=ax, column="effective_aadt", cmap=cmap, norm=norm,
               linewidth=0.25, alpha=0.75, zorder=1)

    ax.scatter(unselected["lon"], unselected["lat"],
               c="#551111", s=3, alpha=0.25, zorder=2, linewidths=0)
    ax.scatter(existing["lon"], existing["lat"],
               c="#223366", s=4, alpha=0.30, zorder=3, linewidths=0)

    p_col = "p_built_kw" if "p_built_kw" in normal_built.columns else None
    for p_val, color, label in [
        (50,  P_COLORS[50],  "50 kW"),
        (100, P_COLORS[100], "100 kW"),
        (150, P_COLORS[150], "150 kW"),
        (200, P_COLORS[200], "200 kW"),
        (250, P_COLORS[250], "250 kW"),
        (350, P_COLORS[350], "350 kW"),
    ]:
        sub = normal_built[normal_built[p_col] == p_val] if p_col else normal_built
        if len(sub) == 0:
            continue
        sizes = sub["c_built"].map(SIZE_MAP).fillna(80)
        ax.scatter(sub["lon"], sub["lat"],
                   c=color, s=sizes, alpha=1.0, zorder=5,
                   linewidths=0.8, edgecolors="white",
                   label=f"{label} ({len(sub)})")
        if p_col is None:
            break

    if len(saturated):
        ax.scatter(saturated["lon"], saturated["lat"],
                   c="#ff00ff", s=120, alpha=1.0, zorder=6,
                   linewidths=0.8, edgecolors="white", marker="X",
                   label=f"Saturated ({len(saturated)})")

    n_c = int(built["c_built"].sum()) if len(built) else 0
    power_dist = dict(built["p_built_kw"].value_counts().sort_index()) if (len(built) and p_col) else {}

    # Legend
    ax.legend(loc="lower right", framealpha=0.35,
              facecolor="black", edgecolor="white",
              labelcolor="white", fontsize=8, markerscale=1.2)

    # Stats box
    stat_lines = [
        f"Stations: {len(built)}",
        f"Chargers: {n_c}",
    ]
    if power_dist:
        dominant = max(power_dist, key=power_dist.get)
        stat_lines.append(f"Dominant: {dominant} kW ({power_dist[dominant]})")

    ax.text(0.02, 0.02, "\n".join(stat_lines),
            transform=ax.transAxes, color="white", fontsize=9,
            verticalalignment="bottom",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="#444444"))

    return ax


# ── Build figure ──────────────────────────────────────────────────────────────
print("Plotting …")
fig, axes = plt.subplots(1, 2, figsize=(28, 11), facecolor="black")
fig.subplots_adjust(wspace=0.04)

built_f, uns_f, sat_f, nb_f = _prep_solution(RESULTS_FIXED)
built_v, uns_v, sat_v, nb_v = _prep_solution(RESULTS_VARIABLE)

_draw_panel(axes[0],
            "Fixed stop rate (5% flat)",
            built_f, uns_f, sat_f, nb_f)

_draw_panel(axes[1],
            "Variable stop rate (corridor-aware logistic)",
            built_v, uns_v, sat_v, nb_v)

# Shared colorbar
sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.010, pad=0.01, aspect=40)
cbar.set_label("Effective AADT — vehicles/day", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
cbar.outline.set_edgecolor("white")
tick_vals = [t for t in [1000, 5000, 10000, 25000, 50000, 100000] if vmin <= t <= vmax]
cbar.set_ticks(tick_vals)
cbar.set_ticklabels([f"{t:,}" for t in tick_vals])

fig.suptitle("Spain EV Charging — Fixed vs Variable Stop Rate  |  Roads coloured by AADT",
             color="white", fontsize=14, y=1.01)

plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="black")
print(f"Saved → {OUT_PNG}")
