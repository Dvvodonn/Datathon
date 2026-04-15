"""
plot_congestion_solution.py
===========================
Map of the congestion-extended Benders solution.

  Background   : Spain interurban road network (grey)
  Grey dots    : feasible but not selected
  Blue dots    : existing EV chargers (always open)
  Coloured dots: built stations, colour = charger count (1→4)
  Dot size     : proportional to λ_k (arrival rate)
  Halo ring    : W_q waiting time (radius ∝ log W_q)

Usage:
  python visualizations/plot_congestion_solution.py [--tag v2]

Output: visualizations/congestion_solution[_<tag>].png
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import geopandas as gpd

parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default="",
                    help="Result file tag, e.g. 'v2' reads results_congestion_v2.csv")
parser.add_argument("--results", type=str, default="",
                    help="Explicit path to results CSV (overrides --tag)")
parser.add_argument("--out", type=str, default="",
                    help="Explicit output PNG path (overrides default)")
args = parser.parse_args()

tag_str      = f"_{args.tag}" if args.tag else ""
RESULTS_PATH = args.results if args.results else f"congestion/outputs/results_congestion{tag_str}.csv"
NODES_PATH   = "data_main/nodes.csv"
ROADS_GPKG   = "data/raw/road_network/spain_interurban_edges.gpkg"
OUT_PNG      = args.out if args.out else f"visualizations/congestion_solution{tag_str}.png"

XLIM = (-9.5, 4.5)
YLIM = (35.8, 44.0)

# Charger-count colour palette (extended to c=7)
C_COLORS = {
    1: "#ffe066",
    2: "#ff9900",
    3: "#e63030",
    4: "#9b0000",
    5: "#cc44ff",
    6: "#7700cc",
    7: "#ffffff",
}

print("Loading data …")
results  = pd.read_csv(RESULTS_PATH)
nodes    = pd.read_csv(NODES_PATH)

# ── Road network background ───────────────────────────────────────────────────
print("Loading road network …")
roads = gpd.read_file(ROADS_GPKG).to_crs("EPSG:4326")

# ── Categorise nodes ──────────────────────────────────────────────────────────
built     = results[results["x_built"] == 1].copy()
existing  = nodes[nodes["is_existing_charger"] == 1].copy()
feasible  = nodes[nodes["is_feasible_location"] == 1].copy()

built_coords = set(zip(built["lon"].round(6), built["lat"].round(6)))
unselected = feasible[
    ~feasible.apply(
        lambda r: (round(r["lon"], 6), round(r["lat"], 6)) in built_coords, axis=1
    )
]

print(f"  Built: {len(built)}  |  Unselected feasible: {len(unselected)}  |  "
      f"Existing chargers: {len(existing)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
print("Plotting …")
fig, ax = plt.subplots(figsize=(22, 14), facecolor="#111111")
ax.set_facecolor("#111111")
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_aspect("equal")
ax.axis("off")

# Road network
print(f"  Drawing {len(roads):,} road edges …")
for _, row in roads.iterrows():
    ax.plot(*row.geometry.xy, color="#333333", linewidth=0.08, alpha=0.6, zorder=1)

# Unselected feasible locations
ax.scatter(
    unselected["lon"], unselected["lat"],
    s=4, color="#555555", alpha=0.4, zorder=2, label="Feasible (not selected)"
)

# Existing chargers
ax.scatter(
    existing["lon"], existing["lat"],
    s=18, color="#4499ff", alpha=0.8, zorder=3, label="Existing charger", marker="^"
)

# Built stations — coloured by charger count, sized by λ_k
for c_val in sorted(built["c_built"].unique()):
    sub = built[built["c_built"] == c_val]
    color = C_COLORS.get(int(c_val), "#ffffff")
    # Size: base 60 + 30 × log(λ_k)  (ensures visibility even at low λ)
    sizes = 60 + 30 * np.log1p(sub["lambda_k"].clip(lower=0.01))
    ax.scatter(
        sub["lon"], sub["lat"],
        s=sizes, color=color, alpha=0.95, zorder=5,
        edgecolors="white", linewidths=0.5,
        label=f"{int(c_val)} charger{'s' if c_val > 1 else ''}"
    )

# W_q halo rings on built stations (radius ∝ log W_q, skip penalty=45)
normal_built = built[built["wq_minutes"] < 45]
for _, row in normal_built.iterrows():
    wq_norm = np.clip(np.log1p(row["wq_minutes"]) / np.log1p(100), 0, 1)
    ring_size = 120 + 200 * wq_norm
    ax.scatter(
        row["lon"], row["lat"],
        s=ring_size, facecolors="none",
        edgecolors="white", linewidths=0.8, alpha=0.35, zorder=4
    )

# Saturated station marker (W_q = 45 penalty)
saturated = built[built["wq_minutes"] >= 45]
if len(saturated):
    ax.scatter(
        saturated["lon"], saturated["lat"],
        s=180, color="#ff00ff", alpha=0.9, zorder=6,
        edgecolors="white", linewidths=1.0, marker="X",
        label=f"Saturated (W_q=45 min)"
    )

# ── Stats annotation ──────────────────────────────────────────────────────────
charger_dist = dict(built["c_built"].value_counts().sort_index())
dist_str = "  ".join(f"c={k}: {v}" for k, v in charger_dist.items())
normal_wq = built[built["wq_minutes"] < 45]["wq_minutes"]

gamma_val = float(built["gamma"].iloc[0]) if "gamma" in built.columns else 0.1
ax.text(
    0.01, 0.99,
    f"Congestion-Extended Benders Solution\n"
    f"  γ = {gamma_val}  β = 1.0  φ = 0\n"
    f"  Stations built : {len(built)}\n"
    f"  Chargers total : {int(built['c_built'].sum())}\n"
    f"  Total kW       : {int((built['c_built']*built['p_built_kw']).sum()):,}\n"
    f"  {dist_str}\n"
    f"  Mean W_q (stable): {normal_wq.mean():.1f} min\n"
    f"  Saturated stations: {len(saturated)}",
    transform=ax.transAxes,
    color="white", fontsize=10, va="top", fontfamily="monospace",
    bbox=dict(facecolor="#111111", alpha=0.85,
              edgecolor="#555555", boxstyle="round,pad=0.5"),
    zorder=10
)

# ── Legend ────────────────────────────────────────────────────────────────────
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, labels,
    loc="lower right", framealpha=0.5,
    facecolor="#111111", edgecolor="#555555",
    labelcolor="white", fontsize=10
)

ax.set_title(
    f"Spain EV Fast-Charging Network — Congestion-Optimal Solution  "
    f"({len(built)} stations, {int(built['c_built'].sum())} chargers, γ={gamma_val})",
    color="white", fontsize=14, fontweight="bold", pad=12
)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="#111111")
print(f"Saved → {OUT_PNG}")
