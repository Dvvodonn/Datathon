"""
road_flow_validation.py
=======================
Two-panel validation map for data_main/road_edges_flow.csv:

  Left  — tier assignment: cyan=tier1, yellow=tier2, grey=tier3
  Right — effective AADT: white=low, yellow→red=high  (log scale)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

FLOW_CSV = "data_main/road_edges_flow.csv"
OSM_GPKG = "data/raw/road_network/spain_interurban_edges.gpkg"
OUT_PNG  = "visualizations/road_flow_validation.png"

CRS_DEG = "EPSG:4326"
XLIM = (-9.5, 4.5)
YLIM = (35.8, 44.0)

print("Loading …")
df  = pd.read_csv(FLOW_CSV)
osm = gpd.read_file(OSM_GPKG).to_crs(CRS_DEG)

# Attach AADT columns to OSM geometry (same order)
assert len(df) == len(osm), "Row count mismatch"
osm["effective_aadt"] = df["effective_aadt"].values
osm["aadt_source"]    = df["aadt_source"].values

# ── Tier colour mapping ───────────────────────────────────────────────────────
TIER_COLORS = {
    "tier1_name_spatial": "#00ccff",
    "tier2_road_idw":     "#ffcc00",
    "tier3_knn":          "#444444",
}
osm["tier_color"] = osm["aadt_source"].map(TIER_COLORS)

# ── AADT colormap ─────────────────────────────────────────────────────────────
cmap = mcolors.LinearSegmentedColormap.from_list(
    "flow", ["#ffffff", "#ffff66", "#ff8800", "#cc0000"]
)
vmin = max(osm["effective_aadt"].quantile(0.05), 1)
vmax = osm["effective_aadt"].quantile(0.99)
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

# ── Plot ──────────────────────────────────────────────────────────────────────
print("Plotting …")
fig, axes = plt.subplots(1, 2, figsize=(28, 10), facecolor="#111111")
fig.suptitle(
    "Spain Interurban Road Network — AADT Flow Assignment  (all 326,183 OSM edges)",
    color="white", fontsize=14, fontweight="bold", y=0.99
)

for ax in axes:
    ax.set_facecolor("#111111")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect("equal"); ax.axis("off")

# ─── Left panel: tier assignment ─────────────────────────────────────────────
ax = axes[0]
ax.set_title("Tier assignment  (cyan=direct MITMA  yellow=road IDW  grey=KNN)",
             color="white", fontsize=10, pad=6)

# Draw tier3 first (background), then tier1/2 on top
for tier, lw, zorder in [("tier3_knn", 0.12, 1),
                          ("tier2_road_idw", 0.6, 2),
                          ("tier1_name_spatial", 0.5, 3)]:
    sub = osm[osm["aadt_source"] == tier]
    color = TIER_COLORS[tier]
    alpha = 0.25 if tier == "tier3_knn" else 0.85
    for _, row in sub.iterrows():
        ax.plot(*row.geometry.xy, color=color, linewidth=lw, alpha=alpha, zorder=zorder)

counts = osm["aadt_source"].value_counts()
total  = len(osm)
legend = [
    Line2D([0],[0], color="#00ccff", lw=1.5,
           label=f"Tier 1 – name+spatial  ({counts.get('tier1_name_spatial',0):,},  "
                 f"{100*counts.get('tier1_name_spatial',0)/total:.1f}%)"),
    Line2D([0],[0], color="#ffcc00", lw=1.5,
           label=f"Tier 2 – road IDW      ({counts.get('tier2_road_idw',0):,},  "
                 f"{100*counts.get('tier2_road_idw',0)/total:.1f}%)"),
    Line2D([0],[0], color="#888888", lw=1.5,
           label=f"Tier 3 – global KNN    ({counts.get('tier3_knn',0):,},  "
                 f"{100*counts.get('tier3_knn',0)/total:.1f}%)"),
]
ax.legend(handles=legend, loc="lower right", framealpha=0.5,
          facecolor="#111111", edgecolor="#555555", labelcolor="white", fontsize=9)

# ─── Right panel: effective AADT ─────────────────────────────────────────────
ax = axes[1]
ax.set_title("Effective AADT  (log scale: white=low  yellow→red=high)",
             color="white", fontsize=10, pad=6)

osm_sorted = osm.sort_values("effective_aadt")

print(f"  Drawing {len(osm_sorted):,} edges …")
for _, row in osm_sorted.iterrows():
    v = max(row["effective_aadt"], 1)
    c  = cmap(norm(v))
    lw = 0.12 + 0.6 * norm(v)
    ax.plot(*row.geometry.xy, color=c, linewidth=lw, alpha=0.75, zorder=1)

# Stats annotation
for tier_src in ["tier1_name_spatial", "tier2_road_idw", "tier3_knn"]:
    sub = osm[osm["aadt_source"] == tier_src]["effective_aadt"]
    print(f"  {tier_src}: mean={sub.mean():,.0f}  median={sub.median():,.0f}")

ax.text(0.01, 0.99,
    f"AADT statistics\n"
    f"  Overall mean  : {osm['effective_aadt'].mean():,.0f} veh/day\n"
    f"  Overall median: {osm['effective_aadt'].median():,.0f} veh/day\n"
    f"  Tier 1 median : {osm[osm['aadt_source']=='tier1_name_spatial']['effective_aadt'].median():,.0f}\n"
    f"  Tier 2 median : {osm[osm['aadt_source']=='tier2_road_idw']['effective_aadt'].median():,.0f}\n"
    f"  Tier 3 median : {osm[osm['aadt_source']=='tier3_knn']['effective_aadt'].median():,.0f}",
    transform=ax.transAxes, color="white", fontsize=9, va="top", fontfamily="monospace",
    bbox=dict(facecolor="#111111", alpha=0.8, edgecolor="#555555", boxstyle="round,pad=0.4"))

# Colorbar
import matplotlib.cm as mcm
sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.01, aspect=35)
cbar.set_label("Effective AADT — vehicles/day", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
cbar.outline.set_edgecolor("white")
tick_vals = [1000, 5000, 10000, 25000, 50000, 100000]
tick_vals = [t for t in tick_vals if vmin <= t <= vmax]
cbar.set_ticks(tick_vals)
cbar.set_ticklabels([f"{t:,}" for t in tick_vals])

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="#111111")
print(f"Saved → {OUT_PNG}")
