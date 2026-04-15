"""
flow_coverage_analysis.py
=========================
Compare optimal alpha=45 route edges with MITMA 2022 traffic flow segments.

Statistics:
  - What % of the flow network (by segment count and km) is covered by optimal paths
  - What % of optimal path edges have flow data
  - Traffic-weighted coverage (AADT * length)

Visualization:
  Left panel  — Full flow network coloured by AADT, optimal path overlay in cyan
  Right panel — Only flow segments that are used in optimal paths, coloured by AADT
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from shapely.geometry import LineString

EDGES_CSV   = "models/routes_250_alpha45_edges.csv"
FLOW_GEOJSON = "data_main/traffic/imd_total_por_tramo.geojson"
OUT_PNG     = "visualizations/flow_coverage_alpha45.png"

BUFFER_M    = 1_500   # 1.5 km corridor — tight enough to avoid parallel roads
CRS_M       = "EPSG:3857"
CRS_DEG     = "EPSG:4326"

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
edges_df = pd.read_csv(EDGES_CSV)
flow_gdf = gpd.read_file(FLOW_GEOJSON).to_crs(CRS_DEG)
flow_gdf = flow_gdf[flow_gdf["aadt_total"].notna() & (flow_gdf["aadt_total"] > 0)].copy()
print(f"  {len(edges_df):,} optimal edges  |  {len(flow_gdf):,} flow segments with AADT>0")

# ── build corridors from optimal edges ────────────────────────────────────────
print("Building corridors...")
geoms = [LineString([(r.lon_a, r.lat_a), (r.lon_b, r.lat_b)]) for r in edges_df.itertuples()]
corridors = gpd.GeoDataFrame(
    {"usage_count": edges_df["usage_count"].values},
    geometry=geoms, crs=CRS_DEG
).to_crs(CRS_M)
corridors["geometry"] = corridors.geometry.buffer(BUFFER_M)
corridors = corridors.to_crs(CRS_DEG)

# ── spatial join: which flow segments lie within any corridor ─────────────────
print("Spatial joining...")
flow_idx = flow_gdf[["geometry", "aadt_total", "shape_length_m"]].copy().reset_index().rename(columns={"index": "flow_idx"})
joined = gpd.sjoin(flow_idx, corridors[["geometry", "usage_count"]], how="left", predicate="intersects")

# Max usage per flow segment
coverage = joined.groupby("flow_idx").agg(
    max_usage=("usage_count", "max")
).reset_index()
flow_gdf = flow_gdf.merge(coverage, left_index=True, right_on="flow_idx", how="left")
flow_gdf["max_usage"] = flow_gdf["max_usage"].fillna(0)
flow_gdf["covered"] = flow_gdf["max_usage"] > 0

# ── statistics ────────────────────────────────────────────────────────────────
flow_m = flow_gdf.to_crs(CRS_M)
total_segs   = len(flow_gdf)
covered_segs = flow_gdf["covered"].sum()

total_km   = flow_m.geometry.length.sum() / 1000
covered_km = flow_m[flow_gdf["covered"]].geometry.length.sum() / 1000

# Traffic-weighted: AADT × length
flow_gdf["aadt_km"] = flow_gdf["aadt_total"] * flow_m.geometry.length / 1000
total_aadt_km   = flow_gdf["aadt_km"].sum()
covered_aadt_km = flow_gdf.loc[flow_gdf["covered"], "aadt_km"].sum()

print(f"\n{'='*55}")
print(f"Coverage of MITMA 2022 flow network by alpha=45 optimal paths")
print(f"{'='*55}")
print(f"  Segments covered : {covered_segs:,} / {total_segs:,}  ({100*covered_segs/total_segs:.1f}%)")
print(f"  Length covered   : {covered_km:,.0f} km / {total_km:,.0f} km  ({100*covered_km/total_km:.1f}%)")
print(f"  Traffic-weighted : {100*covered_aadt_km/total_aadt_km:.1f}%  (AADT × km)")
print(f"\n  Avg AADT — covered segments : {flow_gdf.loc[flow_gdf['covered'],'aadt_total'].mean():,.0f}")
print(f"  Avg AADT — all segments     : {flow_gdf['aadt_total'].mean():,.0f}")
print(f"{'='*55}\n")

# ── plot ──────────────────────────────────────────────────────────────────────
print("Plotting...")
XLIM = (-9.5, 4.5)
YLIM = (35.8, 44.0)

cmap = mcolors.LinearSegmentedColormap.from_list(
    "flow", ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
)
vmin = max(flow_gdf["aadt_total"].quantile(0.05), 1)
vmax = flow_gdf["aadt_total"].quantile(0.99)
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(1, 2, figsize=(22, 10), facecolor="#0f1117")
fig.suptitle(
    "MITMA 2022 Traffic Flow vs Alpha=45 Optimal Paths (250km range)",
    color="white", fontsize=14, fontweight="bold", y=0.98
)

for ax in axes:
    ax.set_facecolor("#0f1117")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect("equal"); ax.axis("off")

# ── Left: full network + optimal overlay ─────────────────────────────────────
ax = axes[0]
ax.set_title("Full flow network  +  optimal path coverage", color="white", fontsize=11, pad=8)

# All segments coloured by AADT
for _, row in flow_gdf.iterrows():
    c = cmap(norm(max(row["aadt_total"], 1)))
    lw = 0.3 + 1.2 * norm(max(row["aadt_total"], 1))
    ax.plot(*row.geometry.xy, color=c, linewidth=lw, alpha=0.6, zorder=2)

# Covered segments highlighted in cyan
covered_gdf = flow_gdf[flow_gdf["covered"]]
for _, row in covered_gdf.iterrows():
    ax.plot(*row.geometry.xy, color="#00ffff", linewidth=1.2, alpha=0.85, zorder=3)

ax.text(0.02, 0.98,
    f"Covered: {covered_segs:,}/{total_segs:,} segments ({100*covered_segs/total_segs:.1f}%)\n"
    f"Length: {covered_km:,.0f}/{total_km:,.0f} km ({100*covered_km/total_km:.1f}%)\n"
    f"Traffic-weighted: {100*covered_aadt_km/total_aadt_km:.1f}%",
    transform=ax.transAxes, color="white", fontsize=9,
    verticalalignment="top",
    bbox=dict(facecolor="#0f1117", alpha=0.7, edgecolor="#444444", boxstyle="round,pad=0.4")
)

# Legend patches
from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], color="#d7191c", lw=2, label="High AADT (flow network)"),
    Line2D([0],[0], color="#1a9641", lw=2, label="Low AADT (flow network)"),
    Line2D([0],[0], color="#00ffff", lw=2, label="Covered by optimal paths"),
]
ax.legend(handles=handles, loc="lower right", framealpha=0.3,
          facecolor="#0f1117", edgecolor="#444444", labelcolor="white", fontsize=9)

# ── Right: only covered segments, coloured by AADT ───────────────────────────
ax = axes[1]
ax.set_title("Covered segments coloured by AADT", color="white", fontsize=11, pad=8)

uncovered = flow_gdf[~flow_gdf["covered"]]
for _, row in uncovered.iterrows():
    ax.plot(*row.geometry.xy, color="#1e1e1e", linewidth=0.3, alpha=0.5, zorder=1)

for _, row in covered_gdf.iterrows():
    c = cmap(norm(max(row["aadt_total"], 1)))
    lw = 0.5 + 1.5 * norm(max(row["aadt_total"], 1))
    ax.plot(*row.geometry.xy, color=c, linewidth=lw, alpha=0.95, zorder=2)

# Colorbar
sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[1], fraction=0.025, pad=0.01, aspect=30)
cbar.set_label("AADT — vehicles/day (2022)", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
cbar.outline.set_edgecolor("white")
tick_vals = [1000, 5000, 10000, 25000, 50000, 100000]
tick_vals = [t for t in tick_vals if vmin <= t <= vmax]
cbar.set_ticks(tick_vals)
cbar.set_ticklabels([f"{t:,}" for t in tick_vals])

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="#0f1117")
print(f"Saved → {OUT_PNG}")
