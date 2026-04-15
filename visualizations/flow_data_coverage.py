"""
flow_data_coverage.py
=====================
Two intersection analyses:
  1. Of the alpha=45 optimal path edges — what fraction has MITMA flow data?
  2. Of the full OSM interurban network — what fraction has MITMA flow data?

Visualization:
  Full OSM network — white if no flow data, yellow→red by AADT if flow data exists
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import LineString

EDGES_CSV    = "models/routes_250_alpha45_edges.csv"
FLOW_GEOJSON = "data_main/traffic/imd_total_por_tramo.geojson"
OSM_GPKG     = "data/raw/road_network/spain_interurban_edges.gpkg"
OUT_PNG      = "visualizations/flow_data_coverage.png"

CRS_M   = "EPSG:3857"
CRS_DEG = "EPSG:4326"
XLIM    = (-9.5, 4.5)
YLIM    = (35.8, 44.0)

# Buffer around MITMA segments to find matching OSM/optimal edges
# 500m: tight enough to avoid parallel roads, wide enough to bridge alignment gaps
BUFFER_M = 500

# ── load ─────────────────────────────────────────────────────────────────────
print("Loading data...")
opt  = pd.read_csv(EDGES_CSV)
flow = gpd.read_file(FLOW_GEOJSON).to_crs(CRS_DEG)
flow = flow[flow["aadt_total"].notna() & (flow["aadt_total"] > 0)].copy()
osm  = gpd.read_file(OSM_GPKG).to_crs(CRS_DEG)
print(f"  {len(opt):,} optimal edges | {len(flow):,} flow segments | {len(osm):,} OSM edges")

# ── build MITMA corridors ────────────────────────────────────────────────────
print(f"Building {BUFFER_M}m corridors around MITMA segments...")
flow_m = flow.to_crs(CRS_M).copy()
flow_m["geometry"] = flow_m.geometry.buffer(BUFFER_M)
flow_corridors = flow_m.to_crs(CRS_DEG)[["aadt_total", "geometry"]].reset_index().rename(columns={"index": "fid"})

# ── Intersection 1: optimal paths vs flow data ────────────────────────────────
print("Intersection 1: optimal paths vs MITMA flow...")
opt_geoms = [LineString([(r.lon_a, r.lat_a), (r.lon_b, r.lat_b)]) for r in opt.itertuples()]
opt_gdf   = gpd.GeoDataFrame({"usage": opt.usage_count.values}, geometry=opt_geoms, crs=CRS_DEG)
opt_gdf_idx = opt_gdf.reset_index().rename(columns={"index": "oid"})

joined_opt = gpd.sjoin(opt_gdf_idx[["oid","geometry"]], flow_corridors[["fid","aadt_total","geometry"]],
                       how="left", predicate="intersects")
opt_coverage = joined_opt.groupby("oid")["aadt_total"].max().reset_index()
opt_gdf = opt_gdf.merge(opt_coverage, left_index=True, right_on="oid", how="left")
opt_gdf["has_flow"] = opt_gdf["aadt_total"].notna()

opt_m = opt_gdf.to_crs(CRS_M)
opt_cov_segs = opt_gdf["has_flow"].sum()
opt_tot_segs = len(opt_gdf)
opt_cov_km   = opt_m[opt_gdf["has_flow"]].geometry.length.sum() / 1000
opt_tot_km   = opt_m.geometry.length.sum() / 1000

# usage-weighted
opt_cov_usage = opt_gdf.loc[opt_gdf["has_flow"], "usage"].sum()
opt_tot_usage = opt_gdf["usage"].sum()

print(f"""
  Optimal paths with flow data:
    Segments : {opt_cov_segs:,} / {opt_tot_segs:,}  ({100*opt_cov_segs/opt_tot_segs:.1f}%)
    Length   : {opt_cov_km:,.0f} km / {opt_tot_km:,.0f} km  ({100*opt_cov_km/opt_tot_km:.1f}%)
    Usage-wtd: {100*opt_cov_usage/opt_tot_usage:.1f}%  (city-pair path traversals)
""")

# ── Intersection 2: full OSM network vs flow data ─────────────────────────────
print("Intersection 2: full OSM network vs MITMA flow...")
osm_idx = osm[["geometry"]].reset_index().rename(columns={"index": "oid"})
joined_osm = gpd.sjoin(osm_idx, flow_corridors[["fid","aadt_total","geometry"]],
                       how="left", predicate="intersects")
osm_coverage = joined_osm.groupby("oid")["aadt_total"].max().reset_index()
osm = osm.merge(osm_coverage, left_index=True, right_on="oid", how="left")
osm["has_flow"] = osm["aadt_total"].notna()

osm_m = osm.to_crs(CRS_M)
osm_cov_segs = osm["has_flow"].sum()
osm_tot_segs = len(osm)
osm_cov_km   = osm_m[osm["has_flow"]].geometry.length.sum() / 1000
osm_tot_km   = osm_m.geometry.length.sum() / 1000

print(f"""
  Full OSM network with flow data:
    Segments : {osm_cov_segs:,} / {osm_tot_segs:,}  ({100*osm_cov_segs/osm_tot_segs:.1f}%)
    Length   : {osm_cov_km:,.0f} km / {osm_tot_km:,.0f} km  ({100*osm_cov_km/osm_tot_km:.1f}%)
""")

# ── plot ──────────────────────────────────────────────────────────────────────
print("Plotting...")
cmap = mcolors.LinearSegmentedColormap.from_list("flow", ["#ffff99", "#ffaa00", "#dd0000"])
vmin = max(osm.loc[osm["has_flow"], "aadt_total"].quantile(0.05), 1)
vmax = osm.loc[osm["has_flow"], "aadt_total"].quantile(0.99)
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(20, 15), facecolor="#111111")
ax.set_facecolor("#111111")
ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
ax.set_aspect("equal"); ax.axis("off")

# No flow: white (dim)
no_flow = osm[~osm["has_flow"]]
has_flow = osm[osm["has_flow"]].copy()
has_flow_sorted = has_flow.sort_values("aadt_total")  # draw low AADT first, high on top

print(f"  Drawing {len(no_flow):,} no-flow segments...")
for _, row in no_flow.iterrows():
    ax.plot(*row.geometry.xy, color="#ffffff", linewidth=0.15, alpha=0.25, zorder=1)

print(f"  Drawing {len(has_flow):,} flow segments...")
for _, row in has_flow_sorted.iterrows():
    c  = cmap(norm(max(row["aadt_total"], 1)))
    lw = 0.4 + 1.2 * norm(max(row["aadt_total"], 1))
    ax.plot(*row.geometry.xy, color=c, linewidth=lw, alpha=0.9, zorder=2)

# Stats box
ax.text(0.01, 0.99,
    f"OSM network with MITMA 2022 flow data\n"
    f"  Segments: {osm_cov_segs:,}/{osm_tot_segs:,}  ({100*osm_cov_segs/osm_tot_segs:.1f}%)\n"
    f"  Length:   {osm_cov_km:,.0f}/{osm_tot_km:,.0f} km  ({100*osm_cov_km/osm_tot_km:.1f}%)\n\n"
    f"Optimal paths with MITMA 2022 flow data\n"
    f"  Segments: {opt_cov_segs:,}/{opt_tot_segs:,}  ({100*opt_cov_segs/opt_tot_segs:.1f}%)\n"
    f"  Length:   {opt_cov_km:,.0f}/{opt_tot_km:,.0f} km  ({100*opt_cov_km/opt_tot_km:.1f}%)\n"
    f"  Usage-wtd:{100*opt_cov_usage/opt_tot_usage:.1f}%",
    transform=ax.transAxes, color="white", fontsize=10, va="top", fontfamily="monospace",
    bbox=dict(facecolor="#111111", alpha=0.85, edgecolor="#555555", boxstyle="round,pad=0.5"))

# Colorbar
import matplotlib.cm as mcm
sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.01, aspect=35)
cbar.set_label("AADT — vehicles/day (2022)", color="white", fontsize=10)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=9)
cbar.outline.set_edgecolor("white")
tick_vals = [1000, 5000, 10000, 25000, 50000, 100000]
tick_vals = [t for t in tick_vals if vmin <= t <= vmax]
cbar.set_ticks(tick_vals)
cbar.set_ticklabels([f"{t:,}" for t in tick_vals])

from matplotlib.lines import Line2D
legend = [
    Line2D([0],[0], color="white",   alpha=0.4, lw=1.5, label="No flow data"),
    Line2D([0],[0], color="#ffff99", lw=1.5,    label="Low AADT"),
    Line2D([0],[0], color="#ffaa00", lw=1.5,    label="Medium AADT"),
    Line2D([0],[0], color="#dd0000", lw=2,      label="High AADT"),
]
ax.legend(handles=legend, loc="lower right", framealpha=0.4,
          facecolor="#111111", edgecolor="#555555", labelcolor="white", fontsize=10)
ax.set_title("Spain Interurban Road Network — MITMA 2022 Traffic Flow Coverage",
             color="white", fontsize=14, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="#111111")
print(f"Saved → {OUT_PNG}")
