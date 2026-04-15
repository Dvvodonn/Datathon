"""
network_coverage_full.py
========================
Three-panel visualization comparing alpha=45 optimal paths against:
  Left   — MITMA 2022 flow segments (7,279 measured roads, 1.5km buffer)
  Centre — Full OSM interurban network (326,183 edges, 200m buffer)
  Right  — MITMA segments coloured by AADT, covered ones highlighted
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely.geometry import LineString

EDGES_CSV    = "models/routes_250_alpha45_edges.csv"
FLOW_GEOJSON = "data_main/traffic/imd_total_por_tramo.geojson"
OSM_GPKG     = "data/raw/road_network/spain_interurban_edges.gpkg"
OSM_NODES    = "data/raw/road_network/spain_interurban_nodes.gpkg"
OUT_PNG      = "visualizations/network_coverage_full.png"

CRS_M   = "EPSG:3857"
CRS_DEG = "EPSG:4326"
XLIM    = (-9.5, 4.5)
YLIM    = (35.8, 44.0)

# ── load optimal edges ────────────────────────────────────────────────────────
print("Loading data...")
opt = pd.read_csv(EDGES_CSV)

def make_corridors(buffer_m):
    geoms = [LineString([(r.lon_a, r.lat_a), (r.lon_b, r.lat_b)]) for r in opt.itertuples()]
    c = gpd.GeoDataFrame({"use": opt.usage_count.values}, geometry=geoms, crs=CRS_DEG)
    c = c.to_crs(CRS_M)
    c["geometry"] = c.geometry.buffer(buffer_m)
    return c.to_crs(CRS_DEG)

# ── MITMA coverage (1.5 km buffer) ───────────────────────────────────────────
print("MITMA coverage...")
flow = gpd.read_file(FLOW_GEOJSON).to_crs(CRS_DEG)
flow = flow[flow["aadt_total"].notna() & (flow["aadt_total"] > 0)].copy()
corridors_15 = make_corridors(1_500)
flow_idx = flow[["geometry"]].reset_index().rename(columns={"index": "fid"})
joined_f = gpd.sjoin(flow_idx, corridors_15[["geometry"]], how="left", predicate="intersects")
flow["covered"] = flow.index.isin(set(joined_f.dropna(subset=["index_right"]).fid))
flow_m = flow.to_crs(CRS_M)
mitma_segs_pct  = 100 * flow.covered.sum() / len(flow)
mitma_km_pct    = 100 * flow_m[flow.covered].length.sum() / flow_m.length.sum()
flow_cov        = flow[flow.covered]["aadt_total"]
total_aadt_km   = (flow["aadt_total"] * flow_m.length).sum()
covered_aadt_km = (flow.loc[flow.covered,"aadt_total"] * flow_m[flow.covered].length).sum()
mitma_traffic_pct = 100 * covered_aadt_km / total_aadt_km

# ── OSM coverage (200 m buffer) ───────────────────────────────────────────────
print("OSM coverage...")
osm = gpd.read_file(OSM_GPKG).to_crs(CRS_DEG)
osm_m = osm.to_crs(CRS_M)
corridors_02 = make_corridors(200)
osm_idx = osm[["geometry"]].reset_index().rename(columns={"index": "oid"})
joined_o = gpd.sjoin(osm_idx, corridors_02[["geometry"]], how="left", predicate="intersects")
osm["covered"] = osm.index.isin(set(joined_o.dropna(subset=["index_right"]).oid))
osm_segs_pct = 100 * osm.covered.sum() / len(osm)
osm_km_pct   = 100 * osm_m[osm.covered].length.sum() / osm_m.length.sum()

print(f"MITMA: {mitma_segs_pct:.1f}% segs | {mitma_km_pct:.1f}% km | {mitma_traffic_pct:.1f}% traffic-wtd")
print(f"OSM:   {osm_segs_pct:.1f}% segs | {osm_km_pct:.1f}% km")

# ── optimal path GDF for plotting ─────────────────────────────────────────────
opt_geoms = [LineString([(r.lon_a, r.lat_a), (r.lon_b, r.lat_b)]) for r in opt.itertuples()]
opt_gdf = gpd.GeoDataFrame({"usage": opt.usage_count.values}, geometry=opt_geoms, crs=CRS_DEG)
norm_use = mcolors.LogNorm(vmin=1, vmax=opt.usage_count.max())
cmap_use = plt.cm.plasma

# ── AADT colormap ─────────────────────────────────────────────────────────────
cmap_aadt = mcolors.LinearSegmentedColormap.from_list(
    "flow", ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
)
vmin_a = max(flow["aadt_total"].quantile(0.05), 1)
vmax_a = flow["aadt_total"].quantile(0.99)
norm_a = mcolors.LogNorm(vmin=vmin_a, vmax=vmax_a)

# ── plot ──────────────────────────────────────────────────────────────────────
print("Plotting...")
fig, axes = plt.subplots(1, 3, figsize=(30, 11), facecolor="#0f1117")
fig.suptitle(
    "Alpha=45 Optimal Path Coverage vs Road Networks  (250 km EV range)",
    color="white", fontsize=15, fontweight="bold", y=0.99
)

for ax in axes:
    ax.set_facecolor("#0f1117")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect("equal"); ax.axis("off")

# ─── Panel 1: MITMA flow network ─────────────────────────────────────────────
ax = axes[0]
ax.set_title("MITMA 2022 measured roads  (1.5 km buffer)", color="white", fontsize=10, pad=6)

for _, row in flow[~flow.covered].iterrows():
    ax.plot(*row.geometry.xy, color="#2a2a2a", linewidth=0.4, zorder=1)
for _, row in flow[flow.covered].iterrows():
    c = cmap_aadt(norm_a(max(row["aadt_total"], 1)))
    ax.plot(*row.geometry.xy, color=c, linewidth=0.8, alpha=0.9, zorder=2)

ax.text(0.02, 0.98,
    f"Segments: {flow.covered.sum():,}/{len(flow):,}  ({mitma_segs_pct:.1f}%)\n"
    f"Length:   {mitma_km_pct:.1f}% of {flow_m.length.sum()/1000:,.0f} km\n"
    f"Traffic-weighted: {mitma_traffic_pct:.1f}%",
    transform=ax.transAxes, color="white", fontsize=8.5, va="top",
    bbox=dict(facecolor="#0f1117", alpha=0.8, edgecolor="#555", boxstyle="round,pad=0.4"))

# ─── Panel 2: Full OSM network ───────────────────────────────────────────────
ax = axes[1]
ax.set_title("Full OSM interurban network  (200 m buffer)", color="white", fontsize=10, pad=6)

for _, row in osm[~osm.covered].iterrows():
    ax.plot(*row.geometry.xy, color="#1e1e1e", linewidth=0.15, zorder=1)
for _, row in osm[osm.covered].iterrows():
    ax.plot(*row.geometry.xy, color="#00ccff", linewidth=0.5, alpha=0.7, zorder=2)

ax.text(0.02, 0.98,
    f"Segments: {osm.covered.sum():,}/{len(osm):,}  ({osm_segs_pct:.1f}%)\n"
    f"Length:   {osm_km_pct:.1f}% of {osm_m.length.sum()/1000:,.0f} km",
    transform=ax.transAxes, color="white", fontsize=8.5, va="top",
    bbox=dict(facecolor="#0f1117", alpha=0.8, edgecolor="#555", boxstyle="round,pad=0.4"))

# ─── Panel 3: Optimal paths coloured by usage ────────────────────────────────
ax = axes[2]
ax.set_title("Optimal paths — usage count (plasma)", color="white", fontsize=10, pad=6)

for _, row in osm[~osm.covered].iterrows():
    ax.plot(*row.geometry.xy, color="#1a1a1a", linewidth=0.15, zorder=1)
for _, row in opt_gdf.iterrows():
    c = cmap_use(norm_use(max(row["usage"], 1)))
    ax.plot(*row.geometry.xy, color=c, linewidth=0.6, alpha=0.85, zorder=2)

sm = plt.cm.ScalarMappable(cmap=cmap_use, norm=norm_use)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[2], fraction=0.025, pad=0.01, aspect=30)
cbar.set_label("City-pair paths using edge", color="white", fontsize=8)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
cbar.outline.set_edgecolor("white")

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="#0f1117")
print(f"Saved → {OUT_PNG}")
