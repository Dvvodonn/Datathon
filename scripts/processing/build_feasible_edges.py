"""
build_feasible_edges.py
-----------------------
Compute directed road distances between all pairs of nodes in nodes.csv
using the OSM interurban graph. Output: data/processed/feasible_edges.csv.gz

Runtime estimate: ~25-40 min
"""

import time
import csv
import gzip
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import igraph as ig
from shapely.geometry import Point

T0 = time.time()

def log(msg):
    elapsed = time.time() - T0
    print(f"[{elapsed:6.1f}s] {msg}", flush=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRAPHML   = "data/raw/road_routes/spain_interurban.graphml"
NODES_CSV = "data/processed/nodes.csv"
NODES_GPK = "data/raw/road_routes/spain_interurban_nodes.gpkg"
OUTPUT    = "data/processed/feasible_edges.csv.gz"
MAX_DIST_M = 500_000.0

# ---------------------------------------------------------------------------
# Step 1 — Load OSM graph into igraph
# ---------------------------------------------------------------------------
log("Loading GraphML with networkx (~3 min)...")
nx_graph = nx.read_graphml(GRAPHML)
log(f"NetworkX: {nx_graph.number_of_nodes():,} nodes, {nx_graph.number_of_edges():,} edges")

log("Converting to igraph...")
# NetworkX node ids are strings (OSM integer ids); map to igraph integer indices
nx_nodes = list(nx_graph.nodes())                          # list of str osm ids
nx_node_to_idx = {nid: i for i, nid in enumerate(nx_nodes)}

edges   = []
lengths = []
for u, v, data in nx_graph.edges(data=True):
    # Edge length attribute is called "length" in this graphml
    try:
        l = float(data.get("length", data.get("length_m", 1.0)))
    except (TypeError, ValueError):
        l = 1.0
    edges.append((nx_node_to_idx[u], nx_node_to_idx[v]))
    lengths.append(l)

ig_graph = ig.Graph(n=len(nx_nodes), edges=edges, directed=True)
ig_graph.es["length_m"] = lengths
log(f"igraph: {ig_graph.vcount():,} vertices, {ig_graph.ecount():,} edges")

# ---------------------------------------------------------------------------
# Step 2 — Snap 8,897 nodes to nearest OSM road node
# ---------------------------------------------------------------------------
log("Loading nodes.csv...")
nodes_df = pd.read_csv(NODES_CSV)
log(f"nodes.csv: {len(nodes_df):,} rows")

log("Loading OSM nodes gpkg...")
osm_nodes = gpd.read_file(NODES_GPK)   # osmid col, EPSG:4326
log(f"OSM nodes: {len(osm_nodes):,}, CRS={osm_nodes.crs}")

# Project to metric CRS for accurate snap distances
TARGET_CRS = "EPSG:25830"
log("Projecting to EPSG:25830...")
osm_proj = osm_nodes[["osmid", "geometry"]].to_crs(TARGET_CRS)

nodes_gdf = gpd.GeoDataFrame(
    nodes_df.reset_index(drop=True),
    geometry=[Point(row.lon, row.lat) for _, row in nodes_df.iterrows()],
    crs="EPSG:4326"
).to_crs(TARGET_CRS)

log("Snapping to nearest OSM node (sjoin_nearest)...")
snapped = gpd.sjoin_nearest(
    nodes_gdf,
    osm_proj,
    how="left",
    distance_col="snap_dist_m"
)
# sjoin_nearest may duplicate rows if equidistant — keep first match per input row
snapped = snapped.groupby(level=0).first()
log("Snap done")

far = snapped[snapped["snap_dist_m"] > 5000]
if len(far):
    log(f"WARNING: {len(far)} nodes snapped >5 km (island/disconnected):")
    for _, r in far.iterrows():
        log(f"  ({r.lat:.4f},{r.lon:.4f}) dist={r.snap_dist_m/1000:.1f}km")

# Map each of our 8,897 nodes to an igraph vertex index
# osmid in gpkg is int64; nx_node_to_idx keys are strings
log("Mapping osmid → igraph vertex index...")
snapped_igraph_idx = []
missing = 0
for osmid in snapped["osmid"]:
    idx = nx_node_to_idx.get(str(int(osmid)))
    if idx is None:
        missing += 1
    snapped_igraph_idx.append(idx)

if missing:
    log(f"WARNING: {missing} nodes could not be mapped to igraph vertex")

valid_mask  = [idx is not None for idx in snapped_igraph_idx]
valid_pos   = [i for i, m in enumerate(valid_mask) if m]   # position in nodes_df
valid_vids  = [snapped_igraph_idx[i] for i in valid_pos]    # igraph vertex ids (may have duplicates)
N           = len(valid_vids)
log(f"{N} nodes mapped to igraph vertices")

# Deduplicate vertex ids for igraph (it rejects duplicate sources/targets)
seen = {}
unique_vids = []
for vid in valid_vids:
    if vid not in seen:
        seen[vid] = len(unique_vids)
        unique_vids.append(vid)
# Map each of our N node positions to its index in unique_vids
node_to_uniq = [seen[vid] for vid in valid_vids]
U = len(unique_vids)
log(f"{U} unique igraph vertex ids (from {N} nodes)")

# ---------------------------------------------------------------------------
# Step 3 — Directed all-pairs shortest paths
# ---------------------------------------------------------------------------
log(f"Computing {U}x{U} = {U*U:,} directed distances (igraph)...")
dist_matrix = ig_graph.distances(
    source=unique_vids,
    target=unique_vids,
    weights="length_m",
    mode="out",
)
log("Distance matrix computed")

# ---------------------------------------------------------------------------
# Step 4 — Flatten, filter, write
# ---------------------------------------------------------------------------
# Pre-extract attribute arrays (aligned with valid_pos)
lons      = nodes_df["lon"].values[valid_pos]
lats      = nodes_df["lat"].values[valid_pos]
is_city   = nodes_df["is_city"].values[valid_pos].astype(np.int8)
is_feas   = nodes_df["is_feasible_location"].values[valid_pos].astype(np.int8)
is_chrg   = nodes_df["is_existing_charger"].values[valid_pos].astype(np.int8)

log(f"Flattening matrix and writing to {OUTPUT}...")
rows_written = 0
INF = float("inf")

with gzip.open(OUTPUT, "wt", newline="") as gz:
    writer = csv.writer(gz)
    writer.writerow([
        "lon_a", "lat_a",
        "lon_b", "lat_b",
        "a_is_city", "a_is_feasible", "a_is_charger",
        "b_is_city", "b_is_feasible", "b_is_charger",
        "distance_km",
    ])

    for i in range(N):
        ui = node_to_uniq[i]
        row = dist_matrix[ui]
        for j in range(N):
            if i == j:
                continue
            d = row[node_to_uniq[j]]
            if d == INF or d >= MAX_DIST_M:
                continue
            writer.writerow([
                round(float(lons[i]), 6), round(float(lats[i]), 6),
                round(float(lons[j]), 6), round(float(lats[j]), 6),
                int(is_city[i]), int(is_feas[i]), int(is_chrg[i]),
                int(is_city[j]), int(is_feas[j]), int(is_chrg[j]),
                round(d / 1000.0, 4),
            ])
            rows_written += 1

        if (i + 1) % 200 == 0:
            pct = (i + 1) / N * 100
            log(f"  {i+1}/{N} ({pct:.1f}%) — {rows_written:,} edges written")

log(f"DONE: {rows_written:,} edges → {OUTPUT}")
log(f"Total elapsed: {time.time()-T0:.1f}s")
