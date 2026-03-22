"""
patch_missing_nodes.py
----------------------
Finds POI nodes that were excluded from edges.csv because their snap distance
fell between MIN_SNAP_M and MAX_SNAP_M, snaps them, runs Dijkstra, and appends
the resulting edges to edges.csv and edges_250.csv.

Much faster than a full re-run — only processes the missing nodes.
"""

import time
import heapq
import multiprocessing as mp
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from collections import defaultdict

# ── config ────────────────────────────────────────────────────────────────────
EDGES_PATH    = "data/raw/road_network/spain_interurban_edges.gpkg"
POI_PATH      = "data_main/nodes.csv"
OUT_EDGES     = "data_main/edges.csv"
OUT_EDGES_250 = "data_main/edges_250.csv"

MIN_SNAP_M    = 5_000.0    # nodes already in edges.csv were snapped within this
MAX_SNAP_M    = 10_000.0   # upper bound — beyond this is genuinely bad data
DIST_LIMIT_M  = 500_000.0
EPSG_PROJ     = 25830
N_WORKERS     = max(1, mp.cpu_count() - 1)

ADJ       = None
N_OSM     = None
VIRT_IDXS = None


def build_base_adj(edges_gdf):
    all_ids = pd.concat([edges_gdf["u"], edges_gdf["v"]]).unique()
    osmid_to_idx = {int(oid): i for i, oid in enumerate(all_ids)}
    n = len(osmid_to_idx)
    adj = [[] for _ in range(n)]
    for row in edges_gdf.itertuples(index=False):
        u = osmid_to_idx[int(row.u)]
        v = osmid_to_idx[int(row.v)]
        tt = float(row.travel_time_s)
        lm = float(row.length_m)
        adj[u].append((v, tt, lm))
        if not row.oneway:
            adj[v].append((u, tt, lm))
    return adj, osmid_to_idx, n


def _worker_init(adj_shared, n_osm_shared, virt_idxs_shared):
    global ADJ, N_OSM, VIRT_IDXS
    ADJ       = adj_shared
    N_OSM     = n_osm_shared
    VIRT_IDXS = virt_idxs_shared


def dijkstra_dual(src_virt_idx):
    INF    = float("inf")
    n      = len(ADJ)
    best_t = [INF] * n
    best_l = [INF] * n
    best_t[src_virt_idx] = 0.0
    best_l[src_virt_idx] = 0.0
    pq = [(0.0, src_virt_idx)]
    while pq:
        t, u = heapq.heappop(pq)
        if t > best_t[u]:
            continue
        for v, tt, lm in ADJ[u]:
            new_t = t + tt
            if new_t < best_t[v]:
                best_t[v] = new_t
                best_l[v] = best_l[u] + lm
                heapq.heappush(pq, (new_t, v))
    return src_virt_idx, [best_t[vi] for vi in VIRT_IDXS], [best_l[vi] for vi in VIRT_IDXS]


def main():
    t0 = time.time()

    print("Loading OSM edges …")
    edges_gdf = gpd.read_file(EDGES_PATH)
    print(f"  {len(edges_gdf):,} OSM edges")

    print("Building adjacency list …")
    adj, osmid_to_idx, n_osm = build_base_adj(edges_gdf)
    print(f"  {n_osm:,} OSM nodes")

    print("Loading POI nodes …")
    poi_df    = pd.read_csv(POI_PATH)
    poi_proj  = gpd.GeoDataFrame(
        poi_df.reset_index(drop=True),
        geometry=[Point(r.lon, r.lat) for r in poi_df.itertuples()],
        crs="EPSG:4326",
    ).to_crs(EPSG_PROJ)
    edges_proj = edges_gdf.to_crs(EPSG_PROJ).reset_index(drop=True)

    print("Finding nodes with snap distance in (MIN, MAX] range …")
    joined_all = gpd.sjoin_nearest(
        poi_proj[["geometry"]],
        edges_proj[["geometry", "u", "v", "length_m", "travel_time_s", "oneway"]],
        how="left",
        distance_col="snap_dist_m",
    )
    joined_all = joined_all[~joined_all.index.duplicated(keep="first")]

    patch_mask = (joined_all["snap_dist_m"] > MIN_SNAP_M) & \
                 (joined_all["snap_dist_m"] <= MAX_SNAP_M)
    patch_indices = joined_all.index[patch_mask].tolist()

    if not patch_indices:
        print("No nodes in the snap range — nothing to patch.")
        return

    print(f"  {len(patch_indices)} node(s) to patch:")
    for i in patch_indices:
        name = poi_df.iloc[i].get("name", f"node_{i}") if "name" in poi_df.columns else f"node_{i}"
        print(f"    {name} (snap={joined_all.loc[i,'snap_dist_m']:.0f} m)")

    # ── snap patch nodes and build virtual nodes ──────────────────────────────
    # We need virtual indices for ALL poi nodes (existing + patch) so Dijkstra
    # targets are consistent with existing edges.csv.
    # Strategy: assign virtual indices starting at n_osm + len(poi_df).
    # Patch nodes get indices n_osm + patch_indices[k].

    adj.extend([] for _ in range(len(poi_df)))  # reserve space for all POIs

    patch_virt_idxs = []
    for i in patch_indices:
        row = joined_all.loc[i]
        snap_dist = float(row.snap_dist_m)
        edge_idx  = int(row.index_right)
        edge_geom = edges_proj.geometry.iloc[edge_idx]
        pt        = poi_proj.geometry.iloc[i]

        foot_t = edge_geom.project(pt, normalized=True)
        u_idx  = osmid_to_idx[int(row.u)]
        v_idx  = osmid_to_idx[int(row.v)]
        tt     = float(row.travel_time_s)
        lm     = float(row.length_m)
        oneway = bool(row.oneway)

        vi = n_osm + i  # virtual index for this POI

        # Remove original u→v edge and rebuild chain through virtual node
        adj[u_idx] = [(nb, t_, l_) for nb, t_, l_ in adj[u_idx] if nb != v_idx]
        adj[u_idx].append((vi, foot_t * tt, foot_t * lm))
        adj[vi].append((v_idx, (1 - foot_t) * tt, (1 - foot_t) * lm))

        if not oneway:
            adj[v_idx] = [(nb, t_, l_) for nb, t_, l_ in adj[v_idx] if nb != u_idx]
            adj[v_idx].append((vi, (1 - foot_t) * tt, (1 - foot_t) * lm))
            adj[vi].append((u_idx, foot_t * tt, foot_t * lm))

        patch_virt_idxs.append(vi)

    # ── also snap ALL existing POIs so Dijkstra can reach them as targets ──────
    print("Snapping all existing POIs as targets …")
    existing_indices = [i for i in range(len(poi_df)) if i not in patch_indices]
    existing_virt_idxs = []

    joined_existing = joined_all.loc[existing_indices]
    for i in existing_indices:
        row = joined_all.loc[i]
        if float(row.snap_dist_m) > MAX_SNAP_M:
            continue
        edge_idx  = int(row.index_right)
        edge_geom = edges_proj.geometry.iloc[edge_idx]
        pt        = poi_proj.geometry.iloc[i]
        foot_t    = edge_geom.project(pt, normalized=True)
        u_idx     = osmid_to_idx[int(row.u)]
        v_idx     = osmid_to_idx[int(row.v)]
        tt        = float(row.travel_time_s)
        lm        = float(row.length_m)
        oneway    = bool(row.oneway)
        vi        = n_osm + i

        if not any(nb == vi for nb, _, _ in adj[u_idx]):
            adj[u_idx] = [(nb, t_, l_) for nb, t_, l_ in adj[u_idx] if nb != v_idx]
            adj[u_idx].append((vi, foot_t * tt, foot_t * lm))
            adj[vi].append((v_idx, (1 - foot_t) * tt, (1 - foot_t) * lm))
            if not oneway:
                adj[v_idx] = [(nb, t_, l_) for nb, t_, l_ in adj[v_idx] if nb != u_idx]
                adj[v_idx].append((vi, (1 - foot_t) * tt, (1 - foot_t) * lm))
                adj[vi].append((u_idx, foot_t * tt, foot_t * lm))

        existing_virt_idxs.append(vi)

    all_target_virt_idxs = existing_virt_idxs + patch_virt_idxs
    print(f"  {len(all_target_virt_idxs)} total targets")

    # ── Dijkstra from patch nodes only ────────────────────────────────────────
    print(f"Running Dijkstra for {len(patch_virt_idxs)} patch node(s) …")
    with mp.Pool(N_WORKERS, initializer=_worker_init,
                 initargs=(adj, n_osm, all_target_virt_idxs)) as pool:
        results_from_patch = pool.map(dijkstra_dual, patch_virt_idxs)

    # ── Dijkstra from existing nodes TO patch nodes (reverse direction) ────────
    # For directed graph we need paths existing→patch too.
    # Run Dijkstra from all existing nodes but only keep patch node targets.
    print(f"Running Dijkstra from {len(existing_virt_idxs)} existing nodes to patch targets …")
    with mp.Pool(N_WORKERS, initializer=_worker_init,
                 initargs=(adj, n_osm, patch_virt_idxs)) as pool:
        results_to_patch = pool.map(dijkstra_dual, existing_virt_idxs)

    # ── write new edges ───────────────────────────────────────────────────────
    INF  = float("inf")
    COLS = ["lon_a","lat_a","lon_b","lat_b",
            "a_is_city","a_is_feasible","a_is_charger",
            "b_is_city","b_is_feasible","b_is_charger",
            "distance_km","estimated_time_min"]

    rows = []
    poi_arr = poi_df.reset_index(drop=True)

    vj_base = n_osm

    # patch → all
    for src_vi, target_t, target_l in results_from_patch:
        i = src_vi - vj_base
        a = poi_arr.iloc[i]
        for k, j in enumerate(existing_virt_idxs + patch_virt_idxs):
            tgt_i = j - vj_base
            if tgt_i == i:
                continue
            d = target_l[k]
            if d == INF or d >= DIST_LIMIT_M:
                continue
            b = poi_arr.iloc[tgt_i]
            rows.append((a.lon, a.lat, b.lon, b.lat,
                         int(a.is_city), int(a.is_feasible_location), int(a.is_existing_charger),
                         int(b.is_city), int(b.is_feasible_location), int(b.is_existing_charger),
                         round(d/1000, 4), round(target_t[k]/60, 4)))

    # existing → patch
    for src_vi, target_t, target_l in results_to_patch:
        i = src_vi - vj_base
        a = poi_arr.iloc[i]
        for k, j in enumerate(patch_virt_idxs):
            tgt_i = j - vj_base
            d = target_l[k]
            if d == INF or d >= DIST_LIMIT_M:
                continue
            b = poi_arr.iloc[tgt_i]
            rows.append((a.lon, a.lat, b.lon, b.lat,
                         int(a.is_city), int(a.is_feasible_location), int(a.is_existing_charger),
                         int(b.is_city), int(b.is_feasible_location), int(b.is_existing_charger),
                         round(d/1000, 4), round(target_t[k]/60, 4)))

    new_df = pd.DataFrame(rows, columns=COLS)
    print(f"  {len(new_df):,} new edges generated")

    # Append to edges.csv
    new_df.to_csv(OUT_EDGES, mode="a", header=False, index=False)
    print(f"  Appended to {OUT_EDGES}")

    # Append to edges_250.csv (filter to <250 km)
    new_df250 = new_df[new_df["distance_km"] < 250]
    new_df250.to_csv(OUT_EDGES_250, mode="a", header=False, index=False)
    print(f"  Appended {len(new_df250):,} edges to {OUT_EDGES_250}")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
