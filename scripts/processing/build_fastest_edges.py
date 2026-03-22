"""
build_fastest_edges.py
======================
Fastest-path (minimize travel_time_s) Dijkstra with edge snapping.
Output: data_main/edges.csv

Schema: lon_a, lat_a, lon_b, lat_b,
        a_is_city, a_is_feasible, a_is_charger,
        b_is_city, b_is_feasible, b_is_charger,
        distance_km, estimated_time_min

Filter: distance_km < 500
"""

import sys
import time
import heapq
import multiprocessing as mp
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ── paths ────────────────────────────────────────────────────────────────────
EDGES_PATH  = "data/raw/road_network/spain_interurban_edges.gpkg"
NODES_PATH  = "data/raw/road_network/spain_interurban_nodes.gpkg"
POI_PATH    = "data_main/nodes.csv"
OUT_PATH    = "data_main/edges.csv"

# ── constants ────────────────────────────────────────────────────────────────
DIST_LIMIT_M  = 500_000.0          # 500 km in metres
MAX_SNAP_M    = 10_000.0           # 10 km — covers rural nodes like El Casar (6.1 km from nearest road)
EPSG_PROJ     = 25830              # UTM zone 30N — Spain
N_WORKERS     = max(1, mp.cpu_count() - 1)

# ── module-level globals (shared across worker processes) ─────────────────────
ADJ        = None   # list of lists: adj[node_idx] = [(neigh_idx, travel_time_s, length_m), ...]
N_OSM      = None   # number of OSM base nodes
VIRT_IDXS  = None   # list of virtual node indices (targets we care about)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — load edges + build adjacency list
# ─────────────────────────────────────────────────────────────────────────────

def build_base_adj(edges_gdf):
    """Return (adj, osmid_to_idx, n_osm_nodes)."""
    t0 = time.time()
    print(f"  building OSM node index …", flush=True)

    # Collect unique OSM node ids
    all_ids = pd.concat([edges_gdf["u"], edges_gdf["v"]]).unique()
    osmid_to_idx = {int(oid): i for i, oid in enumerate(all_ids)}
    n = len(osmid_to_idx)
    print(f"  {n} unique OSM nodes", flush=True)

    adj = [[] for _ in range(n)]

    for row in edges_gdf.itertuples(index=False):
        u_idx = osmid_to_idx[int(row.u)]
        v_idx = osmid_to_idx[int(row.v)]
        tt  = float(row.travel_time_s)
        lm  = float(row.length_m)
        adj[u_idx].append((v_idx, tt, lm))
        # bidirectional: add reverse edge if not oneway
        if not row.oneway:
            adj[v_idx].append((u_idx, tt, lm))

    print(f"  base adj built in {time.time()-t0:.1f}s", flush=True)
    return adj, osmid_to_idx, n


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — edge-snap POI nodes
# ─────────────────────────────────────────────────────────────────────────────

def edge_snap(poi_df, edges_gdf, osmid_to_idx):
    """
    Return snap_info: list of dicts per POI with keys:
        virt_idx, edge_row_idx, u_idx, v_idx, foot_t,
        time_to_virt, time_from_virt,
        len_to_virt, len_from_virt

    POIs are snapped only onto edges whose both endpoints lie within the
    Largest Strongly Connected Component (LSCC) of the road graph.  This
    prevents virtual nodes from landing on one-way dead-ends or isolated
    road segments that have no path back to the main network.
    """
    import networkx as nx

    t0 = time.time()

    # ── find LSCC so we only snap onto well-connected edges ──────────────────
    print(f"  building directed graph for LSCC …", flush=True)
    G = nx.DiGraph()
    G.add_edges_from(zip(edges_gdf["u"].astype(int), edges_gdf["v"].astype(int)))
    lscc = max(nx.strongly_connected_components(G), key=len)
    del G
    print(f"  LSCC: {len(lscc):,} / {edges_gdf['u'].nunique() + edges_gdf['v'].nunique():,} nodes", flush=True)

    edges_lscc = edges_gdf[
        edges_gdf["u"].isin(lscc) & edges_gdf["v"].isin(lscc)
    ].reset_index(drop=True)
    print(f"  {len(edges_lscc):,} / {len(edges_gdf):,} edges retained after LSCC filter", flush=True)

    # ── project ───────────────────────────────────────────────────────────────
    print(f"  projecting to EPSG:{EPSG_PROJ} …", flush=True)

    poi_geom = gpd.GeoDataFrame(
        poi_df.reset_index(drop=True),
        geometry=[Point(row.lon, row.lat) for row in poi_df.itertuples()],
        crs="EPSG:4326",
    ).to_crs(EPSG_PROJ)

    edges_proj = edges_lscc.to_crs(EPSG_PROJ).reset_index(drop=True)

    print(f"  sjoin_nearest (LSCC edges only) …", flush=True)
    joined = gpd.sjoin_nearest(
        poi_geom[["geometry"]],
        edges_proj[["geometry", "u", "v", "length_m", "travel_time_s", "oneway"]],
        how="left",
        distance_col="snap_dist_m",
    )
    # sjoin_nearest can duplicate rows if ties — keep first match per POI
    joined = joined[~joined.index.duplicated(keep="first")]

    print(f"  computing perpendicular feet …", flush=True)
    snap_info = []
    n_osm = len(osmid_to_idx)

    n_excluded = 0
    for i, row in enumerate(joined.itertuples()):
        snap_dist = float(row.snap_dist_m)
        if snap_dist > MAX_SNAP_M:
            # Node is too far from any road (e.g. Canary Islands) — exclude entirely
            snap_info.append({"virt_idx": n_osm + i, "excluded": True})
            n_excluded += 1
            continue

        edge_idx  = row.index_right
        edge_geom = edges_proj.geometry.iloc[edge_idx]
        pt        = poi_geom.geometry.iloc[i]

        foot_t = edge_geom.project(pt, normalized=True)   # 0..1

        u_idx = osmid_to_idx[int(row.u)]
        v_idx = osmid_to_idx[int(row.v)]
        tt    = float(row.travel_time_s)
        lm    = float(row.length_m)

        snap_info.append({
            "virt_idx"       : n_osm + i,
            "excluded"       : False,
            "edge_row_idx"   : edge_idx,
            "u_idx"          : u_idx,
            "v_idx"          : v_idx,
            "foot_t"         : foot_t,
            "time_to_virt"   : foot_t * tt,
            "time_from_virt" : (1 - foot_t) * tt,
            "len_to_virt"    : foot_t * lm,
            "len_from_virt"  : (1 - foot_t) * lm,
            "oneway"         : bool(row.oneway),
        })

    print(f"  {n_excluded} nodes excluded (snap > {MAX_SNAP_M/1000:.0f} km)", flush=True)

    print(f"  edge-snap done in {time.time()-t0:.1f}s", flush=True)
    return snap_info


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — insert virtual nodes into adjacency list
# ─────────────────────────────────────────────────────────────────────────────

def insert_virtual_nodes(adj, snap_info, n_osm):
    """
    Mutate adj in-place.
    For each original edge (u→v) that has k virtual nodes sorted by foot_t:
        Build chain u → virt1 → virt2 → … → virtk → v
        (and v → virtk → … → virt1 → u if bidirectional)
    Remove original u→v edge entries for modified edges.
    """
    t0 = time.time()
    total_virt = len(snap_info)
    print(f"  inserting {total_virt} virtual nodes …", flush=True)

    # Extend adj list
    adj.extend([] for _ in range(total_virt))

    # Group virtual nodes by (u_idx, v_idx, edge_row_idx) — skip excluded nodes
    from collections import defaultdict
    edge_groups = defaultdict(list)
    for info in snap_info:
        if info.get("excluded"):
            continue
        key = (info["u_idx"], info["v_idx"], info["edge_row_idx"])
        edge_groups[key].append(info)

    # For each edge group, sort by foot_t and build chain
    for (u_idx, v_idx, _), infos in edge_groups.items():
        infos_sorted = sorted(infos, key=lambda x: x["foot_t"])

        # Remove original u→v edge
        adj[u_idx] = [(nb, tt, lm) for nb, tt, lm in adj[u_idx] if nb != v_idx]

        # Build forward chain: u → virt0 → … → virtk → v
        prev_idx  = u_idx
        prev_time = 0.0
        prev_len  = 0.0
        cumulative_time = 0.0
        cumulative_len  = 0.0
        for info in infos_sorted:
            vi = info["virt_idx"]
            # edge from prev to this virtual node
            seg_t = info["time_to_virt"] - cumulative_time
            seg_l = info["len_to_virt"]  - cumulative_len
            seg_t = max(seg_t, 0.0)
            seg_l = max(seg_l, 0.0)
            adj[prev_idx].append((vi, seg_t, seg_l))
            cumulative_time = info["time_to_virt"]
            cumulative_len  = info["len_to_virt"]
            prev_idx = vi

        # last virtual → v
        last_info = infos_sorted[-1]
        seg_t = last_info["time_from_virt"]
        seg_l = last_info["len_from_virt"]
        adj[prev_idx].append((v_idx, max(seg_t, 0.0), max(seg_l, 0.0)))

        # If bidirectional, also build reverse chain v → virtk → … → virt0 → u
        if not infos_sorted[0]["oneway"]:
            # Remove original v→u edge
            adj[v_idx] = [(nb, tt, lm) for nb, tt, lm in adj[v_idx] if nb != u_idx]

            # Reverse chain: v → virtkN → … → virt0 → u
            # Segments in reverse order
            prev_idx = v_idx
            prev_infos = list(reversed(infos_sorted))
            # v → virtkN segment
            seg_t = prev_infos[0]["time_from_virt"]
            seg_l = prev_infos[0]["len_from_virt"]
            adj[prev_idx].append((prev_infos[0]["virt_idx"], max(seg_t, 0.0), max(seg_l, 0.0)))
            prev_idx = prev_infos[0]["virt_idx"]

            for k in range(1, len(prev_infos)):
                curr = prev_infos[k]
                prev_ = prev_infos[k-1]
                seg_t = prev_["time_to_virt"] - curr["time_to_virt"]
                seg_l = prev_["len_to_virt"]  - curr["len_to_virt"]
                adj[prev_idx].append((curr["virt_idx"], max(seg_t, 0.0), max(seg_l, 0.0)))
                prev_idx = curr["virt_idx"]

            # last virtual → u: the portion from foot_t=0 to first virtual
            seg_t = infos_sorted[0]["time_to_virt"]
            seg_l = infos_sorted[0]["len_to_virt"]
            adj[prev_idx].append((u_idx, max(seg_t, 0.0), max(seg_l, 0.0)))

    print(f"  virtual node insertion done in {time.time()-t0:.1f}s", flush=True)
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Dijkstra worker
# ─────────────────────────────────────────────────────────────────────────────

def _worker_init(adj_shared, n_osm_shared, virt_idxs_shared):
    global ADJ, N_OSM, VIRT_IDXS
    ADJ       = adj_shared
    N_OSM     = n_osm_shared
    VIRT_IDXS = virt_idxs_shared


def dijkstra_dual(src_virt_idx):
    """
    Minimize travel_time; accumulate length along same path.
    Returns only values for the virtual node indices (targets), not all 203K nodes.
    """
    INF = float("inf")
    n   = len(ADJ)
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
    # Only return values for target virtual nodes — avoids 200K-element pipe transfer
    target_t = [best_t[vi] for vi in VIRT_IDXS]
    target_l = [best_l[vi] for vi in VIRT_IDXS]
    return src_virt_idx, target_t, target_l


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — flatten + write
# ─────────────────────────────────────────────────────────────────────────────

def flatten_and_write(results, snap_info, poi_df, out_path):
    import gzip

    t0 = time.time()

    # included_poi_indices: positions in poi_df that are NOT excluded
    # target_t[k] from Dijkstra corresponds to included_poi_indices[k]
    vj_base = snap_info[0]["virt_idx"]   # = n_osm
    included_poi_indices = [i for i, info in enumerate(snap_info) if not info.get("excluded")]
    n_incl = len(included_poi_indices)
    print(f"  flattening {n_incl}×{n_incl} included pairs …", flush=True)

    # Map src_virt_idx → (target_t, target_l) arrays (length = n_incl)
    result_map = {}
    for src_virt_idx, target_t, target_l in results:
        poi_i = src_virt_idx - vj_base
        result_map[poi_i] = (target_t, target_l)

    COLS = [
        "lon_a", "lat_a", "lon_b", "lat_b",
        "a_is_city", "a_is_feasible", "a_is_charger",
        "b_is_city", "b_is_feasible", "b_is_charger",
        "distance_km", "estimated_time_min",
    ]
    BATCH = 200_000
    INF   = float("inf")
    poi_arr = poi_df.reset_index(drop=True)
    total_written = 0
    rows = []
    header_written = False

    with open(out_path, "w") as gz:
        for i in included_poi_indices:
            if i not in result_map:
                continue
            target_t, target_l = result_map[i]
            a = poi_arr.iloc[i]

            # k-th element of target arrays corresponds to included_poi_indices[k]
            for k, j in enumerate(included_poi_indices):
                if i == j:
                    continue
                d = target_l[k]
                if d == INF or d >= DIST_LIMIT_M:
                    continue
                t = target_t[k]
                b = poi_arr.iloc[j]
                rows.append((
                    a.lon, a.lat, b.lon, b.lat,
                    int(a.is_city), int(a.is_feasible_location), int(a.is_existing_charger),
                    int(b.is_city), int(b.is_feasible_location), int(b.is_existing_charger),
                    round(d / 1000, 4),
                    round(t / 60,   4),
                ))

            if len(rows) >= BATCH:
                df = pd.DataFrame(rows, columns=COLS)
                df.to_csv(gz, index=False, header=not header_written)
                header_written = True
                total_written += len(rows)
                rows = []
                if total_written % 2_000_000 == 0:
                    print(f"    {total_written:,} rows written …", flush=True)

        if rows:
            df = pd.DataFrame(rows, columns=COLS)
            df.to_csv(gz, index=False, header=not header_written)
            total_written += len(rows)

    print(f"  wrote {total_written:,} rows to {out_path} in {time.time()-t0:.1f}s", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    overall = time.time()

    # ── Step 1: Load edges ──────────────────────────────────────────────────
    print("Step 1: loading edges …", flush=True)
    edges_gdf = gpd.read_file(EDGES_PATH)
    print(f"  {len(edges_gdf):,} edges loaded", flush=True)

    adj, osmid_to_idx, n_osm = build_base_adj(edges_gdf)
    print(f"  adj list: {n_osm} OSM nodes, {sum(len(x) for x in adj):,} directed edges", flush=True)

    # ── Step 2: Load POI nodes + edge-snap ──────────────────────────────────
    print("Step 2: edge-snapping POI nodes …", flush=True)
    poi_df = pd.read_csv(POI_PATH)
    snap_info = edge_snap(poi_df, edges_gdf, osmid_to_idx)

    # ── Step 3: Insert virtual nodes ─────────────────────────────────────────
    print("Step 3: inserting virtual nodes …", flush=True)
    adj = insert_virtual_nodes(adj, snap_info, n_osm)
    total_nodes = n_osm + len(snap_info)
    print(f"  total graph nodes: {total_nodes:,}", flush=True)

    # ── Step 4: Dijkstra ─────────────────────────────────────────────────────
    print(f"Step 4: Dijkstra ({len(snap_info)} sources, {N_WORKERS} workers) …", flush=True)
    src_virt_idxs = [info["virt_idx"] for info in snap_info if not info.get("excluded")]
    print(f"  {len(src_virt_idxs)} included sources ({len(snap_info)-len(src_virt_idxs)} excluded)", flush=True)

    t_dijk = time.time()
    with mp.Pool(N_WORKERS, initializer=_worker_init, initargs=(adj, n_osm, src_virt_idxs)) as pool:  # src_virt_idxs used as VIRT_IDXS targets
        results = pool.map(dijkstra_dual, src_virt_idxs, chunksize=4)
    print(f"  Dijkstra done in {time.time()-t_dijk:.1f}s", flush=True)

    # ── Step 5: Flatten + write ──────────────────────────────────────────────
    print("Step 5: flattening + writing …", flush=True)
    flatten_and_write(results, snap_info, poi_df, OUT_PATH)

    print(f"\nDone in {time.time()-overall:.1f}s total. Output: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
