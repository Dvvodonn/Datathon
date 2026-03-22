"""
extract_checkpoint.py — rebuild candidate list and save checkpoint of already-found improvements
Usage: python models/extract_checkpoint.py
Outputs: models/local_search_checkpoint.json
"""
import heapq
import json
from collections import defaultdict

import pandas as pd

ALPHA        = 100
N_CITIES     = 235
NODES_PATH   = "data_main/nodes.csv"
EDGES_PATH   = "data_main/edges_150.csv"
RESULTS_PATH = "models/results_nodes.csv"
CHECKPOINT_PATH = "models/local_search_checkpoint.json"

# Ranks (1-based) that were confirmed improvements in the current run
IMPROVED_RANKS = {2, 4, 7, 8, 9, 14, 16, 17, 19, 21, 25, 30, 31, 35, 36,
                  51, 54, 57, 59, 72, 83, 86, 100}
LAST_TESTED_RANK = 100  # resume from rank 101


def nkey(lon, lat):
    return (round(float(lon), 6), round(float(lat), 6))


def dijkstra(src, open_nodes, adj):
    INF  = float("inf")
    dist = {src: 0.0}
    prev = {src: None}
    pq   = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF):
            continue
        for v, w in adj.get(u, []):
            if v not in open_nodes:
                continue
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def reconstruct_path(prev, src, dst):
    if dst not in prev:
        return []
    path, node = [], dst
    while node is not None:
        path.append(node)
        node = prev.get(node)
    return list(reversed(path))


print("Loading data …")
nodes_df   = pd.read_csv(NODES_PATH)
edges_df   = pd.read_csv(EDGES_PATH)
results_df = pd.read_csv(RESULTS_PATH)

nodes_df["key"]   = [nkey(r.lon, r.lat) for r in nodes_df.itertuples()]
results_df["key"] = [nkey(r.lon, r.lat) for r in results_df.itertuples()]

city_keys     = set(nodes_df.loc[nodes_df.is_city == 1, "key"])
feasible_keys = set(nodes_df.loc[nodes_df.is_feasible_location == 1, "key"])
charger_keys  = set(nodes_df.loc[nodes_df.is_existing_charger == 1, "key"])
all_keys      = city_keys | feasible_keys | charger_keys
built_keys    = set(results_df.loc[results_df.x_built == 1, "key"])

print(f"  {len(built_keys)} chargers in original solution")

adj = {}
for row in edges_df.itertuples(index=False):
    fk = nkey(row.lon_a, row.lat_a)
    tk = nkey(row.lon_b, row.lat_b)
    if fk in all_keys and tk in all_keys:
        adj.setdefault(fk, []).append((tk, float(row.estimated_time_min)))

city_list  = sorted(city_keys)[:N_CITIES]
city_set   = set(city_list)
full_nodes = city_set | charger_keys | feasible_keys
open_nodes = city_set | charger_keys | built_keys

print("Running baseline Dijkstra to identify candidates …")
results = []
for src in city_list:
    dist_opt,  prev_opt  = dijkstra(src, open_nodes, adj)
    dist_full, prev_full = dijkstra(src, full_nodes,  adj)
    for dst in city_list:
        if dst == src:
            continue
        t_opt  = dist_opt.get(dst,  float("inf"))
        t_full = dist_full.get(dst, float("inf"))
        path_full = reconstruct_path(prev_full, src, dst) if t_full < float("inf") else []
        results.append((src, dst, t_opt, t_full, path_full))

remaining = feasible_keys - built_keys
candidate_counts = defaultdict(float)
for src, dst, t_opt, t_full, path_full in results:
    if t_full >= t_opt - 0.01:
        continue
    for node in path_full:
        if node in remaining:
            candidate_counts[node] += (t_opt - t_full)

candidates = [(savings, k) for k, savings in candidate_counts.items() if savings > ALPHA]
candidates.sort(reverse=True)
print(f"  {len(candidates)} candidates total")

# Extract coordinates of improved locations
improved_nodes = []
for rank, (savings, k) in enumerate(candidates, 1):
    if rank in IMPROVED_RANKS:
        improved_nodes.append({"lon": k[0], "lat": k[1], "rank": rank, "savings": savings})

print(f"\nFound {len(improved_nodes)} improved nodes (expected {len(IMPROVED_RANKS)})")
for n in improved_nodes:
    print(f"  rank={n['rank']}  lon={n['lon']}  lat={n['lat']}")

checkpoint = {
    "last_tested_rank": LAST_TESTED_RANK,
    "total_candidates": len(candidates),
    "improved_nodes": improved_nodes,
}

with open(CHECKPOINT_PATH, "w") as f:
    json.dump(checkpoint, f, indent=2)

print(f"\nCheckpoint saved → {CHECKPOINT_PATH}")
print(f"New run will start at rank {LAST_TESTED_RANK + 1}/{len(candidates)}")
