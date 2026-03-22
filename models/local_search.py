"""
local_search.py — 1-opt local search to close the Benders gap  (optimized)
===========================================================================

Optimizations vs original:
  1. Full-graph Dijkstras computed ONCE at startup, reused per candidate
     (original recomputed them for every candidate → 2x wasted work)
  2. Pool created once per improvement round, not once per candidate
     (original created/destroyed a pool for each of 330 candidates)
  3. Batch evaluation: all remaining candidates evaluated in one pool.map
     per round; the BEST improvement is accepted each round (vs first found)
  4. Checkpoint: improvements written to JSON as found; resumes on restart

Algorithm
---------
  Round loop:
    1. pool.map(_eval_candidate, remaining_candidates)
         each task: restricted Dijkstra for all 235 city pairs with candidate added
    2. accept the candidate with the highest objective improvement
    3. checkpoint; update open set; reinit pool; repeat
"""

import heapq
import json
import multiprocessing as mp
from collections import defaultdict

import pandas as pd

# ── parameters ────────────────────────────────────────────────────────────────
ALPHA           = 100
N_CITIES        = 235
NODES_PATH      = "data_main/nodes.csv"
EDGES_PATH      = "data_main/edges_150.csv"
RESULTS_PATH    = "models/results_nodes.csv"
CHECKPOINT_PATH = "models/local_search_checkpoint.json"

# ── worker globals ─────────────────────────────────────────────────────────────
_open_base_g = None   # city_set | charger_keys | current_built (no candidate)
_city_list_g = None
_adj_g       = None


def nkey(lon, lat):
    return (round(float(lon), 6), round(float(lat), 6))


# ── Dijkstra ──────────────────────────────────────────────────────────────────

def dijkstra(src, open_nodes, adj):
    """Single-source shortest path. Returns dist dict (no prev tracking)."""
    INF  = float("inf")
    dist = {src: 0.0}
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
                heapq.heappush(pq, (nd, v))
    return dist


def dijkstra_with_prev(src, open_nodes, adj):
    """Single-source shortest path with predecessor tracking."""
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


def reconstruct_path(prev, dst):
    if dst not in prev:
        return []
    path, node = [], dst
    while node is not None:
        path.append(node)
        node = prev.get(node)
    return list(reversed(path))


# ── multiprocessing ────────────────────────────────────────────────────────────

def _worker_init(open_base, city_list, adj):
    global _open_base_g, _city_list_g, _adj_g
    _open_base_g = open_base
    _city_list_g = city_list
    _adj_g       = adj


def _eval_candidate(candidate_key):
    """Return (total_restricted_travel_time, candidate_key).
    Runs restricted Dijkstra for all source cities with candidate added.
    Full-graph Dijkstras are NOT needed here — precomputed upstream.
    """
    open_nodes = _open_base_g | {candidate_key}
    total = 0.0
    for src in _city_list_g:
        dist = dijkstra(src, open_nodes, _adj_g)
        for dst in _city_list_g:
            if dst != src:
                total += dist.get(dst, float("inf"))
    return total, candidate_key


def make_pool(n_workers, open_base, city_list, adj):
    return mp.Pool(n_workers, initializer=_worker_init,
                   initargs=(open_base, city_list, adj))


# ── candidate identification ───────────────────────────────────────────────────

def identify_candidates(city_list, open_nodes, full_nodes, adj, remaining, full_dists):
    """Return sorted list of (estimated_savings, key) for feasible candidates."""
    candidate_counts = defaultdict(float)
    for src in city_list:
        dist_opt = dijkstra(src, open_nodes, adj)
        _, prev_full = dijkstra_with_prev(src, full_nodes, adj)
        for dst in city_list:
            if dst == src:
                continue
            t_opt  = dist_opt.get(dst,  float("inf"))
            t_full = full_dists[src].get(dst, float("inf"))
            if t_full >= t_opt - 0.01:
                continue
            for node in reconstruct_path(prev_full, dst):
                if node in remaining:
                    candidate_counts[node] += (t_opt - t_full)
    candidates = [(s, k) for k, s in candidate_counts.items() if s > ALPHA]
    candidates.sort(reverse=True)
    return candidates


def total_travel_time(city_list, open_nodes, adj):
    total = 0.0
    for src in city_list:
        dist = dijkstra(src, open_nodes, adj)
        for dst in city_list:
            if dst != src:
                total += dist.get(dst, float("inf"))
    return total


# ── checkpoint ─────────────────────────────────────────────────────────────────

def load_checkpoint():
    try:
        with open(CHECKPOINT_PATH) as f:
            cp = json.load(f)
        nodes = cp.get("improved_nodes", [])
        last  = cp.get("last_tested_rank", 0)
        total = cp.get("total_candidates", 0)
        print(f"Checkpoint: {len(nodes)} prior improvements, "
              f"last rank {last}/{total}")
        return cp
    except FileNotFoundError:
        return None


def save_checkpoint(improved_keys, last_rank, total_candidates):
    cp = {
        "last_tested_rank": last_rank,
        "total_candidates": total_candidates,
        "improved_nodes": [{"lon": k[0], "lat": k[1]} for k in improved_keys],
    }
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(cp, f, indent=2)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Local search (optimized)  |  {n_workers} workers\n")

    # ── load data ──────────────────────────────────────────────────────────────
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
    orig_built    = set(results_df.loc[results_df.x_built == 1, "key"])

    adj = {}
    for row in edges_df.itertuples(index=False):
        fk = nkey(row.lon_a, row.lat_a)
        tk = nkey(row.lon_b, row.lat_b)
        if fk in all_keys and tk in all_keys:
            adj.setdefault(fk, []).append((tk, float(row.estimated_time_min)))

    city_list  = sorted(city_keys)[:N_CITIES]
    city_set   = set(city_list)
    full_nodes = city_set | charger_keys | feasible_keys

    # ── load checkpoint ────────────────────────────────────────────────────────
    cp = load_checkpoint()
    if cp:
        prior_keys = {nkey(n["lon"], n["lat"]) for n in cp["improved_nodes"]}
        # resume_after only valid if no new chargers were added (same solution base).
        # If the checkpoint added chargers, the candidate list is regenerated from
        # a different solution — all new candidates must be tested from scratch.
        resume_after = cp["last_tested_rank"] if not prior_keys else 0
    else:
        prior_keys   = set()
        resume_after = 0

    current_built = orig_built | prior_keys
    print(f"  Original built: {len(orig_built)}  +  checkpoint: {len(prior_keys)}  "
          f"=  {len(current_built)} chargers")

    # ── full-graph Dijkstra — computed ONCE ───────────────────────────────────
    print("\nComputing full-graph Dijkstra (once for all candidates) …")
    full_dists = {}
    for src in city_list:
        full_dists[src] = dijkstra(src, full_nodes, adj)
    print("  Done.")

    # ── baseline ───────────────────────────────────────────────────────────────
    open_nodes = city_set | charger_keys | current_built
    print("\nEvaluating baseline …")
    base_t   = total_travel_time(city_list, open_nodes, adj)
    base_obj = base_t + ALPHA * len(current_built)
    print(f"  Travel time : {base_t:,.1f} min")
    print(f"  Chargers    : {len(current_built)}  (cost {ALPHA * len(current_built):,})")
    print(f"  Objective   : {base_obj:,.1f}")

    # ── identify candidates ────────────────────────────────────────────────────
    print("\nIdentifying candidates …")
    remaining  = feasible_keys - current_built
    candidates = identify_candidates(city_list, open_nodes, full_nodes,
                                     adj, remaining, full_dists)
    total_candidates = len(candidates)
    print(f"  {total_candidates} candidates with estimated savings > {ALPHA} min")

    # Skip already-tested ranks
    if resume_after > 0 and resume_after < total_candidates:
        print(f"  Skipping first {resume_after} (already tested) → "
              f"resuming at rank {resume_after + 1}")
        candidates = candidates[resume_after:]
    elif resume_after >= total_candidates:
        print("  All candidates already tested.")
        candidates = []

    if not candidates:
        print("\nNo candidates to test.")
        _finish(results_df, orig_built, list(prior_keys), base_obj, base_obj)
        return

    # ── batch evaluation rounds ────────────────────────────────────────────────
    best_obj     = base_obj
    best_built   = set(current_built)
    all_improved = list(prior_keys)
    rank_base    = resume_after  # offset for display

    print(f"\nBatch-evaluating {len(candidates)} candidates …\n")

    round_num = 0
    while candidates:
        round_num += 1
        open_base = city_set | charger_keys | best_built

        # One pool.map evaluates ALL remaining candidates simultaneously
        pool = make_pool(n_workers, open_base, city_list, adj)
        try:
            keys        = [k for _, k in candidates]
            map_results = pool.map(_eval_candidate, keys, chunksize=4)
        finally:
            pool.close()
            pool.join()

        # Find best improvement this round
        best_delta = 0.0
        best_k     = None
        best_new_t = None
        for (new_t, k) in map_results:
            new_obj = new_t + ALPHA * (len(best_built) + 1)
            delta   = best_obj - new_obj
            if delta > best_delta:
                best_delta = delta
                best_k     = k
                best_new_t = new_t

        if best_k is None:
            print(f"Round {round_num}: no improvement in {len(candidates)} candidates "
                  f"— locally optimal.")
            break

        # Accept
        best_obj   = best_new_t + ALPHA * (len(best_built) + 1)
        best_built = best_built | {best_k}
        all_improved.append(best_k)

        accepted_rank = rank_base + next(
            i + 1 for i, (_, k) in enumerate(candidates) if k == best_k
        )
        print(f"Round {round_num}  |  rank {accepted_rank}/{total_candidates}  "
              f"lon={best_k[0]:.6f}  lat={best_k[1]:.6f}  "
              f"Δ={best_delta:+,.1f} min  obj={best_obj:,.1f}")

        # Checkpoint
        save_checkpoint(all_improved, accepted_rank, total_candidates)

        # Remove accepted candidate; next round evaluates against updated solution
        candidates = [(s, k) for s, k in candidates if k != best_k]
        rank_base  = 0  # ranks reset each round after first acceptance

    # ── save results ───────────────────────────────────────────────────────────
    _finish(results_df, orig_built, all_improved, base_obj, best_obj)


def _finish(results_df, orig_built, all_improved, base_obj, best_obj):
    new_keys = {k for k in all_improved}
    print(f"\n{'='*60}")
    print(f"Baseline objective  : {base_obj:,.1f}")
    print(f"Improved objective  : {best_obj:,.1f}")
    print(f"Total improvement   : {base_obj - best_obj:+,.1f} min")
    print(f"Locations added     : {len(new_keys)}")

    if new_keys:
        results_df["x_built"] = results_df.apply(
            lambda r: 1 if (nkey(r.lon, r.lat) in new_keys and r.x_built == 0)
                      else r.x_built, axis=1
        )
        results_df.drop(columns="key").to_csv(RESULTS_PATH, index=False)
        print(f"Updated → {RESULTS_PATH}")
    else:
        print("No improvement — solution unchanged.")


if __name__ == "__main__":
    main()
