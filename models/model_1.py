"""
model_1.py — EV Charging Network Optimisation (Benders Decomposition)
======================================================================

Globally optimal via Benders Decomposition:
  - Master problem  : choose x[k] (where to build chargers) — solved with OR-Tools CP-SAT
  - Subproblems     : for fixed x[k], find fastest path for each city pair — solved with Dijkstra
  - Iteration       : subproblems return cuts to master until lower bound == upper bound

Decision variables
------------------
  x[k]          =  1 if we build a charger at feasible location k
  η[u,v]        =  estimated travel time (minutes) for city pair (u,v)  [master only]

Objective
---------
  Minimize  Σ_{u≠v} η[u,v]  +  α · Σ_k x[k]

  where η[u,v] is tightened each iteration by Benders cuts from the Dijkstra subproblems.
  d_ij = estimated_time_min (travel time, not distance).

Benders cuts
------------
  Feasibility cut  : if no path exists for (u,v) given x,
                     force at least one node on the full-graph path to open.
  Optimality cut   : if path found with time t*, using feasible nodes S:
                     η[u,v] ≥ t* - M · Σ_{k∈S} (1 - x[k])
                     (if all nodes in S are open, η[u,v] must be at least t*)

Parameters
----------
  ALPHA        : penalty per charger (higher → fewer chargers, longer paths)
  N_CITIES     : number of city clusters to optimise over
  MAX_ITER     : Benders iteration cap
  N_WORKERS    : parallel Dijkstra workers
"""

import heapq
import json
import multiprocessing as mp
import time as timer
from collections import defaultdict

import pandas as pd
from ortools.linear_solver import pywraplp

# ── parameters ────────────────────────────────────────────────────────────────
ALPHA              = 100      # penalty per charger built (≈ minutes saved threshold to justify one charger)
PENALTY_INFEASIBLE = 5_000    # virtual travel time (min) assigned to disconnected city pairs
N_CITIES     = 235       # all city clusters
MAX_ITER     = 50        # max Benders iterations
CONV_TOL     = 1e-2      # convergence tolerance (% gap — stops when gap < 0.01%)
N_WORKERS    = max(1, mp.cpu_count() - 1)

NODES_PATH   = "data_main/nodes.csv"
EDGES_PATH   = "data_main/edges_250.csv"  # pre-filtered to distance < 250 km (realistic EV range)
CUTS_PATH    = "models/benders_cuts.json"

# ── single-source Dijkstra (returns dist + prev for ALL reachable nodes) ─────
def dijkstra_single_source(src, open_nodes, adj, city_keys=None):
    """
    Full single-source Dijkstra from src over open_nodes.
    Returns (dist dict, prev dict).

    city_keys: if provided, cities other than src are treated as TERMINALS.
               The EV can ARRIVE at them (as the destination) but cannot
               continue routing THROUGH them as intermediate charging stops.
               Only highway chargers (existing or newly placed) can be used
               for mid-route charging. This is the correct model for highway
               charger placement: cities are journey origins/destinations,
               not en-route charging stops.
    """
    INF  = float("inf")
    dist = {src: 0.0}
    prev = {src: None}
    pq   = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF):
            continue
        # Cities (other than the source) are terminals: record arrival but
        # do not expand outward — the EV has reached its destination.
        if city_keys and u != src and u in city_keys:
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


# ── worker (parallelised subproblem — one worker per source city) ─────────────
_open_nodes_global = None
_full_nodes_global = None
_city_list_global  = None
_city_keys_global  = None
_adj_global        = None

def _worker_init(open_nodes, full_nodes, city_list, city_keys, adj):
    global _open_nodes_global, _full_nodes_global, _city_list_global, _city_keys_global, _adj_global
    _open_nodes_global = open_nodes
    _full_nodes_global = full_nodes
    _city_list_global  = city_list
    _city_keys_global  = city_keys
    _adj_global        = adj

def _solve_source(src):
    """
    Run two single-source Dijkstras from src.
    open graph  : cities + existing chargers + newly opened chargers
    full graph  : all nodes (cities + existing + all feasible locations)
    Comparing t_opt vs t_full reveals where highway chargers beat city-hop paths.
    Returns list of (src, dst, t_opt, path_opt, t_full, path_full) for all dst cities.
    """
    dist_opt,  prev_opt  = dijkstra_single_source(src, _open_nodes_global, _adj_global)
    dist_full, prev_full = dijkstra_single_source(src, _full_nodes_global, _adj_global)
    results = []
    for dst in _city_list_global:
        if dst == src:
            continue
        t_opt  = dist_opt.get(dst,  float("inf"))
        t_full = dist_full.get(dst, float("inf"))
        path_opt  = reconstruct_path(prev_opt,  src, dst) if t_opt  < float("inf") else []
        path_full = reconstruct_path(prev_full, src, dst) if t_full < float("inf") else []
        results.append((src, dst, t_opt, path_opt, t_full, path_full))
    return results


def build_master(feas_list, city_pairs, cuts):
    """
    Rebuild the master MIP from scratch each iteration with all accumulated cuts.
    Uses OR-Tools pywraplp (SCIP backend).
    Returns (solver, x_vars, eta_vars).
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    solver.SuppressOutput()
    solver.SetTimeLimit(300_000)  # 5 min per master solve

    feas_idx = {k: i for i, k in enumerate(feas_list)}

    # x[k] ∈ {0,1}
    x = [solver.BoolVar(f"x_{i}") for i in range(len(feas_list))]

    # η[u,v] ≥ 0  only for reachable pairs (permanently unreachable pairs excluded)
    eta = {(u, v): solver.NumVar(0, solver.infinity(), f"eta_{u}_{v}")
           for (u, v) in city_pairs}

    # Objective
    obj = solver.Objective()
    for xi in x:
        obj.SetCoefficient(xi, ALPHA)
    for e in eta.values():
        obj.SetCoefficient(e, 1.0)
    obj.SetMinimization()

    # Re-add all accumulated cuts
    for cut in cuts:
        if cut["type"] == "feasibility":
            # Σ x[k] ≥ 1  for k in cut["nodes"]
            ct = solver.Constraint(1, solver.infinity())
            for k in cut["nodes"]:
                ct.SetCoefficient(x[feas_idx[k]], 1.0)

        elif cut["type"] == "optimality":
            # η[u,v] ≥ t* - t* * Σ(1-x[k])  for k in cut["nodes"]
            # → η[u,v] - t* * Σ x[k] ≥ t* - t* * |S|
            u, v, t_star, nodes = cut["u"], cut["v"], cut["t"], cut["nodes"]
            rhs = t_star - t_star * len(nodes)
            ct  = solver.Constraint(rhs, solver.infinity())
            ct.SetCoefficient(eta[(u, v)], 1.0)
            for k in nodes:
                ct.SetCoefficient(x[feas_idx[k]], -t_star)

        elif cut["type"] == "fixed_eta":
            # η[u,v] ≥ t*  (path uses no feasible nodes)
            u, v, t_star = cut["u"], cut["v"], cut["t"]
            ct = solver.Constraint(t_star, solver.infinity())
            ct.SetCoefficient(eta[(u, v)], 1.0)

        elif cut["type"] == "cond_lower":
            # η[u,v] + (t_opt − t_full) · Σ_{k∈S} x_k ≥ t_opt
            # Semantics: without opening S you pay t_opt; opening any k in S
            # relaxes this and (combined with fixed_eta at t_full) lets η drop
            # to t_full once all of S is open.  Proof of validity: for any x,
            # T(u,v,x) ≥ t_full and T(u,v,x) + (t_opt−t_full)·Σ x_k ≥ t_opt.
            u, v = cut["u"], cut["v"]
            t_opt_c, t_full_c, nodes = cut["t_opt"], cut["t_full"], cut["nodes"]
            delta = t_opt_c - t_full_c          # always > 0
            ct = solver.Constraint(t_opt_c, solver.infinity())
            ct.SetCoefficient(eta[(u, v)], 1.0)
            for k in nodes:
                ct.SetCoefficient(x[feas_idx[k]], delta)

    return solver, x, eta, feas_idx


def nkey(lon, lat):
    return (round(float(lon), 6), round(float(lat), 6))


def main():
    # ── load + filter data ────────────────────────────────────────────────────
    print("Loading data …")
    nodes_df = pd.read_csv(NODES_PATH)
    edges_df = pd.read_csv(EDGES_PATH)
    print(f"  {len(nodes_df):,} nodes  |  {len(edges_df):,} edges (distance < 250 km, edges_250)")

    # ── node keys and sets ────────────────────────────────────────────────────
    nodes_df["key"] = [nkey(r.lon, r.lat) for r in nodes_df.itertuples()]

    city_keys     = set(nodes_df.loc[nodes_df.is_city == 1,              "key"])
    feasible_keys = set(nodes_df.loc[nodes_df.is_feasible_location == 1, "key"])
    charger_keys  = set(nodes_df.loc[nodes_df.is_existing_charger == 1,  "key"])
    all_keys      = city_keys | feasible_keys | charger_keys

    # Select N_CITIES cities
    city_list  = sorted(city_keys)[:N_CITIES]
    city_set   = set(city_list)
    city_pairs = [(u, v) for u in city_list for v in city_list if u != v]
    print(f"  {len(city_list)} cities  |  {len(city_pairs)} directed city pairs")

    # ── build adjacency list ──────────────────────────────────────────────────
    # Edge weight = pure driving time. City-detour penalty is applied inside
    # Dijkstra when *departing* from an intermediate city (not the source).
    # This correctly models: highway chargers have zero detour overhead;
    # city stops require exiting the highway, charging, and re-entering.
    print("Building adjacency list …")
    adj = defaultdict(list)  # node_key → [(neighbour_key, time_min)]
    for row in edges_df.itertuples(index=False):
        fk = nkey(row.lon_a, row.lat_a)
        tk = nkey(row.lon_b, row.lat_b)
        if fk in all_keys and tk in all_keys:
            adj[fk].append((tk, float(row.estimated_time_min)))

    print(f"  {sum(len(v) for v in adj.values()):,} directed adjacency entries")

    # ── pre-filter city pairs to those reachable on the full graph ────────────
    # Pairs with t_full=inf are physically unreachable (islands, missing road
    # data) — no charger can fix them. Exclude them upfront so they don't
    # distort the objective or waste feasibility cuts.
    full_nodes  = city_set | charger_keys | feasible_keys
    print("Pre-filtering city pairs for full-graph reachability …")
    adj_dict = dict(adj)
    reachable_pairs = set()
    for src in city_list:
        dist_full, _ = dijkstra_single_source(src, full_nodes, adj_dict)
        for dst in city_list:
            if dst != src and dist_full.get(dst, float("inf")) < float("inf"):
                reachable_pairs.add((src, dst))
    city_pairs = [(u, v) for u, v in
                  [(u, v) for u in city_list for v in city_list if u != v]
                  if (u, v) in reachable_pairs]
    n_unreachable = len(city_list) * (len(city_list) - 1) - len(city_pairs)
    print(f"  {len(city_pairs):,} reachable pairs  |  {n_unreachable:,} permanently unreachable (islands/gaps) — excluded")

    feas_list   = sorted(feasible_keys)
    upper_bound = float("inf")
    lower_bound = 0.0
    best_x      = {k: 0 for k in feas_list}
    cuts        = []   # accumulated Benders cuts (list of dicts)
    seen_cuts   = set()  # deduplication keys for cuts

    print(f"Starting Benders decomposition "
          f"({MAX_ITER} max iterations, {N_WORKERS} workers) …\n")

    for iteration in range(1, MAX_ITER + 1):
        t_iter = timer.time()

        # ── rebuild and solve master ──────────────────────────────────────────
        solver, x, eta, feas_idx = build_master(feas_list, city_pairs, cuts)
        status = solver.Solve()

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            print(f"Master status {status} — stopping.")
            break

        master_proven = (status == pywraplp.Solver.OPTIMAL)
        obj_val = solver.Objective().Value()
        # Only update LB when SCIP proves MIP optimality — FEASIBLE means B&B
        # timed out, so the bound is unreliable and should not raise the LB.
        if master_proven:
            lower_bound = obj_val
        # Always proceed with the incumbent solution for subproblem/UB computation
        x_vals   = {feas_list[i]: x[i].solution_value() for i in range(len(feas_list))}
        eta_vals = {(u, v): eta[(u, v)].solution_value() for (u, v) in eta}
        open_nodes = city_set | charger_keys | {k for k in feasible_keys if x_vals[k] > 0.5}

        # ── solve subproblems in parallel (235 sources, not 54,990 pairs) ─────
        with mp.Pool(N_WORKERS, initializer=_worker_init,
                     initargs=(open_nodes, full_nodes, city_list, city_set, dict(adj))) as pool:
            source_results = pool.map(_solve_source, city_list, chunksize=1)

        # flatten list-of-lists into flat list of (u,v,...) tuples
        results = [row for src_rows in source_results for row in src_rows]

        # ── generate cuts + compute upper bound ───────────────────────────────
        # Nodes that are always open regardless of x (cities + existing chargers).
        # These must be excluded from cut support sets: a node already in open_nodes
        # cannot be a "swing" variable in a Benders cut — including it produces a
        # weak big-M cut (η[u,v] ≥ 0 when x[k]=0) instead of a tight fixed_eta cut.
        always_open = city_set | charger_keys

        cuts_added       = 0
        total_t_opt      = 0.0
        infeasible_pairs = 0

        reachable_set = set(city_pairs)
        for (u, v, t_opt, path_opt, t_full, path_full) in results:
            if (u, v) not in reachable_set:
                continue  # permanently unreachable (islands) — skip entirely
            if t_opt == float("inf"):
                infeasible_pairs += 1
                total_t_opt += PENALTY_INFEASIBLE  # penalise disconnected pairs
                # Bound η[u,v] from below at PENALTY_INFEASIBLE so the master
                # accounts for the cost of leaving this pair disconnected.
                pen_key = ("fixed", u, v, float(PENALTY_INFEASIBLE))
                if pen_key not in seen_cuts:
                    seen_cuts.add(pen_key)
                    cuts.append({"type": "fixed_eta", "u": u, "v": v, "t": PENALTY_INFEASIBLE})
                    cuts_added += 1
                feas_on_full = [k for k in path_full
                                if k in feasible_keys and k not in always_open]
                if feas_on_full:
                    cut_key = ("feas", frozenset(feas_on_full))
                    if cut_key not in seen_cuts:
                        seen_cuts.add(cut_key)
                        cuts.append({"type": "feasibility", "nodes": feas_on_full})
                        cuts_added += 1
            else:
                # UB always uses t_opt — what's achievable with CURRENT open chargers.
                total_t_opt += t_opt

                highway_better = (t_full < float("inf") and t_full < t_opt - 0.01)
                t_lb = t_full if highway_better else t_opt

                # ── Cut 1: fixed_eta at t_full (global lower bound) ────────────
                # t_full is the best achievable time with any open set.
                # η[u,v] ≥ t_full is ALWAYS valid as an unconditional lower bound.
                fixed_key = ("fixed", u, v, round(t_lb, 4))
                if fixed_key not in seen_cuts:
                    seen_cuts.add(fixed_key)
                    cuts.append({"type": "fixed_eta", "u": u, "v": v, "t": t_lb})
                    cuts_added += 1

                if highway_better:
                    feas_on_full = [k for k in path_full
                                    if k in feasible_keys and k not in always_open]

                    if feas_on_full:
                        # ── Cut 2: optimality cut from path_full ──────────────
                        # "Open S → η can be as low as t_full."
                        # Generated when master hasn't yet accounted for t_full.
                        if eta_vals[(u, v)] < t_full - 0.01:
                            cut_key = ("opt", u, v, round(t_full, 4), frozenset(feas_on_full))
                            if cut_key not in seen_cuts:
                                seen_cuts.add(cut_key)
                                cuts.append({"type": "optimality",
                                             "u": u, "v": v, "t": t_full,
                                             "nodes": feas_on_full})
                                cuts_added += 1

                        # ── Cut 3: conditioned lower bound ────────────────────
                        # η[u,v] + (t_opt − t_full)·Σ x_k ≥ t_opt
                        # "Without opening S you cannot claim η < t_opt."
                        # Generated when master claims η < t_opt (i.e. believes
                        # highway path is free, but chargers haven't been opened).
                        # Proof: T(u,v,x) + (t_opt−t_full)·Σ x_k ≥ t_opt ∀x
                        if eta_vals[(u, v)] < t_opt - 0.01:
                            cond_key = ("cond", u, v,
                                        round(t_opt, 4), round(t_full, 4),
                                        frozenset(feas_on_full))
                            if cond_key not in seen_cuts:
                                seen_cuts.add(cond_key)
                                cuts.append({"type": "cond_lower",
                                             "u": u, "v": v,
                                             "t_opt": t_opt, "t_full": t_full,
                                             "nodes": feas_on_full})
                                cuts_added += 1

        n_chargers = sum(1 for k in feasible_keys if x_vals[k] > 0.5)
        current_ub = total_t_opt + ALPHA * n_chargers
        if current_ub < upper_bound:
            upper_bound = current_ub
            best_x      = dict(x_vals)

        gap = (upper_bound - lower_bound) / max(upper_bound, 1e-9) * 100
        proven_str = "OPT" if master_proven else "feas"
        print(f"  Iter {iteration:3d} | LB {lower_bound:10.1f} | UB {upper_bound:10.1f} | "
              f"Gap {gap:6.2f}% | Master {proven_str} | New cuts {cuts_added:4d} | "
              f"Total cuts {len(cuts):5d} | Infeasible {infeasible_pairs} | {timer.time()-t_iter:.1f}s")

        if gap < CONV_TOL:
            print(f"\nConverged at iteration {iteration}.")
            break
        if cuts_added == 0:
            print(f"\nNo new cuts — search exhausted at iteration {iteration}.")
            break

    # ── save cuts for efficient frontier / warm-start ─────────────────────────
    def _key_to_list(k):
        return list(k) if isinstance(k, tuple) else k

    def _serialise_cut(cut):
        c = dict(cut)
        for field in ("u", "v"):
            if field in c:
                c[field] = _key_to_list(c[field])
        if "nodes" in c:
            c["nodes"] = [_key_to_list(k) for k in c["nodes"]]
        return c

    cuts_payload = {
        "alpha":        ALPHA,
        "edges_path":   EDGES_PATH,
        "feas_list":    [list(k) for k in feas_list],
        "city_pairs":   [[list(u), list(v)] for u, v in city_pairs],
        "lower_bound":  lower_bound,
        "upper_bound":  upper_bound,
        "cuts":         [_serialise_cut(c) for c in cuts],
    }
    with open(CUTS_PATH, "w") as f:
        json.dump(cuts_payload, f)
    print(f"Cuts saved → {CUTS_PATH}  ({len(cuts):,} cuts)")

    # ── results ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Objective (UB) : {upper_bound:.2f}")
    print(f"Lower bound    : {lower_bound:.2f}")
    chargers_built = [k for k in feas_list if best_x.get(k, 0) > 0.5]
    print(f"Chargers built : {len(chargers_built)}")

    result_df = nodes_df.copy()
    result_df["x_built"] = result_df["key"].apply(
        lambda k: 1 if k in feasible_keys and best_x.get(k, 0) > 0.5 else 0
    )
    result_df.drop(columns="key").to_csv("models/results_nodes_250.csv", index=False)
    print("Saved → models/results_nodes_250.csv")


if __name__ == "__main__":
    main()
