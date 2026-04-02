"""
congestion/model.py — Extended EV charger location model with M/M/c congestion penalty.

Extension over models/model_1.py
---------------------------------
Decision variables
  x[k]        ∈ {0,1}   station k is opened                    (same as model_1)
  w[k][c][p]  ∈ {0,1}   station k opened with c chargers at p kW  (replaces y)

Constraints
  Σ_{c,p} w[k][c][p] == x[k]   for all k  (open station → exactly one (c,p) assignment)

Objective
  Minimize:
    Σ_{u,v} η[u,v]                                    travel-time term  (Benders)
  + GAMMA · Σ_{k,c,p} w[k][c][p] · c · p              grid-connection cost (total station kW)
  + BETA  · Σ_{k,c,p} w[k][c][p] · W[k,c,p]          total driver stop time

  W[k,c,p] = W_q(λ_k, μ(p), c) + 60·E_SESSION_KWH/p   [minutes]
    ↑ queue waiting time            ↑ actual charging session time

  Using W (total sojourn) instead of W_q alone ensures the model prefers higher-kW
  chargers at busy corridors where shorter session times offset the queue wait.
  Both terms are in minutes; all three objective components are min-equivalent.

Benders subproblems
  Unchanged from model_1.py.  Routing depends only on x[k] (open/closed);
  (c,p) decisions live entirely in the master.

Warm start
  Cuts loaded from models/benders_cuts.json (produced by model_1.py).
  New cuts written to congestion/outputs/congestion_cuts.json.

Run
---
    python congestion/model.py                    # default gamma and beta
    python congestion/model.py --beta 2.0
    python congestion/model.py --gamma 0.5
    python congestion/model.py --gamma 1.0 --beta 1.0
"""

import argparse
import heapq
import json
import multiprocessing as mp
import sys
import time as timer
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import pandas as pd
from ortools.linear_solver import pywraplp

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from demand import build_demand
from queuing import precompute_wq_table


# ── Node key (mirrors model_1.py exactly) ────────────────────────────────────

def nkey(lon, lat):
    return (round(float(lon), 6), round(float(lat), 6))


# ── Dijkstra (identical logic to model_1.py) ─────────────────────────────────

def dijkstra_single_source(src, open_nodes, adj, city_keys=None):
    INF  = float("inf")
    dist = {src: 0.0}
    prev = {src: None}
    pq   = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF):
            continue
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


# ── Parallel worker (mirrors model_1.py globals) ──────────────────────────────

_open_nodes_global = None
_full_nodes_global = None
_city_list_global  = None
_city_keys_global  = None
_adj_global        = None


def _worker_init(open_nodes, full_nodes, city_list, city_keys, adj):
    global _open_nodes_global, _full_nodes_global
    global _city_list_global, _city_keys_global, _adj_global
    _open_nodes_global = open_nodes
    _full_nodes_global = full_nodes
    _city_list_global  = city_list
    _city_keys_global  = city_keys
    _adj_global        = adj


def _solve_source(src):
    dist_opt,  prev_opt  = dijkstra_single_source(
        src, _open_nodes_global, _adj_global, city_keys=_city_keys_global)
    dist_full, prev_full = dijkstra_single_source(
        src, _full_nodes_global, _adj_global)
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


# ── Master problem with congestion extension ──────────────────────────────────

def build_master_congestion(
    feas_list, city_pairs, cuts, w_total_lookup,
    beta, gamma, c_min, c_max, power_tiers,
):
    """
    Master MIP with w[i][j_c][j_p] ∈ {0,1} decision variables.

    w[i][j_c][j_p] = 1  ↔  station i is open with
                             c = c_min + j_c  chargers at
                             p = power_tiers[j_p]  kW

    Objective coefficient for w[i][j_c][j_p]:
        gamma * c * p                  (station grid-connection cost ∝ total kW)
      + beta  * W[k, c, p]             (total driver stop time = W_q + session)

    x[i] is kept for Benders cut compatibility:
        x[i] == Σ_{c,p} w[i][j_c][j_p]

    Parameters
    ----------
    w_total_lookup : dict  (lon, lat, c, p_kw) → W_q + session_minutes
    gamma          : float  cost per kW of total station power
    power_tiers    : list   of kW values, e.g. [50, 100, 150, 200, 250, 350]
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    solver.SuppressOutput()
    solver.SetTimeLimit(300_000)  # 5 min per master solve

    feas_idx  = {k: i for i, k in enumerate(feas_list)}
    n         = len(feas_list)
    n_c       = c_max - c_min + 1
    n_p       = len(power_tiers)

    # x[i] ∈ {0,1}  — station open/closed (Benders cuts reference x)
    x = [solver.BoolVar(f"x_{i}") for i in range(n)]

    # w[i][j_c][j_p] ∈ {0,1}
    # j_c ∈ 0..n_c-1  (actual c = c_min + j_c)
    # j_p ∈ 0..n_p-1  (actual p = power_tiers[j_p])
    w = [[[solver.BoolVar(f"w_{i}_{j_c}_{j_p}")
           for j_p in range(n_p)]
          for j_c in range(n_c)]
         for i in range(n)]

    # η[u,v] ≥ 0
    eta = {(u, v): solver.NumVar(0, solver.infinity(), f"eta_{u}_{v}")
           for (u, v) in city_pairs}

    # ── Linking: Σ_{c,p} w[i][j_c][j_p] == x[i] ──────────────────────────────
    for i in range(n):
        ct = solver.Constraint(0, 0)
        ct.SetCoefficient(x[i], -1.0)
        for j_c in range(n_c):
            for j_p in range(n_p):
                ct.SetCoefficient(w[i][j_c][j_p], 1.0)

    # ── Objective ─────────────────────────────────────────────────────────────
    obj = solver.Objective()
    for evar in eta.values():
        obj.SetCoefficient(evar, 1.0)

    for i, k in enumerate(feas_list):
        klon, klat = float(k[0]), float(k[1])
        for j_c, c in enumerate(range(c_min, c_max + 1)):
            for j_p, p in enumerate(power_tiers):
                w_total = w_total_lookup.get((klon, klat, c, p), 0.0)
                coeff   = gamma * c * p + beta * w_total
                obj.SetCoefficient(w[i][j_c][j_p], coeff)

    obj.SetMinimization()

    # ── Benders cuts (reference x[i], unchanged from model_1.py) ─────────────
    for cut in cuts:
        if cut["type"] == "feasibility":
            ct = solver.Constraint(1, solver.infinity())
            for k in cut["nodes"]:
                ct.SetCoefficient(x[feas_idx[tuple(k)]], 1.0)

        elif cut["type"] == "optimality":
            u, v, t_star, nodes = cut["u"], cut["v"], cut["t"], cut["nodes"]
            rhs = t_star - t_star * len(nodes)
            ct  = solver.Constraint(rhs, solver.infinity())
            ct.SetCoefficient(eta[(tuple(u), tuple(v))], 1.0)
            for k in nodes:
                ct.SetCoefficient(x[feas_idx[tuple(k)]], -t_star)

        elif cut["type"] == "fixed_eta":
            u, v, t_star = cut["u"], cut["v"], cut["t"]
            ct = solver.Constraint(t_star, solver.infinity())
            ct.SetCoefficient(eta[(tuple(u), tuple(v))], 1.0)

        elif cut["type"] == "cond_lower":
            u, v = cut["u"], cut["v"]
            t_opt_c, t_full_c, nodes = cut["t_opt"], cut["t_full"], cut["nodes"]
            delta = t_opt_c - t_full_c
            ct = solver.Constraint(t_opt_c, solver.infinity())
            ct.SetCoefficient(eta[(tuple(u), tuple(v))], 1.0)
            for k in nodes:
                ct.SetCoefficient(x[feas_idx[tuple(k)]], delta)

    return solver, x, w, eta, feas_idx


# ── Upper-bound helper ────────────────────────────────────────────────────────

def _compute_ub(total_t_opt, w_vals, w_total_lookup,
                feas_list, beta, gamma, c_min, c_max, power_tiers):
    """Compute upper bound: routing cost + grid cost + total driver stop time."""
    grid_cost   = 0.0
    driver_cost = 0.0
    for i, k in enumerate(feas_list):
        klon, klat = float(k[0]), float(k[1])
        for j_c, c in enumerate(range(c_min, c_max + 1)):
            for j_p, p in enumerate(power_tiers):
                if w_vals[i][j_c][j_p] > 0.5:
                    grid_cost   += gamma * c * p
                    driver_cost += beta * w_total_lookup.get((klon, klat, c, p), 0.0)
    return total_t_opt + grid_cost + driver_cost


# ── Main Benders loop ─────────────────────────────────────────────────────────

def main(
    beta:         float = cfg.BETA,
    gamma:        float = cfg.GAMMA,
    c_min:        int   = cfg.C_MIN,
    c_max:        int   = cfg.C_MAX,
    power_tiers:  list  = None,
    cuts_in_path: str   = cfg.BENDERS_CUTS,
    tag:          str   = "",
):
    """
    Run the congestion-extended Benders decomposition with joint (c, p) decisions.

    Objective per station:
        gamma * c * p_kW           (grid-connection cost ∝ total station power)
      + beta  * W(λ_k, μ(p), c)   (total driver stop time = W_q + session time)

    Using W instead of W_q alone drives the solver to prefer higher-kW chargers at
    high-demand sites (shorter session times outweigh any increase in queue wait).

    Existing chargers use actual (n_chargers, mean_power_kw) from DGT data — their
    W_q is a fixed constant reported alongside the optimised cost but not in the MIP.

    Saves results to congestion/outputs/results_congestion{tag}.csv.
    """
    if power_tiers is None:
        power_tiers = cfg.POWER_TIERS

    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, mp.cpu_count() - 1)

    print(f"\n{'='*65}")
    print(f"Congestion model  |  β={beta}  γ={gamma}  c∈[{c_min},{c_max}]  "
          f"tiers={power_tiers} kW")
    print(f"  Objective: γ·c·p (grid cost) + β·W(λ,μ,c) (total stop time)")
    print(f"{'='*65}")

    # ── Candidate demand ──────────────────────────────────────────────────────
    demand_path = cfg.OUTPUTS_DIR / "candidate_demand.csv"
    if demand_path.exists():
        demand_df = pd.read_csv(demand_path)
        # Invalidate if built with flat stop rate (missing through_gap_km)
        if "through_gap_km" not in demand_df.columns:
            print("Demand file uses flat stop rate — regenerating with variable SR …")
            demand_df = build_demand()
        else:
            print("Loading pre-computed candidate demand (variable stop rate) …")
    else:
        print("Building candidate demand …")
        demand_df = build_demand()

    # ── Existing charger demand + fixed W_q ──────────────────────────────────
    from demand import build_existing_demand
    from queuing import compute_existing_wq

    existing_path = cfg.OUTPUTS_DIR / "existing_demand.csv"
    if existing_path.exists():
        print("Loading pre-computed existing charger demand …")
        existing_df = pd.read_csv(existing_path)
    else:
        print("Building existing charger demand …")
        existing_df = build_existing_demand()

    print("Computing fixed W_q for existing chargers …")
    fixed_existing_wq, existing_wq_df = compute_existing_wq(existing_df)

    # ── W_q table for candidates: (lon, lat, c, p_kw) → wq_minutes ───────────
    wq_path = cfg.OUTPUTS_DIR / "wq_table.csv"
    need_regen = True
    if wq_path.exists():
        wq_df = pd.read_csv(wq_path)
        if "p_kw" in wq_df.columns:
            existing_tiers = sorted(wq_df["p_kw"].unique().astype(int).tolist())
            if existing_tiers == sorted(power_tiers):
                need_regen = False
                print("Loading pre-computed W_q table …")
            else:
                print(f"  W_q table tiers {existing_tiers} ≠ {sorted(power_tiers)} — regenerating")
        else:
            print("  W_q table missing p_kw column — regenerating")
    if need_regen:
        print("Pre-computing W_q table …")
        wq_df = precompute_wq_table(demand_df, c_min=c_min, c_max=c_max,
                                    power_tiers=power_tiers)

    # Build two lookups:
    #   wq_lookup       : (lon, lat, c, p) → W_q minutes only  (for results reporting)
    #   w_total_lookup  : (lon, lat, c, p) → W_q + session_min  (for objective)
    wq_lookup       = {}
    w_total_lookup  = {}
    for _, row in wq_df.iterrows():
        key         = (round(row["lon"], 6), round(row["lat"], 6),
                       int(row["c"]), int(row["p_kw"]))
        wq          = float(row["wq_minutes"])
        session_min = 60.0 * cfg.E_SESSION_KWH / float(row["p_kw"])
        wq_lookup[key]      = wq
        w_total_lookup[key] = wq + session_min

    # ── Load nodes and edges ──────────────────────────────────────────────────
    print("Loading nodes and edges …")
    nodes_df = pd.read_csv(cfg.NODES_CSV)
    edges_df = pd.read_csv(cfg.EDGES_250_CSV)
    print(f"  {len(nodes_df):,} nodes  |  {len(edges_df):,} edges")

    nodes_df["key"] = [nkey(r.lon, r.lat) for r in nodes_df.itertuples()]
    city_keys     = set(nodes_df.loc[nodes_df.is_city == 1,              "key"])
    feasible_keys = set(nodes_df.loc[nodes_df.is_feasible_location == 1, "key"])
    charger_keys  = set(nodes_df.loc[nodes_df.is_existing_charger == 1,  "key"])
    all_keys      = city_keys | feasible_keys | charger_keys

    city_list  = sorted(city_keys)[:cfg.N_CITIES]
    city_set   = set(city_list)
    city_pairs_full = [(u, v) for u in city_list for v in city_list if u != v]

    adj = defaultdict(list)
    for row in edges_df.itertuples(index=False):
        fk = nkey(row.lon_a, row.lat_a)
        tk = nkey(row.lon_b, row.lat_b)
        if fk in all_keys and tk in all_keys:
            adj[fk].append((tk, float(row.estimated_time_min)))
    print(f"  {sum(len(v) for v in adj.values()):,} adjacency entries")

    # ── Load warm-start cuts ──────────────────────────────────────────────────
    print(f"Loading warm-start cuts from {cuts_in_path} …")
    with open(cuts_in_path) as f:
        saved = json.load(f)

    # feas_list from cuts file is authoritative (edges were built against it)
    feas_list   = [tuple(k) for k in saved["feas_list"]]
    city_pairs  = [(tuple(u), tuple(v)) for u, v in saved["city_pairs"]]
    cuts        = saved["cuts"]
    seen_cuts   = set()

    print(f"  {len(feas_list):,} feasible locations  |  "
          f"{len(city_pairs):,} city pairs  |  "
          f"{len(cuts):,} warm-start cuts")

    full_nodes = city_set | charger_keys | set(feas_list)
    adj_dict   = dict(adj)

    upper_bound = float("inf")
    lower_bound = 0.0
    n_c = c_max - c_min + 1
    n_p = len(power_tiers)
    best_x = {k: 0 for k in feas_list}
    best_w = [[[0 for _ in range(n_p)] for _ in range(n_c)]
              for _ in range(len(feas_list))]

    n_w_vars = len(feas_list) * n_c * n_p
    print(f"\nStarting Benders loop ({cfg.MAX_ITER} max iters, {n_workers} workers) …")
    print(f"  Master MIP: {n_w_vars:,} w-vars + "
          f"{len(feas_list):,} x-vars + {len(city_pairs):,} η-vars\n")

    for iteration in range(1, cfg.MAX_ITER + 1):
        t0 = timer.time()

        # ── Solve master ──────────────────────────────────────────────────────
        solver, x, w, eta, feas_idx = build_master_congestion(
            feas_list, city_pairs, cuts, w_total_lookup,
            beta, gamma, c_min, c_max, power_tiers,
        )
        status = solver.Solve()

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            print(f"Master returned status {status} — stopping.")
            break

        master_proven = (status == pywraplp.Solver.OPTIMAL)
        if master_proven:
            lower_bound = solver.Objective().Value()

        # Extract solution values
        x_vals = {feas_list[i]: x[i].solution_value() for i in range(len(feas_list))}
        w_vals = [[[w[i][j_c][j_p].solution_value()
                    for j_p in range(n_p)]
                   for j_c in range(n_c)]
                  for i in range(len(feas_list))]
        eta_vals = {(u, v): eta[(u, v)].solution_value() for (u, v) in eta}

        open_nodes = city_set | charger_keys | {k for k in feas_list if x_vals[k] > 0.5}

        # ── Subproblems (parallel Dijkstra — unchanged from model_1.py) ───────
        with mp.Pool(n_workers, initializer=_worker_init,
                     initargs=(open_nodes, full_nodes, city_list, city_set, adj_dict)) as pool:
            source_results = pool.map(_solve_source, city_list, chunksize=1)

        results = [row for src_rows in source_results for row in src_rows]

        # ── Generate cuts + compute upper bound ───────────────────────────────
        always_open  = city_set | charger_keys
        reachable    = set(city_pairs)
        cuts_added   = 0
        total_t_opt  = 0.0
        infeasible   = 0

        for (u, v, t_opt, path_opt, t_full, path_full) in results:
            if (u, v) not in reachable:
                continue

            if t_opt == float("inf"):
                infeasible += 1
                total_t_opt += cfg.PENALTY_INFEASIBLE

                pen_key = ("fixed", u, v, float(cfg.PENALTY_INFEASIBLE))
                if pen_key not in seen_cuts:
                    seen_cuts.add(pen_key)
                    cuts.append({"type": "fixed_eta", "u": list(u),
                                 "v": list(v), "t": cfg.PENALTY_INFEASIBLE})
                    cuts_added += 1

                feas_on_full = [k for k in path_full
                                if k in set(feas_list) and k not in always_open]
                if feas_on_full:
                    fkey = ("feas", frozenset(map(tuple, feas_on_full)))
                    if fkey not in seen_cuts:
                        seen_cuts.add(fkey)
                        cuts.append({"type": "feasibility",
                                     "nodes": [list(k) for k in feas_on_full]})
                        cuts_added += 1
            else:
                total_t_opt += t_opt
                highway_better = t_full < float("inf") and t_full < t_opt - 0.01
                t_lb = t_full if highway_better else t_opt

                fixed_key = ("fixed", u, v, round(t_lb, 4))
                if fixed_key not in seen_cuts:
                    seen_cuts.add(fixed_key)
                    cuts.append({"type": "fixed_eta", "u": list(u),
                                 "v": list(v), "t": t_lb})
                    cuts_added += 1

                if highway_better:
                    feas_on_full = [k for k in path_full
                                    if k in set(feas_list) and k not in always_open]
                    if feas_on_full:
                        if eta_vals[(u, v)] < t_full - 0.01:
                            okey = ("opt", u, v, round(t_full, 4),
                                    frozenset(map(tuple, feas_on_full)))
                            if okey not in seen_cuts:
                                seen_cuts.add(okey)
                                cuts.append({"type": "optimality",
                                             "u": list(u), "v": list(v),
                                             "t": t_full,
                                             "nodes": [list(k) for k in feas_on_full]})
                                cuts_added += 1

                        if eta_vals[(u, v)] < t_opt - 0.01:
                            ckey = ("cond", u, v, round(t_opt, 4),
                                    round(t_full, 4),
                                    frozenset(map(tuple, feas_on_full)))
                            if ckey not in seen_cuts:
                                seen_cuts.add(ckey)
                                cuts.append({"type": "cond_lower",
                                             "u": list(u), "v": list(v),
                                             "t_opt": t_opt, "t_full": t_full,
                                             "nodes": [list(k) for k in feas_on_full]})
                                cuts_added += 1

        current_ub = _compute_ub(total_t_opt, w_vals, w_total_lookup,
                                  feas_list, beta, gamma, c_min, c_max, power_tiers)

        if current_ub < upper_bound:
            upper_bound = current_ub
            best_x = dict(x_vals)
            best_w = [[[w_vals[i][j_c][j_p] for j_p in range(n_p)]
                       for j_c in range(n_c)]
                      for i in range(len(feas_list))]

        gap = (upper_bound - lower_bound) / max(upper_bound, 1e-9) * 100
        proven_str = "OPT" if master_proven else "feas"
        print(f"  Iter {iteration:3d} | LB {lower_bound:12.1f} | UB {upper_bound:12.1f} | "
              f"Gap {gap:6.2f}% | {proven_str} | "
              f"Cuts +{cuts_added:4d} (tot {len(cuts):5d}) | "
              f"Infeas {infeasible} | {timer.time()-t0:.1f}s")

        if gap < cfg.CONV_TOL:
            print(f"\nConverged at iteration {iteration}.")
            break
        if cuts_added == 0:
            print(f"\nNo new cuts — search exhausted at iteration {iteration}.")
            break

    # ── Save cuts ─────────────────────────────────────────────────────────────
    tag_str = f"_{tag}" if tag else ""
    cuts_out = cfg.OUTPUTS_DIR / f"congestion_cuts{tag_str}.json"
    with open(cuts_out, "w") as f:
        json.dump({
            "beta": beta, "gamma": gamma, "c_min": c_min, "c_max": c_max,
            "power_tiers": power_tiers,
            "feas_list":  [list(k) for k in feas_list],
            "city_pairs": [[list(u), list(v)] for u, v in city_pairs],
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "fixed_existing_wq": fixed_existing_wq,
            "cuts": cuts,
        }, f)
    print(f"Cuts saved → {cuts_out}  ({len(cuts):,} cuts)")

    # ── Build result DataFrame ────────────────────────────────────────────────
    nodes_result = pd.read_csv(cfg.NODES_CSV)
    nodes_result["key"] = [nkey(r.lon, r.lat) for r in nodes_result.itertuples()]

    # Determine built (c, p) for each selected feasible station
    feas_set     = set(feas_list)
    c_built_map  = {}
    p_built_map  = {}
    wq_built_map = {}
    for i, k in enumerate(feas_list):
        if best_x.get(k, 0) > 0.5:
            klon, klat = float(k[0]), float(k[1])
            for j_c, c in enumerate(range(c_min, c_max + 1)):
                for j_p, p in enumerate(power_tiers):
                    if best_w[i][j_c][j_p] > 0.5:
                        c_built_map[k]  = c
                        p_built_map[k]  = p
                        wq_built_map[k] = wq_lookup.get((klon, klat, c, p), 0.0)
                        break

    nodes_result["x_built"]    = nodes_result["key"].apply(
        lambda k: 1 if k in feas_set and best_x.get(k, 0) > 0.5 else 0)
    nodes_result["c_built"]    = nodes_result["key"].apply(
        lambda k: c_built_map.get(k, 0))
    nodes_result["p_built_kw"] = nodes_result["key"].apply(
        lambda k: p_built_map.get(k, 0))
    nodes_result["wq_minutes"] = nodes_result["key"].apply(
        lambda k: round(wq_built_map.get(k, 0.0), 4))

    # Merge λ_k from demand
    demand_keys = {
        nkey(r["lon"], r["lat"]): r["lambda_k"]
        for _, r in demand_df.iterrows()
    }
    nodes_result["lambda_k"] = nodes_result["key"].apply(
        lambda k: round(demand_keys.get(k, 0.0), 6))

    nodes_result = nodes_result.drop(columns=["key"])
    results_out = cfg.OUTPUTS_DIR / f"results_congestion{tag_str}.csv"
    nodes_result.to_csv(results_out, index=False)
    print(f"Results saved → {results_out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    built = nodes_result[nodes_result["x_built"] == 1]
    total_system_wq = (built["wq_minutes"].sum() + fixed_existing_wq
                       if len(built) else fixed_existing_wq)
    print(f"\n{'='*65}")
    print(f"Objective (UB)               : {upper_bound:.2f}")
    print(f"Lower bound                  : {lower_bound:.2f}")
    print(f"Stations built               : {len(built)}")
    if len(built) > 0:
        print(f"Chargers total               : {int(built['c_built'].sum())}")
        print(f"Power dist (kW)              : "
              f"{dict(built['p_built_kw'].value_counts().sort_index())}")
        print(f"Charger dist                 : "
              f"{dict(built['c_built'].value_counts().sort_index())}")
        # W_q (queue wait only) and W (total stop = W_q + session)
        new_wq = built["wq_minutes"].sum()
        new_w  = sum(
            w_total_lookup.get(
                (round(float(r["lon"]), 6), round(float(r["lat"]), 6),
                 int(r["c_built"]), int(r["p_built_kw"])), 0.0)
            for _, r in built.iterrows()
        )
        print(f"New station W_q (wait only)  : {new_wq:.1f} min")
        print(f"New station W   (total stop) : {new_w:.1f} min")
    print(f"Existing charger W_q (fixed) : {fixed_existing_wq:.1f} min")
    print(f"Total system W_q             : {total_system_wq:.1f} min")
    print(f"{'='*65}\n")

    return nodes_result, upper_bound, lower_bound


# ── Efficient frontier: sweep BETA ───────────────────────────────────────────

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Congestion-extended EV charger location model"
    )
    parser.add_argument("--beta",  type=float, default=cfg.BETA,
                        help="Driver stop-time weight β (default %(default)s)")
    parser.add_argument("--gamma", type=float, default=cfg.GAMMA,
                        help="Grid-connection cost per kW γ (default %(default)s)")
    parser.add_argument("--c-max", type=int,   default=cfg.C_MAX,
                        help="Maximum chargers per station (default %(default)s)")
    parser.add_argument("--tag",   type=str,   default="",
                        help="Output file suffix, e.g. 'g05' → results_congestion_g05.csv")
    args = parser.parse_args()

    main(beta=args.beta, gamma=args.gamma, c_min=cfg.C_MIN,
         c_max=args.c_max, tag=args.tag)
