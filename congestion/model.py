"""
congestion/model.py — Extended EV charger location model with M/M/c congestion penalty.

Extension over models/model_1.py
---------------------------------
Decision variables
  x[k]     ∈ {0,1}   station k is opened               (same as model_1)
  y[k][c]  ∈ {0,1}   station k is opened with c chargers  (new)

Constraints
  Σ_c y[k][c]  ==  x[k]     for all k   (if open, choose exactly one capacity)
  y[k][c]      ≤   x[k]     for all k,c  (capacity only if station open)

Objective
  Minimize:
    Σ_{u,v} η[u,v]                            travel-time term  (same)
  + ALPHA · Σ_{k,c} c · y[k][c]               charger build cost (ALPHA per charger)
  + BETA  · Σ_{k,c} y[k][c] · W_q[k][c]       congestion penalty (precomputed)

  W_q[k][c] is the expected waiting time (minutes) at station k with c chargers,
  computed by congestion/queuing.py before the solver runs.  The combined coefficient
  (ALPHA·c + BETA·W_q[k][c]) is a scalar per (k,c) — the MIP stays purely linear.

Benders subproblems
  Unchanged from model_1.py.  The routing subproblems depend only on which stations
  are open (x[k]), not on how many chargers are installed (y[k][c]).  The congestion
  term lives entirely in the master problem.

Warm start
  Cuts are loaded from models/benders_cuts.json (produced by model_1.py).  These
  cuts constrain η[u,v] as a function of x[k] and are valid regardless of the
  cost structure.  New cuts generated here are written to
  congestion/outputs/congestion_cuts.json and do NOT overwrite the original.

Run
---
    # single solve at default BETA
    python congestion/model.py

    # override BETA
    python congestion/model.py --beta 2.0

    # Pareto frontier sweep over BETA values
    python congestion/model.py --frontier
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

def build_master_congestion(feas_list, city_pairs, cuts, wq_lookup, alpha, beta, c_min, c_max):
    """
    Rebuild the master MIP with the congestion extension.

    New decision variables: y[i][c] ∈ {0,1} for each candidate i and charger
    count c.  x[i] is kept as an explicit variable linked to y via equality.

    Objective coefficient for y[i][c]:
        ALPHA * c   (charger build cost)
      + BETA * W_q[k][c]   (congestion waiting penalty, precomputed)

    Parameters
    ----------
    wq_lookup : dict mapping (lon, lat, c) → wq_minutes
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    solver.SuppressOutput()
    solver.SetTimeLimit(300_000)  # 5 min per master solve

    feas_idx = {k: i for i, k in enumerate(feas_list)}
    n        = len(feas_list)

    # x[i] ∈ {0,1}  — station open/closed (needed for cut compatibility)
    x = [solver.BoolVar(f"x_{i}") for i in range(n)]

    # y[i][c] ∈ {0,1}  — station i open with exactly c chargers
    y = [[solver.BoolVar(f"y_{i}_{c}") for c in range(c_min, c_max + 1)]
         for i in range(n)]
    # y[i] is indexed 0..c_max-c_min; actual charger count = c_min + j

    # η[u,v] ≥ 0  for reachable city pairs
    eta = {(u, v): solver.NumVar(0, solver.infinity(), f"eta_{u}_{v}")
           for (u, v) in city_pairs}

    # ── Linking constraints: Σ_c y[i][c] == x[i] ─────────────────────────────
    for i in range(n):
        ct = solver.Constraint(0, 0)          # Σ y[i][c] - x[i] == 0
        ct.SetCoefficient(x[i], -1.0)
        for yic in y[i]:
            ct.SetCoefficient(yic, 1.0)

    # ── Objective ─────────────────────────────────────────────────────────────
    obj = solver.Objective()
    for evar in eta.values():
        obj.SetCoefficient(evar, 1.0)

    for i, k in enumerate(feas_list):
        klon, klat = float(k[0]), float(k[1])
        for j, c in enumerate(range(c_min, c_max + 1)):
            wq  = wq_lookup.get((klon, klat, c), 0.0)
            coeff = alpha * c + beta * wq
            obj.SetCoefficient(y[i][j], coeff)

    obj.SetMinimization()

    # ── Benders cuts (identical structure to model_1.py) ─────────────────────
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

    return solver, x, y, eta, feas_idx


# ── Upper-bound helper ────────────────────────────────────────────────────────

def _compute_ub(total_t_opt, y_vals, wq_lookup, feas_list, alpha, beta, c_min, c_max):
    """Compute upper bound from routing cost + charger costs + congestion penalty."""
    charger_cost    = 0.0
    congestion_cost = 0.0
    for i, k in enumerate(feas_list):
        klon, klat = float(k[0]), float(k[1])
        for j, c in enumerate(range(c_min, c_max + 1)):
            yval = y_vals[i][j]
            if yval > 0.5:
                charger_cost    += alpha * c
                congestion_cost += beta * wq_lookup.get((klon, klat, c), 0.0)
    return total_t_opt + charger_cost + congestion_cost


# ── Main Benders loop ─────────────────────────────────────────────────────────

def main(
    alpha:  float = cfg.ALPHA,
    beta:   float = cfg.BETA,
    mu:     float = cfg.MU_PER_CHARGER,
    c_min:  int   = cfg.C_MIN,
    c_max:  int   = cfg.C_MAX,
    cuts_in_path: str = cfg.BENDERS_CUTS,
    tag:    str   = "",
):
    """
    Run the congestion-extended Benders decomposition.

    Warm-starts from models/benders_cuts.json (cuts are independent of the
    cost structure and remain valid for the extended objective).

    Saves results to congestion/outputs/results_congestion{tag}.csv
    and updated cuts to congestion/outputs/congestion_cuts{tag}.json.
    """
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, mp.cpu_count() - 1)

    print(f"\n{'='*65}")
    print(f"Congestion model  |  α={alpha}  β={beta}  μ={mu}  "
          f"c∈[{c_min},{c_max}]")
    print(f"{'='*65}")

    # ── Demand and W_q ────────────────────────────────────────────────────────
    demand_path = cfg.OUTPUTS_DIR / "candidate_demand.csv"
    if demand_path.exists():
        print("Loading pre-computed demand …")
        demand_df = pd.read_csv(demand_path)
    else:
        print("Building demand signal …")
        demand_df = build_demand()

    wq_path = cfg.OUTPUTS_DIR / "wq_table.csv"
    if wq_path.exists():
        print("Loading pre-computed W_q table …")
        wq_df = pd.read_csv(wq_path)
    else:
        print("Pre-computing W_q table …")
        wq_df = precompute_wq_table(demand_df, mu=mu, c_min=c_min, c_max=c_max)

    # Build (lon, lat, c) → wq_minutes lookup
    wq_lookup = {
        (round(row["lon"], 6), round(row["lat"], 6), int(row["c"])): float(row["wq_minutes"])
        for _, row in wq_df.iterrows()
    }

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
    best_x      = {k: 0 for k in feas_list}
    best_y      = {(i, j): 0 for i in range(len(feas_list))
                              for j in range(c_max - c_min + 1)}

    print(f"\nStarting Benders loop ({cfg.MAX_ITER} max iters, {n_workers} workers) …\n")

    for iteration in range(1, cfg.MAX_ITER + 1):
        t0 = timer.time()

        # ── Solve master ──────────────────────────────────────────────────────
        solver, x, y, eta, feas_idx = build_master_congestion(
            feas_list, city_pairs, cuts, wq_lookup, alpha, beta, c_min, c_max
        )
        status = solver.Solve()

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            print(f"Master returned status {status} — stopping.")
            break

        master_proven = (status == pywraplp.Solver.OPTIMAL)
        if master_proven:
            lower_bound = solver.Objective().Value()

        # Extract x and y solution values
        x_vals = {feas_list[i]: x[i].solution_value() for i in range(len(feas_list))}
        y_vals = [[y[i][j].solution_value() for j in range(c_max - c_min + 1)]
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

        current_ub = _compute_ub(total_t_opt, y_vals, wq_lookup,
                                  feas_list, alpha, beta, c_min, c_max)

        if current_ub < upper_bound:
            upper_bound = current_ub
            best_x = dict(x_vals)
            best_y = [[y_vals[i][j] for j in range(c_max - c_min + 1)]
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
            "alpha": alpha, "beta": beta, "mu": mu,
            "c_min": c_min, "c_max": c_max,
            "feas_list":  [list(k) for k in feas_list],
            "city_pairs": [[list(u), list(v)] for u, v in city_pairs],
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "cuts": cuts,
        }, f)
    print(f"Cuts saved → {cuts_out}  ({len(cuts):,} cuts)")

    # ── Build result DataFrame ────────────────────────────────────────────────
    nodes_result = pd.read_csv(cfg.NODES_CSV)
    nodes_result["key"] = [nkey(r.lon, r.lat) for r in nodes_result.itertuples()]

    # Determine built capacity for each feasible station
    feas_set = set(feas_list)
    c_built_map = {}
    wq_built_map = {}
    for i, k in enumerate(feas_list):
        if best_x.get(k, 0) > 0.5:
            for j, c in enumerate(range(c_min, c_max + 1)):
                if best_y[i][j] > 0.5:
                    c_built_map[k]  = c
                    klon, klat = float(k[0]), float(k[1])
                    wq_built_map[k] = wq_lookup.get((klon, klat, c), 0.0)
                    break

    nodes_result["x_built"]    = nodes_result["key"].apply(
        lambda k: 1 if k in feas_set and best_x.get(k, 0) > 0.5 else 0)
    nodes_result["c_built"]    = nodes_result["key"].apply(
        lambda k: c_built_map.get(k, 0))
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
    print(f"\n{'='*65}")
    print(f"Objective (UB) : {upper_bound:.2f}")
    print(f"Lower bound    : {lower_bound:.2f}")
    print(f"Stations built : {len(built)}")
    if len(built) > 0:
        print(f"Chargers total : {built['c_built'].sum()}")
        print(f"Mean W_q built : {built['wq_minutes'].mean():.2f} min")
        print(f"Charger dist   : {dict(built['c_built'].value_counts().sort_index())}")
    print(f"{'='*65}\n")

    return nodes_result, upper_bound, lower_bound


# ── Efficient frontier: sweep BETA ───────────────────────────────────────────

def run_efficient_frontier(
    beta_values: list = None,
    alpha: float = cfg.ALPHA,
    mu:    float = cfg.MU_PER_CHARGER,
    c_min: int   = cfg.C_MIN,
    c_max: int   = cfg.C_MAX,
):
    """
    Sweep BETA over beta_values and record (beta, n_stations, n_chargers,
    travel_time, wq_total, objective) for each solve.

    Reuses the congestion_cuts.json from the first solve as warm start for
    subsequent solves (same pattern as models/efficient_frontier.py reusing
    benders_cuts.json).

    Saves congestion/outputs/efficient_frontier_congestion.csv.
    """
    if beta_values is None:
        beta_values = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    first_cuts = cfg.BENDERS_CUTS   # initial warm start from base model

    for beta in beta_values:
        print(f"\n--- β = {beta} ---")
        tag = f"beta{str(beta).replace('.', 'p')}"
        result_df, ub, lb = main(
            alpha=alpha, beta=beta, mu=mu,
            c_min=c_min, c_max=c_max,
            cuts_in_path=first_cuts, tag=tag,
        )
        # Use this solve's cuts as warm start for next β
        new_cuts = cfg.OUTPUTS_DIR / f"congestion_cuts_{tag}.json"
        if new_cuts.exists():
            first_cuts = str(new_cuts)

        built = result_df[result_df["x_built"] == 1]
        records.append({
            "beta":        beta,
            "n_stations":  len(built),
            "n_chargers":  int(built["c_built"].sum()),
            "wq_total":    round(built["wq_minutes"].sum(), 2),
            "objective":   round(ub, 2),
            "lower_bound": round(lb, 2),
        })

    frontier = pd.DataFrame(records)
    out = cfg.OUTPUTS_DIR / "efficient_frontier_congestion.csv"
    frontier.to_csv(out, index=False)
    print(f"\nFrontier saved → {out}")
    print(frontier.to_string(index=False))
    return frontier


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Congestion-extended EV charger location model"
    )
    parser.add_argument("--beta",     type=float, default=None,
                        help="Congestion weight β (overrides config.BETA)")
    parser.add_argument("--alpha",    type=float, default=cfg.ALPHA,
                        help="Cost per charger α (minutes)")
    parser.add_argument("--mu",       type=float, default=cfg.MU_PER_CHARGER,
                        help="Service rate μ (vehicles/hour/charger)")
    parser.add_argument("--c-max",    type=int,   default=cfg.C_MAX,
                        help="Maximum chargers per station")
    parser.add_argument("--frontier", action="store_true",
                        help="Sweep β values and produce Pareto frontier CSV")
    args = parser.parse_args()

    if args.frontier:
        run_efficient_frontier(alpha=args.alpha, mu=args.mu, c_max=args.c_max)
    else:
        beta = args.beta if args.beta is not None else cfg.BETA
        main(alpha=args.alpha, beta=beta, mu=args.mu,
             c_min=cfg.C_MIN, c_max=args.c_max)
