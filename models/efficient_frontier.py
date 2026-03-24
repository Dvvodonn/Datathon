"""
efficient_frontier.py — sweep alpha to trace the Pareto frontier
================================================================
Loads saved Benders cuts and re-solves the master MIP for each alpha
value without re-running subproblems.  Each solve takes seconds.

Output:
  models/efficient_frontier.csv   — alpha, n_chargers, travel_time, objective
  visualizations/efficient_frontier.png
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp

CUTS_PATH    = "models/benders_cuts.json"
OUT_CSV      = "models/efficient_frontier.csv"
OUT_PNG      = "visualizations/efficient_frontier.png"

# ── load cuts ─────────────────────────────────────────────────────────────────
print("Loading cuts …")
with open(CUTS_PATH) as f:
    payload = json.load(f)

feas_list  = [tuple(k) for k in payload["feas_list"]]
city_pairs = [(tuple(u), tuple(v)) for u, v in payload["city_pairs"]]
cuts_raw   = payload["cuts"]

def _load_cut(c):
    c = dict(c)
    for field in ("u", "v"):
        if field in c:
            c[field] = tuple(c[field])
    if "nodes" in c:
        c["nodes"] = [tuple(k) for k in c["nodes"]]
    return c

cuts = [_load_cut(c) for c in cuts_raw]
print(f"  {len(feas_list):,} feasible locations  |  {len(city_pairs):,} city pairs  |  {len(cuts):,} cuts")

feas_idx = {k: i for i, k in enumerate(feas_list)}


# ── master solve for a given alpha ────────────────────────────────────────────
def solve_master(alpha):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    solver.SuppressOutput()
    solver.SetTimeLimit(60_000)  # 60s per solve

    x   = [solver.BoolVar(f"x_{i}") for i in range(len(feas_list))]
    eta = {(u, v): solver.NumVar(0, solver.infinity(), f"eta_{u}_{v}")
           for (u, v) in city_pairs}

    obj = solver.Objective()
    for xi in x:
        obj.SetCoefficient(xi, alpha)
    for e in eta.values():
        obj.SetCoefficient(e, 1.0)
    obj.SetMinimization()

    for cut in cuts:
        if cut["type"] == "feasibility":
            ct = solver.Constraint(1, solver.infinity())
            for k in cut["nodes"]:
                ct.SetCoefficient(x[feas_idx[k]], 1.0)

        elif cut["type"] == "optimality":
            u, v, t_star, nodes = cut["u"], cut["v"], cut["t"], cut["nodes"]
            rhs = t_star - t_star * len(nodes)
            ct  = solver.Constraint(rhs, solver.infinity())
            ct.SetCoefficient(eta[(u, v)], 1.0)
            for k in nodes:
                ct.SetCoefficient(x[feas_idx[k]], -t_star)

        elif cut["type"] == "fixed_eta":
            u, v, t_star = cut["u"], cut["v"], cut["t"]
            ct = solver.Constraint(t_star, solver.infinity())
            ct.SetCoefficient(eta[(u, v)], 1.0)

        elif cut["type"] == "cond_lower":
            u, v = cut["u"], cut["v"]
            t_opt_c, t_full_c, nodes = cut["t_opt"], cut["t_full"], cut["nodes"]
            delta = t_opt_c - t_full_c
            ct = solver.Constraint(t_opt_c, solver.infinity())
            ct.SetCoefficient(eta[(u, v)], 1.0)
            for k in nodes:
                ct.SetCoefficient(x[feas_idx[k]], delta)

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None

    n_chargers  = sum(1 for xi in x if xi.solution_value() > 0.5)
    travel_time = sum(e.solution_value() for e in eta.values())
    objective   = solver.Objective().Value()
    return n_chargers, travel_time, objective


# ── alpha sweep ───────────────────────────────────────────────────────────────
# Start at alpha=1, grow geometrically until we hit 0 chargers.
# Then fill in any large jumps in charger count with bisection.

print("\nSweeping alpha …\n")
results = []

def run(alpha):
    t0  = time.time()
    res = solve_master(alpha)
    elapsed = time.time() - t0
    if res is None:
        print(f"  alpha={alpha:>10.1f}  →  solver failed")
        return None
    n, tt, obj = res
    print(f"  alpha={alpha:>10.1f}  →  {n:3d} chargers  |  travel={tt:,.0f}  |  obj={obj:,.0f}  ({elapsed:.1f}s)")
    return {"alpha": alpha, "n_chargers": n, "travel_time": tt, "objective": obj}

# Phase 1: geometric sweep to find zero-charger threshold
alpha = 1.0
prev_n = None
while True:
    r = run(alpha)
    if r is not None:
        results.append(r)
        if r["n_chargers"] == 0:
            break
        prev_n = r["n_chargers"]
    alpha = alpha * 1.5 if alpha < 100 else alpha * 1.3

# Phase 2: bisect large gaps in charger count for smoother frontier
print("\nFilling gaps …")
results.sort(key=lambda r: r["alpha"])
i = 0
while i < len(results) - 1:
    a1, n1 = results[i]["alpha"],   results[i]["n_chargers"]
    a2, n2 = results[i+1]["alpha"], results[i+1]["n_chargers"]
    if abs(n1 - n2) > 2 and (a2 - a1) > 1:
        mid = (a1 + a2) / 2
        r   = run(mid)
        if r is not None:
            results.append(r)
            results.sort(key=lambda r: r["alpha"])
            continue  # re-check this interval
    i += 1

# ── save + plot ───────────────────────────────────────────────────────────────
df = pd.DataFrame(results).sort_values("alpha").reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="black")
fig.suptitle("EV Charging Network — Efficient Frontier (250km range)", color="white", fontsize=14)

for ax in axes:
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

# Left: travel time vs n_chargers
ax = axes[0]
ax.plot(df["n_chargers"], df["travel_time"] / 1e6, color="#00ee66", linewidth=2, marker="o", markersize=4)
ax.set_xlabel("Number of chargers built")
ax.set_ylabel("Total travel time (M min)")
ax.set_title("Travel time vs Chargers", color="white")
ax.invert_xaxis()

# Right: objective vs alpha
ax = axes[1]
ax.plot(df["alpha"], df["objective"] / 1e6, color="#4488ff", linewidth=2, marker="o", markersize=4)
ax.set_xlabel("Alpha (cost per charger, min)")
ax.set_ylabel("Objective (M min)")
ax.set_title("Objective vs Alpha", color="white")
ax.set_xscale("log")

# Mark alpha=100 point
row100 = df.iloc[(df["alpha"] - 100).abs().argsort()[:1]]
for ax_i, col in zip(axes, ["n_chargers", "alpha"]):
    x_val = row100[col].values[0]
    y_col = "travel_time" if col == "n_chargers" else "objective"
    ax_i.axvline(x_val, color="#ffaa00", linestyle="--", alpha=0.6, label="α=100")

axes[0].legend(facecolor="#111111", edgecolor="#444444", labelcolor="white")
axes[1].legend(facecolor="#111111", edgecolor="#444444", labelcolor="white")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="black")
print(f"Saved → {OUT_PNG}")
print(f"\nFrontier: {len(df)} points, alpha {df.alpha.min():.1f} → {df.alpha.max():.1f}, "
      f"chargers {df.n_chargers.max()} → {df.n_chargers.min()}")
