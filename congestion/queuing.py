"""
congestion/queuing.py — M/M/c queuing model for EV charging stations.

Theory
------
Arrival process : Poisson with rate λ (EV vehicles/hour)
Service process : Exponential with rate μ per charger (vehicles/hour/charger)
Servers         : c chargers (integer decision variable, 1 ≤ c ≤ C_MAX)

Stability condition: ρ = λ / (c · μ) < 1

Erlang-C gives the probability that an arriving vehicle must wait:

    C(c, a) = [ a^c / (c! · (1 − ρ)) ] / [ Σ_{k=0}^{c−1} a^k/k!  +  a^c/(c!·(1−ρ)) ]

where a = λ/μ is the offered traffic in Erlangs.

Expected waiting time in queue (minutes):

    W_q = C(c, a) / (c · μ − λ) · 60      [converts hours → minutes]

This is the penalty fed into the location model.

Usage
-----
    from congestion.queue import precompute_wq_table

    wq = precompute_wq_table(demand_df)   # demand_df has 'lambda_k' column
    # wq columns: lat, lon, c, wq_minutes

Run standalone (prints a demo table):
    python congestion/queuing.py
"""

import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ── Core M/M/c formulas ───────────────────────────────────────────────────────

def erlang_c(a: float, c: int) -> float:
    """
    Erlang-C formula: probability that an arriving customer must wait.

    Parameters
    ----------
    a : offered traffic in Erlangs = λ / μ  (not per-server utilisation)
    c : number of servers (chargers)

    Returns
    -------
    float in [0, 1].  Returns 1.0 when the queue is unstable (a ≥ c).
    """
    if a <= 0.0:
        return 0.0
    if a >= c:
        return 1.0

    rho = a / c  # per-server utilisation

    # Numerically stable computation using log-factorials avoids overflow
    # for large c values (c up to C_MAX=10 is fine with direct floats, but
    # log-space is safe for future extension to larger C_MAX).
    log_a = math.log(a)

    # Σ_{k=0}^{c-1} a^k / k!
    log_sum_terms = [k * log_a - sum(math.log(i) for i in range(1, k + 1))
                     for k in range(c)]
    # Use log-sum-exp for numerical stability
    max_log = max(log_sum_terms)
    sum_terms = math.exp(max_log) * sum(math.exp(lt - max_log) for lt in log_sum_terms)

    # a^c / (c! · (1 − ρ))
    log_last = c * log_a - sum(math.log(i) for i in range(1, c + 1)) - math.log(1.0 - rho)
    last = math.exp(log_last)

    return last / (sum_terms + last)


def wq_minutes(
    lam: float,
    mu: float  = cfg.MU_PER_CHARGER,
    c:   int   = 1,
    large_penalty: float = cfg.WQ_LARGE_PENALTY,
) -> float:
    """
    Expected waiting time in queue (minutes) for an M/M/c station.

    Parameters
    ----------
    lam   : arrival rate (EV vehicles / hour)
    mu    : service rate per charger (vehicles / hour / charger)
    c     : number of chargers
    large_penalty : returned when queue is unstable (ρ ≥ 1)
    """
    if lam <= 0.0:
        return 0.0
    if mu <= 0.0 or c < 1:
        raise ValueError(f"Invalid queue parameters: mu={mu}, c={c}")

    rho = lam / (c * mu)
    if rho >= 1.0:
        return large_penalty

    # Small epsilon guard to avoid near-singular division
    if rho > 0.999:
        return large_penalty

    a      = lam / mu
    pc     = erlang_c(a, c)
    # W_q in hours = C(c,a) / (c·μ − λ)
    wq_h   = pc / (c * mu - lam)
    return wq_h * 60.0  # convert to minutes


# ── Precompute W_q table ──────────────────────────────────────────────────────

def precompute_wq_table(
    demand_df: pd.DataFrame,
    mu:    float = cfg.MU_PER_CHARGER,
    c_min: int   = cfg.C_MIN,
    c_max: int   = cfg.C_MAX,
) -> pd.DataFrame:
    """
    For every candidate station k and every charger count c ∈ [c_min, c_max],
    compute the expected waiting time W_q(λ_k, μ, c).

    Parameters
    ----------
    demand_df : DataFrame with columns 'lat', 'lon', 'lambda_k'
                (output of demand.build_demand)

    Returns
    -------
    DataFrame with columns: lat, lon, c, wq_minutes
    Saved to congestion/outputs/wq_table.csv.

    The optimizer reads W_q[k, c] as a precomputed coefficient — the Erlang-C
    formula never appears inside the MIP.
    """
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in demand_df.iterrows():
        lam = float(row["lambda_k"])
        for c in range(c_min, c_max + 1):
            wq = wq_minutes(lam, mu, c)
            rows.append({
                "lat":        row["lat"],
                "lon":        row["lon"],
                "c":          c,
                "wq_minutes": round(wq, 6),
            })

    wq_df = pd.DataFrame(rows)

    # Diagnostic: utilisation at c=1 for all candidates
    c1 = wq_df[wq_df["c"] == 1].copy()
    c1 = c1.merge(demand_df[["lat", "lon", "lambda_k"]], on=["lat", "lon"], how="left")
    rho_c1 = c1["lambda_k"] / mu
    print(f"  W_q table: {len(wq_df):,} rows  "
          f"({len(demand_df):,} candidates × {c_max - c_min + 1} capacity levels)")
    print(f"  Utilisation ρ at c=1: "
          f"median={rho_c1.median():.4f}  "
          f"p90={rho_c1.quantile(0.90):.4f}  "
          f"max={rho_c1.max():.4f}")
    unstable_c1 = (rho_c1 >= 1.0).sum()
    if unstable_c1 > 0:
        print(f"  WARNING: {unstable_c1} candidates have ρ ≥ 1 at c=1 "
              f"(assigned WQ_LARGE_PENALTY={cfg.WQ_LARGE_PENALTY} min)")

    out_path = cfg.OUTPUTS_DIR / "wq_table.csv"
    wq_df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    return wq_df


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== M/M/c demo ===")
    print(f"  Service rate μ = {cfg.MU_PER_CHARGER} vehicles/hour/charger")
    print()

    # Load demand if available, otherwise generate a small demo
    demand_path = cfg.OUTPUTS_DIR / "candidate_demand.csv"
    if demand_path.exists():
        demand = pd.read_csv(demand_path)
        print(f"  Loaded demand from {demand_path}  ({len(demand):,} candidates)")
        wq = precompute_wq_table(demand)
        # Show W_q for 5 stations with highest λ at various c values
        top5 = demand.nlargest(5, "lambda_k")[["lat", "lon", "lambda_k"]]
        print("\n  Top-5 highest-demand stations:")
        for _, r in top5.iterrows():
            row_wq = wq[(wq["lat"] == r["lat"]) & (wq["lon"] == r["lon"])]
            vals = {int(rr["c"]): round(rr["wq_minutes"], 2) for _, rr in row_wq.iterrows()}
            print(f"    λ={r['lambda_k']:.4f}  W_q(c=1..5): "
                  + "  ".join(f"c={c}: {vals.get(c, '?')} min" for c in range(1, 6)))
    else:
        print(f"  {demand_path} not found — run demand.py first, or viewing demo values:")
        print()
        # Demo table: λ = 0.5 vehicles/hour
        lam = 0.5
        print(f"  λ = {lam} veh/h, μ = {cfg.MU_PER_CHARGER} veh/h/charger")
        print(f"  {'c':>4}  {'ρ':>6}  {'W_q (min)':>12}")
        print("  " + "-" * 26)
        for c in range(1, cfg.C_MAX + 1):
            rho = lam / (c * cfg.MU_PER_CHARGER)
            wq  = wq_minutes(lam, cfg.MU_PER_CHARGER, c)
            print(f"  {c:>4}  {rho:>6.4f}  {wq:>12.4f}")
