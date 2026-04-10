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
    # for larger c values; log-space keeps the function safe as C_MAX grows.
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
    c_min: int = cfg.C_MIN,
    c_max: int = cfg.C_MAX,
    power_tiers: list = None,
    e_session_kwh: float = cfg.E_SESSION_KWH,
) -> pd.DataFrame:
    """
    For every candidate station k, charger count c ∈ [c_min, c_max], and
    power tier p ∈ power_tiers, compute W_q(λ_k, μ(p), c).

    Service rate μ(p) = p_kW / e_session_kwh  (vehicles / hour / charger)

    Parameters
    ----------
    demand_df     : DataFrame with columns 'lat', 'lon', 'lambda_k'
    power_tiers   : list of kW values, default cfg.POWER_TIERS = [50, 150, 350]

    Returns
    -------
    DataFrame with columns: lat, lon, c, p_kw, wq_minutes
    Saved to congestion/outputs/wq_table.csv.
    """
    if power_tiers is None:
        power_tiers = cfg.POWER_TIERS

    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in demand_df.iterrows():
        lam = float(row["lambda_k"])
        for c in range(c_min, c_max + 1):
            for p in power_tiers:
                mu_p = p / e_session_kwh
                wq   = wq_minutes(lam, mu_p, c)
                rows.append({
                    "lat":        row["lat"],
                    "lon":        row["lon"],
                    "c":          c,
                    "p_kw":       p,
                    "wq_minutes": round(wq, 6),
                })

    wq_df = pd.DataFrame(rows)

    n_tiers = len(power_tiers)
    n_cands = len(demand_df)
    n_c     = c_max - c_min + 1
    print(f"  W_q table: {len(wq_df):,} rows  "
          f"({n_cands:,} candidates × {n_c} capacity levels × {n_tiers} power tiers)")

    # Diagnostics per tier at c=1
    for p in power_tiers:
        mu_p = p / e_session_kwh
        rho  = demand_df["lambda_k"] / mu_p
        unstable = (rho >= 1.0).sum()
        print(f"  {p:>3} kW  μ={mu_p:.2f}  ρ(c=1) median={rho.median():.3f}  "
              f"p90={rho.quantile(0.9):.3f}  unstable={unstable}")

    out_path = cfg.OUTPUTS_DIR / "wq_table.csv"
    wq_df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return wq_df


def compute_existing_wq(
    existing_df: pd.DataFrame,
    e_session_kwh: float = cfg.E_SESSION_KWH,
) -> tuple:
    """
    Compute W_q for each existing charger station at its actual (n_chargers, power).

    Parameters
    ----------
    existing_df : DataFrame from build_existing_demand() with columns
                  lat, lon, lambda_k, n_chargers, mean_power_kw

    Returns
    -------
    (total_wq, per_station_df)
      total_wq       — sum of W_q across all existing stations (minutes)
      per_station_df — existing_df with added wq_minutes and mu_k columns
    """
    result = existing_df.copy()
    wq_list = []
    mu_list  = []

    for _, row in existing_df.iterrows():
        lam  = float(row["lambda_k"])
        c_k  = max(1, int(row["n_chargers"]))
        p_k  = float(row["mean_power_kw"])
        mu_k = p_k / e_session_kwh if p_k > 0 else 22.0 / e_session_kwh
        wq   = wq_minutes(lam, mu_k, c_k)
        wq_list.append(round(wq, 4))
        mu_list.append(round(mu_k, 4))

    result["mu_k"]       = mu_list
    result["wq_minutes"] = wq_list
    total_wq = result["wq_minutes"].sum()

    saturated = (result["wq_minutes"] >= cfg.WQ_LARGE_PENALTY).sum()
    print(f"  Existing charger W_q: total={total_wq:,.1f} min  "
          f"mean={result['wq_minutes'].mean():.1f}  "
          f"saturated={saturated}")
    return total_wq, result


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
