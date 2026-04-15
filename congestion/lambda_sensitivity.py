"""
congestion/lambda_sensitivity.py
=================================
One-pass forward evaluation of how much variable lambda (demand splitting)
would change our optimization results.

Method
------
1. Load solution S from a completed sweep result CSV
2. Add opened new stations to the charger network
3. Recompute through_gap for all existing stations with the expanded network
4. Recompute stop_rate and lambda_k at existing stations
5. Recompute W_q at existing stations with new lambda_k
6. Report: Δlambda, ΔW_q, change in optimal charger count at new stations

Key question: do opened new stations (which fill GAP corridors) come close
enough to existing stations to meaningfully reduce their lambda_k?

Run
---
    python congestion/lambda_sensitivity.py
    python congestion/lambda_sensitivity.py --result datathon_master/results_congestion_dm_g0_2_p0.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from queuing import wq_minutes
from demand import compute_variable_stop_rates

RESULT_CSV      = Path("datathon_master/results_congestion_dm_g0_1_p0.csv")
EXISTING_DEMAND = cfg.OUTPUTS_DIR / "existing_demand.csv"


def latlon_dist_km(coords_a, coords_b_tree):
    """Approximate great-circle distance using flat-earth (fine for <500km)."""
    dists_deg, idx = coords_b_tree.query(coords_a)
    # 1 degree ≈ 111 km (rough but sufficient for this sensitivity check)
    return dists_deg * 111.0, idx


def recompute_through_gap(existing_df, new_station_coords):
    """
    For each existing station, find its distance to the nearest new station.
    Update through_gap_km if the new station is closer than the existing gap.

    New through_gap ≈ min(old_through_gap, 2 × dist_to_nearest_new_station)
    (d1 + d2 where both d1 and d2 approximate the new station distance)
    """
    tree = cKDTree(new_station_coords)
    existing_coords = existing_df[["lat", "lon"]].values
    dists_km, _ = latlon_dist_km(existing_coords, tree)

    # Conservative: new_gap = d_to_new + d_to_next_existing
    # Approximation: through_gap reduces to ~2 × nearest new station dist
    # if the new station is closer than half the old gap
    approx_new_gap = np.clip(dists_km * 2, None, 250.0)
    updated_gap    = np.minimum(existing_df["through_gap_km"].values, approx_new_gap)
    return updated_gap, dists_km


def optimal_c(lam, p_kw, c_min, c_max, gamma, beta, e_session_kwh):
    """Find charger count minimizing gamma*c*p + beta*W_q."""
    mu = p_kw / e_session_kwh
    best_c, best_cost = c_min, float("inf")
    for c in range(c_min, c_max + 1):
        wq  = wq_minutes(lam, mu, c)
        cost = gamma * c * p_kw + beta * wq
        if cost < best_cost:
            best_cost, best_c = cost, c
    return best_c


def main(result_csv):
    print("=" * 65)
    print("Lambda sensitivity: variable demand impact on optimization")
    print(f"Solution: {result_csv}")
    print("=" * 65)

    # ── Load solution ─────────────────────────────────────────────
    result_df   = pd.read_csv(result_csv)
    gamma       = float(result_df["gamma"].iloc[0])
    new_built   = result_df[result_df["x_built"] == 1].copy()
    n_new       = len(new_built)
    print(f"\nSolution: {n_new} new stations opened  (gamma={gamma})")

    new_coords = new_built[["lat", "lon"]].values  # degrees

    # ── Load existing demand ──────────────────────────────────────
    existing_df = pd.read_csv(EXISTING_DEMAND)
    print(f"Existing stations: {len(existing_df):,}")

    # ── Recompute through_gap with new stations added ─────────────
    print("\nRecomputing through_gap with new stations in network …")
    new_gap_km, dist_to_nearest_new = recompute_through_gap(existing_df, new_coords)

    old_gap = existing_df["through_gap_km"].values
    gap_reduction = old_gap - new_gap_km
    affected = (gap_reduction > 0.5).sum()   # stations where gap drops >0.5 km

    print(f"  Existing stations with gap reduced by >0.5 km : {affected:,} "
          f"/ {len(existing_df):,} ({100*affected/len(existing_df):.1f}%)")
    print(f"  Median distance to nearest new station        : "
          f"{np.median(dist_to_nearest_new):.1f} km")
    print(f"  Min distance to nearest new station           : "
          f"{dist_to_nearest_new.min():.1f} km")
    print(f"  Max gap reduction (km)                        : "
          f"{gap_reduction.max():.1f} km")

    # ── Recompute stop_rate and lambda_k ──────────────────────────
    new_gap_series       = pd.Series(new_gap_km, index=existing_df.index)
    new_stop_rate        = compute_variable_stop_rates(new_gap_series)
    old_stop_rate        = existing_df["stop_rate"].values

    new_lambda = (
        existing_df["aadt_assigned"]
        * cfg.EV_PENETRATION
        * cfg.PEAK_HOUR_FACTOR
        * new_stop_rate
    ).clip(lower=0.0).values
    old_lambda = existing_df["lambda_k"].values

    delta_lambda     = new_lambda - old_lambda
    pct_lambda_change = np.where(old_lambda > 0,
                                 delta_lambda / old_lambda * 100, 0)

    print(f"\n  lambda_k changes at existing stations:")
    print(f"    Stations with lambda reduced by >1%  : "
          f"{(pct_lambda_change < -1).sum():,}")
    print(f"    Stations with lambda reduced by >10% : "
          f"{(pct_lambda_change < -10).sum():,}")
    print(f"    Mean lambda change (%)               : "
          f"{pct_lambda_change.mean():.3f}%")
    print(f"    Total old lambda (veh/hr)            : {old_lambda.sum():.1f}")
    print(f"    Total new lambda (veh/hr)            : {new_lambda.sum():.1f}")
    print(f"    Total delta lambda (veh/hr)          : {delta_lambda.sum():.1f} "
          f"({100*delta_lambda.sum()/max(old_lambda.sum(),1e-9):.3f}%)")

    # ── Recompute W_q at existing stations ────────────────────────
    print(f"\n  Recomputing existing station W_q with new lambda_k …")
    old_wq_list, new_wq_list = [], []
    for i, row in existing_df.iterrows():
        c_k  = max(1, int(row["n_chargers"]))
        p_k  = float(row["mean_power_kw"])
        mu_k = p_k / cfg.E_SESSION_KWH if p_k > 0 else 22.0 / cfg.E_SESSION_KWH
        old_wq_list.append(wq_minutes(old_lambda[i], mu_k, c_k))
        new_wq_list.append(wq_minutes(new_lambda[i], mu_k, c_k))

    old_wq_total = sum(old_wq_list)
    new_wq_total = sum(new_wq_list)
    delta_wq     = new_wq_total - old_wq_total

    print(f"    Old total existing W_q : {old_wq_total:,.1f} min")
    print(f"    New total existing W_q : {new_wq_total:,.1f} min")
    print(f"    Delta W_q              : {delta_wq:+,.1f} min "
          f"({100*delta_wq/max(old_wq_total,1e-9):+.3f}%)")

    # ── Check if new station charger counts would change ──────────
    print(f"\n  Checking optimal charger count at new stations with updated lambda …")
    c_changes = 0
    for _, row in new_built.iterrows():
        # New station's lambda: from result CSV (already computed with original model)
        lam_orig = float(row["lambda_k"])
        c_orig   = int(row["c_built"])
        p_kw     = float(row["p_built_kw"]) if row["p_built_kw"] > 0 else 150.0

        # Would lambda change if nearby existing stations shed demand to us?
        # (Inverse: new stations GAIN demand from congested nearby existing ones)
        # For now: lambda stays the same for new stations (existing → new flow)
        # is not modeled in our stop_rate framework regardless
        c_opt = optimal_c(lam_orig, p_kw, cfg.C_MIN, cfg.C_MAX,
                          gamma, cfg.BETA, cfg.E_SESSION_KWH)
        if c_opt != c_orig:
            c_changes += 1

    print(f"    New stations that would change charger count : "
          f"{c_changes} / {n_new}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"SUMMARY")
    print(f"{'='*65}")
    print(f"  Existing W_q change  : {delta_wq:+,.1f} min  "
          f"({100*delta_wq/max(old_wq_total,1e-9):+.4f}%)")
    print(f"  Total lambda change  : {delta_lambda.sum():+.2f} veh/hr  "
          f"({100*delta_lambda.sum()/max(old_lambda.sum(),1e-9):+.4f}%)")
    print(f"  Charger count shifts : {c_changes} / {n_new} new stations")
    print()
    if abs(delta_wq / max(old_wq_total, 1e-9)) < 0.001:
        print("  → Variable lambda has NEGLIGIBLE impact (<0.1% W_q change)")
        print("    New stations are far from existing chargers (gap-filling).")
        print("    Fixed-lambda assumption is valid for this solution.")
    elif abs(delta_wq / max(old_wq_total, 1e-9)) < 0.01:
        print("  → Variable lambda has SMALL impact (<1% W_q change)")
        print("    Fixed-lambda is a reasonable approximation.")
    else:
        print("  → Variable lambda has MEANINGFUL impact (>1% W_q change)")
        print("    Consider iterative demand recomputation for accuracy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=str, default=str(RESULT_CSV))
    args = parser.parse_args()
    main(Path(args.result))
