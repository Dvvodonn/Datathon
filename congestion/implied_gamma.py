"""
congestion/implied_gamma.py
===========================
Compute the implied gamma (grid-cost shadow price) of Spain's existing
highway charging network via inverse optimization.

For each existing station with c >= 2 chargers, using actual per-connector
power records from data/raw/chargers/sites.csv:

    1. Extract the per-connector power list [p1, p2, ..., pc] (kW, sorted asc)
    2. "Remove" the weakest charger (p_min = most marginal install decision)
    3. Recompute mu from mean of remaining c-1 chargers
    4. gamma_k = (W_q(c-1, mu_remaining) - W_q(c, mu_all)) / p_min

This is strictly more accurate than using mean_power_kw for both sides.

Run
---
    python congestion/implied_gamma.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from queuing import wq_minutes

SITES_CSV       = Path("data/raw/chargers/sites.csv")
EXISTING_DEMAND = cfg.OUTPUTS_DIR / "existing_demand.csv"
OUTPUT_PNG      = Path("visualizations/implied_gamma_existing.png")


def load_per_connector_powers(sites_csv: Path) -> pd.DataFrame:
    """
    Load sites.csv and return a DataFrame with columns:
        latitude, longitude, powers  (list of kW values, one per connector)
    """
    print("Loading per-connector power data from sites.csv …")
    sites = pd.read_csv(sites_csv, low_memory=False)
    print(f"  {len(sites):,} charging sites")

    pw_cols = [
        c for c in sites.columns
        if "refillPoint" in c and "connector[1]/maxPowerAtSocket[1]" in c
    ]
    pw_data = sites[pw_cols].apply(pd.to_numeric, errors="coerce") / 1000.0  # W → kW

    def _powers(row):
        vals = row.dropna().tolist()
        return sorted(vals) if vals else []

    sites["powers"] = pw_data.apply(_powers, axis=1)
    return sites[["latitude", "longitude", "powers"]].copy()


def match_to_existing(existing_df: pd.DataFrame,
                      sites_df: pd.DataFrame) -> pd.DataFrame:
    """
    Nearest-neighbour match existing_demand stations to sites by lat/lon.
    Returns existing_df with an added 'powers' column.
    """
    sites_xy    = np.radians(sites_df[["latitude", "longitude"]].values)
    existing_xy = np.radians(existing_df[["lat", "lon"]].values)

    tree = cKDTree(sites_xy)
    dists_rad, idx = tree.query(existing_xy, k=1)
    dists_m = dists_rad * 6_371_000

    print(f"  Match distances (m): "
          f"p50={np.median(dists_m):.1f}  p95={np.percentile(dists_m, 95):.1f}  "
          f"max={dists_m.max():.1f}")

    result = existing_df.copy()
    result["powers"] = [sites_df["powers"].iloc[i] for i in idx]
    return result


def compute_implied_gamma(existing_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each station with >= 2 connectors in the per-connector power list,
    compute the implied gamma by removing the weakest charger.
    """
    rows = []
    for _, row in existing_df.iterrows():
        powers = row["powers"]          # sorted list of kW values (ascending)
        lam    = float(row["lambda_k"])

        if len(powers) < 2:
            continue
        if lam < 0.001:
            continue

        c_all = len(powers)
        p_removed = powers[0]           # weakest charger (most marginal)
        p_remaining = powers[1:]        # c-1 remaining

        if p_removed <= 0:
            continue

        mu_all       = np.mean(powers)       / cfg.E_SESSION_KWH
        mu_remaining = np.mean(p_remaining)  / cfg.E_SESSION_KWH

        wq_c   = wq_minutes(lam, mu_all,       c_all)        # W_q with all c
        wq_cm1 = wq_minutes(lam, mu_remaining, c_all - 1)    # W_q without weakest

        # Skip if both are saturated (no information content)
        if wq_c >= cfg.WQ_LARGE_PENALTY and wq_cm1 >= cfg.WQ_LARGE_PENALTY:
            continue

        delta_wq = max(0.0, wq_cm1 - wq_c)     # minutes saved by weakest charger
        gamma_k  = delta_wq / p_removed         # implied gamma (min / kW)

        rows.append({
            "lat":           row["lat"],
            "lon":           row["lon"],
            "n_chargers":    c_all,
            "p_removed_kw":  p_removed,
            "mu_all":        round(mu_all, 4),
            "mu_remaining":  round(mu_remaining, 4),
            "lambda_k":      lam,
            "wq_c":          round(wq_c, 4),
            "wq_cm1":        round(wq_cm1, 4),
            "delta_wq":      round(delta_wq, 4),
            "gamma_implied": round(gamma_k, 6),
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Implied gamma of existing charging network")
    print("(using per-connector power from sites.csv)")
    print("=" * 60)

    existing_df = pd.read_csv(EXISTING_DEMAND)
    print(f"\nLoaded {len(existing_df):,} existing stations from demand CSV")

    sites_df    = load_per_connector_powers(SITES_CSV)
    existing_df = match_to_existing(existing_df, sites_df)

    n_multi = existing_df["powers"].apply(len).ge(2).sum()
    print(f"  Stations with >= 2 per-connector records: {n_multi:,}")

    result_df = compute_implied_gamma(existing_df)
    n_valid   = len(result_df)
    print(f"  Valid stations for gamma estimation: {n_valid:,}")

    if n_valid == 0:
        print("No valid stations found.")
        return

    gammas = result_df["gamma_implied"]
    gammas_clipped = gammas.clip(upper=gammas.quantile(0.99))

    print(f"\n--- Implied gamma distribution ({n_valid:,} stations) ---")
    print(f"  Min    : {gammas.min():.4f}")
    print(f"  p25    : {gammas.quantile(0.25):.4f}")
    print(f"  Median : {gammas.median():.4f}   ← best single estimate")
    print(f"  Mean   : {gammas.mean():.4f}")
    print(f"  p75    : {gammas.quantile(0.75):.4f}")
    print(f"  p99    : {gammas.quantile(0.99):.4f}")
    print(f"  Max    : {gammas.max():.4f}")

    # kW-weighted median
    weights  = result_df["p_removed_kw"] * result_df["n_chargers"]
    weights /= weights.sum()
    sorted_idx    = gammas.argsort().values
    cum_w         = weights.iloc[sorted_idx].cumsum().values
    weighted_med  = gammas.iloc[sorted_idx[np.searchsorted(cum_w, 0.5)]]
    print(f"\n  Weighted median (by station kW): {weighted_med:.4f}")

    implied = gammas.median()
    print(f"\n  Sweep range: gamma ∈ [0.05, 1.0]")
    if implied < 0.05:
        print(f"  → Implied gamma {implied:.4f} BELOW sweep range "
              f"(existing network over-built vs our cost model)")
    elif implied > 1.0:
        print(f"  → Implied gamma {implied:.4f} ABOVE sweep range "
              f"(existing network under-built vs our cost model)")
    else:
        print(f"  → Implied gamma {implied:.4f} WITHIN sweep range ✓")

    # ── Plot ──────────────────────────────────────────────────────────────────
    Path("visualizations").mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(gammas_clipped, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(implied, color="crimson", lw=2, label=f"Median = {implied:.3f}")
    ax.axvline(weighted_med, color="darkorange", lw=2, ls="--",
               label=f"Weighted median = {weighted_med:.3f}")
    for g in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        ax.axvline(g, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Implied γ (min / kW)", fontsize=11)
    ax.set_ylabel("Number of stations", fontsize=11)
    ax.set_title("Implied γ per existing station\n(weakest-charger removal, per-connector powers)",
                 fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    sorted_g = np.sort(gammas.values)
    cdf = np.arange(1, len(sorted_g) + 1) / len(sorted_g)
    ax.plot(sorted_g, cdf, color="steelblue", lw=2)
    ax.axvline(implied, color="crimson", lw=2, ls="--",
               label=f"Median = {implied:.3f}")
    ax.axvline(weighted_med, color="darkorange", lw=2, ls=":",
               label=f"Weighted median = {weighted_med:.3f}")
    for g in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        ax.axvline(g, color="grey", lw=0.8, ls=":", alpha=0.6)
        ax.text(g, 0.02, f"{g}", fontsize=7, color="grey", ha="center")
    ax.set_xlabel("Implied γ (min / kW)", fontsize=11)
    ax.set_ylabel("Cumulative fraction of stations", fontsize=11)
    ax.set_title("CDF of implied γ\n(sweep gamma values marked in grey)", fontsize=11)
    ax.set_xlim(left=0, right=min(gammas.quantile(0.98), 2.0))
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
