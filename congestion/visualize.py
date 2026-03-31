"""
congestion/visualize.py — Maps for the congestion demand signal and solution.

Two plots
---------
plot_demand_map()
    Spain map with all candidate stations coloured by λ_k (EV arrival rate).
    Tier-2 and Tier-3 imputed candidates are marked differently so data-gap
    areas are immediately visible.

plot_wq_solution()
    Spain map showing built stations sized by c_built (charger count) and
    coloured by W_q (expected waiting time).  Overlays existing charger
    locations as a reference layer.

Run standalone (produces both maps from outputs/ CSVs):
    python congestion/visualize.py
"""

import sys
from pathlib import Path

import contextily as ctx
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ── Shared helpers ────────────────────────────────────────────────────────────

_SPAIN_3857 = dict(xmin=-1_100_000, xmax=420_000, ymin=4_200_000, ymax=5_500_000)

def _wgs84_to_3857(lon, lat):
    """Inline WGS84 → Web Mercator (no pyproj required for plotting)."""
    x = lon * 20_037_508.342789244 / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    y = y * 20_037_508.342789244 / 180.0
    return x, y


def _add_basemap(ax):
    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.CartoDB.DarkMatterNoLabels,
                        reset_extent=False, zoom=6)
    except Exception as exc:
        print(f"  [basemap] skipped (offline?): {exc}")


def _project_df(df):
    """Add x3857, y3857 columns from lon/lat."""
    xy = [_wgs84_to_3857(r.lon, r.lat) for r in df.itertuples()]
    df = df.copy()
    df["x3857"] = [p[0] for p in xy]
    df["y3857"] = [p[1] for p in xy]
    return df


# ── Plot 1: demand map ────────────────────────────────────────────────────────

def plot_demand_map(
    demand_df: pd.DataFrame = None,
    output_path: str = None,
):
    """
    Candidate stations coloured by λ_k.
    Tier 1 = filled circle; Tier 2 = triangle; Tier 3 = cross.
    """
    if demand_df is None:
        p = cfg.OUTPUTS_DIR / "candidate_demand.csv"
        if not p.exists():
            print(f"  {p} not found — run demand.py first"); return
        demand_df = pd.read_csv(p)

    if output_path is None:
        output_path = cfg.OUTPUTS_DIR / "demand_map.png"

    df = _project_df(demand_df)

    cmap = plt.cm.plasma
    lam  = df["lambda_k"].values
    vmin, vmax = np.percentile(lam[lam > 0], [5, 95]) if (lam > 0).any() else (0, 1)
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, 1e-5))

    tier_markers = {1: ("o", 30, 1.0), 2: ("^", 20, 0.8), 3: ("x", 15, 0.6)}

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    ax.set_xlim(_SPAIN_3857["xmin"], _SPAIN_3857["xmax"])
    ax.set_ylim(_SPAIN_3857["ymin"], _SPAIN_3857["ymax"])
    ax.set_aspect("equal")
    ax.axis("off")

    _add_basemap(ax)

    for tier, (marker, size, alpha_v) in tier_markers.items():
        sub = df[df["impute_tier"] == tier]
        if sub.empty:
            continue
        lam_t = sub["lambda_k"].values
        colors = cmap(norm(np.clip(lam_t, vmin, vmax)))
        ax.scatter(sub["x3857"], sub["y3857"],
                   c=colors, s=size, marker=marker,
                   alpha=alpha_v, linewidths=0.3,
                   label=f"Tier {tier} ({len(sub):,})")

    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.018, pad=0.01, aspect=35)
    cbar.set_label("λ_k  EV vehicles / hour (log scale)",
                   color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=9)
    cbar.outline.set_edgecolor("white")

    leg = ax.legend(loc="lower left", fontsize=9, framealpha=0.4,
                    labelcolor="white", facecolor="#0f1117",
                    title="Imputation tier", title_fontsize=9)
    leg.get_title().set_color("white")

    ax.set_title("EV Charging Demand Signal — Candidate Stations  (λ_k, vehicles/hour)",
                 color="white", fontsize=14, pad=12, fontweight="bold")

    plt.tight_layout(pad=0.5)
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ── Plot 2: solution map with W_q ─────────────────────────────────────────────

def plot_wq_solution(
    results_df: pd.DataFrame = None,
    output_path: str = None,
):
    """
    Built stations sized by c_built and coloured by wq_minutes.
    Existing charger locations shown as a faint background layer.
    """
    if results_df is None:
        p = cfg.OUTPUTS_DIR / "results_congestion.csv"
        if not p.exists():
            print(f"  {p} not found — run model.py first"); return
        results_df = pd.read_csv(p)

    if output_path is None:
        output_path = cfg.OUTPUTS_DIR / "solution_congestion.png"

    df = _project_df(results_df)

    existing   = df[df["is_existing_charger"] == 1]
    built      = df[(df["x_built"] == 1) & (df["is_feasible_location"] == 1)]

    cmap = plt.cm.RdYlGn_r
    wq   = built["wq_minutes"].values if len(built) > 0 else np.array([0.0])
    vmin = max(wq.min(), 0)
    vmax = wq.max() if wq.max() > vmin else vmin + 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    ax.set_xlim(_SPAIN_3857["xmin"], _SPAIN_3857["xmax"])
    ax.set_ylim(_SPAIN_3857["ymin"], _SPAIN_3857["ymax"])
    ax.set_aspect("equal")
    ax.axis("off")

    _add_basemap(ax)

    # Existing chargers — faint background
    if not existing.empty:
        ax.scatter(existing["x3857"], existing["y3857"],
                   c="steelblue", s=6, alpha=0.25, marker=".",
                   label=f"Existing chargers ({len(existing):,})", zorder=2)

    # Built stations — sized by charger count, coloured by W_q
    if not built.empty:
        sizes  = (built["c_built"].values * 40).clip(40, 400)
        colors = cmap(norm(wq))
        sc = ax.scatter(built["x3857"], built["y3857"],
                        c=colors, s=sizes, marker="o",
                        edgecolors="white", linewidths=0.6,
                        alpha=0.95, zorder=4,
                        label=f"Built stations ({len(built):,})")

        # Annotate charger count
        for _, r in built.iterrows():
            ax.text(r["x3857"], r["y3857"], str(int(r["c_built"])),
                    color="white", fontsize=6, ha="center", va="center",
                    fontweight="bold", zorder=5)

    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.018, pad=0.01, aspect=35)
    cbar.set_label("Expected waiting time W_q  (minutes)",
                   color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=9)
    cbar.outline.set_edgecolor("white")

    leg = ax.legend(loc="lower left", fontsize=9, framealpha=0.4,
                    labelcolor="white", facecolor="#0f1117")
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)

    n_chargers = int(built["c_built"].sum()) if not built.empty else 0
    mean_wq    = round(float(built["wq_minutes"].mean()), 2) if not built.empty else 0.0
    ax.set_title(
        f"Congestion-aware Solution — {len(built)} stations  |  "
        f"{n_chargers} chargers  |  mean W_q = {mean_wq} min",
        color="white", fontsize=13, pad=12, fontweight="bold",
    )

    plt.tight_layout(pad=0.5)
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Visualise congestion outputs ===")
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_demand_map()
    plot_wq_solution()
    print("Done.")
