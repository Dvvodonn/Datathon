"""
congestion/efficient_frontier_gamma.py
=======================================
Sweep γ (grid-connection cost per kW) and trace the Pareto frontier of:

    x-axis : total installed capacity  Σ c·p_kW  (kW)
    y-axis : total driver stop time    Σ W(λ,μ,c) (minutes)
    colour : number of stations built

Each solve warm-starts from congestion/outputs/congestion_cuts.json so
subsequent runs are fast (cuts already dense from the γ=1.0 run).

Output
------
    congestion/outputs/gamma_frontier.csv
    visualizations/gamma_frontier.png

Run
---
    python congestion/efficient_frontier_gamma.py
"""

import multiprocessing
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ── Constants (module-level so worker subprocesses can import safely) ─────────
GAMMA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70,
                1.00, 1.50, 2.00, 3.00, 5.00, 7.00, 10.0]

CUTS_PATH  = str(cfg.OUTPUTS_DIR / "congestion_cuts.json")
OUTPUT_CSV = cfg.OUTPUTS_DIR / "gamma_frontier.csv"
OUTPUT_PNG = Path("visualizations/gamma_frontier.png")


def _run_sweep():
    """All execution inside this function, called only from __main__."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    import pandas as pd
    from model import main as run_model

    records = []

    for gamma in GAMMA_VALUES:
        tag = f"g{str(gamma).replace('.', '_')}"
        print(f"\n{'='*60}")
        print(f"  γ = {gamma}  (tag={tag})")
        print(f"{'='*60}")

        try:
            results_df, ub, lb = run_model(
                gamma=gamma,
                beta=cfg.BETA,
                c_min=cfg.C_MIN,
                c_max=cfg.C_MAX,
                cuts_in_path=CUTS_PATH,
                tag=tag,
            )

            built = results_df[results_df["x_built"] == 1]
            n_stations     = len(built)
            total_chargers = int(built["c_built"].sum()) if n_stations else 0
            total_kw       = int((built["c_built"] * built["p_built_kw"]).sum()) if n_stations else 0
            total_wq       = float(built["wq_minutes"].sum()) if n_stations else 0.0

            total_w = 0.0
            if n_stations:
                for _, r in built.iterrows():
                    p = float(r["p_built_kw"])
                    if p > 0:
                        total_w += float(r["wq_minutes"]) + 60.0 * cfg.E_SESSION_KWH / p

            power_dist = dict(built["p_built_kw"].value_counts().sort_index()) if n_stations else {}

            rec = {
                "gamma":        gamma,
                "n_stations":   n_stations,
                "n_chargers":   total_chargers,
                "total_kw":     total_kw,
                "total_wq_min": total_wq,
                "total_w_min":  total_w,
                "objective_ub": ub,
                "objective_lb": lb,
                "gap_pct":      (ub - lb) / max(ub, 1e-9) * 100,
                "power_dist":   str(power_dist),
            }
            records.append(rec)
            print(f"  → {n_stations} stations | {total_chargers} chargers | "
                  f"{total_kw:,} kW total | W={total_w:.0f} min | gap={rec['gap_pct']:.2f}%")

        except Exception as e:
            print(f"  ERROR at γ={gamma}: {e}")
            import traceback; traceback.print_exc()
            records.append({"gamma": gamma, "error": str(e)})

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFrontier saved → {OUTPUT_CSV}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_df = df.dropna(subset=["total_kw", "total_w_min"]).copy()
    plot_df = plot_df.sort_values("total_kw")

    if plot_df.empty:
        print("No valid results to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0d0d0d")
    fig.suptitle("γ Frontier — Grid Cost vs Driver Stop Time",
                 color="white", fontsize=14, y=1.01)

    cmap_n = mcm.get_cmap("plasma")
    norm_n = mcolors.Normalize(vmin=plot_df["n_stations"].min(),
                               vmax=plot_df["n_stations"].max())

    for ax in axes:
        ax.set_facecolor("#0d0d0d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # Left: total kW installed vs total W
    ax = axes[0]
    sc = ax.scatter(plot_df["total_kw"], plot_df["total_w_min"],
                    c=plot_df["n_stations"], cmap=cmap_n, norm=norm_n,
                    s=120, zorder=3, edgecolors="white", linewidths=0.5)
    ax.plot(plot_df["total_kw"], plot_df["total_w_min"],
            color="#666666", linewidth=1, zorder=2)
    for _, r in plot_df.iterrows():
        ax.annotate(f"γ={r['gamma']}", (r["total_kw"], r["total_w_min"]),
                    textcoords="offset points", xytext=(6, 4),
                    color="white", fontsize=7, alpha=0.8)
    ax.set_xlabel("Total installed capacity (kW)", fontsize=11)
    ax.set_ylabel("Total driver stop time — W (minutes)", fontsize=11)
    ax.set_title("Pareto Frontier: Infrastructure vs Service Quality")
    ax.grid(color="#333333", linewidth=0.5)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Stations built", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.outline.set_edgecolor("white")

    # Right: gamma vs station count
    ax = axes[1]
    ax.plot(plot_df["gamma"], plot_df["n_stations"],
            color="#44dd44", linewidth=2, marker="o",
            markersize=7, markerfacecolor="white", markeredgecolor="#44dd44")
    ax.set_xlabel("γ — grid cost per kW (min-equivalent)", fontsize=11)
    ax.set_ylabel("Number of stations built", fontsize=11)
    ax.set_title("Station Count vs γ")
    ax.set_xscale("log")
    ax.grid(color="#333333", linewidth=0.5)
    for _, r in plot_df.iterrows():
        ax.annotate(f"{int(r['n_stations'])}",
                    (r["gamma"], r["n_stations"]),
                    textcoords="offset points", xytext=(5, 4),
                    color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"Plot saved → {OUTPUT_PNG}")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    _run_sweep()
