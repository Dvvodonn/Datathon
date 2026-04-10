"""
congestion/sweep_phi.py
=======================
Sweep the congestion model over a 2D grid of:

    gamma  : grid-connection cost per installed kW
    phi    : min fraction of opened stations with through_gap_km ≤ 100

Outputs go to datathon_master/.

Run
---
    python congestion/sweep_phi.py
    python congestion/sweep_phi.py --overwrite
    python congestion/sweep_phi.py --gammas 0.1,0.3,0.5 --phis 0.0,0.5,1.0
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from model import main as run_model

OUTPUT_DIR  = Path("datathon_master")
SUMMARY_CSV = OUTPUT_DIR / "gamma_phi_sweep.csv"
DEFAULT_CUTS = cfg.OUTPUTS_DIR / "congestion_cuts_g10_0.json"

DEFAULT_GAMMAS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
DEFAULT_PHIS   = [0.0, 0.25, 0.5, 0.75, 1.0]


def _parse_float_list(raw):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _tag_token(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "_")


def _safe_gap_pct(ub, lb):
    if ub is None or lb is None or not math.isfinite(ub):
        return float("nan")
    return (ub - lb) / max(ub, 1e-9) * 100.0


def _summarize(results_path, cuts_path, gamma, phi, tag):
    results_df = pd.read_csv(results_path)
    built = results_df[results_df["x_built"] == 1].copy()

    n_stations = len(built)
    n_chargers = int(built["c_built"].sum()) if n_stations else 0
    total_kw   = int((built["c_built"] * built["p_built_kw"]).sum()) if n_stations else 0
    total_wq   = float(built["wq_minutes"].sum()) if n_stations else 0.0
    total_w = 0.0
    if n_stations:
        total_w = float(
            (built["wq_minutes"] + 60.0 * cfg.E_SESSION_KWH / built["p_built_kw"]).sum()
        )

    ub, lb, n_cuts = float("nan"), float("nan"), float("nan")
    if cuts_path.exists():
        with open(cuts_path) as f:
            payload = json.load(f)
        ub     = float(payload.get("upper_bound", float("nan")))
        lb     = float(payload.get("lower_bound", float("nan")))
        n_cuts = len(payload.get("cuts", []))

    return {
        "gamma":    gamma,
        "phi":      phi,
        "tag":      tag,
        "results_csv": str(results_path),
        "cuts_json":   str(cuts_path),
        "n_stations":  n_stations,
        "n_chargers":  n_chargers,
        "avg_chargers_per_station": (n_chargers / n_stations) if n_stations else 0.0,
        "total_kw":       total_kw,
        "total_wq_min":   total_wq,
        "total_w_min":    total_w,
        "objective_ub":   ub,
        "objective_lb":   lb,
        "gap_pct":        _safe_gap_pct(ub, lb),
        "n_cuts":         n_cuts,
        "power_dist": json.dumps(
            {int(k): int(v) for k, v in built["p_built_kw"].value_counts().sort_index().items()}
        ),
        "charger_dist": json.dumps(
            {int(k): int(v) for k, v in built["c_built"].value_counts().sort_index().items()}
        ),
    }


def run_sweep(gammas, phis, cuts_path, summary_csv, overwrite=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cuts_path   = Path(cuts_path)
    summary_csv = Path(summary_csv)

    records   = []
    total     = len(gammas) * len(phis)
    job_idx   = 0

    for phi in phis:
        for gamma in gammas:
            job_idx += 1
            tag  = f"dm_g{_tag_token(gamma)}_p{_tag_token(phi)}"
            results_path = OUTPUT_DIR / f"results_congestion_{tag}.csv"
            cuts_out     = OUTPUT_DIR / f"congestion_cuts_{tag}.json"

            print(f"\n{'='*72}")
            print(f"[{job_idx:02d}/{total:02d}]  gamma={gamma}  phi={phi}  tag={tag}")
            print(f"{'='*72}")

            if results_path.exists() and cuts_out.exists() and not overwrite:
                print("Reusing existing artifacts …")
                records.append(_summarize(results_path, cuts_out, gamma, phi, tag))
                pd.DataFrame(records).sort_values(["phi", "gamma"]).to_csv(summary_csv, index=False)
                continue

            try:
                run_model(
                    beta=cfg.BETA,
                    gamma=gamma,
                    fixed_cost=0.0,
                    c_min=cfg.C_MIN,
                    c_max=cfg.C_MAX,
                    power_tiers=cfg.POWER_TIERS,
                    cuts_in_path=str(cuts_path),
                    tag=tag,
                    variable_sr=True,
                    phi=phi,
                    outputs_dir=OUTPUT_DIR,
                )
                records.append(_summarize(results_path, cuts_out, gamma, phi, tag))
                last = records[-1]
                print(f"  -> stations={last['n_stations']}  chargers={last['n_chargers']}  "
                      f"total_kw={last['total_kw']:,}  gap={last['gap_pct']:.3f}%")
            except Exception as exc:
                print(f"  ERROR: {exc}")
                records.append({
                    "gamma": gamma, "phi": phi, "tag": tag,
                    "results_csv": str(results_path),
                    "cuts_json":   str(cuts_out),
                    "error": str(exc),
                })

            pd.DataFrame(records).sort_values(["phi", "gamma"]).to_csv(summary_csv, index=False)

    summary_df = pd.DataFrame(records).sort_values(["phi", "gamma"])
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSweep complete → {summary_csv}")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D sweep over gamma and phi")
    parser.add_argument("--gammas", type=str,
                        default=",".join(str(g) for g in DEFAULT_GAMMAS))
    parser.add_argument("--phis",   type=str,
                        default=",".join(str(p) for p in DEFAULT_PHIS))
    parser.add_argument("--cuts-path", type=str, default=str(DEFAULT_CUTS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_sweep(
        gammas=_parse_float_list(args.gammas),
        phis=_parse_float_list(args.phis),
        cuts_path=args.cuts_path,
        summary_csv=SUMMARY_CSV,
        overwrite=args.overwrite,
    )
