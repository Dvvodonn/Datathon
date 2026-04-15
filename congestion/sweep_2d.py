"""
congestion/sweep_2d.py
======================
Sweep the congestion model over a 2D grid of:

    gamma      : grid-connection cost per installed kW
    fixed_cost : station opening cost per selected site

Each solve warm-starts from a shared Benders cut file because cuts depend only
on x[k] and not on (gamma, fixed_cost, c, p). Results are written per scenario
plus a summary CSV that drives the interactive map.

Run
---
    python congestion/sweep_2d.py
    python congestion/sweep_2d.py --overwrite
    python congestion/sweep_2d.py --gammas 0.05,0.10,0.20 --fixed-costs 0,15000,60000
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


DEFAULT_GAMMAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
DEFAULT_FIXED_COSTS = [0, 5_000, 15_000, 30_000, 60_000, 120_000, 250_000]
DEFAULT_CUTS_PATH = cfg.OUTPUTS_DIR / "congestion_cuts_g10_0.json"
DEFAULT_SUMMARY_CSV = cfg.OUTPUTS_DIR / "gamma_fixed_cost_sweep.csv"


def _parse_float_list(raw, cast=float):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def _tag_token(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "_")


def _scenario_tag(tag_prefix, gamma, fixed_cost):
    return f"{tag_prefix}_g{_tag_token(gamma)}_f{_tag_token(fixed_cost)}"


def _safe_gap_pct(ub, lb):
    if ub is None or lb is None or not math.isfinite(ub):
        return float("nan")
    return (ub - lb) / max(ub, 1e-9) * 100.0


def _summarize_artifacts(results_path, cuts_path, gamma, fixed_cost, tag):
    results_df = pd.read_csv(results_path)
    built = results_df[results_df["x_built"] == 1].copy()

    n_stations = len(built)
    n_chargers = int(built["c_built"].sum()) if n_stations else 0
    total_kw = int((built["c_built"] * built["p_built_kw"]).sum()) if n_stations else 0
    total_wq = float(built["wq_minutes"].sum()) if n_stations else 0.0
    total_w = 0.0
    if n_stations:
        total_w = float(
            (built["wq_minutes"] + 60.0 * cfg.E_SESSION_KWH / built["p_built_kw"]).sum()
        )

    ub = float("nan")
    lb = float("nan")
    n_cuts = float("nan")
    if cuts_path.exists():
        with open(cuts_path) as f:
            cuts_payload = json.load(f)
        ub = float(cuts_payload.get("upper_bound", float("nan")))
        lb = float(cuts_payload.get("lower_bound", float("nan")))
        n_cuts = len(cuts_payload.get("cuts", []))

    return {
        "gamma": gamma,
        "fixed_cost": fixed_cost,
        "tag": tag,
        "results_csv": str(results_path),
        "cuts_json": str(cuts_path),
        "n_stations": n_stations,
        "n_chargers": n_chargers,
        "avg_chargers_per_station": (n_chargers / n_stations) if n_stations else 0.0,
        "total_kw": total_kw,
        "total_wq_min": total_wq,
        "total_w_min": total_w,
        "total_open_cost": n_stations * fixed_cost,
        "objective_ub": ub,
        "objective_lb": lb,
        "gap_pct": _safe_gap_pct(ub, lb),
        "n_cuts": n_cuts,
        "power_dist": json.dumps(
            {int(k): int(v) for k, v in built["p_built_kw"].value_counts().sort_index().items()}
        ),
        "charger_dist": json.dumps(
            {int(k): int(v) for k, v in built["c_built"].value_counts().sort_index().items()}
        ),
    }


def run_sweep(
    gammas,
    fixed_costs,
    cuts_path,
    summary_csv,
    tag_prefix,
    overwrite=False,
    variable_sr=True,
    c_max=cfg.C_MAX,
):
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    cuts_path = Path(cuts_path)
    summary_csv = Path(summary_csv)

    records = []
    total_jobs = len(gammas) * len(fixed_costs)
    job_idx = 0

    for fixed_cost in fixed_costs:
        for gamma in gammas:
            job_idx += 1
            tag = _scenario_tag(tag_prefix, gamma, fixed_cost)
            results_path = cfg.OUTPUTS_DIR / f"results_congestion_{tag}.csv"
            cuts_out_path = cfg.OUTPUTS_DIR / f"congestion_cuts_{tag}.json"

            print(f"\n{'=' * 72}")
            print(f"[{job_idx:02d}/{total_jobs:02d}] gamma={gamma}  fixed_cost={fixed_cost}  tag={tag}")
            print(f"{'=' * 72}")

            if results_path.exists() and cuts_out_path.exists() and not overwrite:
                print("Reusing existing artifacts …")
                records.append(_summarize_artifacts(results_path, cuts_out_path, gamma, fixed_cost, tag))
                pd.DataFrame(records).sort_values(["fixed_cost", "gamma"]).to_csv(summary_csv, index=False)
                continue

            try:
                run_model(
                    beta=cfg.BETA,
                    gamma=gamma,
                    fixed_cost=fixed_cost,
                    c_min=cfg.C_MIN,
                    c_max=c_max,
                    cuts_in_path=str(cuts_path),
                    tag=tag,
                    variable_sr=variable_sr,
                )
                records.append(_summarize_artifacts(results_path, cuts_out_path, gamma, fixed_cost, tag))
                last = records[-1]
                print(f"  -> stations={last['n_stations']}  chargers={last['n_chargers']}  "
                      f"total_kw={last['total_kw']:,}  W={last['total_w_min']:.1f}  "
                      f"gap={last['gap_pct']:.3f}%")
            except Exception as exc:
                print(f"  ERROR: {exc}")
                records.append({
                    "gamma": gamma,
                    "fixed_cost": fixed_cost,
                    "tag": tag,
                    "results_csv": str(results_path),
                    "cuts_json": str(cuts_out_path),
                    "error": str(exc),
                })

            pd.DataFrame(records).sort_values(["fixed_cost", "gamma"]).to_csv(summary_csv, index=False)
            print(f"Partial summary saved -> {summary_csv}")

    summary_df = pd.DataFrame(records).sort_values(["fixed_cost", "gamma"])
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSweep summary saved -> {summary_csv}")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D sweep over gamma and fixed station cost")
    parser.add_argument("--gammas", type=str,
                        default=",".join(str(x) for x in DEFAULT_GAMMAS),
                        help="Comma-separated gamma values")
    parser.add_argument("--fixed-costs", type=str,
                        default=",".join(str(x) for x in DEFAULT_FIXED_COSTS),
                        help="Comma-separated fixed opening costs")
    parser.add_argument("--cuts-path", type=str, default=str(DEFAULT_CUTS_PATH),
                        help="Warm-start Benders cuts JSON")
    parser.add_argument("--summary-csv", type=str, default=str(DEFAULT_SUMMARY_CSV),
                        help="Output summary CSV path")
    parser.add_argument("--tag-prefix", type=str, default="sweep2d",
                        help="Prefix used for per-scenario output tags")
    parser.add_argument("--c-max", type=int, default=cfg.C_MAX,
                        help="Maximum chargers per station")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run scenarios even if artifacts already exist")
    parser.add_argument("--fixed-sr", action="store_true",
                        help="Use flat stop rate instead of corridor-aware variable SR")
    args = parser.parse_args()

    run_sweep(
        gammas=_parse_float_list(args.gammas, float),
        fixed_costs=_parse_float_list(args.fixed_costs, float),
        cuts_path=args.cuts_path,
        summary_csv=args.summary_csv,
        tag_prefix=args.tag_prefix,
        overwrite=args.overwrite,
        variable_sr=not args.fixed_sr,
        c_max=args.c_max,
    )
