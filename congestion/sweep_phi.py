"""
congestion/sweep_phi.py
=======================
Sweep the corridor-coverage model over phi values (0 → 1).

gamma is fixed at 0.10 (per-kW grid cost).
phi controls what fraction of each Iberdrola corridor's 100 km gaps must be
covered.  At phi=1 every coverable gap on all 9 corridors is filled.

EV penetration: 2.5% (cfg.EV_PENETRATION = 0.025)

Outputs go to datathon_master/corridor_sweep/.

Run
---
    python congestion/sweep_phi.py
    python congestion/sweep_phi.py --overwrite
    python congestion/sweep_phi.py --phis 0.0,0.5,1.0
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

# ── Fixed parameters ──────────────────────────────────────────────────────────
GAMMA      = 0.10               # fixed grid-connection cost per kW
OUTPUT_DIR = Path("datathon_master/corridor_sweep")
SUMMARY_CSV = OUTPUT_DIR / "corridor_phi_sweep.csv"

# Warm-start cuts: use the phi=0, gamma=0.1 run from the previous sweep
DEFAULT_CUTS = Path("datathon_master/congestion_cuts_dm_g0_1_p0.json")
# Fall back to gamma=1.0 cuts if that file is missing
_FALLBACK_CUTS = cfg.OUTPUTS_DIR / "congestion_cuts_g1_0.json"

DEFAULT_PHIS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _parse_float_list(raw):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _phi_tag(phi):
    phi = float(phi)
    if phi == 0.0:
        return "0"
    if phi == 1.0:
        return "1"
    return str(phi).replace(".", "_")


def _safe_gap_pct(ub, lb):
    if ub is None or lb is None or not math.isfinite(ub):
        return float("nan")
    return (ub - lb) / max(ub, 1e-9) * 100.0


def _summarize(results_path, cuts_path, phi, tag):
    results_df = pd.read_csv(results_path)
    built = results_df[results_df["x_built"] == 1].copy()

    n_stations = len(built)
    n_chargers = int(built["c_built"].sum()) if n_stations else 0
    total_kw   = int((built["c_built"] * built["p_built_kw"]).sum()) if n_stations else 0
    total_wq   = float(built["wq_minutes"].sum()) if n_stations else 0.0
    total_w    = 0.0
    if n_stations:
        total_w = float(
            (built["wq_minutes"] + 60.0 * cfg.E_SESSION_KWH / built["p_built_kw"]).sum()
        )

    ub = lb = n_cuts = float("nan")
    if Path(cuts_path).exists():
        with open(cuts_path) as f:
            payload = json.load(f)
        ub     = float(payload.get("upper_bound", float("nan")))
        lb     = float(payload.get("lower_bound", float("nan")))
        n_cuts = len(payload.get("cuts", []))

    return {
        "phi":          phi,
        "gamma":        GAMMA,
        "tag":          tag,
        "results_csv":  str(results_path),
        "cuts_json":    str(cuts_path),
        "n_stations":   n_stations,
        "n_chargers":   n_chargers,
        "avg_chargers_per_station": (n_chargers / n_stations) if n_stations else 0.0,
        "total_kw":          total_kw,
        "total_wq_min":      total_wq,
        "total_w_min":       total_w,
        "objective_ub":      ub,
        "objective_lb":      lb,
        "gap_pct":           _safe_gap_pct(ub, lb),
        "n_cuts":            n_cuts,
        "power_dist": json.dumps(
            {int(k): int(v)
             for k, v in built["p_built_kw"].value_counts().sort_index().items()}
        ) if n_stations else "{}",
        "charger_dist": json.dumps(
            {int(k): int(v)
             for k, v in built["c_built"].value_counts().sort_index().items()}
        ) if n_stations else "{}",
    }


def run_sweep(phis, cuts_path, summary_csv, overwrite=False, master_time_ms=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cuts_path   = Path(cuts_path) if Path(cuts_path).exists() else Path(_FALLBACK_CUTS)
    summary_csv = Path(summary_csv)

    print(f"Corridor phi sweep  |  gamma={GAMMA}  EV={cfg.EV_PENETRATION}")
    print(f"Warm-start cuts : {cuts_path}")
    print(f"Output dir      : {OUTPUT_DIR}\n")

    records  = []
    total    = len(phis)

    for job_idx, phi in enumerate(phis, 1):
        tag          = f"corridor_p{_phi_tag(phi)}"
        results_path = OUTPUT_DIR / f"results_congestion_{tag}.csv"
        cuts_out     = OUTPUT_DIR / f"congestion_cuts_{tag}.json"

        print(f"\n{'='*72}")
        print(f"[{job_idx:02d}/{total:02d}]  phi={phi}  tag={tag}")
        print(f"{'='*72}")

        if results_path.exists() and cuts_out.exists() and not overwrite:
            print("Reusing existing artifacts …")
            records.append(_summarize(results_path, cuts_out, phi, tag))
            pd.DataFrame(records).sort_values("phi").to_csv(summary_csv, index=False)
            continue

        # Each phi run warm-starts from the previous phi's cuts (richer cuts)
        effective_cuts = cuts_out if cuts_out.exists() else cuts_path

        try:
            run_model(
                beta=cfg.BETA,
                gamma=GAMMA,
                fixed_cost=cfg.FIXED_COST,
                c_min=cfg.C_MIN,
                c_max=cfg.C_MAX,
                power_tiers=cfg.POWER_TIERS,
                cuts_in_path=str(effective_cuts),
                tag=tag,
                variable_sr=True,
                phi=phi,
                outputs_dir=OUTPUT_DIR,
                master_time_limit_ms=master_time_ms,
            )
            records.append(_summarize(results_path, cuts_out, phi, tag))
            last = records[-1]
            print(f"  -> stations={last['n_stations']}  chargers={last['n_chargers']}  "
                  f"total_kw={last['total_kw']:,}  gap={last['gap_pct']:.3f}%")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            records.append({
                "phi": phi, "gamma": GAMMA, "tag": tag,
                "results_csv": str(results_path),
                "cuts_json":   str(cuts_out),
                "error": str(exc),
            })

        pd.DataFrame(records).sort_values("phi").to_csv(summary_csv, index=False)

    summary_df = pd.DataFrame(records).sort_values("phi")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSweep complete → {summary_csv}")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phi sweep over corridor coverage (gamma fixed at 0.10)"
    )
    parser.add_argument("--phis", type=str,
                        default=",".join(str(p) for p in DEFAULT_PHIS))
    parser.add_argument("--cuts-path", type=str, default=str(DEFAULT_CUTS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--master-time-ms", type=int, default=None,
                        help="Master MIP time limit in ms per iteration")
    args = parser.parse_args()

    run_sweep(
        phis=_parse_float_list(args.phis),
        cuts_path=args.cuts_path,
        summary_csv=SUMMARY_CSV,
        overwrite=args.overwrite,
        master_time_ms=args.master_time_ms,
    )
