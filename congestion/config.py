"""
congestion/config.py — All tunable parameters for the congestion pipeline.

Every other module in this package imports constants from here.
To run a sensitivity sweep, pass overrides directly to the top-level
functions (demand.build_demand, queue.precompute_wq_table, model.main)
rather than editing this file.
"""

from pathlib import Path

# ── EV demand ─────────────────────────────────────────────────────────────────
EV_PENETRATION   = 0.05    # fraction of highway traffic assumed to be EV
                            # if the input AADT data contains a per-segment
                            # column named 'ev_penetration', that overrides this
PEAK_HOUR_FACTOR = 0.10    # share of daily AADT that falls in the single
                            # busiest hour (standard value for Spanish highways)
STOP_RATE        = 0.05    # fraction of passing EVs that stop to charge

# ── Charger power tiers (new stations only) ────────────────────────────────────
POWER_TIERS      = [50, 100, 150, 200, 250, 350]  # kW options for new station builds
GAMMA            = 1.0              # cost per kW of total station power (minutes-equivalent)
                                    # station grid-connection cost = GAMMA * c * p_kW
                                    # calibrated so 1 kW ≈ 1 min-equivalent;
                                    # sweep via --gamma flag for Pareto frontier
E_SESSION_KWH    = 42.0             # average energy per charging stop (kWh)
                                    # 60% charge on 70 kWh battery
                                    # μ(p) = p_kW / E_SESSION_KWH  (veh/hr)

# ── Spatial assignment ────────────────────────────────────────────────────────
TIER1_MAX_KM     = 10.0    # use actual IMD measurement if nearest segment
                            # is within this distance
TIER2_MAX_KM     = 20.0    # between TIER1 and TIER2: class-mean imputation,
                            # marked tier=2; beyond TIER2: same formula but
                            # marked tier=3 to flag higher uncertainty

# ── M/M/c queuing ─────────────────────────────────────────────────────────────
MU_PER_CHARGER   = 2.0     # service rate: vehicles per charger per hour
                            # 2.0 ≈ 30-min session at ~150 kW
                            # set to 4.0 for 350 kW, 1.33 for 50 kW
C_MIN            = 1       # minimum chargers at an opened station
C_MAX            = 10      # maximum chargers at an opened station
                            # (decision variable upper bound)
WQ_LARGE_PENALTY = 480.0   # waiting time (min) when queue is unstable (ρ ≥ 1)
                            # 480 = 8-hour hard penalty; keeps solver away from
                            # solutions that saturate a station

# ── Optimisation ──────────────────────────────────────────────────────────────
ALPHA            = 100     # cost per charger built (minutes equivalent)
                            # must equal ALPHA in models/model_1.py for
                            # comparable results across the two models
BETA             = 1.0     # weight of congestion waiting penalty relative to
                            # travel time (both in minutes, so 1.0 is neutral)
                            # sweep this via model.run_efficient_frontier()

PENALTY_INFEASIBLE = 5_000 # virtual travel time (min) for disconnected pairs
N_CITIES         = 235     # city clusters — must match models/model_1.py
MAX_ITER         = 50      # Benders iteration cap
CONV_TOL         = 1e-2    # stop when (UB − LB) / UB < 0.01 %

# ── File paths (relative to project root) ─────────────────────────────────────
IMD_GEOJSON      = "data_main/traffic/imd_total_por_tramo.geojson"
ANNUAL_FLOW_GPKG = "annual_flow_2024.gpkg"   # 847 detector points; future use
EDGES_GPKG       = "data/raw/road_network/spain_interurban_edges.gpkg"
ROAD_FLOW_CSV    = "data_main/road_edges_flow.csv"  # calibrated AADT for all OSM edges
NODES_CSV        = "data_main/nodes.csv"
EDGES_250_CSV    = "data_main/edges_250.csv"
BENDERS_CUTS     = "models/benders_cuts.json"
OUTPUTS_DIR      = Path("congestion/outputs")
