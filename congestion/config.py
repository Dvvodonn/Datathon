"""
congestion/config.py — All tunable parameters for the congestion pipeline.

Every other module in this package imports constants from here.
To run a sensitivity sweep, pass overrides directly to the top-level
functions (demand.build_demand, queue.precompute_wq_table, model.main)
rather than editing this file.
"""

from pathlib import Path

# ── EV demand ─────────────────────────────────────────────────────────────────
EV_PENETRATION   = 0.025   # fraction of highway traffic assumed to be EV
                            # if the input AADT data contains a per-segment
                            # column named 'ev_penetration', that overrides this
PEAK_HOUR_FACTOR = 0.13    # share of daily AADT that falls in the single
                            # busiest hour; 0.13 ≈ busy-weekend design standard
                            # for Spanish interurban highways (was 0.10)
STOP_RATE        = 0.05    # flat stop rate — used only for existing charger demand

# ── Variable stop rate (corridor-aware, logistic of through-gap) ───────────────
STOP_RATE_MIN    = 0.02    # opportunistic charging floor
STOP_RATE_MAX    = 0.92    # near-forced stop ceiling
D50_KM           = 125.0   # through-gap (km) at which 50% of EVs stop
                            # = MAX_EDGE_KM / 2 = 250 / 2
STOP_STEEPNESS   = 40.0    # logistic steepness (km); saturates at ~250 km gap
                            # = MAX_EDGE_KM / 6

# ── Charger power tiers (new stations only) ────────────────────────────────────
POWER_TIERS      = [150]                           # kW options for new station builds
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
MU_PER_CHARGER   = 2.0     # legacy fallback only; pipeline uses μ(p)=p/E_SESSION_KWH
                            # kept for wq_minutes() default args and demo mode
C_MIN            = 2       # minimum chargers at an opened station
C_MAX            = 7       # maximum chargers at an opened station
WQ_LARGE_PENALTY = 45.0    # waiting time (min) when queue is unstable (ρ ≥ 1)
                            # 45 min ≈ realistic upper bound; beyond this drivers
                            # divert to another charger or abandon the stop

# ── Optimisation ──────────────────────────────────────────────────────────────
ALPHA            = 100     # legacy baseline-model parameter (not used here)
                            # must equal ALPHA in models/model_1.py for
                            # comparable results across the two models
BETA             = 1.0     # weight of congestion waiting penalty relative to
                            # travel time (both in minutes, so 1.0 is neutral)
                            # sweep this via model.run_efficient_frontier()
FIXED_COST       = 0.0     # station opening cost F·x_k (minutes equivalent)
                            # captures site-prep / civil-work cost independent
                            # of charger count and power tier

PHI              = 0.0     # fraction of corridor gaps that must be covered
                            # 0 → unconstrained; 1 → all 9 corridors fully covered
                            # Applied per-corridor: each corridor must cover
                            # ceil(phi × its own gap count) independently.
PHI_MAX_GAP_KM   = 100.0   # consecutive-stop gap threshold (km); Iberdrola target
PHI_CORRIDOR_BUFFER_KM  = 5.0    # max distance from corridor geometry to assign a node
PHI_MIN_EXISTING_KW    = 150.0  # only existing chargers ≥ this kW count as gap-covering
                                 # stops for the corridor constraint; lower-power AC
                                 # chargers do not satisfy Iberdrola's 150 kW mandate

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
