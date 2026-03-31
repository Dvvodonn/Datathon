# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spanish EV charging network optimization project. Determines optimal placement of fast-charging stations across Spain by combining OSM road network analysis, combinatorial optimization (Benders decomposition), and real-time DGT traffic data. There is also a secondary pipeline for collecting live highway traffic metrics.

## Environment Setup

```bash
source venv/bin/activate
# requirements.txt is present; install with:
pip install -r requirements.txt
```

Key packages: `ortools`, `gurobipy`, `geopandas`, `igraph`, `networkx`, `shapely`, `osmnx`, `contextily`, `matplotlib`, `duckdb`, `pdfplumber`.

## Common Commands

### EV Charging Network Pipeline (run in order)

```bash
# 1. Download raw data (DGT chargers, Ministry road GPKGs, Endesa CSVs, GitHub repo)
python data_download.py

# 2. Inspect road schema, filter to primary arteries
python inspect_road_columns.py --input <path_to_rt_viaria.gpkg> --describe --save-filtered

# 3. Build road network graph, compute centrality metrics
python road_network_analysis.py --save-dataset

# 4. Build fastest-path edges (Dijkstra, 500 km limit)
python scripts/processing/build_fastest_edges.py

# 5. Filter edges and process feasible charger locations
python scripts/processing/filter_edges_250.py
python scripts/processing/process_feasible_locations.py

# 6. Run Benders decomposition optimizer
python models/model_1.py

# 7. Sweep alpha parameter for Pareto frontier (reuses precomputed cuts)
python models/efficient_frontier.py

# 8. Close remaining optimality gap with 1-opt local search
python models/local_search.py

# Visualize results
python visualize_routes.py --show
python visualizations/plot_solution.py
```

### Traffic Flow Pipeline (DGT DATEX2)

```bash
python explore_gpkg.py                          # inspect GeoPackage layers
python fetch_dgt.py                             # download 7,257 detector records
python spatial_join.py [--buffer 50]            # match detectors to road segments
python init_db.py [--db traffic.db]             # create SQLite schema (safe to re-run)
python collect_daily.py [--date YYYY-MM-DD] [--dry-run]  # fetch snapshot
python compute_seasons.py [--year 2026] [--season DJF]   # aggregate to seasonal averages
```

## Architecture

### Two Distinct Pipelines

**A. EV Charging Network Optimization** (`scripts/`, `models/`)

1. **`data_download.py`** — Acquires raw data: DGT EV charger XML (DATEX2 API), Ministry road geometries (scraped GPKGs), Endesa/i-DE grid CSVs, GitHub open-data repo. Outputs to `data/raw/` and `data/meta/`.

2. **`inspect_road_columns.py`** — Fuzzy-detects type/name/order columns in the Ministry GPKG, filters to primary interurban arteries → `data/processed/road_routes/spain_primary_interurban_arteries.gpkg`.

3. **`scripts/processing/build_fastest_edges.py`** — Runs parallel Dijkstra over the OSM graph (Geofabrik Spain PBF), snapping virtual city/station nodes within 10 km to the road network. Outputs fastest-path edges capped at 500 km.

4. **`scripts/processing/process_feasible_locations.py`** — Intersects 12,215 gas station locations with high-population municipalities, producing ~1,200 feasible charger candidate sites.

5. **`models/model_1.py`** — Core Benders decomposition:
   - *Master* (OR-Tools CP-SAT): chooses binary charger placements `x[k]`
   - *Subproblems* (parallel Dijkstra): one per city pair (~55k pairs), returns feasibility/optimality cuts
   - Converges when lower bound = upper bound; cuts persisted to `models/benders_cuts.json` (21 MB) for reuse

6. **`models/efficient_frontier.py`** — Sweeps the cost parameter `α` (cost per charger) over the precomputed cuts to trace the Pareto frontier of {n_chargers, total_travel_time}. Results in `models/efficient_frontier.csv`.

7. **`models/local_search.py`** — 1-opt heuristic to close any remaining Benders optimality gap. Evaluates candidates in parallel; checkpoints improvements to `local_search_checkpoint.json` for fault tolerance.

**B. Traffic Flow Monitoring** (root-level scripts)

Fetches ~1-minute DGT DATEX2 snapshots, spatially joins to road segments (UTM 30N, 50 m buffer), stores hourly readings in SQLite (`traffic.db`), and aggregates to seasonal averages (DJF/MAM/JJA/SON). See README.md for the full database schema and cron setup.

### Key Data Directories

| Path | Contents |
|---|---|
| `data/raw/` | Downloaded sources (DGT XML, Ministry GPKG, grid CSVs) |
| `data/processed/` | Filtered/derived datasets and matched segments |
| `data/meta/` | Scraped link inventories (JSON), HTML pages, warnings |
| `models/` | Optimization scripts + precomputed artifacts |
| `scripts/acquisition/` | Data download & preprocessing |
| `scripts/processing/` | Graph building, edge computation, location filtering |
| `scripts/visualizations/` | Map and flow plots |
| `visualizations/` | Output PNGs and the `plot_solution.py` script |

### Projections

- Spatial operations use **EPSG:25830** (UTM zone 30N) for metric accuracy
- Web display uses **EPSG:3857** (Web Mercator)
- Spain spans UTM zones 29–31; zone 30N is the standard choice for national analysis

### Tunable Constants

`models/model_1.py` and `models/efficient_frontier.py` expose `α` (cost per charger, default 100) at the top of each file. `road_network_analysis.py` exposes `BETWEENNESS_THRESHOLD`, `DEGREE_THRESHOLD`, `BETWEENNESS_SAMPLES`, and `SNAP_TOLERANCE_M`.
