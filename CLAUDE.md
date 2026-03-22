# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spanish EV charging network optimization/planning project. Uses road network analysis, EV charger data, and power grid capacity data to optimize placement of charging infrastructure across Spain.

## Environment Setup

```bash
source venv/bin/activate
```

No `requirements.txt` exists — dependencies are managed via the venv. Key packages: geopandas, igraph, networkx, contextily, shapely, matplotlib, requests, beautifulsoup4.

## Common Commands

```bash
# Download all data (DGT chargers, Ministry road routes, grid capacity, GitHub repo)
python data_download.py

# Inspect road schema and filter to major roads (autopistas, autovías, N-/A-/AP- routes)
python inspect_road_columns.py --input <path_to_rt_viaria.gpkg> --describe --save-filtered

# Analyze road network centrality (betweenness + degree), outputs road_network_spain.png
python road_network_analysis.py --save-dataset

# Visualize filtered roads with EV charger overlay
python visualize_routes.py --show
python data/raw/visualize_with_chargers.py --roads data/processed/road_routes/spain_primary_interurban_arteries.gpkg --chargers-csv data/raw/chargers/sites.csv --show
```

## Architecture

**Linear data pipeline:**

1. **`data_download.py`** — Acquires all raw data: DGT EV charger XML (DATEX2 API), Ministry road geometries (scraped), Endesa/i-DE grid capacity CSVs (scraped), GitHub open-data repo (shallow clone). Outputs to `data/raw/` and `data/meta/`.

2. **`inspect_road_columns.py`** — Reads raw Ministry GPKG, fuzzy-detects type/name/order columns, filters to primary interurban arteries. Saves to `data/processed/road_routes/spain_primary_interurban_arteries.gpkg`.

3. **`road_network_analysis.py`** — Builds directed igraph from road segments, snaps near-coincident endpoints (11m tolerance), computes betweenness and degree centrality, filters by thresholds, plots nodes on CartoDB basemap (color = betweenness, size = degree). Key constants at top of file: `BETWEENNESS_THRESHOLD`, `DEGREE_THRESHOLD`, `BETWEENNESS_SAMPLES`, `SNAP_TOLERANCE_M`.

4. **`visualize_routes.py` / `visualize_with_chargers.py`** — Load filtered roads + optional EV charger CSV overlay, output high-DPI PNG.

## Data

- `data/raw/` — Downloaded source files (DGT XML, Ministry GPKG, grid CSVs, cloned repo)
- `data/processed/` — Filtered/derived datasets
- `data/meta/` — Scraped link inventories (JSON), downloaded HTML pages, warning logs
- `ev_charging_sources.csv` — Bibliography of 20 academic papers on EV charger location optimization

## Key Outputs

- `road_network_spain.png` — Network visualization
- `road_network_dataset.csv` — Node/edge statistics (when `--save-dataset` passed)
- `data/processed/road_routes/spain_primary_interurban_arteries.gpkg` — Filtered road network
