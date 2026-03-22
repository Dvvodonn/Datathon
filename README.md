# Spain Traffic Flow Pipeline

Collects real-time traffic data (speed + flow) from DGT's DATEX2 API, matches
sensors to interurban road segments, stores hourly readings in SQLite, and
computes seasonal averages.

## Data Sources

| Endpoint | Content | Update freq |
|---|---|---|
| `infocar.dgt.es/datex2/dgt/MeasuredDataPublication/detectores/content.xml` | Live speed + flow for ~5,500 loop detectors | ~1 min |
| `infocar.dgt.es/datex2/dgt/PredefinedLocationsPublication/detectores/content.xml` | Detector locations (lat/lon, road name) | Static |
| `infocar.dgt.es/datex2/dgt/PredefinedLocationsPublication/tramos_invive/content.xml` | INVIVE speed-enforcement zone segments | ~4 months |

All feeds are DATEX2 v1.0 XML (namespace `http://datex2.eu/schema/1_0/1_0`), free
to use under CC-BY. Coverage excludes Basque Country and Catalonia (separate authorities).

## Dependencies

```bash
source venv/bin/activate
# Required packages (all present in project venv):
# geopandas, requests, lxml, sqlite3, shapely, pandas, fiona, pyproj
```

## Scripts — Run Order

### Step 1 — Explore the GeoPackage
```bash
python explore_gpkg.py
# or with a custom path:
python explore_gpkg.py path/to/roads.gpkg
```
Prints layers, CRS, column names, road classification values, and bounding box.
Falls back to `data/processed/road_routes/spain_primary_interurban_arteries.gpkg`
if `roads.gpkg` is not found in the current directory.

### Step 2 — Fetch DGT data
```bash
python fetch_dgt.py
```
Downloads detector locations + live measurements, merges by GUID, saves:
- `data/raw/dgt_sample.json` — 100 sample records for schema inspection
- `data/raw/dgt_detectors_full.json` — all 7,257 detector records with geometry

### Step 3 — Spatial join
```bash
python spatial_join.py [--buffer 50]
```
Matches DGT detector points to road segments within 50m buffer (UTM zone 30N
for metric accuracy). Filters roads to interurban classes:
- `clase 1001`: Autopista de peaje
- `clase 1002`: Autopista libre / autovía
- `clase 1003`: Carretera convencional — first order only (`orden = '1'`)
- `clase 1005`: Carretera multicarril

Output: `data/processed/matched_segments.gpkg` (~6,500 matched detectors).

If `roads.gpkg` is not present, re-runs with the processed arteries file.
Re-run this step whenever `roads.gpkg` changes.

### Step 4 — Initialise database
```bash
python init_db.py [--db traffic.db]
```
Creates `traffic.db` with `daily_readings` and `seasonal_averages` tables.
Safe to re-run (uses `CREATE TABLE IF NOT EXISTS`).

### Step 5 — Daily collection
```bash
python collect_daily.py [--date 2026-03-15] [--dry-run]
```
Fetches the current measurement snapshot, filters to matched interurban segments,
and upserts into `daily_readings`. The DGT feed is a snapshot (not historical),
so running once per hour gives one reading per detector per hour.

Logs to `logs/collection.log`.

### Step 6 — Seasonal aggregation
```bash
python compute_seasons.py [--year 2026] [--season DJF]
```
Reads `daily_readings`, assigns seasons, computes per-segment averages, writes
to `seasonal_averages` table, and exports `data/processed/seasonal_averages.gpkg`
with one layer per season (`seasonal_djf`, `seasonal_mam`, etc.).

## Database Schema

### `daily_readings`
| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `dgt_segment_id` | TEXT | Detector GUID (e.g. `GUID_DET_138083`) |
| `road_id` | TEXT | Matched road `id_tramo` from GeoPackage |
| `date` | TEXT | ISO date `YYYY-MM-DD` |
| `hour` | INTEGER | 0–23 UTC hour |
| `speed_kph` | REAL | Average vehicle speed |
| `flow_veh_per_hour` | REAL | Vehicle flow rate |
| `data_quality` | REAL | Quality indicator (null if not provided) |
| `source` | TEXT | Source endpoint name |
| `fetched_at` | TEXT | Insert timestamp |

Unique index on `(dgt_segment_id, date, hour)` — duplicate fetches are replaced.

### `seasonal_averages`
| Column | Type | Description |
|---|---|---|
| `road_id` | TEXT | Road segment ID |
| `dgt_segment_id` | TEXT | Detector GUID |
| `season` | TEXT | `DJF`, `MAM`, `JJA`, or `SON` |
| `year` | INTEGER | Calendar year (DJF year = year of Jan/Feb) |
| `avg_speed_kph` | REAL | Mean speed across season |
| `avg_flow_veh_per_hour` | REAL | Mean hourly flow across season |
| `peak_hour_flow` | REAL | Max per-hour average flow within season |
| `sample_count` | INTEGER | Number of hourly readings contributing |

## Cron Setup

Add to crontab (`crontab -e`):

```cron
# Collect DGT traffic snapshot daily at 06:00 local time
0 6 * * * /Users/daveedvodonenko/Desktop/Datathon/venv/bin/python /Users/daveedvodonenko/Desktop/Datathon/collect_daily.py >> /Users/daveedvodonenko/Desktop/Datathon/logs/cron.log 2>&1

# Optionally collect hourly for finer time resolution
0 * * * * /Users/daveedvodonenko/Desktop/Datathon/venv/bin/python /Users/daveedvodonenko/Desktop/Datathon/collect_daily.py >> /Users/daveedvodonenko/Desktop/Datathon/logs/cron.log 2>&1

# Recompute seasonal averages weekly on Sunday at 07:00
0 7 * * 0 /Users/daveedvodonenko/Desktop/Datathon/venv/bin/python /Users/daveedvodonenko/Desktop/Datathon/compute_seasons.py >> /Users/daveedvodonenko/Desktop/Datathon/logs/cron.log 2>&1
```

> **Note**: The DGT feed provides one snapshot per fetch (not a 24-hour history).
> Running hourly gives hourly resolution; once daily gives a single daily reading
> per detector. For true daily profiles, collect hourly.

## Re-running the Spatial Join

If `roads.gpkg` changes (new road data, different filtering):

```bash
python spatial_join.py --roads new_roads.gpkg --buffer 50
# Then re-run collection so road_id mapping is updated:
python collect_daily.py
```

## Seasonal Definitions

| Season | Months | Year assignment |
|---|---|---|
| DJF | Dec, Jan, Feb | Year of Jan/Feb (Dec belongs to following year's DJF) |
| MAM | Mar, Apr, May | Calendar year |
| JJA | Jun, Jul, Aug | Calendar year |
| SON | Sep, Oct, Nov | Calendar year |
