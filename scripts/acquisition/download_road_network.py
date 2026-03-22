"""
Download Spain interurban drive network from Geofabrik OSM extract.

Pipeline:
  1. Download spain-latest.osm.pbf from Geofabrik (~1.4GB, current as of download date)
  2. Filter to drive-relevant highways with osmium-tool (42MB filtered PBF)
  3. Convert filtered PBF → OSM XML (584MB)
  4. Load via osmnx, add speeds/travel-times, save GraphML + GeoPackages

Outputs → data/raw/road_routes/:
  spain-latest.osm.pbf              raw Spain OSM extract (~1.4GB, cached)
  spain_interurban_filtered.osm.pbf filtered to drive highways (42MB)
  spain_interurban_filtered.osm     OSM XML version of above (584MB)
  spain_interurban.graphml          NetworkX DiGraph with speed/time attrs
  spain_interurban_nodes.gpkg       node points (osmid, x, y, geometry)
  spain_interurban_edges.gpkg       edges: length_m, speed_kph, travel_time_s,
                                    name, highway, oneway, osmid

Edge attributes for routing:
  length_m        float  metres (exact along geometry)
  speed_kph       float  OSM maxspeed tags → highway-type defaults for missing
  travel_time_s   float  seconds = length_m / (speed_kph/3.6)
  oneway          bool
  name            str    e.g. "Autovía del Mediterráneo"
  highway         str    motorway | trunk | primary | secondary | *_link

Requirements:
  osmium-tool  (brew install osmium-tool)  → already at /opt/homebrew/bin/osmium
  osmnx        (pip install osmnx)

Runtime: ~1h first run (dominated by OSM XML parse); subsequent runs skip
existing files automatically.

Usage:
  source venv/bin/activate
  python download_road_network.py
"""

import subprocess
import time
from pathlib import Path

import requests

OUT_DIR = Path("data/raw/road_routes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PBF_URL        = "https://download.geofabrik.de/europe/spain-latest.osm.pbf"
RAW_PBF        = OUT_DIR / "spain-latest.osm.pbf"
FILTERED_PBF   = OUT_DIR / "spain_interurban_filtered.osm.pbf"
FILTERED_OSM   = OUT_DIR / "spain_interurban_filtered.osm"
GRAPHML_PATH   = OUT_DIR / "spain_interurban.graphml"
NODES_GPKG     = OUT_DIR / "spain_interurban_nodes.gpkg"
EDGES_GPKG     = OUT_DIR / "spain_interurban_edges.gpkg"

OSMIUM_BIN = "/opt/homebrew/bin/osmium"

# Interurban-relevant highway types
HIGHWAY_TYPES = [
    "motorway", "trunk", "primary", "secondary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
]


def download_pbf():
    if RAW_PBF.exists():
        size_mb = RAW_PBF.stat().st_size / 1e6
        print(f"  Cached PBF found ({size_mb:.0f} MB) — skipping download")
        return
    print(f"  Downloading {PBF_URL} …")
    t0 = time.time()
    with requests.get(PBF_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(RAW_PBF, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded/total*100:5.1f}%  {downloaded/1e6:.0f}/{total/1e6:.0f} MB",
                          end="", flush=True)
    print()
    print(f"  Downloaded {RAW_PBF.stat().st_size/1e6:.0f} MB in {time.time()-t0:.0f}s")


def filter_pbf():
    if FILTERED_PBF.exists():
        print(f"  Filtered PBF cached ({FILTERED_PBF.stat().st_size/1e6:.1f} MB) — skipping")
        return
    hw_filter = "w/highway=" + ",".join(HIGHWAY_TYPES)
    print(f"  Running osmium tags-filter …")
    result = subprocess.run(
        [OSMIUM_BIN, "tags-filter", str(RAW_PBF), hw_filter,
         "-o", str(FILTERED_PBF), "--overwrite"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"osmium failed: {result.stderr}")
    print(f"  Filtered PBF: {FILTERED_PBF.stat().st_size/1e6:.1f} MB")


def convert_to_xml():
    if FILTERED_OSM.exists():
        print(f"  OSM XML cached ({FILTERED_OSM.stat().st_size/1e6:.0f} MB) — skipping")
        return
    print(f"  Converting PBF → OSM XML …")
    result = subprocess.run(
        [OSMIUM_BIN, "cat", str(FILTERED_PBF), "-o", str(FILTERED_OSM), "--overwrite"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"osmium cat failed: {result.stderr}")
    print(f"  OSM XML: {FILTERED_OSM.stat().st_size/1e6:.0f} MB")


def build_graph():
    if GRAPHML_PATH.exists() and NODES_GPKG.exists() and EDGES_GPKG.exists():
        print(f"  GraphML and GeoPackages cached — skipping graph build")
        return

    import osmnx as ox
    import numpy as np

    ox.settings.log_console = False

    print(f"  Loading OSM XML into osmnx (this takes ~50 min first run) …")
    t0 = time.time()
    G = ox.graph_from_xml(str(FILTERED_OSM), retain_all=False)
    print(f"  Loaded {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"in {time.time()-t0:.0f}s")

    print("  Adding speed / travel time …")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    print(f"  Saving GraphML → {GRAPHML_PATH}")
    ox.save_graphml(G, filepath=str(GRAPHML_PATH))

    print("  Building GeoDataFrames …")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    edges_gdf = edges_gdf.rename(columns={"length": "length_m",
                                           "travel_time": "travel_time_s"})
    for col in edges_gdf.columns:
        if edges_gdf[col].apply(lambda v: isinstance(v, list)).any():
            edges_gdf[col] = edges_gdf[col].apply(
                lambda v: "|".join(str(x) for x in v) if isinstance(v, list) else v
            )

    print(f"  Saving nodes → {NODES_GPKG}")
    nodes_gdf.to_file(str(NODES_GPKG), driver="GPKG", layer="nodes")
    print(f"  Saving edges → {EDGES_GPKG}")
    edges_gdf.to_file(str(EDGES_GPKG), driver="GPKG", layer="edges")

    lens = edges_gdf["length_m"].astype(float)
    spds = edges_gdf["speed_kph"].astype(float)
    print()
    print("=== Graph Summary ===")
    print(f"  Nodes : {len(nodes_gdf):>8,}")
    print(f"  Edges : {len(edges_gdf):>8,}")
    print(f"  Length  min/med/max: {lens.min():.0f} / {lens.median():.0f} / {lens.max():.0f} m")
    print(f"  Speed   min/med/max: {spds.min():.0f} / {spds.median():.0f} / {spds.max():.0f} km/h")
    print()
    for p in [RAW_PBF, FILTERED_PBF, GRAPHML_PATH, NODES_GPKG, EDGES_GPKG]:
        if p.exists():
            print(f"  {p}  ({p.stat().st_size/1e6:.1f} MB)")


def main():
    print("=== Spain Interurban Road Network Download ===")
    print()
    print("Step 1/4  Download Spain OSM PBF from Geofabrik …")
    download_pbf()
    print()
    print("Step 2/4  Filter to interurban highways (osmium) …")
    filter_pbf()
    print()
    print("Step 3/4  Convert filtered PBF → OSM XML …")
    convert_to_xml()
    print()
    print("Step 4/4  Build NetworkX routing graph …")
    build_graph()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
