"""
Road Network Analysis for Spain Primary Interurban Arteries
===========================================================
Builds a graph where nodes = road intersections, edges = road segments.
Computes betweenness centrality (via igraph C backend) and degree, then
plots surviving nodes on a map of Spain with:
  - node size   ~ degree
  - node colour ~ betweenness centrality (white → red gradient)

Global thresholds (top of file) filter out low-importance nodes.
Set BETWEENNESS_SAMPLES to an integer to use approximate (sampled) betweenness
— much faster on large graphs; set to None for exact computation.

Pass --save-dataset (or -s) to export a CSV with full node/edge stats.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from shapely.geometry import Point

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL THRESHOLDS  ← adjust these to filter the network
# ─────────────────────────────────────────────────────────────────────────────
BETWEENNESS_THRESHOLD = 0.0001   # keep nodes with betweenness >= this value
DEGREE_THRESHOLD      = 3        # keep nodes with degree >= this value

# Set to an integer (e.g. 500) to use approximate betweenness via random
# pivot sampling — dramatically faster, slightly less accurate.
# Set to None for exact betweenness centrality.
BETWEENNESS_SAMPLES   = 500
# ─────────────────────────────────────────────────────────────────────────────

GPKG_PATH   = Path("data/processed/road_routes/spain_primary_interurban_arteries.gpkg")
OUTPUT_PLOT = Path("road_network_spain.png")
OUTPUT_CSV  = Path("road_network_dataset.csv")

SNAP_TOLERANCE_M = 11.0   # metres — snaps near-coincident endpoints together


def load_and_project(path: Path) -> gpd.GeoDataFrame:
    """Load GeoPackage and reproject to metric CRS (EPSG:25830, UTM 30N)."""
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs("EPSG:25830")
    gdf = gdf[["nombre", "geometry"]].copy()
    # Strip Z so endpoint tuples compare cleanly
    gdf["geometry"] = gdf["geometry"].apply(
        lambda g: g.__class__([(x, y) for x, y, *_ in g.coords])
    )
    return gdf


def round_coord(coord):
    t = SNAP_TOLERANCE_M
    return (round(coord[0] / t) * t, round(coord[1] / t) * t)


def build_igraph(gdf: gpd.GeoDataFrame):
    """
    Build an igraph Graph (C backend).
    Returns (g, node_coords, node_roads, edge_roads, edge_lengths)
      node_coords : list of (x, y) in EPSG:25830
      node_roads  : list of sets — road names touching each node
      edge_data   : list of dicts with length_m, road_name
    """
    # ── 1. collect all unique snapped endpoints ───────────────────────────────
    node_index: dict = {}   # coord → integer id

    def get_node(coord):
        c = round_coord(coord)
        if c not in node_index:
            node_index[c] = len(node_index)
        return node_index[c], c

    edges_raw = []   # (u_id, v_id, length, road, snapped_u, snapped_v)

    for _, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        uid, uc = get_node(coords[0])
        vid, vc = get_node(coords[-1])
        if uid == vid:
            continue
        length = row.geometry.length
        edges_raw.append((uid, vid, length, row["nombre"], uc, vc))

    n_nodes = len(node_index)
    node_coords = [None] * n_nodes
    node_roads  = [set() for _ in range(n_nodes)]
    for coord, idx in node_index.items():
        node_coords[idx] = coord

    # ── 2. deduplicate edges (keep shortest per node pair; merge road names) ──
    edge_map: dict = {}   # (u,v) → {length, road, roads}
    for uid, vid, length, road, uc, vc in edges_raw:
        key = (min(uid, vid), max(uid, vid))
        node_roads[uid].add(road)
        node_roads[vid].add(road)
        if key not in edge_map:
            edge_map[key] = {"length": length, "road": road, "roads": {road}}
        else:
            if length < edge_map[key]["length"]:
                edge_map[key]["length"] = length
                edge_map[key]["road"]   = road
            edge_map[key]["roads"].add(road)

    # ── 3. build igraph ───────────────────────────────────────────────────────
    edge_list    = list(edge_map.keys())
    edge_lengths = [edge_map[k]["length"] for k in edge_list]
    edge_roads   = ["; ".join(sorted(edge_map[k]["roads"])) for k in edge_list]

    g = ig.Graph(n=n_nodes, edges=edge_list, directed=False)
    g.es["length"] = edge_lengths
    g.es["road"]   = edge_roads

    return g, node_coords, node_roads, edge_roads, edge_lengths


def compute_metrics(g: ig.Graph):
    """
    Compute betweenness centrality (C backend) and degree.
    When BETWEENNESS_SAMPLES is set, passes a random subset of nodes as
    `sources` to igraph — trades a little accuracy for large speed gains.
    Returns (bc_array, deg_array) — numpy arrays indexed by vertex id.
    """
    n = g.vcount()
    print(f"  Computing betweenness centrality on {n:,} nodes "
          f"({'exact' if BETWEENNESS_SAMPLES is None else f'~{BETWEENNESS_SAMPLES} pivot samples'}) …")

    if BETWEENNESS_SAMPLES is None or BETWEENNESS_SAMPLES >= n:
        bc_raw = g.betweenness(weights="length", directed=False)
        scale  = 1.0
    else:
        rng     = np.random.default_rng(42)
        sources = rng.choice(n, size=BETWEENNESS_SAMPLES, replace=False).tolist()
        bc_raw  = g.betweenness(weights="length", directed=False, sources=sources)
        # Re-scale so values are comparable to the full normalised range
        scale   = n / BETWEENNESS_SAMPLES

    # Normalise to [0,1]  (same formula as networkx: / ((n-1)(n-2)/2))
    norm_factor = (n - 1) * (n - 2) / 2 if n > 2 else 1.0
    bc  = np.array(bc_raw) * scale / norm_factor
    deg = np.array(g.degree())
    return bc, deg


def filter_nodes(bc, deg):
    """Return boolean mask of nodes passing both thresholds."""
    return (bc >= BETWEENNESS_THRESHOLD) & (deg >= DEGREE_THRESHOLD)


def nodes_to_wgs84(coords_metric: list) -> list:
    """Batch-convert EPSG:25830 (x,y) list to WGS-84 (lon,lat) list."""
    pts = gpd.GeoSeries(
        [Point(x, y) for x, y in coords_metric], crs="EPSG:25830"
    ).to_crs("EPSG:4326")
    return [(p.x, p.y) for p in pts]


def plot_network(kept_coords, bc_kept, deg_kept, save_path: Path):
    """Plot surviving nodes on a CartoDB basemap of Spain."""
    import contextily as ctx

    lons_lats = nodes_to_wgs84(kept_coords)
    lons = np.array([p[0] for p in lons_lats])
    lats = np.array([p[1] for p in lons_lats])

    bc_norm = (bc_kept - bc_kept.min()) / (bc_kept.max() - bc_kept.min() + 1e-12)
    sizes   = (deg_kept / deg_kept.max()) * 40 + 4

    cmap    = mcolors.LinearSegmentedColormap.from_list("bl_red", ["blue", "red"])
    colours = cmap(bc_norm)

    fig, ax = plt.subplots(figsize=(14, 12))

    # Spain bbox → Web Mercator
    pts_bbox = gpd.GeoSeries.from_wkt(
        ["POINT(-9.5 35.8)", "POINT(4.5 44.1)"], crs="EPSG:4326"
    ).to_crs("EPSG:3857")
    xmin, ymin = pts_bbox[0].x, pts_bbox[0].y
    xmax, ymax = pts_bbox[1].x, pts_bbox[1].y

    node_pts = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(lons, lats)], crs="EPSG:4326"
    ).to_crs("EPSG:3857")
    xs = np.array([p.x for p in node_pts])
    ys = np.array([p.y for p in node_pts])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.Positron, zoom=6)

    ax.scatter(xs, ys, s=sizes, c=colours, edgecolors="none",
               alpha=0.85, zorder=5)

    # Colour bar
    sm = cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(vmin=bc_kept.min(), vmax=bc_kept.max())
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Betweenness Centrality", fontsize=11)

    # Degree legend
    legend_degrees = sorted(set(
        int(np.percentile(deg_kept, p)) for p in [25, 50, 75, 100]
    ))
    legend_handles = [
        matplotlib.lines.Line2D(
            [], [], marker="o", linestyle="None",
            markersize=np.sqrt((d / deg_kept.max()) * 40 + 4),
            color="salmon", label=f"degree {d}"
        )
        for d in legend_degrees
    ]
    ax.legend(handles=legend_handles, title="Node Degree",
              loc="lower right", framealpha=0.8, fontsize=9)

    ax.set_title(
        f"Spain Road Network — Intersection Nodes\n"
        f"(betweenness ≥ {BETWEENNESS_THRESHOLD}, degree ≥ {DEGREE_THRESHOLD})",
        fontsize=13
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"  Plot saved → {save_path}")
    plt.show()


def build_dataset(g, mask, node_coords, node_roads, bc, deg) -> pd.DataFrame:
    """
    Edge-level CSV where both endpoints pass the filter mask.
    Columns: node_a_x/y, node_a_lon/lat, node_b_x/y, node_b_lon/lat,
             length_m, degree_a, degree_b, betweenness_a, betweenness_b, road_name
    """
    kept_ids = set(np.where(mask)[0])

    # Batch lon/lat conversion for kept nodes only
    kept_list   = sorted(kept_ids)
    kept_coords = [node_coords[i] for i in kept_list]
    ll          = dict(zip(kept_list, nodes_to_wgs84(kept_coords)))

    rows = []
    for e in g.es:
        u, v = e.source, e.target
        if u not in kept_ids or v not in kept_ids:
            continue
        lon_a, lat_a = ll[u]
        lon_b, lat_b = ll[v]
        rows.append({
            "node_a_x":      node_coords[u][0],
            "node_a_y":      node_coords[u][1],
            "node_a_lon":    lon_a,
            "node_a_lat":    lat_a,
            "node_b_x":      node_coords[v][0],
            "node_b_y":      node_coords[v][1],
            "node_b_lon":    lon_b,
            "node_b_lat":    lat_b,
            "length_m":      e["length"],
            "degree_a":      int(deg[u]),
            "degree_b":      int(deg[v]),
            "betweenness_a": bc[u],
            "betweenness_b": bc[v],
            "road_name":     e["road"],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build and visualise Spain's interurban road network."
    )
    p.add_argument(
        "-s", "--save-dataset",
        action="store_true",
        default=False,
        help="Export a CSV dataset with node/edge statistics."
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("1/5  Loading GeoPackage …")
    gdf = load_and_project(GPKG_PATH)
    print(f"     {len(gdf):,} segments loaded.")

    print("2/5  Building graph …")
    g, node_coords, node_roads, edge_roads, edge_lengths = build_igraph(gdf)
    print(f"     Graph: {g.vcount():,} nodes, {g.ecount():,} edges.")

    print("3/5  Computing centrality metrics …")
    bc, deg = compute_metrics(g)

    print("4/5  Filtering nodes …")
    mask = filter_nodes(bc, deg)
    n_kept = mask.sum()
    print(f"     {n_kept:,} nodes survive "
          f"(betweenness ≥ {BETWEENNESS_THRESHOLD}, degree ≥ {DEGREE_THRESHOLD}).")

    if n_kept == 0:
        print("  No nodes pass the thresholds — try lowering them.")
        return

    kept_ids    = np.where(mask)[0]
    kept_coords = [node_coords[i] for i in kept_ids]
    bc_kept     = bc[mask]
    deg_kept    = deg[mask].astype(float)

    print("5/5  Plotting …")
    plot_network(kept_coords, bc_kept, deg_kept, OUTPUT_PLOT)

    if args.save_dataset:
        print("  Building dataset …")
        df = build_dataset(g, mask, node_coords, node_roads, bc, deg)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"  Dataset saved → {OUTPUT_CSV}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
