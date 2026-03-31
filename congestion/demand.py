"""
congestion/demand.py — Road traffic → EV charging demand signal per candidate station.

Pipeline
--------
1. load_imd()              load and reproject the IMD GeoJSON (AADT by road segment)
2. build_class_means()     compute length-weighted mean AADT per IMD road type,
                           then map to OSM highway classes — avoids the cross-class
                           spatial-join contamination identified during data analysis
3. load_edge_centroids()   load OSM edge centroids for Tier-2 highway-class lookup
4. assign_aadt()           three-tier AADT assignment for each candidate station:
                             Tier 1 — actual measurement   (nearest IMD ≤ TIER1_MAX_KM)
                             Tier 2 — class-mean imputation (TIER1 < dist ≤ TIER2_MAX_KM)
                             Tier 3 — class-mean, higher uncertainty (dist > TIER2_MAX_KM)
5. add_usage_crosscheck()  append usage_count from Benders cuts as a quality signal
6. compute_lambda()        convert AADT → λ_k (EV vehicles/hour) via EV penetration
                           and peak-hour factor; respects per-segment override column
7. build_demand()          top-level entry point; saves outputs/candidate_demand.csv

Run standalone
--------------
    python congestion/demand.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

# ensure Unicode prints correctly on Windows (cp1252 terminals)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import geopandas as gpd
import numpy as np
import pandas as pd

# allow running from project root as  python congestion/demand.py
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ── 1. Load IMD GeoJSON ───────────────────────────────────────────────────────

def load_imd(path: str = cfg.IMD_GEOJSON) -> gpd.GeoDataFrame:
    """
    Load imd_total_por_tramo.geojson, drop the single zero-AADT row,
    reproject to EPSG:25830 (UTM zone 30N) for metric spatial operations.
    """
    gdf = gpd.read_file(path)
    gdf = gdf[gdf["aadt_total"] > 0].copy()
    gdf = gdf.to_crs("EPSG:25830")
    print(f"  IMD segments loaded: {len(gdf):,}  "
          f"(AADT range {gdf['aadt_total'].min():.0f} – {gdf['aadt_total'].max():.0f})")
    return gdf


# ── 2. Build class means from IMD data ───────────────────────────────────────

def build_class_means(imd_gdf: gpd.GeoDataFrame) -> dict:
    """
    Compute length-weighted mean AADT per IMD road-type prefix (AP-, A-, N-)
    directly from the GeoJSON — NOT from a spatial join to OSM edges.

    The spatial-join-derived class means are contaminated: secondary/primary
    OSM edges near major junctions pick up A-road or N-road AADT values,
    inflating their apparent mean by ~7×.  Computing from IMD prefixes avoids
    this entirely.

    Returns a dict mapping OSM highway class → imputed AADT.
    """
    imd = imd_gdf.copy()

    # Extract road-type prefix; order matters: AP- must match before A-
    imd["road_prefix"] = (
        imd["road"]
        .str.extract(r"^(AP-|A-|N-)", expand=False)
        .str.rstrip("-")
        .fillna("other")
    )

    def _wt_mean(grp):
        total = grp["shape_length_m"].sum()
        if total == 0:
            return grp["aadt_total"].mean()
        return (grp["aadt_total"] * grp["shape_length_m"]).sum() / total

    prefix_means = imd.groupby("road_prefix").apply(_wt_mean, include_groups=False)

    ap = float(prefix_means.get("AP", 31_256))
    a  = float(prefix_means.get("A",  28_521))
    n  = float(prefix_means.get("N",   4_578))

    # secondary roads are not measured by MITMA; use N- × 0.25 as a
    # conservative lower bound (documented assumption, not a spatial fact)
    sec = n * 0.25

    means = {
        "motorway":       ap,
        "motorway_link":  ap * 0.70,   # ramps carry a fraction of trunk flow
        "trunk":          a,
        "trunk_link":     a  * 0.70,
        "primary":        n,
        "primary_link":   n  * 0.70,
        "secondary":      sec,
        "secondary_link": sec * 0.70,
    }

    print("  Class-mean AADT (from IMD prefixes, length-weighted):")
    for cls, val in means.items():
        print(f"    {cls:<20} {val:>8,.0f} veh/day")

    return means


# ── 3. Load OSM edge centroids for Tier-2 highway-class lookup ───────────────

def load_edge_centroids(path: str = cfg.EDGES_GPKG) -> gpd.GeoDataFrame:
    """
    Load spain_interurban_edges.gpkg and return a GeoDataFrame of edge
    centroids with a cleaned highway class column.  Centroids avoid loading
    the full LineString geometry into the spatial index for the Tier-2 join.
    """
    print("  Loading OSM edges for Tier-2 class lookup …")
    edges = gpd.read_file(path, columns=["highway", "length_m", "geometry"])
    edges = edges.to_crs("EPSG:25830")

    # Normalise multi-value highway tags (e.g. "motorway|motorway_link" → "motorway")
    edges["highway_clean"] = edges["highway"].str.split("|").str[0]

    centroids = edges.copy()
    centroids["geometry"] = edges.geometry.centroid

    print(f"  OSM edge centroids: {len(centroids):,}")
    return centroids[["highway_clean", "length_m", "geometry"]]


# ── 4. Three-tier AADT assignment ─────────────────────────────────────────────

def assign_aadt(
    candidates_gdf: gpd.GeoDataFrame,
    imd_gdf: gpd.GeoDataFrame,
    edge_centroids: gpd.GeoDataFrame,
    class_means: dict,
) -> gpd.GeoDataFrame:
    """
    Assign an AADT value and imputation tier to every candidate station.

    All inputs must be in EPSG:25830.

    Returns candidates_gdf extended with columns:
      aadt_assigned  — AADT value used (veh/day)
      impute_tier    — 1 (direct), 2 (class mean, nearby), 3 (class mean, distant)
      imd_dist_m     — distance to nearest IMD segment (metres)
    """
    result = candidates_gdf.copy()

    # ── Tier 1: nearest IMD segment ───────────────────────────────────────────
    imd_join = gpd.sjoin_nearest(
        result[["geometry"]],
        imd_gdf[["aadt_total", "geometry"]],
        how="left",
        distance_col="imd_dist_m",
    )
    # sjoin_nearest can produce 1-to-many when two IMD segments are equidistant;
    # keep the first match (lowest aadt_total in tie — doesn't matter for Tier 1)
    imd_join = imd_join[~imd_join.index.duplicated(keep="first")]

    result["imd_dist_m"]    = imd_join["imd_dist_m"]
    result["aadt_direct"]   = imd_join["aadt_total"]

    t1_mask = result["imd_dist_m"] <= cfg.TIER1_MAX_KM * 1000
    result.loc[t1_mask, "aadt_assigned"] = result.loc[t1_mask, "aadt_direct"]
    result.loc[t1_mask, "impute_tier"]   = 1

    # ── Tier 2 / 3: class-mean imputation ─────────────────────────────────────
    t23_idx = result.index[~t1_mask]

    if len(t23_idx) > 0:
        edge_join = gpd.sjoin_nearest(
            result.loc[t23_idx, ["geometry"]],
            edge_centroids[["highway_clean", "geometry"]],
            how="left",
            distance_col="edge_dist_m",
        )
        edge_join = edge_join[~edge_join.index.duplicated(keep="first")]

        # Map highway class → AADT; fall back to 'primary' mean if class unknown
        fallback = class_means.get("primary", 4_578.0)
        imputed_aadt = (
            edge_join.reindex(t23_idx)["highway_clean"]
            .map(class_means)
            .fillna(fallback)
        )

        result.loc[t23_idx, "aadt_assigned"] = imputed_aadt.values

        # Tier 2 vs 3 distinguished by distance to nearest IMD segment
        t2_mask = ~t1_mask & (result["imd_dist_m"] <= cfg.TIER2_MAX_KM * 1000)
        t3_mask = ~t1_mask & (result["imd_dist_m"]  > cfg.TIER2_MAX_KM * 1000)
        result.loc[result.index[t2_mask], "impute_tier"] = 2
        result.loc[result.index[t3_mask], "impute_tier"] = 3

    # ── Summary ───────────────────────────────────────────────────────────────
    tier_counts = result["impute_tier"].value_counts().sort_index()
    print("  AADT assignment summary:")
    labels = {1: "direct measurement", 2: "class mean (<=20 km)", 3: "class mean (>20 km)"}
    for tier, count in tier_counts.items():
        print(f"    Tier {tier:.0f} ({labels[int(tier)]}): {count:,} candidates")

    return result


# ── 5. Usage-count cross-check from Benders cuts ─────────────────────────────

def add_usage_crosscheck(
    result: pd.DataFrame,
    cuts_path: str = cfg.BENDERS_CUTS,
) -> pd.DataFrame:
    """
    Append usage_count: how many times each candidate's (lon, lat) node
    appeared in the 'nodes' field of any Benders cut.

    This is a topological signal (routing frequency), not a traffic volume.
    It cross-checks AADT imputation: candidates with low AADT but high
    usage_count may be underestimating demand due to data gaps.
    """
    try:
        with open(cuts_path) as f:
            cuts_data = json.load(f)
    except FileNotFoundError:
        print(f"  [usage_count] {cuts_path} not found — skipping cross-check")
        result["usage_count"] = 0
        return result

    counter = Counter()
    for cut in cuts_data.get("cuts", []):
        for node in cut.get("nodes", []):
            counter[tuple(node)] += 1

    # Align to candidates by (lon, lat) rounded to 6 dp (matches nkey in model_1.py)
    def _lookup(row):
        key = (round(float(row["lon"]), 6), round(float(row["lat"]), 6))
        return counter.get(key, 0)

    result["usage_count"] = result.apply(_lookup, axis=1)

    n_nonzero = (result["usage_count"] > 0).sum()
    print(f"  usage_count: {n_nonzero:,} candidates appeared on an optimal Benders path  "
          f"({n_nonzero / len(result) * 100:.1f}%)")

    return result


# ── 6. Convert AADT → λ_k ────────────────────────────────────────────────────

def compute_lambda(
    result: pd.DataFrame,
    ev_penetration:  float = cfg.EV_PENETRATION,
    peak_hour_factor: float = cfg.PEAK_HOUR_FACTOR,
    stop_rate:        float = cfg.STOP_RATE,
) -> pd.DataFrame:
    """
    Convert AADT (vehicles/day) to EV arrival rate λ_k (vehicles/hour).

        λ_k = AADT × ev_penetration × peak_hour_factor × stop_rate

    ev_penetration  : fraction of vehicles that are EVs
    peak_hour_factor: fraction of daily AADT in the single busiest hour
    stop_rate       : fraction of passing EVs that actually stop to charge
                      at this specific station (~5% is a near-future estimate)

    If the DataFrame contains an 'ev_penetration' column (per-segment override),
    that takes precedence over the global parameter for those rows.
    """
    pen = result.get("ev_penetration", pd.Series(ev_penetration, index=result.index))
    pen = pen.fillna(ev_penetration)

    result = result.copy()
    result["lambda_k"] = result["aadt_assigned"] * pen * peak_hour_factor * stop_rate
    result["lambda_k"] = result["lambda_k"].clip(lower=0.0)

    print(f"  λ_k stats (EV vehicles/hour):")
    print(f"    min={result['lambda_k'].min():.4f}  "
          f"median={result['lambda_k'].median():.4f}  "
          f"max={result['lambda_k'].max():.4f}")
    print(f"    zero-demand candidates: {(result['lambda_k'] == 0).sum():,}")

    return result


# ── 7. Top-level entry point ──────────────────────────────────────────────────

def build_demand(
    nodes_path:       str   = cfg.NODES_CSV,
    imd_path:         str   = cfg.IMD_GEOJSON,
    edges_path:       str   = cfg.EDGES_GPKG,
    cuts_path:        str   = cfg.BENDERS_CUTS,
    ev_penetration:   float = cfg.EV_PENETRATION,
    peak_hour_factor: float = cfg.PEAK_HOUR_FACTOR,
    stop_rate:        float = cfg.STOP_RATE,
) -> pd.DataFrame:
    """
    Full demand pipeline. Returns a DataFrame of feasible candidate locations
    with columns: lat, lon, name, aadt_assigned, impute_tier, imd_dist_m,
                  usage_count, lambda_k.

    Also writes congestion/outputs/candidate_demand.csv.
    """
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Demand pipeline ===")

    # Load candidate stations (feasible locations only)
    nodes = pd.read_csv(nodes_path)
    candidates = nodes[nodes["is_feasible_location"] == 1].copy().reset_index(drop=True)
    print(f"  Candidates (feasible locations): {len(candidates):,}")

    candidates_gdf = gpd.GeoDataFrame(
        candidates,
        geometry=gpd.points_from_xy(candidates["lon"], candidates["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:25830")

    # Load IMD and compute class means
    imd_gdf     = load_imd(imd_path)
    class_means = build_class_means(imd_gdf)
    edge_cents  = load_edge_centroids(edges_path)

    # Three-tier AADT assignment
    assigned = assign_aadt(candidates_gdf, imd_gdf, edge_cents, class_means)

    # Drop GeoDataFrame back to plain DataFrame for downstream use
    result = pd.DataFrame(assigned.drop(columns=["geometry", "aadt_direct"]))

    # Attach Benders usage-count cross-check
    result = add_usage_crosscheck(result, cuts_path)

    # Compute λ_k
    result = compute_lambda(result, ev_penetration, peak_hour_factor, stop_rate)

    # Keep a clean ordered column set
    keep_cols = [
        "lat", "lon", "name",
        "aadt_assigned", "impute_tier", "imd_dist_m",
        "usage_count", "lambda_k",
    ]
    result = result[[c for c in keep_cols if c in result.columns]]

    out_path = cfg.OUTPUTS_DIR / "candidate_demand.csv"
    result.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}  ({len(result):,} rows)")

    return result


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    demand = build_demand()
    print("\nDone.")
    print(demand.head(10).to_string())
