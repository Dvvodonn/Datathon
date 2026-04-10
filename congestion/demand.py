"""
congestion/demand.py — Road traffic → EV charging demand signal per candidate station.

Pipeline
--------
1. assign_aadt_from_flow() assign AADT to each candidate from data_main/road_edges_flow.csv
                           (calibrated three-tier AADT for all 326,183 OSM edges).
                           Finds nearest OSM edge midpoint via cKDTree and reads its
                           effective_aadt and aadt_source directly.
2. add_usage_crosscheck()  append usage_count from Benders cuts as a quality signal
3. compute_lambda()        convert AADT → λ_k (EV vehicles/hour) via EV penetration
                           and peak-hour factor; respects per-segment override column
4. build_demand()          top-level entry point; saves outputs/candidate_demand.csv

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


# ── AADT assignment from road_edges_flow.csv ─────────────────────────────────

def assign_aadt_from_flow(
    candidates_gdf: gpd.GeoDataFrame,
    flow_csv:   str = cfg.ROAD_FLOW_CSV,
    edges_gpkg: str = cfg.EDGES_GPKG,
) -> gpd.GeoDataFrame:
    """
    Assign AADT to every candidate station using data_main/road_edges_flow.csv
    (calibrated three-tier AADT for all 326,183 OSM edges).

    Finds the nearest OSM edge midpoint to each candidate via cKDTree and reads
    its effective_aadt and aadt_source directly.  No IMD spatial join needed.

    Returns candidates_gdf extended with:
      aadt_assigned  — effective_aadt of the nearest OSM edge (veh/day)
      impute_tier    — 1/2/3 mapped from aadt_source
      edge_dist_m    — distance to nearest OSM edge midpoint (metres)
    """
    from scipy.spatial import cKDTree

    print("  Loading road_edges_flow.csv …")
    flow = pd.read_csv(flow_csv, usecols=["effective_aadt", "aadt_source", "highway"])

    print("  Loading OSM edge geometries for midpoints …")
    edges = gpd.read_file(edges_gpkg, columns=["geometry"])
    edges = edges.to_crs("EPSG:25830")
    midpoints = edges.geometry.interpolate(0.5, normalized=True)

    mid_x = midpoints.x.values
    mid_y = midpoints.y.values
    valid  = np.isfinite(mid_x) & np.isfinite(mid_y)
    mid_x, mid_y    = mid_x[valid], mid_y[valid]
    flow_valid      = flow[valid].reset_index(drop=True)

    print(f"  Building KD-tree from {len(mid_x):,} edge midpoints …")
    tree = cKDTree(np.column_stack([mid_x, mid_y]))

    cand_x = candidates_gdf.geometry.x.values
    cand_y = candidates_gdf.geometry.y.values
    dists, idxs = tree.query(np.column_stack([cand_x, cand_y]), k=1)

    result = candidates_gdf.copy()
    result["aadt_assigned"]  = flow_valid["effective_aadt"].values[idxs]
    result["edge_dist_m"]    = dists
    # highway_clean: first token only (handles "motorway|motorway_link" tags)
    result["highway_class"]  = (
        pd.Series(flow_valid["highway"].values[idxs])
        .str.split("|").str[0]
        .fillna("secondary")
        .values
    )

    tier_map = {
        "tier1_name_spatial": 1,
        "tier2_road_idw":     2,
        "tier3_knn":          3,
    }
    result["impute_tier"] = (
        pd.Series(flow_valid["aadt_source"].values[idxs]).map(tier_map).fillna(3).values
    )

    tier_counts = result["impute_tier"].value_counts().sort_index()
    labels = {1: "direct MITMA (name+spatial)",
              2: "road IDW interpolation",
              3: "global KNN (calibrated)"}
    print("  AADT assignment summary:")
    for tier, count in tier_counts.items():
        med = result.loc[result["impute_tier"] == tier, "aadt_assigned"].median()
        print(f"    Tier {tier} ({labels[int(tier)]}): "
              f"{count:,} candidates  median AADT={med:,.0f}")

    return result


# ── (legacy stubs kept for reference — no longer called) ─────────────────────

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


# ── 6a. Corridor-aware through-gap computation ────────────────────────────────

def compute_corridor_gaps(
    demand_df:  pd.DataFrame,
    edges_path: str = cfg.EDGES_250_CSV,
) -> pd.Series:
    """
    For each candidate k, compute the through-gap G_k (km) — the total road
    distance between the nearest existing charger behind k and the nearest
    existing charger ahead of k along the same corridor.

    Uses edges_250.csv (pre-computed Dijkstra distances between all node pairs)
    to find road-network distance from each candidate to existing chargers.
    Through-gap is approximated as d1 + d2, where d1 and d2 are the two
    shortest distances to distinct existing chargers. If only one charger is
    reachable within the 250 km graph horizon, fall back to 2 × d1.

    Returns a Series of through_gap_km indexed as demand_df.
    """
    print("  Computing corridor gaps from edges_250 …")
    edges = pd.read_csv(edges_path,
                        usecols=["lon_a", "lat_a", "lon_b", "lat_b",
                                 "a_is_charger", "b_is_charger", "distance_km"])

    # Case 1: candidate is endpoint A, existing charger is endpoint B
    c1 = (edges[edges["b_is_charger"] == 1]
          [["lon_a", "lat_a", "lon_b", "lat_b", "distance_km"]]
          .rename(columns={
              "lon_a": "lon", "lat_a": "lat",
              "lon_b": "charger_lon", "lat_b": "charger_lat",
          }))

    # Case 2: existing charger is endpoint A, candidate is endpoint B
    c2 = (edges[edges["a_is_charger"] == 1]
          [["lon_b", "lat_b", "lon_a", "lat_a", "distance_km"]]
          .rename(columns={
              "lon_b": "lon", "lat_b": "lat",
              "lon_a": "charger_lon", "lat_a": "charger_lat",
          }))

    all_to_ec = pd.concat([c1, c2], ignore_index=True)
    for col in ["lon", "lat", "charger_lon", "charger_lat"]:
        all_to_ec[col] = all_to_ec[col].round(6)

    # Keep only the closest path from a candidate to each distinct charger.
    all_to_ec = (all_to_ec
                 .sort_values(["lon", "lat", "distance_km"])
                 .drop_duplicates(subset=["lon", "lat", "charger_lon", "charger_lat"]))
    all_to_ec["rank"] = all_to_ec.groupby(["lon", "lat"]).cumcount()

    nearest_two = (all_to_ec[all_to_ec["rank"] < 2]
                   .pivot(index=["lon", "lat"], columns="rank", values="distance_km"))
    d1_lookup = nearest_two.get(0, pd.Series(dtype=float))
    d2_lookup = nearest_two.get(1, pd.Series(dtype=float))

    # Join back to demand_df
    demand_df = demand_df.copy()
    demand_df["_lon"] = demand_df["lon"].round(6)
    demand_df["_lat"] = demand_df["lat"].round(6)

    idx = demand_df.set_index(["_lon", "_lat"]).index
    d1 = pd.Series(idx.map(d1_lookup.to_dict()), index=demand_df.index).fillna(250.0)
    d2 = pd.Series(idx.map(d2_lookup.to_dict()), index=demand_df.index)
    d1 = d1.clip(upper=250.0)
    d2 = d2.clip(upper=250.0)

    # If only one charger is reachable, revert to the old symmetric fallback.
    through_gap = (d1 + d2.fillna(d1)).clip(upper=250.0)

    n_single = d2.isna().sum()
    n_missing = (d1 >= 250.0).sum()
    print(f"  Through-gap: median={through_gap.median():.1f} km  "
          f"min={through_gap.min():.1f}  max={through_gap.max():.1f}  "
          f"single-sided fallback: {n_single:,}  capped-at-250: {n_missing:,}")
    return through_gap


# ── 6a-ex. Corridor gap for existing chargers ────────────────────────────────

def compute_corridor_gaps_existing(
    existing_df: pd.DataFrame,
    edges_path:  str = cfg.EDGES_250_CSV,
) -> pd.Series:
    """
    For each existing charger, compute through_gap_km using the same d1+d2
    method as compute_corridor_gaps() for candidates.

    Uses edges_250.csv rows where BOTH endpoints are existing chargers
    (a_is_charger=1 AND b_is_charger=1) — these are pre-computed road-network
    Dijkstra distances, so no Euclidean approximation is needed.

    Returns a Series of through_gap_km aligned to existing_df.
    """
    print("  Computing corridor gaps for existing chargers from edges_250 …")
    edges = pd.read_csv(edges_path,
                        usecols=["lon_a", "lat_a", "lon_b", "lat_b",
                                 "a_is_charger", "b_is_charger", "distance_km"])

    ec = edges[(edges["a_is_charger"] == 1) & (edges["b_is_charger"] == 1)].copy()

    # Build bidirectional: each charger appears as both "source" and "target"
    fwd = ec[["lon_a", "lat_a", "lon_b", "lat_b", "distance_km"]].rename(
              columns={"lon_a": "lon", "lat_a": "lat",
                       "lon_b": "other_lon", "lat_b": "other_lat"})
    bwd = ec[["lon_b", "lat_b", "lon_a", "lat_a", "distance_km"]].rename(
              columns={"lon_b": "lon", "lat_b": "lat",
                       "lon_a": "other_lon", "lat_a": "other_lat"})
    both = pd.concat([fwd, bwd], ignore_index=True)
    for col in ["lon", "lat", "other_lon", "other_lat"]:
        both[col] = both[col].round(6)

    both = (both.sort_values(["lon", "lat", "distance_km"])
                .drop_duplicates(subset=["lon", "lat", "other_lon", "other_lat"]))
    both["rank"] = both.groupby(["lon", "lat"]).cumcount()

    nearest_two = (both[both["rank"] < 2]
                   .pivot(index=["lon", "lat"], columns="rank", values="distance_km"))
    d1_lookup = nearest_two.get(0, pd.Series(dtype=float))
    d2_lookup = nearest_two.get(1, pd.Series(dtype=float))

    ex = existing_df.copy()
    ex["_lon"] = ex["lon"].round(6)
    ex["_lat"] = ex["lat"].round(6)
    idx = ex.set_index(["_lon", "_lat"]).index
    d1 = pd.Series(idx.map(d1_lookup.to_dict()), index=ex.index).fillna(250.0).clip(upper=250.0)
    d2 = pd.Series(idx.map(d2_lookup.to_dict()), index=ex.index).clip(upper=250.0)

    through_gap = (d1 + d2.fillna(d1)).clip(upper=250.0)
    print(f"  Through-gap (existing): median={through_gap.median():.1f} km  "
          f"min={through_gap.min():.1f}  max={through_gap.max():.1f}")
    return through_gap


# ── 6b. Variable stop rate (logistic of through-gap) ─────────────────────────

def compute_variable_stop_rates(
    through_gap_km: pd.Series,
    r_min:      float = cfg.STOP_RATE_MIN,
    r_max:      float = cfg.STOP_RATE_MAX,
    d50:        float = cfg.D50_KM,
    steepness:  float = cfg.STOP_STEEPNESS,
) -> pd.Series:
    """
    Logistic stop rate as a function of corridor through-gap:

        stop_rate(G) = r_min + (r_max - r_min) × σ((G - d50) / steepness)

    Parameters calibrated so:
      - stop_rate ≈ r_min  when G ≈ 0  (alternative charger very close)
      - stop_rate = 0.47   when G = d50 = 125 km (half comfortable range)
      - stop_rate ≈ r_max  when G = 250 km (model range limit → forced stop)
    """
    from scipy.special import expit
    rates = r_min + (r_max - r_min) * expit((through_gap_km - d50) / steepness)
    print(f"  Variable stop rates: min={rates.min():.3f}  "
          f"median={rates.median():.3f}  max={rates.max():.3f}")
    return rates


# ── 6c. Convert AADT → λ_k ───────────────────────────────────────────────────

def compute_lambda(
    result: pd.DataFrame,
    ev_penetration:   float = cfg.EV_PENETRATION,
    peak_hour_factor: float = cfg.PEAK_HOUR_FACTOR,
    stop_rate = cfg.STOP_RATE,   # float or pd.Series
) -> pd.DataFrame:
    """
    Convert AADT (vehicles/day) to EV arrival rate λ_k (vehicles/hour).

        λ_k = AADT × ev_penetration × peak_hour_factor × stop_rate

    stop_rate may be a scalar (flat) or a per-row Series (corridor-aware).
    """
    pen = result.get("ev_penetration", pd.Series(ev_penetration, index=result.index))
    pen = pen.fillna(ev_penetration)

    result = result.copy()
    result["stop_rate"] = stop_rate   # scalar broadcasts; Series aligns by index
    result["lambda_k"]  = result["aadt_assigned"] * pen * peak_hour_factor * result["stop_rate"]
    result["lambda_k"]  = result["lambda_k"].clip(lower=0.0)

    print(f"  λ_k stats (EV vehicles/hour):")
    print(f"    min={result['lambda_k'].min():.4f}  "
          f"median={result['lambda_k'].median():.4f}  "
          f"max={result['lambda_k'].max():.4f}")
    print(f"    zero-demand candidates: {(result['lambda_k'] == 0).sum():,}")

    return result


# ── 7. Top-level entry point ──────────────────────────────────────────────────

def build_demand(
    nodes_path:       str   = cfg.NODES_CSV,
    flow_csv:         str   = cfg.ROAD_FLOW_CSV,
    edges_path:       str   = cfg.EDGES_GPKG,
    edges_250_path:   str   = cfg.EDGES_250_CSV,
    cuts_path:        str   = cfg.BENDERS_CUTS,
    ev_penetration:   float = cfg.EV_PENETRATION,
    peak_hour_factor: float = cfg.PEAK_HOUR_FACTOR,
    variable_sr:      bool  = True,
) -> pd.DataFrame:
    """
    Full demand pipeline.

    variable_sr=True  : stop rate is a logistic function of through-gap to
                        nearest existing charger (corridor-aware).
    variable_sr=False : flat STOP_RATE constant for every candidate.

    Returns a DataFrame with columns:
      lat, lon, name, aadt_assigned, impute_tier, edge_dist_m,
      highway_class, [through_gap_km, stop_rate,] usage_count, lambda_k

    Writes congestion/outputs/candidate_demand.csv.
    """
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    mode = "variable stop rate" if variable_sr else f"flat stop rate ({cfg.STOP_RATE})"
    print(f"=== Demand pipeline ({mode}) ===")

    nodes = pd.read_csv(nodes_path)
    candidates = nodes[nodes["is_feasible_location"] == 1].copy().reset_index(drop=True)
    print(f"  Candidates (feasible locations): {len(candidates):,}")

    candidates_gdf = gpd.GeoDataFrame(
        candidates,
        geometry=gpd.points_from_xy(candidates["lon"], candidates["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:25830")

    assigned = assign_aadt_from_flow(candidates_gdf, flow_csv, edges_path)
    result = pd.DataFrame(assigned.drop(columns=["geometry"]))
    result = add_usage_crosscheck(result, cuts_path)

    if variable_sr:
        through_gap = compute_corridor_gaps(result, edges_250_path)
        stop_rates  = compute_variable_stop_rates(through_gap)
        result["through_gap_km"] = through_gap.values
        result = compute_lambda(result, ev_penetration, peak_hour_factor, stop_rates.values)
    else:
        result = compute_lambda(result, ev_penetration, peak_hour_factor, cfg.STOP_RATE)

    keep_cols = [
        "lat", "lon", "name",
        "aadt_assigned", "impute_tier", "edge_dist_m",
        "highway_class", "through_gap_km", "stop_rate",
        "usage_count", "lambda_k",
    ]
    result = result[[c for c in keep_cols if c in result.columns]]

    out_path = cfg.OUTPUTS_DIR / "candidate_demand.csv"
    result.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}  ({len(result):,} rows)")
    return result


# ── 8. Existing charger demand (fixed capacity, actual power) ─────────────────

def build_existing_demand(
    nodes_path:       str   = cfg.NODES_CSV,
    flow_csv:         str   = cfg.ROAD_FLOW_CSV,
    edges_path:       str   = cfg.EDGES_GPKG,
    ev_penetration:   float = cfg.EV_PENETRATION,
    peak_hour_factor: float = cfg.PEAK_HOUR_FACTOR,
    stop_rate:        float = cfg.STOP_RATE,
    variable_sr:      bool  = False,
) -> pd.DataFrame:
    """
    Compute λ_k for every existing charger station, paired with its actual
    n_chargers and mean_power_kw from the enriched nodes.csv.

    Returns a DataFrame with columns:
      lat, lon, n_chargers, mean_power_kw, aadt_assigned, lambda_k

    Also writes congestion/outputs/existing_demand.csv.
    """
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Existing charger demand ===")
    nodes = pd.read_csv(nodes_path)

    if "n_chargers" not in nodes.columns:
        raise RuntimeError(
            "nodes.csv missing 'n_chargers' — run "
            "scripts/processing/enrich_nodes_capacity.py first"
        )

    existing = nodes[nodes["is_existing_charger"] == 1].copy().reset_index(drop=True)
    print(f"  Existing charger stations: {len(existing):,}")

    existing_gdf = gpd.GeoDataFrame(
        existing,
        geometry=gpd.points_from_xy(existing["lon"], existing["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:25830")

    assigned = assign_aadt_from_flow(existing_gdf, flow_csv, edges_path)
    result = pd.DataFrame(assigned.drop(columns=["geometry"]))

    pen = ev_penetration
    if variable_sr:
        through_gap = compute_corridor_gaps_existing(result)
        result["through_gap_km"] = through_gap.values
        result["stop_rate"] = compute_variable_stop_rates(through_gap).values
    else:
        result["stop_rate"] = stop_rate

    result["lambda_k"] = (
        result["aadt_assigned"] * pen * peak_hour_factor * result["stop_rate"]
    ).clip(lower=0.0)

    keep = ["lat", "lon", "n_chargers", "mean_power_kw",
            "aadt_assigned", "highway_class", "through_gap_km", "stop_rate", "lambda_k"]
    result = result[[c for c in keep if c in result.columns]]

    out = cfg.OUTPUTS_DIR / "existing_demand.csv"
    result.to_csv(out, index=False)
    print(f"  Saved → {out}  ({len(result):,} rows)")
    print(f"  λ_k: min={result['lambda_k'].min():.4f}  "
          f"median={result['lambda_k'].median():.4f}  "
          f"max={result['lambda_k'].max():.4f}")
    return result


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    demand = build_demand()
    existing = build_existing_demand()
    print("\nDone.")
    print(demand.head(5).to_string())
    print(existing.head(5).to_string())
