"""
build_road_flow_network.py
==========================
Assigns effective AADT to every OSM interurban edge using a three-tier cascade:

  Tier 1 — Road name match + 100 m spatial buffer
            OSM edge midpoint must fall within 100 m of a MITMA segment
            that shares the same normalised road name.  Assigns exact AADT
            from the MITMA segment whose buffer the midpoint lies in.

  Tier 2 — Same road name, inverse-distance-weighted average of the 2
            nearest MITMA centroids on the same road (within 30 km).
            Used for edges whose ref matches a MITMA road but that sat
            outside every 100 m buffer.

  Tier 3 — KNN K=5, same OSM highway class.
            Spatial mean of the 5 nearest MITMA centroids belonging to the
            matching highway class group.  Covers all remaining edges.

Output: data_main/road_edges_flow.csv
  Columns: u, v, key, osmid, highway, length_m, travel_time_s, ref,
           effective_aadt, directional_aadt, aadt_source
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import re

# ── Paths ────────────────────────────────────────────────────────────────────
MITMA_GEOJSON = "data_main/traffic/imd_total_por_tramo.geojson"
OSM_GPKG      = "data/raw/road_network/spain_interurban_edges.gpkg"
OUT_CSV       = "data_main/road_edges_flow.csv"

CRS_M   = "EPSG:3857"
CRS_DEG = "EPSG:4326"

TIER1_BUFFER_M = 100    # strict: same carriageway
TIER2_MAX_KM   = 30     # max distance to use same-road neighbour
TIER3_K        = 5      # KNN neighbours

# ── Highway class grouping ───────────────────────────────────────────────────
# Maps any OSM highway value to one of 4 tier-groups for KNN
def _hwy_group(hwy):
    h = str(hwy).lower()
    if "motorway" in h:
        return "motorway"
    if "trunk" in h:
        return "trunk"
    if "primary" in h:
        return "primary"
    return "secondary"

# MITMA road prefix → KNN group (for building KNN trees from MITMA data)
def _mitma_group(road_name):
    if pd.isna(road_name):
        return "secondary"
    r = str(road_name).upper()
    if r.startswith("AP"):
        return "motorway"
    if r.startswith("A-") or r.startswith("A "):
        return "trunk"
    if r.startswith("N-") or r.startswith("N "):
        return "primary"
    return "secondary"

# ── Name normalisation ───────────────────────────────────────────────────────
def norm_mitma(name):
    """AP-7N → AP-7  (strip trailing directional letter)"""
    if pd.isna(name):
        return None
    return re.sub(r"[NSEW]$", "", str(name).strip().upper())

def norm_osm_refs(ref_str):
    """'N-6;A-6' → ['N-6', 'A-6']  (split by semicolons/commas/slashes)"""
    if pd.isna(ref_str):
        return []
    parts = re.split(r"[;,/]", str(ref_str))
    return [p.strip().upper() for p in parts if p.strip()]

# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading data …")
flow = gpd.read_file(MITMA_GEOJSON).to_crs(CRS_DEG)
flow = flow[flow["aadt_total"].notna() & (flow["aadt_total"] > 0)].copy()
flow["road_norm"] = flow["road"].apply(norm_mitma)
flow["pk_mid"] = (flow["pk_start_km"] + flow["pk_end_km"]) / 2
flow.index = range(len(flow))  # clean integer index

osm = gpd.read_file(OSM_GPKG).to_crs(CRS_DEG)
osm["eidx"] = np.arange(len(osm))
osm["hwy_group"] = osm["highway"].apply(_hwy_group)
print(f"  MITMA: {len(flow):,} segments  |  OSM: {len(osm):,} edges")

# ── Precompute OSM midpoints in projected CRS ─────────────────────────────────
print("Computing OSM midpoints …")
osm_m   = osm.to_crs(CRS_M)
osm_mid = osm_m.geometry.interpolate(0.5, normalized=True)  # midpoint in meters
osm["mid_x"] = osm_mid.x.values
osm["mid_y"] = osm_mid.y.values

# ── Precompute MITMA centroids in projected CRS ───────────────────────────────
flow_m  = flow.to_crs(CRS_M)
flow["cen_x"] = flow_m.geometry.centroid.x.values
flow["cen_y"] = flow_m.geometry.centroid.y.values

# ── OSM ref → normalised road name set (one per edge) ────────────────────────
osm["ref_norm_list"] = osm["ref"].apply(norm_osm_refs)
# Build a flat map: eidx → set of road names
osm_road_set = {row.eidx: set(row.ref_norm_list) for row in osm.itertuples()}

# All normalised MITMA road names (fast lookup set)
mitma_road_names = set(flow["road_norm"].dropna().unique())

# Edges where at least one ref matches a MITMA road
print("Finding OSM edges with road name matches …")
has_road_match = np.array([
    bool(osm_road_set[idx] & mitma_road_names)
    for idx in osm["eidx"]
])
print(f"  {has_road_match.sum():,} / {len(osm):,} OSM edges have a road name in MITMA")

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 — 100 m buffer spatial + road name match
# ═══════════════════════════════════════════════════════════════════════════════
print("\nTier 1: 100 m buffer + road name match …")

# Buffer MITMA segments
flow_m_buf = flow.to_crs(CRS_M).copy()
flow_m_buf["geometry"] = flow_m_buf.geometry.buffer(TIER1_BUFFER_M)
flow_buf = flow_m_buf.to_crs(CRS_DEG)[["road_norm", "aadt_total", "cen_x", "cen_y", "geometry"]].copy()
flow_buf["fid"] = np.arange(len(flow_buf))

# Only join OSM edges that have a road name match (big performance win)
osm_with_road = osm[has_road_match][["eidx", "mid_x", "mid_y", "ref_norm_list", "geometry"]].copy()

# Build midpoint GDF for spatial join
mid_gdf = gpd.GeoDataFrame(
    {"eidx": osm_with_road["eidx"].values,
     "ref_norm_list": osm_with_road["ref_norm_list"].values},
    geometry=gpd.points_from_xy(
        osm_with_road["mid_x"].values,
        osm_with_road["mid_y"].values,
    ),
    crs=CRS_M,
).to_crs(CRS_DEG)

# Spatial join midpoints → buffered MITMA polygons
print("  Spatial join …")
joined = gpd.sjoin(mid_gdf, flow_buf[["fid", "road_norm", "aadt_total", "geometry"]],
                   how="left", predicate="within")

# Keep only rows where road name matches
def _road_match(row):
    return not pd.isna(row["road_norm"]) and row["road_norm"] in row["ref_norm_list"]

joined = joined[joined.apply(_road_match, axis=1)].copy()

# For each OSM edge: pick the MITMA segment with closest centroid
# (resolves ties when multiple MITMA buffers overlap)
joined["dist2"] = (
    (joined["eidx"].map(osm.set_index("eidx")["mid_x"]) - joined["fid"].map(flow.set_index(flow.index)["cen_x"])) ** 2 +
    (joined["eidx"].map(osm.set_index("eidx")["mid_y"]) - joined["fid"].map(flow.set_index(flow.index)["cen_y"])) ** 2
)
tier1_result = joined.sort_values("dist2").groupby("eidx").first()[["aadt_total"]].rename(
    columns={"aadt_total": "effective_aadt"}
)
tier1_result["aadt_source"] = "tier1_name_spatial"
print(f"  Tier 1 matched: {len(tier1_result):,} edges")

# Mark matched
matched = set(tier1_result.index)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — Same road name, IDW from 2 nearest MITMA centroids on same road
# ═══════════════════════════════════════════════════════════════════════════════
print("\nTier 2: road name IDW (2 nearest same-road MITMA centroids) …")

# Edges: has road match but NOT in Tier 1
tier2_mask = has_road_match & ~np.isin(osm["eidx"].values, list(matched))
osm_tier2 = osm[tier2_mask].copy()
print(f"  Candidates: {len(osm_tier2):,}")

# Build per-road cKDTree from MITMA centroids
road_trees = {}
road_aadt  = {}
for road_name, grp in flow.groupby("road_norm"):
    if road_name is None:
        continue
    pts  = grp[["cen_x", "cen_y"]].values.astype(float)
    aadt = grp["aadt_total"].values.astype(float)
    valid = np.isfinite(pts).all(axis=1)
    pts, aadt = pts[valid], aadt[valid]
    if len(pts) == 0:
        continue
    road_trees[road_name] = (cKDTree(pts), aadt)

tier2_rows = []
MAX_DIST_M = TIER2_MAX_KM * 1000

for row in osm_tier2.itertuples(index=False):
    refs = row.ref_norm_list
    best_aadt = None
    best_dist = np.inf

    for road in refs:
        if road not in road_trees:
            continue
        tree, aadts = road_trees[road]
        k = min(2, len(aadts))
        dists, idxs = tree.query([row.mid_x, row.mid_y], k=k)
        dists = np.atleast_1d(dists)
        idxs  = np.atleast_1d(idxs)

        if dists[0] > MAX_DIST_M:
            continue  # too far — fall to Tier 3

        if k == 1 or dists[1] > MAX_DIST_M:
            a = float(aadts[idxs[0]])
        else:
            # Inverse-distance weighting (avoid div-by-zero)
            w0 = 1.0 / max(dists[0], 1.0)
            w1 = 1.0 / max(dists[1], 1.0)
            a  = (w0 * aadts[idxs[0]] + w1 * aadts[idxs[1]]) / (w0 + w1)

        if dists[0] < best_dist:
            best_dist = dists[0]
            best_aadt = a

    if best_aadt is not None:
        tier2_rows.append({"eidx": row.eidx, "effective_aadt": best_aadt,
                           "aadt_source": "tier2_road_idw"})

tier2_result = pd.DataFrame(tier2_rows).set_index("eidx") if tier2_rows else pd.DataFrame()
print(f"  Tier 2 matched: {len(tier2_result):,} edges")

matched |= set(tier2_result.index)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3 — KNN K=5 global (all MITMA data, no class filter)
# ═══════════════════════════════════════════════════════════════════════════════
# We use global KNN rather than class-specific KNN because the secondary
# MITMA group is sparse (~519 Madrid M-roads with very high AADT) and would
# assign unrealistic traffic to rural secondary roads.  Global KNN picks the
# geographically nearest measured roads, which is more spatially coherent.
print("\nTier 3: KNN K=5 global MITMA …")

_all = flow[["cen_x", "cen_y", "aadt_total"]].dropna()
all_pts  = _all[["cen_x", "cen_y"]].values.astype(float)
all_aadt = _all["aadt_total"].values.astype(float)
global_tree = cKDTree(all_pts)

# Edges not yet matched
tier3_mask = ~np.isin(osm["eidx"].values, list(matched))
osm_tier3  = osm[tier3_mask].copy()
print(f"  Candidates: {len(osm_tier3):,}")

tier3_pts = osm_tier3[["mid_x", "mid_y"]].values.astype(float)
k = min(TIER3_K, len(all_aadt))
dists_all, idxs_all = global_tree.query(tier3_pts, k=k)
aadt_vals = np.mean(all_aadt[idxs_all], axis=1)

tier3_rows = [
    {"eidx": row.eidx, "effective_aadt": float(aadt_vals[i]), "aadt_source": "tier3_knn"}
    for i, row in enumerate(osm_tier3.itertuples(index=False))
]

tier3_result = pd.DataFrame(tier3_rows).set_index("eidx") if tier3_rows else pd.DataFrame()
print(f"  Tier 3 matched: {len(tier3_result):,} edges")

# ═══════════════════════════════════════════════════════════════════════════════
# Combine all tiers
# ═══════════════════════════════════════════════════════════════════════════════
print("\nCombining tiers …")

# Tier 1 has higher priority over 2 over 3 — already disjoint by construction
all_aadt_df = pd.concat([tier1_result[["effective_aadt", "aadt_source"]],
                          tier2_result[["effective_aadt", "aadt_source"]],
                          tier3_result[["effective_aadt", "aadt_source"]]])

assert len(all_aadt_df) == len(osm), \
    f"Coverage mismatch: {len(all_aadt_df)} vs {len(osm)} edges"

# Merge back to OSM
osm_out = osm.set_index("eidx").join(all_aadt_df)
osm_out["effective_aadt"] = osm_out["effective_aadt"].clip(lower=0)

# ── Tier 3 calibration ───────────────────────────────────────────────────────
# Tier 3 global KNN systematically overestimates secondary/primary/trunk roads
# because MITMA only measures major roads and urban ring roads.
# Calibration factor computed dynamically:
#   factor = median(Tier1+2 AADT for group) / median(Tier3 KNN AADT for group)
# This rescales Tier 3 so each class's median matches the actual measured median.
print("\nComputing Tier 3 calibration factors …")
tier3_mask_out = osm_out["aadt_source"] == "tier3_knn"
t12_mask_out   = ~tier3_mask_out

CAL_FACTORS = {}
for grp in ["motorway", "trunk", "primary", "secondary"]:
    t12_med = osm_out.loc[t12_mask_out & (osm_out["hwy_group"] == grp),
                          "effective_aadt"].median()
    t3_med  = osm_out.loc[tier3_mask_out & (osm_out["hwy_group"] == grp),
                          "effective_aadt"].median()
    if pd.isna(t12_med) or pd.isna(t3_med) or t3_med == 0:
        factor = 1.0   # no data to calibrate against — leave unchanged
    else:
        factor = t12_med / t3_med
    CAL_FACTORS[grp] = factor
    print(f"  {grp:<12}  tier1+2 median={t12_med:>8,.0f}  "
          f"tier3 median={t3_med:>8,.0f}  factor={factor:.3f}")

for grp, factor in CAL_FACTORS.items():
    mask = tier3_mask_out & (osm_out["hwy_group"] == grp)
    osm_out.loc[mask, "effective_aadt"] = (
        osm_out.loc[mask, "effective_aadt"] * factor
    ).clip(lower=1)
print("Tier 3 calibration applied.")

osm_out["directional_aadt"] = osm_out["effective_aadt"] / 2.0

# ── Output ───────────────────────────────────────────────────────────────────
keep_cols = ["u", "v", "key", "osmid", "highway", "length_m", "travel_time_s",
             "ref", "effective_aadt", "directional_aadt", "aadt_source"]
out = osm_out[keep_cols].reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}  ({len(out):,} edges)")

# ── Validation ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COVERAGE SUMMARY")
print("=" * 60)
counts = out["aadt_source"].value_counts()
total  = len(out)
for src, cnt in counts.items():
    km = out.loc[out["aadt_source"] == src, "length_m"].sum() / 1000
    print(f"  {src:<25}  {cnt:>7,}  ({100*cnt/total:4.1f}%)  "
          f"  {km:>8,.0f} km")

print()
print("Effective AADT by tier:")
for src in ["tier1_name_spatial", "tier2_road_idw", "tier3_knn"]:
    sub = out[out["aadt_source"] == src]["effective_aadt"]
    if len(sub) == 0:
        continue
    print(f"  {src:<25}  mean={sub.mean():,.0f}  "
          f"median={sub.median():,.0f}  max={sub.max():,.0f}")

print()
print("Calibrated Tier 3 AADT by highway group (vs Tier 1+2 actual):")
out["hwy_group"] = out["highway"].apply(_hwy_group)
for grp in ["motorway", "trunk", "primary", "secondary"]:
    t3  = out[(out["aadt_source"] == "tier3_knn") & (out["hwy_group"] == grp)]["effective_aadt"]
    t12 = out[(out["aadt_source"] != "tier3_knn") & (out["hwy_group"] == grp)]["effective_aadt"]
    if len(t3) == 0:
        continue
    t12_med = f"{t12.median():,.0f}" if len(t12) > 0 else "n/a"
    print(f"  {grp:<12}  tier3 median={t3.median():>7,.0f}  |  tier1+2 median={t12_med}")

print()
print("Overall effective AADT:")
print(f"  Mean  : {out['effective_aadt'].mean():,.0f} veh/day")
print(f"  Median: {out['effective_aadt'].median():,.0f} veh/day")

total_km = out["length_m"].sum() / 1000
print(f"\nTotal network: {total:,} edges  |  {total_km:,.0f} km")
print("=" * 60)
