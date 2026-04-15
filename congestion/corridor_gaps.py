"""
congestion/corridor_gaps.py
============================
Corridor-aware gap detection for the Iberdrola 9-corridor coverage constraint.

For each of the 9 named corridors (A-1…A-6, Mediterranean, Cantabrian, Silver):
  1. Load corridor geometry from the OSM road-network GPKG.
  2. Assign nearby stop nodes (cities + existing chargers) and feasible
     candidate stations to the corridor, ordered by 1-D axis projection.
  3. Find consecutive stop pairs whose road-network distance exceeds gap_km.
  4. For each such gap, find existing feasible candidates that can cover it.
     If none exist ("uncoverable"), synthesise a midpoint site placed exactly
     on the corridor geometry and return it as a new feasible candidate.

Buffer:
  PHI_CORRIDOR_BUFFER_KM (default 5 km) controls which existing nodes are
  considered "on" a corridor.  Synthetic midpoints are placed at arc-length
  position (pos_u + pos_v) / 2 on the corridor geometry — buffer = 0 km.

Return value of compute_corridor_gaps():
  (gap_data, synthetic_sites)
  gap_data       : {corridor_name: [[feas_idx, ...], ...]}
  synthetic_sites: [(key_tuple, corridor_name, gap_label), ...]
    key_tuple = (lon, lat) rounded to 6dp, placed on the corridor geometry.
    Indices in gap_data for synthetic sites are len(original_feas_list)+i.

The gap_data feeds the per-corridor MIP constraints in model.py:
    for each corridor c:  Σ_{g in c} z_c_g  ≥  ceil(phi * |gaps_c|)
which guarantees phi-fraction progress on EVERY corridor independently.
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.ops import linemerge, unary_union
from shapely.geometry import Point

# Spain bounding box — clips the Mediterranean corridor (AP-7 extends into France)
_SPAIN_BBOX = (-9.5, 35.5, 4.5, 44.5)   # (minlon, minlat, maxlon, maxlat)

# Ref patterns for the 9 Iberdrola corridors
CORRIDOR_PATTERNS = {
    "A-1":           r"\bA-1\b",
    "A-2":           r"\bA-2\b",
    "A-3":           r"\bA-3\b",
    "A-4":           r"\bA-4\b",
    "A-5":           r"\bA-5\b",
    "A-6":           r"\bA-6\b",
    "Mediterranean": r"\bAP-7\b|\bA-7\b",
    "Cantabrian":    r"\bA-8\b|\bAP-8\b",
    "Silver":        r"\bA-66\b",
}

# Corridors that run primarily south (sort pieces by lat descending)
_SOUTHWARD = {"A-4", "A-5", "Silver"}


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _sort_pieces(multiline, corridor_name):
    """Re-order MultiLineString pieces into geographic route order."""
    if multiline.geom_type == "LineString":
        return multiline
    pieces = list(multiline.geoms)
    if len(pieces) == 1:
        return pieces[0]

    coords = [c for piece in pieces for c in piece.coords]
    lon_range = max(c[0] for c in coords) - min(c[0] for c in coords)
    lat_range = max(c[1] for c in coords) - min(c[1] for c in coords)

    if lat_range > lon_range:
        reverse  = corridor_name in _SOUTHWARD
        key_fn   = lambda p: (-p.centroid.y if reverse else p.centroid.y)
    else:
        key_fn   = lambda p: p.centroid.x

    return linemerge(sorted(pieces, key=key_fn))


def load_corridor_geometries(edges_gpkg_path: str) -> dict:
    """
    Load and merge corridor geometries from the OSM road network GPKG.
    Returns dict {corridor_name: geometry} clipped to Spain bounding box.
    """
    minlon, minlat, maxlon, maxlat = _SPAIN_BBOX
    print("  Loading corridor geometries from GPKG …")
    edges = gpd.read_file(edges_gpkg_path)
    edges = edges.cx[minlon:maxlon, minlat:maxlat]

    geoms = {}
    for name, pat in CORRIDOR_PATTERNS.items():
        mask = edges["ref"].str.contains(pat, na=False, regex=True)
        if not mask.any():
            print(f"    WARNING: no edges found for corridor {name!r}")
            continue
        merged  = _sort_pieces(linemerge(unary_union(edges[mask].geometry)), name)
        geoms[name] = merged
        n_pieces  = len(merged.geoms) if merged.geom_type == "MultiLineString" else 1
        length_km = edges[mask]["length_m"].sum() / 1000
        print(f"    {name}: {int(mask.sum())} segments, {length_km:.0f} km "
              f"→ {n_pieces} piece(s)")

    return geoms


def _axis_project(lon: float, lat: float, bbox: tuple) -> float:
    """
    Scalar projection of (lon, lat) onto the corridor's bounding-box diagonal.
    Monotonically increases along the dominant corridor direction.
    """
    minx, miny, maxx, maxy = bbox
    dx, dy = maxx - minx, maxy - miny
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-9:
        return 0.0
    return ((lon - minx) * dx + (lat - miny) * dy) / norm


def _project_to_corridor(pt: Point, corridor) -> float:
    """
    Arc-length position of pt along the corridor (LineString or MultiLineString).
    For MultiLineString, returns cumulative arc-length position on the nearest piece.
    """
    if corridor.geom_type == "LineString":
        return corridor.project(pt)

    best_arc  = 0.0
    best_dist = float("inf")
    cumul     = 0.0
    for piece in corridor.geoms:
        d = piece.distance(pt)
        if d < best_dist:
            best_dist = d
            best_arc  = cumul + piece.project(pt)
        cumul += piece.length
    return best_arc


def _interpolate_on_corridor(corridor, arc_pos: float) -> tuple:
    """
    Return (lon, lat) of the point at arc_pos along the corridor.
    Clamps to the end if arc_pos exceeds the total length.
    """
    if corridor.geom_type == "LineString":
        pt = corridor.interpolate(min(arc_pos, corridor.length))
        return pt.x, pt.y

    cumul = 0.0
    for piece in corridor.geoms:
        if arc_pos <= cumul + piece.length:
            pt = piece.interpolate(arc_pos - cumul)
            return pt.x, pt.y
        cumul += piece.length
    # Beyond end → clamp to last piece endpoint
    last = list(corridor.geoms)[-1]
    pt   = last.interpolate(last.length)
    return pt.x, pt.y


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_corridor_gaps(
    nodes_df:          pd.DataFrame,
    edges_df:          pd.DataFrame,
    feas_list:         list,
    corridor_geoms:    dict,
    buffer_km:         float = 5.0,
    gap_km:            float = 100.0,
    min_existing_kw:   float = 150.0,
) -> tuple:
    """
    Compute per-corridor coverage gaps for the phi MIP constraint.

    For gaps where no existing feasible candidate can provide coverage, a
    synthetic site is placed at the exact midpoint along the corridor geometry
    and returned so model.py can inject it into the feas_list.

    Parameters
    ----------
    nodes_df         : DataFrame with lat, lon, key (tuple), is_city,
                       is_existing_charger, is_feasible_location, mean_power_kw.
    edges_df         : edges_250 DataFrame (already loaded in model.py).
    feas_list        : authoritative list of (lon, lat) tuples (cuts-file order).
    corridor_geoms   : dict from load_corridor_geometries().
    buffer_km        : max distance from corridor geometry to assign a node (5 km).
    gap_km           : gap threshold to enforce (100 km for Iberdrola mandate).
    min_existing_kw  : only existing chargers ≥ this kW count as gap-covering stops.
                       Lower-power AC chargers do not satisfy the 150 kW mandate.

    Returns
    -------
    gap_data : dict {corridor_name: [[feas_idx, ...], ...]}
        One inner list per coverable (or synthesised) gap.
        Indices for synthetic sites are len(original_feas_list) + i.
    synthetic_sites : list of (key_tuple, corridor_name, label)
        New sites placed exactly on the corridor geometry.
        model.py appends these to feas_list with lambda=0.
    """
    print(f"\n  Computing corridor gaps "
          f"(buffer={buffer_km} km, gap threshold={gap_km} km, "
          f"min_existing_kw={min_existing_kw:.0f}) …")

    feas_idx_map   = {k: i for i, k in enumerate(feas_list)}
    n_orig_feas    = len(feas_list)
    synthetic_sites: list = []   # (key_tuple, corridor_name, label)

    def _nk(lon, lat):
        return (round(float(lon), 6), round(float(lat), 6))

    # ── Identify qualifying stop nodes ────────────────────────────────────────
    # Cities:           always count (they are corridor endpoints / demand nodes)
    # Existing chargers: ONLY those >= min_existing_kw satisfy the 150 kW mandate
    hq_charger_mask = (
        (nodes_df["is_existing_charger"] == 1) &
        (nodes_df["mean_power_kw"] >= min_existing_kw)
    )
    hq_charger_keys = set(nodes_df.loc[hq_charger_mask, "key"])
    hq_stop_set     = hq_charger_keys   # cities do NOT close gaps

    n_hq = len(hq_charger_keys)
    n_low = int((nodes_df["is_existing_charger"] == 1).sum()) - n_hq
    print(f"  Existing chargers: {n_hq} qualify (≥{min_existing_kw:.0f} kW), "
          f"{n_low} excluded (<{min_existing_kw:.0f} kW AC/DC)")

    # ── Build stop-stop and stop-feas distance tables ─────────────────────────
    # Pre-filter by any charger/city flag (fast vectorised), then check hq_stop_set
    print("  Building distance tables …", flush=True)

    # Only charger-endpoint edges are relevant for stop lookups (cities excluded)
    a_any_stop = edges_df["a_is_charger"].astype(bool)
    b_any_stop = edges_df["b_is_charger"].astype(bool)
    a_feas     = edges_df["a_is_feasible"].astype(bool)
    b_feas     = edges_df["b_is_feasible"].astype(bool)
    within     = edges_df["distance_km"] <= gap_km

    stop_to_stop: dict = {}
    stop_to_feas: dict = {}

    # stop-stop (both endpoints are any city/charger, then check HQ)
    for row in edges_df[a_any_stop & b_any_stop & within].itertuples(index=False):
        ka = _nk(row.lon_a, row.lat_a)
        kb = _nk(row.lon_b, row.lat_b)
        if ka not in hq_stop_set or kb not in hq_stop_set:
            continue
        d = float(row.distance_km)
        if d < stop_to_stop.get(ka, {}).get(kb, float("inf")):
            stop_to_stop.setdefault(ka, {})[kb] = d
            stop_to_stop.setdefault(kb, {})[ka] = d

    # stop→feas (a=stop, b=feasible)
    for row in edges_df[a_any_stop & b_feas & within].itertuples(index=False):
        sk = _nk(row.lon_a, row.lat_a)
        if sk not in hq_stop_set:
            continue
        fk = _nk(row.lon_b, row.lat_b)
        d  = float(row.distance_km)
        if d < stop_to_feas.get(sk, {}).get(fk, float("inf")):
            stop_to_feas.setdefault(sk, {})[fk] = d

    # feas→stop (a=feasible, b=stop)
    for row in edges_df[b_any_stop & a_feas & within].itertuples(index=False):
        sk = _nk(row.lon_b, row.lat_b)
        if sk not in hq_stop_set:
            continue
        fk = _nk(row.lon_a, row.lat_a)
        d  = float(row.distance_km)
        if d < stop_to_feas.get(sk, {}).get(fk, float("inf")):
            stop_to_feas.setdefault(sk, {})[fk] = d

    print(f"  Distance tables: "
          f"{sum(len(v) for v in stop_to_stop.values()):,} stop-stop pairs, "
          f"{sum(len(v) for v in stop_to_feas.values()):,} stop-feas pairs "
          f"(≤ {gap_km:.0f} km)")

    # ── Separate node types ───────────────────────────────────────────────────
    # Only ≥ min_existing_kw chargers count as gap-closing stops.
    # Cities are NOT stops: a city without a 150 kW charger does not satisfy
    # the mandate for a driver needing to charge.
    stop_df = nodes_df[
        (nodes_df["is_existing_charger"] == 1) &
        (nodes_df["mean_power_kw"] >= min_existing_kw)
    ].copy()
    feas_df = nodes_df[nodes_df["is_feasible_location"] == 1].copy()
    buffer_deg = buffer_km / 111.0   # 1° ≈ 111 km

    gap_data: dict = {}
    total_gaps = total_cov = total_uncov = total_synth = 0

    for corridor_name, corridor_geom in corridor_geoms.items():
        minx, miny, maxx, maxy = corridor_geom.bounds
        margin = buffer_deg + 0.02

        # ── Corridor arc length (WGS-84 degrees; multiply by ~111 for km) ─────
        if corridor_geom.geom_type == "MultiLineString":
            total_arc = sum(p.length for p in corridor_geom.geoms)
        else:
            total_arc = corridor_geom.length

        # ── Stops near this corridor ──────────────────────────────────────────
        bbox_stops = stop_df[
            (stop_df["lon"] >= minx - margin) & (stop_df["lon"] <= maxx + margin) &
            (stop_df["lat"] >= miny - margin) & (stop_df["lat"] <= maxy + margin)
        ]
        real_stops = []   # (axis_pos, key, lon, lat, is_virtual)
        for _, row in bbox_stops.iterrows():
            pt = Point(row["lon"], row["lat"])
            if corridor_geom.distance(pt) <= buffer_deg:
                pos = _axis_project(row["lon"], row["lat"], corridor_geom.bounds)
                real_stops.append((pos, row["key"], float(row["lon"]), float(row["lat"]), False))

        # ── Add virtual corridor endpoints ────────────────────────────────────
        # These mark where the corridor begins/ends so gaps at the boundary
        # (start → first charger, last charger → end) are detected.
        v_start_lon, v_start_lat = _interpolate_on_corridor(corridor_geom, 0.0)
        v_end_lon,   v_end_lat   = _interpolate_on_corridor(corridor_geom, total_arc)
        vs_pos = _axis_project(v_start_lon, v_start_lat, corridor_geom.bounds)
        ve_pos = _axis_project(v_end_lon,   v_end_lat,   corridor_geom.bounds)
        # Sentinel keys (never appear in hq_stop_set / stop_to_feas)
        _VSTART = f"__VSTART_{corridor_name}__"
        _VEND   = f"__VEND_{corridor_name}__"

        assigned_stops = real_stops + [
            (vs_pos, _VSTART, v_start_lon, v_start_lat, True),
            (ve_pos, _VEND,   v_end_lon,   v_end_lat,   True),
        ]
        assigned_stops.sort(key=lambda t: t[0])

        if not real_stops:
            print(f"    {corridor_name}: 0 existing ≥{min_existing_kw:.0f}kW chargers — "
                  f"corridor endpoints added as gap boundaries")

        # ── Existing feasible candidates near this corridor ───────────────────
        bbox_feas = feas_df[
            (feas_df["lon"] >= minx - margin) & (feas_df["lon"] <= maxx + margin) &
            (feas_df["lat"] >= miny - margin) & (feas_df["lat"] <= maxy + margin)
        ]
        corridor_feas_keys = set()
        for _, row in bbox_feas.iterrows():
            pt = Point(row["lon"], row["lat"])
            if corridor_geom.distance(pt) <= buffer_deg:
                corridor_feas_keys.add(row["key"])

        # Pre-compute arc-length projections for corridor feas candidates
        # (used for virtual-endpoint distance approximation)
        feas_arc: dict = {}
        for fk in corridor_feas_keys:
            if fk in feas_idx_map:
                feas_arc[fk] = _project_to_corridor(Point(fk[0], fk[1]), corridor_geom)

        def _arc_of(lon, lat, is_virtual):
            """Arc-length position (degrees) along the corridor."""
            if is_virtual:
                # Virtual endpoints project to exactly 0 or total_arc
                return _project_to_corridor(Point(lon, lat), corridor_geom)
            return _project_to_corridor(Point(lon, lat), corridor_geom)

        def _feas_dist_from_arc(arc_ref):
            """fk → distance (km) via corridor arc-length from a virtual endpoint."""
            return {fk: abs(feas_arc[fk] - arc_ref) * 111.0
                    for fk in corridor_feas_keys if fk in feas_arc}

        # ── Walk ordered stops, detect and cover gaps ─────────────────────────
        corridor_gap_list = []
        n_gaps = n_cov = n_uncov = n_synth = 0

        for i in range(len(assigned_stops) - 1):
            pos_u, u_key, u_lon, u_lat, u_virt = assigned_stops[i]
            pos_v, v_key, v_lon, v_lat, v_virt = assigned_stops[i + 1]

            # ── Gap distance ──────────────────────────────────────────────────
            if u_virt or v_virt:
                # Use arc-length approximation (degrees × 111 km/°) for boundary gaps
                arc_u = _project_to_corridor(Point(u_lon, u_lat), corridor_geom)
                arc_v = _project_to_corridor(Point(v_lon, v_lat), corridor_geom)
                d_uv  = abs(arc_v - arc_u) * 111.0
            else:
                d_uv = stop_to_stop.get(u_key, {}).get(v_key,
                       stop_to_stop.get(v_key, {}).get(u_key, None))

            if d_uv is not None and d_uv <= gap_km:
                continue   # already covered

            n_gaps += 1

            # ── Feas distance lookups for each endpoint ───────────────────────
            if u_virt:
                arc_u = _project_to_corridor(Point(u_lon, u_lat), corridor_geom)
                u_feas = _feas_dist_from_arc(arc_u)
            else:
                u_feas = stop_to_feas.get(u_key, {})

            if v_virt:
                arc_v = _project_to_corridor(Point(v_lon, v_lat), corridor_geom)
                v_feas = _feas_dist_from_arc(arc_v)
            else:
                v_feas = stop_to_feas.get(v_key, {})

            # Existing candidates within gap_km of both endpoints
            covering = [
                feas_idx_map[fk]
                for fk in corridor_feas_keys
                if fk in feas_idx_map
                and u_feas.get(fk, float("inf")) <= gap_km
                and v_feas.get(fk, float("inf")) <= gap_km
            ]

            if covering:
                corridor_gap_list.append(covering)
                n_cov += 1
            else:
                # ── Synthesise a midpoint site exactly on the corridor ────────
                arc_u   = _project_to_corridor(Point(u_lon, u_lat), corridor_geom)
                arc_v   = _project_to_corridor(Point(v_lon, v_lat), corridor_geom)
                mid_arc = (arc_u + arc_v) / 2.0
                syn_lon, syn_lat = _interpolate_on_corridor(corridor_geom, mid_arc)
                syn_key  = _nk(syn_lon, syn_lat)
                syn_idx  = n_orig_feas + len(synthetic_sites)
                label    = f"synthetic_{corridor_name}_gap{i}"
                synthetic_sites.append((syn_key, corridor_name, label))
                # Update local lookups so subsequent gaps on same corridor see them
                feas_idx_map[syn_key] = syn_idx
                corridor_feas_keys.add(syn_key)
                feas_arc[syn_key]     = mid_arc
                # Register arc-length distances for the new synthetic stop
                half = gap_km / 2.0
                if not u_virt:
                    stop_to_feas.setdefault(u_key, {})[syn_key] = half
                if not v_virt:
                    stop_to_feas.setdefault(v_key, {})[syn_key] = half
                corridor_gap_list.append([syn_idx])
                n_uncov += 1
                n_synth += 1

        print(
            f"    {corridor_name}: {len(real_stops)} x ≥{min_existing_kw:.0f}kW charger(s) "
            f"on corridor | "
            f"{n_gaps} gap(s)>{gap_km:.0f}km "
            f"[{n_cov} covered by existing candidates, {n_synth} synthetic midpoint(s)]"
        )

        total_gaps  += n_gaps
        total_cov   += n_cov
        total_uncov += n_synth

        if corridor_gap_list:
            gap_data[corridor_name] = corridor_gap_list

    print(f"\n  All corridors: {total_gaps} total gaps, "
          f"{total_cov} covered by existing candidates, "
          f"{total_uncov} filled by synthetic midpoints\n")

    if synthetic_sites:
        print(f"  Synthetic sites added ({len(synthetic_sites)}):")
        for key, corr, label in synthetic_sites:
            print(f"    {label}  lon={key[0]:.4f}  lat={key[1]:.4f}  (on {corr})")

    return gap_data, synthetic_sites
