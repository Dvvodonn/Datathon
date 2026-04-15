"""
visualizations/corridor_phi_slider_map.py
==========================================
Interactive Plotly HTML map for the corridor phi sweep results.

Reads  datathon_master/corridor_sweep/corridor_phi_sweep.csv
       datathon_master/corridor_sweep/results_congestion_corridor_p*.csv

Features
--------
  • Phi slider (0 → 0.25 → 0.5 → 0.75 → 1.0)
  • 9 Iberdrola corridor routes drawn as coloured lines on the map
  • New stations (green circles, size ∝ charger count)
  • Corridor-mandated synthetic midpoints (amber triangles)
  • Stats panel: stations, chargers, kW, gap coverage counts
  • Dark Carto basemap

Run
---
    python visualizations/corridor_phi_slider_map.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# ── Paths ─────────────────────────────────────────────────────────────────────
SUMMARY_CSV = Path("datathon_master/corridor_sweep/corridor_phi_sweep.csv")
OUT_HTML    = Path("visualizations/corridor_phi_slider_map.html")
EDGES_GPKG  = Path("data/raw/road_network/spain_interurban_edges.gpkg")

SPAIN_CENTER = {"lat": 40.2, "lon": -3.7}
ZOOM = 5.25

# ── Colours ───────────────────────────────────────────────────────────────────
STATION_COLOR   = "#5fcf65"   # green — real new stations
SYNTHETIC_COLOR = "#f5a623"   # amber — corridor-mandated synthetic sites
CORRIDOR_COLORS = {
    "A-1":           "#7eb8f7",
    "A-2":           "#a78bfa",
    "A-3":           "#f472b6",
    "A-4":           "#fb923c",
    "A-5":           "#facc15",
    "A-6":           "#34d399",
    "Mediterranean": "#f87171",
    "Cantabrian":    "#22d3ee",
    "Silver":        "#c084fc",
}

# Known gap counts per corridor (from dry-run output, 150 kW filter + endpoints)
CORRIDOR_GAPS = {
    "A-1": 1, "A-2": 4, "A-3": 2, "A-4": 2, "A-5": 2,
    "A-6": 3, "Mediterranean": 7, "Cantabrian": 3, "Silver": 4,
}
TOTAL_GAPS = sum(CORRIDOR_GAPS.values())   # 28


def _min_covered(phi: float) -> int:
    import math
    return sum(math.ceil(phi * n) for n in CORRIDOR_GAPS.values())


def marker_size(c):
    return 10 + int(c) * 5


def _fmt(v, d=0):
    if pd.isna(v):
        return "n/a"
    return f"{float(v):,.{d}f}"


# ── Corridor geometry ─────────────────────────────────────────────────────────

def _deduplicate_parallel_tracks(pieces):
    """
    Motorways have two separate carriageway edges in OSM (northbound/southbound).
    After linemerge they appear as parallel or contained pieces.  Keep only the
    longest representative in each such group.

    Two pieces are considered duplicates when either:
      (a) Containment: the smaller piece's bounding box lies entirely inside the
          larger piece's bounding box, OR
      (b) Overlap + similarity: bounding-box overlap ≥ 70 % in both axes AND
          lengths within 25 % of each other.
    """
    n = len(pieces)
    # Process longest pieces first so we always keep the most informative one
    order = sorted(range(n), key=lambda k: pieces[k].length, reverse=True)
    skip = [False] * n

    for rank_i, i in enumerate(order):
        if skip[i]:
            continue
        b1 = pieces[i].bounds    # (minx, miny, maxx, maxy)
        l1 = pieces[i].length
        for j in order[rank_i + 1:]:
            if skip[j]:
                continue
            b2 = pieces[j].bounds
            l2 = pieces[j].length

            # (a) Containment: j's bbox fully inside i's bbox → always remove j
            # Use eps≈500 m tolerance so near-containment (e.g., 22 m slack) is caught.
            _eps = 0.005
            if (b2[0] >= b1[0] - _eps and b2[2] <= b1[2] + _eps and
                    b2[1] >= b1[1] - _eps and b2[3] <= b1[3] + _eps):
                skip[j] = True
                continue

            # (b) Overlap + length-similarity: parallel bidirectional tracks
            if abs(l1 - l2) / max(l1, l2) > 0.25:
                continue
            ox = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            oy = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            min_wx = min(b1[2] - b1[0], b2[2] - b2[0])
            min_wy = min(b1[3] - b1[1], b2[3] - b2[1])
            if min_wx < 1e-9 or min_wy < 1e-9:
                continue   # degenerate bbox — skip
            if (ox / min_wx) >= 0.70 and (oy / min_wy) >= 0.70:
                skip[j] = True

    return [p for k, p in enumerate(pieces) if not skip[k]]


def _stitch_corridor_pieces(pieces, max_stitch_km=200.0):
    """
    Build a continuous line by greedy nearest-neighbour piece selection.

    Starting from the westernmost endpoint, repeatedly pick the unvisited piece
    whose nearest endpoint is closest to the current chain tip and orient it so
    that endpoint connects forward.  Pieces further than max_stitch_km apart get
    a None separator (genuine route break, not an OSM tagging gap).

    This handles bidirectional motorway tracks and short OSM ref discontinuities
    without leaving the chain tip stranded at the wrong end of a long segment.
    """
    if not pieces:
        return [], []
    if len(pieces) == 1:
        c = list(pieces[0].coords)
        return [x for x, _ in c] + [None], [y for _, y in c] + [None]

    # Remove bidirectional parallel duplicates before stitching
    pieces = _deduplicate_parallel_tracks(pieces)

    # Find the globally westernmost endpoint as the chain start
    remaining = []
    for p in pieces:
        c = list(p.coords)
        remaining.append((c[0][0], c[0][1], c[-1][0], c[-1][1], c))

    west_val = min(min(r[0], r[2]) for r in remaining)
    best_start = None
    best_start_fwd = True
    for idx, r in enumerate(remaining):
        if min(r[0], r[2]) == west_val:
            best_start = idx
            best_start_fwd = (r[0] <= r[2])
            break

    out_lons: list = []
    out_lats: list = []
    coords_list = [r[4] for r in remaining]
    used = [False] * len(coords_list)

    c = coords_list[best_start]
    if not best_start_fwd:
        c = c[::-1]
    used[best_start] = True
    out_lons += [x for x, _ in c]
    out_lats += [y for _, y in c]
    chain_tip = c[-1]   # (lon, lat)

    while True:
        # Find nearest unused piece endpoint to chain_tip
        best_i, best_d, best_fwd = None, float("inf"), True
        for i, c in enumerate(coords_list):
            if used[i]:
                continue
            d_fwd = ((c[0][0]  - chain_tip[0]) ** 2 +
                     (c[0][1]  - chain_tip[1]) ** 2) ** 0.5 * 111.0
            d_rev = ((c[-1][0] - chain_tip[0]) ** 2 +
                     (c[-1][1] - chain_tip[1]) ** 2) ** 0.5 * 111.0
            d = min(d_fwd, d_rev)
            fwd = d_fwd <= d_rev
            if d < best_d:
                best_d, best_i, best_fwd = d, i, fwd

        if best_i is None:
            break

        used[best_i] = True
        c = coords_list[best_i]
        if not best_fwd:
            c = c[::-1]

        if best_d <= max_stitch_km:
            out_lons.append(c[0][0])
            out_lats.append(c[0][1])
        else:
            out_lons.append(None)
            out_lats.append(None)

        out_lons += [x for x, _ in c]
        out_lats += [y for _, y in c]
        chain_tip = c[-1]

    out_lons.append(None)
    out_lats.append(None)
    return out_lons, out_lats


def load_corridor_traces():
    """
    Return a list of scattermapbox line traces (one per corridor).
    Uses only motorway-type edges for clean, minimal fragment count.
    Stitches OSM ref-tagging gaps so corridors render as continuous lines.
    """
    try:
        import geopandas as gpd
        from shapely.ops import linemerge, unary_union
    except ImportError:
        print("  geopandas not available — corridors will not be drawn")
        return []

    sys.path.insert(0, "congestion")
    from corridor_gaps import CORRIDOR_PATTERNS, _sort_pieces

    _SPAIN_BBOX = (-9.5, 35.5, 4.5, 44.5)
    minlon, minlat, maxlon, maxlat = _SPAIN_BBOX

    print("  Loading corridor edges from GPKG …")
    edges = gpd.read_file(str(EDGES_GPKG))
    edges = edges.cx[minlon:maxlon, minlat:maxlat]
    # Motorway-type only: eliminates ramp/link fragments that cause visual noise
    edges = edges[edges["highway"] == "motorway"]
    print(f"  Motorway edges: {len(edges):,}")

    traces = []
    for name, pat in CORRIDOR_PATTERNS.items():
        mask = edges["ref"].str.contains(pat, na=False, regex=True)
        if not mask.any():
            print(f"    {name}: no motorway edges — skipped")
            continue

        color = CORRIDOR_COLORS.get(name, "#ffffff")
        merged = _sort_pieces(linemerge(unary_union(edges[mask].geometry)), name)

        # Simplify: 0.008° ≈ 900 m — preserves shape at zoom 5, cuts coord count
        geom_s = merged.simplify(0.008, preserve_topology=True)

        if geom_s.geom_type == "LineString":
            raw_pieces = [geom_s]
        else:
            raw_pieces = list(geom_s.geoms)

        # Filter micro-fragments (< 5 km) that are sub-pixel at zoom 5
        raw_pieces = [p for p in raw_pieces if p.length * 111 >= 5.0]

        n_before = len(raw_pieces)
        lons, lats = _stitch_corridor_pieces(raw_pieces, max_stitch_km=200.0)
        n_gaps = lons.count(None) - 1  # terminal None doesn't count
        print(f"    {name}: {n_before} piece(s) → {max(n_gaps,0)} remaining gap(s)")

        traces.append({
            "type": "scattermapbox",
            "mode": "lines",
            "name": name,
            "lat": lats,
            "lon": lons,
            "line": {"width": 3, "color": color},
            "opacity": 0.75,
            "hoverinfo": "name",
            "showlegend": True,
        })

    print(f"  {len(traces)} corridor traces loaded")
    return traces


# ── Per-scenario station traces ───────────────────────────────────────────────

def build_station_traces(built_df):
    """Return [real_trace, synthetic_trace] for the built stations dataframe."""
    real  = built_df[built_df["lambda_k"] > 0].copy()
    synth = built_df[built_df["lambda_k"] == 0].copy()

    traces = []

    if not real.empty:
        hover_real = [
            f"<b>{r.get('name', 'Station')}</b><br>"
            f"Chargers: {int(r.c_built)} × 150 kW<br>"
            f"λ: {r.lambda_k:.2f} EV/hr<br>"
            f"Wq: {r.wq_minutes:.1f} min"
            for _, r in real.iterrows()
        ]
        traces.append({
            "type": "scattermapbox",
            "mode": "markers",
            "name": "New station",
            "lat": real["lat"].tolist(),
            "lon": real["lon"].tolist(),
            "text": hover_real,
            "hovertemplate": "%{text}<extra></extra>",
            "marker": {
                "size": [marker_size(c) for c in real["c_built"].tolist()],
                "color": STATION_COLOR,
                "opacity": 0.92,
                "symbol": "circle",
            },
            "showlegend": True,
        })

    if not synth.empty:
        corr_labels = [
            r.get("name", "synthetic").replace("synthetic_", "").replace("_gap", " gap ")
            for _, r in synth.iterrows()
        ]
        hover_synth = [
            f"<b>Synthetic midpoint</b><br>{lbl}<br>Chargers: {int(r.c_built)} × 150 kW"
            for lbl, (_, r) in zip(corr_labels, synth.iterrows())
        ]
        traces.append({
            "type": "scattermapbox",
            "mode": "markers",
            "name": "Corridor midpoint (synthetic)",
            "lat": synth["lat"].tolist(),
            "lon": synth["lon"].tolist(),
            "text": hover_synth,
            "hovertemplate": "%{text}<extra></extra>",
            "marker": {
                "size": 16,
                "color": SYNTHETIC_COLOR,
                "opacity": 0.95,
                "symbol": "star",
            },
            "showlegend": True,
        })

    return traces


# ── Stats HTML panel ──────────────────────────────────────────────────────────

def _stats_html(row, phi):
    min_cov = _min_covered(phi)
    lines = [
        f"<b>γ = {_fmt(row['gamma'], 2)} (fixed) &nbsp;|&nbsp; φ = {_fmt(row['phi'], 2)}</b>",
        "",
        f"Stations built:          <b>{_fmt(row['n_stations'])}</b>",
        f"Chargers installed:      <b>{_fmt(row['n_chargers'])}</b>  "
        f"(avg {_fmt(row['avg_chargers_per_station'], 2)}/station)",
        f"Installed capacity:      <b>{_fmt(row['total_kw'])} kW</b>",
        "",
        f"Corridor gaps required:  <b>≥ {min_cov} / {TOTAL_GAPS}</b>  (across 9 corridors)",
        "",
        f"Driver stop time W:      {_fmt(row['total_w_min'])} min",
        f"Queue wait Wq:           {_fmt(row['total_wq_min'])} min",
        f"Optimality gap:          {_fmt(row['gap_pct'], 3)}%",
    ]
    return "<br>".join(lines)


# ── Main builder ──────────────────────────────────────────────────────────────

def build_payload(summary_df, corridor_traces):
    valid = summary_df.dropna(subset=["phi"]).copy()
    if "error" in valid.columns:
        valid = valid[valid["error"].isna() | (valid["error"] == "")]
    valid = valid[valid["n_stations"] > 0].sort_values("phi").reset_index(drop=True)

    if valid.empty:
        raise RuntimeError("No valid scenarios in summary CSV.")

    phis = sorted(valid["phi"].unique().tolist())
    scenarios = {}

    for _, row in valid.iterrows():
        rpath = Path(row["results_csv"])
        if not rpath.exists():
            print(f"  Skipping missing results: {rpath}")
            continue

        built = pd.read_csv(rpath)
        built = built[built["x_built"] == 1].copy()
        if built.empty:
            continue

        phi = float(row["phi"])
        station_traces = build_station_traces(built)

        pi = phis.index(phi)
        scenarios[pi] = {
            "title": (
                f"Spain EV Charging  |  γ=0.10  |  "
                f"φ={_fmt(phi, 2)}  "
                f"({_fmt(row['n_stations'])} stations)"
            ),
            "stats_html": _stats_html(row, phi),
            "station_traces": station_traces,
        }
        n_real  = sum(1 for t in station_traces if t["name"] == "New station"
                      and t.get("lat"))
        n_synth = sum(len(t["lat"]) for t in station_traces
                      if t["name"] == "Corridor midpoint (synthetic)"
                      and t.get("lat"))
        print(f"  φ={phi}  → {len(built)} stations built  "
              f"(synthetic midpoints: {n_synth})")

    return phis, scenarios


# ── HTML writer ───────────────────────────────────────────────────────────────

def write_html(phis, scenarios, corridor_traces, out_html):
    base_layout = {
        "paper_bgcolor": "#0d1117",
        "plot_bgcolor":  "#0d1117",
        "font": {"color": "#f5f7fa"},
        "margin": {"l": 0, "r": 0, "t": 52, "b": 0},
        "height": 780,
        "mapbox": {
            "style": "carto-darkmatter",
            "center": SPAIN_CENTER,
            "zoom":   ZOOM,
        },
        "legend": {
            "bgcolor":     "rgba(8,12,18,0.82)",
            "bordercolor": "#2f3947",
            "borderwidth": 1,
            "x": 0.01, "y": 0.99,
            "xanchor": "left", "yanchor": "top",
            "font": {"size": 11},
        },
        "title": {"text": "Spain EV Charging Network", "x": 0.5},
    }

    grid = [scenarios.get(i) for i in range(len(phis))]

    # Legend labels for corridor lines — one entry per corridor
    corridor_legend_html = "".join(
        f'<span style="display:inline-block;width:22px;height:3px;'
        f'background:{CORRIDOR_COLORS.get(n,"#fff")};'
        f'vertical-align:middle;border-radius:2px;margin-right:4px"></span>'
        f'<span style="font-size:11px;color:#b7c2cf">{n}</span>&nbsp;&nbsp;'
        for n in CORRIDOR_COLORS
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spain EV Charging | Corridor φ Sweep</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top, #16202d 0%, #0d1117 58%, #070b11 100%);
      color: #f5f7fa;
      font-family: "IBM Plex Sans","Segoe UI",sans-serif;
    }}
    .wrap {{
      max-width: 1540px;
      margin: 0 auto;
      padding: 18px 18px 22px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr 390px;
      gap: 14px;
      align-items: start;
      margin-bottom: 14px;
    }}
    .card {{
      background: rgba(12,17,24,0.78);
      border: 1px solid #273344;
      border-radius: 14px;
      padding: 14px 18px;
      box-shadow: 0 16px 40px rgba(0,0,0,0.22);
      backdrop-filter: blur(8px);
    }}
    .slider-label {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 10px;
      font-size: 13px;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      color: #b7c2cf;
    }}
    .value {{ color: #ffffff; font-weight: 700; font-size: 15px; }}
    input[type="range"] {{ width: 100%; accent-color: #5fcf65; cursor: pointer; }}
    .ticks {{
      display: flex;
      justify-content: space-between;
      margin-top: 8px;
      font-size: 11px;
      color: #8ea0b4;
    }}
    #stats {{
      font-family: "IBM Plex Mono","SFMono-Regular",monospace;
      font-size: 11.5px;
      line-height: 1.7;
      white-space: pre;
    }}
    .corr-legend {{
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid #273344;
      line-height: 2;
    }}
    #plot {{
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid #273344;
      box-shadow: 0 20px 48px rgba(0,0,0,0.25);
    }}
    @media (max-width: 900px) {{
      .controls {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="controls">
      <div class="card">
        <div class="slider-label">
          <span>Corridor gap coverage &phi;
            <small style="text-transform:none;font-weight:400;font-size:11px;color:#7a8fa4">
              &mdash; fraction of each corridor's 100 km gaps that must be filled &mdash; &gamma; = 0.10 fixed
            </small>
          </span>
          <span class="value" id="phi-value"></span>
        </div>
        <input id="phi-slider" type="range" min="0" max="{len(phis)-1}" value="0" step="1">
        <div class="ticks" id="phi-ticks"></div>
        <div class="corr-legend">{corridor_legend_html}</div>
      </div>
      <div class="card">
        <div id="stats"></div>
      </div>
    </div>
    <div id="plot"></div>
  </div>

  <script>
    const PHIS          = {json.dumps(phis)};
    const GRID          = {json.dumps(grid)};
    const CORR_TRACES   = {json.dumps(corridor_traces)};
    const BASE_LAYOUT   = {json.dumps(base_layout)};

    const slider   = document.getElementById("phi-slider");
    const phiLabel = document.getElementById("phi-value");
    const statsBox = document.getElementById("stats");

    document.getElementById("phi-ticks").innerHTML =
      PHIS.map(p => `<span>${{p.toFixed(2)}}</span>`).join("");

    function render() {{
      const pi = Number(slider.value);
      const sc = GRID[pi];
      phiLabel.textContent = PHIS[pi].toFixed(2);

      if (!sc) {{
        statsBox.innerHTML = "<b>No data for this φ.</b>";
        Plotly.react("plot", CORR_TRACES, BASE_LAYOUT,
                     {{responsive:true, displaylogo:false, scrollZoom:true}});
        return;
      }}

      const layout = JSON.parse(JSON.stringify(BASE_LAYOUT));
      layout.title.text = sc.title;
      statsBox.innerHTML = sc.stats_html;

      const allTraces = [...CORR_TRACES, ...sc.station_traces];
      Plotly.react("plot", allTraces, layout,
                   {{responsive:true, displaylogo:false, scrollZoom:true}});
    }}

    slider.addEventListener("input", render);
    render();
  </script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"\nSaved → {out_html}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading corridor geometries for map lines …")
    corridor_traces = load_corridor_traces()

    print("\nBuilding per-phi station data …")
    summary_df = pd.read_csv(SUMMARY_CSV)
    phis, scenarios = build_payload(summary_df, corridor_traces)

    write_html(phis, scenarios, corridor_traces, OUT_HTML)
