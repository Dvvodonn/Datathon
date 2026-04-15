"""
visualizations/phi_slider_map.py
=================================
Standalone Plotly HTML map with a single phi slider, gamma fixed at 0.1.

Reads datathon_master/gamma_phi_sweep.csv and per-scenario result CSVs.

Run
---
    python visualizations/phi_slider_map.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

SUMMARY_CSV  = Path("datathon_master/gamma_phi_sweep.csv")
OUT_HTML     = Path("visualizations/phi_slider_map.html")
FIXED_GAMMA  = 0.1
SPAIN_CENTER = {"lat": 40.2, "lon": -3.7}
ZOOM = 5.25

CHARGER_COLOR = "#5fcf65"


def marker_size(chargers):
    return 10 + int(chargers) * 5


def _fmt(value, decimals=0):
    if pd.isna(value):
        return "n/a"
    if decimals == 0:
        return f"{float(value):,.0f}"
    return f"{float(value):,.{decimals}f}"


def _stats_html(row):
    lines = [
        f"<b>γ = {_fmt(row['gamma'], 2)} (fixed) &nbsp;|&nbsp; φ = {_fmt(row['phi'], 2)}</b>",
        f"Stations: {_fmt(row['n_stations'])}",
        f"Chargers: {_fmt(row['n_chargers'])}  "
        f"(avg {_fmt(row['avg_chargers_per_station'], 2)}/station)",
        f"Installed capacity: {_fmt(row['total_kw'])} kW",
        f"Driver stop time W: {_fmt(row['total_w_min'])} min",
        f"Queue wait Wq: {_fmt(row['total_wq_min'])} min",
        f"Objective UB/LB: {_fmt(row['objective_ub'])} / {_fmt(row['objective_lb'])}",
        f"Gap: {_fmt(row['gap_pct'], 3)}%",
    ]
    return "<br>".join(lines)


def build_payload(summary_df):
    valid = summary_df.dropna(subset=["gamma", "phi"]).copy()
    if "error" in valid.columns:
        valid = valid[valid["error"].isna() | (valid["error"] == "")]
    valid = valid[valid["gap_pct"].lt(50)]
    valid = valid[valid["n_stations"] > 0]
    # Fix gamma = 0.1
    valid = valid[valid["gamma"] == FIXED_GAMMA].sort_values("phi").reset_index(drop=True)

    if valid.empty:
        raise RuntimeError(f"No valid scenarios found for gamma={FIXED_GAMMA}.")

    phis = sorted(valid["phi"].unique().tolist())
    scenarios = {}

    for _, row in valid.iterrows():
        rpath = Path(row["results_csv"])
        if not rpath.exists():
            print(f"Skipping missing: {rpath}")
            continue

        built = pd.read_csv(rpath)
        built = built[built["x_built"] == 1].copy()
        if built.empty:
            continue

        hover = [
            f"<b>150 kW</b><br>"
            f"Chargers: {int(r.c_built)}<br>"
            f"λ: {r.lambda_k:.2f} EV/hr<br>"
            f"Wq: {r.wq_minutes:.1f} min"
            for _, r in built.iterrows()
        ]

        trace = {
            "type": "scattermapbox",
            "mode": "markers",
            "name": "150 kW",
            "showlegend": False,
            "lat": built["lat"].tolist(),
            "lon": built["lon"].tolist(),
            "text": hover,
            "hovertemplate": "%{text}<extra></extra>",
            "marker": {
                "size": [marker_size(c) for c in built["c_built"].tolist()],
                "color": CHARGER_COLOR,
                "opacity": 0.92,
            },
        }

        pi = phis.index(float(row["phi"]))
        scenarios[pi] = {
            "title": f"Spain EV Charging | γ=0.10 (fixed) | φ={_fmt(row['phi'], 2)}",
            "stats_html": _stats_html(row),
            "traces": [trace],
        }
        print(f"  loaded φ={row['phi']}  → {len(built)} stations")

    if not scenarios:
        raise RuntimeError("No valid scenarios found.")

    return phis, scenarios


def write_html(phis, scenarios, out_html):
    base_layout = {
        "paper_bgcolor": "#0d1117",
        "plot_bgcolor":  "#0d1117",
        "font": {"color": "#f5f7fa"},
        "margin": {"l": 0, "r": 0, "t": 56, "b": 0},
        "height": 780,
        "mapbox": {
            "style": "carto-darkmatter",
            "center": SPAIN_CENTER,
            "zoom":   ZOOM,
        },
        "legend": {
            "bgcolor":     "rgba(8, 12, 18, 0.82)",
            "bordercolor": "#2f3947",
            "borderwidth": 1,
            "x": 0.99, "y": 0.99,
            "xanchor": "right", "yanchor": "top",
        },
        "title": {"text": "Spain EV Charging Network", "x": 0.5},
    }

    # Build a list indexed 0..len(phis)-1; None for missing entries
    grid = [scenarios.get(i) for i in range(len(phis))]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spain EV Charging | γ=0.10 | φ slider</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      background: radial-gradient(circle at top, #16202d 0%, #0d1117 58%, #070b11 100%);
      color: #f5f7fa;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    .wrap {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 18px 18px 22px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 16px;
      align-items: start;
      margin-bottom: 14px;
    }}
    .card {{
      background: rgba(12, 17, 24, 0.76);
      border: 1px solid #273344;
      border-radius: 16px;
      padding: 14px 16px;
      box-shadow: 0 16px 40px rgba(0,0,0,0.24);
      backdrop-filter: blur(8px);
    }}
    .label {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
      font-size: 13px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      color: #b7c2cf;
    }}
    .value {{ color: #ffffff; font-weight: 700; }}
    input[type="range"] {{ width: 100%; accent-color: #5fcf65; }}
    .ticks {{
      display: flex;
      justify-content: space-between;
      margin-top: 9px;
      font-size: 11px;
      color: #8ea0b4;
    }}
    #stats {{
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 12px;
      line-height: 1.6;
    }}
    #plot {{
      border-radius: 18px;
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
        <div class="label">
          <span>Coverage &phi; &nbsp;<small style="text-transform:none;font-weight:400">(short-gap fraction &mdash; &gamma; = 0.10 fixed)</small></span>
          <span class="value" id="phi-value"></span>
        </div>
        <input id="phi-slider" type="range" min="0" max="{len(phis)-1}" value="0" step="1">
        <div class="ticks" id="phi-ticks"></div>
      </div>
      <div class="card">
        <div id="stats"></div>
      </div>
    </div>
    <div id="plot"></div>
  </div>

  <script>
    const PHIS       = {json.dumps(phis)};
    const GRID       = {json.dumps(grid)};
    const BASE_LAYOUT = {json.dumps(base_layout)};

    const slider   = document.getElementById("phi-slider");
    const phiValue = document.getElementById("phi-value");
    const statsBox = document.getElementById("stats");

    function fmt(v, d=2) {{
      return Number(v).toLocaleString(undefined, {{minimumFractionDigits:d, maximumFractionDigits:d}});
    }}

    document.getElementById("phi-ticks").innerHTML =
      PHIS.map(p => `<span>${{fmt(p)}}</span>`).join("");

    function render() {{
      const pi = Number(slider.value);
      const sc = GRID[pi];
      phiValue.textContent = fmt(PHIS[pi]);

      if (!sc) {{
        statsBox.innerHTML = "<b>No data for this φ.</b>";
        Plotly.react("plot", [], BASE_LAYOUT, {{responsive:true, displaylogo:false, scrollZoom:true}});
        return;
      }}

      const layout = JSON.parse(JSON.stringify(BASE_LAYOUT));
      layout.title.text = sc.title;
      statsBox.innerHTML = sc.stats_html;
      Plotly.react("plot", sc.traces, layout, {{responsive:true, displaylogo:false, scrollZoom:true}});
    }}

    slider.addEventListener("input", render);
    render();
  </script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"Saved → {out_html}")


if __name__ == "__main__":
    summary_df = pd.read_csv(SUMMARY_CSV)
    phis, scenarios = build_payload(summary_df)
    write_html(phis, scenarios, OUT_HTML)
