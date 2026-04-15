"""
visualizations/gamma_frontier_slider_map.py
============================================
Single-slider HTML map: gamma frontier at phi=0, variable kW tiers.

Reads congestion/outputs/gamma_frontier_v2.csv and per-scenario result CSVs.

Run
---
    python visualizations/gamma_frontier_slider_map.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

SUMMARY_CSV  = Path("congestion/outputs/gamma_frontier_v2.csv")
RESULTS_DIR  = Path("congestion/outputs")
OUT_HTML     = Path("visualizations/gamma_frontier_slider_map.html")
SPAIN_CENTER = {"lat": 40.2, "lon": -3.7}
ZOOM = 5.25

TIER_COLORS = {
    50:  "#cfeec8",
    100: "#96d98b",
    150: "#62bf59",
    200: "#34993b",
    250: "#1f6f2a",
    350: "#0d4416",
}


def marker_size(chargers):
    return 10 + int(chargers) * 4


def _fmt(value, decimals=0):
    if pd.isna(value):
        return "n/a"
    if decimals == 0:
        return f"{float(value):,.0f}"
    return f"{float(value):,.{decimals}f}"


def _decode_json_obj(raw):
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return {str(k): int(v) for k, v in data.items()}


def _results_path(gamma):
    tag = str(gamma).replace(".", "_")
    return RESULTS_DIR / f"results_congestion_v2_g{tag}.csv"


def _stats_html(row):
    power_dist   = _decode_json_obj(row.get("power_dist", ""))
    power_line   = ", ".join(f"{k}kW×{v}" for k, v in power_dist.items()) or "—"
    lines = [
        f"<b>γ = {_fmt(row['gamma'], 2)}</b> &nbsp;(φ = 0, variable kW)",
        f"Stations : {_fmt(row['n_stations'])}",
        f"Chargers : {_fmt(row['n_chargers'])}  (avg {_fmt(row['n_chargers']/max(row['n_stations'],1), 2)}/station)",
        f"Capacity : {_fmt(row['total_kw'])} kW",
        f"Stop time W : {_fmt(row['total_w_min'])} min",
        f"Queue wait Wq : {_fmt(row['total_wq_min'])} min",
        f"Obj UB / LB : {_fmt(row['objective_ub'])} / {_fmt(row['objective_lb'])}",
        f"Gap : {_fmt(row['gap_pct'], 3)}%",
        f"Power tiers : {power_line}",
    ]
    return "<br>".join(lines)


def build_payload(summary_df):
    valid = summary_df.sort_values("gamma").reset_index(drop=True)
    gammas = valid["gamma"].tolist()
    scenarios = {}

    for _, row in valid.iterrows():
        rpath = _results_path(row["gamma"])
        if not rpath.exists():
            print(f"Skipping missing: {rpath}")
            continue

        built = pd.read_csv(rpath)
        built = built[built["x_built"] == 1].copy()

        traces = []
        for tier, color in TIER_COLORS.items():
            sub = built[built["p_built_kw"] == tier]
            if sub.empty:
                continue
            hover = [
                f"<b>{tier} kW</b><br>"
                f"Chargers: {int(r.c_built)}<br>"
                f"λ: {r.lambda_k:.2f} EV/hr<br>"
                f"Wq: {r.wq_minutes:.1f} min"
                for _, r in sub.iterrows()
            ]
            traces.append({
                "type": "scattermapbox",
                "mode": "markers",
                "name": f"{tier} kW",
                "showlegend": False,
                "lat": sub["lat"].tolist(),
                "lon": sub["lon"].tolist(),
                "text": hover,
                "hovertemplate": "%{text}<extra></extra>",
                "marker": {
                    "size": [marker_size(c) for c in sub["c_built"].tolist()],
                    "color": color,
                    "opacity": 0.92,
                },
            })

        scenarios[str(row["gamma"])] = {
            "title": f"Spain EV Charging | γ={_fmt(row['gamma'], 2)} | φ=0 | variable kW",
            "stats_html": _stats_html(row),
            "traces": traces,
        }
        print(f"  γ={row['gamma']:.2f}  {len(built)} stations  tiers={sorted(built['p_built_kw'].unique().tolist())}")

    gammas = [g for g in gammas if str(g) in scenarios]
    return gammas, scenarios


def write_html(gammas, scenarios, out_html):
    legend_traces = [
        {
            "type": "scattermapbox",
            "mode": "markers",
            "lat": [None], "lon": [None],
            "name": f"{tier} kW",
            "showlegend": True,
            "marker": {"size": 12, "color": color, "opacity": 0.95},
            "hoverinfo": "skip",
        }
        for tier, color in TIER_COLORS.items()
    ]

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
            "title": {"text": "Power tier"},
            "bgcolor":     "rgba(8, 12, 18, 0.82)",
            "bordercolor": "#2f3947",
            "borderwidth": 1,
            "x": 0.99, "y": 0.99,
            "xanchor": "right", "yanchor": "top",
        },
        "title": {"text": "Spain EV Charging Network", "x": 0.5},
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spain EV Charging | γ Frontier (φ=0, variable kW)</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      background: radial-gradient(circle at top, #16202d 0%, #0d1117 58%, #070b11 100%);
      color: #f5f7fa;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 18px 18px 22px; }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr 360px;
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
      margin-bottom: 10px;
      font-size: 13px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      color: #b7c2cf;
    }}
    .value {{ color: #fff; font-weight: 700; }}
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
    @media (max-width: 900px) {{ .controls {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="controls">
      <div class="card">
        <div class="label">
          <span>Grid Cost &gamma; &nbsp;<small style="text-transform:none;font-weight:400">(φ = 0, variable kW tiers)</small></span>
          <span class="value" id="gamma-value"></span>
        </div>
        <input id="gamma-slider" type="range" min="0" max="{len(gammas)-1}" value="0" step="1">
        <div class="ticks" id="gamma-ticks"></div>
      </div>
      <div class="card"><div id="stats"></div></div>
    </div>
    <div id="plot"></div>
  </div>

  <script>
    const GAMMAS        = {json.dumps(gammas)};
    const SCENARIOS     = {json.dumps(scenarios)};
    const LEGEND_TRACES = {json.dumps(legend_traces)};
    const BASE_LAYOUT   = {json.dumps(base_layout)};

    const slider    = document.getElementById("gamma-slider");
    const gammaVal  = document.getElementById("gamma-value");
    const statsBox  = document.getElementById("stats");

    function fmt(v, d=2) {{
      return Number(v).toLocaleString(undefined, {{minimumFractionDigits:d, maximumFractionDigits:d}});
    }}

    document.getElementById("gamma-ticks").innerHTML =
      GAMMAS.map(g => `<span>${{fmt(g)}}</span>`).join("");

    function render() {{
      const gamma = GAMMAS[Number(slider.value)];
      const sc    = SCENARIOS[String(gamma)];
      gammaVal.textContent = fmt(gamma);
      if (!sc) {{
        statsBox.innerHTML = "<b>No data for this γ.</b>";
        Plotly.react("plot", LEGEND_TRACES, BASE_LAYOUT, {{responsive:true, displaylogo:false, scrollZoom:true}});
        return;
      }}
      const layout = JSON.parse(JSON.stringify(BASE_LAYOUT));
      layout.title.text = sc.title;
      statsBox.innerHTML = sc.stats_html;
      Plotly.react("plot", sc.traces.concat(LEGEND_TRACES), layout,
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
    print(f"Saved → {out_html}")


if __name__ == "__main__":
    summary_df = pd.read_csv(SUMMARY_CSV)
    gammas, scenarios = build_payload(summary_df)
    write_html(gammas, scenarios, OUT_HTML)
