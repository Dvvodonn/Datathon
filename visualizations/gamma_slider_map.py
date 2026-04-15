"""
visualizations/gamma_slider_map.py
==================================
Build a standalone Plotly HTML map with two independent controls:

    gamma      : grid-connection cost per installed kW
    fixed_cost : station opening cost per selected site

The script reads the 2D sweep summary plus the per-scenario result CSVs written
by congestion/sweep_2d.py and emits a shareable HTML file.

Run
---
    python visualizations/gamma_slider_map.py
    python visualizations/gamma_slider_map.py --summary-csv congestion/outputs/gamma_fixed_cost_sweep.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


SUMMARY_CSV = Path("congestion/outputs/gamma_fixed_cost_sweep.csv")
OUT_HTML = Path("visualizations/gamma_fixed_cost_slider_map.html")
SPAIN_CENTER = {"lat": 40.2, "lon": -3.7}
ZOOM = 5.25

TIER_COLORS = {
    50: "#cfeec8",
    100: "#96d98b",
    150: "#62bf59",
    200: "#34993b",
    250: "#1f6f2a",
    350: "#0d4416",
}


def marker_size(chargers):
    return 10 + int(chargers) * 4


def _fmt_num(value, decimals=0):
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


def _stats_html(row):
    n_stations = int(row["n_stations"])
    power_dist = _decode_json_obj(row.get("power_dist", ""))
    charger_dist = _decode_json_obj(row.get("charger_dist", ""))

    power_line = ", ".join(f"{k}kW:{v}" for k, v in power_dist.items()) or "none"
    charger_line = ", ".join(f"{k}:{v}" for k, v in charger_dist.items()) or "none"

    lines = [
        f"<b>gamma = {_fmt_num(row['gamma'], 2)}</b>",
        f"<b>fixed cost = {_fmt_num(row['fixed_cost'])}</b>",
        f"Stations: {_fmt_num(row['n_stations'])}",
        f"Chargers: {_fmt_num(row['n_chargers'])} (avg {_fmt_num(row['avg_chargers_per_station'], 2)}/station)",
        f"Installed capacity: {_fmt_num(row['total_kw'])} kW",
        f"Driver stop time W: {_fmt_num(row['total_w_min'])} min",
        f"Queue wait Wq: {_fmt_num(row['total_wq_min'])} min",
        f"Opening cost total: {_fmt_num(row['total_open_cost'])}",
        f"Objective UB/LB: {_fmt_num(row['objective_ub'])} / {_fmt_num(row['objective_lb'])}",
        f"Gap: {_fmt_num(row['gap_pct'], 3)}%",
        f"Power tiers: {power_line}",
        f"Charger counts: {charger_line}",
    ]
    return "<br>".join(lines)


def _scenario_key(gamma, fixed_cost):
    fc_str = str(int(fixed_cost)) if float(fixed_cost) == int(fixed_cost) else str(fixed_cost)
    return f"{gamma}|{fc_str}"


def build_payload(summary_df):
    valid = summary_df.dropna(subset=["gamma", "fixed_cost"]).copy()
    if "error" in valid.columns:
        valid = valid[valid["error"].isna() | (valid["error"] == "")]
    valid = valid.sort_values(["fixed_cost", "gamma"])

    gammas = sorted(valid["gamma"].unique().tolist())
    fixed_costs = sorted(valid["fixed_cost"].unique().tolist())

    scenarios = {}
    kept_rows = 0

    for _, row in valid.iterrows():
        results_path = Path(row["results_csv"])
        if not results_path.exists():
            print(f"Skipping missing results file: {results_path}")
            continue

        built = pd.read_csv(results_path)
        built = built[built["x_built"] == 1].copy()

        traces = []
        for tier, color in TIER_COLORS.items():
            sub = built[built["p_built_kw"] == tier]
            hover = [
                (
                    f"<b>{tier} kW</b><br>"
                    f"Chargers: {int(r.c_built)}<br>"
                    f"lambda: {r.lambda_k:.2f} EV/hr<br>"
                    f"Wq: {r.wq_minutes:.1f} min<br>"
                    f"rho: {r.lambda_k / max(r.c_built * tier / 42.0, 1e-9):.2f}"
                )
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

        key = _scenario_key(row["gamma"], row["fixed_cost"])
        scenarios[key] = {
            "title": (
                "Spain EV Charging Network"
                f" | gamma={_fmt_num(row['gamma'], 2)}"
                f" | fixed cost={_fmt_num(row['fixed_cost'])}"
            ),
            "stats_html": _stats_html(row),
            "traces": traces,
        }
        kept_rows += 1

    if kept_rows == 0:
        raise RuntimeError("No valid scenarios found. Run congestion/sweep_2d.py first.")

    gammas = [g for g in gammas if any(_scenario_key(g, f) in scenarios for f in fixed_costs)]
    fixed_costs = [f for f in fixed_costs if any(_scenario_key(g, f) in scenarios for g in gammas)]
    return gammas, fixed_costs, scenarios


def write_html(gammas, fixed_costs, scenarios, out_html):
    legend_traces = [{
        "type": "scattermapbox",
        "mode": "markers",
        "lat": [None],
        "lon": [None],
        "name": f"{tier} kW",
        "showlegend": True,
        "marker": {"size": 12, "color": color, "opacity": 0.95},
        "hoverinfo": "skip",
    } for tier, color in TIER_COLORS.items()]

    base_layout = {
        "paper_bgcolor": "#0d1117",
        "plot_bgcolor": "#0d1117",
        "font": {"color": "#f5f7fa"},
        "margin": {"l": 0, "r": 0, "t": 56, "b": 0},
        "height": 780,
        "mapbox": {
            "style": "carto-darkmatter",
            "center": SPAIN_CENTER,
            "zoom": ZOOM,
        },
        "legend": {
            "title": {"text": "Power tier"},
            "bgcolor": "rgba(8, 12, 18, 0.82)",
            "bordercolor": "#2f3947",
            "borderwidth": 1,
            "x": 0.99,
            "y": 0.99,
            "xanchor": "right",
            "yanchor": "top",
        },
        "title": {"text": "Spain EV Charging Network", "x": 0.5},
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spain EV Charging | gamma x fixed cost</title>
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
      grid-template-columns: minmax(280px, 1fr) minmax(280px, 1fr) 360px;
      gap: 16px;
      align-items: start;
      margin-bottom: 14px;
    }}
    .card {{
      background: rgba(12, 17, 24, 0.76);
      border: 1px solid #273344;
      border-radius: 16px;
      padding: 14px 16px;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.24);
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
    .value {{
      color: #ffffff;
      font-weight: 700;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: #5fcf65;
    }}
    .ticks {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-top: 9px;
      font-size: 11px;
      color: #8ea0b4;
    }}
    #stats {{
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 12px;
      line-height: 1.55;
      white-space: normal;
    }}
    #plot {{
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid #273344;
      box-shadow: 0 20px 48px rgba(0, 0, 0, 0.25);
    }}
    @media (max-width: 1200px) {{
      .controls {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="controls">
      <div class="card">
        <div class="label">
          <span>Grid Cost gamma</span>
          <span class="value" id="gamma-value"></span>
        </div>
        <input id="gamma-slider" type="range" min="0" max="{len(gammas) - 1}" value="0" step="1">
        <div class="ticks" id="gamma-ticks"></div>
      </div>
      <div class="card">
        <div class="label">
          <span>Fixed Station Cost</span>
          <span class="value" id="fixed-value"></span>
        </div>
        <input id="fixed-slider" type="range" min="0" max="{len(fixed_costs) - 1}" value="0" step="1">
        <div class="ticks" id="fixed-ticks"></div>
      </div>
      <div class="card">
        <div id="stats"></div>
      </div>
    </div>
    <div id="plot"></div>
  </div>

  <script>
    const GAMMAS = {json.dumps(gammas)};
    const FIXED_COSTS = {json.dumps(fixed_costs)};
    const SCENARIOS = {json.dumps(scenarios)};
    const LEGEND_TRACES = {json.dumps(legend_traces)};
    const BASE_LAYOUT = {json.dumps(base_layout)};

    const gammaSlider = document.getElementById("gamma-slider");
    const fixedSlider = document.getElementById("fixed-slider");
    const gammaValue = document.getElementById("gamma-value");
    const fixedValue = document.getElementById("fixed-value");
    const statsBox = document.getElementById("stats");

    function fmtNumber(value, digits = 0) {{
      return Number(value).toLocaleString(undefined, {{
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
      }});
    }}

    function setTicks(containerId, values, digits = 0) {{
      const node = document.getElementById(containerId);
      node.innerHTML = values.map(v => `<span>${{fmtNumber(v, digits)}}</span>`).join("");
    }}

    function scenarioKey(gamma, fixedCost) {{
      return `${{gamma}}|${{fixedCost}}`;
    }}

    function render() {{
      const gamma = GAMMAS[Number(gammaSlider.value)];
      const fixedCost = FIXED_COSTS[Number(fixedSlider.value)];
      const scenario = SCENARIOS[scenarioKey(gamma, fixedCost)];

      gammaValue.textContent = fmtNumber(gamma, 2);
      fixedValue.textContent = fmtNumber(fixedCost);

      if (!scenario) {{
        statsBox.innerHTML = "<b>No scenario for this parameter pair.</b>";
        Plotly.react("plot", LEGEND_TRACES, BASE_LAYOUT, {{
          responsive: true,
          displaylogo: false,
          scrollZoom: true
        }});
        return;
      }}

      const layout = JSON.parse(JSON.stringify(BASE_LAYOUT));
      layout.title.text = scenario.title;

      statsBox.innerHTML = scenario.stats_html;
      Plotly.react("plot", scenario.traces.concat(LEGEND_TRACES), layout, {{
        responsive: true,
        displaylogo: false,
        scrollZoom: true
      }});
    }}

    setTicks("gamma-ticks", GAMMAS, 2);
    setTicks("fixed-ticks", FIXED_COSTS, 0);
    gammaSlider.addEventListener("input", render);
    fixedSlider.addEventListener("input", render);
    render();
  </script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"Saved -> {out_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive gamma x fixed-cost map")
    parser.add_argument("--summary-csv", type=str, default=str(SUMMARY_CSV),
                        help="2D sweep summary CSV")
    parser.add_argument("--out-html", type=str, default=str(OUT_HTML),
                        help="Output HTML path")
    args = parser.parse_args()

    summary_df = pd.read_csv(args.summary_csv)
    gammas, fixed_costs, scenarios = build_payload(summary_df)
    write_html(gammas, fixed_costs, scenarios, Path(args.out_html))
