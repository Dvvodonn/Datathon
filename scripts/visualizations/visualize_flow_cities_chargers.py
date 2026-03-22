"""
Road flow choropleth + city overlay + EV charger locations — Spain (2022).

Layers (bottom → top):
  1. CartoDB DarkMatter basemap
  2. Road segments colored green → yellow → red by AADT (log-normalized)
  3. Cities (pop > 10,000) as white dots scaled by population, labeled above 200k
  4. EV charger sites as small cyan markers
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.collections as mc
import contextily as ctx

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE           = os.path.dirname(__file__)
GEOJSON_ROADS  = os.path.join(HERE, "..", "download_flow_data", "imd_total_por_tramo.geojson")
GEOJSON_CITIES = os.path.join(HERE, "..", "download_cities", "filtered_cities.geojson")
CSV_CHARGERS   = os.path.join(HERE, "..", "data", "raw", "chargers", "sites.csv")
OUT_PNG        = os.path.join(HERE, "spain_flow_cities_chargers_2022.png")

# ── Spain bounding box in EPSG:3857 ──────────────────────────────────────────
SPAIN = dict(xmin=-1_100_000, xmax=420_000, ymin=4_200_000, ymax=5_500_000)

CMAP = mcolors.LinearSegmentedColormap.from_list(
    "flow", ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
)

LABEL_THRESHOLD = 200_000


def _wgs84_to_3857(lon, lat):
    x = lon * 20037508.342789244 / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    y = y * 20037508.342789244 / 180.0
    return x, y


def load_segments(path):
    with open(path, encoding="utf-8") as f:
        fc = json.load(f)
    segments = []
    for feat in fc["features"]:
        aadt = feat["properties"].get("aadt_total")
        geom = feat.get("geometry")
        if aadt is None or geom is None:
            continue
        for path in geom["coordinates"]:
            xy = [_wgs84_to_3857(lon, lat) for lon, lat in path]
            if len(xy) >= 2:
                segments.append((xy, aadt))
    return segments


def load_cities(path):
    with open(path, encoding="utf-8") as f:
        fc = json.load(f)
    cities = []
    for feat in fc["features"]:
        lon, lat = feat["geometry"]["coordinates"]
        x, y = _wgs84_to_3857(lon, lat)
        cities.append({
            "x": x, "y": y,
            "name": feat["properties"].get("name", ""),
            "population": feat["properties"].get("population", 0) or 0,
        })
    return cities


def load_chargers(path):
    df = pd.read_csv(path, usecols=["latitude", "longitude"]).dropna()
    xs, ys = zip(*[_wgs84_to_3857(row.longitude, row.latitude)
                   for row in df.itertuples()])
    return list(xs), list(ys)


def main():
    print("Loading road segments…")
    segments = load_segments(GEOJSON_ROADS)
    print(f"  {len(segments):,} polyline paths")

    print("Loading cities…")
    cities = load_cities(GEOJSON_CITIES)
    print(f"  {len(cities):,} cities (pop > 10,000)")

    print("Loading EV chargers…")
    cx_ch, cy_ch = load_chargers(CSV_CHARGERS)
    print(f"  {len(cx_ch):,} charger sites")

    # ── Road colour/width ─────────────────────────────────────────────────────
    aadt_vals = np.clip([s[1] for s in segments], 1, None)
    vmin = np.percentile(aadt_vals, 5)
    vmax = np.percentile(aadt_vals, 99)
    norm = mcolors.LogNorm(vmin=max(vmin, 1), vmax=vmax)

    segments.sort(key=lambda s: s[1])
    lines  = [s[0] for s in segments]
    colors = [CMAP(norm(s[1])) for s in segments]
    widths = [0.3 + 1.4 * norm(s[1]) for s in segments]
    lc = mc.LineCollection(lines, colors=colors, linewidths=widths, zorder=3)

    # ── City dot sizes ────────────────────────────────────────────────────────
    pops     = np.array([c["population"] for c in cities], dtype=float)
    pop_norm = mcolors.LogNorm(vmin=pops.min(), vmax=pops.max())
    dot_sizes = 4 + 120 * pop_norm(pops)
    cx_ci = [c["x"] for c in cities]
    cy_ci = [c["y"] for c in cities]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 14), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    ax.set_xlim(SPAIN["xmin"], SPAIN["xmax"])
    ax.set_ylim(SPAIN["ymin"], SPAIN["ymax"])
    ax.set_aspect("equal")
    ax.axis("off")

    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.DarkMatterNoLabels,
        crs="EPSG:3857",
        reset_extent=False,
        zoom=6,
    )

    # Roads
    ax.add_collection(lc)

    # Cities
    ax.scatter(cx_ci, cy_ci, s=dot_sizes, color="white", alpha=0.85,
               edgecolors="#cccccc", linewidths=0.4, zorder=5)

    for c in cities:
        if c["population"] >= LABEL_THRESHOLD:
            ax.text(
                c["x"], c["y"] + 18_000, c["name"],
                fontsize=7, color="white", fontweight="bold",
                ha="center", va="bottom", zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="#0f1117", ec="none", alpha=0.55),
            )

    # EV chargers
    ax.scatter(cx_ch, cy_ch, s=4, color="#00e5ff", alpha=0.6,
               edgecolors="none", zorder=6, label="EV charger")

    # ── Colorbar (roads) ──────────────────────────────────────────────────────
    sm = mcm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.018, pad=0.01, aspect=35)
    cbar.set_label("AADT — vehicles / day (2022)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=9)
    cbar.outline.set_edgecolor("white")

    tick_vals = [t for t in [500, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
                 if vmin <= t <= vmax]
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{t:,}" for t in tick_vals])

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.scatter([], [], s=40, color="white", label="City (pop > 10k)")
    ax.scatter([], [], s=10, color="#00e5ff", alpha=0.8, label="EV charger site")
    ax.legend(loc="lower left", framealpha=0.3, facecolor="#0f1117",
              edgecolor="white", labelcolor="white", fontsize=9)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "Spain — AADT by Road Segment, Cities & EV Charger Sites  (2022)",
        color="white", fontsize=14, pad=14, fontweight="bold",
    )

    plt.tight_layout(pad=0.5)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
