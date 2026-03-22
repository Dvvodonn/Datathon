"""
Road flow choropleth + city overlay — Spain national road network (2022 AADT).

Roads colored green → yellow → red by aadt_total (log-normalized).
Cities (population > 10,000) drawn on top as white dots scaled by population,
with name labels for the largest ones.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.collections as mc
import contextily as ctx

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(__file__)
GEOJSON_ROADS  = os.path.join(HERE, "..", "download_flow_data", "imd_total_por_tramo.geojson")
GEOJSON_CITIES = os.path.join(HERE, "..", "download_cities", "filtered_cities.geojson")
OUT_PNG        = os.path.join(HERE, "spain_road_flow_cities_2022.png")

# ── Spain bounding box in EPSG:3857 ──────────────────────────────────────────
SPAIN = dict(xmin=-1_100_000, xmax=420_000, ymin=4_200_000, ymax=5_500_000)

# ── Colormap: green → yellow → red ───────────────────────────────────────────
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "flow", ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
)

# Cities above this population get a name label
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
        coords = feat["geometry"]["coordinates"]   # [lon, lat]
        props  = feat["properties"]
        x, y = _wgs84_to_3857(coords[0], coords[1])
        cities.append({
            "x": x, "y": y,
            "name": props.get("name", ""),
            "population": props.get("population", 0) or 0,
        })
    return cities


def main():
    print("Loading road segments…")
    segments = load_segments(GEOJSON_ROADS)
    print(f"  {len(segments):,} polyline paths")

    print("Loading cities…")
    cities = load_cities(GEOJSON_CITIES)
    print(f"  {len(cities):,} cities (pop > 10,000)")

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

    # ── City dot size: log-scaled by population ───────────────────────────────
    pops = np.array([c["population"] for c in cities], dtype=float)
    pop_norm = mcolors.LogNorm(vmin=pops.min(), vmax=pops.max())
    dot_sizes = 4 + 120 * pop_norm(pops)   # 4 pt² (small town) → ~124 pt² (Madrid)

    cx = [c["x"] for c in cities]
    cy = [c["y"] for c in cities]

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

    ax.add_collection(lc)

    # City dots: white outline + semi-transparent fill
    ax.scatter(cx, cy, s=dot_sizes, color="white", alpha=0.85,
               edgecolors="#cccccc", linewidths=0.4, zorder=5)

    # Labels for large cities only
    for c in cities:
        if c["population"] >= LABEL_THRESHOLD:
            ax.text(
                c["x"], c["y"] + 18_000, c["name"],
                fontsize=7, color="white", fontweight="bold",
                ha="center", va="bottom", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="#0f1117",
                          ec="none", alpha=0.55),
            )

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

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "Spain — Annual Average Daily Traffic by Road Segment  (2022)\n"
        "White dots = cities with population > 10,000",
        color="white", fontsize=14, pad=14, fontweight="bold",
    )

    plt.tight_layout(pad=0.5)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
