"""
Road flow choropleth — Spain national road network (2022 AADT).

Each road segment is colored green → yellow → red by aadt_total.
Color scale is log-normalized to handle the wide range (~0 – 190,000 veh/day).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.collections as mc
import contextily as ctx
from shapely.geometry import shape

# ── Paths ────────────────────────────────────────────────────────────────────
GEOJSON = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "intermediate", "traffic_flow", "imd_total_por_tramo.geojson"
)
OUT_PNG = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "maps", "spain_road_flow_2022.png")

# ── Spain bounding box in EPSG:3857 (Web Mercator) ───────────────────────────
SPAIN = dict(xmin=-1_100_000, xmax=420_000, ymin=4_200_000, ymax=5_500_000)

# ── Colormap: green → yellow → red ───────────────────────────────────────────
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "flow", ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
)


def load_segments(path):
    with open(path, encoding="utf-8") as f:
        fc = json.load(f)

    segments = []   # list of (lines_in_3857, aadt)
    for feat in fc["features"]:
        aadt = feat["properties"].get("aadt_total")
        geom = feat.get("geometry")
        if aadt is None or geom is None:
            continue
        # Project each path from WGS84 lon/lat → Web Mercator (EPSG:3857)
        coords = geom["coordinates"]
        # LineString: list of points; MultiLineString or paths: list of lists
        if coords and not isinstance(coords[0][0], (list, tuple)):
            coords = [coords]  # wrap single LineString
        for path in coords:
            xy = [_wgs84_to_3857(pt[0], pt[1]) for pt in path]  # handles 2D and 3D points
            if len(xy) >= 2:
                segments.append((xy, aadt))

    return segments


def _wgs84_to_3857(lon, lat):
    """Fast inline WGS84 → EPSG:3857 conversion (no pyproj dependency)."""
    x = lon * 20037508.342789244 / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    y = y * 20037508.342789244 / 180.0
    return x, y


def main():
    print("Loading segments…")
    segments = load_segments(GEOJSON)
    print(f"  {len(segments):,} polyline paths loaded")

    aadt_vals = np.array([s[1] for s in segments], dtype=float)
    aadt_vals = np.clip(aadt_vals, 1, None)   # avoid log(0)

    # Log-normalize: low-traffic roads stay green, high-traffic glow red
    vmin = np.percentile(aadt_vals, 5)
    vmax = np.percentile(aadt_vals, 99)
    norm = mcolors.LogNorm(vmin=max(vmin, 1), vmax=vmax)

    # Sort so high-traffic roads are drawn last (on top)
    segments.sort(key=lambda s: s[1])

    # ── Build LineCollection ──────────────────────────────────────────────────
    lines = [s[0] for s in segments]
    colors = [CMAP(norm(s[1])) for s in segments]

    # Line width also scales with traffic (subtle)
    widths = [0.3 + 1.4 * norm(s[1]) for s in segments]

    lc = mc.LineCollection(lines, colors=colors, linewidths=widths, zorder=3)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 14), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    ax.set_xlim(SPAIN["xmin"], SPAIN["xmax"])
    ax.set_ylim(SPAIN["ymin"], SPAIN["ymax"])
    ax.set_aspect("equal")
    ax.axis("off")

    # Basemap (dark)
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.DarkMatterNoLabels,
        crs="EPSG:3857",
        reset_extent=False,
        zoom=6,
    )

    ax.add_collection(lc)

    # ── Colorbar ─────────────────────────────────────────────────────────────
    sm = mcm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.018, pad=0.01, aspect=35)
    cbar.set_label("AADT — vehicles / day (2022)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=9)
    cbar.outline.set_edgecolor("white")

    # Log-scale tick labels as readable integers
    tick_vals = [500, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
    tick_vals = [t for t in tick_vals if vmin <= t <= vmax]
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{t:,}" for t in tick_vals])

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "Spain — Annual Average Daily Traffic by Road Segment  (2022)",
        color="white", fontsize=15, pad=14, fontweight="bold",
    )

    plt.tight_layout(pad=0.5)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
