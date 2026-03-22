"""
Map of Spain showing:
  - Exact city polygons (pop > 10,000) in blue
  - EV chargers outside those cities as red dots
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.collections as mc
import contextily as ctx
import geopandas as gpd

HERE          = os.path.dirname(__file__)
CHARGERS      = os.path.join(HERE, "..", "data", "processed", "chargers_outside_cities.csv")
BOUNDARIES    = os.path.join(HERE, "..", "data", "raw", "spain_municipal_boundaries.geojson")
CITIES_GJ     = os.path.join(HERE, "..", "download_cities", "filtered_cities.geojson")
GEOJSON_ROADS = os.path.join(HERE, "..", "download_flow_data", "imd_total_por_tramo.geojson")
GPKG_ALL      = os.path.join(HERE, "..", "data", "processed", "road_routes",
                              "spain_primary_interurban_arteries.gpkg")
OUT_PNG       = os.path.join(HERE, "spain_chargers_outside_cities.png")

# Only load autopistas (1001), autovías (1002), multicarril (1005) for the base layer
# Skipping clase 1003 (397k conventional road segments) to keep render time reasonable
BASE_ROAD_CLASSES = {"1001", "1002", "1005"}

ROAD_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "flow", ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
)

SPAIN = dict(xmin=-1_100_000, xmax=420_000, ymin=4_200_000, ymax=5_500_000)
PROJ_CRS = "EPSG:25830"


def _wgs84_to_3857(lon, lat):
    x = lon * 20037508.342789244 / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    y = y * 20037508.342789244 / 180.0
    return x, y


def load_road_segments(path):
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


def load_base_roads(path):
    print("Loading base road network…")
    gdf = gpd.read_file(path)
    gdf = gdf[gdf["clase"].astype(str).isin(BASE_ROAD_CLASSES)].to_crs("EPSG:3857")
    print(f"  {len(gdf):,} base road segments (autopistas/autovías/multicarril)")
    return gdf


def main():
    # ── Load base road network (no flow data, dim grey) ───────────────────────
    base_roads = load_base_roads(GPKG_ALL)

    # ── Load AADT-coloured road segments ─────────────────────────────────────
    print("Loading AADT road segments…")
    segments = load_road_segments(GEOJSON_ROADS)
    aadt_vals = np.clip([s[1] for s in segments], 1, None)
    vmin = np.percentile(aadt_vals, 5)
    vmax = np.percentile(aadt_vals, 99)
    norm = mcolors.LogNorm(vmin=max(vmin, 1), vmax=vmax)
    segments.sort(key=lambda s: s[1])
    road_lc = mc.LineCollection(
        [s[0] for s in segments],
        colors=[ROAD_CMAP(norm(s[1])) for s in segments],
        linewidths=[0.3 + 1.4 * norm(s[1]) for s in segments],
        zorder=3,
    )
    print(f"  {len(segments):,} road paths")

    # ── Load municipal polygons ───────────────────────────────────────────────
    print("Loading municipal boundaries…")
    munis = gpd.read_file(BOUNDARIES)
    munis = munis[munis["nationallevelname"] == "Municipio"].copy().to_crs(PROJ_CRS)

    # ── Identify city polygons spatially ─────────────────────────────────────
    cities = gpd.read_file(CITIES_GJ).to_crs(PROJ_CRS)
    join = gpd.sjoin(cities[["geometry"]], munis[["nameunit", "geometry"]],
                     how="inner", predicate="within")
    city_munis = munis.loc[list(set(join["index_right"]))].copy().to_crs("EPSG:3857")
    print(f"  {len(city_munis):,} city polygons")

    # ── Load chargers ─────────────────────────────────────────────────────────
    df = pd.read_csv(CHARGERS)
    xs, ys = zip(*[_wgs84_to_3857(r.longitude, r.latitude) for r in df.itertuples()])
    print(f"  {len(df):,} chargers outside cities")

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

    # Base road network — dim grey underneath
    base_roads.plot(ax=ax, color="#ff69b4", linewidth=0.4, alpha=0.6, zorder=2)

    # AADT roads — green → red on top
    ax.add_collection(road_lc)

    # City polygons — semi-transparent blue fill, white edge
    city_munis.plot(ax=ax, facecolor="#1a6fb5", edgecolor="white",
                    linewidth=0.3, alpha=0.45, zorder=4)

    # Chargers outside cities — red dots on top
    ax.scatter(xs, ys, s=8, color="#ff3333", alpha=0.85, edgecolors="none", zorder=5)

    # ── Road colorbar ─────────────────────────────────────────────────────────
    sm = mcm.ScalarMappable(cmap=ROAD_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.018, pad=0.01, aspect=35)
    cbar.set_label("AADT — vehicles / day (2022)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    cbar.outline.set_edgecolor("white")
    tick_vals = [t for t in [500, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
                 if vmin <= t <= vmax]
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{t:,}" for t in tick_vals])

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="#1a6fb5", edgecolor="white", alpha=0.6,
                       label=f"City boundary (pop > 10k, {len(city_munis):,})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff3333",
                   markersize=6, linestyle="none",
                   label=f"EV charger outside city ({len(df):,})"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", framealpha=0.3,
              facecolor="#0f1117", edgecolor="white", labelcolor="white", fontsize=9)

    ax.set_title(
        "Spain — Road Traffic, City Boundaries & EV Chargers Outside Cities  (2022)",
        color="white", fontsize=14, pad=14, fontweight="bold",
    )

    plt.tight_layout(pad=0.5)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
