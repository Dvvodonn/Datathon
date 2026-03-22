"""
Step — Visualise seasonal traffic flow across Spain.

Produces:
  outputs/seasonal_flow_maps.png      — 2×2 grid, one map per season
  outputs/seasonal_flow_comparison.png — bar + violin charts
  outputs/seasonal_flow_by_road.png   — top-20 roads heatmap
  outputs/season_<NAME>_flow.csv      — per-detector CSV for each season
  outputs/season_<NAME>_flow.gpkg     — per-season GeoPackage (GIS-ready)

Usage:
  python visualize_seasons.py [--db traffic.db]
                              [--segments data/processed/matched_segments.gpkg]
                              [--year 2025]
"""
import os
import argparse
import logging
import sqlite3
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import contextily as ctx
from shapely.geometry import box

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

os.makedirs("outputs", exist_ok=True)

SEASON_META = {
    "DJF": {"label": "Winter (DJF)",   "color": "#4a90d9", "months": "Dec–Feb"},
    "MAM": {"label": "Spring (MAM)",   "color": "#5cb85c", "months": "Mar–May"},
    "JJA": {"label": "Summer (JJA)",   "color": "#e8a838", "months": "Jun–Aug"},
    "SON": {"label": "Autumn (SON)",   "color": "#c0392b", "months": "Sep–Nov"},
}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

# Spain bounding box (EPSG:4326)
SPAIN_BBOX = (-9.5, 35.8, 4.5, 43.9)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_seasonal_data(db_path, year=None):
    """Load seasonal_averages joined with matched_segments geometry."""
    con = sqlite3.connect(db_path)
    where = f"WHERE year = {year}" if year else ""
    df = pd.read_sql(f"""
        SELECT road_id, dgt_segment_id, season, year,
               avg_speed_kph, avg_flow_veh_per_hour, peak_hour_flow, sample_count
        FROM seasonal_averages
        {where}
    """, con)
    con.close()
    log.info(f"Loaded {len(df):,} seasonal_averages rows")
    return df


def load_segments(gpkg_path):
    """Load matched_segments with road geometry."""
    gdf = gpd.read_file(gpkg_path)
    gdf["road_id"] = gdf["road_id"].astype(str)
    log.info(f"Loaded {len(gdf):,} matched road segments")
    return gdf


def build_season_gdfs(df, segments_gdf):
    """Return dict season → GeoDataFrame with flow data + geometry."""
    segments_gdf = segments_gdf.copy()
    segments_gdf["road_id"] = segments_gdf["road_id"].astype(str)
    df["road_id"] = df["road_id"].astype(str)

    season_gdfs = {}
    for season in SEASON_ORDER:
        sub = df[df["season"] == season].copy()
        if sub.empty:
            log.warning(f"No data for season {season}")
            continue
        # Drop dgt_segment_id from sub to avoid _x/_y suffixes on merge
        sub_cols = ["road_id", "avg_flow_veh_per_hour", "avg_speed_kph",
                    "peak_hour_flow", "sample_count", "year"]
        merged = segments_gdf.merge(sub[sub_cols], on="road_id", how="inner")
        merged = merged.dropna(subset=["avg_flow_veh_per_hour"])
        # Drop Z coordinates for cleaner plotting
        merged["geometry"] = merged["geometry"].apply(
            lambda g: g.__class__([(x, y) for x, y, *_ in g.coords])
            if g is not None and g.geom_type == "LineString" else g
        )
        season_gdfs[season] = merged
        log.info(f"  {season}: {len(merged):,} segments with flow data")
    return season_gdfs


# ── Plot helpers ──────────────────────────────────────────────────────────────

def flow_colormap():
    """Continuous colormap: light yellow → orange → dark red (flow intensity)."""
    return plt.cm.YlOrRd


def get_flow_norm(all_gdfs, percentile_high=97):
    """Common flow normalisation across all seasons."""
    all_flows = pd.concat([g["avg_flow_veh_per_hour"] for g in all_gdfs.values()])
    vmin = 0
    vmax = np.percentile(all_flows.dropna(), percentile_high)
    return mcolors.Normalize(vmin=vmin, vmax=vmax), vmin, vmax


def add_spain_basemap(ax, crs="EPSG:3857"):
    """Add CartoDB Positron basemap."""
    try:
        ctx.add_basemap(ax, crs=crs, source=ctx.providers.CartoDB.Positron,
                        zoom=6, alpha=0.6)
    except Exception as exc:
        log.warning(f"Basemap failed (offline?): {exc}")


def clip_to_spain(gdf):
    """Clip to Spain bounding box, reproject to Web Mercator for plotting."""
    bbox_geom = box(*SPAIN_BBOX)
    spain_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")
    clipped = gpd.clip(gdf, spain_gdf)
    return clipped.to_crs("EPSG:3857")


# ── Plot 1: 2×2 Seasonal Maps ─────────────────────────────────────────────────

def plot_seasonal_maps(season_gdfs, norm, out_path):
    log.info("Plotting 2×2 seasonal maps...")
    cmap = flow_colormap()

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.patch.set_facecolor("#1a1a2e")
    axes_flat = axes.flatten()

    for i, season in enumerate(SEASON_ORDER):
        ax = axes_flat[i]
        ax.set_facecolor("#1a1a2e")
        meta = SEASON_META[season]

        if season not in season_gdfs or season_gdfs[season].empty:
            ax.text(0.5, 0.5, f"No data\n({season})",
                    ha="center", va="center", color="white", fontsize=14,
                    transform=ax.transAxes)
            ax.set_title(meta["label"], color="white", fontsize=16, fontweight="bold")
            continue

        gdf = clip_to_spain(season_gdfs[season])
        if gdf.empty:
            continue

        flows = gdf["avg_flow_veh_per_hour"].values
        colors = cmap(norm(flows))

        # Sort by flow so high-flow roads plot on top
        gdf = gdf.sort_values("avg_flow_veh_per_hour")
        flow_vals = gdf["avg_flow_veh_per_hour"].values
        line_widths = 0.4 + (flow_vals / flow_vals.max()) * 1.6

        gdf.plot(
            column="avg_flow_veh_per_hour",
            cmap=cmap,
            norm=norm,
            linewidth=line_widths,
            ax=ax,
            legend=False,
        )
        add_spain_basemap(ax, crs="EPSG:3857")

        # Stats box
        p50 = np.percentile(flow_vals, 50)
        p90 = np.percentile(flow_vals, 90)
        n = len(gdf)
        stats_text = (
            f"Detectors: {n:,}\n"
            f"Median flow: {p50:.0f} veh/h\n"
            f"P90 flow: {p90:.0f} veh/h"
        )
        ax.text(0.02, 0.03, stats_text, transform=ax.transAxes,
                fontsize=9, color="white", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#000000aa", edgecolor="none"))

        ax.set_title(
            f"{meta['label']}  ·  {meta['months']}",
            color="white", fontsize=15, fontweight="bold", pad=8,
        )
        ax.set_axis_off()

    # Shared colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.018])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Average Flow (vehicles / hour)", color="white", fontsize=12)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white")
    cbar.outline.set_edgecolor("white")

    fig.suptitle(
        "Spain Interurban Road Traffic Flow — Seasonal Averages",
        color="white", fontsize=20, fontweight="bold", y=0.97,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


# ── Plot 2: Comparison Charts ─────────────────────────────────────────────────

def plot_comparison(season_gdfs, out_path):
    log.info("Plotting seasonal comparison charts...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#f8f8f8")

    seasons_present = [s for s in SEASON_ORDER if s in season_gdfs and not season_gdfs[s].empty]
    colors = [SEASON_META[s]["color"] for s in seasons_present]
    labels = [SEASON_META[s]["label"] for s in seasons_present]

    # -- Panel A: Median flow per season (bar) --
    ax = axes[0]
    medians = [season_gdfs[s]["avg_flow_veh_per_hour"].median() for s in seasons_present]
    p25s = [season_gdfs[s]["avg_flow_veh_per_hour"].quantile(0.25) for s in seasons_present]
    p75s = [season_gdfs[s]["avg_flow_veh_per_hour"].quantile(0.75) for s in seasons_present]
    yerr_low = [max(0.0, m - p) for m, p in zip(medians, p25s)]
    yerr_high = [max(0.0, p - m) for m, p in zip(p75s, medians)]
    x = np.arange(len(seasons_present))
    bars = ax.bar(x, medians, color=colors, edgecolor="white", linewidth=1.2,
                  yerr=[yerr_low, yerr_high], capsize=6, error_kw={"elinewidth": 1.5, "ecolor": "#555"})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=10)
    ax.set_ylabel("Median Flow (veh/h)", fontsize=11)
    ax.set_title("A  Median Hourly Flow by Season\n(bars = IQR)", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # -- Panel B: Flow distribution violin --
    ax = axes[1]
    data = [season_gdfs[s]["avg_flow_veh_per_hour"].dropna().values for s in seasons_present]
    parts = ax.violinplot(data, positions=range(len(seasons_present)),
                          showmedians=True, showextrema=False)
    for i, (body, color) in enumerate(zip(parts["bodies"], colors)):
        body.set_facecolor(color)
        body.set_alpha(0.75)
        body.set_edgecolor("white")
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    ax.set_xticks(range(len(seasons_present)))
    ax.set_xticklabels(labels, fontsize=11, rotation=10)
    ax.set_ylabel("Flow (veh/h)", fontsize=11)
    ax.set_title("B  Flow Distribution by Season", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    # -- Panel C: Speed vs Flow scatter (sampled) --
    ax = axes[2]
    for season, color in zip(seasons_present, colors):
        gdf = season_gdfs[season].dropna(subset=["avg_speed_kph", "avg_flow_veh_per_hour"])
        sample = gdf.sample(min(500, len(gdf)), random_state=42)
        ax.scatter(
            sample["avg_flow_veh_per_hour"],
            sample["avg_speed_kph"],
            c=color, alpha=0.35, s=12, label=SEASON_META[season]["label"],
        )
    ax.set_xlabel("Average Flow (veh/h)", fontsize=11)
    ax.set_ylabel("Average Speed (km/h)", fontsize=11)
    ax.set_title("C  Speed vs Flow (sample per season)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, markerscale=2)
    ax.yaxis.grid(True, alpha=0.4)
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    fig.suptitle("Spain Interurban Traffic — Seasonal Flow Analysis",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


# ── Plot 3: Top-20 Roads Heatmap ──────────────────────────────────────────────

def plot_road_heatmap(season_gdfs, out_path):
    log.info("Plotting top-20 roads heatmap...")

    seasons_present = [s for s in SEASON_ORDER if s in season_gdfs and not season_gdfs[s].empty]

    # Aggregate flow per road_name per season
    records = []
    for season in seasons_present:
        gdf = season_gdfs[season].dropna(subset=["avg_flow_veh_per_hour"])
        agg = (
            gdf.groupby("road_name")["avg_flow_veh_per_hour"]
            .median()
            .reset_index()
        )
        agg["season"] = season
        records.append(agg)

    df = pd.concat(records)
    pivot = df.pivot_table(index="road_name", columns="season",
                           values="avg_flow_veh_per_hour", aggfunc="median")
    pivot = pivot.reindex(columns=SEASON_ORDER, fill_value=np.nan)

    # Select top 20 roads by mean flow across seasons
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.nlargest(20, "mean").drop(columns="mean")
    pivot = pivot.sort_values("JJA", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(
        pivot.values,
        cmap="YlOrRd",
        aspect="auto",
        vmin=0,
        vmax=np.nanpercentile(pivot.values, 95),
    )

    ax.set_xticks(range(len(SEASON_ORDER)))
    ax.set_xticklabels(
        [f"{SEASON_META[s]['label']}\n{SEASON_META[s]['months']}" for s in SEASON_ORDER],
        fontsize=12,
    )
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Annotate cells
    for r in range(len(pivot)):
        for c, season in enumerate(SEASON_ORDER):
            val = pivot.iloc[r, c]
            if not np.isnan(val):
                text_color = "white" if val > np.nanpercentile(pivot.values, 60) else "black"
                ax.text(c, r, f"{val:.0f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Median Flow (veh/h)", fontsize=11)

    ax.set_title("Top 20 Roads by Traffic Flow — Seasonal Comparison",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Road", fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


# ── CSV + GPKG exports ────────────────────────────────────────────────────────

def export_datasets(season_gdfs, year):
    """Export per-season CSV and GeoPackage files."""
    for season in SEASON_ORDER:
        if season not in season_gdfs or season_gdfs[season].empty:
            log.warning(f"  Skipping {season} (no data)")
            continue

        gdf = season_gdfs[season].copy()
        label = SEASON_META[season]["label"].replace(" ", "_").replace("(", "").replace(")", "")

        # CSV (no geometry)
        csv_path = f"outputs/season_{season.lower()}_{label.lower()}_flow.csv"
        cols = ["road_id", "road_name", "road_class",
                "avg_flow_veh_per_hour", "avg_speed_kph", "peak_hour_flow", "sample_count"]
        gdf[cols].to_csv(csv_path, index=False)
        log.info(f"  CSV: {csv_path}  ({len(gdf):,} rows)")

        # GPKG (with geometry, 2D)
        gpkg_path = f"outputs/season_{season.lower()}_{label.lower()}_flow.gpkg"
        gdf_2d = gdf.copy()
        gdf_2d["geometry"] = gdf_2d["geometry"].apply(
            lambda g: g.__class__([(x, y) for x, y, *_ in g.coords])
            if g is not None and g.geom_type == "LineString" else g
        )
        gdf_2d.to_file(gpkg_path, driver="GPKG", layer=f"flow_{season.lower()}")
        log.info(f"  GPKG: {gpkg_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise seasonal traffic flow")
    parser.add_argument("--db", default="traffic.db")
    parser.add_argument("--segments", default="data/processed/matched_segments.gpkg")
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()

    # Load data
    df = load_seasonal_data(args.db, year=args.year)
    if df.empty:
        log.error("No seasonal_averages found. Run compute_seasons.py first.")
        return

    segments = load_segments(args.segments)
    season_gdfs = build_season_gdfs(df, segments)

    if not season_gdfs:
        log.error("No season GeoDataFrames built — check data.")
        return

    # Shared flow normalisation
    norm, vmin, vmax = get_flow_norm(season_gdfs)
    log.info(f"Flow normalisation: {vmin:.0f} – {vmax:.0f} veh/h (P97)")

    # Generate plots
    plot_seasonal_maps(season_gdfs, norm, "outputs/seasonal_flow_maps.png")
    plot_comparison(season_gdfs, "outputs/seasonal_flow_comparison.png")
    plot_road_heatmap(season_gdfs, "outputs/seasonal_flow_by_road.png")

    # Export datasets
    log.info("Exporting per-season datasets...")
    export_datasets(season_gdfs, args.year)

    # Summary table
    print(f"\n{'='*65}")
    print(f"{'Season':<12} {'Detectors':>10} {'Med flow':>10} {'P90 flow':>10} {'Med speed':>10}")
    print("-" * 65)
    for season in SEASON_ORDER:
        if season not in season_gdfs or season_gdfs[season].empty:
            print(f"{SEASON_META[season]['label']:<12} {'no data':>10}")
            continue
        gdf = season_gdfs[season]
        f = gdf["avg_flow_veh_per_hour"].dropna()
        s = gdf["avg_speed_kph"].dropna()
        print(
            f"{SEASON_META[season]['label']:<12}"
            f"{len(gdf):>10,}"
            f"{f.median():>10.0f}"
            f"{f.quantile(0.90):>10.0f}"
            f"{s.median():>10.1f}"
        )
    print("=" * 65)
    print("\nOutputs:")
    for f in sorted(os.listdir("outputs")):
        size = os.path.getsize(f"outputs/{f}")
        print(f"  outputs/{f}  ({size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
