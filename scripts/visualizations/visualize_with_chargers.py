#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROAD_ROUTES_DIR = Path("data/processed/road_routes")
DEFAULT_CHARGERS_CSV = Path("data/raw/chargers/sites.csv")
DEFAULT_OUTPUT = Path("data/raw/chargers/spain_roads_with_chargers.png")


def _detect_roads_file() -> Path:
    geojson_candidates = sorted(ROAD_ROUTES_DIR.glob("*.geojson")) + sorted(
        ROAD_ROUTES_DIR.glob("*.json")
    )
    if geojson_candidates:
        return geojson_candidates[0]

    gpkg_candidates = sorted(ROAD_ROUTES_DIR.glob("*.gpkg"))
    if gpkg_candidates:
        return gpkg_candidates[0]

    return ROAD_ROUTES_DIR / "spain_primary_interurban_arteries.geojson"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Spain roads from data/processed/road_routes and overlay chargers "
            "from a CSV as small red dots."
        )
    )
    parser.add_argument(
        "--roads",
        type=Path,
        default=_detect_roads_file(),
        help=(
            "Roads file path. GeoJSON is preferred, but GPKG also works "
            f"(default: {_detect_roads_file()})."
        ),
    )
    parser.add_argument(
        "--roads-layer",
        default=None,
        help=(
            "Layer name for GeoPackage inputs. If omitted, the first layer is used."
        ),
    )
    parser.add_argument(
        "--chargers-csv",
        type=Path,
        default=DEFAULT_CHARGERS_CSV,
        help=f"CSV file with charger coordinates (default: {DEFAULT_CHARGERS_CSV}).",
    )
    parser.add_argument(
        "--lat-column",
        default="latitude",
        help="Latitude column in chargers CSV (default: latitude).",
    )
    parser.add_argument(
        "--lon-column",
        default="longitude",
        help="Longitude column in chargers CSV (default: longitude).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--road-linewidth",
        type=float,
        default=0.40,
        help="Road line width (default: 0.40).",
    )
    parser.add_argument(
        "--charger-size",
        type=float,
        default=5.0,
        help="Marker size for charger dots (default: 5.0).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image DPI (default: 300).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in addition to saving it.",
    )
    return parser.parse_args()


def _read_roads(args, gpd):
    roads_path = args.roads
    if not roads_path.exists():
        raise FileNotFoundError(f"Roads file not found: {roads_path}")

    if roads_path.suffix.lower() == ".gpkg":
        layer = args.roads_layer
        if layer is None:
            try:
                layers = gpd.list_layers(roads_path)
                if layers.empty:
                    raise ValueError("No layers found in GeoPackage.")
                layer = layers.iloc[0]["name"]
            except Exception as exc:
                raise ValueError(
                    "Unable to determine GeoPackage layer. "
                    "Provide --roads-layer explicitly."
                ) from exc
        roads = gpd.read_file(roads_path, layer=layer)
    else:
        roads = gpd.read_file(roads_path)

    roads = roads[roads.geometry.notna()].copy()
    roads = roads[roads.geom_type.isin({"LineString", "MultiLineString"})].copy()
    if roads.empty:
        raise ValueError("Roads data does not contain line geometries.")
    return roads


def _read_chargers(args, pd, gpd):
    if not args.chargers_csv.exists():
        raise FileNotFoundError(f"Chargers CSV not found: {args.chargers_csv}")

    usecols = [args.lat_column, args.lon_column]
    chargers = pd.read_csv(args.chargers_csv, usecols=usecols)
    chargers[args.lat_column] = pd.to_numeric(chargers[args.lat_column], errors="coerce")
    chargers[args.lon_column] = pd.to_numeric(chargers[args.lon_column], errors="coerce")
    chargers = chargers.dropna(subset=[args.lat_column, args.lon_column]).copy()
    if chargers.empty:
        raise ValueError("No valid charger coordinates after cleaning CSV values.")

    points = gpd.GeoDataFrame(
        chargers,
        geometry=gpd.points_from_xy(chargers[args.lon_column], chargers[args.lat_column]),
        crs="EPSG:4326",
    )
    return points


def main() -> int:
    args = parse_args()

    try:
        import geopandas as gpd
    except ImportError:
        print("Missing dependency: geopandas (pip install geopandas)", file=sys.stderr)
        return 1

    try:
        import pandas as pd
    except ImportError:
        print("Missing dependency: pandas (pip install pandas)", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Missing dependency: matplotlib (pip install matplotlib)", file=sys.stderr)
        return 1

    try:
        roads = _read_roads(args, gpd)
        chargers = _read_chargers(args, pd, gpd)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    target_crs = roads.crs if roads.crs is not None else "EPSG:4326"
    if roads.crs is not None:
        roads = roads.to_crs(target_crs)
    chargers = chargers.to_crs(target_crs)

    fig, ax = plt.subplots(figsize=(11, 11), dpi=args.dpi)
    ax.set_facecolor("#f7fafc")

    roads.plot(
        ax=ax,
        color="#1f4e79",
        linewidth=args.road_linewidth,
        alpha=0.85,
    )
    chargers.plot(
        ax=ax,
        color="red",
        markersize=args.charger_size,
        alpha=0.3,
    )

    minx, miny, maxx, maxy = roads.total_bounds
    padx = (maxx - minx) * 0.02
    pady = (maxy - miny) * 0.02
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)
    ax.set_title("Spain Road Network with EV Chargers", fontsize=14, pad=12)
    ax.set_axis_off()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved map: {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
