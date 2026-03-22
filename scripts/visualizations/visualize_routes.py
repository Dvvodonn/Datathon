#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_INPUT = Path(
    "data/raw/road_routes/IGR_RT_Espana_pormodos_gpkg/"
    "IGR_RT_Espana_por_modos_gpkg_RT_VIARIA_CARRETERAS/rt_viaria.gpkg"
)
DEFAULT_LAYER = "rt_tramo_vial"
DEFAULT_OUTPUT = Path("data/raw/road_routes/spain_road_network_map.png")
DEFAULT_FILTERED_OUTPUT = Path("data/processed/road_routes/spain_major_roads.gpkg")


PREFERRED_TYPE_COLUMNS = [
    "clased",
    "tipo_viald",
    "tipo_via",
    "tipo",
    "clase",
    "categoria",
    "jerarquia",
    "funcion",
    "funcional",
    "class",
    "road_type",
]

PREFERRED_NAME_COLUMNS = [
    "nombre",
    "denominacion",
    "id_via",
    "codigo",
    "carretera",
    "rotulo",
    "name",
]

PREFERRED_ORDER_COLUMNS = [
    "ordend",
    "orden",
    "jerarquia",
    "categoria",
]

INCLUDED_HIGH_CAPACITY_TYPES = {
    "autopista libre / autovía",
    "autopista libre / autovia",
    "autopista de peaje",
    "carretera multicarril",
}

CONDITIONAL_TYPE = "carretera convencional"

TARGET_NAME_PREFIXES = (
    "AP-",
    "A-",
    "N-",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect, filter, and plot Spain's road network from a GeoPackage."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to rt_viaria geopackage (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--layer",
        default=DEFAULT_LAYER,
        help=f"GeoPackage layer name (default: {DEFAULT_LAYER}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--filtered-output",
        type=Path,
        default=DEFAULT_FILTERED_OUTPUT,
        help=(
            "Output GeoPackage path for filtered major roads "
            f"(default: {DEFAULT_FILTERED_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=0.20,
        help="Road line width (default: 0.20).",
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
        help="Show the map window in addition to saving the image.",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print all GeoPackage layers and exit.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help=(
            "Print columns and sample unique values for non-geometry fields before "
            "plotting."
        ),
    )
    parser.add_argument(
        "--sample-values",
        type=int,
        default=10,
        help="How many sample unique values to print per column (default: 10).",
    )
    parser.add_argument(
        "--major-roads-only",
        action="store_true",
        default=True,
        help=(
            "Keep only the primary interurban arteries needed for the project: "
            "autopista libre/autovía, autopista de peaje, carretera multicarril, "
            "and carretera convencional only when its road code starts with AP-, A-, or N-."
        ),
    )
    parser.add_argument(
        "--save-filtered",
        action="store_true",
        help="Save the filtered roads to a new GeoPackage.",
    )
    parser.add_argument(
        "--print-head",
        type=int,
        default=5,
        help="How many rows to print from the attribute table (default: 5).",
    )
    return parser.parse_args()



def _list_layers(gpkg_path: Path, gpd) -> list[str]:
    try:
        layers_df = gpd.list_layers(gpkg_path)
        return layers_df["name"].tolist()
    except Exception:
        return []



def _print_layers(gpkg_path: Path, gpd) -> None:
    layers = _list_layers(gpkg_path, gpd)
    if not layers:
        print("No layers found or unable to list layers.")
        return
    print("Layers in GeoPackage:")
    for layer in layers:
        print(f"- {layer}")



def _print_description(roads, sample_values: int, print_head: int) -> None:
    non_geom_cols = [c for c in roads.columns if c != "geometry"]

    print("\nColumns:")
    for col in roads.columns:
        print(f"- {col}")

    print(f"\nCRS: {roads.crs}")
    print(f"Rows: {len(roads)}")
    print(f"Geometry types: {sorted(roads.geom_type.dropna().unique().tolist())}")

    if non_geom_cols:
        print(f"\nFirst {print_head} rows (without geometry):")
        print(roads[non_geom_cols].head(print_head).to_string())

    print("\nSample unique values by column:")
    for col in non_geom_cols:
        values = roads[col].dropna()
        if values.empty:
            print(f"\n=== {col} ===")
            print("<all values missing>")
            continue

        try:
            sample = values.astype(str).unique()[:sample_values]
        except Exception:
            sample = values.head(sample_values).astype(str).tolist()

        print(f"\n=== {col} ===")
        for val in sample:
            print(val)



def _first_matching_column(columns: list[str], preferred: list[str]) -> str | None:
    lower_to_original = {col.lower(): col for col in columns}
    for candidate in preferred:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    for col in columns:
        lowered = col.lower()
        if any(token in lowered for token in preferred):
            return col
    return None



def _filter_major_roads(roads):
    columns = [c for c in roads.columns if c != "geometry"]
    type_col = _first_matching_column(columns, PREFERRED_TYPE_COLUMNS)
    name_col = _first_matching_column(columns, PREFERRED_NAME_COLUMNS)
    order_col = _first_matching_column(columns, PREFERRED_ORDER_COLUMNS)

    if type_col is None:
        raise ValueError("Could not identify a usable road-type column for filtering.")

    type_values = roads[type_col].fillna("").astype(str).str.strip().str.lower()
    print(f"Using type column for filtering: {type_col}")

    high_capacity_mask = type_values.isin(INCLUDED_HIGH_CAPACITY_TYPES)
    conventional_mask = type_values.eq(CONDITIONAL_TYPE)

    if name_col is not None:
        name_values = roads[name_col].fillna("").astype(str).str.strip().str.upper()
        name_mask = name_values.str.startswith(TARGET_NAME_PREFIXES)
        print(f"Using name/code column for filtering: {name_col}")
    else:
        name_mask = False
        print("No obvious road-name/code column found; name-based filtering disabled.")

    if order_col is not None:
        order_values = roads[order_col].fillna("").astype(str).str.strip().str.lower()
        order_mask = order_values.isin({"primer orden", "principal"})
        print(f"Using order column for filtering: {order_col}")
    else:
        order_mask = False
        print("No obvious road-order column found; order-based filtering disabled.")

    major_conventional_mask = name_mask | order_mask
    mask = high_capacity_mask | (conventional_mask & major_conventional_mask)

    filtered = roads[mask].copy()
    print(f"Filtered major roads: {len(filtered)} of {len(roads)} rows kept.")
    return filtered, type_col, name_col



def main() -> int:
    args = parse_args()

    try:
        import geopandas as gpd
    except ImportError:
        print(
            "Missing dependency: geopandas. Install with:\n"
            "  pip install geopandas",
            file=sys.stderr,
        )
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Missing dependency: matplotlib. Install with:\n"
            "  pip install matplotlib",
            file=sys.stderr,
        )
        return 1

    input_path = args.input
    output_path = args.output
    filtered_output_path = args.filtered_output

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    if args.list_layers:
        _print_layers(input_path, gpd)
        return 0

    try:
        roads = gpd.read_file(input_path, layer=args.layer)
    except Exception as exc:
        available_layers = _list_layers(input_path, gpd)
        layer_msg = (
            f" Available layers: {', '.join(available_layers)}."
            if available_layers
            else ""
        )
        print(
            f"Failed to read layer '{args.layer}' from {input_path}.{layer_msg}\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 1

    roads = roads[roads.geometry.notna()].copy()
    if roads.empty:
        print("No geometries found in the selected layer.", file=sys.stderr)
        return 1

    roads = roads[roads.geom_type.isin({"LineString", "MultiLineString"})].copy()
    if roads.empty:
        print("Selected layer does not contain line geometries.", file=sys.stderr)
        return 1

    if args.describe:
        _print_description(roads, sample_values=args.sample_values, print_head=args.print_head)

    plot_roads = roads
    title_suffix = ""

    try:
        plot_roads, _, _ = _filter_major_roads(roads)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if plot_roads.empty:
        print("Filtering removed all rows.", file=sys.stderr)
        return 1

    title_suffix = " - Primary Interurban Arteries"

    if args.save_filtered:
        filtered_output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_roads.to_file(
            filtered_output_path,
            layer="major_roads",
            driver="GPKG",
        )
        print(f"Saved filtered GeoPackage: {filtered_output_path}")

    if plot_roads.crs is not None:
        plot_roads = plot_roads.to_crs(epsg=4326)

    fig, ax = plt.subplots(figsize=(11, 11), dpi=args.dpi)
    ax.set_facecolor("#f8fbfd")

    plot_roads.plot(
        ax=ax,
        color="#155e75",
        linewidth=args.linewidth,
        alpha=0.9,
    )

    minx, miny, maxx, maxy = plot_roads.total_bounds
    padx = (maxx - minx) * 0.02
    pady = (maxy - miny) * 0.02
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)
    ax.set_title(f"Spain Road Network (rt_viaria){title_suffix}", fontsize=14, pad=14)
    ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved map: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
