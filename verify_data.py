from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PREVIEW_MAP_PATH = PROCESSED_DATA_DIR / "preview_map.png"

EXPECTED_FILES = {
    "roads": RAW_DATA_DIR / "hk_roads.geojson",
    "buildings": RAW_DATA_DIR / "hk_buildings.geojson",
    "greenspace": RAW_DATA_DIR / "hk_greenspace.geojson",
    "pois": RAW_DATA_DIR / "hk_pois.geojson",
    "blocks": RAW_DATA_DIR / "hk_blocks.geojson",
}

LAYER_STYLES = {
    "roads": {"color": "gray", "linewidth": 0.6, "alpha": 0.8},
    "buildings": {"color": "lightblue", "edgecolor": "none", "alpha": 0.6},
    "greenspace": {"color": "lightgreen", "edgecolor": "green", "linewidth": 0.3, "alpha": 0.7},
    "pois": {"color": "red", "markersize": 10, "alpha": 0.9},
    "blocks": {"facecolor": "none", "edgecolor": "orange", "linewidth": 1.0, "alpha": 1.0},
    "default": {"color": "black", "linewidth": 0.8, "alpha": 0.7},
}


def infer_layer_name(file_path: Path) -> str:
    """Infer a logical layer name from a GeoJSON filename."""
    stem = file_path.stem.lower()
    if "road" in stem:
        return "roads"
    if "build" in stem:
        return "buildings"
    if "green" in stem or "park" in stem:
        return "greenspace"
    if "poi" in stem or "supermarket" in stem or "transit" in stem:
        return "pois"
    if "block" in stem:
        return "blocks"
    return stem


def summarize_layer(file_path: Path, gdf: gpd.GeoDataFrame) -> None:
    """Print summary statistics and sample rows for a layer."""
    geometry_types = ", ".join(sorted(str(geom_type) for geom_type in gdf.geom_type.dropna().unique()))
    bounds = tuple(round(value, 6) for value in gdf.total_bounds) if not gdf.empty else None

    print(f"\nFile: {file_path.name}")
    print(f"Geometry type: {geometry_types or 'No geometry'}")
    print(f"Feature count: {len(gdf)}")
    print(f"CRS: {gdf.crs}")
    print(f"Bounding box: {bounds}")
    print("First 2 rows:")
    print(gdf.head(2).to_string())


def plot_layer(ax: plt.Axes, layer_name: str, gdf: gpd.GeoDataFrame) -> None:
    """Plot a layer using the requested style settings."""
    if gdf.empty:
        print(f"Warning: {layer_name} is empty and was skipped in the plot.")
        return

    style = LAYER_STYLES.get(layer_name, LAYER_STYLES["default"])

    if layer_name == "pois":
        gdf.plot(ax=ax, color=style["color"], markersize=style["markersize"], alpha=style["alpha"])
    elif layer_name == "blocks":
        gdf.plot(
            ax=ax,
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
        )
    elif layer_name == "buildings":
        gdf.plot(ax=ax, color=style["color"], edgecolor=style["edgecolor"], alpha=style["alpha"])
    elif layer_name == "greenspace":
        gdf.plot(
            ax=ax,
            color=style["color"],
            edgecolor=style["edgecolor"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
        )
    else:
        gdf.plot(ax=ax, color=style["color"], linewidth=style["linewidth"], alpha=style["alpha"])


def main() -> None:
    """Load and verify raw GeoJSON layers, then create a preview plot."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for layer_name, expected_path in EXPECTED_FILES.items():
        if not expected_path.exists():
            print(f"Warning: expected {layer_name} file is missing: {expected_path}")

    geojson_files = sorted(RAW_DATA_DIR.glob("*.geojson"))
    if not geojson_files:
        print(f"Warning: no GeoJSON files found in {RAW_DATA_DIR}")
        return

    loaded_layers: list[tuple[str, Path, gpd.GeoDataFrame]] = []

    for file_path in geojson_files:
        try:
            gdf = gpd.read_file(file_path)
            summarize_layer(file_path, gdf)
            loaded_layers.append((infer_layer_name(file_path), file_path, gdf))
        except Exception as exc:
            print(f"Warning: failed to load {file_path.name}: {exc}")

    if not loaded_layers:
        print("Warning: no layers were loaded successfully, so no preview map was created.")
        return

    fig, ax = plt.subplots(figsize=(12, 12))
    plotted_any = False

    for layer_name in ["greenspace", "buildings", "roads", "blocks", "pois"]:
        matching_layers = [item for item in loaded_layers if item[0] == layer_name]
        for _, file_path, gdf in matching_layers:
            try:
                plot_layer(ax, layer_name, gdf)
                plotted_any = plotted_any or not gdf.empty
            except Exception as exc:
                print(f"Warning: failed to plot {file_path.name}: {exc}")

    other_layers = [item for item in loaded_layers if item[0] not in {"greenspace", "buildings", "roads", "blocks", "pois"}]
    for layer_name, file_path, gdf in other_layers:
        try:
            plot_layer(ax, "default", gdf)
            plotted_any = plotted_any or not gdf.empty
        except Exception as exc:
            print(f"Warning: failed to plot {file_path.name}: {exc}")

    if not plotted_any:
        print("Warning: no non-empty layers were available to plot.")
        plt.close(fig)
        return

    ax.set_title("Urban Renewal Input Data Preview")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(PREVIEW_MAP_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved preview map to {PREVIEW_MAP_PATH}")


if __name__ == "__main__":
    main()
