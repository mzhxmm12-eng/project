from pathlib import Path
import argparse

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


INPUT_PATH = Path("data/processed/districts_scored.geojson")
OUTPUT_PATH = Path("data/processed/results_preview.png")


def find_label_column(gdf: gpd.GeoDataFrame) -> str | None:
    """Pick a readable district label column for the bar chart."""
    candidates = [
        "name",
        "NAME",
        "district",
        "DISTRICT",
        "TPU_NUMBER",
        "CSDI_TPU_ID",
        "OBJECTID",
    ]
    for column in candidates:
        if column in gdf.columns:
            return column
    return None


def main() -> None:
    """Create a map + bar-chart preview of scored districts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(INPUT_PATH),
        help="Input scored GeoJSON path.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing scored districts file: {input_path}")

    gdf = gpd.read_file(input_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    sorted_gdf = gdf.sort_values("renewal_priority", ascending=False).copy()

    label_column = find_label_column(sorted_gdf)
    if label_column is None:
        sorted_gdf["district_label"] = [f"District {idx}" for idx in sorted_gdf.index]
        label_column = "district_label"
    else:
        sorted_gdf[label_column] = sorted_gdf[label_column].fillna("").astype(str)
        empty_mask = sorted_gdf[label_column].str.strip().eq("")
        sorted_gdf.loc[empty_mask, label_column] = [f"District {idx}" for idx in sorted_gdf.loc[empty_mask].index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Left plot: choropleth map
    sorted_gdf.plot(
        column="renewal_priority",
        cmap="RdYlGn_r",
        linewidth=0.6,
        edgecolor="black",
        ax=axes[0],
    )
    axes[0].set_title("Urban Renewal Priority Score")
    axes[0].set_axis_off()

    norm = Normalize(vmin=sorted_gdf["renewal_priority"].min(), vmax=sorted_gdf["renewal_priority"].max())
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="RdYlGn_r"),
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("renewal_priority")

    # Right plot: top-10 bar chart
    top_ten = sorted_gdf.head(10).copy()
    color_map = {"High": "red", "Medium": "orange", "Low": "green"}
    bar_colors = top_ten["priority_level"].map(color_map).fillna("gray")

    axes[1].bar(top_ten[label_column], top_ten["renewal_priority"], color=bar_colors)
    axes[1].set_title("Top 10 Renewal Priority Districts")
    axes[1].set_xlabel("District")
    axes[1].set_ylabel("renewal_priority")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved results preview to {output_path}")
    print("\nScored districts sorted by renewal_priority descending:")
    print(sorted_gdf.to_string(index=False))


if __name__ == "__main__":
    main()
