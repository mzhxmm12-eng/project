import os
import sys
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


INDICATOR_COLUMNS = [
    "road_density",
    "transit_coverage_rate",
    "supermarket_coverage_rate",
    "greenspace_coverage_rate",
    "building_floor_area",
]


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Apply min-max normalization and keep a stable 0 result for constant columns."""
    min_value = series.min()
    max_value = series.max()

    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.0, index=series.index)

    return (series - min_value) / (max_value - min_value)


def build_composite_index(
    gdf: gpd.GeoDataFrame,
    w_skeleton: float = 0.4,
    w_metabolism: float = 0.3,
    w_circulatory: float = 0.3,
) -> gpd.GeoDataFrame:
    """Build normalized indicators, dimension scores, and final renewal priority."""
    missing_columns = [column for column in INDICATOR_COLUMNS if column not in gdf.columns]
    if missing_columns:
        raise ValueError(f"Missing required indicator columns: {missing_columns}")

    result = gdf.copy()

    for indicator in INDICATOR_COLUMNS:
        result[indicator] = pd.to_numeric(result[indicator], errors="coerce").fillna(0)
        result[f"{indicator}_norm"] = min_max_normalize(result[indicator]).astype(float)

    result["skeleton_score"] = result["building_floor_area_norm"]
    result["metabolism_score"] = (
        0.5 * result["supermarket_coverage_rate_norm"]
        + 0.5 * result["greenspace_coverage_rate_norm"]
    )
    result["circulatory_score"] = (
        0.5 * result["road_density_norm"]
        + 0.5 * result["transit_coverage_rate_norm"]
    )

    result["renewal_priority"] = 1 - (
        w_skeleton * result["skeleton_score"]
        + w_metabolism * result["metabolism_score"]
        + w_circulatory * result["circulatory_score"]
    )
    result["renewal_priority"] = result["renewal_priority"].clip(0, 1)

    result["renewal_priority_rank"] = (
        result["renewal_priority"].rank(method="dense", ascending=False).astype(int)
    )
    result["priority_level"] = np.select(
        [
            result["renewal_priority"] >= 0.67,
            result["renewal_priority"] >= 0.33,
        ],
        [
            "High",
            "Medium",
        ],
        default="Low",
    )

    return result


def find_display_column(gdf: gpd.GeoDataFrame) -> str | None:
    """Pick a readable district identifier column for console output."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/districts_with_indicators.geojson",
        help="Input GeoJSON path with indicator columns.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/districts_scored.geojson",
        help="Output GeoJSON path for scored districts.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    districts = gpd.read_file(input_path)
    scored = build_composite_index(districts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_file(output_path, driver="GeoJSON")

    display_column = find_display_column(scored)
    top_five = scored.sort_values("renewal_priority", ascending=False).head(5).copy()
    columns_to_print = ["renewal_priority", "renewal_priority_rank", "priority_level"]
    if display_column is not None:
        columns_to_print = [display_column] + columns_to_print

    print(f"Saved scored districts to {output_path}")
    print("\nTop 5 highest priority districts:")
    print(top_five[columns_to_print].to_string(index=False))
