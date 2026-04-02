import os
import sys
import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Calculator.PointIndex import poi_coverage_cal
from Calculator.AoiIndex import aoi_coverage_cal, building_floor_area_cal
from Calculator.LineIndex import road_dens_cal


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DATA_DIR / "districts_with_indicators.geojson"

MERGE_KEY = "__district_id__"
INDICATOR_COLUMNS = [
    "road_density",
    "transit_coverage_rate",
    "supermarket_coverage_rate",
    "greenspace_coverage_rate",
    "building_floor_area",
]


def load_geojson(path: Path, label: str) -> gpd.GeoDataFrame:
    """Load a GeoJSON file with a clear error message."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label} file: {path}")
    gdf = gpd.read_file(path)
    print(f"Loaded {label}: {path} ({len(gdf)} features, CRS={gdf.crs})")
    return gdf


def filter_geometry_types(
    gdf: gpd.GeoDataFrame, allowed_types: set[str], label: str
) -> gpd.GeoDataFrame:
    """Keep only the requested geometry types and report how many remain."""
    filtered = gdf[gdf.geometry.geom_type.isin(allowed_types)].copy()
    print(
        f"Filtered {label} to geometry types {sorted(allowed_types)}: "
        f"{len(filtered)} of {len(gdf)} features kept"
    )
    return filtered


def load_districts(districts_path: Path | None = None) -> gpd.GeoDataFrame:
    """Load district polygons using the blocks file as the analysis unit."""
    path = districts_path or (RAW_DATA_DIR / "hk_blocks.geojson")
    return load_geojson(path, "districts")


def load_transit_pois() -> gpd.GeoDataFrame:
    """Load transit POIs, with fallback filtering from hk_pois.geojson."""
    primary_path = RAW_DATA_DIR / "hk_poi_transit.geojson"
    fallback_path = RAW_DATA_DIR / "hk_pois.geojson"

    if primary_path.exists():
        return filter_geometry_types(
            load_geojson(primary_path, "transit POIs"),
            {"Point", "MultiPoint"},
            "transit POIs",
        )

    pois = load_geojson(fallback_path, "combined POIs for transit fallback")
    transit_mask = (
        pois.get("amenity", pd.Series(index=pois.index)).eq("bus_station")
        | pois.get("public_transport", pd.Series(index=pois.index)).eq("stop_position")
    )
    transit_pois = pois.loc[transit_mask].copy()
    print(f"Derived transit POIs from {fallback_path}: {len(transit_pois)} features")
    return filter_geometry_types(transit_pois, {"Point", "MultiPoint"}, "transit POIs")


def load_supermarket_pois() -> gpd.GeoDataFrame:
    """Load supermarket POIs, with fallback filtering from hk_pois.geojson."""
    primary_path = RAW_DATA_DIR / "hk_poi_supermarket.geojson"
    fallback_path = RAW_DATA_DIR / "hk_pois.geojson"

    if primary_path.exists():
        return filter_geometry_types(
            load_geojson(primary_path, "supermarket POIs"),
            {"Point", "MultiPoint"},
            "supermarket POIs",
        )

    pois = load_geojson(fallback_path, "combined POIs for supermarket fallback")
    supermarket_mask = pois.get("amenity", pd.Series(index=pois.index)).eq("supermarket")
    supermarket_pois = pois.loc[supermarket_mask].copy()
    print(f"Derived supermarket POIs from {fallback_path}: {len(supermarket_pois)} features")
    return filter_geometry_types(
        supermarket_pois, {"Point", "MultiPoint"}, "supermarket POIs"
    )


def load_greenspace() -> gpd.GeoDataFrame:
    """Load greenspace polygons using the greenspace file directly."""
    greenspace = load_geojson(RAW_DATA_DIR / "hk_greenspace.geojson", "greenspace AOIs")
    return filter_geometry_types(
        greenspace, {"Polygon", "MultiPolygon"}, "greenspace AOIs"
    )


def load_buildings() -> gpd.GeoDataFrame:
    """Load buildings and ensure a numeric height field is available."""
    buildings = load_geojson(RAW_DATA_DIR / "hk_buildings.geojson", "buildings").copy()
    buildings = filter_geometry_types(buildings, {"Polygon", "MultiPolygon"}, "buildings")

    if "height" not in buildings.columns:
        buildings["height"] = 120
        print("Note: 'height' field missing in buildings. Assigned default height=120 meters.")
    else:
        buildings["height"] = pd.to_numeric(buildings["height"], errors="coerce").fillna(120)
        print("Prepared building heights using 'height' field with default fill=120 meters.")

    return buildings


def prepare_districts_for_calc(districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create a fresh district copy for each indicator calculation."""
    district_copy = districts.copy()
    district_copy[MERGE_KEY] = districts.index
    return district_copy


def merge_indicator(
    merged: gpd.GeoDataFrame, result: gpd.GeoDataFrame, indicator_column: str
) -> gpd.GeoDataFrame:
    """Merge a single indicator back onto the master districts layer."""
    indicator_frame = result[[MERGE_KEY, indicator_column]].copy()
    return merged.merge(indicator_frame, on=MERGE_KEY, how="left")


def zero_indicator_result(districts: gpd.GeoDataFrame, indicator_column: str) -> gpd.GeoDataFrame:
    """Create a zero-filled indicator result when an input layer is empty."""
    result = prepare_districts_for_calc(districts)
    result[indicator_column] = 0.0
    return result


def print_summary_table(result: gpd.GeoDataFrame) -> None:
    """Print min, max, and mean for each indicator column."""
    summary = (
        result[INDICATOR_COLUMNS]
        .agg(["min", "max", "mean"])
        .transpose()
        .round(4)
        .reset_index()
        .rename(columns={"index": "indicator"})
    )

    print("\nIndicator summary:")
    print(summary.to_string(index=False))


def main() -> None:
    """Run all district-level indicator calculations and save the merged output."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--districts",
        default=str(RAW_DATA_DIR / "hk_blocks.geojson"),
        help="Input district/block GeoJSON path.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Output GeoJSON path for indicator results.",
    )
    args = parser.parse_args()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    districts_path = Path(args.districts)
    output_path = Path(args.output)

    districts = load_districts(districts_path)
    roads = load_geojson(RAW_DATA_DIR / "hk_roads.geojson", "roads")
    buildings = load_buildings()
    transit_pois = load_transit_pois()
    supermarket_pois = load_supermarket_pois()
    greenspace = load_greenspace()

    merged_result = districts.copy()
    merged_result[MERGE_KEY] = merged_result.index

    road_result = road_dens_cal(roads.copy(), prepare_districts_for_calc(districts))
    merged_result = merge_indicator(merged_result, road_result, "road_density")

    transit_result = poi_coverage_cal(
        transit_pois.copy(),
        prepare_districts_for_calc(districts),
        poi_type="transit",
        buffer_distance=500,
    )
    merged_result = merge_indicator(merged_result, transit_result, "transit_coverage_rate")

    if supermarket_pois.empty:
        print("Note: supermarket POI layer is empty. Filling supermarket_coverage_rate with 0.")
        supermarket_result = zero_indicator_result(districts, "supermarket_coverage_rate")
    else:
        supermarket_result = poi_coverage_cal(
            supermarket_pois.copy(),
            prepare_districts_for_calc(districts),
            poi_type="supermarket",
            buffer_distance=300,
        )
    merged_result = merge_indicator(
        merged_result, supermarket_result, "supermarket_coverage_rate"
    )

    greenspace_result = aoi_coverage_cal(
        greenspace.copy(),
        prepare_districts_for_calc(districts),
        aoi_type="greenspace",
        threshold=500,
    )
    merged_result = merge_indicator(
        merged_result, greenspace_result, "greenspace_coverage_rate"
    )

    building_result = building_floor_area_cal(
        buildings.copy(),
        prepare_districts_for_calc(districts),
        bd_type="building",
        height_field="height",
        height_per_floor=3,
    )
    merged_result = merge_indicator(merged_result, building_result, "building_floor_area")

    merged_result[INDICATOR_COLUMNS] = merged_result[INDICATOR_COLUMNS].fillna(0)
    merged_result = gpd.GeoDataFrame(merged_result, geometry="geometry", crs=districts.crs)
    merged_result = merged_result.drop(columns=[MERGE_KEY])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_result.to_file(output_path, driver="GeoJSON")

    print(f"\nSaved merged indicator file to {output_path}")
    print_summary_table(merged_result)


if __name__ == "__main__":
    main()
