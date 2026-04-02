import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calculator.PointIndex import poi_coverage_cal
from calculator.AoiIndex import aoi_coverage_cal, building_floor_area_cal
from calculator.LineIndex import road_dens_cal


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


def load_districts() -> gpd.GeoDataFrame:
    """Load district polygons, with fallback to the current block filename."""
    primary_path = RAW_DATA_DIR / "hk_districts.geojson"
    fallback_path = RAW_DATA_DIR / "hk_blocks.geojson"

    if primary_path.exists():
        return load_geojson(primary_path, "districts")

    if fallback_path.exists():
        print(f"Note: {primary_path} not found, using {fallback_path} as districts.")
        return load_geojson(fallback_path, "districts")

    raise FileNotFoundError(
        f"Missing district file. Expected {primary_path} or fallback {fallback_path}."
    )


def load_transit_pois() -> gpd.GeoDataFrame:
    """Load transit POIs, with fallback filtering from hk_pois.geojson."""
    primary_path = RAW_DATA_DIR / "hk_poi_transit.geojson"
    fallback_path = RAW_DATA_DIR / "hk_pois.geojson"

    if primary_path.exists():
        return load_geojson(primary_path, "transit POIs")

    pois = load_geojson(fallback_path, "combined POIs for transit fallback")
    transit_mask = (
        pois.get("amenity", pd.Series(index=pois.index)).eq("bus_station")
        | pois.get("public_transport", pd.Series(index=pois.index)).eq("stop_position")
    )
    transit_pois = pois.loc[transit_mask].copy()
    print(f"Derived transit POIs from {fallback_path}: {len(transit_pois)} features")
    return transit_pois


def load_supermarket_pois() -> gpd.GeoDataFrame:
    """Load supermarket POIs, with fallback filtering from hk_pois.geojson."""
    primary_path = RAW_DATA_DIR / "hk_poi_supermarket.geojson"
    fallback_path = RAW_DATA_DIR / "hk_pois.geojson"

    if primary_path.exists():
        return load_geojson(primary_path, "supermarket POIs")

    pois = load_geojson(fallback_path, "combined POIs for supermarket fallback")
    supermarket_mask = pois.get("amenity", pd.Series(index=pois.index)).eq("supermarket")
    supermarket_pois = pois.loc[supermarket_mask].copy()
    print(f"Derived supermarket POIs from {fallback_path}: {len(supermarket_pois)} features")
    return supermarket_pois


def load_greenspace() -> gpd.GeoDataFrame:
    """Load greenspace polygons, with fallback to hk_greenspace.geojson."""
    primary_path = RAW_DATA_DIR / "hk_poi_greenspace.geojson"
    fallback_path = RAW_DATA_DIR / "hk_greenspace.geojson"

    if primary_path.exists():
        return load_geojson(primary_path, "greenspace AOIs")

    if fallback_path.exists():
        print(f"Note: {primary_path} not found, using {fallback_path} for greenspace coverage.")
        return load_geojson(fallback_path, "greenspace AOIs")

    raise FileNotFoundError(
        f"Missing greenspace file. Expected {primary_path} or fallback {fallback_path}."
    )


def load_buildings() -> gpd.GeoDataFrame:
    """Load buildings and ensure a numeric height field is available."""
    buildings = load_geojson(RAW_DATA_DIR / "hk_buildings.geojson", "buildings").copy()

    if "height" not in buildings.columns:
        buildings["height"] = 15
        print("Note: 'height' field missing in buildings. Assigned default height=15 meters.")
    else:
        buildings["height"] = pd.to_numeric(buildings["height"], errors="coerce").fillna(15)
        print("Prepared building heights using 'height' field with default fill=15 meters.")

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
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    districts = load_districts()
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
    merged_result.to_file(OUTPUT_PATH, driver="GeoJSON")

    print(f"\nSaved merged indicator file to {OUTPUT_PATH}")
    print_summary_table(merged_result)


if __name__ == "__main__":
    main()
