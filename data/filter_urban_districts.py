import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DATA_DIR / "urban_districts.geojson"
URBAN_CORE = box(114.10, 22.26, 114.25, 22.37)
URBAN_DISTRICT_WHITELIST = {
    "central & western",
    "central and western",
    "wan chai",
    "eastern",
    "southern",
    "yau tsim mong",
    "sham shui po",
    "kowloon city",
    "wong tai sin",
    "kwun tong",
    "kwai tsing",
    "tsuen wan",
    "sha tin",
    "tuen mun",
    "yuen long",
}


def normalize_name(value: object) -> str:
    """Normalize district names for loose matching."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def find_label_column(gdf: gpd.GeoDataFrame) -> str | None:
    """Pick a likely district name/code column if it exists."""
    candidates = [
        "district",
        "DISTRICT",
        "district_name",
        "DISTRICT_NAME",
        "name",
        "NAME",
        "name_en",
        "NAME_EN",
        "dc_name",
        "DC_NAME",
        "district_code",
        "DISTRICT_CODE",
    ]
    for column in candidates:
        if column in gdf.columns:
            return column
    return None


def load_source_districts() -> tuple[gpd.GeoDataFrame, Path]:
    """Load hk_districts.geojson when present, otherwise fall back to hk_blocks.geojson."""
    primary_path = RAW_DATA_DIR / "hk_districts.geojson"
    fallback_path = RAW_DATA_DIR / "hk_blocks.geojson"

    if primary_path.exists():
        return gpd.read_file(primary_path), primary_path

    if fallback_path.exists():
        print(f"Note: {primary_path} not found. Falling back to {fallback_path}.")
        return gpd.read_file(fallback_path), fallback_path

    raise FileNotFoundError(
        f"Missing input districts file. Expected {primary_path} or {fallback_path}."
    )


def main() -> None:
    """Filter districts/blocks to the urban core analysis extent."""
    districts, source_path = load_source_districts()
    if districts.crs is None:
        districts = districts.set_crs(epsg=4326)
    else:
        districts = districts.to_crs(epsg=4326)

    total_count = len(districts)
    bbox_mask = districts.intersects(URBAN_CORE)

    label_column = find_label_column(districts)
    if label_column is not None:
        name_mask = districts[label_column].apply(normalize_name).isin(URBAN_DISTRICT_WHITELIST)
        filtered = districts.loc[bbox_mask & name_mask].copy()
        print(f"Using whitelist column: {label_column}")
    else:
        filtered = districts.loc[bbox_mask].copy()
        print("Note: no district name/code column found, so only bbox intersection was applied.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_file(OUTPUT_PATH, driver="GeoJSON")

    print(f"Loaded source districts from {source_path}")
    print(f"Kept {len(filtered)} of {total_count} districts/blocks")
    print(f"Saved filtered urban districts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
