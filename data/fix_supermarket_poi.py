from pathlib import Path
import re

import geopandas as gpd
import osmnx as ox
import pandas as pd


RAW_DATA_DIR = Path("data/raw")
INPUT_PATH = RAW_DATA_DIR / "hk_pois.geojson"
OUTPUT_PATH = RAW_DATA_DIR / "hk_poi_supermarket.geojson"
DOWNLOAD_BBOX = (114.10, 22.26, 114.25, 22.37)  # (left, bottom, right, top)
NAME_PATTERNS = [
    "wellcome",
    "parknshop",
    "park n shop",
    "fusion",
    "taste",
    "city super",
    "citysuper",
    "marketplace",
    "great",
    "aeon",
    "yata",
    "jtc",
    "dchfood",
    "百佳",
    "惠康",
    "超市",
    "超級市場",
]
SHOP_VALUES = {
    "supermarket",
    "convenience",
    "grocery",
    "department_store",
    "mall",
    "wholesale",
    "hypermarket",
}


def print_value_counts(gdf: gpd.GeoDataFrame, column: str) -> None:
    """Print value_counts for a column, handling missing columns safely."""
    print(f"\nValue counts for '{column}':")
    if column not in gdf.columns:
        print(f"Column '{column}' does not exist.")
        return

    counts = gdf[column].fillna("<NA>").astype(str).value_counts(dropna=False)
    print(counts.to_string())


def normalize_text(value: object) -> str:
    """Normalize text for case-insensitive matching and de-duplication."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def filter_supermarket_candidates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extract supermarket-like POIs from existing data using broad rules."""
    shop_series = (
        gdf["shop"].fillna("").astype(str).str.lower()
        if "shop" in gdf.columns
        else pd.Series("", index=gdf.index)
    )
    name_series = (
        gdf["name"].fillna("").astype(str)
        if "name" in gdf.columns
        else pd.Series("", index=gdf.index)
    )

    shop_mask = shop_series.isin(SHOP_VALUES)
    name_mask = pd.Series(False, index=gdf.index)
    for pattern in NAME_PATTERNS:
        name_mask = name_mask | name_series.str.contains(pattern, case=False, na=False, regex=False)

    candidates = gdf.loc[shop_mask | name_mask].copy()
    print(f"\nFiltered supermarket-like POIs from existing file: {len(candidates)} features")
    return candidates


def download_fresh_supermarkets() -> gpd.GeoDataFrame:
    """Download fresh supermarket-related features from OSM."""
    tags = {
        "shop": ["supermarket", "convenience", "grocery"],
        "amenity": "marketplace",
    }
    print("\nDownloading fresh supermarket data from OSM...")
    fresh = ox.features_from_bbox(DOWNLOAD_BBOX, tags)
    fresh = fresh.reset_index(drop=True)
    if fresh.crs is None:
        fresh = fresh.set_crs(epsg=4326)
    else:
        fresh = fresh.to_crs(epsg=4326)
    print(f"Downloaded fresh supermarket-related features: {len(fresh)}")
    return fresh


def deduplicate_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop exact duplicates and near-duplicates with the same normalized name."""
    if gdf.empty:
        return gdf.copy()

    deduped = gdf.copy()
    deduped["__name_norm"] = deduped.get("name", pd.Series("", index=deduped.index)).apply(normalize_text)
    deduped["__geom_wkb"] = deduped.geometry.apply(lambda geom: geom.wkb if geom is not None else None)
    deduped = deduped.drop_duplicates(subset="__geom_wkb").reset_index(drop=True)

    projected = deduped.to_crs(epsg=3857)
    keep_indices: list[int] = []
    seen_by_name: dict[str, list[tuple[int, object]]] = {}

    for idx, row in projected.iterrows():
        geom = row.geometry
        name_norm = row["__name_norm"]

        if geom is None or geom.is_empty:
            continue

        if not name_norm:
            keep_indices.append(idx)
            continue

        existing_items = seen_by_name.setdefault(name_norm, [])
        is_duplicate = False
        for _, existing_geom in existing_items:
            if existing_geom.distance(geom) <= 30:
                is_duplicate = True
                break

        if not is_duplicate:
            keep_indices.append(idx)
            existing_items.append((idx, geom))

    deduped = deduped.loc[sorted(set(keep_indices))].copy()
    deduped = deduped.drop(columns=["__name_norm", "__geom_wkb"], errors="ignore")
    return deduped.to_crs(epsg=4326)


def main() -> None:
    """Fix and export supermarket POIs."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input POI file: {INPUT_PATH}")

    pois = gpd.read_file(INPUT_PATH)
    if pois.crs is None:
        pois = pois.set_crs(epsg=4326)
    else:
        pois = pois.to_crs(epsg=4326)

    print(f"Loaded {INPUT_PATH} ({len(pois)} features, CRS={pois.crs})")

    print_value_counts(pois, "shop")
    print_value_counts(pois, "amenity")

    supermarket_pois = filter_supermarket_candidates(pois)

    if len(supermarket_pois) < 10:
        fresh_supermarkets = download_fresh_supermarkets()
        supermarket_pois = pd.concat([supermarket_pois, fresh_supermarkets], ignore_index=True)
        supermarket_pois = gpd.GeoDataFrame(supermarket_pois, geometry="geometry", crs="EPSG:4326")
        print(f"Combined filtered + downloaded supermarket features: {len(supermarket_pois)}")

    supermarket_pois = deduplicate_features(supermarket_pois)
    supermarket_pois = supermarket_pois.to_crs(epsg=4326)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    supermarket_pois.to_file(OUTPUT_PATH, driver="GeoJSON")

    sample_columns = [col for col in ["name", "shop", "amenity"] if col in supermarket_pois.columns]
    print(f"\nSaved supermarket POIs to {OUTPUT_PATH}")
    print(f"Final feature count: {len(supermarket_pois)}")
    print("\nSample rows:")
    if sample_columns:
        print(supermarket_pois[sample_columns].head(5).to_string(index=False))
    else:
        print("No name/shop/amenity columns available to display.")


if __name__ == "__main__":
    main()
