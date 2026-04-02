from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd


# Section 1: Global configuration
BOUNDING_BOX = (114.09, 22.19, 114.28, 22.37)  # (left, bottom, right, top)
RAW_DATA_DIR = Path("data/raw")


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame in EPSG:4326."""
    if gdf.crs is None:
        return gdf.set_crs(epsg=4326)
    return gdf.to_crs(epsg=4326)


def save_geojson(gdf: gpd.GeoDataFrame, output_path: Path) -> dict[str, str | int]:
    """Persist a GeoDataFrame and return summary metadata."""
    cleaned = ensure_wgs84(gdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_file(output_path, driver="GeoJSON")
    return {
        "file": str(output_path),
        "count": len(cleaned),
        "crs": str(cleaned.crs),
    }


def fetch_roads() -> gpd.GeoDataFrame:
    """Download and merge drive and walk road edges for Hong Kong."""
    drive_graph = ox.graph_from_bbox(BOUNDING_BOX, network_type="drive")
    walk_graph = ox.graph_from_bbox(BOUNDING_BOX, network_type="walk")

    drive_edges = ox.graph_to_gdfs(drive_graph, nodes=False, edges=True)
    walk_edges = ox.graph_to_gdfs(walk_graph, nodes=False, edges=True)

    roads = gpd.GeoDataFrame(
        pd.concat([drive_edges, walk_edges], ignore_index=False),
        geometry="geometry",
        crs=drive_edges.crs,
    ).reset_index(drop=True)

    roads["geometry_wkb"] = roads.geometry.apply(lambda geom: geom.wkb if geom is not None else None)
    roads = roads.drop_duplicates(subset="geometry_wkb").drop(columns="geometry_wkb")
    return roads


def fetch_features(tags: dict[str, bool | str | list[str]]) -> gpd.GeoDataFrame:
    """Download OSM features for the configured bounding box."""
    return ox.features_from_bbox(BOUNDING_BOX, tags)


def main() -> None:
    """Download Hong Kong source data and print a save summary."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, str | int]] = []

    # Section 2: Download Hong Kong road network
    try:
        roads = fetch_roads()
        summaries.append(save_geojson(roads, RAW_DATA_DIR / "hk_roads.geojson"))
        print("Saved roads to data/raw/hk_roads.geojson")
    except Exception as exc:
        print(f"Failed to download roads: {exc}")

    # Section 3: Download OSM building footprints
    try:
        buildings = fetch_features({"building": True})
        summaries.append(save_geojson(buildings, RAW_DATA_DIR / "hk_buildings.geojson"))
        print("Saved buildings to data/raw/hk_buildings.geojson")
    except Exception as exc:
        print(f"Failed to download buildings: {exc}")

    # Section 4: Download OSM green spaces
    try:
        greenspace = fetch_features(
            {
                "leisure": ["park", "garden", "pitch"],
                "landuse": ["grass", "recreation_ground"],
            }
        )
        summaries.append(save_geojson(greenspace, RAW_DATA_DIR / "hk_greenspace.geojson"))
        print("Saved green spaces to data/raw/hk_greenspace.geojson")
    except Exception as exc:
        print(f"Failed to download green spaces: {exc}")

    # Section 5: Download OSM POIs
    try:
        pois = fetch_features(
            {
                "amenity": ["supermarket", "bus_station"],
                "public_transport": ["stop_position"],
            }
        )
        summaries.append(save_geojson(pois, RAW_DATA_DIR / "hk_pois.geojson"))
        print("Saved POIs to data/raw/hk_pois.geojson")
    except Exception as exc:
        print(f"Failed to download POIs: {exc}")

    # Section 6: Print saved file summary
    print("\nDownload summary:")
    if not summaries:
        print("No datasets were saved.")
        return

    for summary in summaries:
        print(
            f"- {summary['file']}: "
            f"features={summary['count']}, crs={summary['crs']}"
        )


if __name__ == "__main__":
    main()
