import os
import sys
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from calculator.CompositeIndex import build_composite_index
except ModuleNotFoundError:
    from Calculator.CompositeIndex import build_composite_index


DATA_PATH = Path("data/processed/urban_districts_scored.geojson")
MAP_CENTER = [22.32, 114.17]
MAP_ZOOM = 11


@st.cache_data
def load_scored_geojson(path: str) -> gpd.GeoDataFrame:
    """Load scored district data once and reuse it across reruns."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def get_district_id_column(gdf: gpd.GeoDataFrame) -> str:
    """Find a district identifier column using the requested preference list."""
    candidates = [
        "DCCODE",
        "DC_CODE",
        "district",
        "District",
        "id",
        "ID",
        "name",
        "Name",
        "ENAME",
        "index",
    ]
    matching = [column for column in gdf.columns if column in candidates]
    return matching[0] if matching else gdf.columns[0]


def prepare_tooltip_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add rounded display columns for map tooltips."""
    tooltip_gdf = gdf.copy()
    score_columns = [
        "renewal_priority",
        "skeleton_score",
        "metabolism_score",
        "circulatory_score",
    ]
    for column in score_columns:
        tooltip_gdf[f"{column}_display"] = tooltip_gdf[column].map(lambda value: f"{value:.2f}")
    return tooltip_gdf


def build_priority_map(gdf: gpd.GeoDataFrame, id_col: str) -> folium.Map:
    """Create the Folium choropleth map with tooltip overlay."""
    map_gdf = prepare_tooltip_columns(gdf)
    folium_map = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=map_gdf.to_json(),
        data=map_gdf[[id_col, "renewal_priority"]],
        columns=[id_col, "renewal_priority"],
        key_on=f"feature.properties.{id_col}",
        fill_color="RdYlGn_r",
        fill_opacity=0.7,
        line_opacity=0.6,
        line_color="black",
        legend_name="Renewal Priority Score",
        highlight=True,
    ).add_to(folium_map)

    folium.GeoJson(
        map_gdf.to_json(),
        style_function=lambda _: {
            "fillColor": "transparent",
            "color": "black",
            "weight": 0.7,
            "fillOpacity": 0.0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                id_col,
                "renewal_priority_display",
                "priority_level",
                "skeleton_score_display",
                "metabolism_score_display",
                "circulatory_score_display",
            ],
            aliases=[
                f"{id_col}:",
                "Renewal priority:",
                "Priority level:",
                "Skeleton score:",
                "Metabolism score:",
                "Circulatory score:",
            ],
            localize=True,
            sticky=False,
            labels=True,
        ),
    ).add_to(folium_map)

    return folium_map


def format_top_district(top_row: pd.Series, id_col: str) -> str:
    """Format the top district identifier for the metric card."""
    value = top_row[id_col]
    return str(value)


def main() -> None:
    """Render the Hong Kong Urban Renewal Index Streamlit app."""
    st.set_page_config(layout="wide", page_title="HK Urban Renewal Index")

    st.sidebar.title("Urban Organism Weights")
    st.sidebar.subheader("Adjust dimension weights (must sum to 1.0)")

    w_skeleton = st.sidebar.slider("Skeleton (Building)", 0.0, 1.0, 0.4, 0.05)
    w_metabolism = st.sidebar.slider("Metabolism (Services)", 0.0, 1.0, 0.3, 0.05)
    w_circulatory = st.sidebar.slider("Circulatory (Roads)", 0.0, 1.0, 0.3, 0.05)

    total = w_skeleton + w_metabolism + w_circulatory
    if abs(total - 1.0) > 0.01:
        st.sidebar.warning(f"Weights sum to {total:.2f}, should be 1.0")
    else:
        st.sidebar.success("Weights valid")

    st.sidebar.divider()
    st.sidebar.caption("Urban Organism Theory")
    st.sidebar.caption("Skeleton = building density & floor area")
    st.sidebar.caption("Metabolism = access to supermarkets & green space")
    st.sidebar.caption("Circulatory = road network & transit access")

    st.title("Hong Kong Urban Renewal Index Calculator")
    st.markdown(
        "An interactive tool based on the **Urban Organism** theory. "
        "Adjust dimension weights in the sidebar to see how renewal "
        "priorities shift across Hong Kong's urban districts."
    )
    st.divider()

    gdf = load_scored_geojson(str(DATA_PATH))
    scored_gdf = build_composite_index(gdf, w_skeleton, w_metabolism, w_circulatory)
    id_col = get_district_id_column(scored_gdf)

    left_col, right_col = st.columns([6, 4])

    with left_col:
        st.subheader("Interactive Priority Map")
        folium_map = build_priority_map(scored_gdf, id_col)
        st_folium(folium_map, width=700, height=500)

    with right_col:
        st.subheader("Urban Renewal Summary")

        high_count = int((scored_gdf["priority_level"] == "High").sum())
        mean_priority = float(scored_gdf["renewal_priority"].mean())
        top_district = scored_gdf.sort_values("renewal_priority", ascending=False).iloc[0]

        metric_cols = st.columns(3)
        metric_cols[0].metric("High priority districts", high_count)
        metric_cols[1].metric("Mean priority score", f"{mean_priority:.3f}")
        metric_cols[2].metric("Top district ID", format_top_district(top_district, id_col))

        st.markdown("**Average Dimension Scores**")
        avg_scores = (
            scored_gdf[["skeleton_score", "metabolism_score", "circulatory_score"]]
            .mean()
            .rename(
                {
                    "skeleton_score": "Skeleton",
                    "metabolism_score": "Metabolism",
                    "circulatory_score": "Circulatory",
                }
            )
            .sort_values()
        )
        st.bar_chart(avg_scores, horizontal=True)

        st.markdown("**Top 10 Districts**")
        top_ten = scored_gdf.sort_values("renewal_priority", ascending=False).head(10).copy()
        top10_df = top_ten[
            [
                id_col,
                "renewal_priority",
                "priority_level",
                "skeleton_score",
                "metabolism_score",
                "circulatory_score",
            ]
        ].copy()
        st.dataframe(
            top10_df.style.format(
                {
                    "renewal_priority": "{:.3f}",
                    "skeleton_score": "{:.3f}",
                    "metabolism_score": "{:.3f}",
                    "circulatory_score": "{:.3f}",
                }
            ),
            use_container_width=True,
        )
        csv = top10_df.to_csv(index=False)
        st.download_button(
            label="Download top 10 as CSV",
            data=csv,
            file_name="hk_renewal_priority_top10.csv",
            mime="text/csv",
        )

    st.caption(
        "Data sources: OpenStreetMap contributors · "
        "HK Government Open Data · "
        "MSc Urban Informatics, PolyU 2025"
    )


if __name__ == "__main__":
    main()
