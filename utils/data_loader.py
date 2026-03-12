import geopandas as gpd
import pandas as pd
import streamlit as st

@st.cache_data
def load_gdf(place, scale):
    return gpd.read_parquet(
        f"data/error_results_{place}_{scale}.geoparquet"
    )

@st.cache_data
def load_base_gdf():
    return gpd.read_parquet(
        "data/error_results_all_h10.geoparquet"
    )

@st.cache_data
def load_metrics():
    return pd.read_excel("data/model_comparison_results.xlsx") 

@st.cache_data
def load_gdf_features():
    return gpd.read_parquet("data/hex_features_explore_1.geoparquet")

@st.cache_data
def load_feature_summary():
    df = pd.read_csv(
        "data/feature_summary_for_team.csv"
    )
    
    # df = df.iloc[:, 1:7]

    return df








