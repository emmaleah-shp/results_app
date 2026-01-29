import geopandas as gpd
import pandas as pd
import streamlit as st

@st.cache_data
def load_gdf():
    return gpd.read_parquet("data/error_results_a_b_c.geoparquet")

@st.cache_data
def load_metrics():
    return pd.read_csv("data/model_metrics.csv")

@st.cache_data
def load_gdf_features():
    return gpd.read_parquet("data/hex_features_explore_1.geoparquet")

@st.cache_data
def load_feature_summary():
    df = pd.read_csv(
        "data/feature_summary_for_team.csv"
    )
    # keep only relevant columns (as you already do)
    df = df.iloc[:, 1:7]

    return df

