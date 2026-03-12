import geopandas as gpd
import pandas as pd
import streamlit as st

@st.cache_data
def load_h9_gdf(place):
    return gpd.read_parquet(f"data/error_results_{place}_h9.geoparquet")

@st.cache_data
def load_h10_gdf(place):
    return gpd.read_parquet(f"data/error_results_{place}_h10.geoparquet")

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





