import geopandas as gpd
import pandas as pd
import streamlit as st

@st.cache_data
def load_gdf(place, scale):
    return gpd.read_parquet(
        f"data/error_results_{place}_{scale}.geoparquet"
    )

@st.cache_data
def load_metrics():
    return pd.read_excel("data/final_model_metrics.xlsx") 

@st.cache_data
def load_gdf_features():
    return gpd.read_parquet("data/hex_features_explore.geoparquet")

@st.cache_data
def load_feature_summary():
    # df = pd.read_excel(
    #     "data/feature_selection.xlsx"
    # )
    # df = df.iloc[:, 1:7]
    return pd.read_excel("data/feature_selection.xlsx")








