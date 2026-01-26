import streamlit as st
import plotly.express as px
from utils.data_loader import load_gdf_features, load_feature_summary

gdf = load_gdf_features()
feat_summary = load_feature_summary()
st.title("Feature Explorer")

cities = sorted(gdf["city"].dropna().unique())
city = st.selectbox("City", cities)

FEATURE_COLS = sorted([c for c in gdf.columns if c.startswith("feat_")]) 
feature_col = st.selectbox("Feature", FEATURE_COLS) 
st.subheader("Feature Documentation")

feat_row = feat_summary.loc[
    feat_summary["feature_name"] == feature_col
]

if feat_row.empty:
    st.warning("No documentation found for this feature.")
else:
    feat_row = feat_row.iloc[0]

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown(f"**Feature name** `{feat_row['feature_name']}`")
        st.markdown("**Description**")
        st.write(feat_row["description"])

    with col2:
        st.markdown(f"""
        **Non-zero count:** {feat_row['non_zero_count']:,}  
        **Coverage (%):** {feat_row['coverage_pct']:.1f}% 
        **Data range:** {feat_row['data_range']}  
        **Type:** {feat_row['feature_type']} 
        """)

df = gdf[gdf["city"] == city].copy()

fig = px.choropleth_mapbox(
    df,
    geojson=df.geometry,
    locations=df.index,
    color=feature_col,
    color_continuous_scale="sunset",
    mapbox_style="carto-positron",
    zoom=11.5,
    center={
        "lat": df.geometry.centroid.y.mean(),
        "lon": df.geometry.centroid.x.mean(),
    },
)


fig.update_layout(
    height=900  # try 650–850 depending on screen
    )


st.plotly_chart(fig, use_container_width=True) 

st.divider()
