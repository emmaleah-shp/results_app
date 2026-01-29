import streamlit as st
import plotly.express as px
from utils.data_loader import load_gdf_features, load_feature_summary
LISA_COLORS = {
    "High-High": "#d7191c",
    "Low-Low": "#2c7bb6",
    "High-Low": "#fdae61",
    "Low-High": "#abd9e9",
    "Not significant": "#dddddd",
}

gdf = load_gdf_features()
feat_summary = load_feature_summary()
st.title("Explorador de variables")

cities = sorted(gdf["city"].dropna().unique())

FEATURE_COLS = [c for c in gdf.columns if c.startswith("feat_") | c.startswith("target_")]
transform_cols = [c for c in gdf.columns if c.startswith("log_") | c.startswith("lisa_")]
feature_col = st.selectbox("Feature", FEATURE_COLS) 
feat_row = feat_summary.loc[
    feat_summary["feature_name"] == feature_col
]

city = st.sidebar.selectbox("Ciudad", cities)

scale = st.sidebar.radio(
    "Map Scale",
    ["Raw", "Log", "LISA Clusters"],
    horizontal=True
)


if feat_row.empty:
    st.warning("No se ha encontrado documentación para esta variable.")
else:
    feat_row = feat_row.iloc[0]
    st.subheader(f"**Nombre de feature:** `{feat_row['feature_name']}`")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown(f"""##### **Descripción (en inglés):**    
        {feat_row['description']}
        """)
        st.markdown(f"##### **Número de valores NON cero:** `{feat_row['non_zero_count']:,}`")

    with col2:
        #st.subheader(f"**Número de valores NON cero:** `{feat_row['non_zero_count']:,}`")  
        st.markdown(f"""
        ##### **Cobertura (%):** {feat_row['coverage_pct']:.1f}%    
        ##### **Rango de datos:** {feat_row['data_range']}  
        ##### **Tipo:** {feat_row['feature_type']} 
        """)

df = gdf[gdf["city"] == city].copy()
df_test = df[df["is_test"]].copy()

if scale == "Log":
    value_col = f"log_{feature_col}"
elif scale == "Raw":
    value_col = feature_col
elif scale == "LISA Clusters":
    value_col = f"lisa_cluster_{feature_col}"
    # lisa_i_col = f"lisa_i_{feature_col}"
    # lisa_p_col = f"lisa_p_{feature_col}"

if scale =="LISA Clusters": 
    fig_map = px.choropleth_mapbox(
        df,
        geojson=df.geometry,
        locations=df.index,
        color=value_col,
        color_discrete_map=LISA_COLORS,
        mapbox_style="carto-positron",
        zoom=11.5,
        center={
            "lat": df.geometry.centroid.y.mean(),
            "lon": df.geometry.centroid.x.mean(),
        },
    )

    fig_map.update_layout(
        height=900,  # try 650–850 depending on screen
        title=f"LISA Clusters: {feature_col}"
        )
    
    fig_test = px.choropleth_mapbox(
        df_test,
        geojson=df_test.geometry,
        locations=df_test.index,
        color_discrete_sequence=["rgba(0,0,0,0)"],  # fully transparent fill
        hover_data={value_col: value_col},
    )

    fig_test.update_traces(
        marker_line_color="black",
        marker_line_width=2.0,
        # showlegend=False,
    )

    for trace in fig_test.data:
        fig_map.add_trace(trace)

    st.plotly_chart(fig_map, use_container_width=True)
else: 
    fig_map = px.choropleth_mapbox(
        df,
        geojson=df.geometry,
        locations=df.index,
        color=value_col,
        color_continuous_scale="sunset",
        mapbox_style="carto-positron",
        zoom=11.5,
        center={
            "lat": df.geometry.centroid.y.mean(),
            "lon": df.geometry.centroid.x.mean(),
        },
    )

    fig_map.update_layout(
        height=900,  # try 650–850 depending on screen
        title=f"Distribución espacial de {feature_col} ({scale})"
        )
    
    fig_test = px.choropleth_mapbox(
        df_test,
        geojson=df_test.geometry,
        locations=df_test.index,
        color_discrete_sequence=["rgba(0,0,0,0)"],  # fully transparent fill
        hover_data={value_col: value_col},
    )

    fig_test.update_traces(
        marker_line_color="black",
        marker_line_width=2.0,
        # showlegend=False,
    )

    for trace in fig_test.data:
        fig_map.add_trace(trace)

    st.plotly_chart(fig_map, use_container_width=True) 

st.subheader("Distribución de features")
st.markdown("_*En todos los hexágonos_")

# Controls
col1, col2 = st.columns(2)

with col1:
    bins = st.slider("Bins", min_value=5, max_value=50, value=10)

with col2:
    normalize = st.checkbox("Normalizar (densidad)", value=False)

# Histogram
fig = px.histogram(
    gdf, # to use data filtered by city, use df
    x=value_col,
    nbins=bins,
    histnorm="probability density" if normalize else None,
    marginal="box",               # shows boxplot above histogram
    template="plotly_white",
    title=f"Distribución de {value_col}",
)

fig.update_layout(
    bargap=0.05,
    xaxis_title=value_col,
    yaxis_title="Densidad" if normalize else "Conteo",
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

feat_summary = feat_summary[
    (feat_summary["feature_name"].isin(FEATURE_COLS))
]

type = st.multiselect(
    "Tipo de variable",
    sorted(feat_summary["feature_type"].unique()),
    default=sorted(feat_summary["feature_type"].unique())
) 

fs = feat_summary[
    (feat_summary["feature_type"].isin(type))
]

st.dataframe(fs.sort_values("coverage_pct"), use_container_width=True)















