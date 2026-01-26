import streamlit as st
import plotly.express as px
from utils.data_loader import load_gdf_features, load_feature_summary

gdf = load_gdf_features()
feat_summary = load_feature_summary()
st.title("Explorador de variables")

cities = sorted(gdf["city"].dropna().unique())
city = st.selectbox("Ciudad", cities)

FEATURE_COLS = [c for c in gdf.columns if c.startswith("feat_")]
feature_col = st.selectbox("Feature", FEATURE_COLS) 
st.subheader("Documentación de las variables (features)")

feat_row = feat_summary.loc[
    feat_summary["feature_name"] == feature_col
]

if feat_row.empty:
    st.warning("No se ha encontrado documentación para esta variable.")
else:
    feat_row = feat_row.iloc[0]

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown(f"### **Nombre de feature** `{feat_row['feature_name']}`")
        st.markdown("##### **Descripción (en inglés):**")
        st.write(feat_row["description"])

    with col2:
        #st.subheader(f"**Número de valores NON cero:** `{feat_row['non_zero_count']:,}`")  
        st.markdown(f"##### **Número de valores NON cero:** `{feat_row['non_zero_count']:,}`")
        st.markdown(f"""
        **Cobertura (%):** {feat_row['coverage_pct']:.1f}%    
        **Rango de datos:** {feat_row['data_range']}  
        **Tipo:** {feat_row['feature_type']} 
        """)

df = gdf[gdf["city"] == city].copy()

fig_map = px.choropleth_mapbox(
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

fig_map.update_layout(
    height=900  # try 650–850 depending on screen
    )

st.plotly_chart(fig_map, use_container_width=True) 

st.subheader("Distribución de features")

# Controls
col1, col2 = st.columns(2)

with col1:
    bins = st.slider("Bins", min_value=5, max_value=50, value=15)

with col2:
    normalize = st.checkbox("Normalizar (densidad)", value=False)

# Histogram
fig = px.histogram(
    gdf, # to use data filtered by city, use df
    x=feature_col,
    nbins=bins,
    histnorm="probability density" if normalize else None,
    marginal="box",               # shows boxplot above histogram
    template="plotly_white",
    title=f"Distribución de {feature_col}",
)

fig.update_layout(
    bargap=0.05,
    xaxis_title=feature_col,
    yaxis_title="Densidad" if normalize else "Conteo",
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

type = st.multiselect(
    "Tipo de variable",
    sorted(feat_summary["feature_type"].unique()),
    default=sorted(feat_summary["feature_type"].unique())
) 

target_cols = ['target_residential_m2', 'target_commercial_m2', 'target_office_m2']
select_cols = list(set(FEATURE_COLS + target_cols))

fs = feat_summary[
    (feat_summary["feature_type"].isin(type)) &
    (feat_summary["feature_name"].isin(select_cols))
]

st.dataframe(fs.sort_values("coverage_pct"), use_container_width=True)





