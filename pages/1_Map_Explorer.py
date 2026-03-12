import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_gdf
from utils.constants import MODEL_ITERS, CITIES, USES

st.title("Map & Prediction Explorer")   
# ---------------- Sidebar ----------------
st.sidebar.header("Filtros espaciales")
city = st.sidebar.selectbox("City", CITIES)
scale = st.sidebar.radio("Escala", ["h9", "h10"])

if city in ["medellin","pucon","curico"]:
    place = city 
else: 
    place = "all"

gdf = load_gdf(place, scale)

st.sidebar.header("Selección del modelo")

model_iter = st.sidebar.selectbox("Iteración del modelo", MODEL_ITERS) 
use = st.sidebar.selectbox("Uso del suelo", USES)
var_type = st.sidebar.radio("Tipo variable", ["error", "predicción", "true"])



# ---------------- Column logic ----------------
target_col = f"target_{use}_m2"

if var_type == "predicción":
    if place == "all":
        value_col = f"flaml_{use}_{model_iter}_pred_m2"
        hover_col = f"flaml_{use}_{model_iter}_error"
    else:
        value_col = f"flaml_{city}_{use}_{model_iter}_pred_m2"
        hover_col = f"flaml_{city}_{use}_{model_iter}_error"
elif var_type == "error":
    if place == "all":
        value_col = f"flaml_{use}_{model_iter}_error"
        hover_col = f"flaml_{use}_{model_iter}_pred_m2"
    else:
        value_col = f"flaml_{city}_{use}_{model_iter}_error"
        hover_col = f"flaml_{city}_{use}_{model_iter}_pred_m2"
elif var_type == "true":
    value_col = target_col
    if place == "all":
        hover_col = f"flaml_{use}_{model_iter}_pred_m2"
    else:
        hover_col = f"flaml_{city}_{use}_{model_iter}_pred_m2"

if city == "temuco":
    city_name = "Temuco"
elif city == "valdivia":
    city_name = "Valdivia"
elif city == "osorno":
    city_name = "Osorno"
elif city == "losangeles":
    city_name = "Los Ángeles"
elif city == "talca":
    city_name = "Talca"
elif city == "puertovaras":
    city_name = "Puerto Varas"
elif city == "pucon":
    city_name = "Pucón"
elif city == "curico":
    city_name = "Curicó"
elif city == "medellin":
    city_name = "Medellín"


# ---------------- Filter data ----------------
df = gdf[gdf["city"] == city].copy()

if value_col not in df.columns:
    st.warning(f"Column not found: {value_col}")
    st.stop()

df_city = df.copy()

df_valid = df_city[df_city[value_col].notna()]
df_nan = df_city[df_city[value_col].isna()]


# ---------------- Layout ----------------
col_map, col_scatter = st.columns([0.6, 0.4])
percentiles, guide = st.columns([0.35, 0.65])
# ---------------- Map ---------------- 

with col_map:
    st.subheader("Distribución espacial")
    st.markdown(f"###### Modeling _{value_col}_")
    max_abs = max(
        abs(df_valid[value_col].min()),
        abs(df_valid[value_col].max()),
    )


    fig_map = px.choropleth_mapbox(
        df_valid,
        geojson=df_valid.geometry,
        locations=df_valid.index,
        color=value_col,
        hover_data={
            target_col: True,
            value_col: ':.2f',
            hover_col: ':.2f'
        },
        mapbox_style="carto-positron",
        zoom=11.5,
        center={
            "lat": df_valid.geometry.centroid.y.mean(),
            "lon": df_valid.geometry.centroid.x.mean(),
        },
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
    )
    
    fig_map.update_traces(
        zmin=-max_abs,
        zmax=max_abs,
        marker_line_width=0.2,
        marker_line_color="rgba(0,0,0,0.2)",
    )
    if len(df_nan) > 0:
        fig_nan = px.choropleth_mapbox(
            df_nan,
            geojson=df_nan.geometry,
            locations=df_nan.index,
            color_discrete_sequence=["rgba(0,0,0,0)"],  # fully transparent fill
            hover_data={value_col: False},
        )
    
        fig_nan.update_traces(
            marker_line_color="black",
            marker_line_width=1.0,
            showlegend=False,
        )
    
        for trace in fig_nan.data:
            fig_map.add_trace(trace)

    st.caption(
        "_*Error = True − Prediction, i.e. Positivo=Subestimación | Negativo=Sobreestimación_"
    )

    fig_map.update_layout(
        height=800  # try 650–850 depending on screen
    )

    st.plotly_chart(fig_map, use_container_width=True)


# ---------------- Scatter ----------------
with col_scatter:
    st.subheader("True vs Predicted")
    if place == "all":
        pred_col = f"flaml_{use}_{model_iter}_pred_m2"
    else:
        pred_col = f"flaml_{city}_{use}_{model_iter}_pred_m2"
    
    # Scatter plot
    fig_scatter = px.scatter(
        df_valid,
        x=target_col,
        y=pred_col,
        opacity=0.3,
        labels={"x": "True m²", "y": "Predicted m²"},
    )
    
    # Limits
    max_val = max(
        df_valid[target_col].max(),
        df_valid[pred_col].max()
    )
    
    fig_scatter.update_xaxes(range=[0, max_val])
    fig_scatter.update_yaxes(range=[0, max_val])
    
    # --------------------------------------------------
    # Ideal line (y = x) — BLACK dashed
    # --------------------------------------------------
    fig_scatter.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Ideal (y = x)",
            line=dict(color="black", dash="dash"),
        )
    )
    
    # --------------------------------------------------
    # Best-fit line (y = ax + b) — RED solid
    # --------------------------------------------------
    y_true = df_valid[target_col].values
    y_pred = df_valid[pred_col].values
    
    a, b = np.polyfit(y_true, y_pred, 1)
    
    x_line = np.array([0, max_val])
    y_line = a * x_line + b
    
    fig_scatter.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"Fit: y = {a:.2f}x + {b:.1f}",
            line=dict(color="crimson", width=3),
        )
    )
    
    # Layout
    fig_scatter.update_layout(
        height=800,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
with percentiles: 
    st.markdown(f"""
                ##### Error percentile for {city_name} model:     
                ##### _{use.title()}_
                """)
    st.markdown(f"""
                ##### 50th percentile: {np.nanpercentile(np.abs(df_valid[value_col]), 50):.2f} 
                """)
    st.markdown(f"""
                ##### 75th percentile: {np.nanpercentile(np.abs(df_valid[value_col]), 75):.2f} 
                """)
    st.markdown(f"""
                ##### 90th percentile: {np.nanpercentile(np.abs(df_valid[value_col]), 90):.2f}
                """)
    st.markdown(f"""
                ##### Max: {max(np.abs(df_valid[value_col])):.2f}
                """)

# with guide: 
st.markdown("""
### Notes:

##### Variable Selection (A2 and A4): 
        - Step 1: Create a df with all the feature correlations and filter to correlations >0.80. Selects the first variable that appears 
        in this list and eliminates the covarying feature. 
            Correlated features dropped: 45 
            Number of variables after correlation filter: 42
        - Step 2: Perform a Variance analysis (from sklearn.feature_selection) with a conservative threshold of 0.01. 
        VarianceThreshold is a feature selector that removes all low-variance features. This feature selection algorithm looks 
        only at the features (X), not the desired outputs (y), and can thus be used for unsupervised learning. 
            Low-variance features removed: 4
            After variance filter: 38
        - Step 3: Train an importance model using XGBoost regressor to assess the importance of the 38 variables remaining after Step 2. 
        Output is a df with importance values, which were ranked. 
        - Step 4: Manually select the features based on the two prior analyses, plus a contextual understanding of inputs as well as previous 
        correlation analysis between each of the targets and the features. 
        See Feature Explorer for more details.

##### Outlier Elimination (A3 and A4): 
    - hexagons["target_commercial_m2"] <= 10000] # 99.495 percentile
    - hexagons["target_office_m2"] <= 5000] # 99.751 percentile
    - hexagons["target_residential_m2"] <= 8250] #99.02 percentile
    - hexagons["target_commercial_office_m2"] <= 10500] #99.45 percentile

| Metric | Pre-Outlier Removal | Post-Outlier Removal |
|------|------|------|
| Test Cells | 4851 | 4787 |
| Train Cells | 19299 | 19014 |
| Total Cells | 24150 | 23801 |

---

## Residential

| Metric | Pre Test | Post Test | Pre Train | Post Train |
|------|------|------|------|------|
| Avg Area (m²) | 1317.71 | 1224.33 | 1260.77 | 1174.16 |
| Avg Area Excluding 0s (m²) | 1947.06 | 1814.51 | 7411.40 | 6911.91 |
| Cells > 0 | 3283 (67.677%) | 3230 (67.474%) | 12271 (63.584%) | 12033 (63.285%) |

---

## Commercial

| Metric | Pre Test | Post Test | Pre Train | Post Train |
|------|------|------|------|------|
| Avg Area (m²) | 215.86 | 156.82 | 257.07 | 156.22 |
| Avg Area Excluding 0s (m²) | 1342.48 | 1010.34 | 6360.40 | 3997.76 |
| Cells > 0 | 780 (16.079%) | 743 (15.521%) | 3080 (15.959%) | 2895 (15.226%) |

---

## Office

| Metric | Pre Test | Post Test | Pre Train | Post Train |
|------|------|------|------|------|
| Avg Area (m²) | 52.46 | 35.96 | 57.85 | 35.46 |
| Avg Area Excluding 0s (m²) | 912.05 | 659.57 | 4001.36 | 2583.22 |
| Cells > 0 | 279 (5.751%) | 261 (5.452%) | 1068 (5.534%) | 968 (5.091%) |

---

## Commercial + Office

| Metric | Pre Test | Post Test | Pre Train | Post Train |
|------|------|------|------|------|
| Avg Area (m²) | 183.63 | 192.78 | 188.72 | 191.68 |
| Avg Area Excluding 0s (m²) | 1032.08 | 1079.34 | 4215.53 | 4262.64 |
| Cells > 0 | 851 (17.792%) | 855 (17.861%) | 3238 (17.034%) | 3243 (17.056%) |

---

## Total Built Area

| Metric | Pre Test | Post Test | Pre Train | Post Train |
|------|------|------|------|------|
| Avg Area (m²) | 2044.43 | 1844.50 | 1975.09 | 1722.47 |
| Avg Area Excluding 0s (m²) | 2752.57 | 2494.95 | 10579.30 | 9254.33 |
| Cells > 0 | 3603 (74.273%) | 3539 (73.929%) | 13419 (69.532%) | 13134 (69.075%) |

## Medellín

| Land use | Metric | Total |
|------|------|------|
| Residential | Avg Area (m²) | 6039.39 | 
| Residential | Avg Area Excluding 0s (m²) | 7706.66 |
| Residential | Cells > 0 | 5716 (78.366%) |
| Commercial + Office | Avg Area (m²) | 1412.83 | 
| Commercial + Office | Avg Area Excluding 0s (m²) | 2554.58 |
| Commercial + Office | Cells > 0 | 4034 (55.306%) |
| Industrial | Avg Area (m²) | 359.79 | 
| Industrial | Avg Area Excluding 0s (m²) | 1782.81 |
| Industrial | Cells > 0 | 1472 (20.181%) |
| Total | Avg Area (m²) | 12835.73 | 
| Total | Avg Area Excluding 0s (m²) | 13313.96 |
| Total | Cells > 0 | 7032 (96.408%) |

---
##### Decision Trees:
        Tree-based Models (Random Forests, Gradient Boosting Machines like XGBoost, LightGBM): These models are generally 
        robust to the distribution shape of the target variable and features because they work by splitting data based on 
        thresholds, rather than assuming a specific underlying distribution. 
""")







