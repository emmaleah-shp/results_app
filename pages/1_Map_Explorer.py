import streamlit as st
import numpy as np
import plotly.express as px
from utils.data_loader import load_gdf
from utils.constants import MODEL_ITERS, MODELS, USES

gdf = load_gdf()

st.title("Map & Prediction Explorer")

# ---------------- Sidebar ----------------
st.sidebar.header("Filtros espaciales")

cities = sorted(gdf["city"].dropna().unique())
city = st.sidebar.selectbox("City", cities)

st.sidebar.header("Selección del modelo")

model_iter = st.sidebar.selectbox("Iteración del modelo", MODEL_ITERS)
model = st.sidebar.selectbox("Modelo", MODELS)
use = st.sidebar.selectbox("Uso del suelo", USES)
var_type = st.sidebar.radio("Tipo variable", ["error", "predicción", "true"])



# ---------------- Column logic ----------------
target_col = f"target_{use}_m2"

if var_type == "predicción":
    if model == "nn" or model == "nn_log":
        gdf = gdf.assign(c1_nn_m2 = gdf[target_col] - gdf[f"c1_nn_{use}_error"])
        value_col = "c1_nn_m2"
        hover_col = f"{model_iter}_nn_pct_{use}_error"
    else:
        value_col = f"{model_iter}_{model}_{use}_m2"
        hover_col = f"{model_iter}_{model}_{use}_error"
elif var_type == "error":
    value_col = f"{model_iter}_{model}_{use}_error"
    if model == "nn" or model == "nn_log":
        hover_col = f"{model_iter}_nn_pct_{use}_error"
    else: 
        hover_col = f"{model_iter}_{model}_{use}_m2"
elif var_type == "true":
    value_col = target_col
    hover_col = f"{model_iter}_{model}_{use}_error"

if model == "rf":
    model_name = "Random Forest"
elif model == "xgb":
    model_name = "XGBoost"
elif model == "lgbm":
    model_name = "LightGBM"
elif model == "cat":
    model_name = "CatBoost"
elif model == "nn" or model == "nn_log":
    model_name = "Multilayer Perceptron (red neuronal)"


# ---------------- Filter data ----------------
df = gdf[gdf["city"] == city].copy()

if value_col not in df.columns:
    st.warning(f"Column not found: {value_col}")
    st.stop()

df_city = df.copy()

df_valid = df_city[df_city[value_col].notna()]
df_nan = df_city[df_city[value_col].isna()]


# ---------------- Layout ----------------
col_map, col_scatter = st.columns([1.2, 1])
percentiles, guide = st.columns([0.3, 0.7])
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
    
    if model == "nn" or model == "nn_log":
        df_valid = df_valid.assign(c1_nn_m2 = df_valid[target_col] - df_valid[f"c1_nn_{use}_error"])
        pred_col = "c1_nn_m2"
    else:
        pred_col = f"{model_iter}_{model}_{use}_m2"

    fig_scatter = px.scatter(
        df_valid,
        x=target_col,
        y=pred_col,
        opacity=0.3,
        labels={"x": "True m²", "y": "Predicted m²"},
    )

    max_val = max(df_valid[target_col].max(), df_valid[pred_col].max())

    fig_scatter.add_shape(
        type="line",
        x0=0, y0=0, 
        x1=max_val, y1=max_val,
        line=dict(dash="dash", color="crimson"),
    )

    fig_scatter.update_layout(
        height=800  # try 650–850 depending on screen
    )

    st.plotly_chart(fig_scatter, use_container_width=True)
with percentiles: 
    st.markdown(f"""
                Error percentile for {model_name} model: {use.title()}
                """)
    st.markdown(f"""
                50th percentile: {np.nanpercentile(np.abs(df_valid[value_col]), 50):.2f} 
                """)
    st.markdown(f"""
                75th percentile: {np.nanpercentile(np.abs(df_valid[value_col]), 75):.2f} 
                """)
    st.markdown(f"""
                90th percentile: {np.nanpercentile(np.abs(df_valid[value_col]), 90):.2f}
                """)
    st.markdown(f"""
                Max: {max(np.abs(df_valid[value_col])):.2f}
                """)

with guide: 
    st.markdown(f"""
                #### Iteraciones - Models - Uses
| **Iteraciones**   | **Modelos**                                                        | **Usos**                             | **Total**    |
| ----------------- | ------------------------------------------------------------------ | ------------------------------------ | -------------|
| **A1**            | (4) Decision trees: RandomForest, XGBoost, LightGBM, CatBoost      | (3) Residential, Commercial, Office  | 12           |
| **A2**            | (4) Decision trees (see above) with correlated variable selection  | (3) Residential, Commercial, Office  | 12           |
| **A3**            | (4) Decision trees (see above) with outlier elimination            | (3) Residential, Commercial, Office  | 12           |
| **B1**            | (2) Decision trees: RandomForest & XGBoost                         | (1) Residential                      | 2            |
| **B1_pruned**     | (2) Decision trees: RandomForest & XGBoost, with parameter tuning  | (1) Residential                      | 2            |
| **C1**            | (1) MLP: Multilayer Perceptron with transformed variables + LOG    | (2) Residential & Commercial         | 2            |
| **C2**            | (1) MLP with base variables (no transform) + LOG                   | (2) Residential & Commercial         | 2            |
                """)

st.divider()
st.markdown("""
### Explicación de iteración y modelos:

##### Iteración:
  - A1 = rendimiento original (hierarchical decision tree) sin selección de variables
  - A2 = rendimiento original CON seleccion de variables con alta correlación con los 3 usos 'target'
  - A3 = Eliminar todos los hexágonos outliers (pre-split) que se encuentren fuera del percentil ~99.
  - B1 = Hierarchical decision tree estructura con búsqueda aleatoria en cuadrícula para el ajuste de parámetros.
  - B1_pruned = B1 con análisis de importancia de permutación  y posterior ajuste de características
  - C1 = neural network resultados utilizando variables transformadas (LISA Moran, log1p)
  - C2 = neural network resultados utilizando variables base (sin transformar)

##### Modelos:
  - XGBoost
  - RandomForest
  - Light GBM
  - CatBoost
  - MLP (neural network)

##### Next Steps:
  - Implement soft hierarchical routing 
  - Spatial cross-validation 
  - "distance from center" variable
""")


st.markdown("""
### Notes:

##### B1 Parameter choices: 
        - RandomizedSearchCV
        - Keep raw target, use Tweedie which explicitly models mass at 0 and continuous positive variables
        - shallow trees with a smaller learning rate
        - subsampling to help with correlated but different vars like POIs and spatial redundancy
        - SUGGESTED BUT NOT TAKEN: feature transform / regularization

##### Permutation Importance:
        Scikit-learn's permutation importance is a model inspection technique that measures the contribution of
        each feature to a fitted model’s statistical performance on a given dataset. This technique is particularly
        useful for non-linear or opaque estimators, and involves randomly shuffling the values of a single feature
        and observing the resulting degradation of the model’s score. By breaking the relationship between the
        feature and the target, we determine how much the model relies on such particular feature.
        (Source: https://scikit-learn.org/stable/modules/permutation_importance.html)
        
##### Decision Trees:
        Tree-based Models (Random Forests, Gradient Boosting Machines like XGBoost, LightGBM): These models are generally 
        robust to the distribution shape of the target variable and features because they work by splitting data based on 
        thresholds, rather than assuming a specific underlying distribution. 

| Aspect                   | **XGBoost**                               | **RandomForest**          |
| ------------------------ | ----------------------------------------------- | -------------------------------- |
| **Algorithm**            | XGBoost                                         | RandomForest                     |
| **Hierarchy**            | 3-level decision system                         | 2-level per-use model            |
| **Stage 1**              | Predict *residential-dominant* hex              | Predict *has_use* binary per use |
| **Stage 2A/2B**          | Two separate regressors depending on context    | One regressor per use            |
| **Data split per model** | Split into residential-dominant vs non-dominant | All data used for each use       |
| **Complexity**           | Very high                                       | Low/clean                        |
| **Interpretability**     | Hard                                            | Easy                             |
| **Weights**              | Custom sample weights                           | None                             |
            
| Aspect                                        | **CatBoost**                                                                     | **LightGBM**                                     |
| --------------------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Algorithm Type**                            | Gradient boosting over oblivious decision trees                                  | Gradient boosting over leaf-wise decision trees                        |
| **Handling of Categorical Data**              | Native categorical handling (best-in-class), but you currently feed numeric-only | Requires manual encoding (not relevant for      numeric-only features) |
| **Default Bias**                              | Stronger regularization, tends to avoid overfitting                              | More aggressive fitting, can overfit small samples                     |
| **Performance on Small/Medium Datasets**      | Excellent stability                                                              | Very fast but more sensitive to noise                                  |
| **Performance on Large Feature Sets**         | Good, but slower than LGBM                                                       | Extremely fast, scales very well                                       |
| **Interpretability**                          | Medium (oblivious trees easier to visualize)                                     | Medium–low (leaf-wise trees complex)                                   |
| **Hyperparameter Sensitivity**                | Low (good out-of-the-box results)                                                | High (defaults can underperform for regression)                        |
| **Training Speed**                            | Slower (especially CPU mode)                                                     | Fastest tree booster available                                         |
| **Prediction Speed**                          | Moderate                                                                         | Very fast                                                              |
| **Missing Value Handling**                    | Native                                                                           | Native                                                                 |
| **Monotonic Constraints Support**             | Yes                                                                              | Yes                                                                    |
| **GPU Support**                               | Very strong, often faster than LGBM GPU                                          | Strong GPU support, extremely fast                                     |
| **Robustness to Noisy / Correlated Features** | Very high → good for many features                                               | Medium → benefits from feature reduction         |
| **Works Well with Sparse Data**               | Yes                                                                              | Very strong                                                            |
| **Model Stability Across Splits**             | High                                                                             | Medium (can vary more)                                                 |
| **Good for Hierarchical Setup?**              | Very — handles unbalanced levels well                                            | Yes — excels at regression stage                                       |
| **Binary Classifier Stage Performance**       | Often stronger & more stable probabilities                                       | Fast and competitive but needs tuning                                  |
| **Regression Stage (m²) Performance**         | Good & stable                                                                    | Often best raw accuracy if tuned                                       |
| **Best Use Case in Your Workflow**            | Stage 1 (binary “has_use”) or full pipeline when max stability matters           | Stage 2 (m² regression) when maximizing accuracy                       |
| **Overall Strengths**                         | Stability, robust defaults, good for noisy + small data                          | Speed, high accuracy when tuned, ideal with reduced features           |
| **Overall Weaknesses**                        | Slower, may underfit slightly                                                    | Sensitive to hyperparameters & correlation                             |



""")









