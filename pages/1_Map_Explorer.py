import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    
    if model == "nn" or model == "nn_log":
        df_valid = df_valid.assign(c1_nn_m2 = df_valid[target_col] - df_valid[f"c1_nn_{use}_error"])
        pred_col = "c1_nn_m2"
    else:
        pred_col = f"{model_iter}_{model}_{use}_m2"
    
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
                ##### Error percentile for {model_name} model:     
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

with guide: 
    st.markdown(f"""
                #### Iteraciones - Modelos - Usos
| **Iteraciones**   | **Modelos**                                                        | **Usos**                             | **Total**    |
| ----------------- | ------------------------------------------------------------------ | ------------------------------------ | -------------|
| **A1**            | (4) Decision trees: RandomForest, XGBoost, LightGBM, CatBoost      | (3) Residential, Commercial, Office  | 12           |
| **A2**            | (4) Decision trees (see above) with variable selection             | (3) Residential, Commercial, Office  | 12           |
| **A3**            | (4) Decision trees (see above) with outlier elimination            | (3) Residential, Commercial, Office  | 12           |
| **B1**            | (2) Decision trees: RandomForest & XGBoost                         | (1) Residential                      | 2            |
| **B1_pruned**     | (2) Decision trees: RandomForest & XGBoost, with parameter tuning  | (1) Residential                      | 2            |
| **C1**            | (1) MLP: Multilayer Perceptron with transformed variables + LOG    | (2) Residential & Commercial         | 2            |
| **C2**            | (1) MLP with base variables (no transform) + LOG                   | (2) Residential & Commercial         | 2            |
                """)

st.divider()
st.markdown("""
### Explicación de iteración y modelos:

##### Iteración A#.1 Hierarchical (Hurdle) Random Forest Modeling
Many hexagons are either almost entirely residential or contain residential use alongside several smaller non-residential components. These two regimes exhibit very different relationships between features and land-use intensity, and treating them with a single regression model leads to systematic bias.
To address the strong zero-inflation and class imbalance present in the land-use targets, we employ a hierarchical (hurdle) modeling strategy using Random Forests.         

For each land-use category, we first train a binary classifier to predict whether the use is present in a hexagon (m² > 0). This explicitly separates the detection problem (does this use exist here?) from the intensity problem (how much area is present if it exists). This is particularly important for residential use, which is more common overall but still appears alongside other uses in mixed-use hexagons. 
Conditional on presence, we then train a regression model only on hexagons where the use exists (according to the classifier which reaches over 95% accuracy), predicting the log-transformed built area to reduce skewness and stabilize variance (reduce the influence of extreme values). 
The final prediction is computed as the probability of presence multiplied by the predicted area, yielding a continuous expected m² value.         

Random Forests are used for both stages because they handle nonlinear relationships, interactions, and heterogeneous spatial features well. They also require minimal preprocessing and are robust in imbalanced and sparse-data settings. Balanced class weights are applied in the classifier to prevent majority residential-only hexagons from overwhelming minority cases, while ensemble size (300 trees) improves stability, while shallow regularization / leaf constraints (min_samples_leaf=2) reduces overfitting in sparse regimes, providing stable yet flexible models.        

##### Iteración A#.2 Hierarchical XGBoost with Residential Dominance Splitting        
This model extends the basic binary + regression (hurdle) framework by explicitly accounting for residential dominance in two stages.         

In Stage 1, an XGBoost classifier is trained to identify residential-dominant hexagons, defined as hexes where residential built area exceeds the combined area of all other land uses. This is a more informative split than a simple residential presence indicator, as it separates hexagons that are structurally residential from genuinely mixed-use or non-residential contexts. Class imbalance at this stage is addressed through balanced sample weights.        

In Stage 2, we condition on the predicted dominance regime (Stage 1) and train separate regression models for each land-use category. For residential-dominant hexagons, regressors are trained on the residential subset only, allowing the model to learn how secondary land uses (e.g., small commercial or office components) behave within primarily residential environments. For non-residential or mixed hexagons, a separate set of regressors is trained, capturing fundamentally different spatial and functional patterns. Again, all regressors predict log-transformed built area (to reduce skewness and stabilize learning). Sample weights are used to upweight positive (non-zero) observations—especially for sparse uses such as office and commercial in mixed contexts—so that the models focus on learning meaningful signals rather than minimizing error on zeros. If we didn’t do that, the model would learn to predict zero in all cases and would achieve higher “accuracy.”        

Final predictions follow the same hierarchical logic: each test hexagon is first classified as residential-dominant or not, then routed to the corresponding land-use–specific regressor, and finally transformed back to m² space.     

XGBoost is similarly used throughout due to its strong performance on tabular data, ability to model complex nonlinear interactions, and fine-grained control over bias–variance tradeoffs via depth, learning rate, and subsampling.         

##### Iteración A#.3 Hierarchical LightGBM (Binary + Regression)        
This model follows the same two-stage hierarchical (hurdle) structure as RandomForest (A#.1), where each land-use category is modeled independently using a binary classifier for presence/absence and a regressor for built-area magnitude. In the first stage, a LightGBM classifier predicts whether a given land use is present in a hexagon. In the second stage, a LightGBM regressor is trained only on positive samples to predict log-transformed built area, which is then converted back to m² space.        

Compared to the first Random Forest model, this approach replaces bagged trees with gradient-boosted decision trees, allowing for more efficient learning of complex nonlinear interactions with fewer trees and better handling of feature interactions. Unlike XGBoost, this model does not introduce an additional residential-dominance split; instead, it treats all hexagons uniformly within each land-use–specific model. This makes Model A#.3 simpler and more directly comparable to the baseline hierarchical approach, while leveraging LightGBM’s speed and regularization to improve performance and scalability.

##### Iteración A#.4 — Hierarchical CatBoost (Binary + Regression)        
This model preserves the same two-stage hierarchical structure used in Models A#.1 and A#.3, with a binary classification stage to predict land-use presence followed by a regression stage to estimate built area conditional on presence. For each land-use category, a CatBoost classifier first estimates the probability that the use exists in a hexagon, and a CatBoost regressor is then trained on positive samples to predict log-transformed built area, which is converted back to square meters at inference time.        

Relative to Random Forest and LightGBM, CatBoost introduces ordered boosting and symmetric tree structures, which improve robustness to overfitting and reduce sensitivity to feature scaling and noisy predictors. Class weights are explicitly applied in the classification stage to emphasize positive (present) samples, while the regression stage focuses on learning conditional magnitude. This model serves as a strong gradient-boosted baseline that retains the interpretability and modularity of the hierarchical framework while offering improved stability in heterogeneous feature spaces.        

| Model ID | Model Type    | Hierarchical Strategy                            | Key Stages                                                                                            | Strengths                                                                      | Limitations                                                                                       |
| -------- | ------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| **A#.1** | Random Forest | Binary → Regression (per use)                    | (1) RF classifier for presence<br>(2) RF regressor on log(m²) for positives                           | Simple, interpretable, robust baseline<br>Handles nonlinearities well          | Less efficient at modeling complex interactions<br>Can underperform with many correlated features |
| **A#.2** | XGBoost       | Residential dominance split + per-use regression | (1) Classify residential dominance<br>(2) Separate regressors for residential-dominant vs mixed hexes | Explicitly encodes land-use hierarchy<br>Addresses residential underestimation | Most complex pipeline<br>Higher risk of error propagation                                         |
| **A#.3** | LightGBM      | Binary → Regression (per use)                    | (1) LGBM classifier for presence<br>(2) LGBM regressor on log(m²)                                     | Fast, scalable, strong performance<br>Efficient gradient boosting              | Less interpretable than RF<br>No explicit residential hierarchy                                   |
| **A#.4** | CatBoost      | Binary → Regression (per use)                    | (1) CatBoost classifier with class weights<br>(2) CatBoost regressor on log(m²)                       | Robust to noisy features<br>Stable training, minimal preprocessing             | Slower than LightGBM<br>Less transparent model internals                                          |

""")


st.markdown("""
### Notes:

##### Variable Selection (A2 and A4): 
        - Step 1: Create a df with all the feature correlations and filter to correlations >0.80. Select the first variable that appears in \n
        this list and eliminate the covarying feature. 
            Correlated features dropped: 45 
            Number of variables after correlation filter: 42
        - Step 2: Perform a Variance analysis (from sklearn.feature_selection) with a conservative threshold of 0.01. VarianceThreshold is a \n
        feature selector that removes all low-variance features. This feature selection algorithm looks only at the features (X), not the desired \n
        outputs (y), and can thus be used for unsupervised learning. 
            Low-variance features removed: 4
            After variance filter: 38
        - Step 3: Train an importance model using XGBoost regressor to assess the importance of the 38 variables remaining after Step 2. \n
        Output is a df with importance values, which were ranked. 
        - Step 4: Manually select the features based on the two prior analyses, plus a contextual understanding of inputs as well as previous \n
        correlation analysis between each of the targets and the features. 
        See Feature Explorer for more details.

##### Outlier Elimination (A3 and A4): 
        - hexagons["target_commercial_m2"] <= 10000] # 99.495 percentile
        - hexagons["target_office_m2"] <= 5000] # 99.751 percentile
        - hexagons["target_residential_m2"] <= 8250] #99.02 percentile

    **Stats pre-outlier elimination:**
        N of TEST cells: 2749        
        N of train cells: 10928        
        Total cells: 13677        
        =========RESIDENTIAL=======================================
        Average area for target_residential_m2 TEST cells: 1305.72
        Average area for target_residential_m2 train cells: 1327.23
        Average area for target_residential_m2 TEST cells EXCLUDING 0s: 1990.81
        Average area for target_residential_m2 train cells EXCLUDING 0s: 8044.35
        N of cells with target_residential_m2 > 0 in TEST: 1803, or 65.587%
        N of cells with target_residential_m2 > 0 in train: 6971, or 63.790%
        =========COMMERCIAL=======================================
        Average area for target_commercial_m2 TEST cells: 252.93
        Average area for target_commercial_m2 train cells: 291.62
        Average area for target_commercial_m2 TEST cells EXCLUDING 0s: 1671.40
        Average area for target_commercial_m2 train cells EXCLUDING 0s: 7660.54
        N of cells with target_commercial_m2 > 0 in TEST: 416, or 15.133%
        N of cells with target_commercial_m2 > 0 in train: 1881, or 17.213%
        =========OFFICE=======================================
        Average area for target_office_m2 TEST cells: 53.94
        Average area for target_office_m2 train cells: 63.28
        Average area for target_office_m2 TEST cells EXCLUDING 0s: 1029.67
        Average area for target_office_m2 train cells EXCLUDING 0s: 4801.90
        N of cells with target_office_m2 > 0 in TEST: 144, or 5.238%
        N of cells with target_office_m2 > 0 in train: 673, or 6.158%
    
    **Stats post-outlier elimination:**
        N of TEST cells: 2708        
        N of train cells: 10737        
        Total cells: 13445        
        =========RESIDENTIAL=======================================
        Average area for target_residential_m2 TEST cells: **1231.76**
        Average area for target_residential_m2 train cells: **1218.30**
        Average area for target_residential_m2 TEST cells EXCLUDING 0s: **1880.27**
        Average area for target_residential_m2 train cells EXCLUDING 0s: **7373.68**
        N of cells with target_residential_m2 > 0 in TEST: 1774, or 65.510%
        N of cells with target_residential_m2 > 0 in train: 6807, or 63.398%
        =========COMMERCIAL=======================================
        Average area for target_commercial_m2 TEST cells: **138.39**
        Average area for target_commercial_m2 train cells: **180.51**
        Average area for target_commercial_m2 TEST cells EXCLUDING 0s: **958.48**
        Average area for target_commercial_m2 train cells EXCLUDING 0s: **4956.80**
        N of cells with target_commercial_m2 > 0 in TEST: 391, or 14.439%
        N of cells with target_commercial_m2 > 0 in train: 1750, or 16.299%
        =========OFFICE=======================================
        Average area for target_office_m2 TEST cells: **30.46**
        Average area for target_office_m2 train cells: **39.02**
        Average area for target_office_m2 TEST cells EXCLUDING 0s: **634.48**
        Average area for target_office_m2 train cells EXCLUDING 0s: **3223.03**
        N of cells with target_office_m2 > 0 in TEST: 130, or 4.801%
        N of cells with target_office_m2 > 0 in train: 601, or 5.597%    

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
""")

















