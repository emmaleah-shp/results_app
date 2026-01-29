import streamlit as st
import plotly.express as px
from utils.data_loader import load_metrics

metrics = load_metrics()

st.title("Model Performance Metrics")

use = st.selectbox("Uso del suelo", sorted(metrics["use"].unique()))
model = st.multiselect(
    "Modelo",
    sorted(metrics["model"].unique()),
    default=sorted(metrics["model"].unique())
)

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


iteration = st.multiselect(
    "Iteración",
    sorted(metrics["model_iteration"].unique()),
    default=sorted(metrics["model_iteration"].unique())
)

df = metrics[
    (metrics["use"] == use) &
    (metrics["model"].isin(model)) &
    (metrics["model_iteration"].isin(iteration))
]

st.dataframe(df.sort_values("MAE"), use_container_width=True)

barshow = st.selectbox("Ver estadística", ["R2","MAE"])
if barshow =="MAE": 
    fig = px.bar(
        df,
        x="model",
        y="MAE",
        color="model_iteration",
        barmode="group",
        title="MAE by Model and Iteration",
    )
    
else: 
    fig = px.bar(
        df,
        x="model",
        y="R2",
        color="model_iteration",
        barmode="group",
        title="R2 by Model and Iteration",
    )

st.plotly_chart(fig, use_container_width=True)



