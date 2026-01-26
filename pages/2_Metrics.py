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

fig = px.bar(
    df,
    x="model",
    y="MAE",
    color="model_iteration",
    barmode="group",
    title="MAE by Model and Iteration",
)

st.plotly_chart(fig, use_container_width=True)

