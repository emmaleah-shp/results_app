import streamlit as st
import plotly.express as px
from utils.data_loader import load_metrics

metrics = load_metrics()

st.title("Model Performance Metrics")

use = st.selectbox("Uso del suelo", sorted(metrics["land_use"].unique()))

cty = st.selectbox("Prueba", sorted(metrics["city"].unique()))

iteration = st.multiselect(
    "Iteración",
    sorted(metrics["iteration"].unique()),
    default=sorted(metrics["iteration"].unique())
)

scale = st.multiselect(
    "Escala",
    sorted(metrics["scale"].unique()),
    default=sorted(metrics["scale"].unique())
)

df = metrics[
    (metrics["land_use"] == use) &
    (metrics["city"] == cty) & 
    (metrics["iteration"].isin(iteration)) &
    (metrics["scale"].isin(scale)) 
]

st.dataframe(df.sort_values("mae"), use_container_width=True)

barshow = st.selectbox("Ver estadística", ["Log R2", "R2","MAE"])
if barshow =="Log R2": 
    fig = px.bar(
        df,
        x="scale",
        y="log_r2",
        color="iteration",
        barmode="group",
        title="Log R2 by Scale and Iteration",
    )
    
elif barshow =="MAE": 
    fig = px.bar(
        df,
        x="scale",
        y="mae",
        color="iteration",
        barmode="group",
        title="MAE by Scale and Iteration",
    )
    
elif barshow =="R2": 
    fig = px.bar(
        df,
        x="scale",
        y="r2",
        color="iteration",
        barmode="group",
        title="R2 by Scale and Iteration",
    )

st.plotly_chart(fig, use_container_width=True)



