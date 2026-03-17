import streamlit as st
import plotly.express as px
from utils.data_loader import load_metrics

metrics = load_metrics()

st.title("Model Performance Metrics")

use = st.selectbox("Uso del suelo", sorted(metrics["land_use"].unique()))

iteration = st.multiselect(
    "Iteración",
    sorted(metrics["iteration"].unique()),
    default=sorted(metrics["iteration"].unique())
)

cty = st.multiselect(
    "Prueba",
    sorted(metrics["city"].unique()),
    default=sorted(metrics["city"].unique())
)

scale = st.multiselect(
    "Escala",
    sorted(metrics["scale"].unique()),
    default=sorted(metrics["scale"].unique())
)

df = metrics[
    (metrics["land_use"] == use) &
    (metrics["iteration"].isin(iteration)) &
    (metrics["city"] == cty) & 
    (metrics["scale"] == scale) 
]

st.dataframe(df.sort_values("mae"), use_container_width=True)

barshow = st.selectbox("Ver estadística", ["Log R2", "R2","MAE"])
if barshow =="Log R2": 
    fig = px.bar(
        df,
        x="use",
        y="log_r2",
        color="iteration",
        barmode="group",
        title="Log R2 by Model and Iteration",
    )
    
elif barshow =="MAE": 
    fig = px.bar(
        df,
        x="use",
        y="mae",
        color="iteration",
        barmode="group",
        title="MAE by Model and Iteration",
    )
    
elif barshow =="R2": 
    fig = px.bar(
        df,
        x="use",
        y="r2",
        color="iteration",
        barmode="group",
        title="R2 by Model and Iteration",
    )

st.plotly_chart(fig, use_container_width=True)



