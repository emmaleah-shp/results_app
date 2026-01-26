import streamlit as st
import plotly
import plotly.express as px

st.set_page_config(
    page_title="MIUS Results Explorer",
    layout="wide"
)

st.title("Explorador de resultados MIUS")

st.markdown("""
Esta aplicación permite explorar:

- Errores de predicción espacial (hexágonos H3)
- Métricas de rendimiento del modelo (MAE, R²)
- Distribuciones y relaciones de características
- Comparaciones entre iteraciones del modelo

Utilice el panel de navegación de la izquierda para comenzar.
""")

st.divider()
st.subheader("Páginas y descripciones")

st.markdown("""
#### Map Explorer:
Mapa de los errores de predicción espacial y un scatterplot de "True vs Predicted"

#### Metrics: 
Rendimiento de los modelos

#### Feature Explorer: 
Explorar la distribución y descripción de todas las variables, con un mapa y un histograma
""")


