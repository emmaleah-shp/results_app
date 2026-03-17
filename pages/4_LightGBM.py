import streamlit as st
import plotly.express as px
from utils.data_loader import load_metrics


st.title("Modelo Seleccionado: Light GBM")

st.markdown("""
## Descripción del Modelo: Regresión Jerárquica con LightGBM

### Propósito
Este modelo estima el área construida (m²) por categoría de uso de suelo a nivel de hexágono.  
Muchos hexágonos **no contienen área para un uso de suelo dado**, mientras que otros presentan **valores continuos con alta asimetría**.  
Para abordar esto, el modelo separa la **detección de presencia** de la **estimación de área** mediante un enfoque jerárquico.

---

### Arquitectura del Modelo

Para cada categoría de uso de suelo, el modelo se entrena en **dos etapas**:

**1. Detección de Presencia (Clasificación Binaria)**  
- Predice si un uso de suelo está presente en un hexágono (m² > 0)
- Maneja el fuerte desbalance de clases y la inflación de ceros
- Entrega una probabilidad de presencia

**2. Estimación de Área (Regresión Condicional)**  
- Entrenado solo en hexágonos donde el uso de suelo existe
- Predice el área construida transformada logarítmicamente
- La transformación logarítmica reduce la asimetría y limita la influencia de valores extremos

**Predicción Final**  
El área construida esperada se calcula como:

""")

st.divider()
st.markdown(""" 

Esto produce una estimación continua preservando una dispersión realista.

---

### ¿Por qué LightGBM?

LightGBM es adecuado para este problema porque:

- Aprende eficientemente **relaciones no lineales e interacciones entre variables**
- Maneja eficazmente **objetivos dispersos y desbalanceados**
- Escala bien al entrenar **modelos separados por categoría de uso de suelo**
- Incluye regularización para reducir el sobreajuste en espacios de alta dimensionalidad

En comparación con métodos de árboles con bagging, el gradient boosting logra precisión similar o superior con menos árboles y menor costo computacional.

---

### Comportamiento Simplificado del Modelo

En términos simples, el modelo:

- Primero pregunta **"¿Existe este uso de suelo aquí?"**
- Si es así, estima **"¿Cuánta área ocupa?"**
- Aplica la misma lógica de forma consistente en todos los hexágonos
- Limita la influencia de valores extremos
- Produce predicciones interpretables y estables para áreas de uso mixto

---

### Variantes de Modelo Consideradas

| ID Modelo | Tipo de Modelo | Estrategia Jerárquica | Fortalezas | Limitaciones |
|--------|-----------|----------------------|-----------|-------------|
| A#.1 | Random Forest | Binario → Regresión | Línea base simple e interpretable | Menos eficiente para interacciones complejas |
| A#.2 | XGBoost | División por dominancia residencial | Codifica jerarquía explícita | Pipeline más complejo |
| **A#.3** | **LightGBM** | **Binario → Regresión** | **Rápido, escalable, estable** | **Menos interpretable que RF** |
| A#.4 | CatBoost | Binario → Regresión | Robusto ante variables ruidosas | Más lento, menos transparente |

---

### Justificación del Diseño

El modelo jerárquico con LightGBM fue seleccionado para equilibrar:

- Precisión del modelo
- Eficiencia computacional
- Simplicidad del pipeline
- Reproducibilidad entre categorías de uso de suelo

Esta estructura evita ramificaciones innecesarias mientras aborda explícitamente la inflación de ceros y los hexágonos de uso mixto.
""")

st.divider()
st.markdown(""" 
### Optimización de Hiperparámetros con FLAML

El rendimiento y la estabilidad del modelo dependen no solo de la estructura del modelo, sino también de la elección de los hiperparámetros (p. ej., profundidad de árboles, tasa de aprendizaje, número de hojas).  
Para evitar la sintonización manual y garantizar la reproducibilidad, la selección de hiperparámetros se realiza automáticamente mediante **FLAML**.

---

#### Qué hace FLAML

FLAML (Fast Lightweight AutoML) realiza una **búsqueda automatizada de hiperparámetros** con énfasis en:

- Exploración eficiente del espacio de hiperparámetros
- Alto rendimiento bajo presupuestos limitados de tiempo o cómputo
- Evitar el sobreajuste mediante parada temprana y búsqueda adaptativa

En lugar de buscar exhaustivamente todas las combinaciones de parámetros, FLAML prioriza configuraciones prometedoras basándose en el rendimiento observado, lo que le permite converger rápidamente hacia configuraciones de buen desempeño.

---

#### Cómo se usa FLAML en este Pipeline

Para cada modelo LightGBM (clasificador y regresor):

- FLAML busca sobre un rango predefinido de hiperparámetros de LightGBM
- El rendimiento del modelo se evalúa mediante validación cruzada en los datos de entrenamiento
- Las configuraciones de bajo rendimiento se descartan tempranamente
- La configuración de mejor rendimiento se selecciona y fija para el entrenamiento

Este proceso se aplica de forma independiente para:
- Modelos de presencia (clasificación binaria)
- Modelos de área condicional (regresión)

---

#### Por qué se eligió FLAML

FLAML fue seleccionado porque:

- Reduce la necesidad de sintonización manual de hiperparámetros
- Produce configuraciones de modelo consistentes y reproducibles
- Es computacionalmente eficiente en comparación con búsqueda en grilla o aleatoria
- Se integra directamente con LightGBM

Esto permite que el pipeline de modelado se enfoque en el **diseño de estructura y datos**, mientras que la sintonización de hiperparámetros se maneja de forma fundamentada y automatizada.

---

#### Explicación Simplificada

En términos simples:

- FLAML prueba automáticamente diferentes configuraciones de LightGBM
- Conserva las que funcionan bien
- Detiene la búsqueda una vez que mejoras adicionales son poco probables
- Garantiza que el modelo final esté bien ajustado sin excesivo ensayo y error
""")
