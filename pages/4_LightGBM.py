import streamlit as st
import plotly.express as px
from utils.data_loader import load_metrics


st.title("Model Selected: Light GBM")

st.markdown("""
## Model Overview: Hierarchical LightGBM Regression

### Purpose
This model estimates built area (m²) by land-use category at the hexagon level.  
Many hexagons contain **no area for a given land use**, while others contain **highly skewed, continuous values**.  
To address this, the model separates **presence detection** from **area estimation** using a hierarchical approach.

---

### Model Architecture

For each land-use category, the model is trained in **two stages**:

**1. Presence Detection (Binary Classification)**  
- Predicts whether a land use is present in a hexagon (m² > 0)
- Handles strong class imbalance and zero-inflation
- Outputs a probability of presence

**2. Area Estimation (Conditional Regression)**  
- Trained only on hexagons where the land use exists
- Predicts log-transformed built area
- Log transformation reduces skewness and limits the influence of extreme values

**Final Prediction**  
The expected built area is computed as:

""")

st.divider()
st.markdown(""" 

This yields a continuous estimate while preserving realistic sparsity.

---

### Why LightGBM?

LightGBM is well suited to this problem because it:

- Efficiently learns **nonlinear relationships and feature interactions**
- Handles **sparse and imbalanced targets** effectively
- Scales well when training **separate models per land-use category**
- Includes regularization to reduce overfitting in high-dimensional feature spaces

Compared to bagged tree methods, gradient boosting achieves similar or better accuracy with fewer trees and lower computational cost.

---

### Simplified Model Behavior

In plain terms, the model:

- First asks **“Does this land use exist here?”**
- If yes, estimates **“How much area does it occupy?”**
- Applies the same logic consistently across all hexagons
- Limits the influence of extreme values
- Produces interpretable, stable predictions for mixed-use areas

---

### Model Variants Considered

| Model ID | Model Type | Hierarchical Strategy | Strengths | Limitations |
|--------|-----------|----------------------|-----------|-------------|
| A#.1 | Random Forest | Binary → Regression | Simple, interpretable baseline | Less efficient for complex interactions |
| A#.2 | XGBoost | Residential dominance split | Encodes explicit hierarchy | Most complex pipeline |
| **A#.3** | **LightGBM** | **Binary → Regression** | **Fast, scalable, stable** | **Less interpretable than RF** |
| A#.4 | CatBoost | Binary → Regression | Robust to noisy features | Slower, less transparent |

---

### Design Rationale

The LightGBM hierarchical model was selected to balance:

- Modeling accuracy
- Computational efficiency
- Pipeline simplicity
- Reproducibility across land-use categories

This structure avoids unnecessary branching while explicitly addressing zero-inflation and mixed-use hexagons.
""")
st.divider()
st.markdown(""" 
### Hyperparameter Optimization with FLAML

Model performance and stability depend not only on the model structure, but also on the choice of hyperparameters (e.g., tree depth, learning rate, number of leaves).  
To avoid manual tuning and to ensure reproducibility, hyperparameter selection is handled automatically using **FLAML**.

---

#### What FLAML Does

FLAML (Fast Lightweight AutoML) performs **automated hyperparameter search** with a focus on:

- Efficient exploration of the hyperparameter space
- Strong performance under limited time or compute budgets
- Avoiding overfitting through early stopping and adaptive search

Rather than exhaustively searching all parameter combinations, FLAML prioritizes promising configurations based on observed performance, allowing it to converge quickly to well-performing settings.

---

#### How FLAML Is Used in This Pipeline

For each LightGBM model (classifier and regressor):

- FLAML searches over a predefined range of LightGBM hyperparameters
- Model performance is evaluated using cross-validation on the training data
- Poor-performing configurations are discarded early
- The best-performing configuration is selected and fixed for training

This process is applied independently for:
- Presence (binary classification) models
- Conditional area (regression) models

---

#### Why FLAML Was Chosen

FLAML was selected because it:

- Reduces the need for manual hyperparameter tuning
- Produces consistent and reproducible model configurations
- Is computationally efficient compared to grid or random search
- Integrates directly with LightGBM

This allows the modeling pipeline to focus on **structure and data design**, while hyperparameter tuning is handled in a principled and automated way.

---

#### Simplified Explanation

In simple terms:

- FLAML automatically tests different LightGBM settings
- Keeps the ones that work well
- Stops searching once further improvements are unlikely
- Ensures the final model is well-tuned without excessive trial-and-error
""")

