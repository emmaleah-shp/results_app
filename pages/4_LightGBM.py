import streamlit as st
import plotly.express as px
from utils.data_loader import load_metrics


st.title("Model Selected: Light GBM")

st.markdown("""
### Explicación de iteración y modelos:

##### **Hierarchical LightGBM (Binary + Regression)**

Many hexagons are either almost entirely residential or contain residential use alongside several smaller non-residential components. These two regimes exhibit very different relationships between features and land-use intensity, and treating them with a single regression model leads to systematic bias. To address the strong zero-inflation and class imbalance present in the land-use targets we employ a hierarchical (hurdle) modeling strategy using LightGBM decision trees.      

**Architecture:**

For each land-use category, we first train a binary classifier to predict whether each land-use category is present in a hexagon (m² > 0). This explicitly separates the detection problem (does this use exist here?) from the intensity problem (how much area is present if it exists). This is particularly important for residential use, which is more common overall but still appears alongside other uses in mixed-use hexagons. Conditional on presence, we then train a regression model only on hexagons where the use exists (according to the classifier which reaches over 95% accuracy), predicting the log-transformed built area to reduce skewness and stabilize variance (reduce the influence of extreme values). The final prediction is computed as the probability of presence multiplied by the predicted area, yielding a continuous expected m² value.

**LGBM:**		

In the first stage, a LightGBM classifier predicts whether a given land use is present in a hexagon. In the second stage, a LightGBM regressor is trained only on positive samples to predict log-transformed built area, which is then converted back to m² space.        

This approach replaces bagged trees with gradient-boosted decision trees, allowing for more efficient learning of complex nonlinear interactions with fewer trees and better handling of feature interactions. This model does not introduce an additional residential-dominance split; instead, it treats all hexagons uniformly within each land-use–specific model. This means the model is simpler and more directly comparable to the baseline hierarchical approach, while leveraging LightGBM’s speed and regularization to improve performance and scalability.

| Model ID | Model Type    | Hierarchical Strategy                            | Key Stages                                                                                            | Strengths                                                                      | Limitations                                                                                       |
| -------- | ------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| **A#.1** | Random Forest | Binary → Regression (per use)                    | (1) RF classifier for presence<br>(2) RF regressor on log(m²) for positives                           | Simple, interpretable, robust baseline<br>Handles nonlinearities well          | Less efficient at modeling complex interactions<br>Can underperform with many correlated features |
| **A#.2** | XGBoost       | Residential dominance split + per-use regression | (1) Classify residential dominance<br>(2) Separate regressors for residential-dominant vs mixed hexes | Explicitly encodes land-use hierarchy<br>Addresses residential underestimation | Most complex pipeline<br>Higher risk of error propagation                                         |
| **A#.3** | _LightGBM_    | Binary → Regression (per use)                    | (1) LGBM classifier for presence<br>(2) LGBM regressor on log(m²)                                     | Fast, scalable, strong performance<br>Efficient gradient boosting              | Less interpretable than RF<br>No explicit residential hierarchy                                   |
| **A#.4** | CatBoost      | Binary → Regression (per use)                    | (1) CatBoost classifier with class weights<br>(2) CatBoost regressor on log(m²)                       | Robust to noisy features<br>Stable training, minimal preprocessing             | Slower than LightGBM<br>Less transparent model internals                                          |

""")

st.divider()
st.markdown(""" 
| Model ID | Model Type    | Hierarchical Strategy                            | Key Stages                                                                                            | Strengths                                                                      | Limitations                                                                                       |
| -------- | ------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| **A#.1** | Random Forest | Binary → Regression (per use)                    | (1) RF classifier for presence<br>(2) RF regressor on log(m²) for positives                           | Simple, interpretable, robust baseline<br>Handles nonlinearities well          | Less efficient at modeling complex interactions<br>Can underperform with many correlated features |
| **A#.2** | XGBoost       | Residential dominance split + per-use regression | (1) Classify residential dominance<br>(2) Separate regressors for residential-dominant vs mixed hexes | Explicitly encodes land-use hierarchy<br>Addresses residential underestimation | Most complex pipeline<br>Higher risk of error propagation                                         |
| **A#.3** | LightGBM      | Binary → Regression (per use)                    | (1) LGBM classifier for presence<br>(2) LGBM regressor on log(m²)                                     | Fast, scalable, strong performance<br>Efficient gradient boosting              | Less interpretable than RF<br>No explicit residential hierarchy                                   |
| **A#.4** | CatBoost      | Binary → Regression (per use)                    | (1) CatBoost classifier with class weights<br>(2) CatBoost regressor on log(m²)                       | Robust to noisy features<br>Stable training, minimal preprocessing             | Slower than LightGBM<br>Less transparent model internals                                          |


""")