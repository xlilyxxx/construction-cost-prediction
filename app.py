
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Construction Cost Predictor",
    page_icon="🏗️",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    with open("feature_columns.json", "r") as f:
        features = json.load(f)
    return model, features

st.sidebar.title("🏗️ Cost Predictor")
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Cost Prediction",
        "Model Explanation",
        "Dataset & Model",
        "Program Relevance"
    ]
)

st.title("🏗️ Residential Construction Cost Prediction")
st.caption("AI-assisted cost estimation using Random Forest")

try:
    model, feature_columns = load_model()
    model_ready = True
except Exception:
    model_ready = False

if page == "Home":
    st.header("Project Overview")
    st.write(
        "This project explores machine learning for predicting "
        "residential construction costs using a dataset of building "
        "characteristics and economic indicators."
    )
    st.info(
        "Enter project information on the Cost Prediction page "
        "to explore a model-based estimate."
    )
    st.subheader("Project Features")
    st.markdown(
        "- Construction cost prediction\n"
        "- Model explanation and feature importance\n"
        "- Dataset and model information\n"
        "- Construction and civil engineering relevance"
    )

elif page == "Cost Prediction":
    st.header("Construction Cost Prediction")
    st.write(
        "Enter values for the model inputs. For this initial version, "
        "all model features are exposed as numeric inputs."
    )

    if not model_ready:
        st.error(
            "The model files are missing. Upload "
            "random_forest_model.pkl and feature_columns.json "
            "to the GitHub repository."
        )
    else:
        st.warning(
            "This is a prototype. Inputs must use the same units "
            "and definitions as the training dataset. The estimate "
            "is not a professional quantity-surveying quotation."
        )

        with st.form("prediction_form"):
            values = {}
            columns = st.columns(2)

            for i, feature in enumerate(feature_columns):
                with columns[i % 2]:
                    values[feature] = st.number_input(
                        feature,
                        value=0.0,
                        format="%.4f",
                        key=f"input_{i}"
                    )

            submitted = st.form_submit_button("Predict Construction Cost")

        if submitted:
            input_df = pd.DataFrame(
                [[values[f] for f in feature_columns]],
                columns=feature_columns
            )
            prediction = float(model.predict(input_df)[0])

            st.subheader("Predicted Construction Cost")
            st.metric("Model output", f"{prediction:,.2f}")
            st.caption(
                "The value is displayed in the target's dataset unit. "
                "Confirm the unit before interpreting it as currency."
            )

elif page == "Model Explanation":
    st.header("Model Explanation")
    st.write(
        "Feature importance indicates which input variables the "
        "Random Forest model relies on most. It does not establish "
        "causation."
    )

    if model_ready and hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(15)

        st.dataframe(importance, use_container_width=True)

        chart_data = importance.sort_values("Importance")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(chart_data["Feature"], chart_data["Importance"])
        ax.set_xlabel("Feature importance")
        ax.set_title("Top 15 Model Features")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Model files are needed to display feature importance.")

elif page == "Dataset & Model":
    st.header("Dataset and Model Information")
    st.write("Dataset: Residential Building Data Set")
    st.write("Target variable: V-10 — Actual construction costs")
    st.write("Candidate models: Random Forest and XGBoost")
    st.write("Initial evaluation used an 80:20 train-test split.")
    st.warning(
        "The initial scores require further validation for feature "
        "availability and possible data leakage before being treated "
        "as final performance results."
    )

elif page == "Program Relevance":
    st.header("Relevance to Civil Engineering")
    st.write(
        "The project explores how machine learning can support "
        "early construction cost analysis, project planning, and "
        "decision support. Predictions should be reviewed by qualified "
        "professionals and should not replace detailed cost estimation."
    )

st.divider()
st.caption("CPE126L Artificial Intelligence Laboratory | Academic Prototype")
