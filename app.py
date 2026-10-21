
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Construction Cost Predictor",
    page_icon="🏗️",
    layout="wide"
)

# -----------------------------
# Load trained model and features
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    with open("feature_columns.json", "r") as f:
        features = json.load(f)
    return model, features


# -----------------------------
# Load sample input record
# -----------------------------
def load_sample_input():
    sample_path = Path("sample_input.json")

    if not sample_path.exists():
        raise FileNotFoundError(
            "sample_input.json was not found in the repository."
        )

    with open(sample_path, "r") as f:
        sample = json.load(f)

    if not isinstance(sample, dict):
        raise ValueError("The sample JSON must contain a dictionary.")

    return sample


# -----------------------------
# Navigation
# -----------------------------
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
except Exception as e:
    model_ready = False
    st.error(f"Could not load the model files: {e}")


# -----------------------------
# Home
# -----------------------------
if page == "Home":
    st.header("Project Overview")

    st.write(
        "This project explores machine learning for predicting "
        "residential construction costs using building characteristics "
        "and economic indicators."
    )

    st.info(
        "Use the Cost Prediction page to load a sample record "
        "or enter model input values."
    )

    st.subheader("Project Features")
    st.markdown(
        "- Construction cost prediction\n"
        "- Sample-record testing\n"
        "- Model explanation and feature importance\n"
        "- Dataset and model information\n"
        "- Construction and civil engineering relevance"
    )


# -----------------------------
# Cost Prediction
# -----------------------------
elif page == "Cost Prediction":
    st.header("Construction Cost Prediction")

    if not model_ready:
        st.error("The model files could not be loaded.")
    else:
        st.warning(
            "Academic prototype only. Use the same units and variable "
            "definitions as the training dataset. This prediction is "
            "not a professional construction quotation."
        )

        # Initialize the input values
        for i, feature in enumerate(feature_columns):
            key = f"input_{i}"
            if key not in st.session_state:
                st.session_state[key] = 0.0

        # Load a real record from the test dataset
        if st.button("📂 Use Sample Record from Dataset"):
            try:
                sample = load_sample_input()

                missing_features = [
                    feature for feature in feature_columns
                    if feature not in sample
                ]

                if missing_features:
                    st.error(
                        "The sample file is missing "
                        f"{len(missing_features)} model inputs. "
                        "Check that sample_input.json matches "
                        "feature_columns.json."
                    )
                else:
                    for i, feature in enumerate(feature_columns):
                        value = float(sample[feature])

                        if not np.isfinite(value):
                            raise ValueError(
                                f"Invalid numeric value for {feature}."
                            )

                        st.session_state[f"input_{i}"] = value

                    st.session_state["sample_loaded"] = True
                    st.success(
                        "Sample record loaded! Review the values "
                        "below, then click Predict Construction Cost."
                    )
                    st.rerun()

            except Exception as e:
                st.error(f"Could not load the sample record: {e}")

        if st.session_state.get("sample_loaded", False):
            st.info(
                "Sample record loaded from sample_input.json. "
                "You can review or edit the values before prediction."
            )

        st.subheader("Model Input Values")

        with st.form("prediction_form"):
            values = {}
            columns = st.columns(2)

            for i, feature in enumerate(feature_columns):
                key = f"input_{i}"

                with columns[i % 2]:
                    values[feature] = st.number_input(
                        feature,
                        format="%.4f",
                        key=key
                    )

            submitted = st.form_submit_button(
                "🔍 Predict Construction Cost",
                type="primary"
            )

        if submitted:
            try:
                input_df = pd.DataFrame(
                    [[values[f] for f in feature_columns]],
                    columns=feature_columns
                )

                prediction = float(model.predict(input_df)[0])

                st.subheader("Predicted Construction Cost")
                st.metric("Model output", f"{prediction:,.4f}")

                st.caption(
                    "Prediction is reported in the target variable's "
                    "dataset units. Confirm the unit before interpreting "
                    "it as a monetary amount."
                )

                st.session_state["last_prediction"] = prediction

                st.success("Prediction completed successfully.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# -----------------------------
# Model Explanation
# -----------------------------
elif page == "Model Explanation":
    st.header("Model Explanation")

    st.write(
        "Feature importance indicates which input variables the "
        "Random Forest model relies on most. It does not establish "
        "causation."
    )

    if model_ready and hasattr(model, "feature_importances_"):
        if len(model.feature_importances_) != len(feature_columns):
            st.error(
                "The model's feature count does not match "
                "feature_columns.json."
            )
        else:
            importance = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": model.feature_importances_
            }).sort_values(
                "Importance", ascending=False
            ).head(15)

            st.dataframe(importance, use_container_width=True)

            chart_data = importance.sort_values("Importance")

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(
                chart_data["Feature"],
                chart_data["Importance"]
            )
            ax.set_xlabel("Feature importance")
            ax.set_title("Top 15 Model Features")
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Model files are needed to display feature importance.")


# -----------------------------
# Dataset and Model
# -----------------------------
elif page == "Dataset & Model":
    st.header("Dataset and Model Information")

    st.write("Dataset: Residential Building Data Set")
    st.write("Target variable: V-10 — Actual construction costs")
    st.write("Candidate models: Random Forest and XGBoost")
    st.write("Initial evaluation used an 80:20 train-test split.")

    st.warning(
        "Model performance and input variables require further "
        "validation for data leakage and feature availability."
    )


# -----------------------------
# Program Relevance
# -----------------------------
elif page == "Program Relevance":
    st.header("Relevance to Civil Engineering")

    st.write(
        "The project explores how machine learning can support "
        "early construction cost analysis, project planning, and "
        "decision support. Predictions should be reviewed by qualified "
        "professionals and should not replace detailed cost estimation."
    )


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "CPE126L Artificial Intelligence Laboratory | Academic Prototype"
)
