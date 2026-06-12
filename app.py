import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Bank Deposit Prediction",
    page_icon="🏦",
    layout="wide"
)

# =====================================================
# LOAD MODEL FILES
# =====================================================

model = load_model("model_deep.keras")

scaler = joblib.load("scaler_deep.pkl")

feature_columns = joblib.load("feature_columns_deep.pkl")

# =====================================================
# HEADER
# =====================================================

st.title("🏦 Bank Deposit Prediction System")

st.markdown("""
Predict whether a customer will subscribe
to a term deposit using a Deep Neural Network.
""")

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =====================================================
# PROCESS FILE
# =====================================================

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(data.head())

    if st.button("Predict"):

        prediction_data = data.copy()

        # -------------------------------------------
        # Remove target if exists
        # -------------------------------------------

        if "deposit" in prediction_data.columns:
            prediction_data.drop(
                columns=["deposit"],
                inplace=True
            )

        # -------------------------------------------
        # Convert yes/no columns
        # -------------------------------------------

        for col in ["default", "housing", "loan"]:
            if col in prediction_data.columns:
                prediction_data[col] = prediction_data[col].map(
                    {
                        "yes": 1,
                        "no": 0
                    }
                )

        # -------------------------------------------
        # One Hot Encoding
        # -------------------------------------------

        prediction_data = pd.get_dummies(
            prediction_data,
            drop_first=True
        )

        # -------------------------------------------
        # Match Training Columns
        # -------------------------------------------

        prediction_data = prediction_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # -------------------------------------------
        # Scaling
        # -------------------------------------------

        scaled_data = scaler.transform(
            prediction_data
        )

        # -------------------------------------------
        # Prediction
        # -------------------------------------------

        probabilities = model.predict(
            scaled_data
        )

        predictions = (
            probabilities > 0.5
        ).astype(int)

        # -------------------------------------------
        # Output
        # -------------------------------------------

        results = data.copy()

        results["Prediction"] = predictions

        results["Probability"] = probabilities

        results["Prediction"] = results[
            "Prediction"
        ].map(
            {
                0: "No Deposit",
                1: "Deposit"
            }
        )

        st.success("Prediction Completed Successfully")

        st.subheader("Prediction Results")

        st.dataframe(results)

        # -------------------------------------------
        # Download Button
        # -------------------------------------------

        csv = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="bank_predictions.csv",
            mime="text/csv"
        )