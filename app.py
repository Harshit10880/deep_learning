import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Bank Deposit Prediction",
    page_icon="🏦",
    layout="wide"
)

# ======================================
# LOAD FILES
# ======================================

model = load_model("model_deep.keras")
scaler = joblib.load("scaler_deep.pkl")
feature_columns = joblib.load("feature_columns_deep.pkl")

# ======================================
# HEADER
# ======================================

st.title("🏦 Bank Deposit Prediction Using Deep Learning")

st.write(
    """
    Upload a customer dataset and predict whether
    the customer will subscribe to a term deposit.
    """
)

# ======================================
# MODEL INFO
# ======================================

st.info(
    """
    Deep Neural Network (DNN)

    Accuracy: Enter your actual accuracy here
    Example: 88.45%
    """
)

# ======================================
# FILE UPLOAD
# ======================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ======================================
# PREDICTION
# ======================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    if st.button("Predict"):

        input_data = df.copy()

        # Remove target if present
        if "deposit" in input_data.columns:
            input_data = input_data.drop(
                columns=["deposit"]
            )

        # Convert yes/no columns
        for col in ["default", "housing", "loan"]:
            if col in input_data.columns:
                input_data[col] = input_data[col].map({
                    "yes": 1,
                    "no": 0
                })

        # One Hot Encoding
        input_data = pd.get_dummies(
            input_data,
            drop_first=True
        )

        # Match Training Columns
        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Scaling
        input_scaled = scaler.transform(
            input_data
        )

        # Prediction
        probabilities = model.predict(
            input_scaled
        )

        predictions = (
            probabilities > 0.5
        ).astype(int)

        # Results
        results = df.copy()

        results["Prediction"] = [
            "Deposit"
            if p == 1
            else "No Deposit"
            for p in predictions.flatten()
        ]

        results["Probability (%)"] = (
            probabilities.flatten() * 100
        ).round(2)

        st.success("Prediction Completed")

        # Summary
        deposit_count = (
            results["Prediction"]
            == "Deposit"
        ).sum()

        no_deposit_count = (
            results["Prediction"]
            == "No Deposit"
        ).sum()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Deposit Predictions",
                deposit_count
            )

        with col2:
            st.metric(
                "No Deposit Predictions",
                no_deposit_count
            )

        st.subheader("Prediction Results")

        st.dataframe(results)

        # Download
        csv = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )