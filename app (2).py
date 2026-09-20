
import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the patient's medical information below "
    "to predict the likelihood of diabetes."
)

st.divider()


# -----------------------------
# Patient Information
# -----------------------------

st.subheader("Patient Information")


col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=100
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )


st.divider()


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Predict Diabetes", use_container_width=True):

    # Create patient data
    patient_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    # Scale data
    patient_scaled = scaler.transform(patient_data)

    # Prediction
    prediction = model.predict(patient_scaled)

    # Probability
    probability = model.predict_proba(patient_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error("⚠️ The model predicts: Diabetes")

    else:

        st.success("✅ The model predicts: No Diabetes")

    st.write(
        f"Estimated diabetes probability: **{probability * 100:.2f}%**"
    )

    st.progress(float(probability))


st.divider()


# -----------------------------
# About the Project
# -----------------------------

st.subheader("About This Project")

st.write(
    """
    This project uses Machine Learning to predict the likelihood
    of diabetes based on patient information.

    The model was trained using the Pima Indians Diabetes dataset
    and uses Logistic Regression for binary classification.

    **Important:** This application is for educational purposes
    and should not be used as a medical diagnosis.
    """
)
