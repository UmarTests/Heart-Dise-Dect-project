import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load(r"C:\Users\mohdq\OneDrive\Desktop\internship projects\Heart_dise_dict\Heart Disease\best_knn_heart_disease_model.joblib")
scaler = joblib.load(r"C:\Users\mohdq\OneDrive\Desktop\internship projects\Heart_dise_dict\Heart Disease\scaler.joblib")


## UI Title
st.title("🩺 Heart Disease Prediction App")
st.markdown("### Enter patient details to predict the likelihood of heart disease.")

# User Input Sections
with st.expander("📌 Personal Information"):
    age = st.slider("Age", 20, 100, 50)
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)

with st.expander("❤️ Heart & Blood Vitals"):
    chest_pain_type = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp_s = st.slider("Resting Blood Pressure (mmHg)", 10, 200, 80)
    cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
    max_heart_rate = st.slider("Max Heart Rate Achieved", 30, 220, 140)

with st.expander("🩸 Other Medical Details"):
    fasting_blood_sugar = st.radio("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"], horizontal=True)
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"])
    exercise_angina = st.radio("Exercise-Induced Angina", ["Yes", "No"], horizontal=True)
    oldpeak = st.slider("ST Depression (Oldpeak)", -2.0, 6.0, 1.0, step=0.1)
    ST_slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])

# Convert categorical inputs to numerical values
sex = 1 if sex == "Male" else 0
chest_pain_type = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_type) + 1
fasting_blood_sugar = 1 if fasting_blood_sugar == "Yes" else 0
resting_ecg = ["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
exercise_angina = 1 if exercise_angina == "Yes" else 0
ST_slope = ["Upsloping", "Flat", "Downsloping"].index(ST_slope) + 1

# Prepare input data with correct feature names
input_data = pd.DataFrame([[age, sex, chest_pain_type, resting_bp_s, cholesterol, fasting_blood_sugar,
                            resting_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope]],
                          columns=["age", "sex", "chest pain type", "resting bp s", "cholesterol", "fasting blood sugar",
                                   "resting ecg", "max heart rate", "exercise angina", "oldpeak", "ST slope"])

# Scale input data
input_scaled = scaler.transform(input_data)

# Predict
if st.button("🔍 Predict Heart Disease"):
    prediction = model.predict(input_scaled)
    result = "🛑 **Heart Disease Detected!**" if prediction[0] == 1 else "✅ **No Heart Disease**"
    st.markdown(f"## {result}")
