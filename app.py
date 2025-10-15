import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pickle


# Load model, scaler, and column list
model = pickle.load(open('best_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
trained_columns = pickle.load(open('trained_columns.pkl', 'rb'))

# App title
st.title("❤️ Heart Disease Prediction App")
st.markdown("This app predicts the likelihood of heart disease based on patient health data using a trained machine learning model.")

# Sidebar input section
st.sidebar.header("Input Patient Data")

def user_input_features():
    age = st.sidebar.number_input("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
    chest_pain_type = st.sidebar.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    resting_blood_pressure = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholestoral = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fasting_blood_sugar = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    rest_ecg = st.sidebar.selectbox("Resting ECG (0–2)", [0, 1, 2])
    Max_heart_rate = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exercise_induced_angina = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
    vessels_colored_by_flourosopy = st.sidebar.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
    thalassemia = st.sidebar.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

    # Convert categorical inputs to numerical if necessary
    sex = 1 if sex == "Male" else 0

    # Create dataframe
    data = {
        'age': age,
        'sex': sex,
        'chest_pain_type': chest_pain_type,
        'resting_blood_pressure': resting_blood_pressure,
        'cholestoral': cholestoral,
        'fasting_blood_sugar': fasting_blood_sugar,
        'rest_ecg': rest_ecg,
        'Max_heart_rate': Max_heart_rate,
        'exercise_induced_angina': exercise_induced_angina,
        'oldpeak': oldpeak,
        'slope': slope,
        'vessels_colored_by_flourosopy': vessels_colored_by_flourosopy,
        'thalassemia': thalassemia
    }

    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Button to trigger prediction
if st.button("🔍 Predict Heart Disease"):

    # Encode input
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=trained_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    # Display patient data
    st.subheader("Patient Data Preview")
    st.write(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Show results
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ The model predicts a HIGH likelihood of heart disease.")
    else:
        st.success("✅ The model predicts a LOW likelihood of heart disease.")

    # Probabilities
    st.subheader("Prediction Probability")
    st.write(f"Probability of No Heart Disease: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")

st.markdown("---")
st.caption("Built with Streamlit | Model: Heart Disease Classifier")

