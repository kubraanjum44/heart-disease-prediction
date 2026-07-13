"""
Heart Disease Prediction App
-----------------------------
This file creates a simple webpage where someone can enter patient details
and get a prediction from your trained model.

HOW TO RUN THIS (do this in your terminal, one line at a time):
    1. pip install streamlit
    2. Make sure heart_disease_model.pkl is in the SAME folder as this file
    3. streamlit run app.py

That's it. It will open a webpage automatically in your browser.
"""

import streamlit as st
import pickle
import numpy as np

# -----------------------------------------------------------------------
# STEP 1: Load your saved model (the .pkl file you created earlier)
# This runs once when the app starts, not every time someone clicks predict.
# -----------------------------------------------------------------------
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------------------------------------------------
# STEP 2: Give the page a title
# -----------------------------------------------------------------------
st.title("Heart Disease Prediction")
st.write("Enter patient details below to predict the likelihood of heart disease.")

# -----------------------------------------------------------------------
# STEP 3: Create input boxes for every feature your model needs.
# Each st.number_input / st.selectbox creates one box on the webpage.
# The order and names here MUST match the columns your model was trained on:
# age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
# slope, ca, thal
# -----------------------------------------------------------------------

age = st.number_input("Age", min_value=1, max_value=120, value=50)

sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]

cp = st.selectbox(
    "Chest Pain Type",
    options=[
        ("Typical angina", 0),
        ("Atypical angina", 1),
        ("Non-anginal pain", 2),
        ("Asymptomatic", 3),
    ],
    format_func=lambda x: x[0],
)[1]

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)

chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

restecg = st.selectbox(
    "Resting ECG Result",
    options=[
        ("Normal", 0),
        ("ST-T wave abnormality", 1),
        ("Left ventricular hypertrophy", 2),
    ],
    format_func=lambda x: x[0],
)[1]

thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)

exang = st.selectbox("Exercise Induced Angina?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
    format_func=lambda x: x[0],
)[1]

ca = st.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])

thal = st.selectbox(
    "Thalium Stress Result",
    options=[("Normal", 1), ("Fixed defect", 2), ("Reversible defect", 3)],
    format_func=lambda x: x[0],
)[1]

# -----------------------------------------------------------------------
# STEP 4: The Predict button.
# When clicked, we collect all the inputs above into one row of data,
# feed it to the model, and show the result.
# -----------------------------------------------------------------------
if st.button("Predict"):
    # Put all inputs into one array, in the exact column order the model expects
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # probability of class 1 (disease)

    st.subheader("Result")
    if prediction == 1:
        st.error(f"The model predicts **heart disease present** "
                  f"(confidence: {probability*100:.1f}%)")
    else:
        st.success(f"The model predicts **no heart disease** "
                   f"(confidence: {(1-probability)*100:.1f}%)")

    st.caption("This is a demo model trained on a small dataset (303 patients) "
               "and is not a substitute for medical advice.")
