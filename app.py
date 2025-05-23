import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load artifacts once
@st.cache_resource
def load_artifacts():
    le_gender = pickle.load(open('artifacts/label_encoder_gender.pkl', 'rb'))
    ohe_geo = pickle.load(open('artifacts/onehot_encoder_geo.pkl', 'rb'))
    scaler = pickle.load(open('artifacts/scaler.pkl', 'rb'))
    model = load_model('artifacts/model.h5')
    return le_gender, ohe_geo, scaler, model

le_gender, ohe_geo, scaler, model = load_artifacts()

st.title("Churn Prediction App")

# Input widgets
gender = st.selectbox("Gender", options=['Female', 'Male'])
geography = st.selectbox("Geography", options=['France', 'Spain', 'Germany'])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=1000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=5, value=1)
has_cr_card = st.selectbox("Has Credit Card", options=[0,1])
is_active_member = st.selectbox("Is Active Member", options=[0,1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)

if st.button("Predict Churn"):
    # Encode gender
    gender_encoded = le_gender.transform([gender])[0]

    # Encode geography
    geo_df = pd.DataFrame({'Geography': [geography]})
    geo_encoded = ohe_geo.transform(geo_df)
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.categories_[0])

    # Build feature vector in correct order (make sure matches training)
    input_dict = {
        'CreditScore': credit_score,
        'Gender': gender_encoded,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary,
    }

    # Combine input data and geography onehot columns
    input_df = pd.DataFrame([input_dict])
    input_df = pd.concat([input_df, geo_encoded_df], axis=1)

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0][0]

    st.write(f"**Churn Probability:** {prediction:.4f}")
