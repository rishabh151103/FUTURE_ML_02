import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, feature names
model = joblib.load('Saved_Model/churn_model.pkl')
scaler = joblib.load('Saved_Model/scaler.pkl')
feature_names = joblib.load('Saved_Model/feature_names.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #3B82F6;'>📉 Customer Churn Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: gray;'>Enter customer details to predict likelihood of churn</h4>",
    unsafe_allow_html=True
)

st.markdown("---")

# Layout split into two columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ['Male', 'Female'])
    SeniorCitizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
    Partner = st.selectbox("Partner", ['No', 'Yes'])
    Dependents = st.selectbox("Dependents", ['No', 'Yes'])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
    PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
    MonthlyCharges = st.number_input("Monthly Charges", 10.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges", 10.0, 10000.0, 2500.0)

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    Contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])

with col4:
    InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    PaymentMethod = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])

# Encode inputs
input_dict = {
    'gender': 1 if gender == 'Female' else 0,
    'SeniorCitizen': 1 if SeniorCitizen == 'Yes' else 0,
    'Partner': 1 if Partner == 'Yes' else 0,
    'Dependents': 1 if Dependents == 'Yes' else 0,
    'tenure': tenure,
    'PhoneService': 1 if PhoneService == 'Yes' else 0,
    'PaperlessBilling': 1 if PaperlessBilling == 'Yes' else 0,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,
    'Contract_One year': 1 if Contract == 'One year' else 0,
    'Contract_Two year': 1 if Contract == 'Two year' else 0,
    'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
    'InternetService_No': 1 if InternetService == 'No' else 0,
    'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == 'Credit card (automatic)' else 0,
    'PaymentMethod_Electronic check': 1 if PaymentMethod == 'Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if PaymentMethod == 'Mailed check' else 0
}

input_df = pd.DataFrame([input_dict])

# Scale numeric features
input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
)

# Add missing features and reorder
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_names]

# Predict
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# Result Display
st.markdown("---")
st.subheader("🔍 Prediction Result")

if prediction == 1:
    st.error(f"⚠️ The customer is likely to churn.\n\n**Churn Probability: {probability*100:.2f}%**")
else:
    st.success(f"✅ The customer is not likely to churn.\n\n**Churn Probability: {probability*100:.2f}%**")

st.markdown("### 📊 Confidence Level")
st.progress(min(int(probability * 100), 100))

# Footer
st.markdown("---")
st.caption("Built using Logistic Regression | Telco Churn Dataset | Streamlit UI 💻")


