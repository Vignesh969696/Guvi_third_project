import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Insurance Premium Predictor")
st.write("Enter customer details to predict the premium amount.")

# --- User Input ---
def user_input():
    data = {
        "Age": st.number_input("Age", 18, 100, 35),
        "Gender": st.selectbox("Gender", ["Male", "Female"]),
        "Annual Income": st.number_input("Annual Income", 0, 1_000_000, 50000),
        "Marital Status": st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
        "Number of Dependents": st.number_input("Dependents", 0, 10, 1),
        "Education Level": st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"]),
        "Occupation": st.selectbox("Occupation", ["Salaried", "Self-Employed", "Retired"]),
        "Health Score": st.slider("Health Score", 0, 100, 50),
        "Location": st.selectbox("Location", ["Urban", "Suburban", "Rural"]),
        "Previous Claims": st.number_input("Previous Claims", 0, 10, 0),
        "Vehicle Age": st.number_input("Vehicle Age", 0, 20, 5),
        "Credit Score": st.number_input("Credit Score", 300, 900, 650),
        "Insurance Duration": st.number_input("Insurance Duration", 0, 10, 3),
        "Policy Start Date": st.date_input("Policy Start Date"),
        "Customer Feedback": st.selectbox("Feedback", ["Poor", "Average", "Good"]),
        "Smoking Status": st.selectbox("Smoking Status", ["Yes", "No"]),
        "Exercise Frequency": st.selectbox("Exercise Frequency", ["Never", "Rarely", "Monthly", "Weekly", "Daily"]),
        "Property Type": st.selectbox("Property Type", ["House", "Apartment", "Condo"]),
        "Policy Type": st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    }
    return pd.DataFrame([data])

# --- Preprocessing ---
def preprocess(df):
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
    
    # Calculate missing columns expected by the model
    df['Policy Age (Days)'] = (pd.Timestamp.today() - df['Policy Start Date']).dt.days
    df['Policy Start Year'] = df['Policy Start Date'].dt.year
    df['Policy Start Month'] = df['Policy Start Date'].dt.month
    df['Policy Start Day'] = df['Policy Start Date'].dt.day

    # Interaction features and risk flags (optional, only if model was trained with them)
    df['Age_x_Health'] = df['Age'] * df['Health Score']
    df['Credit_x_Claims'] = df['Credit Score'] * df['Previous Claims']
    df['Income_x_Credit'] = df['Annual Income'] * df['Credit Score']
    df['Is_Smoker'] = df['Smoking Status'].str.lower().map({'yes':1,'no':0})
    df['Low_Credit_Score'] = (df['Credit Score'] < 600).astype(int)
    df['Multiple_Claims'] = (df['Previous Claims'] > 2).astype(int)
    df['Customer_Feedback_Score'] = df['Customer Feedback'].map({'Poor':0,'Average':1,'Good':2})
    df['Exercise_Freq_Score'] = df['Exercise Frequency'].map({'Never':0,'Rarely':1,'Monthly':2,'Weekly':3,'Daily':4})

    # Convert categorical columns to string (matching model)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    return df

# --- Prediction ---
input_df = user_input()
if st.button("Estimate Premium"):
    processed = preprocess(input_df)
    pred_log = model.predict(processed)
    pred = np.expm1(pred_log)  # Reverse log-transform
    st.success(f"Estimated Premium: ₹{pred[0]:,.2f}")

