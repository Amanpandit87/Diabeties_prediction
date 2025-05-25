import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load model
model = pickle.load(open("diabetes.pkl", "rb"))

# Title
st.title("Diabetes Prediction App")

# Sidebar for user input
st.sidebar.header("Patient Information")

# Input fields
pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, step=1)
glucose = st.sidebar.slider("Glucose Level", 70, 180, 100)
blood_pressure = st.sidebar.slider("Blood Pressure", 30, 110, 70)
skin_thickness = st.sidebar.slider("Skin Thickness", 5, 55, 20)
insulin = st.sidebar.slider("Insulin", 0, 500, 100)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
age = st.sidebar.number_input("Age", 10, 100, 25)

# Collecting input into a DataFrame
input_data = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
})

# Scaling input
scaler = StandardScaler()
# You must use the same scaler used during training, or fit a new one like this:
# (Just for this example, fitting it on the fly)
df = pd.read_csv("diabetes.csv")
df = df[(df["Glucose"] > 70) & (df["Glucose"] < 180)]
df = df[(df["BloodPressure"] > 30) & (df["BloodPressure"] < 110)]
df = df[(df["SkinThickness"] > 5) & (df["SkinThickness"] < 55)]
df = df[(df["Insulin"] < 500)]
df = df[(df["BMI"] > 10) & (df["BMI"] < 50)]
x = df.drop(columns=["Outcome"])
scaler.fit(x)
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("The person is likely to have Diabetes.")
    else:
        st.success("The person is not likely to have Diabetes.")
