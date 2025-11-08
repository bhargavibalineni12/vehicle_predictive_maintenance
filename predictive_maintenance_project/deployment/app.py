import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download, login, HfApi
import os

# ---------------------------
# Load Predictive Maintenance Model
# ---------------------------
st.title("Predictive Maintenance App")
st.write("""
This application predicts whether an engine requires **maintenance** or is **operating normally** based on real-time sensor readings.
""")

# Download the model from Hugging Face Hub
model_path = hf_hub_download(repo_id="Bhargavi329/predictive_maintenance_model", filename="best_predictive_maintenance_model_v1.joblib")
model = joblib.load(model_path)

# ---------------------------
# Collect User Inputs
# ---------------------------
st.subheader("Enter Engine Sensor Readings")

engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=5000, value=800)
lub_oil_pressure = st.number_input("Lubricating Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=3.0)
fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=25.0, value=8.0)
coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=2.5)
lub_oil_temp = st.number_input("Lubricating Oil Temperature (°C)", min_value=0.0, max_value=200.0, value=80.0)
coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=0.0, max_value=200.0, value=85.0)

# Create DataFrame for model input
input_data = pd.DataFrame([{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Engine Condition"):
    prediction = model.predict(input_data)[0]
    result = "Maintenance Required" if prediction == 1 else "Engine Operating Normally"
    
    st.subheader("Prediction Result:")
    st.success(result)
    st.write("---")

# ---------------------------
# Optional: Upload Deployment to Hugging Face Space
# ---------------------------
st.subheader("Upload Deployment to Hugging Face Space")

if st.button("Upload Deployment Files"):
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        st.error(" HF_TOKEN is not set. Cannot upload files.")
    else:
        login(token=HF_TOKEN)
        api = HfApi()
        repo_id = "Bhargavi329/vehicle-predictive-maintenance"  # Replace with your actual Space repo name
        folder_path = "predictive_maintenance_project/deployment"

        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="space",
            path_in_repo=""
        )
        st.success(" Deployment files uploaded successfully to Hugging Face Space!")
