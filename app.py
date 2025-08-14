# app.py (for Streamlit)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --- Caching the models and data to improve performance ---
@st.cache_resource
def load_models_and_data():
    """Loads all models, columns, and evaluation results."""
    try:
        lr_model = joblib.load('lr_model.pkl')
        rf_model = joblib.load('rf_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        with open('evaluation_results.json', 'r') as f:
            results = json.load(f)
        return lr_model, rf_model, model_columns, results
    except FileNotFoundError:
        st.error("Model files not found! Please make sure all .pkl and .json files are in the repository.")
        return None, None, None, None

# --- Load all assets ---
lr_model, rf_model, model_columns, results = load_models_and_data()

# --- Page Configuration and Header ---
st.set_page_config(page_title="Model Comparison", layout="wide")

# Header with personal details
st.container(border=True).markdown("""
**Name:** Rehan Zafar  
**Enrollment:** 02-134221-008
""")

# Main title
st.title("Linear Regression vs Random Forest")
st.write("Enter crime details to get predictions from two different ML models and see which one is recommended.")

# --- Input Form using Streamlit columns for a cleaner layout ---
if lr_model is not None:
    col1, col2 = st.columns(2)

    with col1:
        crime_type = st.selectbox("Select Crime Type", ['Homicide', 'Assault', 'Burglary', 'Vandalism', 'Fraud', 'Drug Offense', 'Domestic Violence', 'Arson', 'Theft', 'Robbery'])
        victim_gender = st.selectbox("Select Victim Gender", ['Male', 'Female', 'Other', 'Non-binary'])
        hour = st.number_input("Hour of Crime (0-23)", min_value=0, max_value=23, step=1, value=14)

    with col2:
        city = st.selectbox("Select City", ['Philadelphia', 'Phoenix', 'San Antonio', 'Chicago', 'Houston', 'San Diego', 'New York', 'Dallas', 'Los Angeles', 'San Jose'])
        victim_race = st.selectbox("Select Victim Race", ['Other', 'Black', 'Asian', 'White', 'Hispanic'])

    # --- Prediction Logic ---
    if st.button("Compare Predictions", use_container_width=True):
        # Create a dataframe with the same columns as the training data
        query_df = pd.DataFrame(columns=model_columns)
        query_df.loc[0, :] = 0

        # Update the dataframe with user's input
        query_df.loc[0, 'hour'] = hour
        
        # Handle categorical inputs safely
        for feature, value in {'crime_type': crime_type, 'city': city, 'victim_gender': victim_gender, 'victim_race': victim_race}.items():
            col_name = f"{feature}_{value}"
            if col_name in model_columns:
                query_df.loc[0, col_name] = 1

        # Make predictions
        lr_prediction = int(lr_model.predict(query_df)[0])
        rf_prediction = int(rf_model.predict(query_df)[0])
        best_model_name = results['best_model']

        # --- Display Results ---
        st.divider()
        st.subheader("Prediction Results")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Model 1: Linear Regression", value=f"{lr_prediction} years")
        
        with res_col2:
            st.metric(label="Model 2: Random Forest", value=f"{rf_prediction} years")
        
        st.success(f"🏆 Recommended Model (based on training performance): **{best_model_name}**")
