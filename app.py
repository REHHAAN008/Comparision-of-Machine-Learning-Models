# app.py

import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import joblib
import json

app = Flask(__name__)

# --- Load all necessary files ---
try:
    # Load both models
    lr_model = joblib.load('lr_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    
    # Load model columns and evaluation results
    model_columns = joblib.load('model_columns.pkl')
    with open('evaluation_results.json', 'r') as f:
        results = json.load(f)
    best_model_name = results['best_model']

except FileNotFoundError:
    print("❌ Model files not found! Please run the training script first.")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        query_df = pd.DataFrame(columns=model_columns)
        query_df.loc[0, :] = 0
        
        # --- Create feature vector (same as before) ---
        query_df.loc[0, 'hour'] = int(form_data['hour'])
        for key, value in form_data.items():
            col_name = f"{key}_{value}"
            if col_name in model_columns:
                query_df.loc[0, col_name] = 1
        
        # --- Make predictions with BOTH models ---
        lr_prediction = int(lr_model.predict(query_df)[0])
        rf_prediction = int(rf_model.predict(query_df)[0])
        
        # --- Send all results to the HTML page ---
        return render_template('index.html', 
                               lr_pred=f'Predicted Age: {lr_prediction}',
                               rf_pred=f'Predicted Age: {rf_prediction}',
                               best_model_info=f'Recommended Model: {best_model_name}')
