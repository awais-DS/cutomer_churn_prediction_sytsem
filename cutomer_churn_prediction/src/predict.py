import joblib
import pandas as pd
import numpy as np
from src.data_preprocesssing import clean_raw_data

def run_production_inference(raw_user_input, pipeline_path='models/data_cleaning_pipeline.pkl', model_path='models/winning_churn_model.pkl'):
    """
    Loads saved artifacts, cleans raw user inputs, transforms features,
    and runs predictions complete with diagnostic recommendations.
    """
    # 1. Load the pre-fitted preprocessing pipeline and your model artifact
    try:
        transformer = joblib.load(pipeline_path)
        ml_model = joblib.load(model_path)
    except FileNotFoundError as e:
        return {"status": "error", "message": f"Artifact missing. Details: {e}"}

    # 2. Preprocess and align data features
    cleaned_df = clean_raw_data(raw_user_input, is_training=False)
    
    # Force exact column sequencing to match what your model expects
    expected_features = ['gender', 'tenure', 'Contract', 'Dependents', 'MonthlyCharges']
    cleaned_df = cleaned_df[expected_features]

    # 3. Apply Feature Engineering Pipeline Encoders
    processed_matrix = transformer.transform(cleaned_df)

    # 4. Generate Predictions and Probabilities
    prediction = ml_model.predict(processed_matrix)[0]
    probability = ml_model.predict_proba(processed_matrix)[0][1]

    # 5. Build Explanation Models and Action Plans
    reasons = []
    recommendations = []
    
    # Rule 1: High Monthly Charges Risk Check
    if float(raw_user_input.get('MonthlyCharges', 0)) > 70.0 and probability > 0.45:
        reasons.append(f"High bill pressure: Monthly charges are quite high (${raw_user_input.get('MonthlyCharges')}).")
        recommendations.append("Offer a customized value bundle or discount to ease financial churn risk.")
        
    # Rule 2: Unstable Contract Risk Check
    if str(raw_user_input.get('Contract')).lower().replace('-', ' ') == 'month to month' and probability > 0.45:
        reasons.append("High-risk account state: Customer is on a rolling month-to-month plan.")
        recommendations.append("Proactively offer a long-term upgrade incentive to lock in a 1-Year or 2-Year contract.")

    # Rule 3: Early Stage Risk Check
    if int(raw_user_input.get('tenure', 0)) <= 6 and probability > 0.45:
        reasons.append(f"High vulnerability: Early account tenure stage (Only {raw_user_input.get('tenure')} months active).")
        recommendations.append("Schedule a customer success onboarding call to clear up early service roadblocks.")

    # Fallback if the user profile looks completely secure
    if not reasons:
        reasons.append("Customer shows healthy operational tenure and stable account parameters.")
        recommendations.append("No immediate intervention needed. Maintain baseline customer relation touchpoints.")

    return {
        "status": "success",
        "churn_risk": "Yes" if prediction == 1 else "No",
        "churn_probability": f"{probability:.2%}",
        "risk_factors": reasons,
        "action_plan": recommendations
    }
