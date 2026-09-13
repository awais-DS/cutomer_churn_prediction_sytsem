import pandas as pd

def clean_raw_data(data, is_training=False):
    """
    Cleans raw incoming data. Handles row drops for training
    and normalizes categorical columns.
    """
    # 1. Convert input data to a DataFrame if passed as a dictionary
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()
        
    # 2. Drop rows with null values during training (e.g., your 11 nulls)
    if is_training:
        df = df.dropna(subset=['gender', 'tenure', 'Contract', 'Dependents', 'MonthlyCharges'])
        
    # 3. Clean string anomalies for categorical variables to guarantee matching
    categorical_cols = ['gender', 'Dependents', 'Contract']
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.lower().str.replace('-', ' ').str.strip()
        
    return df
