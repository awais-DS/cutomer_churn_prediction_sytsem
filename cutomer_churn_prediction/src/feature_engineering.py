import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

def build_feature_pipeline():
    """
    Assembles a standalone scikit-learn transformation pipeline.
    It isolates nominal and ordinal features without housing an ML model.
    """
    # Define the precise, lowercased timeline hierarchy for your contracts
    contract_hierarchy = ['month to month', 'one year', 'two year']

    # Configure parallel encoders for specific features
    preprocessor = ColumnTransformer(
        transformers=[
            # Nominal Categories -> OneHot mapping (Returns a single 0 or 1 column)
            ('nominal_gender', OneHotEncoder(drop='first', sparse_output=False), ['gender']),
            ('nominal_dependents', OneHotEncoder(drop='first', sparse_output=False), ['Dependents']),
            
            # Ordinal Category -> Maps strictly to 0, 1, or 2
            ('ordinal_contract', OrdinalEncoder(categories=[contract_hierarchy]), ['Contract'])
        ],
        # CRITICAL: Keeps both 'tenure' and 'MonthlyCharges' completely safe and untouched
        remainder='passthrough' 
    )

    # Wrap it inside a strict Pipeline container
    data_cleaning_pipeline = Pipeline(steps=[
        ('encoder_layer', preprocessor)
    ])
    
    return data_cleaning_pipeline

def save_feature_pipeline(pipeline, X_train, file_path='models/data_cleaning_pipeline.pkl'):
    """Fits the transformation pipeline strictly on X_train and exports it."""
    pipeline.fit(X_train)
    joblib.dump(pipeline, file_path)
    print(f"Feature pipeline successfully exported to: {file_path}")
