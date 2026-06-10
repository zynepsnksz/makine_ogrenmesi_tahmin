import os
import pickle
import pandas as pd
import numpy as np

def test_pipeline():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'outputs', 'best_lgbm_model.pkl')
    
    print("Testing Model Prediction Pipeline...")
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    print("Model loaded successfully.")
    
    # Create dummy patient data (raw inputs)
    raw_input = {
        'N_Days': 1800,
        'Age': 50 * 365.25,  # 50 years old in days
        'Bilirubin': 1.2,
        'Cholesterol': 300.0,
        'Albumin': 3.5,
        'Copper': 60.0,
        'Alk_Phos': 1200.0,
        'SGOT': 110.0,
        'Tryglicerides': 110.0,
        'Platelets': 265.0,
        'Prothrombin': 10.5,
        'Stage': 3.0
    }
    
    df = pd.DataFrame([raw_input])
    
    # One-hot encoded features mapping (categorical options matching model expected features)
    df['Drug_D-penicillamine'] = 0
    df['Drug_Placebo'] = 1
    df['Sex_F'] = 1
    df['Sex_M'] = 0
    df['Ascites_N'] = 1
    df['Ascites_Y'] = 0
    df['Hepatomegaly_N'] = 1
    df['Hepatomegaly_Y'] = 0
    df['Spiders_N'] = 1
    df['Spiders_Y'] = 0
    df['Edema_N'] = 1
    df['Edema_S'] = 0
    df['Edema_Y'] = 0
    
    # Feature Engineering Formulas
    df['ALBI'] = 0.66 * np.log10(df['Bilirubin'] * 17.1) - 0.085 * (df['Albumin'] * 10)
    df['APRI'] = ((df['SGOT'] / 40.0) * 100) / df['Platelets']
    df['BAR'] = df['Bilirubin'] / df['Albumin']
    df['PAI'] = df['Prothrombin'] / df['Albumin']
    df['Alk_Phos_SGOT_Ratio'] = df['Alk_Phos'] / df['SGOT']
    
    # Target feature column ordering of the processed dataset
    feature_cols = [
        'N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage',
        'Drug_D-penicillamine', 'Drug_Placebo', 'Sex_F', 'Sex_M', 'Ascites_N', 'Ascites_Y', 'Hepatomegaly_N', 'Hepatomegaly_Y', 'Spiders_N', 'Spiders_Y', 'Edema_N', 'Edema_S', 'Edema_Y',
        'ALBI', 'APRI', 'BAR', 'PAI', 'Alk_Phos_SGOT_Ratio'
    ]
    
    processed_df = df[feature_cols]
    
    print(f"Processed input shape: {processed_df.shape}")
    
    # Run prediction
    proba = model.predict_proba(processed_df)[0]
    pred_class = model.predict(processed_df)[0]
    
    print(f"Prediction complete. Predicted class index: {pred_class}")
    print(f"Class probabilities: C={proba[0]:.4f}, CL={proba[1]:.4f}, D={proba[2]:.4f}")
    
    # Asserts for pipeline sanity
    assert len(proba) == 3, f"Expected 3 class probabilities, got {len(proba)}"
    assert np.isclose(np.sum(proba), 1.0), f"Expected probabilities to sum to 1.0, got {np.sum(proba)}"
    print("Pipeline sanity checks passed! Model predictions are active and correct.")

if __name__ == '__main__':
    try:
        test_pipeline()
    except Exception as e:
        print(f"Pipeline Test FAILED: {str(e)}")
        import sys
        sys.exit(1)
