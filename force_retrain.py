"""
Simple script to force model retraining.
Run with: python force_retrain.py
"""
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
app_path = Path(__file__).resolve().parent
sys.path.append(str(app_path))

# Import model initialization functions
from app.ml import initialize_model_from_data

def main():
    """Force model retraining"""
    print("Starting model retraining...")
    
    # Create models directory if it doesn't exist
    os.makedirs("data/models", exist_ok=True)
    
    # Force retrain
    model = initialize_model_from_data(
        json_path="appraisals_dataset.json",
        model_path="data/models/recommendation_model.pkl",
        force_retrain=True
    )
    
    print(f"Model retrained successfully with {len(model.test_property_ids)} test property IDs")
    print("Restart the application to use the new model")

if __name__ == "__main__":
    main() 