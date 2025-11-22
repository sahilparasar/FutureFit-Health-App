import pandas as pd
import numpy as np
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))

try:
    from models.preprocessor import HealthDataPreprocessor
    from models.trainer import HealthRiskPredictor
    from utils.helpers import generate_sample_data
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def main():
    """Train and save the health risk prediction model"""
    print("🚀 Starting model training...")
    
    try:
        print("Generating sample health data...")
        df = generate_sample_data()
        print(f"✅ Generated {len(df)} sample records")
        
        print("Preprocessing data...")
        preprocessor = HealthDataPreprocessor()
        preprocessor.fit(df)
        print("✅ Data preprocessing completed")
        
        print("Training model...")
        model = HealthRiskPredictor()
        model.train(df, preprocessor)
        print("✅ Model training completed")
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save model and preprocessor
        model.save('data/trained_model.pkl')
        preprocessor.save('data/trained_preprocessor.pkl')
        
        print("🎉 Model training completed and saved!")
        print(f"📊 Trained on {len(df)} samples")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()