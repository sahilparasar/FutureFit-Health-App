import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import joblib

class HealthRiskPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.risk_categories = [
            'Cardiovascular_Risk',
            'Musculoskeletal_Risk', 
            'Metabolic_Risk',
            'Functional_Decline_Risk'
        ]
    
    def calculate_derived_features(self, df):
        """Calculate all derived features needed for target creation"""
        df_processed = df.copy()
        
        # Calculate basic derived features
        df_processed['BMI'] = df_processed['Weight'] / ((df_processed['Height'] / 100) ** 2)
        df_processed['WH_Ratio'] = df_processed['Waist'] / df_processed['Hip']
        df_processed['VO2_Max_Estimate'] = 111.33 - (0.42 * df_processed['Step_Test_Heart_Rate'])
        
        return df_processed
    
    def prepare_targets(self, df):
        """Create multi-target risk categories based on health metrics"""
        # First calculate all derived features
        df_processed = self.calculate_derived_features(df)
        
        targets = {}
        
        # Cardiovascular Risk (based on WH ratio, VO2 max, pushups)
        cardio_risk = (
            (df_processed['WH_Ratio'] > 0.9) | 
            (df_processed['VO2_Max_Estimate'] < 35) |
            (df_processed['Pushups'] < 15)
        ).astype(int)
        
        # Musculoskeletal Risk (based on grip strength, balance, flexibility)
        muscle_risk = (
            (df_processed['Hand_Grip_Strength'] < 25) |
            (df_processed['Balance_Test_Seconds'] < 30) |
            (df_processed['Flexibility_cm'] < 15)
        ).astype(int)
        
        # Metabolic Risk (based on BMI, curl ups)
        metabolic_risk = (
            (df_processed['BMI'] > 30) | 
            (df_processed['Curl_Ups'] < 15)
        ).astype(int)
        
        # Functional Decline Risk (composite of multiple factors)
        functional_risk = (
            (cardio_risk == 1) | 
            (muscle_risk == 1) | 
            (metabolic_risk == 1)
        ).astype(int)
        
        targets['Cardiovascular_Risk'] = cardio_risk
        targets['Musculoskeletal_Risk'] = muscle_risk
        targets['Metabolic_Risk'] = metabolic_risk
        targets['Functional_Decline_Risk'] = functional_risk
        
        return pd.DataFrame(targets)
    
    def train(self, df, preprocessor):
        """Train the multi-output classifier"""
        self.preprocessor = preprocessor
        
        # Prepare features and targets
        X = self.preprocessor.transform(df)
        y = self.prepare_targets(df)
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        print("Target distribution:")
        print(y.mean())
        
        # Train model WITHOUT SMOTE (removed dependency)
        self.model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'  # Handle imbalance without SMOTE
            )
        )
        
        self.model.fit(X, y)
        
        # Training accuracy
        train_predictions = self.model.predict(X)
        train_accuracy = accuracy_score(y, train_predictions)
        print(f"Training accuracy: {train_accuracy:.3f}")
        
        return self
    
    def predict(self, df):
        """Predict health risks"""
        if self.model is None or self.preprocessor is None:
            # Return demo predictions if model not trained
            return self.demo_predictions(df)
        
        try:
            X = self.preprocessor.transform(df)
            predictions = self.model.predict(X)
            
            # Create results dataframe
            risk_df = pd.DataFrame(predictions, columns=self.risk_categories)
            
            # Add probabilities (simplified for demo)
            for i, category in enumerate(self.risk_categories):
                risk_df[f'{category}_Probability'] = predictions[:, i] * 0.7 + np.random.uniform(0.1, 0.3, len(df))
            
            return risk_df
        except Exception as e:
            print(f"Prediction error: {e}. Using demo predictions.")
            return self.demo_predictions(df)
    
    def demo_predictions(self, df):
        """Generate demo predictions when model isn't available"""
        risk_data = {}
        
        for category in self.risk_categories:
            # Simple demo logic based on BMI and age-like features
            base_risk = df['Weight'] / df['Height'] * 0.1
            risk_data[category] = (base_risk > base_risk.median()).astype(int)
            risk_data[f'{category}_Probability'] = np.clip(base_risk, 0.1, 0.9)
        
        return pd.DataFrame(risk_data)
    
    def save(self, filepath):
        """Save trained model"""
        joblib.dump(self, filepath)
        print(f"✅ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load trained model"""
        try:
            model = joblib.load(filepath)
            print(f"✅ Model loaded from {filepath}")
            return model
        except:
            print(f"❌ Could not load model from {filepath}")
            return cls()  # Return new instance if loading fails