import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

class HealthDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_fitted = False
        
    def calculate_bmi(self, df):
        """Calculate Body Mass Index"""
        return df['Weight'] / ((df['Height'] / 100) ** 2)
    
    def calculate_wh_ratio(self, df):
        """Calculate Waist-to-Hip ratio"""
        return df['Waist'] / df['Hip']
    
    def calculate_vo2_max(self, df):
        """Estimate VO2 Max from step test"""
        # Using Queens College Step Test formula
        return 111.33 - (0.42 * df['Step_Test_Heart_Rate'])
    
    def create_features(self, df):
        """Create engineered features"""
        df_copy = df.copy()
        
        # Basic calculations
        df_copy['BMI'] = self.calculate_bmi(df_copy)
        df_copy['WH_Ratio'] = self.calculate_wh_ratio(df_copy)
        df_copy['VO2_Max_Estimate'] = self.calculate_vo2_max(df_copy)
        
        # Strength ratios
        df_copy['Strength_to_Weight'] = df_copy['Hand_Grip_Strength'] / df_copy['Weight']
        df_copy['Core_Endurance_Index'] = df_copy['Curl_Ups'] / df_copy['BMI']
        
        # Fitness scores
        df_copy['Upper_Body_Score'] = df_copy['Pushups'] / df_copy['Weight']
        df_copy['Balance_Score'] = df_copy['Balance_Test_Seconds']
        df_copy['Flexibility_Score'] = df_copy['Flexibility_cm']
        
        return df_copy
    
    def fit(self, df):
        """Fit preprocessor on training data"""
        df_processed = self.create_features(df)
        
        # Select feature columns
        self.feature_columns = [
            'Height', 'Weight', 'Hip', 'Waist', 'Hand_Grip_Strength',
            'Curl_Ups', 'Pushups', 'Step_Test_Heart_Rate', 
            'Balance_Test_Seconds', 'Flexibility_cm',
            'BMI', 'WH_Ratio', 'VO2_Max_Estimate', 
            'Strength_to_Weight', 'Core_Endurance_Index',
            'Upper_Body_Score', 'Balance_Score', 'Flexibility_Score'
        ]
        
        # Handle missing values
        X = df_processed[self.feature_columns]
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        self.scaler.fit(X_imputed)
        self.is_fitted = True
        
        return self
    
    def transform(self, df):
        """Transform new data"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        df_processed = self.create_features(df)
        X = df_processed[self.feature_columns]
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled
    
    def save(self, filepath):
        """Save preprocessor"""
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load preprocessor"""
        return joblib.load(filepath)