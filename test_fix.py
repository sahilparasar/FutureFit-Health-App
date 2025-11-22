import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

from models.trainer import HealthRiskPredictor
from utils.helpers import generate_sample_data

print("Testing the fix...")

# Generate sample data
df = generate_sample_data()
print("Sample data columns:", df.columns.tolist())

# Test the target preparation
predictor = HealthRiskPredictor()
targets = predictor.prepare_targets(df)

print("Targets shape:", targets.shape)
print("Targets columns:", targets.columns.tolist())
print("Targets sample:")
print(targets.head())

print("✅ Fix verified! Targets created successfully.")