"""
tests/test_preprocessing.py
=============================================================================
Unit Testing Suite for the ML Preprocessing Pipeline.
Validates the data cleaning and feature engineering logic.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import engineer_features, identify_redundant_features

def test_feature_engineering_logic():
    """Verify that derived features (Aggregates, Ratios) are correctly calculated."""
    # Create raw sample data with columns needed for the actual engineer_features function
    data = {
        'MonetaryTotal': [1000.0, 2000.0],
        'Frequency': [10, 20],
        'Recency': [5, 10],
        'CustomerTenureDays': [100, 200],
        'NegativeQuantityCount': [1, 2],
        'TotalTransactions': [10, 20],
        'CancelledTransactions': [0, 1]
    }
    df = pd.DataFrame(data)
    
    # Process
    df_feat = engineer_features(df)
    
    # Assertions for derived ratios
    assert 'AvgBasketValue' in df_feat.columns
    assert df_feat['AvgBasketValue'].iloc[0] == 100.0 # 1000 / 10
    assert 'TenureRatio' in df_feat.columns
    assert df_feat['TenureRatio'].iloc[0] == 0.05 # 5 / 100
    assert 'ReturnRate' in df_feat.columns
    assert df_feat['ReturnRate'].iloc[0] == 0.1 # 1 / 10

def test_protected_feature_whitelist():
    """Ensure that the 'PROTECTED' features are never dropped by the selection logic."""
    data = {
        'Age': [30, 30, 30], # Constant (low variance)
        'SupportTicketsCount': [0, 0, 0], # Constant
        'RandomNoise': np.random.randn(3),
        'Churn': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    # identify_redundant_features should PROTECT Age and SupportTicketsCount 
    # despite zero variance.
    dropped = identify_redundant_features(df, variance_threshold=0.1)
    
    assert "Age" not in dropped
    assert "SupportTicketsCount" not in dropped
    assert "Churn" not in dropped
