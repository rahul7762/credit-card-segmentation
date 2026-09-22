"""
test_preprocessing.py
----------------------
Basic unit tests for preprocessing.py. Run with:
    pytest tests/
or:
    python -m pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import clean_data, select_features, scale_features, detect_outliers, DEFAULT_FEATURES


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "CUST_ID": ["C1", "C2", "C3", "C3"],
        "BALANCE": [100.0, 200.0, 300.0, 300.0],
        "PURCHASES": [10.0, 20.0, 30.0, 30.0],
        "CASH_ADVANCE": [0.0, 5.0, 10.0, 10.0],
        "PURCHASES_FREQUENCY": [0.1, 0.2, 0.3, 0.3],
        "CASH_ADVANCE_FREQUENCY": [0.0, 0.1, 0.2, 0.2],
        "PURCHASES_TRX": [1, 2, 3, 3],
        "CREDIT_LIMIT": [1000.0, np.nan, 3000.0, 3000.0],
        "PAYMENTS": [90.0, 180.0, 270.0, 270.0],
        "MINIMUM_PAYMENTS": [10.0, np.nan, 30.0, 30.0],
        "PRC_FULL_PAYMENT": [0.5, 0.6, 0.7, 0.7],
    })


def test_clean_data_fills_missing_values(sample_df):
    cleaned = clean_data(sample_df)
    assert cleaned["CREDIT_LIMIT"].isnull().sum() == 0
    assert cleaned["MINIMUM_PAYMENTS"].isnull().sum() == 0


def test_clean_data_removes_duplicates(sample_df):
    cleaned = clean_data(sample_df)
    assert cleaned.duplicated().sum() == 0


def test_select_features_excludes_cust_id(sample_df):
    X = select_features(sample_df, DEFAULT_FEATURES)
    assert "CUST_ID" not in X.columns


def test_select_features_returns_expected_columns(sample_df):
    X = select_features(sample_df, DEFAULT_FEATURES)
    assert list(X.columns) == DEFAULT_FEATURES


def test_scale_features_produces_zero_mean(sample_df):
    cleaned = clean_data(sample_df)
    X = select_features(cleaned, DEFAULT_FEATURES)
    X_scaled, scaler = scale_features(X)

    means = X_scaled.mean(axis=0)
    stds = X_scaled.std(axis=0)

    assert np.allclose(means, 0, atol=1e-8)
    assert np.allclose(stds, 1, atol=1e-2)


def test_detect_outliers_returns_report_without_modifying_data(sample_df):
    cleaned = clean_data(sample_df)
    original_shape = cleaned.shape
    report = detect_outliers(cleaned, DEFAULT_FEATURES)

    assert cleaned.shape == original_shape
    assert "outlier_count" in report.columns
    assert "feature" in report.columns
    