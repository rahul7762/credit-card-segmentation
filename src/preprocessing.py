"""
preprocessing.py
------------------
Cleaning + feature selection + scaling + outlier reporting.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from .config import DEFAULT_FEATURES
except ImportError:
    from config import DEFAULT_FEATURES


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "MINIMUM_PAYMENTS" in df.columns:
        df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(
            df["MINIMUM_PAYMENTS"].median()
        )
    if "CREDIT_LIMIT" in df.columns:
        df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())
    df = df.drop_duplicates()
    return df


def select_features(df: pd.DataFrame, features: list = None) -> pd.DataFrame:
    features = features or DEFAULT_FEATURES
    return df[features].copy()


def detect_outliers(df: pd.DataFrame, features: list = None) -> pd.DataFrame:
    """Report (do NOT remove) outliers using the IQR method."""
    features = features or DEFAULT_FEATURES
    report = []
    for col in features:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        report.append({
            "feature": col,
            "outlier_count": int(outlier_count),
            "outlier_pct": round(100 * outlier_count / len(df), 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
        })
    return pd.DataFrame(report)


def scale_features(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler