"""
generate_sample_data.py
------------------------
Creates a SYNTHETIC (fake) credit-card customer dataset with the same 18
columns as the real Kaggle dataset, so students can test the full pipeline
before/without downloading the real data.

Run:
    python src/generate_sample_data.py
"""

import numpy as np
import pandas as pd
import os

def generate_sample_data(n_customers: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic credit-card customer dataframe with 4 rough
    behaviour archetypes baked in, so clustering has something real to find."""
    rng = np.random.default_rng(seed)

    # We create 4 hidden "true" groups so the clustering has real structure
    # to discover (low-activity, regular, high-value, cash-advance-heavy).
    group = rng.choice([0, 1, 2, 3], size=n_customers, p=[0.35, 0.30, 0.20, 0.15])

    balance = np.where(group == 2, rng.normal(4000, 800, n_customers),
               np.where(group == 3, rng.normal(2500, 700, n_customers),
               np.where(group == 1, rng.normal(1200, 400, n_customers),
                                     rng.normal(300, 150, n_customers))))
    balance = np.clip(balance, 0, None)

    purchases = np.where(group == 2, rng.normal(3500, 900, n_customers),
                 np.where(group == 1, rng.normal(900, 300, n_customers),
                 np.where(group == 3, rng.normal(150, 80, n_customers),
                                       rng.normal(50, 40, n_customers))))
    purchases = np.clip(purchases, 0, None)

    cash_advance = np.where(group == 3, rng.normal(2000, 600, n_customers),
                             rng.normal(100, 80, n_customers))
    cash_advance = np.clip(cash_advance, 0, None)

    credit_limit = np.where(group == 2, rng.normal(9000, 1500, n_customers),
                    np.where(group == 3, rng.normal(5000, 1200, n_customers),
                    np.where(group == 1, rng.normal(3500, 900, n_customers),
                                          rng.normal(1500, 500, n_customers))))
    credit_limit = np.clip(credit_limit, 500, None)

    df = pd.DataFrame({
        "CUST_ID": [f"C{10000+i}" for i in range(n_customers)],
        "BALANCE": balance.round(2),
        "BALANCE_FREQUENCY": rng.uniform(0.2, 1.0, n_customers).round(2),
        "PURCHASES": purchases.round(2),
        "ONEOFF_PURCHASES": (purchases * rng.uniform(0.3, 0.7, n_customers)).round(2),
        "INSTALLMENTS_PURCHASES": (purchases * rng.uniform(0.1, 0.5, n_customers)).round(2),
        "CASH_ADVANCE": cash_advance.round(2),
        "PURCHASES_FREQUENCY": rng.uniform(0.0, 1.0, n_customers).round(2),
        "ONEOFF_PURCHASES_FREQUENCY": rng.uniform(0.0, 1.0, n_customers).round(2),
        "PURCHASES_INSTALLMENTS_FREQUENCY": rng.uniform(0.0, 1.0, n_customers).round(2),
        "CASH_ADVANCE_FREQUENCY": np.where(group == 3, rng.uniform(0.3, 0.9, n_customers),
                                            rng.uniform(0.0, 0.2, n_customers)).round(2),
        "CASH_ADVANCE_TRX": rng.integers(0, 20, n_customers),
        "PURCHASES_TRX": rng.integers(0, 60, n_customers),
        "CREDIT_LIMIT": credit_limit.round(2),
        "PAYMENTS": (balance * rng.uniform(0.5, 1.2, n_customers)).round(2),
        "MINIMUM_PAYMENTS": (balance * rng.uniform(0.05, 0.3, n_customers)).round(2),
        "PRC_FULL_PAYMENT": np.where(group == 1, rng.uniform(0.3, 0.9, n_customers),
                                      rng.uniform(0.0, 0.3, n_customers)).round(2),
        "TENURE": rng.choice([6, 7, 8, 9, 10, 11, 12], size=n_customers,
                              p=[0.02, 0.03, 0.05, 0.05, 0.10, 0.15, 0.60]),
    })

    # Sprinkle a few missing values, just like the real dataset has
    for col in ["MINIMUM_PAYMENTS", "CREDIT_LIMIT"]:
        missing_idx = rng.choice(df.index, size=int(0.01 * n_customers), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_sample_data()
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "credit_card_customers.csv")
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset saved to: {out_path}")
    print(f"Shape: {df.shape}")
    print(df.head())
