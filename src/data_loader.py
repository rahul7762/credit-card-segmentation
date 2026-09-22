"""
data_loader.py
---------------
Responsible ONLY for loading the CSV and doing a first inspection.
"""

import pandas as pd

try:
    from .config import DEFAULT_DATA_PATH
except ImportError:
    from config import DEFAULT_DATA_PATH


def load_data(path: str = None) -> pd.DataFrame:
    """Load the credit card dataset from a CSV file."""
    path = path or DEFAULT_DATA_PATH
    df = pd.read_csv(path)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("SHAPE:", df.shape)
    print("=" * 60)
    print("\nCOLUMN INFO:")
    print(df.info())
    print("\nMISSING VALUES PER COLUMN:")
    print(df.isnull().sum())
    print("\nDUPLICATE ROWS:", df.duplicated().sum())
    print("\nBASIC STATISTICS:")
    print(df.describe().T)


if __name__ == "__main__":
    df = load_data()
    inspect_data(df)