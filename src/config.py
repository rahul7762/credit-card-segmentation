"""
config.py
---------
Single source of truth for paths and constants. Uses pathlib + __file__
so paths work correctly NO MATTER where you run the script from.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DEFAULT_DATA_PATH = DATA_DIR / "credit_card_customers.csv"

DEFAULT_FEATURES = [
    "BALANCE",
    "PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "PRC_FULL_PAYMENT",
]

DEFAULT_N_CLUSTERS = 4

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)