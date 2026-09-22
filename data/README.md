# Data Folder

Place `credit_card_customers.csv` here.

**Real dataset (Kaggle):**
https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

**Columns expected (18 total):**
CUST_ID, BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES,
INSTALLMENTS_PURCHASES, CASH_ADVANCE, PURCHASES_FREQUENCY,
ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY,
CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX, PURCHASES_TRX, CREDIT_LIMIT,
PAYMENTS, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT, TENURE

**No Kaggle account yet?** Run this to generate a synthetic file with the same
columns so you can test the whole pipeline immediately:

```bash
python ../src/generate_sample_data.py
```

Remember to record: source URL, download date, and dataset version in your
final report (this is required by the project guide's reproducibility section).
