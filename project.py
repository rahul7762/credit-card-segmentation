

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Load data and basic exploration
df = pd.read_csv('cc general.csv')
df.head()
df.shape
df.info()
print(df.describe().T)
print("Missing values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# EDA

# Distribution of Purchases
df['PURCHASES'].hist(bins=50)
plt.title('Distribution of Purchases')
plt.xlabel('Purchases')
plt.ylabel('Number of Customers')
plt.show()


# Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(
    df.select_dtypes('number').corr(),
    cmap='coolwarm',
    annot=False
)
plt.title('Correlation Matrix')
plt.show()


# Box Plot - Numerical Features
plt.figure(figsize=(15, 8))
sns.boxplot(data=df.select_dtypes('number'))
plt.title('Box Plot of Numerical Features')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Individual Box Plots

plt.figure(figsize=(8, 5))
sns.boxplot(y=df['BALANCE'])
plt.title('Box Plot - Balance')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(y=df['PURCHASES'])
plt.title('Box Plot - Purchases')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(y=df['CASH_ADVANCE'])
plt.title('Box Plot - Cash Advance')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(y=df['CREDIT_LIMIT'])
plt.title('Box Plot - Credit Limit')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(y=df['MINIMUM_PAYMENTS'])
plt.title('Box Plot - Minimum Payments')
plt.show()


# Handle missing values
df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())
df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())

# feature scaling 
features = [
    col for col in df.columns
    if col != 'CUST_ID'
    and df[col].dtype in ['int64', 'float64']
]

X = df[features].copy()
X_scaled = StandardScaler().fit_transform(X)

