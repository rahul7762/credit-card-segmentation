

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

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

# Elbow Method
inertia = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker='o')

plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')

plt.xticks(range(2, 11))
plt.grid(True)
plt.show()

# K-Means Model

optimal_k = 4

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)
df['KMeans_Cluster'] = cluster_labels

print("\nCustomer count in each K-Means cluster:")
print(df['KMeans_Cluster'].value_counts().sort_index())

# Plot cluster distribution

cluster_counts = df['KMeans_Cluster'].value_counts().sort_index()

plt.figure(figsize=(8, 5))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.xlabel('K-Means Cluster')
plt.ylabel('Number of Customers')
plt.title('Customer Distribution Across K-Means Clusters')

plt.show()