
# Credit Card Customer Behaviour Segmentation

**Unsupervised customer segmentation using Agglomerative Hierarchical Clustering**

An end-to-end machine learning pipeline that discovers natural behavioural segments among credit-card customers — without any predefined labels — and translates them into interpretable business profiles and recommendations.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Module Reference](#module-reference)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Tech Stack](#tech-stack)


---

## Overview

Financial institutions often have large volumes of transaction data but no ground-truth label describing customer type (e.g. "premium," "cash-advance-dependent," "low-activity"). This project applies **unsupervised learning** — specifically **Agglomerative Hierarchical Clustering** — to segment ~2,000 credit-card holders based on 9 behavioural features, then profiles each resulting cluster and generates rule-based business recommendations.

**Problem type:** Unsupervised clustering (no target variable)
**Algorithm:** Agglomerative Hierarchical Clustering (Ward linkage)
**Cluster selection:** Dendrogram inspection + Silhouette Score
**Output:** Cluster profiles, visualizations, and auto-generated business recommendations

---

## Key Results

Running the full pipeline on the dataset produces **4 distinct, interpretable customer segments**:

| Cluster | Size | Behaviour Profile | Recommended Action |
|---|---|---|---|
| 0 | 35.2% | Low balance, low purchases, low credit limit | Basic engagement campaigns |
| 1 | 29.4% | Moderate spend, high full-payment rate | Loyalty offers / credit limit increases |
| 2 | 15.7% | High cash-advance usage and frequency | Financial-product education |
| 3 | 19.7% | High purchases, high credit limit, high payments | Premium services / retention priority |

Silhouette scores were evaluated across `k = 2..7`; `k = 3` produced the mathematically highest score (0.459), while `k = 4` (0.353) was selected as the final model for stronger business interpretability — consistent with the project's evaluation philosophy of balancing statistical separation with actionable insight.

---

## Project Structure

```
credit_card_segmentation/
│
├── data/
│   ├── README.md                     Dataset source and setup instructions
│   └── credit_card_customers.csv     Input data (user-provided or generated)
│
├── notebooks/
│   └── customer_segmentation.ipynb   Step-by-step exploratory notebook
│
├── src/
│   ├── __init__.py
│   ├── config.py                     Centralized paths and constants
│   ├── data_loader.py                Data loading and initial inspection
│   ├── preprocessing.py              Cleaning, feature selection, scaling
│   ├── clustering.py                 Core pipeline: dendrogram, clustering, profiling
│   ├── visualization.py              EDA and cluster visualization plots
│   ├── business_rules.py             Rule-based cluster labeling and recommendations
│   └── generate_sample_data.py       Synthetic dataset generator for testing
│
├── outputs/                          Generated artifacts (plots, CSVs)
├── requirements.txt
└── README.md
```

---

## Methodology

The pipeline follows a standard unsupervised-learning workflow:

```
Dataset → Data Understanding → Cleaning → EDA → Feature Selection
   → Standardization → Dendrogram → Cluster Count Selection
   → Agglomerative Clustering → Cluster Labels → Visualization
   → Cluster Profiling → Evaluation → Business Recommendations
```

**1. Data Cleaning**
Missing values in `MINIMUM_PAYMENTS` and `CREDIT_LIMIT` are imputed using the **median**, chosen over the mean because financial variables are typically right-skewed by a small number of high-value customers.

**2. Feature Selection**
Nine behavioural features are retained (`BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `PURCHASES_FREQUENCY`, `CASH_ADVANCE_FREQUENCY`, `PURCHASES_TRX`, `CREDIT_LIMIT`, `PAYMENTS`, `PRC_FULL_PAYMENT`). `CUST_ID` is excluded as it is an identifier, not a behavioural signal.

**3. Standardization**
All features are scaled using `StandardScaler`:

  z = (x − μ) / σ

This is required because Agglomerative Clustering relies on distance calculations; without scaling, features with larger numeric ranges (e.g. `CREDIT_LIMIT`) would dominate the distance metric.

**4. Hierarchical Clustering**
A dendrogram is built using **Ward linkage** with **Euclidean distance**, which minimizes the increase in within-cluster variance at each merge step, tending to produce compact, evenly-sized clusters.

**5. Cluster Count Selection**
The number of clusters is chosen by combining:
- Visual inspection of the dendrogram for a large merge-distance gap
- Silhouette Score across a range of `k` values
- Business interpretability of the resulting profiles

**6. Cluster Profiling**
Each cluster's mean feature values are computed to translate anonymous numeric labels (0, 1, 2, 3) into interpretable behavioural profiles.

**7. Business Recommendations**
A transparent, rule-based system (`business_rules.py`) compares each cluster's averages against the overall population average using fixed thresholds (e.g. cash advance > 1.5× average → "cash-advance heavy") to generate a label and a suggested action — fully explainable, with no black-box model involved.

---

## Installation

```bash
git clone <repository-url>
cd credit_card_segmentation
pip install -r requirements.txt
```

**Requirements:** Python 3.9+

---

## Usage

### 1. Provide the dataset

**Option A — Real data (recommended)**
Download from Kaggle: [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
Save as `data/credit_card_customers.csv`.

**Option B — Synthetic data (for testing without Kaggle access)**
```bash
python src/generate_sample_data.py
```

### 2. Run the full pipeline

```bash
cd src
python clustering.py
```

or, from the project root, as a module:

```bash
python -m src.clustering
```

This executes the complete workflow and writes the following to `outputs/`:

| File | Description |
|---|---|
| `correlation_heatmap.png` | Feature correlation matrix |
| `dendrogram.png` | Hierarchical merge structure |
| `cluster_plot.png` | 2D scatter of clusters (PURCHASES vs CREDIT_LIMIT) |
| `pca_clusters.png` | PCA-reduced 2D cluster visualization |
| `cluster_profile.csv` | Per-cluster averages, labels, and recommendations |

### 3. Explore interactively

```bash
jupyter notebook notebooks/customer_segmentation.ipynb
```

---

## Module Reference

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for file paths (via `pathlib`) and shared constants |
| `data_loader.py` | `load_data()`, `inspect_data()` |
| `preprocessing.py` | `clean_data()`, `select_features()`, `scale_features()` |
| `clustering.py` | `build_dendrogram()`, `find_best_k()`, `train_agglomerative()`, `profile_clusters()`, `run_pipeline()` |
| `visualization.py` | `plot_correlation_heatmap()`, `plot_two_feature_scatter()`, `plot_pca_clusters()` |
| `business_rules.py` | `generate_business_recommendations()`, `evaluate_cluster_health()` |
| `generate_sample_data.py` | Synthetic dataset generator matching the real schema, for pipeline testing |

Imports throughout `src/` use a `try/except` pattern to support execution as a standalone script, as a package module (`python -m src.clustering`), or from within a notebook.

---

## Dataset

**Source:** [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) (Kaggle)
**Size:** ~9,000 active credit-card holders, 6 months of behavioural data, 18 features

| Feature | Description |
|---|---|
| `CUST_ID` | Customer identifier (excluded from clustering) |
| `BALANCE` | Account balance |
| `PURCHASES` | Total purchase amount |
| `CASH_ADVANCE` | Amount drawn as cash advance |
| `PURCHASES_FREQUENCY` | Frequency of purchases |
| `CASH_ADVANCE_FREQUENCY` | Frequency of cash advances |
| `CREDIT_LIMIT` | Assigned credit limit |
| `PAYMENTS` | Amount paid by the customer |
| `PRC_FULL_PAYMENT` | Proportion of full-payment behaviour |
| `TENURE` | Customer relationship duration |

There is no target variable — this is by design, as the project's objective is unsupervised discovery of customer segments.

---

## Evaluation

Since clustering has no ground-truth labels, evaluation relies on:

- **Silhouette Score** — measures how well-separated clusters are (range −1 to 1)
- **Cluster size distribution** — flags degenerate solutions where one cluster dominates (>80%) or another is negligible (<2%)
- **Dendrogram structure** — visual confirmation of natural separation
- **Business interpretability** — whether each cluster tells a coherent, actionable story

---

## Limitations

- Cluster labels (0, 1, 2, 3) are arbitrary and can change between runs — always compare cluster *profiles*, not numeric IDs.
- The rule-based business recommendations are heuristic starting points, not validated financial conclusions; they should not be used to infer creditworthiness or financial risk..
- Results depend on the feature set and scaling choices; alternative feature sets may surface different segment structures.

---

## Tech Stack 

- **Python 3.9+**
- **pandas**, **NumPy** — data manipulation
- **scikit-learn** — `StandardScaler`, `AgglomerativeClustering`, `silhouette_score`, `PCA`
- **SciPy** — hierarchical clustering (`linkage`, `dendrogram`)
- **Matplotlib**, **Seaborn** — visualization
- **Jupyter** — exploratory analysis

