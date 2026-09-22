"""
business_rules.py
------------------
Turns raw cluster averages into human-readable labels and suggested
business actions - automatically, using simple threshold rules.
"""

import pandas as pd

ACTION_MAP = {
    "cash-advance heavy": "Financial-product education / targeted offers to reduce cash-advance reliance",
    "high-value": "Premium services / rewards / retention priority",
    "disciplined full-payer": "Loyalty offers / low credit risk, good candidate for limit increases",
    "low-activity": "Basic engagement campaigns / reactivation offers",
    "regular/mixed behaviour": "Standard engagement, monitor for movement into other segments",
}


def _tag_cluster(row: pd.Series, avg: pd.Series) -> list:
    tags = []

    if "CASH_ADVANCE" in row and avg.get("CASH_ADVANCE", 0) > 0:
        if row["CASH_ADVANCE"] > 1.5 * avg["CASH_ADVANCE"]:
            tags.append("cash-advance heavy")

    if "PURCHASES" in row and "CREDIT_LIMIT" in row:
        if (row["PURCHASES"] > 1.3 * avg.get("PURCHASES", 0)
                and row["CREDIT_LIMIT"] > 1.3 * avg.get("CREDIT_LIMIT", 0)):
            tags.append("high-value")

    if "PRC_FULL_PAYMENT" in row and avg.get("PRC_FULL_PAYMENT", 0) > 0:
        if row["PRC_FULL_PAYMENT"] > 1.3 * avg["PRC_FULL_PAYMENT"]:
            tags.append("disciplined full-payer")

    if "PURCHASES" in row and "BALANCE" in row:
        if (row["PURCHASES"] < 0.6 * avg.get("PURCHASES", 1)
                and row["BALANCE"] < 0.6 * avg.get("BALANCE", 1)):
            tags.append("low-activity")

    if not tags:
        tags.append("regular/mixed behaviour")

    return tags


def generate_business_recommendations(profile: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in profile.columns if c != "count"]
    avg = profile[numeric_cols].mean()

    labels, actions = [], []
    for _, row in profile.iterrows():
        tags = _tag_cluster(row, avg)
        label = " & ".join(tags)
        action = " | ".join(ACTION_MAP[t] for t in tags)
        labels.append(label)
        actions.append(action)

    result = profile.copy()
    result["Label"] = labels
    result["Recommended Action"] = actions
    return result


def evaluate_cluster_health(df, cluster_col: str = "Cluster") -> None:
    counts = df[cluster_col].value_counts(normalize=True).sort_index()
    print("\nCluster size check:")
    for cluster_id, pct in counts.items():
        flag = ""
        if pct > 0.80:
            flag = "  ⚠️  WARNING: this cluster dominates (>80%) - review features/k"
        elif pct < 0.02:
            flag = "  ⚠️  WARNING: very small cluster (<2%) - may be noise/outliers"
        print(f"  Cluster {cluster_id}: {pct:.1%}{flag}")
        
def profile_dbscan_clusters(df, dbscan_labels, features):
    """Same idea as profile_clusters() but for DBSCAN - handles noise
    (-1 label) separately since it isn't a real cluster."""
    df = df.copy()
    df["DBSCAN_Cluster"] = dbscan_labels
    profile = df.groupby("DBSCAN_Cluster")[features].mean().round(2)
    profile["count"] = df["DBSCAN_Cluster"].value_counts().sort_index()
    return profile