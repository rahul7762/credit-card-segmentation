"""
clustering.py
--------------
The "engine" of the project: builds the dendrogram, tests k values with
silhouette score, trains the final AgglomerativeClustering model,
profiles the resulting clusters, runs EDA + visualization, and
auto-generates business recommendations - run_pipeline() covers the
FULL guide workflow end-to-end.

Run with defaults:
    python clustering.py

Run with custom cluster count (CLI arg):
    python clustering.py --k 5

Run as a package module:
    python -m src.clustering --k 5
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans ,DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
import matplotlib.pyplot as plt

try:
    from .config import DEFAULT_DATA_PATH, OUTPUT_DIR, DEFAULT_FEATURES, DEFAULT_N_CLUSTERS
    from .data_loader import load_data, inspect_data
    from .preprocessing import clean_data, select_features, scale_features, detect_outliers
    from .visualization import plot_correlation_heatmap, plot_two_feature_scatter, plot_pca_clusters, plot_outlier_boxplots, plot_dbscan_pca, plot_kmeans_elbow, plot_kmeans_clusters, plot_algorithm_comparison
    from .business_rules import generate_business_recommendations, evaluate_cluster_health, profile_dbscan_clusters
except ImportError:
    from config import DEFAULT_DATA_PATH, OUTPUT_DIR, DEFAULT_FEATURES, DEFAULT_N_CLUSTERS
    from data_loader import load_data, inspect_data
    from preprocessing import clean_data, select_features, scale_features, detect_outliers
    from visualization import plot_correlation_heatmap, plot_two_feature_scatter, plot_pca_clusters, plot_outlier_boxplots, plot_dbscan_pca, plot_kmeans_elbow, plot_kmeans_clusters, plot_algorithm_comparison
    from business_rules import generate_business_recommendations, evaluate_cluster_health, profile_dbscan_clusters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_dendrogram(X_scaled, save_path=None):
    save_path = save_path or (OUTPUT_DIR / "dendrogram.png")
    linked = linkage(X_scaled, method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(linked, truncate_mode="level", p=5)
    plt.xlabel("Customers / merged clusters")
    plt.ylabel("Distance")
    plt.title("Hierarchical Clustering Dendrogram")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Dendrogram saved to {save_path}")
    return linked


def find_best_k(X_scaled, k_range=range(2, 8)) -> dict:
    scores = {}
    for k in k_range:
        model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
        labels = model.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    return scores


def train_agglomerative(X_scaled, n_clusters: int = DEFAULT_N_CLUSTERS):
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    labels = model.fit_predict(X_scaled)
    return model, labels


def compare_with_kmeans(X_scaled, n_clusters: int = DEFAULT_N_CLUSTERS, agglo_labels=None) -> dict:
    """Train K-Means with the same k and compare against Agglomerative
    Clustering. Useful for the common interview question: "Why did you
    pick Agglomerative Clustering over K-Means?"
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

    result = {
        "kmeans_labels": kmeans_labels,
        "kmeans_silhouette": kmeans_silhouette,
        "kmeans_centers": kmeans.cluster_centers_,
    }

    if agglo_labels is not None:
        agglo_silhouette = silhouette_score(X_scaled, agglo_labels)
        result["agglomerative_silhouette"] = agglo_silhouette
        logger.info(
            f"Silhouette comparison — Agglomerative: {agglo_silhouette:.4f} | "
            f"K-Means: {kmeans_silhouette:.4f}"
        )

    return result


def kmeans_elbow_data(X_scaled, k_range=range(1, 11)):
    """Inertia for every k (elbow method) + silhouette for k >= 2."""
    inertias, silhouettes = [], {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        inertias.append(km.inertia_)
        if k >= 2:
            silhouettes[k] = silhouette_score(X_scaled, km.labels_)
    return list(k_range), inertias, silhouettes


def build_algorithm_comparison(X_scaled, n_clusters=DEFAULT_N_CLUSTERS,
                               eps=1.5, min_samples=10):
    """Fit all 3 algorithms on the SAME scaled data and collect one
    side-by-side metrics table (for the presentation's comparison slide).

    Metrics: Silhouette (higher better), Davies-Bouldin (LOWER better),
    Calinski-Harabasz (higher better). For DBSCAN, noise points (-1) are
    excluded before scoring because they belong to no cluster.
    """
    models = {
        "K-Means": ("Yes (k=%d)" % n_clusters,
                    lambda: KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
        "Agglomerative (Ward)": ("Yes (k=%d)" % n_clusters,
                    lambda: AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")),
        "DBSCAN": ("No (eps based)",
                    lambda: DBSCAN(eps=eps, min_samples=min_samples)),
    }

    labels_dict, rows = {}, {}
    for name, (needs_k, make) in models.items():
        t0 = time.perf_counter()
        labels = make().fit_predict(X_scaled)
        elapsed = time.perf_counter() - t0
        labels_dict[name] = labels

        mask = labels != -1
        n_found = len(set(labels[mask]))
        noise_pct = 100 * (~mask).sum() / len(labels)
        if n_found >= 2:
            sil = round(silhouette_score(X_scaled[mask], labels[mask]), 4)
            db = round(davies_bouldin_score(X_scaled[mask], labels[mask]), 3)
            ch = round(calinski_harabasz_score(X_scaled[mask], labels[mask]), 1)
        else:
            sil = db = ch = np.nan
        sizes = pd.Series(labels[mask]).value_counts().sort_index()
        rows[name] = {
            "Needs k?": needs_k,
            "Clusters found": n_found,
            "Noise %": round(noise_pct, 1),
            "Silhouette": sil,
            "Davies-Bouldin": db,
            "Calinski-Harabasz": ch,
            "Cluster sizes": " / ".join(str(v) for v in sizes.values),
            "Fit time (s)": round(elapsed, 3),
        }

    table = pd.DataFrame(rows).T

    # How much do the methods agree? (Adjusted Rand Index: 1 = identical)
    agglo = labels_dict["Agglomerative (Ward)"]
    table["ARI vs Agglo"] = [
        round(adjusted_rand_score(agglo, labels_dict[n]), 3) for n in table.index
    ]
    return table, labels_dict


def compare_with_dbscan(X_scaled, eps: float = 1.5, min_samples: int = 10, agglo_labels=None) -> dict:
    """Train DBSCAN and compare against Agglomerative Clustering.

    Unlike Agglomerative/K-Means, DBSCAN does NOT need a predefined
    number of clusters - it discovers dense regions automatically and
    labels points that don't belong to any dense region as noise (-1).
    This is useful for validating our outlier analysis: points DBSCAN
    calls "noise" should substantially overlap with points our IQR-based
    detect_outliers() flagged as outliers.

    eps and min_samples are the two key DBSCAN parameters:
    - eps: neighborhood radius - how close points must be to count as
      neighbors
    - min_samples: minimum neighbors needed for a point to be a "core"
      point that can start/extend a cluster
    Both need tuning per dataset - the defaults here are a reasonable
    starting point for 9 standardized features, not a universal answer.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    n_clusters_found = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = int((dbscan_labels == -1).sum())
    noise_pct = 100 * n_noise / len(dbscan_labels)

    logger.info(
        f"DBSCAN found {n_clusters_found} clusters, "
        f"{n_noise} noise points ({noise_pct:.1f}%)"
    )

    result = {
        "dbscan_labels": dbscan_labels,
        "n_clusters_found": n_clusters_found,
        "n_noise": n_noise,
        "noise_pct": noise_pct,
    }

    # Silhouette score only makes sense with 2+ clusters and needs the
    # noise points excluded (they aren't part of any cluster).
    if n_clusters_found >= 2:
        mask = dbscan_labels != -1
        dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
        result["dbscan_silhouette"] = dbscan_silhouette
        logger.info(f"DBSCAN silhouette (excluding noise): {dbscan_silhouette:.4f}")

        if agglo_labels is not None:
            agglo_silhouette = silhouette_score(X_scaled, agglo_labels)
            logger.info(
                f"Silhouette comparison — Agglomerative: {agglo_silhouette:.4f} | "
                f"DBSCAN: {dbscan_silhouette:.4f}"
            )
    else:
        logger.info(
            "DBSCAN found fewer than 2 clusters with these parameters - "
            "try adjusting eps/min_samples (smaller eps = more, smaller clusters)"
        )

    return result

def profile_clusters(df: pd.DataFrame, features: list, cluster_col: str = "Cluster") -> pd.DataFrame:
    profile = df.groupby(cluster_col)[features].mean().round(2)
    profile["count"] = df[cluster_col].value_counts().sort_index()
    return profile


def run_pipeline(data_path=None,
                n_clusters: int = DEFAULT_N_CLUSTERS,
                output_dir=None,
                run_kmeans_comparison: bool = True,
                run_dbscan_comparison: bool = True,
                dbscan_eps: float = 1.5,
                dbscan_min_samples: int = 10):
    data_path = data_path or DEFAULT_DATA_PATH
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset...")
    df_raw = load_data(data_path)
    inspect_data(df_raw)

    logger.info("Cleaning data...")
    df_clean = clean_data(df_raw)

    plot_correlation_heatmap(df_clean, save_path=output_dir / "correlation_heatmap.png")
    logger.info(f"Correlation heatmap saved to {output_dir / 'correlation_heatmap.png'}")

    outlier_report = detect_outliers(df_clean)
    logger.info("Outlier report (IQR method, values NOT removed automatically):")
    print(outlier_report.to_string(index=False))

    plot_outlier_boxplots(df_clean, save_path=output_dir / "outlier_boxplots.png")
    logger.info(f"Outlier boxplots saved to {output_dir / 'outlier_boxplots.png'}")

    X = select_features(df_clean, DEFAULT_FEATURES)

    logger.info("Scaling features...")
    X_scaled, scaler = scale_features(X)

    build_dendrogram(X_scaled, save_path=output_dir / "dendrogram.png")

    scores = find_best_k(X_scaled)
    logger.info("Silhouette scores by k:")
    for k, s in scores.items():
        logger.info(f"  k={k}: {s:.4f}")

    logger.info(f"Training Agglomerative Clustering with k={n_clusters}...")
    model, labels = train_agglomerative(X_scaled, n_clusters=n_clusters)
    df_clean = df_clean.copy()
    df_clean["Cluster"] = labels

    if run_kmeans_comparison:
        kmeans_result = compare_with_kmeans(X_scaled, n_clusters=n_clusters, agglo_labels=labels)

        k_values, inertias, km_sil = kmeans_elbow_data(X_scaled)
        plot_kmeans_elbow(k_values, inertias, silhouettes=km_sil, chosen_k=n_clusters,
                          save_path=output_dir / "kmeans_elbow_curve.png")
        plot_kmeans_clusters(X_scaled, kmeans_result["kmeans_labels"],
                             centers=kmeans_result["kmeans_centers"],
                             save_path=output_dir / "kmeans_clusters.png")
        logger.info(f"K-Means plots saved to {output_dir}")
    
    
    if run_dbscan_comparison:
        dbscan_result = compare_with_dbscan(X_scaled, eps=dbscan_eps, min_samples=dbscan_min_samples, agglo_labels=labels)

        # Save DBSCAN labels as a dataframe column
        df_clean["DBSCAN_Cluster"] = dbscan_result["dbscan_labels"]
        logger.info("DBSCAN cluster counts:")
        print(df_clean["DBSCAN_Cluster"].value_counts().sort_index())

        # DBSCAN-specific PCA plot with noise marked
        plot_dbscan_pca(X_scaled, dbscan_result["dbscan_labels"], save_path=output_dir / "dbscan_pca.png")
        logger.info(f"DBSCAN PCA plot saved to {output_dir / 'dbscan_pca.png'}")

        # DBSCAN business profile
        dbscan_profile = profile_dbscan_clusters(df_clean, dbscan_result["dbscan_labels"], DEFAULT_FEATURES)
        dbscan_profile.to_csv(output_dir / "dbscan_profile.csv")
        logger.info("DBSCAN cluster profile:")
        print(dbscan_profile)
        
    if run_kmeans_comparison and run_dbscan_comparison:
        comp_table, comp_labels = build_algorithm_comparison(
            X_scaled, n_clusters=n_clusters, eps=dbscan_eps, min_samples=dbscan_min_samples)
        comp_table.to_csv(output_dir / "algorithm_comparison.csv")
        plot_algorithm_comparison(X_scaled, comp_labels, comp_table,
                                  save_path=output_dir / "algorithm_comparison.png")
        logger.info("Algorithm comparison table:")
        print(comp_table.to_string())
        logger.info(f"Comparison figure + CSV saved to {output_dir}")

    plot_two_feature_scatter(df_clean, x="PURCHASES", y="CREDIT_LIMIT",
                            save_path=output_dir / "cluster_plot.png")
    plot_pca_clusters(X_scaled, labels, save_path=output_dir / "pca_clusters.png")
    logger.info(f"Cluster plots saved to {output_dir}")

    profile = profile_clusters(df_clean, DEFAULT_FEATURES)

    evaluate_cluster_health(df_clean)

    recommendations = generate_business_recommendations(profile)
    recommendations.to_csv(output_dir / "cluster_profile.csv")
    logger.info("Pipeline complete. Cluster profile + recommendations:")
    print(recommendations[["count", "Label", "Recommended Action"]])

    return df_clean, recommendations, scores


def _parse_args():
    parser = argparse.ArgumentParser(description="Credit card customer segmentation pipeline")
    parser.add_argument("--k", type=int, default=DEFAULT_N_CLUSTERS,
                        help=f"Number of clusters (default: {DEFAULT_N_CLUSTERS})")
    parser.add_argument("--no-kmeans-comparison", action="store_true",
                        help="Skip the K-Means comparison step")
    parser.add_argument("--no-dbscan-comparison", action="store_true",
                        help="Skip the DBSCAN comparison step")
    parser.add_argument("--eps", type=float, default=1.5,
                        help="DBSCAN neighborhood radius (default: 1.5)")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="DBSCAN minimum samples per core point (default: 10)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        n_clusters=args.k,
        run_kmeans_comparison=not args.no_kmeans_comparison,
        run_dbscan_comparison=not args.no_dbscan_comparison,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
    )