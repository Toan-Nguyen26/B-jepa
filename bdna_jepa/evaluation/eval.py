"""Downstream evaluation: kNN, linear probe, GC regression, clustering, visualization."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def knn_species_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: list[int] = (1, 3, 5, 10, 20, 50),
    cv: int = 5,
) -> dict[int, dict[str, float]]:
    """k-Nearest Neighbors species classification."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        scores = cross_val_score(knn, embeddings, labels, cv=cv, scoring="accuracy")
        results[k] = {"mean": float(scores.mean()), "std": float(scores.std())}
    return results


def linear_probe_classification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cv: int = 5,
) -> dict[str, float]:
    """Linear probe classification accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    scores = cross_val_score(pipe, embeddings, labels, cv=cv, scoring="accuracy")
    return {"accuracy": float(scores.mean()), "std": float(scores.std())}


def gc_regression(
    embeddings: np.ndarray,
    gc_values: np.ndarray,
    cv: int = 5,
) -> dict[str, float]:
    """GC content regression from embeddings."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])
    scores = cross_val_score(pipe, embeddings, gc_values, cv=cv, scoring="r2")
    return {"r2": float(scores.mean()), "r2_std": float(scores.std())}


def compute_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: Optional[int] = None,
) -> dict[str, float]:
    """Clustering evaluation: silhouette, ARI, NMI."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

    if n_clusters is None:
        n_clusters = len(set(labels))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    pred_labels = kmeans.fit_predict(embeddings)

    return {
        "silhouette": float(silhouette_score(embeddings, pred_labels)),
        "ari": float(adjusted_rand_score(labels, pred_labels)),
        "nmi": float(normalized_mutual_info_score(labels, pred_labels)),
    }


def plot_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    title: str = "UMAP",
    dpi: int = 300,
) -> None:
    """UMAP visualization colored by labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from umap import UMAP

    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        mask_arr = labels == label
        ax.scatter(coords[mask_arr, 0], coords[mask_arr, 1], s=5, alpha=0.5, label=str(label))

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6, markerscale=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_analysis(
    embeddings: np.ndarray,
    output_path: str,
    dpi: int = 300,
) -> None:
    """Plot singular value spectrum and cumulative variance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bdna_jepa.utils.metrics import compute_spectral_analysis

    analysis = compute_spectral_analysis(torch.from_numpy(embeddings))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    s = analysis["singular_values"]
    ax1.loglog(range(1, len(s) + 1), s, "b-", linewidth=1)
    ax1.set_xlabel("Rank")
    ax1.set_ylabel("Singular value")
    ax1.set_title(f"Spectrum (alpha={analysis['power_law_alpha']:.2f})")
    ax1.grid(True, alpha=0.3)

    cumvar = analysis["cumulative_variance"]
    ax2.plot(range(1, len(cumvar) + 1), cumvar, "r-", linewidth=1)
    ax2.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95%")
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative variance")
    ax2.set_title(f"RankMe={analysis['effective_rank']:.1f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
