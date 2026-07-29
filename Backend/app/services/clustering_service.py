"""HDBSCAN clustering and silhouette quality metrics over embedding vectors.

Pure computation: no Redis, no object storage, no model loading. Everything here is
callable from a unit test with a plain list of vectors.

The output contract is consumed by the frontend (`ClusteringResult` in
`Frontend/src/contexts/EmbeddingContext.tsx`), so every key below is always present
even in degenerate cases -- the UI must never have to branch on a partial shape.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Above this dimensionality we pre-reduce with PCA before clustering. Density-based
# clustering degrades in very high dimensions, and mean-pooled encoder embeddings are
# 512-d (whisper-base) to 1280-d (whisper-large).
PCA_THRESHOLD_DIMS = 50

# HDBSCAN needs at least a handful of points before "density" means anything.
MIN_POINTS_FOR_CLUSTERING = 3

NOISE_LABEL = -1


def _empty_result(
    n_points: int, min_cluster_size: int, metric: str, reason: str
) -> dict[str, Any]:
    """A well-formed payload for cases where clustering cannot run."""
    return {
        "labels": [NOISE_LABEL] * n_points,
        "probabilities": [],
        "n_clusters": 0,
        "n_noise": n_points,
        "silhouette_score": None,
        "silhouette_samples": [None] * n_points,
        "cluster_stats": [],
        "params": {
            "min_cluster_size": min_cluster_size,
            "metric": metric,
            "pca_dims": None,
        },
        "skipped_reason": reason,
    }


def _prepare_matrix(vectors: list[list[float]]) -> tuple[np.ndarray, int | None]:
    """L2-normalise, then optionally PCA-reduce. Returns (matrix, pca_dims_used)."""
    matrix = np.asarray(vectors, dtype=float)

    # L2-normalise so euclidean distance on the result is monotonic with cosine
    # distance -- the standard treatment for mean-pooled transformer embeddings.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    n_samples, n_features = matrix.shape
    if n_features <= PCA_THRESHOLD_DIMS:
        return matrix, None

    # PCA cannot produce more components than min(n_samples, n_features).
    pca_dims = int(min(PCA_THRESHOLD_DIMS, n_samples - 1, n_features))
    if pca_dims < 2:
        return matrix, None

    from app.services.model_loader_service import reduce_dimensions

    reduced = np.asarray(reduce_dimensions(list(matrix), method="pca", n_components=pca_dims))
    return reduced, pca_dims


def _medoid_index(matrix: np.ndarray, member_indices: np.ndarray) -> int:
    """Index (into the original input) of the cluster's most central member."""
    members = matrix[member_indices]
    # Pairwise distances within the cluster; the medoid minimises total distance.
    deltas = members[:, None, :] - members[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1).sum(axis=1)
    return int(member_indices[int(np.argmin(distances))])


def cluster_embeddings(
    vectors: list[list[float]],
    min_cluster_size: int = 5,
    metric: str = "euclidean",
) -> dict[str, Any]:
    """Cluster embedding vectors with HDBSCAN and score the result.

    Args:
        vectors: One embedding per audio file, aligned with the caller's file order.
        min_cluster_size: Smallest group HDBSCAN will call a cluster.
        metric: Distance metric passed to HDBSCAN.

    Returns:
        A dict matching the frontend `ClusteringResult` contract. `labels[i]`
        corresponds to `vectors[i]`; `-1` means noise (no cluster).
    """
    n_points = len(vectors)
    if n_points < MIN_POINTS_FOR_CLUSTERING:
        return _empty_result(
            n_points, min_cluster_size, metric, f"needs at least {MIN_POINTS_FOR_CLUSTERING} files"
        )

    # HDBSCAN requires 2 <= min_cluster_size <= n_points.
    effective_min = max(2, min(int(min_cluster_size), n_points))

    matrix, pca_dims = _prepare_matrix(vectors)

    from sklearn.cluster import HDBSCAN

    clusterer = HDBSCAN(min_cluster_size=effective_min, metric=metric)
    labels = np.asarray(clusterer.fit_predict(matrix))
    probabilities = getattr(clusterer, "probabilities_", None)

    unique_labels = sorted({int(label) for label in labels if label != NOISE_LABEL})
    n_noise = int(np.sum(labels == NOISE_LABEL))

    # Silhouette is only defined for >= 2 clusters, and only over clustered points.
    silhouette_overall: float | None = None
    per_point: list[float | None] = [None] * n_points
    non_noise = np.flatnonzero(labels != NOISE_LABEL)
    if len(unique_labels) >= 2 and len(non_noise) > len(unique_labels):
        try:
            from sklearn.metrics import silhouette_samples, silhouette_score

            subset, subset_labels = matrix[non_noise], labels[non_noise]
            silhouette_overall = float(silhouette_score(subset, subset_labels, metric=metric))
            for position, sample in zip(non_noise, silhouette_samples(subset, subset_labels, metric=metric)):
                per_point[int(position)] = float(sample)
        except ValueError as exc:
            # Degenerate geometry (e.g. all-identical vectors) -- report no score
            # rather than failing the whole job.
            logger.warning("silhouette_skipped reason=%s", exc)

    cluster_stats = []
    for label in unique_labels:
        member_indices = np.flatnonzero(labels == label)
        scores = [per_point[int(i)] for i in member_indices if per_point[int(i)] is not None]
        cluster_stats.append(
            {
                "label": label,
                "size": int(member_indices.size),
                "mean_silhouette": float(np.mean(scores)) if scores else None,
                "medoid_index": _medoid_index(matrix, member_indices),
            }
        )

    return {
        "labels": [int(label) for label in labels],
        "probabilities": [float(p) for p in probabilities] if probabilities is not None else [],
        "n_clusters": len(unique_labels),
        "n_noise": n_noise,
        "silhouette_score": silhouette_overall,
        "silhouette_samples": per_point,
        "cluster_stats": cluster_stats,
        "params": {
            "min_cluster_size": effective_min,
            "metric": metric,
            "pca_dims": pca_dims,
        },
    }
