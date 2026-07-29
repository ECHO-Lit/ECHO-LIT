"""
Embedding clustering service tests.
Covers HDBSCAN grouping, silhouette scoring, and the degenerate cases the UI relies on
never raising.
"""

import numpy as np
import pytest

from app.services.clustering_service import NOISE_LABEL, cluster_embeddings


def _blobs(n_clusters: int = 3, per_cluster: int = 20, dim: int = 8, spread: float = 0.05):
    """Well-separated Gaussian blobs, one far-apart centre per cluster."""
    rng = np.random.default_rng(42)
    centres = np.eye(n_clusters, dim) * 10.0
    points = [
        centre + rng.normal(scale=spread, size=(per_cluster, dim)) for centre in centres
    ]
    return np.vstack(points).tolist()


class TestClusterEmbeddings:
    """Happy paths: separable data should produce clean, well-scored clusters."""

    def test_separated_blobs_are_recovered(self):
        result = cluster_embeddings(_blobs(n_clusters=3, per_cluster=20), min_cluster_size=5)

        assert result["n_clusters"] == 3
        assert result["silhouette_score"] > 0.5
        assert len(result["labels"]) == 60
        assert len(result["cluster_stats"]) == 3
        assert sum(stat["size"] for stat in result["cluster_stats"]) + result["n_noise"] == 60

    def test_labels_align_with_input_order(self):
        vectors = _blobs(n_clusters=2, per_cluster=15)
        result = cluster_embeddings(vectors, min_cluster_size=5)

        assert len(result["labels"]) == len(vectors)
        assert len(result["silhouette_samples"]) == len(vectors)
        # The first and last points come from different blobs.
        assert result["labels"][0] != result["labels"][-1]

    def test_medoid_index_points_into_its_own_cluster(self):
        result = cluster_embeddings(_blobs(n_clusters=3, per_cluster=20), min_cluster_size=5)

        for stat in result["cluster_stats"]:
            assert result["labels"][stat["medoid_index"]] == stat["label"]

    def test_high_dimensional_input_is_pca_reduced(self):
        result = cluster_embeddings(_blobs(n_clusters=3, per_cluster=20, dim=512), min_cluster_size=5)

        assert result["params"]["pca_dims"] == 50
        assert result["n_clusters"] == 3

    def test_low_dimensional_input_skips_pca(self):
        result = cluster_embeddings(_blobs(dim=8), min_cluster_size=5)

        assert result["params"]["pca_dims"] is None


class TestDegenerateInputs:
    """The frontend types against one shape -- these must never raise or return partials."""

    EXPECTED_KEYS = {
        "labels",
        "probabilities",
        "n_clusters",
        "n_noise",
        "silhouette_score",
        "silhouette_samples",
        "cluster_stats",
        "params",
    }

    @pytest.mark.parametrize("n_points", [0, 1, 2])
    def test_too_few_points_returns_empty_payload(self, n_points):
        result = cluster_embeddings([[1.0, 2.0, 3.0]] * n_points, min_cluster_size=5)

        assert self.EXPECTED_KEYS <= set(result)
        assert result["n_clusters"] == 0
        assert result["silhouette_score"] is None
        assert result["labels"] == [NOISE_LABEL] * n_points
        assert result["cluster_stats"] == []

    def test_identical_vectors_do_not_crash(self):
        result = cluster_embeddings([[1.0, 2.0, 3.0, 4.0]] * 20, min_cluster_size=5)

        assert self.EXPECTED_KEYS <= set(result)
        # Either one cluster or all noise -- both are valid, neither may raise.
        assert result["n_clusters"] <= 1
        assert result["silhouette_score"] is None

    def test_zero_vectors_survive_normalisation(self):
        """Zero-norm rows would divide by zero if not guarded."""
        result = cluster_embeddings([[0.0, 0.0, 0.0]] * 10, min_cluster_size=3)

        assert self.EXPECTED_KEYS <= set(result)
        assert not any(np.isnan(value) for value in result["labels"])

    def test_min_cluster_size_larger_than_dataset_is_clamped(self):
        result = cluster_embeddings(_blobs(n_clusters=2, per_cluster=5), min_cluster_size=50)

        assert result["params"]["min_cluster_size"] == 10
        assert self.EXPECTED_KEYS <= set(result)

    def test_single_cluster_yields_no_silhouette(self):
        """Silhouette is undefined below two clusters -- must be None, not 0."""
        result = cluster_embeddings(_blobs(n_clusters=1, per_cluster=20), min_cluster_size=5)

        assert result["n_clusters"] <= 1
        assert result["silhouette_score"] is None
