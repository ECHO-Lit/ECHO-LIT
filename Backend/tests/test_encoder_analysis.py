"""Unit tests for the pure math in encoder_analysis_service.

These cover the label-free structural analyses (attention distance profiles,
linear CKA, participation ratio) without loading any model: everything here
runs on synthetic numpy arrays in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.encoder_analysis_service import (
    DIAGONAL_HALF_WIDTH,
    attention_distance_profiles,
    cka_layer_matrix,
    linear_cka,
    offset_bin_edges,
    participation_ratio,
)


def _identity_attention(n: int) -> np.ndarray:
    matrix = np.eye(n)
    return matrix[None, :, :].repeat(2, axis=0)


def test_offset_bin_edges_are_monotonic_and_cover_range():
    edges = offset_bin_edges(100, 16)
    assert edges[0] == 0
    assert edges[-1] == 100
    assert np.all(np.diff(edges) > 0)
    assert np.issubdtype(edges.dtype, np.integer)


def test_identity_attention_puts_all_mass_on_the_diagonal():
    edges = offset_bin_edges(50, 8)
    result = attention_distance_profiles(_identity_attention(50), edges)
    assert result["head_profiles"].shape == (2, len(edges) - 1)
    assert np.allclose(result["head_profiles"].sum(axis=1), 1.0)
    # Offset 0 is the only populated bin.
    assert result["head_profiles"][0, 0] == pytest.approx(1.0)
    assert result["head_profiles"][0, 1:].sum() == pytest.approx(0.0)
    assert result["diagonal_mass"][0] == pytest.approx(1.0)
    # A delta profile is minimally entropic.
    assert result["profile_entropy"][0] == pytest.approx(0.0)


def test_uniform_attention_is_flat_and_highly_entropic():
    n = 40
    edges = offset_bin_edges(n, 8)
    matrix = np.full((1, n, n), 1.0 / n)
    result = attention_distance_profiles(matrix, edges)
    # Offsets 0..n-1 occur n, n-1, ..., 1 times, so bins are only approximately
    # flat; the strong claims are total mass and high (not maximal) entropy --
    # pair counts per offset decay, so uniform attention is not the uniform
    # distribution over bins.
    assert result["head_profiles"].shape == (1, len(edges) - 1)
    assert np.allclose(result["head_profiles"].sum(axis=1), 1.0)
    assert result["profile_entropy"][0] > 0.8
    # Offsets 0, +-1, +-2 pair counts: n, 2(n-1), 2(n-2).
    expected_diag = (5 * n - 6) / (n * n)
    assert result["diagonal_mass"][0] == pytest.approx(expected_diag, rel=1e-9)


def test_diagonal_mass_matches_direct_computation():
    rng = np.random.default_rng(0)
    raw = rng.random((1, 30, 30))
    matrix = raw / raw.sum(axis=(1, 2), keepdims=True)
    result = attention_distance_profiles(matrix, offset_bin_edges(30, 6))
    offsets = np.abs(np.subtract.outer(np.arange(30), np.arange(30)))
    expected = matrix[0][offsets <= DIAGONAL_HALF_WIDTH].sum()
    assert result["diagonal_mass"][0] == pytest.approx(float(expected))


def test_linear_cka_is_one_for_identical_and_rotated_representations():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((64, 32))
    assert linear_cka(x, x) == pytest.approx(1.0)

    q, _ = np.linalg.qr(rng.standard_normal((32, 32)))
    assert linear_cka(x, x @ q) == pytest.approx(1.0)
    assert linear_cka(x, 3.0 * (x @ q)) == pytest.approx(1.0)


def test_linear_cka_is_low_for_independent_representations():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((128, 32))
    y = rng.standard_normal((128, 32))
    assert linear_cka(x, y) < 0.2


def test_linear_cka_ignores_a_constant_offset():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((64, 16))
    assert linear_cka(x, x + 100.0) == pytest.approx(1.0)


def test_participation_ratio_bounds_and_uniform_case():
    rng = np.random.default_rng(4)
    # Far more samples than dimensions: finite-sample eigenvalue spread
    # (Marchenko-Pastur) depresses the ratio, so the isotropic limit needs
    # N >> H to approach 1.0.
    x = rng.standard_normal((2048, 16))
    ratio = participation_ratio(x)
    assert 0.0 < ratio <= 1.0
    assert ratio == pytest.approx(1.0, abs=0.05)

    # Rank-one data lives in a single direction.
    low = np.outer(np.arange(64, dtype=float), np.ones(16))
    assert participation_ratio(low) == pytest.approx(1.0 / 16, rel=1e-6)


def test_cka_layer_matrix_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(5)
    hidden = [
        rng.standard_normal((1, 48, 16)) for _ in range(4)
    ]
    result = cka_layer_matrix(hidden, max_frames=48)
    matrix = np.asarray(result["matrix"])
    assert matrix.shape == (4, 4)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T)
    assert result["adjacent_cka"] == [
        result["matrix"][i][i + 1] for i in range(3)
    ]
    assert len(result["participation_ratio"]) == 4


def test_cka_layer_matrix_subsampling_does_not_change_similarities():
    """Striding frames is a variance/accuracy trade, not a correctness one:
    two near-identical layers must stay near 1.0 after subsampling."""
    rng = np.random.default_rng(6)
    base = rng.standard_normal((200, 16))
    noisy = base + 0.01 * rng.standard_normal((200, 16))
    hidden = [base[None, :, :], noisy[None, :, :]]

    full = linear_cka(base, noisy)
    subsampled = cka_layer_matrix(hidden, max_frames=50)
    assert subsampled["matrix"][0][1] == pytest.approx(full, abs=0.05)
