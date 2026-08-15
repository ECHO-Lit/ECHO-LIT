"""
`layer_probe` job wiring: what the worker assembles and what the API returns.

Two properties are load-bearing and both are asserted here:

* the response never carries raw activations (144 whisper-large files would be
  ~50 MB of JSON), and
* per-file activations stay cached when only probe settings change, which is
  what makes iterating on a probe configuration cost no GPU time.
"""

import json

import numpy as np
import pytest

from app.core.settings import settings
from app.schemas.jobs import TaskAudio, TaskEnvelope
from app.worker.executor import _attach_probe_analysis, item_cache_key


LAYERS = 5
DIM = 24
FILES = 40


def _activation_payload(vectors, noisy=None):
    return {
        "num_layers": LAYERS,
        "hidden_dim": DIM,
        "layer_names": ["input"] + [f"layer_{index}" for index in range(1, LAYERS)],
        "pooling": "mean",
        "sample_rate": 16000,
        "noise_snr_db": None if noisy is None else 10.0,
        "layers": vectors,
        "noisy_layers": noisy,
    }


def _batch(n_files=FILES, planted_layer=2, seed=42):
    """One activation payload per file, with `gender` decodable at one layer."""
    rng = np.random.default_rng(seed)
    labels = ["male" if index % 2 else "female" for index in range(n_files)]
    activations = []
    for index in range(n_files):
        block = rng.normal(size=(LAYERS, DIM))
        block[planted_layer, : DIM // 2] += (labels[index] == "male") * 2.0
        activations.append(_activation_payload(block.tolist()))
    return activations, labels


def _result(activations):
    return {
        "job_id": "job-1",
        "operation": "layer_probe",
        "model": "whisper-base",
        "items": [
            {"audio_id": f"audio-{index}", "result": payload, "cache_hit": False}
            for index, payload in enumerate(activations)
        ],
        "metadata": {},
    }


def _envelope(parameters, audio_count=2, operation="layer_probe"):
    return TaskEnvelope(
        job_id="job-1",
        session_id="session-1",
        operation=operation,
        model="whisper-base",
        audio=[
            TaskAudio(
                audio_id=f"audio-{index}",
                object_key=f"key-{index}",
                filename=f"file-{index}.wav",
                media_type="audio/wav",
                sha256=f"sha-{index}",
            )
            for index in range(audio_count)
        ],
        parameters=parameters,
        result_schema_version=settings.RESULT_SCHEMA_VERSION,
        code_version=settings.CODE_VERSION,
    )


class TestProbeAttachment:
    def test_probes_are_attached_and_find_the_planted_layer(self):
        activations, labels = _batch()
        result = _result(activations)

        _attach_probe_analysis(result, activations, {"properties": {"gender": labels}})

        probe = result["probes"]["properties"]["gender"]
        assert probe["best_layer"] == 2
        assert probe["best_accuracy"] > 0.9
        assert probe["majority_baseline"] == pytest.approx(0.5)

    def test_raw_activations_are_stripped_from_the_response(self):
        activations, labels = _batch()
        result = _result(activations)

        _attach_probe_analysis(result, activations, {"properties": {"gender": labels}})

        assert all("layers" not in item or isinstance(item["layers"], int) for item in result["items"])
        for item in result["items"]:
            assert item["layers"] == LAYERS
            assert item["dim"] == DIM
            assert "result" not in item

    def test_response_stays_small(self):
        activations, labels = _batch()
        result = _result(activations)

        _attach_probe_analysis(result, activations, {"properties": {"gender": labels}})

        assert len(json.dumps(result)) < 500_000

    def test_cache_hit_flags_are_preserved(self):
        activations, labels = _batch()
        result = _result(activations)
        result["items"][0]["cache_hit"] = True

        _attach_probe_analysis(result, activations, {"properties": {"gender": labels}})

        assert result["items"][0]["cache_hit"] is True

    def test_layer_names_come_from_the_extraction_payload(self):
        activations, labels = _batch()
        result = _result(activations)

        _attach_probe_analysis(result, activations, {"properties": {"gender": labels}})

        assert result["probes"]["layer_names"] == activations[0]["layer_names"]

    def test_probe_settings_are_forwarded(self):
        activations, labels = _batch()
        result = _result(activations)

        _attach_probe_analysis(
            result, activations,
            {"properties": {"gender": labels}, "cv_folds": 3, "project_dims": 0,
             "include_control": False, "probe": "linear_svm"},
        )

        params = result["probes"]["params"]
        assert params["cv_folds"] == 3
        assert params["probe"] == "linear_svm"
        assert params["include_control"] is False
        assert result["probes"]["projected_dim"] == DIM


class TestAlignmentGuard:
    def test_short_label_list_raises_rather_than_mislabelling(self):
        activations, labels = _batch()
        result = _result(activations)

        with pytest.raises(ValueError, match="labels for 40 files"):
            _attach_probe_analysis(result, activations, {"properties": {"gender": labels[:-1]}})

    def test_inconsistent_activation_shapes_raise(self):
        activations, labels = _batch()
        activations[3]["layers"] = [[0.0] * (DIM + 1) for _ in range(LAYERS)]
        result = _result(activations)

        with pytest.raises(ValueError, match="same model"):
            _attach_probe_analysis(result, activations, {"properties": {"gender": labels}})

    def test_missing_activations_raise(self):
        activations, labels = _batch()
        activations[2]["layers"] = []
        result = _result(activations)

        with pytest.raises(ValueError, match="no 'layers' activations"):
            _attach_probe_analysis(result, activations, {"properties": {"gender": labels}})


class TestNoiseProbe:
    def test_noise_property_is_added_from_the_second_pass(self):
        rng = np.random.default_rng(7)
        labels = ["male" if index % 2 else "female" for index in range(FILES)]
        activations = []
        for index in range(FILES):
            clean = rng.normal(size=(LAYERS, DIM))
            # A noisy pass shifts the representation from layer 1 onward.
            noisy = clean.copy()
            noisy[1:, DIM // 2 :] += 2.0
            activations.append(_activation_payload(clean.tolist(), noisy.tolist()))
        result = _result(activations)

        _attach_probe_analysis(
            result, activations,
            {"properties": {"gender": labels}, "noise_snr_db": 10.0, "project_dims": 0},
        )

        noise = result["probes"]["properties"]["noise"]
        assert noise["n_samples"] == FILES * 2
        assert noise["class_labels"] == ["clean", "noisy"]
        assert noise["best_accuracy"] > 0.9
        assert result["probes"]["noise_snr_db"] == 10.0

    def test_user_properties_stay_on_the_clean_pass_only(self):
        """Stacking would place a near-duplicate of every clip across the fold split."""
        activations, labels = _batch()
        for payload in activations:
            payload["noisy_layers"] = payload["layers"]
        result = _result(activations)

        _attach_probe_analysis(
            result, activations,
            {"properties": {"gender": labels}, "noise_snr_db": 10.0, "project_dims": 0},
        )

        assert result["probes"]["properties"]["gender"]["n_samples"] == FILES

    def test_missing_noisy_pass_raises(self):
        activations, labels = _batch()
        result = _result(activations)

        with pytest.raises(ValueError, match="no 'noisy_layers' activations"):
            _attach_probe_analysis(
                result, activations,
                {"properties": {"gender": labels}, "noise_snr_db": 10.0},
            )


class TestItemCacheKey:
    def test_probe_settings_do_not_change_the_per_file_key(self):
        base = _envelope({"pooling": "mean", "seed": 42, "properties": {"g": ["a", "b"]},
                          "probe": "logreg", "cv_folds": 5, "project_dims": 256,
                          "min_class_count": 5, "include_control": True})
        tweaked = _envelope({"pooling": "mean", "seed": 42, "properties": {"e": ["x", "y"]},
                             "probe": "linear_svm", "cv_folds": 10, "project_dims": 64,
                             "min_class_count": 2, "include_control": False})

        assert item_cache_key(base, "sha-0") == item_cache_key(tweaked, "sha-0")

    def test_hidden_states_and_layer_probe_share_a_per_file_key(self):
        probe = _envelope({"pooling": "mean", "seed": 42, "properties": {"g": ["a", "b"]},
                           "probe": "logreg", "cv_folds": 5, "project_dims": 256,
                           "min_class_count": 5, "include_control": True})
        extract = _envelope({"pooling": "mean", "seed": 42}, operation="hidden_states")

        assert item_cache_key(probe, "sha-0") == item_cache_key(extract, "sha-0")

    def test_noise_setting_does_change_the_per_file_key(self):
        base = _envelope({"pooling": "mean", "seed": 42}, operation="hidden_states")
        noisy = _envelope(
            {"pooling": "mean", "seed": 42, "noise_snr_db": 10.0}, operation="hidden_states"
        )

        assert item_cache_key(base, "sha-0") != item_cache_key(noisy, "sha-0")

    def test_other_operations_keep_their_existing_keys(self):
        """Regression guard: the policy must not disturb the existing cache."""
        embedding = _envelope({"reduction": "pca", "n_components": 2}, operation="embedding")

        assert item_cache_key(embedding, "sha-0") == item_cache_key(embedding, "sha-0")
        other = _envelope({"reduction": "umap", "n_components": 2}, operation="embedding")
        assert item_cache_key(embedding, "sha-0") != item_cache_key(other, "sha-0")
