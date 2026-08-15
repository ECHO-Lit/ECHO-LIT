"""
Request validation for the `hidden_states` and `layer_probe` operations.

The label payload rides in `parameters` and is joined to files *positionally*.
Most of what is asserted here is therefore about rejecting a malformed join at
the door: a length mismatch produces no error anywhere downstream, just a flat
chart that looks like a real (negative) finding.
"""

import pytest
from pydantic import ValidationError

from app.core.model_catalog import CUSTOM_MODEL_CAPABILITIES, MODEL_DEFINITIONS
from app.schemas.jobs import (
    MAX_PROBE_LABEL_LENGTH,
    MAX_PROBE_PROPERTIES,
    JobCreateRequest,
    JobOperation,
)


def _request(audio_ids, parameters, operation="layer_probe", model="whisper-base"):
    return JobCreateRequest(
        operation=operation, model=model, audio_ids=audio_ids, parameters=parameters
    )


def _labels(count, name="gender"):
    return {name: ["male" if index % 2 else "female" for index in range(count)]}


class TestCapabilities:
    def test_every_builtin_model_declares_both_operations(self):
        for model_id, definition in MODEL_DEFINITIONS.items():
            assert definition.supports("hidden_states"), model_id
            assert definition.supports("layer_probe"), model_id

    def test_every_custom_model_kind_declares_both_operations(self):
        for kind, capabilities in CUSTOM_MODEL_CAPABILITIES.items():
            assert "hidden_states" in capabilities, kind
            assert "layer_probe" in capabilities, kind

    def test_both_operations_require_a_model(self):
        with pytest.raises(ValidationError):
            JobCreateRequest(
                operation=JobOperation.layer_probe,
                audio_ids=["a", "b"],
                parameters={"properties": _labels(2)},
            )
        with pytest.raises(ValidationError):
            JobCreateRequest(operation=JobOperation.hidden_states, audio_ids=["a"], parameters={})


class TestHiddenStatesParameters:
    def test_defaults_are_applied(self):
        request = _request(["a"], {}, operation="hidden_states")

        assert request.parameters == {"pooling": "mean", "seed": 42}

    def test_a_single_file_is_allowed(self):
        assert len(_request(["a"], {}, operation="hidden_states").audio_ids) == 1

    def test_frame_pooling_is_rejected_not_downgraded(self):
        """Frame pooling belongs to the deferred phonetic phase."""
        with pytest.raises(ValidationError):
            _request(["a"], {"pooling": "frames"}, operation="hidden_states")

    @pytest.mark.parametrize("snr", [-11, 41])
    def test_snr_is_bounded(self, snr):
        with pytest.raises(ValidationError):
            _request(["a"], {"noise_snr_db": snr}, operation="hidden_states")

    def test_unset_snr_is_dropped_so_it_cannot_perturb_the_cache_key(self):
        assert "noise_snr_db" not in _request(["a"], {}, operation="hidden_states").parameters

    def test_unknown_parameters_are_rejected(self):
        with pytest.raises(ValidationError):
            _request(["a"], {"nonsense": 1}, operation="hidden_states")


class TestLayerProbeAlignment:
    def test_matching_label_count_is_accepted(self):
        request = _request(["a", "b", "c"], {"properties": _labels(3)})

        assert request.parameters["properties"]["gender"] == ["female", "male", "female"]

    @pytest.mark.parametrize("count", [2, 4])
    def test_label_count_must_equal_audio_count(self, count):
        with pytest.raises(ValidationError, match="labels for 3 audio_ids"):
            _request(["a", "b", "c"], {"properties": _labels(count)})

    def test_every_property_is_checked_not_just_the_first(self):
        parameters = {"properties": {**_labels(3), "emotion": ["happy", "sad"]}}

        with pytest.raises(ValidationError, match="emotion"):
            _request(["a", "b", "c"], parameters)

    def test_none_labels_survive_serialisation(self):
        """`None` marks "not annotated"; dropping it would shift every later label."""
        request = _request(
            ["a", "b", "c"], {"properties": {"accent": ["us", None, "england"]}}
        )

        assert request.parameters["properties"]["accent"] == ["us", None, "england"]


class TestLayerProbeLimits:
    def test_at_least_two_files_are_required(self):
        with pytest.raises(ValidationError, match="at least 2 audio_ids"):
            _request(["a"], {"properties": _labels(1)})

    def test_at_least_one_property_is_required(self):
        with pytest.raises(ValidationError, match="at least one property"):
            _request(["a", "b"], {"properties": {}})

    def test_property_count_is_capped(self):
        parameters = {
            "properties": {f"p{index}": ["x", "y"] for index in range(MAX_PROBE_PROPERTIES + 1)}
        }

        with pytest.raises(ValidationError, match="at most 8 properties"):
            _request(["a", "b"], parameters)

    def test_label_length_is_capped(self):
        parameters = {"properties": {"x": ["a", "b" * (MAX_PROBE_LABEL_LENGTH + 1)]}}

        with pytest.raises(ValidationError, match="longer than"):
            _request(["a", "b"], parameters)

    @pytest.mark.parametrize("folds", [1, 11])
    def test_cv_folds_are_bounded(self, folds):
        with pytest.raises(ValidationError):
            _request(["a", "b"], {"properties": _labels(2), "cv_folds": folds})

    @pytest.mark.parametrize("dims", [16, 2048])
    def test_project_dims_reject_the_useless_range(self, dims):
        with pytest.raises(ValidationError):
            _request(["a", "b"], {"properties": _labels(2), "project_dims": dims})

    @pytest.mark.parametrize("dims", [0, 32, 256, 1024])
    def test_project_dims_accept_zero_and_the_useful_range(self, dims):
        request = _request(["a", "b"], {"properties": _labels(2), "project_dims": dims})

        assert request.parameters["project_dims"] == dims

    def test_unknown_probe_type_is_rejected(self):
        with pytest.raises(ValidationError):
            _request(["a", "b"], {"properties": _labels(2), "probe": "magic"})


class TestCacheIdentity:
    """`layer_probe` must not invalidate cached activations when probes change."""

    def test_probe_settings_are_excluded_from_the_extraction_identity(self):
        from app.worker.cache_policy import item_cache_identity

        base = _request(["a", "b"], {"properties": _labels(2)}).parameters
        tweaked = _request(
            ["a", "b"],
            {"properties": {"emotion": ["happy", "sad"]}, "cv_folds": 10,
             "probe": "linear_svm", "project_dims": 64, "min_class_count": 3,
             "include_control": False},
        ).parameters

        assert item_cache_identity("layer_probe", base) == item_cache_identity("layer_probe", tweaked)

    def test_extraction_settings_do_change_the_identity(self):
        from app.worker.cache_policy import item_cache_identity

        base = _request(["a", "b"], {"properties": _labels(2)}).parameters
        noisy = _request(["a", "b"], {"properties": _labels(2), "noise_snr_db": 10}).parameters

        assert item_cache_identity("layer_probe", base) != item_cache_identity("layer_probe", noisy)

    def test_layer_probe_and_hidden_states_share_one_identity(self):
        """Running either operation must warm the cache for the other."""
        from app.worker.cache_policy import item_cache_identity

        probe = _request(["a", "b"], {"properties": _labels(2)}).parameters
        extract = _request(["a"], {}, operation="hidden_states").parameters

        assert item_cache_identity("layer_probe", probe) == item_cache_identity(
            "hidden_states", extract
        )

    def test_other_operations_are_left_alone(self):
        from app.worker.cache_policy import item_cache_identity

        parameters = {"reduction": "pca", "n_components": 2}

        assert item_cache_identity("embedding", parameters) == ("embedding", parameters)
