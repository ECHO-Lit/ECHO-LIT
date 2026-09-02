"""
Request validation, capability and routing for the `saliency_faithfulness` job.

The metric itself is covered in `test_faithfulness_service.py`. What is asserted
here is the plumbing around it: that the operation is offered by every model that
offers saliency (a faithfulness test that cannot reach a model is useless), that
it is confined to one clip at a time, and that it is not routed onto the fast
queue -- it is a saliency run plus dozens of extra forward passes.
"""

import pytest
from pydantic import ValidationError

from app.core.celery_app import queue_for
from app.core.model_catalog import CUSTOM_MODEL_CAPABILITIES, MODEL_DEFINITIONS
from app.schemas.jobs import (
    JobCreateRequest,
    JobOperation,
    SaliencyFaithfulnessParameters,
)

AUDIO = "a" * 32
OTHER = "b" * 32


def _request(audio_ids=None, parameters=None, model="wav2vec2"):
    return JobCreateRequest(
        operation="saliency_faithfulness",
        model=model,
        audio_ids=audio_ids or [AUDIO],
        parameters=parameters if parameters is not None else {},
    )


class TestCapabilities:
    def test_every_model_that_offers_saliency_offers_the_test(self):
        """A map you cannot check is the thing this feature exists to prevent."""
        for model_id, definition in MODEL_DEFINITIONS.items():
            if definition.supports("saliency"):
                assert definition.supports("saliency_faithfulness"), model_id

    def test_every_custom_model_kind_offers_the_test(self):
        for kind, capabilities in CUSTOM_MODEL_CAPABILITIES.items():
            if "saliency" in capabilities:
                assert "saliency_faithfulness" in capabilities, kind

    def test_requires_a_model(self):
        with pytest.raises(ValidationError):
            JobCreateRequest(
                operation="saliency_faithfulness", audio_ids=[AUDIO], parameters={}
            )

    def test_rejects_a_model_that_does_not_support_it(self, monkeypatch):
        definition = MODEL_DEFINITIONS["wav2vec2"]
        monkeypatch.setitem(
            MODEL_DEFINITIONS,
            "wav2vec2",
            type(definition)(
                definition.model_id,
                definition.revision,
                definition.kind,
                definition.capabilities - {"saliency_faithfulness"},
            ),
        )
        with pytest.raises(ValidationError):
            _request()


class TestAudioSelection:
    def test_accepts_exactly_one_clip(self):
        assert _request().audio_ids == [AUDIO]

    def test_rejects_more_than_one_clip(self):
        """Faithfulness is per-clip; a batch would silently average away a verdict."""
        with pytest.raises(ValidationError, match="exactly one audio_id"):
            _request(audio_ids=[AUDIO, OTHER])


class TestParameters:
    def test_defaults_are_complete(self):
        parameters = _request().parameters
        assert parameters == {
            "method": "gradcam",
            "n_steps": 9,
            "top_fraction": 0.2,
            "random_repeats": 3,
            "seed": 42,
            "include_occlusion": True,
        }

    def test_method_is_inherited_from_saliency(self):
        """The verdict belongs to one attribution method, so the method travels with it."""
        assert _request(parameters={"method": "shap"}).parameters["method"] == "shap"
        with pytest.raises(ValidationError):
            _request(parameters={"method": "occlusion"})

    def test_unknown_parameters_are_rejected(self):
        with pytest.raises(ValidationError):
            _request(parameters={"frame_rate": 25})

    @pytest.mark.parametrize(
        "parameters",
        [
            {"n_steps": 2},          # below the floor: a curve needs points
            {"n_steps": 21},         # above the ceiling: cost runs away
            {"top_fraction": 0},     # removing nothing measures nothing
            {"top_fraction": 1},     # removing everything measures nothing
            {"random_repeats": 0},   # the baseline is not optional
            {"random_repeats": 1},   # one draw gives no error bar to judge against
            {"random_repeats": 11},
            {"seed": -1},
        ],
    )
    def test_out_of_range_parameters_are_rejected(self, parameters):
        with pytest.raises(ValidationError):
            _request(parameters=parameters)

    def test_cost_stays_bounded_at_the_extremes(self):
        """The schema's ceilings must keep the worst case affordable on whisper-large."""
        limits = SaliencyFaithfulnessParameters(n_steps=20, random_repeats=10)
        passes = (3 + limits.random_repeats) * limits.n_steps
        assert passes <= 300


class TestRouting:
    def test_does_not_land_on_the_fast_queue(self):
        for model in MODEL_DEFINITIONS:
            assert queue_for("saliency_faithfulness", model) == "gpu-large"

    def test_shares_the_saliency_queue(self):
        assert queue_for("saliency_faithfulness", "wav2vec2") == queue_for("saliency", "wav2vec2")


def test_operation_is_registered():
    assert JobOperation.saliency_faithfulness.value == "saliency_faithfulness"
