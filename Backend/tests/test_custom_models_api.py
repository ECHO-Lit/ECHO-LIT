"""Function Testing -- FR-12 custom model ingestion and compatibility constraints.

Test Plan Section 3.1.2.  See tests/plans/3.1.2-function-testing.md.

SRS FR-12 acceptance criterion: "A compliant seq2seq/CTC/classification model
loads and exposes exactly the analyses in the matrix above.  A non-compliant
model is rejected with a clear explanation and the platform remains stable."
Realized as SAD S4 use case 3 (Ingest a custom Hugging Face model).

Two seams are used deliberately, and the distinction is load-bearing:

* Tests that *consume* a ready model build the record straight through
  CustomModelRepository.  Fast, and no transformers import.
* Tests that verify the *transition itself* (FT-33..FT-37) call
  `validate_custom_model` with a fake `transformers` injected into sys.modules.
  Using the repository factory for those would assert the fixture's own
  capability list rather than the catalogue's -- which is exactly the failure
  mode of the deleted `test_model_loading_and_initialization`.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from types import ModuleType, SimpleNamespace

import pytest
from httpx import AsyncClient

from app.api.routes import models as models_routes
from app.core.model_catalog import ModelKind, custom_model_capabilities
from app.core.settings import settings
from app.core.storage import get_storage
from app.main import app
from app.repositories.models import CustomModelRepository
from app.schemas.models import CustomModelRecord, CustomModelStatus
from app.worker.custom_model_validation import validate_custom_model

pytestmark = pytest.mark.critical

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


@pytest.fixture(autouse=True)
def fake_broker(monkeypatch):
    """Celery has no eager mode here and no broker runs; see test_function_testing."""
    calls: list[dict] = []
    revoked: list[tuple] = []
    state = SimpleNamespace(raises=None)

    def _send_task(name, args=None, kwargs=None, queue=None, **rest):
        if state.raises is not None:
            raise state.raises
        calls.append({"name": name, "args": args, "queue": queue})
        return SimpleNamespace(id="celery-task-1")

    monkeypatch.setattr(models_routes.celery_app, "send_task", _send_task)
    monkeypatch.setattr(
        models_routes.celery_app.control,
        "revoke",
        lambda task_id, **kw: revoked.append((task_id, kw)),
    )
    return SimpleNamespace(calls=calls, revoked=revoked, state=state)


@pytest.fixture
def ready_model():
    """Write a READY custom-model record without going through the worker."""

    async def _make(session_id: str, *, kind=ModelKind.SEQ2SEQ_ASR, model_id="custom-ready", **over):
        fields = {
            "model_id": model_id,
            "session_id": session_id,
            "hf_repo": "openai/whisper-tiny",
            "status": CustomModelStatus.READY,
            "kind": kind,
            "capabilities": custom_model_capabilities(kind),
            "processor_type": "WhisperProcessor",
            "created_at": T0,
            "updated_at": T0,
        }
        fields.update(over)
        record = CustomModelRecord(**fields)
        await CustomModelRepository().create(record)
        return record

    return _make


def _fake_transformers(monkeypatch, *, config, processor=None, config_error=None):
    """Inject a stand-in `transformers` for the worker's function-local import.

    `validate_custom_model` imports AutoConfig/AutoProcessor inside the function
    body, so replacing the sys.modules entry is sufficient and leaks nothing
    into other tests (monkeypatch restores it).
    """
    module = ModuleType("transformers")

    class _AutoConfig:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            if config_error is not None:
                raise config_error
            return config

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            return processor if processor is not None else _CallableProcessor()

    module.AutoConfig = _AutoConfig
    module.AutoProcessor = _AutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", module)


class _CallableProcessor:
    def __call__(self, *args, **kwargs):  # a real processor is callable
        return None


async def _sid(client) -> str:
    return (await client.get("/session")).json()["sid"]


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


class TestModelRegistration:
    async def test_post_models_returns_202_and_dispatches_validation(self, client, fake_broker):
        """FT-24: FR-12 trigger -- registration is accepted and handed to a worker."""
        response = await client.post(
            "/models", json={"hf_repo": "openai/whisper-tiny", "revision": "main"}
        )

        assert response.status_code == 202
        body = response.json()
        assert body["model_id"].startswith("custom-")
        assert len(body["model_id"]) == len("custom-") + 32
        assert body["status"] == "validating"
        assert body["status_url"] == f"/models/{body['model_id']}"

        # The API must not import transformers or run model code itself.
        dispatched = fake_broker.calls[-1]
        assert dispatched["name"] == "app.worker.tasks.validate_custom_model"
        assert dispatched["args"] == [body["model_id"]]
        assert dispatched["queue"] == "cpu"

    async def test_the_record_is_persisted_with_its_task_id(self, client):
        """FT-25: the record is readable immediately and carries the dispatch handle."""
        created = (await client.post("/models", json={"hf_repo": "a/b"})).json()

        fetched = await client.get(f"/models/{created['model_id']}")

        assert fetched.status_code == 200
        body = fetched.json()
        assert body["hf_repo"] == "a/b"
        assert body["revision"] is None
        assert body["status"] == "validating"
        assert body["task_id"] == "celery-task-1"
        assert body["capabilities"] == []
        assert body["kind"] is None

    @pytest.mark.parametrize(
        "hf_repo",
        ["", "no-slash", "/leading", "trailing/", "a b/c", "-bad/repo", "x" * 101 + "/" + "y" * 101],
        ids=["empty", "no_slash", "leading", "trailing", "space", "leading_dash", "too_long"],
    )
    async def test_a_malformed_repo_id_is_rejected(self, client, hf_repo):
        """FT-26: FR-12 inputs -- the repo id is constrained, not free text."""
        response = await client.post("/models", json={"hf_repo": hf_repo})

        assert response.status_code == 422

    @pytest.mark.parametrize(
        "payload",
        [
            {"hf_repo": "a/b", "unexpected": 1},
            {"hf_repo": "a/b", "revision": ""},
            {"hf_repo": "a/b", "revision": "r" * 129},
        ],
        ids=["extra_field", "empty_revision", "long_revision"],
    )
    async def test_unknown_fields_and_bad_revisions_are_rejected(self, client, payload):
        """FT-27: the request model forbids extras rather than ignoring them."""
        response = await client.post("/models", json=payload)

        assert response.status_code == 422

    async def test_a_broker_outage_returns_503_and_marks_the_record_failed(
        self, client, fake_broker
    ):
        """FT-28: a dispatch failure is reported and does not leave a ghost record."""
        fake_broker.state.raises = RuntimeError("broker down")

        response = await client.post("/models", json={"hf_repo": "a/b"})

        assert response.status_code == 503
        assert response.json()["detail"] == "Model-validation worker is unavailable"

        # The record exists so the owner can see why, rather than vanishing.
        records = (await client.get("/models")).json()
        assert len(records) == 1
        assert records[0]["status"] == "failed"
        assert records[0]["error"] == "Model-validation worker is unavailable"


# --------------------------------------------------------------------------
# Lifecycle and ownership
# --------------------------------------------------------------------------


class TestModelLifecycle:
    async def test_get_and_delete_of_an_unknown_model_are_404(self, client):
        """FT-29: absent ids are refused with the documented message."""
        got = await client.get("/models/custom-nope")
        deleted = await client.delete("/models/custom-nope")

        assert got.status_code == 404
        assert got.json()["detail"] == "Custom model not found"
        assert deleted.status_code == 404
        assert deleted.json()["detail"] == "Custom model not found"

    @pytest.mark.security
    async def test_a_foreign_model_is_invisible_and_is_left_intact(self, client, ready_model):
        """FT-30: FR-12 records are session-owned.

        The 'not found' message is deliberately identical to the unknown-id
        case, so the response does not disclose that the id exists.
        """
        owner_sid = await _sid(client)
        await ready_model(owner_sid, model_id="custom-owned")

        async with AsyncClient(app=app, base_url="http://intruder") as intruder:
            intruder_sid = await _sid(intruder)
            assert intruder_sid != owner_sid

            assert (await intruder.get("/models/custom-owned")).status_code == 404
            assert (await intruder.delete("/models/custom-owned")).status_code == 404
            assert (await intruder.get("/models")).json() == []

        assert (await client.get("/models/custom-owned")).status_code == 200

    async def test_the_listing_is_newest_first(self, client, ready_model):
        """FT-31: list_owned reads the index with ZREVRANGE, unlike jobs."""
        sid = await _sid(client)
        for index in range(3):
            await ready_model(
                sid,
                model_id=f"custom-{index}",
                created_at=T0 + timedelta(seconds=index * 10),
                updated_at=T0 + timedelta(seconds=index * 10),
            )

        listing = await client.get("/models")

        assert [record["model_id"] for record in listing.json()] == [
            "custom-2",
            "custom-1",
            "custom-0",
        ]

    async def test_delete_returns_204_and_revokes_a_running_validation(
        self, client, fake_broker
    ):
        """FT-32: deleting mid-validation stops the worker rather than orphaning it."""
        created = (await client.post("/models", json={"hf_repo": "a/b"})).json()

        response = await client.delete(f"/models/{created['model_id']}")

        assert response.status_code == 204
        assert response.content == b""
        assert fake_broker.revoked == [("celery-task-1", {"terminate": False})]
        assert (await client.get("/models")).json() == []

    async def test_deleting_a_finished_model_does_not_revoke(self, client, ready_model, fake_broker):
        """FT-32b: revoke is limited to work that is actually still running."""
        sid = await _sid(client)
        await ready_model(sid, model_id="custom-done", task_id="celery-old")

        assert (await client.delete("/models/custom-done")).status_code == 204
        assert fake_broker.revoked == []

    async def test_capabilities_are_refreshed_on_read(self, client, ready_model):
        """FT-40: a record created before a catalogue change is brought current."""
        sid = await _sid(client)
        await ready_model(sid, model_id="custom-stale", capabilities=["prediction"])

        body = (await client.get("/models/custom-stale")).json()

        expected = custom_model_capabilities(ModelKind.SEQ2SEQ_ASR)
        assert body["capabilities"] == expected
        # The refresh is written back, not just projected into the response.
        stored = await CustomModelRepository().get("custom-stale")
        assert stored.capabilities == expected

    async def test_lens_listing_404s_for_an_unknown_model_and_the_alias_agrees(
        self, client, ready_model
    ):
        """FT-41: both lens routes gate on model visibility identically."""
        unknown = await client.get("/models/jacobian-lenses/custom-nope")
        assert unknown.status_code == 404
        assert unknown.json()["detail"] == "Model not found"

        # A built-in model id is always visible and simply has no lenses yet.
        builtin = await client.get("/models/jacobian-lenses/whisper-base")
        alias = await client.get("/models/whisper-base/jacobian-lenses")
        assert builtin.status_code == 200
        assert builtin.json() == []
        assert alias.json() == builtin.json()


# --------------------------------------------------------------------------
# FR-12 acceptance criterion -- the validation transition itself
# --------------------------------------------------------------------------


class TestModelValidation:
    @pytest.mark.parametrize(
        ("config", "expected_kind"),
        [
            (SimpleNamespace(is_encoder_decoder=True, architectures=["WhisperForConditionalGeneration"]),
             ModelKind.SEQ2SEQ_ASR),
            (SimpleNamespace(is_encoder_decoder=False, architectures=["Wav2Vec2ForCTC"]),
             ModelKind.CTC_ASR),
            (SimpleNamespace(is_encoder_decoder=False, architectures=["HubertForAudioClassification"],
                             id2label={0: "happy", 1: "sad"}),
             ModelKind.AUDIO_CLASSIFICATION),
        ],
        ids=["seq2seq", "ctc", "classification"],
    )
    async def test_a_compliant_model_exposes_exactly_the_matrix_capabilities(
        self, client, monkeypatch, config, expected_kind
    ):
        """FT-33: FR-12 acceptance criterion, first half.

        'Exactly' is the operative word: the assertion compares against the
        catalogue rather than a list restated in the test, so a capability added
        to one and not the other fails here.
        """
        created = (await client.post("/models", json={"hf_repo": "a/b"})).json()
        _fake_transformers(monkeypatch, config=config)

        await validate_custom_model(created["model_id"])

        body = (await client.get(f"/models/{created['model_id']}")).json()
        assert body["status"] == "ready"
        assert body["kind"] == expected_kind.value
        assert body["capabilities"] == custom_model_capabilities(expected_kind)
        assert body["processor_type"] == "_CallableProcessor"
        assert body["error"] is None

    async def test_a_non_compliant_architecture_is_rejected_with_an_explanation(
        self, client, monkeypatch
    ):
        """FT-34: FR-12 acceptance criterion, second half."""
        created = (await client.post("/models", json={"hf_repo": "a/b"})).json()
        _fake_transformers(
            monkeypatch,
            config=SimpleNamespace(is_encoder_decoder=False, architectures=["BertForMaskedLM"]),
        )

        await validate_custom_model(created["model_id"])

        body = (await client.get(f"/models/{created['model_id']}")).json()
        assert body["status"] == "failed"
        assert body["error"] == (
            "Model must be a standard speech Seq2Seq, CTC, or audio-classification architecture"
        )
        assert body["capabilities"] == []
        assert body["kind"] is None

    async def test_other_non_compliance_modes_are_reported_not_raised(self, client, monkeypatch):
        """FT-35: every rejection path produces a bounded, owner-safe message."""
        # A classification model with no label map cannot report predictions.
        created = (await client.post("/models", json={"hf_repo": "a/b"})).json()
        _fake_transformers(
            monkeypatch,
            config=SimpleNamespace(
                is_encoder_decoder=False, architectures=["HubertForAudioClassification"], id2label=None
            ),
        )
        await validate_custom_model(created["model_id"])
        body = (await client.get(f"/models/{created['model_id']}")).json()
        assert body["status"] == "failed"
        assert body["error"] == "Classification models must define id2label"

        # A repository that exposes no usable audio processor.
        created = (await client.post("/models", json={"hf_repo": "c/d"})).json()
        _fake_transformers(
            monkeypatch,
            config=SimpleNamespace(is_encoder_decoder=True, architectures=["X"]),
            processor=object(),
        )
        await validate_custom_model(created["model_id"])
        body = (await client.get(f"/models/{created['model_id']}")).json()
        assert body["status"] == "failed"
        assert body["error"] == "The model repository does not provide an audio processor"

    async def test_an_upstream_failure_is_truncated_rather_than_leaked(self, client, monkeypatch):
        """FT-35b: an arbitrary loader error is bounded before it is stored."""
        created = (await client.post("/models", json={"hf_repo": "a/b"})).json()
        _fake_transformers(
            monkeypatch, config=None, config_error=OSError("x" * 900)
        )

        await validate_custom_model(created["model_id"])

        body = (await client.get(f"/models/{created['model_id']}")).json()
        assert body["status"] == "failed"
        assert len(body["error"]) == 500

    async def test_the_platform_remains_stable_after_a_failed_validation(
        self, client, monkeypatch
    ):
        """FT-36: FR-12 acceptance criterion -- 'the platform remains stable'."""
        created = (await client.post("/models", json={"hf_repo": "a/b"})).json()
        _fake_transformers(
            monkeypatch,
            config=SimpleNamespace(is_encoder_decoder=False, architectures=["BertForMaskedLM"]),
        )
        await validate_custom_model(created["model_id"])

        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/models")).status_code == 200
        assert (await client.post("/models", json={"hf_repo": "e/f"})).status_code == 202
        # And the failed model is refused for work rather than half-accepted.
        rejected = await client.post(
            "/jobs",
            json={
                "operation": "prediction",
                "model": created["model_id"],
                "audio_ids": ["missing"],
            },
        )
        assert rejected.status_code == 409
        assert rejected.json()["detail"] == "Custom model is not ready"

    async def test_validating_a_deleted_model_is_a_no_op(self, client, monkeypatch):
        """FT-37: a race between delete and the worker must not resurrect a record."""
        created = (await client.post("/models", json={"hf_repo": "a/b"})).json()
        await client.delete(f"/models/{created['model_id']}")
        _fake_transformers(
            monkeypatch, config=SimpleNamespace(is_encoder_decoder=True, architectures=["X"])
        )

        await validate_custom_model(created["model_id"])

        assert await CustomModelRepository().get(created["model_id"]) is None


# --------------------------------------------------------------------------
# FR-12 -> FR-4 handoff: how /jobs gates on a custom model
# --------------------------------------------------------------------------


class TestCustomModelJobGating:
    async def test_jobs_rejects_an_unknown_custom_model(self, client):
        """FT-38a: an unregistered custom id is refused before any audio lookup."""
        response = await client.post(
            "/jobs",
            json={"operation": "prediction", "model": "custom-nope", "audio_ids": ["a"]},
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Custom model not found"

    async def test_jobs_rejects_a_model_that_is_still_validating(self, client):
        """FT-38b: work cannot be queued against an unvalidated model."""
        created = (await client.post("/models", json={"hf_repo": "a/b"})).json()

        response = await client.post(
            "/jobs",
            json={"operation": "prediction", "model": created["model_id"], "audio_ids": ["a"]},
        )

        assert response.status_code == 409
        assert response.json()["detail"] == "Custom model is not ready"

    async def test_jobs_rejects_an_operation_the_model_cannot_do(self, client, ready_model):
        """FT-38c: FR-12's capability matrix is enforced at submission."""
        sid = await _sid(client)
        # A CTC model exposes no attention in the matrix.
        await ready_model(sid, model_id="custom-ctc", kind=ModelKind.CTC_ASR)

        response = await client.post(
            "/jobs",
            json={"operation": "attention", "model": "custom-ctc", "audio_ids": ["a"]},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Custom model does not support attention"

    @pytest.mark.security
    async def test_jobs_rejects_another_sessions_custom_model(self, client, ready_model):
        """FT-38d: a model id from another session is 'not found', not 'not ready'."""
        owner_sid = await _sid(client)
        await ready_model(owner_sid, model_id="custom-owned")

        async with AsyncClient(app=app, base_url="http://intruder") as intruder:
            assert await _sid(intruder) != owner_sid
            response = await intruder.post(
                "/jobs",
                json={"operation": "prediction", "model": "custom-owned", "audio_ids": ["a"]},
            )

        assert response.status_code == 404
        assert response.json()["detail"] == "Custom model not found"
