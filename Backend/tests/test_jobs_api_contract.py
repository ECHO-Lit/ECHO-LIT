"""Function Testing -- the POST /jobs contract and job lifecycle over HTTP.

Test Plan Section 3.1.2.  See tests/plans/3.1.2-function-testing.md.

Covers the submission and lifecycle half of:
  FR-4  "Submitting a batch returns a job_id within the control plane latency
         budget"
  FR-14 "A running job can be canceled, and a completed job's result is
         retrievable until it expires at 24 hours"
plus the operation-level business rules for FR-5 (embedding), FR-6 (saliency),
FR-9 (hidden_states / layer_probe) and the routing rules that push FR-7 and
FR-10 to their own endpoints.

Why this is not duplication of test_layer_probe_schema.py: that module asserts
the same rules with `pytest.raises(ValidationError)` against the pydantic model
directly.  A route that stopped calling the validator would still pass every one
of those tests.  These assert the HTTP status and the message a user actually
receives.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from httpx import AsyncClient

from app.api.routes import jobs as jobs_routes
from app.core.settings import settings
from app.core.storage import get_storage
from app.main import app
from app.repositories.audio import AudioRepository
from app.repositories.jacobian_lenses import JacobianLensRepository
from app.repositories.jobs import JobRepository
from app.schemas.jacobian_lens import JacobianLensRecord, JacobianLensStatus
from app.schemas.jobs import JobProgress, JobStatus

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
    calls: list[dict] = []
    revoked: list[str] = []
    state = SimpleNamespace(raises=None)

    def _send_task(name, args=None, kwargs=None, queue=None, **rest):
        if state.raises is not None:
            raise state.raises
        calls.append({"name": name, "args": args, "queue": queue})
        return SimpleNamespace(id="celery-task-1")

    monkeypatch.setattr(jobs_routes.celery_app, "send_task", _send_task)
    monkeypatch.setattr(
        jobs_routes.celery_app.control, "revoke", lambda tid, **kw: revoked.append(tid)
    )
    return SimpleNamespace(calls=calls, revoked=revoked, state=state)


async def _upload(client, sample_audio_file, name="sample.wav") -> str:
    with sample_audio_file.open("rb") as handle:
        response = await client.post("/upload", files={"file": (name, handle, "audio/wav")})
    assert response.status_code == 201, response.text
    return response.json()["audio_id"]


async def _sid(client) -> str:
    return (await client.get("/session")).json()["sid"]


def _detail(response) -> str:
    """FastAPI validation errors nest the message; route errors do not."""
    body = response.json()["detail"]
    if isinstance(body, list):
        return " | ".join(entry.get("msg", "") for entry in body)
    return body


# --------------------------------------------------------------------------
# Submission contract
# --------------------------------------------------------------------------


class TestSubmissionContract:
    async def test_a_valid_batch_is_accepted_and_immediately_observable(
        self, client, sample_audio_file, fake_broker
    ):
        """FT-78: FR-4 acceptance criterion -- submission returns a job_id."""
        audio_id = await _upload(client, sample_audio_file)

        response = await client.post(
            "/jobs",
            json={"operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id]},
        )

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "queued"
        assert body["status_url"] == f"/jobs/{body['job_id']}"
        assert body["cache_hit"] is False

        # "The UI remains responsive" means the job is pollable at once.
        status = await client.get(f"/jobs/{body['job_id']}")
        assert status.status_code == 200
        assert status.json()["status"] == "queued"
        assert status.json()["result_url"] is None
        assert fake_broker.calls

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (
                {"operation": "prediction", "audio_ids": ["a"]},
                "model must be one of: wav2vec2, whisper-base, whisper-large",
            ),
            (
                {"operation": "attention", "model": "wav2vec2", "audio_ids": ["a"]},
                "wav2vec2 does not support attention",
            ),
            # The capability check runs before the "does not accept a model"
            # check, so a built-in model is refused for the first reason. The
            # second message is only reachable for a model id outside the
            # catalogue, which the route rejects as unknown before validation.
            (
                {"operation": "perturbation", "model": "whisper-base", "audio_ids": ["a"],
                 "parameters": {"perturbations": [{"type": "noise", "params": {}}]}},
                "whisper-base does not support perturbation",
            ),
            (
                {"operation": "audio_features", "model": "whisper-base", "audio_ids": ["a"]},
                "whisper-base does not support audio_features",
            ),
        ],
        ids=["missing_model", "capability", "perturbation_model", "audio_features_model"],
    )
    async def test_model_rules_are_enforced_over_http(self, client, payload, expected):
        """FT-79: FR-4 business rules -- the model/operation pairing is checked."""
        response = await client.post("/jobs", json=payload)

        assert response.status_code == 422
        assert expected in _detail(response)

    @pytest.mark.parametrize(
        ("operation", "count", "expected"),
        [
            ("saliency", 2, "saliency requires exactly one audio_id"),
            ("attention", 2, "attention requires exactly one audio_id"),
            ("layer_probe", 1, "layer_probe requires at least 2 audio_ids"),
        ],
        ids=["saliency", "attention", "layer_probe"],
    )
    async def test_arity_rules_are_enforced_over_http(self, client, operation, count, expected):
        """FT-80: each operation's audio arity is a business rule, not a hint."""
        payload = {
            "operation": operation,
            "model": "whisper-base",
            "audio_ids": [f"a{i}" for i in range(count)],
        }
        if operation == "layer_probe":
            payload["parameters"] = {"properties": {"gender": ["m"]}}

        response = await client.post("/jobs", json=payload)

        assert response.status_code == 422
        assert expected in _detail(response)

    @pytest.mark.parametrize(
        "audio_ids", [[], [f"a{i}" for i in range(1001)]], ids=["empty", "over_max"]
    )
    async def test_the_audio_id_list_is_bounded(self, client, audio_ids):
        """FT-80b: a batch has a documented size range."""
        response = await client.post(
            "/jobs",
            json={"operation": "prediction", "model": "whisper-base", "audio_ids": audio_ids},
        )

        assert response.status_code == 422

    async def test_unknown_parameters_are_rejected_not_ignored(self, client):
        """FT-81a: silently dropping a parameter would produce a plausible wrong answer."""
        response = await client.post(
            "/jobs",
            json={
                "operation": "prediction",
                "model": "whisper-base",
                "audio_ids": ["a"],
                "parameters": {"temperature": 0.5},
            },
        )

        assert response.status_code == 422
        assert "Extra inputs are not permitted" in _detail(response)

    async def test_probe_labels_must_align_with_the_audio_list(self, client):
        """FT-81b: FR-9 -- a positional mismatch silently destroys the signal."""
        response = await client.post(
            "/jobs",
            json={
                "operation": "layer_probe",
                "model": "whisper-base",
                "audio_ids": ["a", "b", "c"],
                "parameters": {"properties": {"gender": ["m", "f"]}},
            },
        )

        assert response.status_code == 422
        assert "property 'gender' has 2 labels for 3 audio_ids" in _detail(response)

    @pytest.mark.parametrize(
        ("operation", "endpoint"),
        [
            ("linguistic_acoustic", "POST /analyses/linguistic-vs-acoustic"),
            ("fairness", "POST /api/v1/analyses/fairness"),
        ],
        ids=["fr7", "fr10"],
    )
    async def test_dedicated_analyses_are_routed_to_their_own_endpoints(
        self, client, operation, endpoint
    ):
        """FT-81c: the generic endpoint names the right one rather than failing late."""
        response = await client.post(
            "/jobs",
            json={"operation": operation, "model": "whisper-base", "audio_ids": ["a"]},
        )

        assert response.status_code == 422
        assert endpoint in _detail(response)

    async def test_parameters_are_normalised_before_dispatch(
        self, client, sample_audio_file, fake_broker
    ):
        """FT-82: defaults are materialised and unset optionals dropped.

        The drop matters beyond tidiness: an explicit null would change the
        content-cache key and turn every request into a miss.
        """
        first = await _upload(client, sample_audio_file, "a.wav")
        second = await _upload(client, sample_audio_file, "b.wav")

        response = await client.post(
            "/jobs",
            json={
                "operation": "hidden_states",
                "model": "whisper-base",
                "audio_ids": [first, second],
                "parameters": {},
            },
        )

        assert response.status_code == 202
        record = await JobRepository().get(response.json()["job_id"])
        assert record.parameters["pooling"] == "mean"
        assert record.parameters["seed"] == 42
        assert "noise_snr_db" not in record.parameters

    @pytest.mark.security
    async def test_an_unowned_audio_id_is_reported_as_not_found(
        self, client, sample_audio_file
    ):
        """FT-83: FR-1 isolation, enforced at submission rather than at execution."""
        missing = await client.post(
            "/jobs",
            json={"operation": "prediction", "model": "whisper-base", "audio_ids": ["ghost"]},
        )
        assert missing.status_code == 404
        assert missing.json()["detail"] == "Audio not found: ghost"

        owner_sid = await _sid(client)
        audio_id = await _upload(client, sample_audio_file)

        async with AsyncClient(app=app, base_url="http://intruder") as intruder:
            assert await _sid(intruder) != owner_sid
            response = await intruder.post(
                "/jobs",
                json={"operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id]},
            )

        assert response.status_code == 404
        assert response.json()["detail"] == f"Audio not found: {audio_id}"

    async def test_a_broker_outage_returns_503_and_records_why(
        self, client, sample_audio_file, fake_broker
    ):
        """FT-84: a job that could not be queued says so, and stays inspectable."""
        audio_id = await _upload(client, sample_audio_file)
        fake_broker.state.raises = RuntimeError("broker down")

        response = await client.post(
            "/jobs",
            json={"operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id]},
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "Job broker is unavailable"

        # The record survives so the owner can see the reason rather than
        # watching a job that never starts.
        records = await JobRepository().list_session_job_ids(await _sid(client))
        assert len(records) == 1
        record = await JobRepository().get(records[0])
        assert record.status == JobStatus.failure
        assert record.error.code == "broker_unavailable"
        assert record.error.retryable is True


# --------------------------------------------------------------------------
# Jacobian lens gating
# --------------------------------------------------------------------------


class TestJacobianLensGating:
    @pytest.mark.important
    async def test_apply_gates_on_lens_existence_readiness_and_model(
        self, client, sample_audio_file
    ):
        """FT-85: applying a lens checks the three things that make it valid."""
        audio_id = await _upload(client, sample_audio_file)
        sid = await _sid(client)

        def _payload(lens_id, model="whisper-base"):
            return {
                "operation": "jacobian_lens_apply",
                "model": model,
                "audio_ids": [audio_id],
                "parameters": {"lens_id": lens_id},
            }

        missing = await client.post("/jobs", json=_payload("jlens-nope"))
        assert missing.status_code == 404
        assert missing.json()["detail"] == "Jacobian lens not found"

        await JacobianLensRepository().create(JacobianLensRecord(
            lens_id="jlens-fitting", session_id=sid, model_id="whisper-base",
            model_revision="r", fit_job_id="j1", status=JacobianLensStatus.FITTING,
            created_at=T0, updated_at=T0, sample_count=2,
        ))
        unready = await client.post("/jobs", json=_payload("jlens-fitting"))
        assert unready.status_code == 409
        assert unready.json()["detail"] == "Jacobian lens is fitting"

        await JacobianLensRepository().create(JacobianLensRecord(
            lens_id="jlens-ready", session_id=sid, model_id="whisper-base",
            model_revision="r", fit_job_id="j2", status=JacobianLensStatus.READY,
            created_at=T0, updated_at=T0, sample_count=2,
        ))
        # whisper-large also declares jacobian_lens_apply, so this reaches the
        # model-identity check rather than being refused for capability first.
        mismatched = await client.post("/jobs", json=_payload("jlens-ready", model="whisper-large"))
        assert mismatched.status_code == 400
        assert mismatched.json()["detail"] == "Jacobian lens belongs to a different model"

    @pytest.mark.important
    async def test_fit_creates_a_lens_and_fails_it_when_the_broker_is_down(
        self, client, sample_audio_file, fake_broker
    ):
        """FT-86: the lens record tracks the fit job's fate rather than dangling."""
        first = await _upload(client, sample_audio_file, "a.wav")
        second = await _upload(client, sample_audio_file, "b.wav")
        fake_broker.state.raises = RuntimeError("broker down")

        response = await client.post(
            "/jobs",
            json={
                "operation": "jacobian_lens_fit",
                "model": "whisper-base",
                "audio_ids": [first, second],
                "parameters": {"samples": [
                    {"audio_id": first, "transcript": "one"},
                    {"audio_id": second, "transcript": "two"},
                ]},
            },
        )

        assert response.status_code == 503
        lenses = await JacobianLensRepository().list_owned(await _sid(client))
        assert len(lenses) == 1
        assert lenses[0].status == JacobianLensStatus.FAILED
        assert lenses[0].error == "Lens-fitting worker is unavailable"


# --------------------------------------------------------------------------
# Lifecycle -- FR-14
# --------------------------------------------------------------------------


class TestJobLifecycle:
    async def test_result_url_appears_only_once_the_job_succeeds(
        self, client, sample_audio_file
    ):
        """FT-87: FR-14 outputs -- a result link that would 409 is not offered."""
        audio_id = await _upload(client, sample_audio_file)
        job_id = (await client.post(
            "/jobs",
            json={"operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id]},
        )).json()["job_id"]

        assert (await client.get(f"/jobs/{job_id}")).json()["result_url"] is None
        conflict = await client.get(f"/jobs/{job_id}/result")
        assert conflict.status_code == 409
        assert conflict.json()["detail"] == "Job is queued"

        get_storage().put_json("results/x/result.json", {"items": []})
        await JobRepository().update(
            job_id, status=JobStatus.success, result_key="results/x/result.json"
        )

        status = await client.get(f"/jobs/{job_id}")
        assert status.json()["result_url"] == f"/jobs/{job_id}/result"
        assert (await client.get(f"/jobs/{job_id}/result")).json() == {"items": []}

    async def test_an_expired_result_object_is_reported_as_gone(
        self, client, sample_audio_file
    ):
        """FT-88: FR-14 acceptance criterion -- retrievable *until it expires*."""
        audio_id = await _upload(client, sample_audio_file)
        job_id = (await client.post(
            "/jobs",
            json={"operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id]},
        )).json()["job_id"]
        # Success recorded, but the payload is no longer in the object store --
        # the state a 24h expiry leaves behind.
        await JobRepository().update(
            job_id, status=JobStatus.success, result_key="results/gone/result.json"
        )

        response = await client.get(f"/jobs/{job_id}/result")

        assert response.status_code == 410
        assert response.json()["detail"] == "Job result has expired"

    async def test_cancelling_a_running_job_revokes_every_child_task(
        self, client, sample_audio_file, fake_broker
    ):
        """FT-90: FR-14 acceptance criterion -- 'a running job can be canceled'.

        Fan-out jobs dispatch child tasks; revoking only the parent would leave
        the shards running and burning GPU.
        """
        audio_id = await _upload(client, sample_audio_file)
        job_id = (await client.post(
            "/jobs",
            json={"operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id]},
        )).json()["job_id"]
        await JobRepository().update(
            job_id,
            status=JobStatus.processing,
            child_task_ids=["child-1", "child-2"],
            progress=JobProgress(current=1, total=2, message="Running"),
        )

        response = await client.delete(f"/jobs/{job_id}")

        assert response.status_code == 202
        assert response.json() == {"job_id": job_id, "status": "cancellation_requested"}
        assert set(fake_broker.revoked) == {"celery-task-1", "child-1", "child-2"}
        assert await JobRepository().cancellation_requested(job_id) is True

    async def test_deleting_a_terminal_job_returns_204_and_clears_its_result(
        self, client, sample_audio_file
    ):
        """FT-89: the DELETE route declares 202 but overrides to 204 when finished."""
        audio_id = await _upload(client, sample_audio_file)
        job_id = (await client.post(
            "/jobs",
            json={"operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id]},
        )).json()["job_id"]
        get_storage().put_json("results/done/result.json", {"items": []})
        await JobRepository().update(
            job_id, status=JobStatus.success, result_key="results/done/result.json"
        )

        response = await client.delete(f"/jobs/{job_id}")

        assert response.status_code == 204
        assert response.content == b""
        assert await JobRepository().get(job_id) is None
        assert get_storage().exists("results/done/result.json") is False

    @pytest.mark.security
    async def test_another_sessions_job_is_invisible(self, client, sample_audio_file):
        """FT-91: FR-14 reads and the cancel route are all session-scoped."""
        owner_sid = await _sid(client)
        audio_id = await _upload(client, sample_audio_file)
        job_id = (await client.post(
            "/jobs",
            json={"operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id]},
        )).json()["job_id"]

        async with AsyncClient(app=app, base_url="http://intruder") as intruder:
            assert await _sid(intruder) != owner_sid
            status = await intruder.get(f"/jobs/{job_id}")
            result = await intruder.get(f"/jobs/{job_id}/result")
            cancelled = await intruder.delete(f"/jobs/{job_id}")

        assert status.status_code == 404
        assert status.json()["detail"] == "Job not found"
        assert result.status_code == 404
        assert cancelled.status_code == 404
        # The owner's job is untouched by the attempt.
        assert (await client.get(f"/jobs/{job_id}")).json()["status"] == "queued"


# --------------------------------------------------------------------------
# FR-10 request validation -- no prior coverage of any kind
# --------------------------------------------------------------------------


class TestFairnessRequestValidation:
    BASE = {
        "dataset": "ravdess",
        "grouping_key": ["emotion"],
        "model": "whisper-base",
    }

    @pytest.mark.parametrize(
        ("override", "expected"),
        [
            ({"grouping_key": ["emotion", "emotion"]},
             "grouping_key contains a duplicate column"),
            ({"include_faithfulness": True, "include_explanations": False},
             "include_faithfulness requires include_explanations"),
            ({"filters": {"emotion": ["happy"]}},
             "'emotion' cannot be both a filter and a grouping column"),
        ],
        ids=["duplicate_group", "faithfulness", "filter_conflict"],
    )
    async def test_the_four_request_validators_are_enforced(self, client, override, expected):
        """FT-92: each cross-field rule reaches the user as a 422 explanation."""
        response = await client.post(
            "/api/v1/analyses/fairness", json={**self.BASE, **override}
        )

        assert response.status_code == 422
        assert expected in _detail(response)

    @pytest.mark.parametrize(
        "override",
        [
            {"grouping_key": []},
            {"grouping_key": ["a", "b", "c"]},
            {"unexpected_field": 1},
            {"metrics": ["not-a-metric"]},
            {"max_items_per_group": 2001},
            {"thresholds": {"alpha": 0.9}},
        ],
        ids=["no_groups", "too_many_groups", "extra_field", "unknown_metric",
             "over_item_cap", "alpha_out_of_range"],
    )
    async def test_fairness_field_bounds_are_enforced(self, client, override):
        """FT-93: the request model is closed, not advisory."""
        response = await client.post(
            "/api/v1/analyses/fairness", json={**self.BASE, **override}
        )

        assert response.status_code == 422

    async def test_the_total_budget_guard_is_unreachable_behind_the_field_cap(self, client):
        """FT-93b: characterises dead validation (OBS-09).

        `max_items_per_group` is capped at 2000 by its own field bound, and the
        budget rule fires at `value * 8 > 20000`, i.e. above 2500.  The guard can
        therefore never trigger.  Asserted rather than removed so that raising
        either bound without revisiting the other is noticed here.
        """
        from app.schemas.fairness import MAX_TOTAL_ITEMS, FairnessRequest

        field_cap = FairnessRequest.model_fields["max_items_per_group"].metadata[-1].le
        assert field_cap * 8 <= MAX_TOTAL_ITEMS

        accepted = await client.post(
            "/api/v1/analyses/fairness", json={**self.BASE, "max_items_per_group": field_cap}
        )
        assert accepted.status_code != 422
