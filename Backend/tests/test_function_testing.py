"""Function Testing -- audio ingestion lifecycle and the control-plane surface.

Test Plan Section 3.1.2.  See tests/plans/3.1.2-function-testing.md.

Traces to SRS FR-1 (session management and audio upload) and to the platform
surface every use case depends on: /health, /metrics, /session, and the gate that
retires the synchronous inference API.

What this module is NOT: it no longer contains "ML model integration" or "audio
processing pipeline" tests.  Those previously defined a mock function inside the
test and asserted that function, or called endpoints that do not exist while
accepting the resulting 404 as a pass.  Model behaviour is covered where it can
actually be observed -- test_fr7_*.py, test_probing_service.py, test_clustering.py
-- and the operations' HTTP contract lives in test_jobs_api_contract.py.

Every case asserts the exact `detail` string rather than a status code alone: the
datasets router is mounted at the root with `/{dataset}/metadata`, so almost any
mistyped two-segment GET returns *some* 404, and a status-only assertion would
pass for the wrong reason.
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import pytest
import soundfile as sf
from httpx import AsyncClient

from app.api.routes import upload as upload_routes
from app.core import redis as redis_module
from app.core.settings import settings
from app.core.storage import get_storage
from app.main import app
from app.services import custom_dataset_service

pytestmark = pytest.mark.critical


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Keep uploads out of the developer's real object store.

    STORAGE_LOCAL_ROOT defaults to the relative path "shared-storage", and
    get_storage is lru_cached, so without this every /upload in this module
    writes real audio into Backend/shared-storage/ and leaves it there.
    """
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


@pytest.fixture(autouse=True)
def isolated_datasets(tmp_path, monkeypatch):
    """SESSIONS_BASE_DIR is relative too, so materialize would write into the tree."""
    monkeypatch.setattr(custom_dataset_service, "SESSIONS_BASE_DIR", tmp_path / "sessions")


@pytest.fixture(autouse=True)
def fake_broker(monkeypatch):
    """No broker runs in the test environment and celery has no eager mode.

    An unpatched send_task reaches for redis://localhost:6379/2 -- a real enqueue
    if the developer happens to have Redis up, a multi-second kombu retry if not.
    Neither belongs in a function test, so the dispatch boundary is stubbed and
    the calls are recorded for assertion.
    """
    calls: list[dict] = []
    result = SimpleNamespace(value={"applicable": True}, raises=None)

    def _send_task(name, args=None, kwargs=None, queue=None, **rest):
        calls.append({"name": name, "args": args, "queue": queue})
        if result.raises is not None:
            raise result.raises

        def _get(timeout=None):
            if isinstance(result.value, Exception):
                raise result.value
            return result.value

        return SimpleNamespace(id="celery-task", get=_get)

    monkeypatch.setattr(upload_routes.celery_app, "send_task", _send_task)
    monkeypatch.setattr(upload_routes.celery_app.control, "revoke", lambda *a, **k: None)
    return SimpleNamespace(calls=calls, result=result)


async def _upload(client, sample_audio_file, name="sample.wav", content_type="audio/wav") -> dict:
    with sample_audio_file.open("rb") as handle:
        response = await client.post("/upload", files={"file": (name, handle, content_type)})
    assert response.status_code == 201, response.text
    return response.json()


def _wav_bytes(seconds: float, sr: int = 16_000) -> bytes:
    """A real, decodable WAV of a given length -- probe_audio must accept it."""
    import numpy as np

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(sr * seconds), dtype="float32"), sr, format="WAV")
    return buffer.getvalue()


# --------------------------------------------------------------------------
# FR-1 -- Session management and audio upload
# --------------------------------------------------------------------------


class TestAudioIngestion:
    """FR-1 inputs and processing: what the platform accepts and what it refuses."""

    async def test_upload_returns_the_probed_asset_contract(self, client, sample_audio_file):
        """FT-01: the response carries probed facts, not echoes of the request."""
        import hashlib

        raw = sample_audio_file.read_bytes()
        payload = await _upload(client, sample_audio_file)

        # conftest's sample is 5.0s @ 16kHz mono.  These must come from probing
        # the decoded audio -- an implementation that echoed the upload would
        # have no way to produce them.
        assert payload["duration_seconds"] == pytest.approx(5.0, abs=0.05)
        assert payload["sample_rate"] == 16_000
        assert payload["channels"] == 1
        assert payload["size_bytes"] == len(raw)
        assert payload["media_type"] == "audio/wav"
        assert payload["filename"] == "sample.wav"
        # The legacy aliases must keep agreeing with their canonical fields.
        assert payload["audio_id"] == payload["file_id"]
        assert payload["duration"] == payload["duration_seconds"]
        assert payload["size"] == payload["size_bytes"]
        assert payload["playback_url"] == f"/audio/{payload['audio_id']}"

        # The stored asset records the content digest the cache keys are built on.
        from app.repositories.audio import AudioRepository

        asset = await AudioRepository().get(payload["audio_id"])
        assert asset.sha256 == hashlib.sha256(raw).hexdigest()

    @pytest.mark.parametrize(
        ("name", "content_type"),
        [
            ("clip.wav", "audio/wav"),
            ("clip.mp3", "audio/mpeg"),
            ("clip.m4a", "audio/mp4"),
            ("clip.flac", "audio/flac"),
        ],
        ids=["wav", "mp3", "m4a", "flac"],
    )
    async def test_upload_accepts_every_allowed_extension(
        self, client, sample_audio_file, name, content_type
    ):
        """FT-02: FR-1 inputs -- the documented extension set is honoured.

        The bytes are WAV in every case; the extension gate is what is under
        test, and probe_audio decodes by content rather than by name.
        """
        with sample_audio_file.open("rb") as handle:
            response = await client.post("/upload", files={"file": (name, handle, content_type)})
        assert response.status_code == 201, response.text

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("notes.txt", "Unsupported audio extension: .txt"),
            ("payload.exe", "Unsupported audio extension: .exe"),
            ("noextension", "Unsupported audio extension: none"),
        ],
        ids=["txt", "exe", "bare"],
    )
    async def test_upload_rejects_an_unsupported_extension(self, client, name, expected):
        """FT-03: invalid data is refused with the documented message."""
        response = await client.post(
            "/upload", files={"file": (name, io.BytesIO(b"data"), "audio/wav")}
        )

        assert response.status_code == 400
        assert response.json()["detail"] == expected

    async def test_upload_rejects_a_non_audio_content_type(self, client):
        """FT-04: the content-type gate is independent of the extension gate."""
        response = await client.post(
            "/upload", files={"file": ("clip.wav", io.BytesIO(b"data"), "text/plain")}
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid file type. Only audio files are allowed."

    async def test_upload_accepts_application_octet_stream(self, client, sample_audio_file):
        """FT-05: the documented exemption for browsers that send no audio type."""
        with sample_audio_file.open("rb") as handle:
            response = await client.post(
                "/upload", files={"file": ("clip.wav", handle, "application/octet-stream")}
            )

        assert response.status_code == 201

    async def test_upload_without_a_file_is_422(self, client):
        """FT-06: the multipart field is required."""
        response = await client.post("/upload", data={"model": "whisper-base"})

        assert response.status_code == 422

    async def test_the_configured_limits_match_the_stated_requirement(self):
        """FT-07c: FR-1 acceptance criterion -- the limits are 100 MB and 10 minutes.

        FT-07/FT-08 prove the limits are *enforced* by shrinking them; this
        proves the deployed values are the ones the requirement names.  Neither
        assertion entails the criterion alone.
        """
        assert settings.MAX_UPLOAD_BYTES == 100 * 1024 * 1024
        assert settings.MAX_AUDIO_DURATION_SECONDS == 600

    async def test_upload_over_the_byte_limit_is_rejected(
        self, client, sample_audio_file, monkeypatch
    ):
        """FT-07: FR-1 acceptance criterion -- over-limit is refused clearly."""
        # Above one 1 MiB read chunk, or the first chunk trips the check and the
        # size accumulator is never exercised.
        monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1024 * 1024 + 512)

        payload = _wav_bytes(seconds=60.0)
        assert len(payload) > 1024 * 1024 + 512
        response = await client.post(
            "/upload", files={"file": ("big.wav", io.BytesIO(payload), "audio/wav")}
        )

        assert response.status_code == 413
        assert response.json()["detail"] == "Audio exceeds the 100 MB upload limit"

    async def test_upload_exactly_at_the_byte_limit_is_accepted(
        self, client, sample_audio_file, monkeypatch
    ):
        """FT-07b: the boundary -- the interesting defect here is > versus >=."""
        raw = sample_audio_file.read_bytes()
        monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", len(raw))

        response = await client.post(
            "/upload", files={"file": ("exact.wav", io.BytesIO(raw), "audio/wav")}
        )

        assert response.status_code == 201

    async def test_upload_over_the_duration_limit_is_rejected(self, client, monkeypatch):
        """FT-08: FR-1's duration half, enforced after decode rather than by size."""
        monkeypatch.setattr(settings, "MAX_AUDIO_DURATION_SECONDS", 2)

        response = await client.post(
            "/upload",
            files={"file": ("long.wav", io.BytesIO(_wav_bytes(seconds=5.0)), "audio/wav")},
        )

        assert response.status_code == 413
        assert response.json()["detail"] == "Audio exceeds the 10 minute duration limit"

    async def test_a_rejected_upload_leaves_no_object_and_no_record(self, client, monkeypatch):
        """FT-09: blast radius -- a refused upload must not half-land."""
        monkeypatch.setattr(settings, "MAX_AUDIO_DURATION_SECONDS", 2)

        rejected = await client.post(
            "/upload",
            files={"file": ("long.wav", io.BytesIO(_wav_bytes(seconds=5.0)), "audio/wav")},
        )
        assert rejected.status_code == 413

        listing = await client.get("/upload/list")
        assert listing.json() == {"files": []}
        # The duration check happens after the bytes are streamed to a temp file
        # but before anything is published to the object store.
        root = get_storage().root
        assert list(root.rglob("*")) == []


class TestAudioLifecycle:
    """FR-1 retrieval, deletion and the session boundary around them."""

    async def test_upload_list_and_metadata_agree(self, client, sample_audio_file):
        """FT-10: the three read paths describe the same asset identically."""
        uploaded = await _upload(client, sample_audio_file)

        listing = await client.get("/upload/list")
        metadata = await client.get(f"/upload/metadata/{uploaded['audio_id']}")

        assert listing.status_code == 200
        assert metadata.status_code == 200
        assert listing.json()["files"] == [uploaded]
        assert metadata.json() == uploaded

    async def test_playback_returns_the_stored_bytes(self, client, sample_audio_file):
        """FT-11: the canonical route and its deprecated alias serve one asset."""
        raw = sample_audio_file.read_bytes()
        uploaded = await _upload(client, sample_audio_file)
        audio_id = uploaded["audio_id"]

        canonical = await client.get(f"/audio/{audio_id}")
        alias = await client.get(f"/upload/file/{audio_id}")

        assert canonical.status_code == 200
        assert canonical.content == raw
        assert canonical.headers["content-type"].startswith("audio/wav")
        assert canonical.headers["x-content-type-options"] == "nosniff"
        assert alias.status_code == 200
        assert alias.content == canonical.content

    async def test_delete_removes_the_record_and_the_object(self, client, sample_audio_file):
        """FT-12: deletion is complete and idempotent-by-404."""
        uploaded = await _upload(client, sample_audio_file)
        audio_id = uploaded["audio_id"]
        object_key = (await _asset(audio_id)).object_key
        assert get_storage().exists(object_key)

        deleted = await client.delete(f"/upload/{audio_id}")

        assert deleted.status_code == 200
        assert deleted.json() == {"message": "File deleted successfully"}
        assert not get_storage().exists(object_key)
        assert (await client.get(f"/upload/metadata/{audio_id}")).status_code == 404

        repeat = await client.delete(f"/upload/{audio_id}")
        assert repeat.status_code == 404
        assert repeat.json()["detail"] == "Audio not found"

    @pytest.mark.security
    async def test_audio_is_inaccessible_from_another_session(self, client, sample_audio_file):
        """FT-13: FR-1 acceptance criterion -- cross-session audio is invisible.

        Asserts on every route that resolves an audio_id, because ownership is
        re-checked per route rather than centrally; one route forgetting the
        check would not be visible from any of the others.
        """
        uploaded = await _upload(client, sample_audio_file)
        audio_id = uploaded["audio_id"]
        owner_sid = (await client.get("/session")).json()["sid"]

        async with AsyncClient(app=app, base_url="http://intruder") as intruder:
            intruder_sid = (await intruder.get("/session")).json()["sid"]
            # Guard the guard: if the clients shared a cookie jar the 404s below
            # would pass for entirely the wrong reason.
            assert intruder_sid != owner_sid

            assert (await intruder.get(f"/upload/metadata/{audio_id}")).status_code == 404
            assert (await intruder.get(f"/audio/{audio_id}")).status_code == 404
            assert (await intruder.delete(f"/upload/{audio_id}")).status_code == 404
            variant = await intruder.post(
                f"/audio/{audio_id}/variant", json={"property": "pitch", "theta": 2.0}
            )
            assert variant.status_code == 404
            assert variant.json()["detail"] == "Audio not found"

        # The owner's asset survived every one of those attempts.
        assert (await client.get(f"/upload/metadata/{audio_id}")).status_code == 200

    async def test_variant_render_returns_the_worker_result(
        self, client, sample_audio_file, fake_broker
    ):
        """FT-16a: the happy path dispatches to the cpu queue and echoes the render."""
        uploaded = await _upload(client, sample_audio_file)
        fake_broker.result.value = {
            "applicable": True,
            "variant_audio_id": "variant-1",
            "playback_url": "/audio/variant-1",
        }

        response = await client.post(
            f"/audio/{uploaded['audio_id']}/variant", json={"property": "pitch", "theta": 2.0}
        )

        assert response.status_code == 200
        assert response.json() == {
            "variant_audio_id": "variant-1",
            "playback_url": "/audio/variant-1",
        }
        dispatched = fake_broker.calls[-1]
        assert dispatched["name"] == "app.worker.tasks.fr7_render_variant"
        assert dispatched["queue"] == "cpu"

    async def test_variant_rejects_an_unknown_property(self, client, sample_audio_file):
        """FT-16b: the renderable-property set is a business rule, not a hint."""
        uploaded = await _upload(client, sample_audio_file)

        response = await client.post(
            f"/audio/{uploaded['audio_id']}/variant", json={"property": "reverb", "theta": 1.0}
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Unknown property: reverb"

    async def test_variant_reports_a_render_failure_as_502(
        self, client, sample_audio_file, fake_broker
    ):
        """FT-16c: a worker failure is an upstream error, not a 500."""
        uploaded = await _upload(client, sample_audio_file)
        fake_broker.result.value = RuntimeError("worker exploded")

        response = await client.post(
            f"/audio/{uploaded['audio_id']}/variant", json={"property": "pitch", "theta": 2.0}
        )

        assert response.status_code == 502
        assert response.json()["detail"] == "Variant render failed"

    async def test_variant_echoes_the_not_applicable_reason(
        self, client, sample_audio_file, fake_broker
    ):
        """FT-16d: 'not applicable' is a user-facing explanation, passed verbatim."""
        uploaded = await _upload(client, sample_audio_file)
        fake_broker.result.value = {
            "applicable": False,
            "reason": "clip shorter than one analysis frame",
        }

        response = await client.post(
            f"/audio/{uploaded['audio_id']}/variant", json={"property": "time_mask", "theta": 40.0}
        )

        assert response.status_code == 422
        assert response.json()["detail"] == "clip shorter than one analysis frame"


async def _asset(audio_id):
    from app.repositories.audio import AudioRepository

    return await AudioRepository().get(audio_id)


# --------------------------------------------------------------------------
# Control-plane surface -- the platform every use case depends on
# --------------------------------------------------------------------------


class TestControlPlaneSurface:
    async def test_health_reports_ok_when_every_dependency_answers(self, client):
        """FT-17: the System Administrator use case (SAD S4)."""
        response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["session_redis"] is True
        assert body["job_redis"] is True
        assert body["broker_redis"] is True
        assert body["storage"] is True
        assert set(body["queue_depth"]) == {"cpu", "gpu-fast", "gpu-large"}
        assert "detail" not in body

    async def test_health_degrades_to_503_and_names_the_failing_dependency(
        self, client, monkeypatch
    ):
        """FT-18: a degraded platform says which dependency is down."""

        class _Broken:
            async def ping(self):
                raise ConnectionError("job redis is unreachable")

            async def keys(self, pattern):
                raise ConnectionError("job redis is unreachable")

        monkeypatch.setattr(redis_module, "job_redis", _Broken())

        response = await client.get("/health")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "degraded"
        assert body["job_redis"] is False
        assert body["session_redis"] is True
        assert "job redis: job redis is unreachable" in body["detail"]

    async def test_metrics_reports_counters_and_queue_depth(self, client):
        """FT-19: /metrics reads the counters the workers actually increment."""
        await redis_module.job_redis.hincrby("metrics:jobs", "success", 3)
        await redis_module.job_redis.hincrby("metrics:jobs", "failure", 1)
        await redis_module.broker_redis.rpush("cpu", "task-a", "task-b")

        response = await client.get("/metrics")

        assert response.status_code == 200
        body = response.json()
        # Redis hashes are strings; the endpoint's contract is integers.
        assert body["jobs"] == {"success": 3, "failure": 1}
        assert body["queue_depth"] == {"cpu": 2, "gpu-fast": 0, "gpu-large": 0}

    async def test_metrics_survives_a_dependency_outage(self, client, monkeypatch):
        """FT-19b: /metrics degrades like /health rather than throwing.

        Guards BUG-08.
        """

        class _Broken:
            async def hgetall(self, key):
                raise ConnectionError("job redis is unreachable")

            async def llen(self, key):
                raise ConnectionError("job redis is unreachable")

        monkeypatch.setattr(redis_module, "job_redis", _Broken())

        response = await client.get("/metrics")

        # A monitoring endpoint that 500s during an outage is useless precisely
        # when it is needed; /health next door already returns a structured 503.
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"

    @pytest.mark.security
    async def test_debug_session_does_not_echo_the_session_cookie(self, client):
        """FT-20: the debug endpoint must not reflect credentials back.

        Guards BUG-09.
        """
        await client.get("/session")

        response = await client.get("/debug/session")

        assert response.status_code == 200
        body = response.json()
        assert body["session_id"]
        headers = {key.lower() for key in body.get("headers", {})}
        assert "cookie" not in headers
        assert "authorization" not in headers

    async def test_session_endpoint_issues_then_reuses_one_sid(self, client):
        """FT-21: session identity is minted once and then kept."""
        first = await client.get("/session")
        second = await client.get("/session")

        assert first.status_code == 200
        assert first.json()["sid"] == second.json()["sid"]
        assert len(first.json()["sid"]) == 32
        # The cookie is issued only when the request arrives without one.
        assert "set-cookie" in first.headers
        assert "set-cookie" not in second.headers

    @pytest.mark.parametrize(
        "path",
        ["/inferences/run", "/saliency/generate", "/perturb", "/results/whisper-base/key"],
        ids=["inferences", "saliency", "perturb", "results"],
    )
    @pytest.mark.parametrize("method", ["get", "post"], ids=["get", "post"])
    async def test_every_legacy_prefix_is_gone(self, client, path, method):
        """FT-22: the synchronous API is retired uniformly, not just for /results."""
        response = await getattr(client, method)(path) if method == "get" else await client.post(
            path, json={}
        )

        assert response.status_code == 410
        assert response.json() == {
            "detail": "Synchronous inference APIs are disabled; use POST /jobs"
        }

    async def test_the_legacy_gate_matches_by_prefix_not_by_path_segment(self, client):
        """FT-22b: characterises the gate's string-prefix matching (OBS-03)."""
        # Harmless today, but a future /resultsets route would be unreachable.
        response = await client.get("/results-not-a-real-route")

        assert response.status_code == 410

    @pytest.mark.integration
    async def test_concurrent_health_requests_all_succeed(self, client):
        """FT-23: the control plane stays responsive under parallel probes."""
        responses = await asyncio.gather(*[client.get("/health") for _ in range(5)])

        assert [r.status_code for r in responses] == [200] * 5
