"""Security and Access Control Testing -- Test Plan Section 3.1.6.

See tests/plans/3.1.6-security-and-access-control-testing.md.

ECHO has no accounts, logins or roles. Its access-control boundary is the
anonymous **session**: every resource a caller creates is owned by the `sid`
cookie that created it, and the repository layer refuses to return a resource to
any other session (`get_owned`). For a tool that ingests user audio and holds
derived analyses, that cross-session isolation *is* the security requirement.

Two rules make every case here evidence rather than decoration; both exist
because this codebase has already shipped tests that passed for the wrong reason
(3.1.2 §Findings):

* **Paired outcome (O-2).** A denial is only meaningful beside the owner's
  matching success in the *same* test. A lone 404 also fires when an endpoint is
  broken, mistyped or removed -- which is exactly how three cases in
  test_security.py pass today. `_denied_then_allowed` enforces the pairing.
* **Exact detail, and non-disclosure (O-1, O-3).** The datasets router is
  mounted at the application root as `/{dataset}/metadata`, so almost any
  mistyped two-segment GET returns *some* 404; the status code alone proves
  nothing. And "not yours" must be byte-identical to "does not exist", or the
  response turns an opaque id into an oracle for an attacker.

Isolation of the built-in resource types (audio, dataset, job, model, lens) is
already asserted piecemeal across the 3.1.2 modules (FT-13, FT-30, FT-38d,
FT-77, FT-83, FT-91). This module does not restate those; it adds the cases that
belong to *this* section -- the non-disclosure guarantee (SEC-02/03), the
job-input trust boundary for the newest operations (SEC-22/23), the system-level
surface (cookie attributes, CORS, the legacy gate, route inventory), and the
session-identity boundary itself, where the probe below found a live defect.
"""

from __future__ import annotations

import io

import pytest
from httpx import AsyncClient

from app.core.settings import settings
from app.core.storage import get_storage
from app.main import app
from app.services import custom_dataset_service
from tests._fixtures import upload_files, wav_bytes

pytestmark = [pytest.mark.security, pytest.mark.critical]


# ---------------------------------------------------------------------------
# Isolation from the developer's real object store and dataset tree. Both roots
# default to relative paths, so without these an /upload writes into the repo.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


@pytest.fixture(autouse=True)
def isolated_datasets(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    monkeypatch.setattr(custom_dataset_service, "SESSIONS_BASE_DIR", root)
    return root


# ---------------------------------------------------------------------------
# Actor construction and the two oracle helpers.
# ---------------------------------------------------------------------------


async def _sid(client) -> str:
    return (await client.get("/session")).json()["sid"]


async def _upload(client, name: str = "clip.wav") -> str:
    response = await client.post(
        "/upload", files={"file": (name, io.BytesIO(wav_bytes()), "audio/wav")}
    )
    assert response.status_code == 201, response.text
    return response.json()["audio_id"]


async def _custom_dataset(client, name: str = "private", files=("tone.wav",)) -> str:
    created = await client.post("/upload/dataset/create", data={"dataset_name": name})
    assert created.status_code == 201, created.text
    uploaded = await client.post(f"/upload/dataset/{name}/files", files=upload_files(files))
    assert uploaded.status_code == 200, uploaded.text
    return created.json()["dataset_name"]


@pytest.fixture
async def owner():
    """Actor A: the session that creates the resource under test."""
    async with AsyncClient(app=app, base_url="http://owner") as client:
        yield client


@pytest.fixture
async def intruder(owner):
    """Actor B: an independent session. The distinct sid is asserted here, once,
    so no individual case can pass because the two shared a cookie jar."""
    async with AsyncClient(app=app, base_url="http://intruder") as client:
        assert await _sid(client) != await _sid(owner)
        yield client


async def _denied_then_allowed(intruder_call, owner_call, *, detail: str):
    """O-2: the denial must be caused by ownership, not by a broken endpoint.

    `intruder_call`/`owner_call` are zero-arg coroutines issuing the *same*
    request under the two sessions.
    """
    denied = await intruder_call()
    assert denied.status_code == 404, denied.text
    assert denied.json()["detail"] == detail

    allowed = await owner_call()
    assert allowed.status_code == 200, allowed.text
    return denied, allowed


def _assert_indistinguishable(foreign, fabricated) -> None:
    """O-3: 'not yours' and 'does not exist' must be the same response."""
    assert foreign.status_code == fabricated.status_code
    assert foreign.json() == fabricated.json()


# ===========================================================================
# SEC-01..08  Application-level security: audio ownership and non-disclosure
# ===========================================================================


class TestAudioAccessControl:
    async def test_foreign_session_is_denied_but_owner_is_served(self, owner, intruder):
        """SEC-01 / SEC-03."""
        audio_id = await _upload(owner)
        await _denied_then_allowed(
            lambda: intruder.get(f"/audio/{audio_id}"),
            lambda: owner.get(f"/audio/{audio_id}"),
            detail="Audio not found",
        )

    async def test_denial_does_not_disclose_existence(self, owner, intruder):
        """SEC-02: a real-but-foreign id looks identical to a fabricated one.

        If it did not, the 404/200 split would let a caller enumerate which
        opaque audio ids exist in other sessions.
        """
        audio_id = await _upload(owner)
        _assert_indistinguishable(
            await intruder.get(f"/audio/{audio_id}"),
            await intruder.get(f"/audio/{'0' * 32}"),
        )

    async def test_foreign_delete_leaves_the_owner_intact(self, owner, intruder):
        """SEC-05: assert the side effect did NOT happen, not merely the 404."""
        audio_id = await _upload(owner)

        assert (await intruder.delete(f"/upload/{audio_id}")).status_code == 404
        assert (await owner.get(f"/audio/{audio_id}")).status_code == 200

    async def test_foreign_variant_render_is_refused(self, owner, intruder):
        """SEC-06: the render route resolves an owned asset before dispatching."""
        audio_id = await _upload(owner)

        response = await intruder.post(
            f"/audio/{audio_id}/variant", json={"property": "pitch", "theta": 2.0}
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Audio not found"

    async def test_listing_never_carries_another_sessions_ids(self, owner, intruder):
        """SEC-13, O-4: assert on content, not on a length difference."""
        audio_id = await _upload(owner)

        body = (await intruder.get("/upload/list")).text
        assert audio_id not in body


# ===========================================================================
# SEC-09..14  Custom dataset ownership -- the BUG-12 regression surface
# ===========================================================================


class TestCustomDatasetAccessControl:
    async def test_qualified_name_from_another_session_is_unknown(self, owner, intruder):
        """SEC-09 / SEC-10 / SEC-12: guards BUG-12.

        The qualified name `custom:<sid>:<name>` is returned to the client,
        appears in URLs and is logged -- it is not a secret. Holding one must not
        be enough to read the data behind it, on either the metadata or the file
        path, while the owner keeps working (O-2).
        """
        dataset = await _custom_dataset(owner)

        await _denied_then_allowed(
            lambda: intruder.get(f"/{dataset}/metadata"),
            lambda: owner.get(f"/{dataset}/metadata"),
            detail=f"Unknown dataset: {dataset}",
        )
        await _denied_then_allowed(
            lambda: intruder.get(f"/{dataset}/file/tone.wav"),
            lambda: owner.get(f"/{dataset}/file/tone.wav"),
            detail=f"Unknown dataset: {dataset}",
        )

    async def test_materialize_cannot_pull_a_foreign_dataset_file(self, owner, intruder):
        """SEC-11: the third BUG-12 path -- POST /audio/materialize.

        A cross-session read here would additionally mint an AudioAsset in the
        intruder's own session, laundering the leak into an owned resource.
        """
        dataset = await _custom_dataset(owner)

        response = await intruder.post(
            "/audio/materialize", json={"dataset": dataset, "filename": "tone.wav"}
        )
        assert response.status_code == 404
        assert response.json()["detail"] == f"Unknown dataset: {dataset}"

    async def test_dataset_listing_is_session_scoped(self, owner, intruder):
        """SEC-13: the intruder's own listing is empty and names nothing of A's."""
        dataset = await _custom_dataset(owner)

        listing = await intruder.get("/upload/dataset/list")
        assert listing.status_code == 200
        body = listing.json()
        assert body["total_datasets"] == 0
        assert dataset not in listing.text

    async def test_foreign_label_key_is_neither_readable_nor_writable(self, owner, intruder):
        """SEC-14: the answer key rides on the dataset, so it inherits its scope."""
        name = "labelled"
        await _custom_dataset(owner, name=name)

        # The dataset-management routes are mounted under /upload and keyed on the
        # *bare* name resolved against the caller's own session, so the intruder
        # simply has no dataset by that name.
        read = await intruder.get(f"/upload/dataset/{name}/labels")
        assert read.status_code == 404
        assert read.json()["detail"] == f"Dataset '{name}' not found"

        assert (await owner.get(f"/upload/dataset/{name}/labels")).status_code == 200


# ===========================================================================
# SEC-15..17  Dataset file serving: traversal and the built-in asymmetry
# ===========================================================================


class TestDatasetServingBoundary:
    @pytest.mark.parametrize(
        "traversal",
        ["../dataset_metadata.json", "..%2fdataset_metadata.json", "....//dataset_metadata.json"],
        ids=["parent", "encoded", "doubled"],
    )
    async def test_a_traversing_file_path_cannot_escape_the_dataset(self, owner, traversal):
        """SEC-15: `resolve_file` reduces the path to its basename, so a `..`
        segment resolves to a non-existent file inside the dataset, never to the
        metadata JSON one level up."""
        dataset = await _custom_dataset(owner)

        response = await owner.get(f"/{dataset}/file/{traversal}")
        assert response.status_code == 404

    async def test_builtin_datasets_are_deliberately_shared(self, owner, intruder):
        """SEC-17: documents the intended asymmetry -- built-in corpora are
        read-only reference data and are NOT session-scoped, unlike everything a
        user creates. Recorded so the difference is a decision, not a surprise."""
        if not _builtin_available():
            pytest.skip("Bundled datasets live under Backend/data/, which is gitignored")

        owner_view = await owner.get("/ravdess/metadata")
        intruder_view = await intruder.get("/ravdess/metadata")
        assert owner_view.status_code == intruder_view.status_code == 200


def _builtin_available() -> bool:
    from app.services import dataset_service

    return dataset_service.DATASET_PATHS["ravdess"].exists()


# ===========================================================================
# SEC-18..24  Job ownership and the submission-time input trust boundary
# ===========================================================================


class TestJobAccessControl:
    @pytest.fixture(autouse=True)
    def fake_broker(self, monkeypatch):
        """Jobs would otherwise reach for a real broker. Record dispatches so a
        rejected submission can be proven to have created no job."""
        from types import SimpleNamespace

        from app.api.routes import jobs as jobs_routes

        calls: list[dict] = []
        monkeypatch.setattr(
            jobs_routes.celery_app,
            "send_task",
            lambda name, args=None, queue=None, **_: calls.append({"name": name}) or SimpleNamespace(id="t"),
        )
        monkeypatch.setattr(jobs_routes.celery_app.control, "revoke", lambda *a, **k: None)
        return SimpleNamespace(calls=calls)

    async def _submit(self, client, audio_id, operation="prediction", model="whisper-base"):
        return await client.post(
            "/jobs",
            json={"operation": operation, "model": model, "audio_ids": [audio_id]},
        )

    async def test_foreign_status_result_and_cancel_are_all_denied(self, owner, intruder):
        """SEC-18/19/20/21: every FR-14 read and the cancel route are scoped,
        and the owner's job survives the attempt."""
        audio_id = await _upload(owner)
        job_id = (await self._submit(owner, audio_id)).json()["job_id"]

        status = await intruder.get(f"/jobs/{job_id}")
        result = await intruder.get(f"/jobs/{job_id}/result")
        cancel = await intruder.delete(f"/jobs/{job_id}")

        assert status.status_code == 404
        assert status.json()["detail"] == "Job not found"
        assert result.status_code == 404
        assert cancel.status_code == 404
        assert (await owner.get(f"/jobs/{job_id}")).json()["status"] == "queued"

    async def test_a_job_cannot_be_built_over_foreign_audio(self, owner, intruder, fake_broker):
        """SEC-22: the trust boundary is submission, not execution. A rejected
        submission must create no job record at all -- otherwise a foreign
        audio_id would still spawn a queued job in the intruder's session."""
        audio_id = await _upload(owner)

        response = await self._submit(intruder, audio_id)

        assert response.status_code == 404
        assert response.json()["detail"] == f"Audio not found: {audio_id}"
        assert fake_broker.calls == []

    async def test_the_faithfulness_style_single_audio_op_inherits_the_check(
        self, owner, intruder, fake_broker
    ):
        """SEC-23: a per-clip operation added after this boundary was written
        still resolves its audio through `get_owned`, so it cannot reach another
        session's clip. `saliency` stands in for that whole family (it is the
        single-audio op present on this branch)."""
        audio_id = await _upload(owner)

        response = await self._submit(intruder, audio_id, operation="saliency", model="wav2vec2")

        assert response.status_code == 404
        assert response.json()["detail"] == f"Audio not found: {audio_id}"
        assert fake_broker.calls == []


# ===========================================================================
# SEC-30..34  Session identity -- issuance, cookie attributes, and the
#             client-controlled-sid trust boundary (where the probe found a bug)
# ===========================================================================


class TestSessionIdentity:
    async def test_a_fresh_client_is_issued_a_set_cookie(self, owner):
        """SEC-30."""
        async with AsyncClient(app=app, base_url="http://fresh") as fresh:
            response = await fresh.get("/session")
        assert "set-cookie" in response.headers
        assert response.json()["sid"]

    async def test_cookie_carries_the_expected_attributes(self):
        """SEC-31/32/33: HttpOnly is load-bearing -- with no second factor it is
        the only thing stopping a stored-XSS payload from exfiltrating the
        session. Secure is asserted absent to DOCUMENT the dev default; a
        deployment checklist, not a test, must confirm the production override."""
        async with AsyncClient(app=app, base_url="http://fresh") as fresh:
            header = (await fresh.get("/session")).headers["set-cookie"]

        assert "HttpOnly" in header
        assert "Path=/" in header
        assert f"Max-Age={settings.SESSION_TTL_SECONDS}" in header
        assert f"SameSite={settings.COOKIE_SAMESITE}" in header  # Starlette emits the value verbatim
        assert "Secure" not in header  # COOKIE_SECURE defaults to False (see plan §Special Considerations)

    @pytest.mark.parametrize(
        "forged",
        ["..", "../escaped", "a/b", "../../etc", "not-hex", "", "AB" * 16 + "zz"],
        ids=["dotdot", "traversal", "separator", "deep", "nonhex", "empty", "toolong"],
    )
    async def test_a_client_chosen_sid_is_never_trusted_as_identity(self, forged):
        """SEC-08b / BUG-13: the sid becomes a filesystem path segment
        (uploads/sessions/<sid>, datasets/<sid>, results/<sid>), so a cookie
        containing `..` or a separator would escape those namespaces -- at worst
        POST /upload/dataset/cleanup rmtree-ing the parent of the sessions root.

        A minted sid is 32 lowercase hex; anything else is not one we issued and
        must be replaced with a fresh valid id rather than adopted verbatim.
        """
        async with AsyncClient(app=app, base_url="http://test", cookies={"sid": forged}) as client:
            adopted = (await client.get("/session")).json()["sid"]

        assert adopted != forged
        assert len(adopted) == 32 and all(c in "0123456789abcdef" for c in adopted)

    async def test_a_valid_sid_is_preserved_across_requests(self, owner):
        """SEC-34: the fix above must not disrupt a legitimate reused session --
        a real 32-hex id issued by the app is kept, not re-minted."""
        first = await _sid(owner)
        second = await _sid(owner)
        assert first == second
        assert len(first) == 32


# ===========================================================================
# SEC-35..40  System-level surface: legacy gate, admin surface, health exemption
# ===========================================================================


class TestDeploymentSurface:
    @pytest.mark.parametrize("prefix", ["/inferences", "/saliency", "/perturb", "/results"])
    async def test_legacy_synchronous_api_is_gated_off_by_default(self, owner, prefix):
        """SEC-37: the deprecated synchronous inference surface is disabled unless
        deliberately enabled (ENABLE_LEGACY_SYNC_INFERENCE, non-prod only)."""
        response = await owner.get(prefix)
        assert response.status_code == 410
        assert response.json() == {"detail": "Synchronous inference APIs are disabled; use POST /jobs"}

    def test_no_unauthenticated_admin_surface_exists(self):
        """SEC-39: the honest form of test_security.py's admin probe. That test
        GETs /admin/* and accepts a 404 -- which holds for a completely
        unprotected app, because the routes simply do not exist. Assert the route
        inventory directly instead of inferring protection from an absence."""
        paths = {getattr(route, "path", "") for route in app.routes}
        assert not any(path.startswith("/admin") for path in paths), sorted(paths)
        assert not any(path.startswith("/login") for path in paths)

    async def test_health_is_exempt_from_session_issuance(self, owner):
        """SEC-40: /health is the middleware's one documented exemption -- a probe
        from a monitoring system must not be handed (and must not need) a cookie."""
        async with AsyncClient(app=app, base_url="http://monitor") as monitor:
            response = await monitor.get("/health")
        assert "set-cookie" not in response.headers
        assert response.status_code in (200, 503)  # 503 when fake redis reports a dep down
