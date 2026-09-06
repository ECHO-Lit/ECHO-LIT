"""Function Testing -- FR-3 dataset exploration and file serving.

Test Plan Section 3.1.2.  See tests/plans/3.1.2-function-testing.md.

SRS FR-3 acceptance criterion: "For a labeled dataset, class counts sum to the
number of items, and imbalance is flagged when the majority/minority ratio
exceeds a configured threshold."  Realized as SAD S4 use case 4 (Explore a
dataset with EDA, clustering, and nearest-neighbor retrieval).

Note on the second half of that criterion: the API emits `class_balance` counts
but no imbalance flag and no threshold -- the flagging is a Frontend behaviour.
This module asserts the API supplies everything a threshold needs (FT-69); the
flag itself belongs to Section 3.1.3.

The bundled datasets under Backend/data/ are gitignored, so the built-in cases
are guarded by a module-level skip rather than failing on a clean checkout.  The
Range/415 cases deliberately use a *custom* dataset instead: resolve_file shares
the entire serving path between the two, custom gives byte-level control of the
file, and it avoids the 0.3s sleep on the built-in missing-file branch.
"""

from __future__ import annotations

import io

import pytest
from httpx import AsyncClient

from app.core.settings import settings
from app.core.storage import get_storage
from app.main import app
from app.services import custom_dataset_service, dataset_service
from tests._fixtures import upload_files, wav_bytes

pytestmark = pytest.mark.critical

BUNDLED_AVAILABLE = dataset_service.DATASET_PATHS["ravdess"].exists()
requires_bundled = pytest.mark.skipif(
    not BUNDLED_AVAILABLE,
    reason="Bundled datasets live under Backend/data/, which is gitignored",
)


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


@pytest.fixture(scope="module", autouse=True)
def warm_metadata_cache():
    """Attribute the first bundled read to the module, not to whichever test runs first.

    `dataset_service._metadata_cache` is a process-global that no fixture clears;
    the first load of a bundled dataset probes ~150 audio files.
    """
    if BUNDLED_AVAILABLE:
        dataset_service.load_metadata("ravdess", None)


async def _sid(client) -> str:
    return (await client.get("/session")).json()["sid"]


async def _custom_dataset(client, name="serving", filenames=("tone.wav",)):
    """Create a custom dataset and return its qualified name."""
    created = await client.post("/upload/dataset/create", data={"dataset_name": name})
    assert created.status_code == 201, created.text
    uploaded = await client.post(
        f"/upload/dataset/{name}/files", files=upload_files(filenames)
    )
    assert uploaded.status_code == 200, uploaded.text
    return created.json()["dataset_name"]


# --------------------------------------------------------------------------
# Metadata retrieval
# --------------------------------------------------------------------------


class TestBuiltinMetadata:
    @requires_bundled
    @pytest.mark.parametrize(
        "dataset", ["common-voice", "cv-valid-dev", "ravdess", "l2-arctic", "saa"]
    )
    async def test_every_bundled_dataset_loads_with_normalised_keys(self, client, dataset):
        """FT-64: FR-3 retrieval -- each registered dataset is readable."""
        response = await client.get(f"/{dataset}/metadata")

        assert response.status_code == 200
        rows = response.json()
        assert rows, f"{dataset} returned no rows"
        # Keys are normalised on load so downstream code need not know which CSV
        # a row came from.
        assert all(key == key.lower() for key in rows[0])
        assert "filename" in rows[0]

    @requires_bundled
    async def test_dataset_names_are_case_insensitive(self, client):
        """FT-65a: the registry lookup lowercases, so display casing still works."""
        response = await client.get("/RAVDESS/metadata")

        assert response.status_code == 200
        assert response.json()

    async def test_an_unknown_dataset_is_404_with_its_name(self, client):
        """FT-65b: invalid data is refused with a message naming what was asked for."""
        response = await client.get("/nope/metadata")

        assert response.status_code == 404
        assert response.json()["detail"] == "Unknown dataset: nope"


# --------------------------------------------------------------------------
# EDA -- FR-3 acceptance criterion
# --------------------------------------------------------------------------


class TestEda:
    @requires_bundled
    async def test_class_counts_sum_to_the_item_count(self, client):
        """FT-66: FR-3 acceptance criterion, first half."""
        response = await client.get("/ravdess/eda")

        assert response.status_code == 200
        body = response.json()
        balance = body["class_balance"]
        summary = body["summary"]

        # The criterion, asserted literally.
        assert sum(balance.values()) == summary["total_files"]
        assert summary["num_classes"] == len(balance)
        assert summary["total_files"] > 0

    @requires_bundled
    async def test_the_duration_histogram_is_self_consistent(self, client):
        """FT-67: FR-3 processing -- the histogram describes the rows it came from."""
        body = (await client.get("/ravdess/eda")).json()
        histogram = body["duration_histogram"]

        # numpy histograms have one more edge than bucket; a mismatch here means
        # the chart would silently misalign.
        assert len(histogram["bins"]) == len(histogram["histogram"]) + 1
        assert sum(histogram["histogram"]) <= body["summary"]["total_files"]
        assert all(isinstance(count, int) for count in histogram["histogram"])
        assert all(edge == edge for edge in histogram["bins"])  # no NaN

    @requires_bundled
    async def test_an_unlabelled_dataset_reports_zero_classes_without_failing(self, client):
        """FT-68: a degenerate but legitimate input -- no labels is not an error."""
        response = await client.get("/common-voice/eda")

        assert response.status_code == 200
        body = response.json()
        assert body["summary"]["num_classes"] == 0
        assert body["class_balance"] == {}
        assert body["summary"]["total_files"] > 0

    @requires_bundled
    async def test_the_imbalance_ratio_is_derivable_from_the_response(self, client):
        """FT-69: FR-3's second half is satisfied across two tiers, not by the API.

        The response carries no imbalance flag and no threshold -- that decision
        lives in the Frontend.  What the API owes is a class_balance a threshold
        can be computed from, which is what this asserts.  The flag itself is
        Section 3.1.3's to verify.
        """
        balance = (await client.get("/ravdess/eda")).json()["class_balance"]

        assert balance
        assert all(isinstance(count, int) and count > 0 for count in balance.values())
        ratio = max(balance.values()) / min(balance.values())
        assert ratio >= 1.0

    async def test_eda_is_404_for_an_unknown_dataset(self, client):
        """FT-70: the EDA route gates on the same registry as metadata."""
        response = await client.get("/nope/eda")

        assert response.status_code == 404
        assert response.json()["detail"] == "Unknown dataset: nope"


# --------------------------------------------------------------------------
# File serving
# --------------------------------------------------------------------------


class TestFileServing:
    async def test_a_range_request_returns_only_the_requested_bytes(self, client):
        """FT-71: FR-3 playback -- seeking works without downloading the file."""
        dataset = await _custom_dataset(client, "serving", ("tone.wav",))
        expected = wav_bytes()

        response = await client.get(
            f"/{dataset}/file/tone.wav", headers={"Range": "bytes=0-9"}
        )

        assert response.status_code == 206
        assert response.content == expected[:10]
        assert response.headers["content-range"] == f"bytes 0-9/{len(expected)}"
        assert response.headers["content-length"] == "10"
        assert response.headers["accept-ranges"] == "bytes"

    async def test_an_open_ended_range_is_clamped_to_the_end_of_the_file(self, client):
        """FT-72a: `bytes=N-` is the common player request."""
        dataset = await _custom_dataset(client, "serving", ("tone.wav",))
        size = len(wav_bytes())

        response = await client.get(
            f"/{dataset}/file/tone.wav", headers={"Range": "bytes=100-"}
        )

        assert response.status_code == 206
        assert response.headers["content-range"] == f"bytes 100-{size - 1}/{size}"
        assert len(response.content) == size - 100

    async def test_a_malformed_range_header_falls_back_to_the_whole_file(self, client):
        """FT-72b: an unparseable header must not fail the request."""
        dataset = await _custom_dataset(client, "serving", ("tone.wav",))

        response = await client.get(
            f"/{dataset}/file/tone.wav", headers={"Range": "bytes=abc"}
        )

        assert response.status_code == 200
        assert response.content == wav_bytes()

    async def test_head_and_options_are_served_on_the_file_route(self, client):
        """FT-73: players preflight before they stream."""
        dataset = await _custom_dataset(client, "serving", ("tone.wav",))

        head = await client.head(f"/{dataset}/file/tone.wav")
        options = await client.options(f"/{dataset}/file/tone.wav")

        assert head.status_code == 200
        assert head.content == b""
        assert options.status_code == 200
        assert options.headers["access-control-allow-methods"] == "GET, HEAD, OPTIONS"

    async def test_an_unsupported_media_type_is_415(self, client, isolated_datasets):
        """FT-74: a file the platform cannot serve is refused, not streamed blind.

        The extension whitelist blocks this through the upload API, so the file
        is seeded directly into the dataset directory -- the state a stray copy
        or an older release would leave behind.
        """
        sid = await _sid(client)
        dataset = await _custom_dataset(client, "serving", ("tone.wav",))
        (isolated_datasets / sid / "datasets" / "serving" / "stray.ogg").write_bytes(b"OggS")

        response = await client.get(f"/{dataset}/file/stray.ogg")

        assert response.status_code == 415
        assert "ogg" in response.json()["detail"].lower()

    async def test_a_missing_file_in_a_real_dataset_is_404(self, client):
        """FT-74b: an absent file is reported rather than served as empty."""
        dataset = await _custom_dataset(client, "serving", ("tone.wav",))

        response = await client.get(f"/{dataset}/file/ghost.wav")

        assert response.status_code == 404


# --------------------------------------------------------------------------
# Routing and cross-session visibility
# --------------------------------------------------------------------------


class TestRoutingAndVisibility:
    async def test_an_unclaimed_metadata_path_falls_through_to_the_dataset_router(self, client):
        """FT-75: the datasets router is mounted at the root, so it catches these.

        Locking this in matters because it means a bare `assert status == 404`
        anywhere in the suite can pass for the wrong reason: almost any mistyped
        two-segment GET lands here.  The detail string is what distinguishes a
        handler 404 from a routing 404.
        """
        response = await client.get("/not-a-router/metadata")

        assert response.status_code == 404
        assert response.json()["detail"] == "Unknown dataset: not-a-router"

    async def test_a_missing_custom_dataset_returns_an_empty_list(self, client):
        """FT-76: characterises the built-in/custom asymmetry (OBS-04).

        A missing built-in dataset is a 404; a missing custom one is an empty
        200.  Recorded rather than corrected -- the frontend distinguishes "no
        files yet" from "no such dataset" by other means.
        """
        sid = await _sid(client)

        response = await client.get(f"/custom:{sid}:never-created/metadata")

        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.security
    async def test_another_sessions_custom_dataset_is_not_readable(self, client):
        """FT-77: FR-1 AC-2 and FR-2 AC-2 at the dataset-serving boundary.

        Guards BUG-12.  `load_metadata` and `resolve_file` detect a session
        mismatch and then serve the other session's data anyway.  The qualified
        name is not a secret -- it is returned to the client, appears in URLs and
        is logged -- so knowing it must not be enough to read the data.
        """
        owner_sid = await _sid(client)
        dataset = await _custom_dataset(client, "private", ("tone.wav",))

        async with AsyncClient(app=app, base_url="http://intruder") as intruder:
            intruder_sid = await _sid(intruder)
            assert intruder_sid != owner_sid

            metadata = await intruder.get(f"/{dataset}/metadata")
            audio = await intruder.get(f"/{dataset}/file/tone.wav")

        assert metadata.status_code == 404
        assert audio.status_code == 404

    @pytest.mark.security
    async def test_a_url_encoded_qualified_name_is_gated_identically(self, client):
        """FT-77b: the route unquotes its parameter, so both forms must agree."""
        owner_sid = await _sid(client)
        dataset = await _custom_dataset(client, "private", ("tone.wav",))
        encoded = dataset.replace(":", "%3A")

        async with AsyncClient(app=app, base_url="http://intruder") as intruder:
            assert await _sid(intruder) != owner_sid
            response = await intruder.get(f"/{encoded}/metadata")

        assert response.status_code == 404

    async def test_the_owner_can_still_read_their_own_dataset(self, client):
        """FT-77c: the isolation fix must not break the legitimate path."""
        dataset = await _custom_dataset(client, "mine", ("tone.wav",))

        metadata = await client.get(f"/{dataset}/metadata")
        audio = await client.get(f"/{dataset}/file/tone.wav")

        assert metadata.status_code == 200
        assert len(metadata.json()) == 1
        assert audio.status_code == 200
        assert audio.content == wav_bytes()
