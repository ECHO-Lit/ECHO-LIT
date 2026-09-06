"""Function Testing -- FR-2 dataset management (custom datasets, manifests, labels).

Test Plan Section 3.1.2.  See tests/plans/3.1.2-function-testing.md.

SRS FR-2 acceptance criteria:
  * "A custom dataset created with N files reports exactly N files with correct
    metadata."
  * "Deleting a user-uploaded dataset removes its files and metadata, ensuring no
    trace of the user's data is kept."
Realized as the SAD S4 Dataset Curator actor.

Scope: the HTTP layer only.  `CustomDatasetManager` and the label parser are
already covered directly by test_custom_dataset_manifest.py and
test_dataset_labels_service.py; duplicating those here would add nothing.  What
is untested before this module is everything between the request and that
service -- status codes, the exact user-facing messages, multipart handling, the
4 MB and UTF-8 guards, and session scoping.
"""

from __future__ import annotations

import io
import json

import pytest
from httpx import AsyncClient

from app.core.settings import settings
from app.core.storage import get_storage
from app.main import app
from app.services import custom_dataset_service
from tests._fixtures import upload_files, wav_bytes

pytestmark = pytest.mark.critical


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


@pytest.fixture(autouse=True)
def isolated_datasets(tmp_path, monkeypatch):
    """SESSIONS_BASE_DIR defaults to the relative path "uploads/sessions".

    Without this every case in this module writes a real session tree into
    Backend/uploads/sessions/ -- and the traversal cases would write outside it.
    """
    root = tmp_path / "sessions"
    monkeypatch.setattr(custom_dataset_service, "SESSIONS_BASE_DIR", root)
    return root


async def _sid(client) -> str:
    return (await client.get("/session")).json()["sid"]


async def _create(client, name="speech"):
    response = await client.post("/upload/dataset/create", data={"dataset_name": name})
    assert response.status_code == 201, response.text
    return response.json()


async def _create_with_files(client, name="speech", filenames=("a.wav", "b.wav", "c.wav")):
    await _create(client, name)
    response = await client.post(
        f"/upload/dataset/{name}/files", files=upload_files(filenames)
    )
    assert response.status_code == 200, response.text
    return response.json()


def _csv(rows: str) -> dict:
    return {"file": ("labels.csv", io.BytesIO(rows.encode("utf-8")), "text/csv")}


# --------------------------------------------------------------------------
# Creation
# --------------------------------------------------------------------------


class TestDatasetCreation:
    async def test_create_returns_the_qualified_name_and_empty_metadata(self, client):
        """FT-42: FR-2 outputs -- a new dataset is addressable and starts empty."""
        sid = await _sid(client)

        body = await _create(client, "speech")

        # The qualified name is what every other API expects back.
        assert body["dataset_name"] == f"custom:{sid}:speech"
        assert body["original_name"] == "speech"
        assert body["session_id"] == sid
        assert body["metadata"]["total_files"] == 0
        assert body["metadata"]["files"] == []

    async def test_creating_a_duplicate_name_is_rejected(self, client):
        """FT-43: names are unique within a session."""
        await _create(client, "speech")

        response = await client.post("/upload/dataset/create", data={"dataset_name": "speech"})

        assert response.status_code == 400
        assert response.json()["detail"] == "Dataset 'speech' already exists in this session"

    async def test_create_without_the_form_field_is_422(self, client):
        """FT-44: the dataset name is required."""
        response = await client.post("/upload/dataset/create")

        assert response.status_code == 422

    @pytest.mark.security
    @pytest.mark.parametrize(
        "name",
        ["../escaped", "../../escaped", "a/../../escaped"],
        ids=["parent", "grandparent", "nested"],
    )
    async def test_a_traversing_dataset_name_cannot_escape_the_session_tree(
        self, client, isolated_datasets, name
    ):
        """FT-45: the dataset name is user input and must be constrained.

        Guards BUG-10.  `dataset_name` arrives as a form field and is joined
        straight onto the session's datasets directory.
        """
        response = await client.post("/upload/dataset/create", data={"dataset_name": name})

        assert response.status_code == 400
        assert "name" in response.json()["detail"].lower()
        # Nothing may exist above the session's own datasets directory.
        escaped = [
            path
            for path in isolated_datasets.rglob("escaped")
            if "datasets" not in path.relative_to(isolated_datasets).parts[:2]
        ]
        assert escaped == []


# --------------------------------------------------------------------------
# File upload -- FR-2 acceptance criterion 1
# --------------------------------------------------------------------------


class TestDatasetFileUpload:
    async def test_n_files_in_reports_n_files_out_with_correct_metadata(self, client):
        """FT-46: FR-2 acceptance criterion -- N in, exactly N out, described correctly."""
        sid = await _sid(client)
        body = await _create_with_files(client, "speech", ("a.wav", "b.wav", "c.wav"))

        assert body["total_files"] == 3
        assert body["dataset_name"] == f"custom:{sid}:speech"
        assert body["dataset_metadata"]["total_files"] == 3
        assert len(body["uploaded_files"]) == 3

        expected_size = len(wav_bytes())
        for entry in body["uploaded_files"]:
            assert entry["filename"] in {"a.wav", "b.wav", "c.wav"}
            assert entry["size"] == expected_size
            # Metadata must be probed, not assumed: a 0.25s tone at 16 kHz.
            assert entry["duration"] == pytest.approx(0.25, abs=0.02)

        listed = await client.get("/upload/dataset/speech/files")
        assert listed.status_code == 200
        assert listed.json()["total_files"] == 3

    async def test_a_duplicate_filename_is_stored_alongside_the_original(self, client):
        """FT-47: a repeated name must not overwrite existing audio."""
        await _create(client, "speech")
        await client.post("/upload/dataset/speech/files", files=upload_files(["a.wav"]))

        second = await client.post(
            "/upload/dataset/speech/files", files=upload_files(["a.wav"])
        )

        assert second.status_code == 200
        names = {entry["filename"] for entry in (
            await client.get("/upload/dataset/speech/files")
        ).json()["files"]}
        assert names == {"a.wav", "a_1.wav"}

    @pytest.mark.parametrize(
        ("filename", "content_type", "expected"),
        [
            ("notes.txt", "text/plain",
             "Invalid file type for notes.txt. Only audio files are allowed."),
            ("clip.ogg", "audio/ogg",
             "Invalid file extension for clip.ogg. Allowed: .wav, .mp3, .m4a, .flac"),
        ],
        ids=["content_type", "extension"],
    )
    async def test_an_invalid_file_is_rejected_before_anything_is_written(
        self, client, filename, content_type, expected
    ):
        """FT-48: validation precedes the write loop, so nothing half-lands."""
        await _create(client, "speech")

        response = await client.post(
            "/upload/dataset/speech/files",
            files=[("files", (filename, io.BytesIO(wav_bytes()), content_type))],
        )

        assert response.status_code == 400
        assert response.json()["detail"] == expected
        assert (await client.get("/upload/dataset/speech/files")).json()["total_files"] == 0

    async def test_one_bad_file_rejects_the_whole_batch(self, client):
        """FT-49: the batch is validated as a unit -- partial acceptance would
        leave the user guessing which files landed."""
        await _create(client, "speech")

        response = await client.post(
            "/upload/dataset/speech/files",
            files=[
                ("files", ("a.wav", io.BytesIO(wav_bytes()), "audio/wav")),
                ("files", ("bad.txt", io.BytesIO(b"x"), "text/plain")),
                ("files", ("c.wav", io.BytesIO(wav_bytes()), "audio/wav")),
            ],
        )

        assert response.status_code == 400
        assert (await client.get("/upload/dataset/speech/files")).json()["total_files"] == 0

    async def test_uploading_to_an_unknown_dataset_reports_not_found(self, client):
        """FT-51: an unknown dataset is a 404, not a 207 with a null body.

        Guards BUG-11.  The per-file `except Exception` swallows the service's
        "does not exist" ValueError, so the route's own 404 is unreachable and
        the client receives a success-shaped response describing nothing.
        """
        response = await client.post(
            "/upload/dataset/ghost/files", files=upload_files(["a.wav"])
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Dataset 'ghost' does not exist"

    async def test_an_absent_files_field_is_422(self, client):
        """FT-52: FastAPI rejects the missing multipart field before the handler.

        The handler's own `if not files: 400 "No files provided"` is therefore
        unreachable; this asserts the status a client actually receives.
        """
        await _create(client, "speech")

        response = await client.post("/upload/dataset/speech/files")

        assert response.status_code == 422


# --------------------------------------------------------------------------
# Manifests and labels
# --------------------------------------------------------------------------


class TestDatasetManifest:
    async def test_a_valid_manifest_is_attached(self, client):
        """FT-53: FR-9 prerequisite -- transcripts are joined to stored audio."""
        await _create_with_files(client, "speech", ("a.wav", "b.wav"))

        response = await client.post(
            "/upload/dataset/speech/manifest",
            files={"manifest": (
                "m.csv",
                io.BytesIO(b"filename,transcript\na.wav,hello there\nb.wav,goodbye now\n"),
                "text/csv",
            )},
        )

        assert response.status_code == 200
        manifest = response.json()["manifest"]
        assert manifest["pair_count"] == 2
        assert manifest["matched_audio_count"] == 2
        assert manifest["unmatched_audio_count"] == 0

    @pytest.mark.parametrize(
        ("body", "fragment"),
        [
            (b"nothing,useful\nx,y\n", "needs a filename column"),
            (b"filename,transcript\na.wav,one\na.wav,two\n", "duplicate"),
            (b"filename,transcript\n", "no filename/transcript pairs"),
        ],
        ids=["no_columns", "duplicate_filename", "empty"],
    )
    async def test_manifest_errors_surface_as_400(self, client, body, fragment):
        """FT-54: every manifest rejection reaches the user as a 4xx explanation."""
        await _create_with_files(client, "speech", ("a.wav",))

        response = await client.post(
            "/upload/dataset/speech/manifest",
            files={"manifest": ("m.csv", io.BytesIO(body), "text/csv")},
        )

        assert response.status_code == 400
        assert fragment in response.json()["detail"].lower()


class TestDatasetLabels:
    async def test_label_patterns_are_listed_with_their_properties(self, client):
        """FT-55: the derive-from-filename options are discoverable."""
        response = await client.get("/upload/dataset/label-patterns")

        assert response.status_code == 200
        patterns = response.json()["patterns"]
        assert patterns
        for pattern in patterns:
            assert {"pattern_id", "label", "description", "properties"} <= set(pattern)

    async def test_a_labels_csv_is_stored_and_previewed(self, client):
        """FT-56: FR-9's answer key is accepted and summarised back."""
        await _create_with_files(client, "speech", ("a.wav", "b.wav", "c.wav"))

        response = await client.post(
            "/upload/dataset/speech/labels",
            files=_csv("filename,emotion\na.wav,happy\nb.wav,sad\nc.wav,happy\n"),
        )

        assert response.status_code == 200
        body = response.json()
        assert body["matched_files"] == 3
        assert body["stored"] == ["emotion"]
        assert body["source"] == "csv"
        # `columns` is the dataset's whole probeable column set, so the uploaded
        # property joins the intrinsic ones rather than replacing them.
        assert "emotion" in body["columns"]

        # Readable back through the GET route.
        fetched = await client.get("/upload/dataset/speech/labels")
        assert fetched.status_code == 200
        assert "emotion" in fetched.json()["columns"]
        assert fetched.json()["source"] == "csv"

    async def test_an_oversized_label_csv_is_rejected(self, client):
        """FT-57a: an answer key is one row per file; a huge one is a mistake."""
        await _create_with_files(client, "speech", ("a.wav",))
        oversized = b"filename,emotion\n" + b"a.wav,happy\n" * 400_000
        assert len(oversized) > 4 * 1024 * 1024

        response = await client.post(
            "/upload/dataset/speech/labels",
            files={"file": ("labels.csv", io.BytesIO(oversized), "text/csv")},
        )

        assert response.status_code == 413
        assert response.json()["detail"] == "Label CSV is too large (limit 4 MB)"

    async def test_a_non_utf8_label_csv_is_rejected(self, client):
        """FT-57b: encoding is checked before parsing, with a fixable message."""
        await _create_with_files(client, "speech", ("a.wav",))

        response = await client.post(
            "/upload/dataset/speech/labels",
            files={"file": (
                "labels.csv",
                io.BytesIO("filename,emotion\na.wav,café\n".encode("latin-1")),
                "text/csv",
            )},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Label CSV must be UTF-8 encoded"

    async def test_a_csv_without_a_filename_column_is_rejected(self, client):
        """FT-57c: the join key is mandatory."""
        await _create_with_files(client, "speech", ("a.wav",))

        response = await client.post(
            "/upload/dataset/speech/labels", files=_csv("name,emotion\na.wav,happy\n")
        )

        assert response.status_code == 400
        assert response.json()["detail"]

    async def test_a_csv_matching_nothing_names_the_expected_filenames(self, client):
        """FT-58: the most likely user error gets the most actionable message."""
        await _create_with_files(client, "speech", ("a.wav", "b.wav"))

        response = await client.post(
            "/upload/dataset/speech/labels",
            files=_csv("filename,emotion\nzz.wav,happy\nyy.wav,sad\n"),
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail.startswith(
            "The CSV parsed but none of its filenames match this dataset. "
            "Dataset files look like: "
        )
        # The hint must name files that actually exist, or it misleads.
        assert "a.wav" in detail or "b.wav" in detail

    async def test_a_partially_matching_csv_warns_instead_of_failing(self, client):
        """FT-59: a partial answer key is usable, so it is accepted with a warning."""
        await _create_with_files(client, "speech", ("a.wav", "b.wav", "c.wav"))

        response = await client.post(
            "/upload/dataset/speech/labels",
            files=_csv("filename,emotion\na.wav,happy\nb.wav,sad\n"),
        )

        assert response.status_code == 200
        assert response.json()["matched_files"] == 2
        assert any("no row in the CSV" in warning for warning in response.json()["warnings"])

    async def test_labels_can_be_derived_from_filenames_and_then_cleared(self, client):
        """FT-60: the no-CSV path, and the recovery path back out of it."""
        await _create_with_files(
            client, "speech", ("03-01-05-01-01-01-12.wav", "03-01-03-01-01-01-06.wav")
        )
        patterns = (await client.get("/upload/dataset/label-patterns")).json()["patterns"]
        ravdess = next(p for p in patterns if "ravdess" in p["pattern_id"].lower())

        derived = await client.post(
            "/upload/dataset/speech/labels/derive", data={"pattern_id": ravdess["pattern_id"]}
        )

        assert derived.status_code == 200
        assert derived.json()["matched_files"] == 2
        # The source records which pattern produced the table, so a user can see
        # where an answer key came from without re-deriving it.
        assert derived.json()["source"] == f"pattern:{ravdess['pattern_id']}"

        cleared = await client.delete("/upload/dataset/speech/labels")
        assert cleared.status_code == 200
        assert (await client.get("/upload/dataset/speech/labels")).json()["source"] is None
        # Clearing the answer key must not touch the audio.
        assert (await client.get("/upload/dataset/speech/files")).json()["total_files"] == 2

    async def test_an_unknown_derive_pattern_is_rejected(self, client):
        """FT-60b: an invalid pattern id is refused, not silently ignored."""
        await _create_with_files(client, "speech", ("a.wav",))

        response = await client.post(
            "/upload/dataset/speech/labels/derive", data={"pattern_id": "nope"}
        )

        assert response.status_code == 400

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/upload/dataset/ghost/labels"),
            ("delete", "/upload/dataset/ghost/labels"),
        ],
        ids=["get", "delete"],
    )
    async def test_label_routes_404_on_an_unknown_dataset(self, client, method, path):
        """FT-61: every label route gates on dataset existence identically."""
        response = await getattr(client, method)(path)

        assert response.status_code == 404
        assert response.json()["detail"] == "Dataset 'ghost' not found"

    async def test_label_write_routes_404_on_an_unknown_dataset(self, client):
        """FT-61b: the two multipart/form label writers gate the same way."""
        posted = await client.post(
            "/upload/dataset/ghost/labels", files=_csv("filename,emotion\na.wav,happy\n")
        )
        derived = await client.post(
            "/upload/dataset/ghost/labels/derive", data={"pattern_id": "ravdess_emotion"}
        )

        assert posted.status_code == 404
        assert posted.json()["detail"] == "Dataset 'ghost' not found"
        assert derived.status_code == 404
        assert derived.json()["detail"] == "Dataset 'ghost' not found"


# --------------------------------------------------------------------------
# Lifecycle -- FR-2 acceptance criterion 2
# --------------------------------------------------------------------------


class TestDatasetLifecycle:
    async def test_delete_removes_every_file_and_all_metadata(self, client, isolated_datasets):
        """FT-62: FR-2 acceptance criterion -- 'no trace of the user's data is kept'."""
        sid = await _sid(client)
        await _create_with_files(client, "speech", ("a.wav", "b.wav"))
        await client.post(
            "/upload/dataset/speech/labels", files=_csv("filename,emotion\na.wav,happy\n")
        )
        dataset_dir = isolated_datasets / sid / "datasets" / "speech"
        assert dataset_dir.is_dir()

        response = await client.delete("/upload/dataset/speech")

        assert response.status_code == 200
        # On disk: gone entirely, including the audio and the answer key.
        assert not dataset_dir.exists()
        # Through the API: absent from every read path.
        assert (await client.get("/upload/dataset/list")).json()["total_datasets"] == 0
        assert (await client.get("/upload/dataset/speech/metadata")).status_code == 404
        assert (await client.get("/upload/dataset/speech/files")).status_code == 404

    async def test_cleanup_removes_the_whole_session_tree(self, client, isolated_datasets):
        """FT-62b: the session-level erase covers every dataset at once."""
        sid = await _sid(client)
        await _create_with_files(client, "one", ("a.wav",))
        await _create_with_files(client, "two", ("b.wav",))

        response = await client.post("/upload/dataset/cleanup")

        assert response.status_code == 200
        assert response.json()["success"] is True
        assert not (isolated_datasets / sid).exists()

    async def test_delete_and_metadata_404_on_an_unknown_dataset(self, client):
        """FT-62c: the lifecycle routes report absence consistently."""
        deleted = await client.delete("/upload/dataset/ghost")
        metadata = await client.get("/upload/dataset/ghost/metadata")

        assert deleted.status_code == 404
        assert deleted.json()["detail"] == "Dataset 'ghost' not found"
        assert metadata.status_code == 404
        assert metadata.json()["detail"] == "Dataset 'ghost' not found"

    @pytest.mark.security
    async def test_every_dataset_endpoint_is_session_scoped(self, client):
        """FT-63: FR-2 isolation -- another session sees nothing and can change nothing."""
        owner_sid = await _sid(client)
        await _create_with_files(client, "speech", ("a.wav",))

        async with AsyncClient(app=app, base_url="http://intruder") as intruder:
            intruder_sid = await _sid(intruder)
            assert intruder_sid != owner_sid

            assert (await intruder.get("/upload/dataset/list")).json()["total_datasets"] == 0
            assert (await intruder.get("/upload/dataset/speech/metadata")).status_code == 404
            assert (await intruder.get("/upload/dataset/speech/files")).status_code == 404
            assert (await intruder.get("/upload/dataset/speech/labels")).status_code == 404
            assert (await intruder.delete("/upload/dataset/speech")).status_code == 404

        # The owner's dataset is untouched by all of that.
        assert (await client.get("/upload/dataset/speech/files")).json()["total_files"] == 1
