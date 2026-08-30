from types import SimpleNamespace

import pytest


def test_jacobian_lens_job_contract_requires_paired_owned_audio():
    from pydantic import ValidationError

    from app.schemas.jobs import JobCreateRequest

    request = JobCreateRequest(
        operation="jacobian_lens_fit",
        model="whisper-base",
        audio_ids=["audio-a", "audio-b"],
        parameters={
            "samples": [
                {"audio_id": "audio-a", "transcript": "first sample"},
                {"audio_id": "audio-b", "transcript": "second sample"},
            ],
        },
    )
    assert request.parameters["probe_count"] == 4
    assert request.parameters["samples"][0]["transcript"] == "first sample"

    with pytest.raises(ValidationError):
        JobCreateRequest(
            operation="jacobian_lens_fit",
            model="whisper-base",
            audio_ids=["audio-a"],
            parameters={
                "samples": [
                    {"audio_id": "audio-a", "transcript": "one"},
                    {"audio_id": "audio-b", "transcript": "two"},
                ],
            },
        )


def test_apply_parameters_reject_removed_bucketing():
    from pydantic import ValidationError

    from app.schemas.jobs import JacobianLensApplyParameters

    parameters = JacobianLensApplyParameters(lens_id="jlens-1")
    assert parameters.top_k == 5
    assert parameters.transcript is None
    assert parameters.max_new_tokens == 64

    with pytest.raises(ValidationError):
        JacobianLensApplyParameters(lens_id="jlens-1", max_frames=96)


def test_speech_adapters_expose_jacobian_lens_architecture():
    from app.core.model_catalog import ModelKind
    from app.schemas.jobs import RuntimeModelSpec
    from app.worker.model_adapters import get_model_adapter

    assert get_model_adapter("whisper-base").jacobian_lens_architecture() == "decoder"
    assert get_model_adapter("wav2vec2").jacobian_lens_architecture() is None
    ctc = get_model_adapter(
        "custom-ctc",
        RuntimeModelSpec(
            hf_repo="org/ctc-model",
            kind=ModelKind.CTC_ASR,
            capabilities=["prediction"],
        ),
    )
    assert ctc.jacobian_lens_architecture() is None
    assert ctc.jacobian_lens_revision() == "org/ctc-model@main"


@pytest.mark.asyncio
async def test_jacobian_lens_repository_is_session_owned():
    from datetime import datetime, timezone

    from app.repositories.jacobian_lenses import JacobianLensRepository
    from app.schemas.jacobian_lens import JacobianLensRecord

    repository = JacobianLensRepository()
    now = datetime.now(timezone.utc)
    record = JacobianLensRecord(
        lens_id="jlens-test",
        session_id="session-a",
        model_id="whisper-base",
        model_revision="openai/whisper-base",
        fit_job_id="job-a",
        created_at=now,
        updated_at=now,
        sample_count=2,
    )
    await repository.create(record)
    assert await repository.get_owned("jlens-test", "session-a") == record
    assert await repository.get_owned("jlens-test", "session-b") is None


@pytest.mark.asyncio
async def test_fit_job_creates_session_owned_lens_and_uses_single_worker_job(client, sample_audio_file, monkeypatch):
    from app.api.routes import jobs as jobs_routes
    from app.repositories.jobs import JobRepository

    audio_ids = []
    for name in ("first.wav", "second.wav"):
        with sample_audio_file.open("rb") as handle:
            upload = await client.post("/upload", files={"file": (name, handle, "audio/wav")})
        assert upload.status_code == 201
        audio_ids.append(upload.json()["audio_id"])

    sent = []
    monkeypatch.setattr(
        jobs_routes.celery_app,
        "send_task",
        lambda *args, **kwargs: sent.append((args, kwargs)) or SimpleNamespace(id="fit-task"),
    )
    response = await client.post("/jobs", json={
        "operation": "jacobian_lens_fit",
        "model": "whisper-base",
        "audio_ids": audio_ids,
        "parameters": {"samples": [
            {"audio_id": audio_ids[0], "transcript": "first transcript"},
            {"audio_id": audio_ids[1], "transcript": "second transcript"},
        ]},
    })
    assert response.status_code == 202
    record = await JobRepository().get(response.json()["job_id"])
    assert record.parameters["lens_id"].startswith("jlens-")
    assert sent[0][0][0] == "app.worker.tasks.execute_job"
    assert sent[0][1]["queue"] == "gpu-large"


class _TinyDecoder:
    """Position-wise linear decoder standing in for a Whisper decoder stack.

    hidden_0 = embed(ids) + encoder frames (the "embedding output" state)
    hidden_1 = hidden_0 + W1 hidden_0          (one transformer block)
    final    = W2 hidden_1                     (the pre-logit readout state)
    """

    def __init__(self, dim):
        import torch

        self.dim = dim
        self.embed = torch.nn.Embedding(16, dim)
        self.block = torch.nn.Linear(dim, dim, bias=False)
        self.head = torch.nn.Linear(dim, dim, bias=False)
        torch.nn.init.normal_(self.block.weight, std=0.3)
        torch.nn.init.normal_(self.head.weight, std=0.3)

    def __call__(self, input_ids, encoder_hidden_states, output_hidden_states, use_cache, return_dict):

        token_part = self.embed(input_ids)
        hidden0 = token_part + encoder_hidden_states[:, : token_part.shape[1], :]
        hidden1 = hidden0 + self.block(hidden0)
        final = self.head(hidden1)
        return SimpleNamespace(last_hidden_state=final, hidden_states=(hidden0, hidden1, final))


class _TinyEncoder:
    def __init__(self, dim, frames):
        # Deterministic "audio" so fit and apply see identical conditions.
        import torch

        grid = torch.linspace(-1.0, 1.0, frames * dim)
        self.frame = grid.reshape(frames, dim)

    def __call__(self, **kwargs):
        features = kwargs["input_features"]
        batch = features.shape[0]
        return SimpleNamespace(last_hidden_state=self.frame.repeat(batch, 1, 1))


class _TinyModel:
    def __init__(self, dim=8, frames=5, vocab=13):
        import torch

        self.dim = dim
        self.config = SimpleNamespace(decoder_start_token_id=1)
        self.encoder = _TinyEncoder(dim, frames)
        self.decoder = _TinyDecoder(dim)
        self.output_head = torch.nn.Linear(dim, vocab, bias=False)
        torch.nn.init.normal_(self.output_head.weight, std=0.5)

    def parameters(self):
        return iter([self.decoder.head.weight])

    def eval(self):
        return self

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.output_head

    def prepare_decoder_input_ids_from_labels(self, labels):
        return _TinyModel.shift_right(labels, self.config.decoder_start_token_id)

    def generate(self, **kwargs):
        import torch

        return self.prepare_decoder_input_ids_from_labels(torch.tensor([[5, 6, 7]]))

    @staticmethod
    def shift_right(labels, start_id):
        import torch

        start = torch.full((labels.shape[0], 1), start_id, dtype=labels.dtype)
        return torch.cat([start, labels[:, :-1]], dim=1)


class _TinyTokenizer:
    def __call__(self, transcript, return_tensors="pt", truncation=False, max_length=None):
        import torch

        ids = [2 + (ord(character) % 11) for character in transcript]
        if max_length is not None:
            ids = ids[:max_length]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def convert_ids_to_tokens(self, token_id):
        return f"tok_{token_id}"


class _TinyProcessor:
    def __init__(self):
        self.feature_extractor = SimpleNamespace(sampling_rate=16000)
        self.tokenizer = _TinyTokenizer()

    def __call__(self, audio, sampling_rate=None, return_tensors="pt"):
        import torch

        return SimpleNamespace(input_features=torch.zeros(1, 4, 8))

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [" ".join(f"tok_{token_id}" for token_id in token_ids[0])]


class _TinyAdapter:
    model_id = "tiny-stub"

    @staticmethod
    def jacobian_lens_architecture():
        return "decoder"

    @staticmethod
    def jacobian_lens_revision():
        return "tiny@main"

    @staticmethod
    def jacobian_lens_components(resource):
        return resource


def test_decoder_jacobian_lens_fit_matches_linear_closed_form(monkeypatch):
    """For a linear decoder the averaged causal Jacobian has a closed form.

    With a position-wise (no cross-position) map M, only diagonal (t, t') pairs
    carry signal, so the triangular mean over t' >= t is exactly 2M/(T+1).
    """
    pytest.importorskip("torch")
    import torch

    from app.services import jacobian_lens_service as service

    model = _TinyModel()
    processor = _TinyProcessor()
    monkeypatch.setattr(
        service,
        "_prepare_audio",
        lambda *_args, **_kwargs: ({"input_features": torch.zeros(1, 4, model.dim)}, 1.0),
    )

    identity = torch.eye(model.dim)
    layer0_map = model.decoder.head.weight @ (identity + model.decoder.block.weight)
    layer1_map = model.decoder.head.weight
    positions = 3  # both transcripts tokenize to three tokens
    expected_matrices = [
        2.0 * layer0_map / (positions + 1),
        2.0 * layer1_map / (positions + 1),
    ]

    artifact = service.fit_decoder_jacobian_lens(
        _TinyAdapter(),
        (processor, model),
        samples=[("unused.wav", "abc"), ("unused.wav", "def")],
        probe_count=256,
        max_audio_seconds=30.0,
    )
    assert artifact["format_version"] == 2
    assert artifact["architecture"] == "decoder"
    assert artifact["sample_count"] == 2
    assert artifact["probe_count"] == 256
    assert "baselines" not in artifact
    assert len(artifact["matrices"]) == 2
    for matrix, expected in zip(artifact["matrices"], expected_matrices):
        assert matrix.shape == (model.dim, model.dim)
        deviation = (matrix - expected).abs().max()
        scale = expected.abs().max()
        assert deviation / scale < 0.15


def test_decoder_jacobian_lens_apply_ranks_through_model_head(monkeypatch):
    pytest.importorskip("torch")
    import torch

    from app.services import jacobian_lens_service as service

    model = _TinyModel()
    processor = _TinyProcessor()
    monkeypatch.setattr(
        service,
        "_prepare_audio",
        lambda *_args, **_kwargs: ({"input_features": torch.zeros(1, 4, model.dim)}, 1.0),
    )
    artifact = service.fit_decoder_jacobian_lens(
        _TinyAdapter(),
        (processor, model),
        samples=[("unused.wav", "hi"), ("unused.wav", "yo")],
        probe_count=64,
        max_audio_seconds=30.0,
    )

    provided = service.apply_decoder_jacobian_lens(
        _TinyAdapter(), (processor, model), artifact, "unused.wav",
        top_k=3, transcript="abcd",
    )
    assert provided["architecture"] == "decoder"
    assert provided["transcript_source"] == "provided"
    assert len(provided["layers"]) == 2
    assert len(provided["positions"]) == 4
    assert provided["positions"][0]["token_id"] == 1  # decoder start token leads
    for layer in provided["layers"]:
        assert len(layer["positions"]) == 4
        for cell in layer["positions"]:
            scores = [token["score"] for token in cell["tokens"]]
            probabilities = [token["probability"] for token in cell["tokens"]]
            assert len(cell["tokens"]) == 3
            assert scores == sorted(scores, reverse=True)
            assert all(0.0 <= probability <= 1.0 for probability in probabilities)

    # The readout must be exactly (J h) @ E.T: recompute one cell independently.
    labels = processor.tokenizer("abcd").input_ids
    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
    encoder_states = model.encoder.frame.unsqueeze(0)
    hidden0 = model.decoder.embed(decoder_input_ids) + encoder_states[:, :4, :]
    transported = hidden0[0, 2] @ artifact["matrices"][0].T
    logits = transported @ model.output_head.weight.T
    top1 = provided["layers"][0]["positions"][2]["tokens"][0]
    assert int(logits.argmax()) == top1["token_id"]

    generated = service.apply_decoder_jacobian_lens(
        _TinyAdapter(), (processor, model), artifact, "unused.wav", top_k=3,
    )
    assert generated["transcript_source"] == "generated"
    assert len(generated["positions"]) == 3
