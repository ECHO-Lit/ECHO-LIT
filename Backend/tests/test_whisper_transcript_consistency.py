"""The word-timed transcript must say exactly what the prediction says.

Saliency and attention label their segments with words from
`transcribe_whisper(..., return_timestamps=True)`; the right-hand panel shows
`transcribe_whisper(...)`. The timed path used to decode with Whisper's
timestamp tokens switched on, which changes what greedy decoding picks, so on
perturbed audio one panel read "I will be over there." and the other
"I will be overexposed.". Both paths now make the same decoding call, and word
timing is aligned afterwards from cross-attention without re-decoding.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.services import model_loader_service as loader


# --------------------------------------------------------------------------
# A Whisper stand-in whose decode and cross-attention are fully known
# --------------------------------------------------------------------------

PIECES = {1: "ĠI", 2: "Ġwill", 3: "Ġbe", 4: "Ġover", 5: "exposed", 6: "."}
SPECIAL = {
    "<|startoftranscript|>": 50258, "<|en|>": 50259,
    "<|transcribe|>": 50360, "<|notimestamps|>": 50364,
}
EOS = 50257
PREFIX = list(SPECIAL.values())
TEXT = [1, 2, 3, 4, 5, 6]
FRAMES_PER_TOKEN = 20  # each text token "sounds" for 0.4 s of encoder frames
NUM_SAMPLES = (len(TEXT) + 2) * FRAMES_PER_TOKEN * 2 * 160


class FakeTokenizer:
    eos_token_id = EOS

    def convert_ids_to_tokens(self, token):
        return PIECES.get(int(token), "<|special|>")

    def convert_tokens_to_ids(self, tokens):
        return [SPECIAL[token] for token in tokens]

    def decode(self, ids, skip_special_tokens=False):
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
        return "".join(PIECES[i].replace("Ġ", " ") for i in ids if i in PIECES)


class FakeProcessor:
    tokenizer = FakeTokenizer()
    feature_extractor = SimpleNamespace(hop_length=160)

    def __call__(self, audio, sampling_rate, return_tensors):
        return SimpleNamespace(input_features=torch.zeros(1, 80, 3000))


class FakeWhisper:
    device = torch.device("cpu")
    config = SimpleNamespace(decoder_layers=1, decoder_attention_heads=1, median_filter_width=7)
    generation_config = SimpleNamespace(alignment_heads=[[0, 0]])

    def __init__(self):
        self.generate_calls = []

    def generate(self, input_features, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[*PREFIX, *TEXT, EOS]])

    def __call__(self, input_features, decoder_input_ids, output_attentions, return_dict):
        rows = decoder_input_ids.shape[-1]
        frames = torch.arange(1500, dtype=torch.float32)
        attention = torch.full((rows, 1500), 1e-4)
        # Row p predicts token p+1: text token k is predicted by row
        # len(PREFIX) - 1 + k and attends to the middle of its 0.4 s slot.
        for k in range(len(TEXT) + 1):
            centre = k * FRAMES_PER_TOKEN + FRAMES_PER_TOKEN / 2
            attention[len(PREFIX) - 1 + k] = torch.exp(-((frames - centre) ** 2) / 20.0)
        return SimpleNamespace(cross_attentions=(attention[None, None],))


@pytest.fixture
def fake_whisper(monkeypatch):
    model = FakeWhisper()
    monkeypatch.setattr(loader, "get_whisper_gen_model", lambda model_id: model)
    monkeypatch.setattr(loader.WhisperProcessor, "from_pretrained", lambda *a, **k: FakeProcessor())
    audio = np.random.default_rng(0).standard_normal(NUM_SAMPLES).astype(np.float32) * 0.1
    monkeypatch.setattr(loader.librosa, "load", lambda path, sr: (audio.copy(), sr))
    return model


class TestSameDecodeForPredictionAndTimedTranscript:
    def test_both_paths_make_the_same_decoding_call(self, fake_whisper):
        loader.transcribe_whisper("openai/whisper-base", "clip.wav")
        loader.transcribe_whisper("openai/whisper-base", "clip.wav", return_timestamps=True)

        prediction_call, timed_call = fake_whisper.generate_calls
        assert timed_call == prediction_call
        # The switch that made them disagree.
        assert "return_timestamps" not in timed_call

    def test_timed_words_are_the_prediction_words(self, fake_whisper):
        prediction = loader.transcribe_whisper("openai/whisper-base", "clip.wav")
        timed = loader.transcribe_whisper("openai/whisper-base", "clip.wav", return_timestamps=True)

        assert prediction.strip() == "I will be overexposed."
        assert timed["text"] == prediction.strip()
        # Sub-word pieces and punctuation join the word they belong to.
        assert [chunk["text"] for chunk in timed["chunks"]] == ["I", "will", "be", "overexposed."]

    def test_word_timings_come_from_the_audio_not_an_even_split(self, fake_whisper):
        timed = loader.transcribe_whisper("openai/whisper-base", "clip.wav", return_timestamps=True)

        slot = FRAMES_PER_TOKEN / 50  # seconds per token in the stand-in
        starts = [chunk["timestamp"][0] for chunk in timed["chunks"]]
        ends = [chunk["timestamp"][1] for chunk in timed["chunks"]]
        assert starts == sorted(starts) and all(end > start for start, end in zip(starts, ends))
        # "overexposed." is three tokens: it spans three slots, not one quarter
        # of the clip as the old even split gave every word.
        overexposed = timed["chunks"][3]["timestamp"]
        assert overexposed[1] - overexposed[0] == pytest.approx(3 * slot, abs=slot / 2)
        assert overexposed[0] == pytest.approx(3 * slot, abs=slot / 2)


class TestDynamicTimeWarping:
    def test_follows_the_cheapest_monotonic_path(self):
        cost = np.ones((3, 6))
        for token, frames in enumerate([(0, 2), (2, 4), (4, 6)]):
            cost[token, frames[0]:frames[1]] = 0.0

        text_indices, time_indices = loader._dynamic_time_warping(cost)

        assert list(np.diff(text_indices) >= 0) == [True] * (len(text_indices) - 1)
        assert list(np.diff(time_indices) >= 0) == [True] * (len(time_indices) - 1)
        first_frame = {int(t): int(f) for t, f in reversed(list(zip(text_indices, time_indices)))}
        assert first_frame == {0: 0, 1: 2, 2: 4}


# --------------------------------------------------------------------------
# The real model, when it is already in the local Hugging Face cache
# --------------------------------------------------------------------------

CLIPS = Path(__file__).resolve().parents[1] / "data" / "common_voice_valid_dev"


def _cached_whisper_base():
    try:
        loader.WhisperProcessor.from_pretrained("openai/whisper-base", local_files_only=True)
    except Exception:
        return False
    return True


@pytest.mark.slow
@pytest.mark.skipif(not CLIPS.is_dir(), reason="bundled Common Voice clips not present")
def test_real_whisper_base_agrees_on_clean_and_noisy_speech(tmp_path):
    """Measured before the fix: the timestamp decode disagreed on 5 of 36
    clean/noisy clips, including this set's clean sample-000037
    ("Mines in the door." vs "Minds in the door.")."""
    if not _cached_whisper_base():
        pytest.skip("openai/whisper-base is not in the local Hugging Face cache")
    import librosa
    import soundfile

    rng = np.random.default_rng(0)
    for name in ("sample-000037.mp3", "sample-000454.mp3"):
        audio, _ = librosa.load(CLIPS / name, sr=16000)
        noise = rng.standard_normal(audio.shape).astype(np.float32)
        for label, wave in (("clean", audio), ("snr5", audio + noise * np.sqrt(np.mean(audio**2) / 10**0.5))):
            path = tmp_path / f"{name}-{label}.wav"
            soundfile.write(path, wave, 16000)
            prediction = loader.transcribe_whisper("openai/whisper-base", str(path)).strip()
            timed = loader.transcribe_whisper("openai/whisper-base", str(path), return_timestamps=True)
            words = " ".join(chunk["text"] for chunk in timed["chunks"])
            assert timed["text"] == prediction, (name, label)
            assert words == prediction, (name, label)
