"""
Masking and decoder-prompt construction — the parts of the model wiring that can
be checked without loading a model.

`mask_spans` is where a saliency timeline becomes an actual edit to the
waveform, and `_decoder_prompt_candidates` is where a Transformers version change
would silently break Whisper scoring. Both are worth pinning down: a fault in
either produces plausible-looking numbers rather than an error.
"""

import numpy as np
import pytest

from app.services.faithfulness_runner import (
    MASK_NOISE_FLOOR,
    SAMPLE_RATE,
    _decoder_prompt_candidates,
    mask_spans,
)


class TestMaskSpans:
    def _audio(self, seconds=2.0):
        return np.ones(int(seconds * SAMPLE_RATE), dtype=np.float32)

    def test_masks_exactly_the_requested_window(self):
        masked = mask_spans(self._audio(), [(0.5, 1.0)])

        assert np.abs(masked[: SAMPLE_RATE // 2]).min() == 1.0
        assert np.abs(masked[SAMPLE_RATE // 2 : SAMPLE_RATE]).max() < MASK_NOISE_FLOOR * 10
        assert np.abs(masked[SAMPLE_RATE:]).min() == 1.0

    def test_uses_a_noise_floor_not_silence(self):
        """Digital silence makes Whisper emit an empty chunk and crash its
        timestamp extractor -- see `pertubation_service.apply_time_masking`."""
        masked = mask_spans(self._audio(), [(0.0, 2.0)])

        assert not np.all(masked == 0.0)
        assert np.abs(masked).max() < 0.01

    def test_leaves_the_input_untouched(self):
        """The clean waveform is reused for every one of the ~70 masked passes."""
        audio = self._audio()
        mask_spans(audio, [(0.0, 1.0)])

        assert audio.min() == 1.0

    def test_no_spans_is_a_no_op(self):
        audio = self._audio()
        assert mask_spans(audio, []) is audio

    def test_out_of_range_spans_are_clipped(self):
        masked = mask_spans(self._audio(1.0), [(-5.0, 0.1), (0.9, 99.0)])

        assert masked.size == SAMPLE_RATE
        assert np.abs(masked[SAMPLE_RATE // 2]).min() == 1.0

    def test_length_is_preserved(self):
        audio = self._audio()
        assert mask_spans(audio, [(0.2, 0.4), (1.1, 1.3)]).size == audio.size


class _FakeTokenizer:
    """Stands in for a WhisperTokenizer: only the id lookups matter here."""

    prefix_tokens = [50258, 50363]
    _ids = {"<|transcribe|>": 50359, "<|notimestamps|>": 50363, "<|en|>": 50259}

    def convert_tokens_to_ids(self, token):
        return self._ids.get(token, -1)


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


class _FakeConfig:
    decoder_start_token_id = 50258


class _FakeModel:
    config = _FakeConfig()

    def __init__(self, detected=None):
        self._detected = detected

    def detect_language(self, features):
        if self._detected is None:
            raise RuntimeError("not supported on this build")
        # Real builds return a tensor; the caller indexes it as `[0, -1]`, which
        # a plain list would not support.
        return np.array([[self._detected]])


class TestDecoderPrompts:
    def test_detected_language_is_offered_first(self):
        candidates = _decoder_prompt_candidates(_FakeModel(50265), _FakeProcessor(), None)

        assert candidates[0] == [50258, 50265, 50359, 50363]

    def test_falls_back_when_language_detection_is_unavailable(self):
        """A build without `detect_language` must still produce a full prompt."""
        candidates = _decoder_prompt_candidates(_FakeModel(None), _FakeProcessor(), None)

        assert [50258, 50259, 50359, 50363] in candidates

    def test_always_offers_the_tokenizer_prefix_and_a_bare_start(self):
        candidates = _decoder_prompt_candidates(_FakeModel(50259), _FakeProcessor(), None)

        assert [50258, 50363] in candidates       # tokenizer.prefix_tokens
        assert [50258] in candidates              # last-resort bare start token

    def test_candidates_are_unique(self):
        """Duplicates would just pay for the same forward pass twice."""
        candidates = _decoder_prompt_candidates(_FakeModel(50259), _FakeProcessor(), None)

        assert len(candidates) == len({tuple(candidate) for candidate in candidates})
