"""Performance Profiling -- worker-side computation that scales with the data.

Test Plan Section 3.1.4.  See tests/plans/3.1.4-performance-profiling.md.

The control plane's budget is PE-1's 500 ms.  Worker computation has no SRS
number of its own, but PE-3 requires that "long-running tasks are bounded", and
FR-10's statistics are pure CPU with no model in the loop -- so their cost is
the product's own and is profiled here at the sizes the API accepts:

  FairnessRequest.max_items_per_group  <= 2000
  FairnessRequest.n_bootstrap          <= 20000

The bootstrap is also subject to RE-3 ("analytical results shall be
reproducible for identical inputs and configuration").  Any speed-up must
therefore reproduce the previous numbers, and TestBootstrapEquivalence holds the
pre-fix implementation verbatim as its oracle.
"""

from __future__ import annotations

import time
import zlib

import numpy as np
import pytest

from app.services.fairness_metrics_service import (
    ItemMetric,
    _block_map,
    _percentile_ci,
    _two_sided_p,
    estimate_gap,
    holm_bonferroni,
)
from app.worker.executor import _aggregate_batch

pytestmark = [pytest.mark.performance, pytest.mark.critical]

MAX_ITEMS_PER_GROUP = 2000
MAX_BOOTSTRAP = 20000


def _items(group: str, n: int, *, speakers: int = 0, contents: int = 0, seed: int = 0) -> list[ItemMetric]:
    # zlib.crc32, not hash(): str hashes are salted per process, which would
    # make every run draw different data.
    rng = np.random.default_rng(zlib.crc32(f"{group}|{n}|{speakers}|{contents}|{seed}".encode()))
    return [
        ItemMetric(
            item_id=f"{group}-{i}",
            speaker_id=f"{group}-spk{i % speakers}" if speakers else None,
            content_id=f"c{i % contents}" if contents else None,
            value=float(rng.random()),
        )
        for i in range(n)
    ]


def _reference_unpaired(ref_items, other_items, *, n_boot: int, seed: int, alpha: float = 0.05):
    """The pre-fix unpaired bootstrap, verbatim from fairness_metrics_service.

    Kept here, not imported, so the oracle cannot drift with the code it checks.
    """
    rng = np.random.default_rng(seed)
    use_speaker = any(i.speaker_id for i in ref_items) and any(i.speaker_id for i in other_items)
    ref_blocks = _block_map(ref_items, use_speaker)
    other_blocks = _block_map(other_items, use_speaker)
    ref_keys, other_keys = list(ref_blocks.keys()), list(other_blocks.keys())
    boot = np.empty(n_boot)
    for k in range(n_boot):
        rk = [ref_keys[i] for i in rng.integers(0, len(ref_keys), size=len(ref_keys))]
        ok = [other_keys[i] for i in rng.integers(0, len(other_keys), size=len(other_keys))]
        rv = np.concatenate([ref_blocks[b] for b in rk])
        ov = np.concatenate([other_blocks[b] for b in ok])
        boot[k] = ov.mean() - rv.mean()
    ref_vals = np.array([i.value for i in ref_items])
    other_vals = np.array([i.value for i in other_items])
    point = float(other_vals.mean() - ref_vals.mean())
    return _percentile_ci(boot, alpha), _two_sided_p(boot, point)


def _timed(fn):
    started = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - started


class TestBootstrapCost:
    def test_unpaired_bootstrap_at_the_group_cap_is_subsecond(self):
        """PP-25: guards BUG-41 -- one (metric, group) gap at max_items_per_group.

        Measured before the fix: 1.2 s at 100 items, 9.9 s at 1 000 and 32.6 s at
        3 000 per call, because every resample built Python lists of block keys
        and concatenated Python lists of floats.  A fairness job calls it once
        per metric per non-reference group -- five groups and two metrics at
        1 000 items was ~100 s of pure statistics before any model ran.
        """
        ref = _items("ref", MAX_ITEMS_PER_GROUP)
        other = _items("other", MAX_ITEMS_PER_GROUP)

        gap, seconds = _timed(lambda: estimate_gap(ref, other, metric="wer", design="unmatched"))

        assert gap["method"] == "unpaired" and gap["block_unit"] == "utterance"
        assert seconds < 1.0, f"{seconds:.2f} s"

    def test_the_largest_requestable_bootstrap_stays_bounded(self):
        """PP-26: PE-3 "long-running tasks are bounded" at n_bootstrap = 20 000.

        The request schema allows ten times the default resample count.  Before
        the fix this configuration extrapolated to ~200 s per gap at 2 000 items.
        """
        ref = _items("ref", MAX_ITEMS_PER_GROUP)
        other = _items("other", MAX_ITEMS_PER_GROUP)

        _, seconds = _timed(
            lambda: estimate_gap(ref, other, metric="wer", design="unmatched", n_boot=MAX_BOOTSTRAP)
        )

        assert seconds < 10.0, f"{seconds:.2f} s"

    def test_paired_bootstrap_at_the_group_cap_is_subsecond(self):
        """PP-27: positive control -- the paired path was already vectorised per resample."""
        ref = _items("ref", MAX_ITEMS_PER_GROUP, contents=400)
        other = _items("other", MAX_ITEMS_PER_GROUP, contents=400)

        gap, seconds = _timed(lambda: estimate_gap(ref, other, metric="wer", design="matched"))

        assert gap["method"] == "paired_by_content"
        assert seconds < 1.0, f"{seconds:.2f} s"


class TestBootstrapEquivalence:
    @pytest.mark.parametrize("seed", [0, 1, 7])
    @pytest.mark.parametrize(
        "speakers", [0, 12], ids=["utterance blocks", "speaker blocks"]
    )
    def test_the_faster_bootstrap_reproduces_the_previous_numbers(self, seed, speakers):
        """PP-28: RE-3 -- same inputs and seed give the same interval and p-value.

        Speaker blocks have unequal sizes, which is the case a naive
        vectorisation (mean of block means) would get wrong.  The fix keeps the
        exact random draw sequence, so the only permitted difference is float
        summation order.
        """
        ref = _items("ref", 300, speakers=speakers, seed=seed)
        other = _items("other", 260, speakers=speakers, seed=seed + 100)
        if speakers:
            # Unequal block sizes: drop a varying number of items per speaker.
            ref = [item for index, item in enumerate(ref) if index % 7 != seed % 7]

        expected_ci, expected_p = _reference_unpaired(ref, other, n_boot=2000, seed=seed)
        gap = estimate_gap(ref, other, metric="wer", design="unmatched", seed=seed)

        assert gap["ci"] == pytest.approx(expected_ci, abs=1e-12)
        assert gap["p_value"] == expected_p


class TestAggregationCost:
    def test_holm_bonferroni_scales_to_many_comparisons(self):
        """PP-29: O(m log m) -- 10 000 p-values, far beyond any real group count."""
        p_values = list(np.random.default_rng(0).random(10_000))

        adjusted, seconds = _timed(lambda: holm_bonferroni(p_values))

        assert len(adjusted) == 10_000
        assert seconds < 0.1, f"{seconds:.3f} s"

    @pytest.mark.parametrize(
        ("operation", "model", "item"),
        [
            ("prediction", "wav2vec2", {"predicted_emotion": "neutral", "confidence": 0.9}),
            ("prediction", "whisper-base", {"text": "the quick brown fox jumps over the lazy dog"}),
            ("audio_features", None, {f"feature_{k}": float(k) for k in range(40)}),
        ],
        ids=["wav2vec2", "whisper", "audio-features"],
    )
    def test_batch_summary_at_the_batch_cap(self, operation, model, item):
        """PP-30: the per-batch summary a 200-file job builds in its finalizer."""
        result = {"operation": operation, "model": model,
                  "items": [{"audio_id": f"a{i}", "result": dict(item)} for i in range(200)]}

        _, seconds = _timed(lambda: _aggregate_batch(result, [f"f{i}.wav" for i in range(200)]))

        assert result["summary"]["total_files"] == 200
        assert seconds < 0.1, f"{seconds:.3f} s"
