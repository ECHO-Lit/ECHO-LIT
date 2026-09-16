"""FR-10 group partitioning and design classification.

Test Plan Section 3.1.2.  Implements docs/FR10plan.md Part 1 S9 tests #1-#5,
which the plan's own file manifest (FR10plan.md:63) reserved this filename for
and which were never written.

SRS FR-10 acceptance criterion: "The analysis reports per-group error rates and
flags groups with significantly weaker performance or grounding."  Everything
downstream of that depends on the partition being right, so this module asserts
the partition itself against the three bundled corpora, each of which exercises
a different comparison design:

  saa           matched            one shared passage, every group reads it
  l2-arctic     partially_matched  overlapping but unequal utterance sets
  common-voice  unmatched          no shared content at all

Backend/data/ is gitignored, so the module skips rather than errors when the
corpora are absent.
"""

from __future__ import annotations

import pytest

from app.services.fairness_service import FairnessInputError, build_index
from tests._corpora import requires_corpora

pytestmark = [pytest.mark.critical, requires_corpora("saa", "l2-arctic", "common-voice")]


def _groups(index) -> dict[str, int]:
    return {group.label: len(group.items) for group in index.groups}


def test_saa_is_classified_as_a_matched_design():
    """FR10plan S9 #1: every speaker reads the same passage, so content is controlled."""
    index = build_index("saa", ["native_language"], None, task="transcription")

    assert index.design == "matched"
    assert index.content_column == "reading_passage"
    assert index.speaker_column == "speakerid"
    evidence = index.design_evidence
    # One passage, shared by all five groups: perfect overlap.
    assert evidence["jaccard"] == 1.0
    assert evidence["n_shared_content"] == 1
    assert set(evidence["per_group_coverage"].values()) == {1.0}
    assert _groups(index) == {
        "arabic": 30, "english": 30, "mandarin": 30, "russian": 30, "spanish": 30
    }
    assert index.excluded == []


def test_l2_arctic_is_classified_as_partially_matched():
    """FR10plan S9 #2: overlapping utterance sets, so the design is partial.

    The plan doc records "33 shared utt_id" for this corpus; the bundled subset
    actually yields 18 of 51.  The number here is measured, and the plan's is
    filed as a documentation defect (DOC-01) rather than silently matched.
    """
    index = build_index("l2-arctic", ["native_language"], None, task="transcription")

    assert index.design == "partially_matched"
    assert index.content_column == "utt_id"
    evidence = index.design_evidence
    assert evidence["n_shared_content"] == 18
    assert evidence["n_total_content"] == 51
    assert evidence["jaccard"] == pytest.approx(0.3529, abs=1e-4)
    assert _groups(index) == {"arabic": 51, "chinese": 51, "spanish": 48}


def test_common_voice_is_unmatched_and_documents_its_exclusions():
    """FR10plan S9 #3: no shared content, and thin groups are dropped explicitly."""
    index = build_index("common-voice", ["accent"], None, task="transcription")

    assert index.design == "unmatched"
    assert index.design_evidence["n_shared_content"] == 0
    assert _groups(index) == {"england": 10, "us": 18}

    # Exclusions are reported, not silently applied: a user must be able to see
    # that two thirds of the corpus was set aside and why.
    reasons = {entry["label"]: entry["reason"] for entry in index.excluded}
    assert reasons["<unlabelled>"] == "missing_group_label"
    assert reasons["australia"] == "below_min_group_size"
    assert reasons["indian"] == "below_min_group_size"
    unlabelled = next(e for e in index.excluded if e["label"] == "<unlabelled>")
    assert unlabelled["n_items"] == 67
    # An excluded label must not also appear as a comparison group.
    assert set(reasons) & set(_groups(index)) == set()


def test_the_speaker_confound_gate_flags_single_speaker_groups():
    """FR10plan S9 #4: a group that is one speaker cannot separate accent from voice.

    test_fr10_dataset_extensions.py documents the corpus fact that L2-ARCTIC has
    one speaker per language; this exercises the gate that acts on it.
    """
    l2 = build_index("l2-arctic", ["native_language"], None, task="transcription")

    assert all(group.speaker_confounded for group in l2.groups)
    assert all(group.n_speakers == 1 for group in l2.groups)

    # SAA has 30 speakers per group, so the same gate stays clear.
    saa = build_index("saa", ["native_language"], None, task="transcription")
    assert not any(group.speaker_confounded for group in saa.groups)

    # Common Voice has no usable speaker column at all -- confounded for a
    # different reason, which the count distinguishes from L2-ARCTIC's.
    cv = build_index("common-voice", ["accent"], None, task="transcription")
    assert cv.speaker_column is None
    assert all(group.speaker_confounded and group.n_speakers == 0 for group in cv.groups)


def test_fewer_than_two_valid_groups_is_rejected_with_a_reason():
    """FR10plan S9 #5: an impossible comparison fails at the door, not mid-run."""
    with pytest.raises(FairnessInputError) as exc:
        build_index("common-voice", ["accent"], None, task="transcription", min_group_size=15)

    assert "Only 1 group(s) meet the minimum size of 15" in str(exc.value)
    assert "A fairness comparison needs at least two." in str(exc.value)


def test_an_absent_grouping_column_is_rejected():
    """Guard: a typo in the grouping key must not silently produce one group."""
    with pytest.raises(FairnessInputError) as exc:
        build_index("saa", ["nope"], None, task="transcription")

    assert str(exc.value) == "Column 'nope' is not present in dataset 'saa'."


def test_subsampling_is_deterministic_for_a_fixed_seed():
    """PE/RE: two identical requests must partition identically, or nothing
    downstream of the partition is reproducible."""
    kwargs = dict(task="transcription", max_items_per_group=10, seed=0)
    first = build_index("saa", ["native_language"], None, **kwargs)
    second = build_index("saa", ["native_language"], None, **kwargs)

    assert _groups(first) == _groups(second)
    for left, right in zip(first.groups, second.groups):
        assert [item.filename for item in left.items] == [
            item.filename for item in right.items
        ]
