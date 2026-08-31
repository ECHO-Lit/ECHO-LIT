"""Custom-dataset label ingestion tests.

The feature this covers is "an uploaded dataset can be probed at all".  The
highest-value test here is `TestPreviewMatchesProbe`: the preview promises the
user what the probe will do *before* they spend minutes of extraction on it, so
if the preview and the probe ever disagree the preview is worse than useless --
it is a confident lie that costs real time.
"""

import csv
import io

import pytest

from app.services.dataset_labels_service import (
    FILENAME_PATTERNS,
    MAX_DISCOVERED_CLASSES,
    MIN_FOLDS,
    MIN_ROWS_FOR_PROBE,
    MISSING_LABELS,
    attach_duration_bands,
    available_patterns,
    band_labels,
    derive_from_filenames,
    label_columns,
    merge_labels,
    normalise_label,
    parse_labels_csv,
    preview_dataset,
    summarise_property,
)


def _rows(labels, column="emotion"):
    """File rows carrying one label column, filenames in order."""
    return [
        {"filename": f"clip_{index:03d}.wav", column: label if label is not None else ""}
        for index, label in enumerate(labels)
    ]


class TestConstantsStayInSync:
    """These are duplicated, not imported -- the API container has no numpy.

    If `probing_service` changes a threshold and this module does not, the
    preview silently starts disagreeing with the probe.  That is exactly the
    failure the preview exists to prevent, so it fails a test instead.
    """

    def test_constants_match_probing_service(self):
        from app.services import probing_service

        assert MISSING_LABELS == probing_service.MISSING_LABELS
        assert MIN_ROWS_FOR_PROBE == probing_service.MIN_ROWS_FOR_PROBE
        assert MIN_FOLDS == probing_service.MIN_FOLDS


class TestNormaliseLabel:
    @pytest.mark.parametrize("value", ["", "unknown", "UNKNOWN", "n/a", "NA", "none", None, "  "])
    def test_missing_markers_become_none(self, value):
        assert normalise_label(value) is None

    def test_real_labels_are_trimmed_and_kept(self):
        assert normalise_label("  anger ") == "anger"
        assert normalise_label(12) == "12"


class TestParseLabelsCsv:
    def test_parses_filename_and_label_columns(self):
        table, warnings = parse_labels_csv(
            "filename,speaker,emotion\nDC_a01.wav,DC,anger\nJE_sa04.wav,JE,sadness\n"
        )
        assert table["DC_a01.wav"] == {"speaker": "DC", "emotion": "anger"}
        assert table["JE_sa04.wav"]["emotion"] == "sadness"
        assert warnings == []

    def test_indexes_by_basename_so_paths_still_join(self):
        """A user's CSV commonly carries the path the corpus shipped with."""
        table, _ = parse_labels_csv("filename,emotion\nAudioData/DC/a01.wav,anger\n")
        assert "a01.wav" in table
        assert table["a01.wav"]["emotion"] == "anger"

    def test_header_is_case_and_space_insensitive(self):
        table, _ = parse_labels_csv(" FileName , Emotion \nx.wav,anger\n")
        assert table["x.wav"] == {"emotion": "anger"}

    def test_missing_values_are_dropped_not_stored_as_a_class(self):
        table, _ = parse_labels_csv(
            "filename,emotion,accent\na.wav,anger,unknown\nb.wav,,us\n"
        )
        assert table["a.wav"] == {"emotion": "anger"}
        assert table["b.wav"] == {"accent": "us"}

    def test_rejects_csv_without_filename_column(self):
        with pytest.raises(ValueError, match="filename"):
            parse_labels_csv("speaker,emotion\nDC,anger\n")

    def test_rejects_csv_with_no_label_columns(self):
        with pytest.raises(ValueError, match="no label columns"):
            parse_labels_csv("filename,duration\na.wav,1.2\n")

    def test_rejects_csv_where_every_row_is_unusable(self):
        with pytest.raises(ValueError, match="No usable rows"):
            parse_labels_csv("filename,emotion\na.wav,unknown\nb.wav,\n")

    def test_reports_duplicate_filenames(self):
        _, warnings = parse_labels_csv("filename,emotion\na.wav,anger\na.wav,sadness\n")
        assert any("duplicate" in warning for warning in warnings)

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            parse_labels_csv("")


class TestFilenamePatterns:
    def test_savee_flat_extracts_speaker_and_emotion(self):
        table, warnings = derive_from_filenames(
            ["DC_a01.wav", "JE_sa04.wav", "KL_su12.wav", "JK_n30.wav"], "savee"
        )
        assert table["DC_a01.wav"] == {"speaker": "DC", "emotion": "anger"}
        # Two-letter codes must beat the one-letter prefix: `sa` is sadness, not
        # `s` + take `a04`. This is the pattern's only real ambiguity.
        assert table["JE_sa04.wav"]["emotion"] == "sadness"
        assert table["KL_su12.wav"]["emotion"] == "surprise"
        assert table["JK_n30.wav"]["emotion"] == "neutral"
        assert warnings == []

    def test_savee_take_number_is_not_probed(self):
        """Take index is one class per clip -- useless as a probe target."""
        assert "take" not in FILENAME_PATTERNS["savee"].properties()
        table, _ = derive_from_filenames(["DC_a01.wav"], "savee")
        assert "take" not in table["DC_a01.wav"]

    def test_savee_nested_takes_speaker_from_the_folder(self):
        table, _ = derive_from_filenames(
            ["AudioData/DC/a01.wav", "AudioData/KL/su12.wav"], "savee-nested"
        )
        assert table["AudioData/DC/a01.wav"] == {"emotion": "anger", "speaker": "DC"}
        assert table["AudioData/KL/su12.wav"]["speaker"] == "KL"

    def test_ravdess_pattern_maps_codes_to_readable_classes(self):
        table, _ = derive_from_filenames(["03-01-05-02-01-01-16.wav"], "ravdess")
        entry = table["03-01-05-02-01-01-16.wav"]
        assert entry["emotion"] == "angry"
        assert entry["intensity"] == "strong"
        assert entry["statement"] == "Kids are talking by the door"
        assert entry["actor"] == "16"

    def test_crema_d_pattern(self):
        table, _ = derive_from_filenames(["1001_DFA_ANG_XX.wav"], "crema-d")
        assert table["1001_DFA_ANG_XX.wav"]["emotion"] == "anger"
        assert table["1001_DFA_ANG_XX.wav"]["actor"] == "1001"

    def test_unmatched_files_are_reported_not_fatal(self):
        table, warnings = derive_from_filenames(
            ["DC_a01.wav", "not-savee-at-all.wav"], "savee"
        )
        assert "DC_a01.wav" in table
        assert "not-savee-at-all.wav" not in table
        assert any("did not match" in warning for warning in warnings)

    def test_unrecognised_code_is_kept_and_named(self):
        """An unmapped code is still a consistent class; dropping it hides it."""
        table, warnings = derive_from_filenames(["DC_zz01.wav"], "savee")
        assert table["DC_zz01.wav"]["emotion"] == "zz"
        assert any("zz" in warning for warning in warnings)

    def test_no_match_at_all_raises(self):
        with pytest.raises(ValueError, match="No filename matched"):
            derive_from_filenames(["totally_wrong.wav"], "savee")

    def test_unknown_pattern_raises_and_lists_options(self):
        with pytest.raises(ValueError, match="savee"):
            derive_from_filenames(["a.wav"], "nonexistent")

    def test_available_patterns_exposes_probeable_properties(self):
        patterns = {entry["pattern_id"]: entry for entry in available_patterns()}
        assert set(patterns["savee"]["properties"]) == {"speaker", "emotion"}
        assert "take" not in patterns["savee"]["properties"]


class TestBandLabels:
    def test_splits_into_equal_count_tertiles(self):
        bands = band_labels(list(range(30)))
        assert bands.count("low") == 10
        assert bands.count("mid") == 10
        assert bands.count("high") == 10

    def test_is_rank_based_so_skew_still_yields_usable_classes(self):
        """Equal-width bins would put all but two values in one band."""
        values = [1.0] * 10 + [1.1] * 10 + [500.0] * 10
        bands = band_labels(values)
        assert {bands.count(name) for name in ("low", "mid", "high")} == {10}

    def test_too_few_values_yields_nothing(self):
        assert band_labels([1, 2, 3]) == [None, None, None]

    def test_all_identical_values_yield_nothing(self):
        """A single-class column would fail downstream, less legibly."""
        assert band_labels([5.0] * 40) == [None] * 40

    def test_missing_values_are_preserved_as_none(self):
        bands = band_labels([1, None, 2, None] + list(range(3, 20)))
        assert bands[1] is None and bands[3] is None

    def test_attach_duration_bands_adds_a_probeable_column(self):
        rows = [
            {"filename": f"c{index}.wav", "duration": str(index * 0.5)}
            for index in range(1, 31)
        ]
        attach_duration_bands(rows)
        assert all("duration_band" in row for row in rows)
        assert {row["duration_band"] for row in rows} == {"low", "mid", "high"}

    def test_attach_duration_bands_tolerates_unreadable_durations(self):
        rows = [{"filename": "a.wav", "duration": "not-a-number"}]
        attach_duration_bands(rows)
        assert "duration_band" not in rows[0]


class TestMergeAndColumns:
    def test_merge_joins_on_filename(self):
        rows = [{"filename": "DC_a01.wav"}, {"filename": "JE_n02.wav"}]
        merge_labels(rows, {"DC_a01.wav": {"emotion": "anger"}})
        assert rows[0]["emotion"] == "anger"
        assert "emotion" not in rows[1]

    def test_merge_falls_back_to_basename(self):
        rows = [{"filename": "a01.wav"}]
        merge_labels(rows, {"AudioData/DC/a01.wav": {"emotion": "anger"}, "a01.wav": {"emotion": "anger"}})
        assert rows[0]["emotion"] == "anger"

    def test_label_columns_excludes_file_facts(self):
        rows = [{
            "filename": "a.wav", "duration": "1.0", "size": "10",
            "sample_rate": "16000", "uploaded_at": "now", "emotion": "anger",
        }]
        assert label_columns(rows) == ["emotion"]


class TestSummariseProperty:
    def test_reports_classes_and_majority_baseline(self):
        summary = summarise_property(_rows(["a"] * 12 + ["b"] * 8), "emotion")
        assert summary["probeable"] is True
        assert summary["n_samples"] == 20
        assert summary["class_counts"] == {"a": 12, "b": 8}
        assert summary["majority_baseline"] == pytest.approx(0.6)

    def test_drops_rare_classes_and_names_them(self):
        summary = summarise_property(_rows(["a"] * 10 + ["b"] * 10 + ["c"] * 2), "emotion")
        assert summary["class_counts"] == {"a": 10, "b": 10}
        assert summary["dropped_classes"] == [{"label": "c", "count": 2}]
        assert summary["n_samples"] == 20

    def test_folds_cannot_exceed_the_smallest_class(self):
        summary = summarise_property(_rows(["a"] * 20 + ["b"] * 6), "emotion", cv_folds=5)
        assert summary["cv_folds_used"] == 5
        summary = summarise_property(_rows(["a"] * 20 + ["b"] * 3), "emotion", min_class_count=2)
        assert summary["cv_folds_used"] == 3

    def test_single_surviving_class_is_skipped(self):
        """SAVEE is all-male, so a gender column would land exactly here."""
        summary = summarise_property(_rows(["male"] * 30, "gender"), "gender")
        assert summary["probeable"] is False
        assert "1 class remains" in summary["skipped_reason"]
        # Mirrors `_probe_property`, which zeroes these in the degenerate case
        # while still naming the class it found.
        assert summary["n_samples"] == 0
        assert summary["class_counts"] == {"male": 30}

    def test_no_labels_at_all_is_skipped(self):
        summary = summarise_property(_rows([None] * 20), "emotion")
        assert summary["probeable"] is False
        assert summary["skipped_reason"] == "no labelled files"

    def test_too_few_rows_is_skipped(self):
        summary = summarise_property(_rows(["a"] * 3 + ["b"] * 3), "emotion", min_class_count=2)
        assert summary["probeable"] is False
        assert "at least 8" in summary["skipped_reason"]

    def test_unannotated_rows_are_counted_separately(self):
        summary = summarise_property(_rows(["a"] * 10 + ["b"] * 10 + [None] * 5), "emotion")
        assert summary["n_missing"] == 5
        assert summary["n_samples"] == 20


class TestPreviewDataset:
    def test_previews_every_probeable_column(self):
        rows = [
            {"filename": f"c{index}.wav", "emotion": "a" if index % 2 else "b", "speaker": "DC"}
            for index in range(20)
        ]
        preview = preview_dataset(rows)
        names = {entry["property"] for entry in preview["properties"]}
        assert names == {"emotion", "speaker"}
        assert preview["probeable_count"] == 1  # speaker is single-class
        assert preview["n_files"] == 20

    def test_identifier_columns_are_not_offered(self):
        """One distinct value per file is an id, and would give one class each."""
        rows = [
            {"filename": f"c{index}.wav", "clip_id": f"id-{index}", "emotion": "a"}
            for index in range(MAX_DISCOVERED_CLASSES + 10)
        ]
        assert {entry["property"] for entry in preview_dataset(rows)["properties"]} == {"emotion"}


class TestPreviewMatchesProbe:
    """The preview must agree with the probe it is previewing.

    A preview that promised five folds and 40 usable files, followed by a probe
    that used three folds over 30, would be worse than no preview at all -- the
    user planned a run around numbers that were never going to happen.
    """

    @pytest.mark.parametrize(
        "labels",
        [
            ["a"] * 12 + ["b"] * 8,
            ["a"] * 10 + ["b"] * 10 + ["c"] * 2,          # a class gets dropped
            ["a"] * 20 + ["b"] * 3,                        # folds reduced
            ["a"] * 9 + ["b"] * 9 + [None] * 4,            # unannotated rows
        ],
    )
    def test_preview_agrees_with_train_layer_probes(self, labels):
        import numpy as np

        from app.services.probing_service import train_layer_probes

        rows = _rows(labels)
        activations = np.random.default_rng(0).normal(size=(len(labels), 3, 8))
        result = train_layer_probes(
            activations.tolist(),
            {"emotion": [row["emotion"] or None for row in rows]},
            project_dims=0,
            include_control=False,
            min_class_count=5,
            cv_folds=5,
            seed=42,
        )
        probe = result["properties"]["emotion"]
        preview = summarise_property(rows, "emotion", min_class_count=5, cv_folds=5)

        assert preview["n_samples"] == probe["n_samples"]
        assert preview["n_missing"] == probe["n_missing"]
        assert preview["n_classes"] == probe["n_classes"]
        assert preview["class_counts"] == probe["class_counts"]
        assert preview["dropped_classes"] == probe["dropped_classes"]
        assert preview["cv_folds_used"] == probe["cv_folds_used"]
        assert preview["probeable"] is (probe["skipped_reason"] is None)
        if preview["majority_baseline"] is not None:
            assert preview["majority_baseline"] == pytest.approx(probe["majority_baseline"])


class TestSaveeEndToEnd:
    """The documented SAVEE workflow, exercised on the real filename inventory."""

    @staticmethod
    def _savee_filenames():
        counts = {"a": 15, "d": 15, "f": 15, "h": 15, "n": 30, "sa": 15, "su": 15}
        return [
            f"{speaker}_{code}{take:02d}.wav"
            for speaker in ("DC", "JE", "JK", "KL")
            for code, total in counts.items()
            for take in range(1, total + 1)
        ]

    def test_full_inventory_is_480_clips(self):
        assert len(self._savee_filenames()) == 480

    def test_every_savee_filename_parses(self):
        names = self._savee_filenames()
        table, warnings = derive_from_filenames(names, "savee")
        assert len(table) >= len(names)
        assert warnings == []

    @staticmethod
    def _preview_for(names):
        table, _ = derive_from_filenames(names, "savee")
        rows = [{"filename": name} for name in names]
        merge_labels(rows, table)
        return {entry["property"]: entry for entry in preview_dataset(rows)["properties"]}

    def test_naive_first_100_clips_yields_a_single_speaker(self):
        """The trap the subset script exists to avoid.

        SAVEE ships grouped by speaker, so "just take the first 100" gives 100
        clips of DC.  The speaker probe then has one class and is correctly
        refused -- which is the honest outcome, but not a useful dataset.  This
        is why `scripts/prepare_savee_subset.py` samples stratified rather than
        slicing, and why the preview matters before anyone spends extraction
        time on it.
        """
        preview = self._preview_for(self._savee_filenames()[:100])
        assert preview["speaker"]["probeable"] is False
        assert "1 class remains" in preview["speaker"]["skipped_reason"]

    def test_stratified_100_clip_subset_is_probeable_on_both_properties(self):
        """Round-robin over speakers within each emotion, as the script does."""
        by_emotion: dict[str, list[str]] = {}
        for name in self._savee_filenames():
            code = name.split("_")[1].split(".")[0].rstrip("0123456789")
            by_emotion.setdefault(code, []).append(name)
        assert len(by_emotion) == 7

        # 7 emotions over 100 clips: five get 14, two get 15.
        names: list[str] = []
        for index, code in enumerate(sorted(by_emotion)):
            quota = 14 + (1 if index < 2 else 0)
            pool = sorted(by_emotion[code], key=lambda entry: (entry.split("_")[1], entry))
            names.extend(pool[:quota])

        assert len(names) == 100
        preview = self._preview_for(names)
        assert preview["emotion"]["probeable"] is True
        assert preview["speaker"]["probeable"] is True
        assert preview["emotion"]["n_samples"] == 100
        assert preview["emotion"]["n_classes"] == 7
        assert preview["speaker"]["n_classes"] == 4
