"""
Layer-wise probing service tests.

This is the highest-value test file in the layer-representation feature: it is the
only place where the ground truth is knowable.  On real audio there is no way to
tell "the probe found speaker identity at layer 3" from "the pipeline is subtly
broken and reports layer 3".  Here the signal is planted, so the answer is known
in advance.

If `test_planted_layer_is_recovered` fails, every downstream number the UI shows is
meaningless -- fix it before looking at anything else.
"""

import numpy as np
import pytest

from app.services.probing_service import train_layer_probes


SEED = 42
DIM = 32


def _noise(n_files: int, num_layers: int, dim: int = DIM, seed: int = SEED) -> np.ndarray:
    """Pure Gaussian noise: [file][layer][dim], carrying no label information."""
    return np.random.default_rng(seed).normal(size=(n_files, num_layers, dim))


def _balanced_labels(n_files: int, classes: list[str]) -> list[str]:
    return [classes[index % len(classes)] for index in range(n_files)]


def _plant(
    activations: np.ndarray,
    labels: list[str],
    layer: int,
    strength: float = 2.0,
    width: int | None = None,
) -> np.ndarray:
    """Give each class its own offset direction at one layer.

    That makes the class linearly separable at exactly that layer and nowhere else.

    Two properties of the construction matter:

    * The offset is spread across a block of dimensions at a modest multiple of
      the noise scale, which is how a distributed representation actually
      encodes a property.  Concentrating the same total signal into three or
      four dimensions would be recovered by an *unscaled* probe but not by this
      one -- the probe standardises features so layers with different activation
      magnitudes stay comparable, and that necessarily trades away signals
      carried by a few outsized dimensions.  See `_cross_validate`.
    * Each class gets a *different* block rather than a larger offset along one
      shared direction.  Equally spaced collinear class means are separable by a
      multinomial probe but not by a one-vs-rest one, because the middle class
      cannot be split from the union of the others -- an artefact of the test
      data that would look like `linear_svm` being broken.
    """
    classes = sorted(set(labels))
    index_of = {label: index for index, label in enumerate(classes)}
    planted = activations.copy()
    dim = activations.shape[2]
    span = width if width is not None else max(1, dim // (2 * len(classes)))
    for row, label in enumerate(labels):
        start = index_of[label] * span
        planted[row, layer, start:start + span] += strength
    return planted


class TestPlantedSignalRecovery:
    """Does the probe find information that is provably there, where it is?"""

    def test_planted_layer_is_recovered(self):
        """The single most important assertion in the feature."""
        n_files, num_layers, planted_layer = 60, 6, 3
        labels = _balanced_labels(n_files, ["a", "b", "c"])
        activations = _plant(_noise(n_files, num_layers), labels, planted_layer)

        result = train_layer_probes(
            activations.tolist(), {"planted": labels}, project_dims=0, seed=SEED
        )
        probe = result["properties"]["planted"]

        assert probe["best_layer"] == planted_layer
        assert probe["layers"][planted_layer]["accuracy"] > 0.9
        # Every other layer is pure noise, so it cannot beat the majority baseline
        # by any meaningful margin.
        for layer in probe["layers"]:
            if layer["layer"] != planted_layer:
                assert layer["accuracy"] < probe["majority_baseline"] + 0.2

    def test_planted_layer_has_high_selectivity(self):
        n_files, num_layers, planted_layer = 60, 6, 3
        labels = _balanced_labels(n_files, ["a", "b", "c"])
        activations = _plant(_noise(n_files, num_layers), labels, planted_layer)

        result = train_layer_probes(
            activations.tolist(), {"planted": labels}, project_dims=0, seed=SEED
        )
        layers = result["properties"]["planted"]["layers"]

        assert layers[planted_layer]["selectivity"] > 0.5
        assert result["properties"]["planted"]["best_selectivity"] > 0.5

    def test_monotone_emergence_peaks_at_the_top(self):
        """Signal-to-noise rising with depth must produce a rising accuracy curve.

        The signal deliberately stops short of saturation: ties resolve to the
        earliest layer, so a curve that reached 1.0 early would legitimately
        report that earlier layer as the peak.
        """
        n_files, num_layers = 80, 6
        labels = _balanced_labels(n_files, ["a", "b"])
        activations = _noise(n_files, num_layers)
        classes = {"a": 0, "b": 1}
        for layer in range(num_layers):
            strength = 1.1 * layer / (num_layers - 1)
            for row, label in enumerate(labels):
                activations[row, layer, :4] += classes[label] * strength

        result = train_layer_probes(
            activations.tolist(), {"emerging": labels}, project_dims=0, seed=SEED
        )
        probe = result["properties"]["emerging"]
        accuracies = [layer["accuracy"] for layer in probe["layers"]]

        assert probe["best_layer"] == num_layers - 1
        assert probe["peak_depth"] == pytest.approx(1.0)
        assert accuracies[-1] > accuracies[0] + 0.3

    def test_peak_depth_is_normalised_position(self):
        n_files, num_layers, planted_layer = 60, 7, 3
        labels = _balanced_labels(n_files, ["a", "b", "c"])
        activations = _plant(_noise(n_files, num_layers), labels, planted_layer)

        result = train_layer_probes(
            activations.tolist(), {"planted": labels}, project_dims=0, seed=SEED
        )

        assert result["properties"]["planted"]["peak_depth"] == pytest.approx(0.5)


class TestBaselines:
    """The numbers that stop an accuracy from being over-read."""

    def test_control_probe_kills_memorisation(self):
        """With no signal anywhere, selectivity must sit at ~0 on every layer."""
        n_files, num_layers = 60, 5
        labels = _balanced_labels(n_files, ["a", "b", "c"])
        activations = _noise(n_files, num_layers)

        result = train_layer_probes(
            activations.tolist(), {"random": labels}, project_dims=0, seed=SEED
        )

        for layer in result["properties"]["random"]["layers"]:
            assert abs(layer["selectivity"]) < 0.2

    def test_majority_baseline_exposes_imbalance(self):
        """A 90/10 split with no signal scores ~0.9 accuracy but ~0.47 macro F1.

        This is exactly why the UI must never show accuracy on its own.
        """
        n_files, num_layers = 100, 3
        labels = ["a"] * 90 + ["b"] * 10
        activations = _noise(n_files, num_layers)

        result = train_layer_probes(
            activations.tolist(), {"imbalanced": labels}, project_dims=0, seed=SEED
        )
        probe = result["properties"]["imbalanced"]

        assert probe["majority_baseline"] == pytest.approx(0.9)
        assert probe["best_accuracy"] == pytest.approx(0.9, abs=0.08)
        # Predicting the majority class for everything gives macro F1 ~0.47.
        assert probe["layers"][0]["macro_f1"] < 0.6

    def test_control_can_be_disabled(self):
        labels = _balanced_labels(40, ["a", "b"])
        activations = _noise(40, 3)

        result = train_layer_probes(
            activations.tolist(), {"x": labels},
            project_dims=0, include_control=False, seed=SEED,
        )

        for layer in result["properties"]["x"]["layers"]:
            assert layer["control_accuracy"] is None
            assert layer["selectivity"] is None


class TestLabelHygiene:
    """Thin, missing and degenerate labels must be reported, never papered over."""

    def test_rare_classes_are_dropped_and_reported(self):
        labels = ["a"] * 20 + ["b"] * 20 + ["c"] * 3 + ["d"] * 2
        activations = _noise(len(labels), 4)

        result = train_layer_probes(
            activations.tolist(), {"thin": labels},
            project_dims=0, min_class_count=5, seed=SEED,
        )
        probe = result["properties"]["thin"]

        assert [entry["label"] for entry in probe["dropped_classes"]] == ["c", "d"]
        assert [entry["count"] for entry in probe["dropped_classes"]] == [3, 2]
        assert probe["n_classes"] == 2
        assert probe["n_samples"] == 40
        assert probe["skipped_reason"] is None

    def test_missing_labels_are_excluded_not_treated_as_a_class(self):
        n_files = 60
        labels = [None if index % 10 < 3 else "a" if index % 2 else "b" for index in range(n_files)]
        activations = _noise(n_files, 4)

        result = train_layer_probes(
            activations.tolist(), {"sparse": labels}, project_dims=0, seed=SEED
        )
        probe = result["properties"]["sparse"]

        assert probe["n_missing"] == 18
        assert probe["n_samples"] == 42
        assert set(probe["class_labels"]) == {"a", "b"}

    def test_unknown_is_treated_as_missing(self):
        """Common Voice writes "unknown" rather than leaving the cell blank."""
        labels = ["unknown"] * 30 + ["us"] * 15 + ["england"] * 15
        activations = _noise(len(labels), 3)

        result = train_layer_probes(
            activations.tolist(), {"accent": labels}, project_dims=0, seed=SEED
        )
        probe = result["properties"]["accent"]

        assert probe["n_missing"] == 30
        assert set(probe["class_labels"]) == {"england", "us"}

    def test_single_class_is_skipped_without_raising(self):
        labels = ["same"] * 40
        activations = _noise(40, 5)

        result = train_layer_probes(
            activations.tolist(), {"degenerate": labels}, project_dims=0, seed=SEED
        )
        probe = result["properties"]["degenerate"]

        assert probe["skipped_reason"] is not None
        assert probe["best_layer"] is None
        assert len(probe["layers"]) == 5
        assert all(layer["accuracy"] is None for layer in probe["layers"])

    def test_all_classes_below_threshold_is_skipped(self):
        labels = ["a", "a", "b", "b", "c", "c", "d", "d"]
        activations = _noise(len(labels), 3)

        result = train_layer_probes(
            activations.tolist(), {"tiny": labels},
            project_dims=0, min_class_count=5, seed=SEED,
        )
        probe = result["properties"]["tiny"]

        assert probe["skipped_reason"] is not None
        assert len(probe["dropped_classes"]) == 4

    def test_folds_shrink_to_the_smallest_class(self):
        """5 folds are impossible when a class has 3 members; use what fits."""
        labels = ["a"] * 30 + ["b"] * 3
        activations = _noise(len(labels), 3)

        result = train_layer_probes(
            activations.tolist(), {"x": labels},
            project_dims=0, cv_folds=5, min_class_count=3, seed=SEED,
        )

        assert result["properties"]["x"]["cv_folds_used"] == 3


class TestContract:
    """The UI branches on none of this, so all of it must always be there."""

    def test_output_shape_is_fully_populated(self):
        labels = _balanced_labels(40, ["a", "b"])
        activations = _noise(40, 4)

        result = train_layer_probes(
            activations.tolist(), {"x": labels}, project_dims=0, seed=SEED
        )

        assert result["num_layers"] == 4
        assert result["layer_names"] == ["input", "layer_1", "layer_2", "layer_3"]
        assert result["hidden_dim"] == DIM
        assert result["n_files"] == 40
        assert result["params"]["probe"] == "logreg"
        probe = result["properties"]["x"]
        for key in (
            "n_samples", "n_missing", "n_classes", "class_counts", "dropped_classes",
            "majority_baseline", "cv_folds_used", "layers", "best_layer", "best_accuracy",
            "best_selectivity", "peak_depth", "confusion_matrix", "class_labels",
            "skipped_reason",
        ):
            assert key in probe, f"missing contract key: {key}"

    def test_confusion_matrix_matches_the_class_labels(self):
        labels = _balanced_labels(60, ["a", "b", "c"])
        activations = _plant(_noise(60, 4), labels, 2)

        result = train_layer_probes(
            activations.tolist(), {"x": labels}, project_dims=0, seed=SEED
        )
        probe = result["properties"]["x"]

        assert len(probe["confusion_matrix"]) == 3
        assert all(len(row) == 3 for row in probe["confusion_matrix"])
        assert sum(sum(row) for row in probe["confusion_matrix"]) == probe["n_samples"]

    def test_multiple_properties_are_independent(self):
        n_files, num_layers = 60, 6
        # The two label sets must be uncorrelated, or a signal planted for one
        # would also separate the other and the test would prove nothing.
        early = ["p" if index % 2 else "q" for index in range(n_files)]
        late = ["x" if (index // 2) % 2 else "y" for index in range(n_files)]
        assert len({(a, b) for a, b in zip(early, late)}) == 4
        activations = _noise(n_files, num_layers)
        for row in range(n_files):
            activations[row, 1, :16] += (early[row] == "q") * 1.5
            activations[row, 5, 16:] += (late[row] == "y") * 1.5

        result = train_layer_probes(
            activations.tolist(), {"early": early, "late": late}, project_dims=0, seed=SEED
        )

        assert result["properties"]["early"]["best_layer"] == 1
        assert result["properties"]["late"]["best_layer"] == 5
        # The ordering claim the whole feature rests on.
        assert result["properties"]["early"]["peak_depth"] < result["properties"]["late"]["peak_depth"]

    def test_mismatched_label_length_raises(self):
        activations = _noise(20, 3)

        with pytest.raises(ValueError, match="19 labels for 20 files"):
            train_layer_probes(activations.tolist(), {"x": ["a"] * 19}, project_dims=0)

    def test_unknown_probe_type_raises(self):
        activations = _noise(20, 3)

        with pytest.raises(ValueError, match="Unknown probe type"):
            train_layer_probes(activations.tolist(), {"x": ["a"] * 20}, probe="magic")

    def test_a_failing_property_does_not_void_the_others(self):
        labels = _balanced_labels(40, ["a", "b"])
        activations = _noise(40, 3)

        result = train_layer_probes(
            activations.tolist(),
            {"good": labels, "empty": [None] * 40},
            project_dims=0, seed=SEED,
        )

        assert result["properties"]["good"]["skipped_reason"] is None
        assert result["properties"]["empty"]["skipped_reason"] is not None


class TestReproducibility:
    def test_identical_input_gives_identical_output(self):
        labels = _balanced_labels(60, ["a", "b", "c"])
        activations = _plant(_noise(60, 5), labels, 2).tolist()

        first = train_layer_probes(activations, {"x": labels}, seed=SEED)
        second = train_layer_probes(activations, {"x": labels}, seed=SEED)

        assert first == second

    def test_projection_preserves_the_finding(self):
        """A random projection must not move the peak or lose the signal.

        The assertion on accuracy is one-sided on purpose: projecting can only
        cost information in principle, but in practice it also regularises, so
        demanding the two match symmetrically would fail for the wrong reason.
        What matters is that the projected run does not find *less*.
        """
        n_files, num_layers, planted_layer = 60, 5, 2
        labels = _balanced_labels(n_files, ["a", "b", "c"])
        activations = _plant(
            _noise(n_files, num_layers, dim=256), labels, planted_layer
        ).tolist()

        full = train_layer_probes(activations, {"x": labels}, project_dims=0, seed=SEED)
        projected = train_layer_probes(activations, {"x": labels}, project_dims=64, seed=SEED)

        assert projected["projected_dim"] == 64
        assert full["projected_dim"] == 256
        assert projected["properties"]["x"]["best_layer"] == planted_layer
        assert full["properties"]["x"]["best_layer"] == planted_layer
        assert projected["properties"]["x"]["best_accuracy"] > 0.9
        assert (
            projected["properties"]["x"]["best_accuracy"]
            >= full["properties"]["x"]["best_accuracy"] - 0.1
        )

    def test_projection_is_skipped_when_wider_than_the_input(self):
        labels = _balanced_labels(40, ["a", "b"])
        activations = _noise(40, 3, dim=32).tolist()

        result = train_layer_probes(activations, {"x": labels}, project_dims=256, seed=SEED)

        assert result["projected_dim"] == 32


class TestProbeTypes:
    @pytest.mark.parametrize("probe", ["logreg", "linear_svm"])
    def test_every_registered_probe_recovers_planted_signal(self, probe):
        labels = _balanced_labels(60, ["a", "b", "c"])
        activations = _plant(_noise(60, 5), labels, 2).tolist()

        result = train_layer_probes(
            activations, {"x": labels}, probe=probe, project_dims=0, seed=SEED
        )

        assert result["properties"]["x"]["best_layer"] == 2
        assert result["properties"]["x"]["best_accuracy"] > 0.9
