"""Layer-wise diagnostic probing over per-layer encoder activations.

Given one pooled activation vector per (file, layer) and a set of per-file
property labels, train a lightweight classifier at every layer and report where
each property is most linearly decodable.  The result answers "at what depth
does this model encode speaker identity / lexical content / emotion?".

Pure computation: no Redis, no object storage, no torch, no model loading.
Everything here is callable from a unit test with plain nested lists, which is
the only place the ground truth is actually knowable -- see
``tests/test_probing_service.py``.

Three numbers are reported per (property, layer) and all three are needed to
read the result honestly:

``accuracy``
    Mean cross-validated accuracy of the probe.
``majority_baseline``
    Share of the largest class.  Accuracy at or below this means *no*
    information was found, however large the accuracy looks.
``selectivity``
    ``accuracy - control_accuracy``, where the control probe is the identical
    pipeline trained on shuffled labels (Hewitt & Liang's control-task
    convention).  A probe that beats the majority baseline but not its control
    is memorising the training rows, not reading the representation.

The output contract is consumed by the frontend (`LayerProbeResult` in
`Frontend/src/lib/probes.ts`), so every key below is always present even in
degenerate cases -- the UI must never have to branch on a partial shape.
"""

from __future__ import annotations

from collections import Counter
import logging
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Labels carrying no information.  Dataset CSVs use "unknown" as an explicit
# not-annotated marker (Common Voice age/gender/accent), which must be treated
# as missing rather than as a class of its own.
MISSING_LABELS = {"", "unknown", "none", "n/a", "na"}

# A probe needs at least this many usable rows before cross-validation means
# anything, regardless of how the classes are distributed.
MIN_ROWS_FOR_PROBE = 8

# Cross-validation cannot use more folds than the smallest class has members.
MIN_FOLDS = 2


def _make_logreg(seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression

    # lbfgs is deterministic, so the probe is reproducible without a seed; the
    # argument is accepted for interface symmetry with the other factories.
    del seed
    return LogisticRegression(max_iter=2000, C=1.0)


def _make_linear_svm(seed: int) -> Any:
    from sklearn.svm import LinearSVC

    return LinearSVC(C=1.0, max_iter=5000, random_state=seed)


# Adding a probe type is a registry entry, not a change to the training loop.
PROBE_FACTORIES: dict[str, Callable[[int], Any]] = {
    "logreg": _make_logreg,
    "linear_svm": _make_linear_svm,
}


def _default_layer_names(num_layers: int) -> list[str]:
    """Layer 0 is the encoder's embedding output, before any encoder block.

    Callers extracting real activations should pass the names from the
    extraction payload instead, so the two contracts cannot drift apart.
    """
    return ["input"] + [f"layer_{index}" for index in range(1, num_layers)]


def _normalise_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in MISSING_LABELS else text


def _empty_property(
    reason: str, num_layers: int, names: Sequence[str], cv_folds: int
) -> dict[str, Any]:
    """A well-formed payload for a property that cannot be probed."""
    return {
        "n_samples": 0,
        "n_missing": 0,
        "n_classes": 0,
        "class_counts": {},
        "dropped_classes": [],
        "majority_baseline": None,
        "cv_folds_used": 0,
        "layers": [
            {
                "layer": index,
                "layer_name": names[index],
                "accuracy": None,
                "accuracy_std": None,
                "macro_f1": None,
                "control_accuracy": None,
                "selectivity": None,
            }
            for index in range(num_layers)
        ],
        "best_layer": None,
        "best_accuracy": None,
        "best_selectivity": None,
        "peak_depth": None,
        "confusion_matrix": [],
        "class_labels": [],
        "skipped_reason": reason,
    }


def _project(matrix: np.ndarray, project_dims: int, seed: int) -> np.ndarray:
    """Seeded Gaussian random projection, or the input unchanged.

    Johnson-Lindenstrauss preserves the linear separability a probe measures
    while making a 1280-d whisper-large stack roughly five times cheaper to
    train on.  It is label-free, so applying it outside the cross-validation
    loop leaks nothing.
    """
    n_features = matrix.shape[1]
    if project_dims <= 0 or project_dims >= n_features:
        return matrix
    from sklearn.random_projection import GaussianRandomProjection

    projector = GaussianRandomProjection(n_components=project_dims, random_state=seed)
    return projector.fit_transform(matrix)


def _cross_validate(
    features: np.ndarray,
    encoded: np.ndarray,
    n_classes: int,
    folds: int,
    probe: str,
    seed: int,
) -> tuple[float, float, float, np.ndarray]:
    """Stratified CV of one probe.

    Returns (mean accuracy, std over folds, macro F1 on out-of-fold predictions,
    out-of-fold predictions).  The scaler is fitted inside each fold so no test
    row ever contributes to its own standardisation.

    Standardisation is not cosmetic here.  Activation magnitudes differ by an
    order of magnitude between the bottom and top of an encoder stack, and L2
    regularisation penalises raw weight size -- without scaling, a layer would
    score better simply for having larger activations, which would make the
    cross-layer comparison this whole feature rests on meaningless.  The cost
    is that a signal concentrated in a handful of unusually high-variance
    dimensions gets pulled down toward the noise floor; distributed signals,
    which is what encoder representations actually produce, are unaffected.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions = np.empty_like(encoded)
    fold_accuracies: list[float] = []
    for train_index, test_index in splitter.split(features, encoded):
        estimator = make_pipeline(StandardScaler(), PROBE_FACTORIES[probe](seed))
        estimator.fit(features[train_index], encoded[train_index])
        predicted = estimator.predict(features[test_index])
        predictions[test_index] = predicted
        fold_accuracies.append(float(np.mean(predicted == encoded[test_index])))

    macro_f1 = float(
        f1_score(encoded, predictions, average="macro", labels=np.arange(n_classes), zero_division=0)
    )
    return float(np.mean(fold_accuracies)), float(np.std(fold_accuracies)), macro_f1, predictions


def _probe_property(
    projected: list[np.ndarray],
    labels: Sequence[Any],
    names: Sequence[str],
    *,
    probe: str,
    cv_folds: int,
    min_class_count: int,
    include_control: bool,
    seed: int,
) -> dict[str, Any]:
    """Train one probe per layer for a single property."""
    num_layers = len(projected)
    normalised = [_normalise_label(value) for value in labels]
    n_missing = sum(1 for value in normalised if value is None)

    present = [(index, value) for index, value in enumerate(normalised) if value is not None]
    raw_counts = Counter(value for _, value in present)
    kept_classes = {label for label, count in raw_counts.items() if count >= min_class_count}
    dropped = [
        {"label": label, "count": int(count)}
        for label, count in sorted(raw_counts.items())
        if label not in kept_classes
    ]

    rows = [index for index, value in present if value in kept_classes]
    kept_labels = [normalised[index] for index in rows]
    counts = Counter(kept_labels)

    if len(counts) < 2:
        reason = (
            "no labelled files" if not raw_counts
            else f"only {len(counts)} class remains after dropping classes with fewer than {min_class_count} files"
        )
        result = _empty_property(reason, num_layers, names, cv_folds)
        result["n_missing"] = n_missing
        result["class_counts"] = {label: int(count) for label, count in sorted(counts.items())}
        result["dropped_classes"] = dropped
        return result

    if len(rows) < MIN_ROWS_FOR_PROBE:
        result = _empty_property(
            f"needs at least {MIN_ROWS_FOR_PROBE} labelled files, got {len(rows)}",
            num_layers, names, cv_folds,
        )
        result["n_missing"] = n_missing
        result["n_samples"] = len(rows)
        result["n_classes"] = len(counts)
        result["class_counts"] = {label: int(count) for label, count in sorted(counts.items())}
        result["dropped_classes"] = dropped
        return result

    class_labels = sorted(counts)
    encoding = {label: index for index, label in enumerate(class_labels)}
    encoded = np.array([encoding[label] for label in kept_labels], dtype=int)
    row_index = np.asarray(rows, dtype=int)

    # StratifiedKFold cannot split a class into more folds than it has members.
    folds = max(MIN_FOLDS, min(int(cv_folds), min(counts.values())))
    majority = max(counts.values()) / len(rows)

    # A permutation keeps the class distribution -- and therefore the majority
    # baseline -- identical, so the control isolates memorisation capacity
    # rather than changing the problem.
    control_encoded = np.random.default_rng(seed).permutation(encoded) if include_control else None

    layers: list[dict[str, Any]] = []
    best_index: int | None = None
    best_predictions: np.ndarray | None = None
    for index in range(num_layers):
        features = projected[index][row_index]
        accuracy, accuracy_std, macro_f1, predictions = _cross_validate(
            features, encoded, len(class_labels), folds, probe, seed
        )
        control_accuracy: float | None = None
        if control_encoded is not None:
            control_accuracy, _, _, _ = _cross_validate(
                features, control_encoded, len(class_labels), folds, probe, seed
            )
        layers.append({
            "layer": index,
            "layer_name": names[index],
            "accuracy": accuracy,
            "accuracy_std": accuracy_std,
            "macro_f1": macro_f1,
            "control_accuracy": control_accuracy,
            "selectivity": None if control_accuracy is None else accuracy - control_accuracy,
        })
        # Strict `>` makes ties resolve to the *earliest* layer, which is the
        # reading the feature claims: once a property is fully decodable, the
        # first layer that achieves it is where it emerged, not the last layer
        # that happens to retain it.
        if best_index is None or accuracy > layers[best_index]["accuracy"]:
            best_index = index
            best_predictions = predictions

    from sklearn.metrics import confusion_matrix

    best = layers[best_index]
    matrix = confusion_matrix(encoded, best_predictions, labels=np.arange(len(class_labels)))
    return {
        "n_samples": len(rows),
        "n_missing": n_missing,
        "n_classes": len(class_labels),
        "class_counts": {label: int(counts[label]) for label in class_labels},
        "dropped_classes": dropped,
        "majority_baseline": float(majority),
        "cv_folds_used": folds,
        "layers": layers,
        "best_layer": best_index,
        "best_accuracy": best["accuracy"],
        "best_selectivity": best["selectivity"],
        # Where in the stack the property peaks, normalised so models with
        # different depths can be compared on one axis.
        "peak_depth": float(best_index / (num_layers - 1)) if num_layers > 1 else 0.0,
        "confusion_matrix": matrix.astype(int).tolist(),
        "class_labels": class_labels,
        "skipped_reason": None,
    }


def train_layer_probes(
    activations: Sequence[Sequence[Sequence[float]]],
    properties: dict[str, Sequence[Any]],
    *,
    probe: str = "logreg",
    cv_folds: int = 5,
    project_dims: int = 256,
    min_class_count: int = 5,
    include_control: bool = True,
    seed: int = 42,
    layer_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Train a linear probe at every layer for every property.

    Args:
        activations: ``[file][layer][dim]`` pooled activations, aligned with the
            caller's file order.  Every file must expose the same layer count
            and hidden size.
        properties: Property name -> one label per file, positionally aligned
            with ``activations``.  ``None`` and "unknown" mean not annotated.
        probe: Key into :data:`PROBE_FACTORIES`.
        cv_folds: Requested stratified folds; reduced per property when a class
            has fewer members than that.
        project_dims: Random-projection target width; 0 disables projection.
        min_class_count: Classes with fewer files than this are dropped and
            reported rather than silently producing a meaningless class count.
        include_control: Also train a shuffled-label control probe per layer.
        seed: Seeds the projection, the fold assignment and the control shuffle.
        layer_names: Display names from the extraction payload. Defaults to
            ``input, layer_1, ...`` when probing synthetic activations.
    Returns:
        A dict matching the frontend `LayerProbeResult` contract.
    """
    if probe not in PROBE_FACTORIES:
        raise ValueError(f"Unknown probe type: {probe}")

    matrix = np.asarray(activations, dtype=float)
    if matrix.ndim != 3:
        raise ValueError(
            f"activations must be [file][layer][dim]; got array with {matrix.ndim} dimensions"
        )
    n_files, num_layers, hidden_dim = matrix.shape
    names = list(layer_names) if layer_names else _default_layer_names(num_layers)
    if len(names) != num_layers:
        raise ValueError(f"got {len(names)} layer names for {num_layers} layers")

    for name, labels in properties.items():
        if len(labels) != n_files:
            raise ValueError(
                f"property '{name}' has {len(labels)} labels for {n_files} files"
            )

    # Projected once for the whole batch so every property and every layer sees
    # the same subspace, and so toggling a probe hyperparameter never re-projects.
    projected = [_project(matrix[:, index, :], int(project_dims), seed) for index in range(num_layers)]
    projected_dim = int(projected[0].shape[1]) if projected else 0

    probes: dict[str, Any] = {}
    for name, labels in properties.items():
        try:
            probes[name] = _probe_property(
                projected, labels, names,
                probe=probe, cv_folds=cv_folds, min_class_count=min_class_count,
                include_control=include_control, seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 - one bad property must not void the rest
            logger.exception("probe_failed property=%s", name)
            failed = _empty_property(f"probe failed: {exc}", num_layers, names, cv_folds)
            probes[name] = failed

    return {
        "num_layers": num_layers,
        "layer_names": names,
        "hidden_dim": int(hidden_dim),
        "projected_dim": projected_dim,
        "n_files": int(n_files),
        "properties": probes,
        "params": {
            "probe": probe,
            "cv_folds": int(cv_folds),
            "project_dims": int(project_dims),
            "min_class_count": int(min_class_count),
            "include_control": bool(include_control),
            "seed": int(seed),
        },
    }
