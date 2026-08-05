"""Which parameters actually change a *per-file* result.

The batch-level cache key legitimately covers every parameter: change anything
and the aggregate answer changes.  The per-file cache is different.  A
``layer_probe`` job carries labels, fold counts, probe type and projection width,
none of which affect the activations extracted from an individual audio file --
they only affect what is computed *from* them afterwards.

Keying per-file entries on the whole parameter dict would therefore throw away
every cached activation the moment a user nudged `cv_folds`, which is precisely
the interaction the feature is designed around: extraction is the expensive part
and probe configuration is what gets iterated on.

So per-file results are stored under an *extraction identity*: the operation that
produced them plus only the parameters that fed the model.  ``hidden_states`` and
``layer_probe`` collapse onto the same identity, which means the two operations
share cache entries in both directions -- running one warms the other.
"""

from __future__ import annotations

from typing import Any

from app.schemas.jobs import HiddenStatesParameters

# Operations whose per-file work is "run the encoder and pool each layer".
SHARED_EXTRACTION_OPERATIONS = frozenset({"hidden_states", "layer_probe"})

# The identity they share. Named for the operation a user can request directly.
EXTRACTION_OPERATION = "hidden_states"

# Derived from the schema rather than restated, so adding an extraction
# parameter cannot silently leave the cache keyed on a stale field set.
EXTRACTION_PARAMETER_NAMES = tuple(HiddenStatesParameters.model_fields)


def item_cache_identity(operation: str, parameters: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the ``(operation, parameters)`` a per-file result is cached under.

    Every operation not sharing an extraction identity is returned unchanged, so
    existing cache behaviour is untouched.
    """
    if operation not in SHARED_EXTRACTION_OPERATIONS:
        return operation, parameters
    extraction = {
        name: parameters[name]
        for name in EXTRACTION_PARAMETER_NAMES
        if name in parameters
    }
    return EXTRACTION_OPERATION, extraction
