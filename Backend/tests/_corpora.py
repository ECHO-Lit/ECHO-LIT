"""Bundled corpora -- skip guards, and a plugin that hides them.

Test Plan Section 5.  See tests/plans/5-risks-dependencies-assumptions-constraints.md.

The bundled corpora (SAA, L2-ARCTIC, Common Voice, RAVDESS) live under
Backend/data/, which is gitignored: their licences keep them out of the
repository, so a clean checkout has none of them.  Every case that reads one
therefore carries `requires_corpora(...)` and skips on a clean checkout rather
than failing (TEST-06).

Loaded with `-p tests._corpora`, this module is also a pytest plugin: it
repoints every corpus path the application reads at a directory that does not
exist, so a machine that has the corpora can still see what a clean checkout
sees (RD-04).  Nothing under Backend/data/ is touched.  `ECHO_KEEP_CORPORA`
keeps the named corpora visible, for partial checkouts:

    ECHO_KEEP_CORPORA=ravdess python -m pytest tests/test_dataset_serving_api.py -p tests._corpora
    ECHO_KEEP_CORPORA=datasets ...    # every dataset visible, the L2-ARCTIC annotations hidden

Only the standard library and pytest are imported at module level.  `-p`
imports this module before tests/conftest.py sets offline mode (TEST-08), so an
`app` import here would import the application -- and transformers -- online.
"""
from __future__ import annotations

import operator
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REASON = "(Backend/data/ is gitignored)"

# A corpus the app reads outside the dataset registry.
ANNOTATIONS = "l2-arctic-annotations"
# `ECHO_KEEP_CORPORA=datasets` keeps every registered dataset.
ALL_DATASETS = "datasets"


@dataclass(frozen=True)
class Disposition:
    """What the plugin does with one app module's paths into Backend/data/."""

    module: str
    repointed: tuple[str, ...] = ()
    exempt: str = ""


# Every app/ file that builds a path into Backend/data/.  RD-05 rescans app/ and
# fails when a file appears or disappears, and checks that hiding really moves
# every Path-valued global in the repointed modules.
HIDDEN: dict[str, Disposition] = {
    "app/services/dataset_service.py": Disposition(
        "app.services.dataset_service", ("DATA_DIR", "DATASET_PATHS", "DATASET_BASE_DIRS")),
    "app/services/l2_arctic_annotations.py": Disposition(
        "app.services.l2_arctic_annotations", ("_ANNOTATIONS_PATH",)),
    "app/api/routes/inferences.py": Disposition(
        "app.api.routes.inferences", ("DATA_DIR", "DATASET_DIRS")),
    "app/main.py": Disposition(
        "app.main", exempt="Path('data') is an allow-list root for legacy file_path payloads; "
                           "nothing is read from it"),
}


def _paths(name: str) -> tuple[Path, ...]:
    if name == ANNOTATIONS:
        from app.services import l2_arctic_annotations

        return (l2_arctic_annotations._ANNOTATIONS_PATH,)
    from app.services import dataset_service

    return (dataset_service.DATASET_PATHS[name], dataset_service.DATASET_BASE_DIRS[name])


def missing(*names: str, files: tuple[str | Path, ...] = ()) -> list[str]:
    """The named corpora and files that are absent.

    A dataset needs both its metadata CSV and its audio directory.  `files` are
    relative to `dataset_service.DATA_DIR`, or absolute.
    """
    from app.services import dataset_service

    absent = [name for name in names if not all(path.exists() for path in _paths(name))]
    absent += [Path(f).name for f in files if not (dataset_service.DATA_DIR / f).exists()]
    return absent


def corpora_present(*names: str, files: tuple[str | Path, ...] = ()) -> bool:
    return not missing(*names, files=files)


def requires_corpora(*names: str, files: tuple[str | Path, ...] = ()):
    """`skipif` for a case that reads bundled corpora.  Evaluated at collection."""
    absent = missing(*names, files=files)
    return pytest.mark.skipif(bool(absent), reason=f"Bundled corpus absent: {', '.join(absent)} {REASON}")


# --------------------------------------------------------------------------- the plugin

def _keep(raw: str) -> set[str]:
    return {key.strip() for key in raw.split(",") if key.strip()}


def hide(root: Path, keep: set[str] = frozenset(), *, setattr_=setattr, setitem=operator.setitem) -> None:
    """Repoint every corpus path in HIDDEN at `root`, which is never created.

    Dict values are replaced in place, because modules that imported the dict
    hold the same object.  `setattr_`/`setitem` let RD-05 route the changes
    through monkeypatch so they are undone.
    """
    from app.services import dataset_service, l2_arctic_annotations

    # The real Backend/data/, computed as the app computes it, and wherever an
    # earlier hide() already moved it -- a module imported after that one (the
    # legacy routes, say) still holds real paths.
    bases = (Path(dataset_service.__file__).resolve().parents[2] / "data", dataset_service.DATA_DIR)
    keep_datasets = ALL_DATASETS in keep

    def rehome(path: Path) -> Path:
        for base in bases:
            try:
                return root / path.relative_to(base)
            except ValueError:
                continue
        return path

    for table in (dataset_service.DATASET_PATHS, dataset_service.DATASET_BASE_DIRS):
        for name, path in list(table.items()):
            if not (keep_datasets or name in keep):
                setitem(table, name, rehome(path))
    setattr_(dataset_service, "DATA_DIR", root)
    dataset_service._metadata_cache.clear()

    if ANNOTATIONS not in keep:
        setattr_(l2_arctic_annotations, "_ANNOTATIONS_PATH", rehome(l2_arctic_annotations._ANNOTATIONS_PATH))
    l2_arctic_annotations._load_annotations.cache_clear()

    # The legacy routes are imported only when ENABLE_LEGACY_SYNC_INFERENCE is on.
    inferences = sys.modules.get("app.api.routes.inferences")
    if inferences is not None:
        for name, path in list(inferences.DATASET_DIRS.items()):
            if not (keep_datasets or name in keep):
                setitem(inferences.DATASET_DIRS, name, rehome(path))
        setattr_(inferences, "DATA_DIR", root)


def hidden_root() -> Path:
    return Path(tempfile.gettempdir()) / f"echo-corpora-hidden-{os.getpid()}"


def pytest_configure(config):
    """Active only via `-p tests._corpora`; tests/conftest.py is loaded by now."""
    hide(hidden_root(), _keep(os.environ.get("ECHO_KEEP_CORPORA", "")))


def pytest_report_header(config):
    kept = os.environ.get("ECHO_KEEP_CORPORA", "") or "none"
    return f"bundled corpora hidden by tests._corpora (kept: {kept})"
