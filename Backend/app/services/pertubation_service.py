"""Deprecated spelling. Kept so existing imports (worker.executor,
api.routes.perturbations) keep working during the FR-7 migration.
Remove one release after all call sites move to perturbation_service."""
from app.services.perturbation_service import (  # noqa: F401
    add_gaussian_noise,
    apply_time_masking,
    apply_frequency_masking,
    apply_pitch_shift,
    apply_time_stretch,
    apply_perturbations,
    perturb_and_save,
)
