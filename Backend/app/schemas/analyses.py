"""FR-7 request/response schemas for POST /analyses/linguistic-vs-acoustic
(docs/FR7plan.md Part 1 S5.2)."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.jobs import JobStatus
from app.services.fr7_planning import MAX_GRID_VARIANTS, STOCHASTIC_PROPERTIES


class _Sweep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: int = Field(default=7, ge=2, le=15)


class PitchSweep(_Sweep):
    property: Literal["pitch"] = "pitch"
    start: float = Field(default=-6.0, ge=-12.0, le=12.0)   # semitones
    stop: float = Field(default=6.0, ge=-12.0, le=12.0)


class SpeedSweep(_Sweep):
    property: Literal["speed"] = "speed"
    start: float = Field(default=0.7, ge=0.5, le=2.0)       # rate multiplier
    stop: float = Field(default=1.4, ge=0.5, le=2.0)


class NoiseSweep(_Sweep):
    property: Literal["noise"] = "noise"
    start: float = Field(default=40.0, ge=-10.0, le=60.0)   # dB SNR, descending
    stop: float = Field(default=0.0, ge=-10.0, le=60.0)
    repeats: int = Field(default=3, ge=1, le=10)            # stochastic -> average


class TimeMaskSweep(_Sweep):
    property: Literal["time_mask"] = "time_mask"
    start: float = Field(default=0.0, ge=0.0, le=80.0)      # % of duration
    stop: float = Field(default=50.0, ge=0.0, le=80.0)


class FreqMaskSweep(_Sweep):
    property: Literal["freq_mask"] = "freq_mask"
    band_low_hz: float = Field(default=0.0, ge=0.0, le=8000.0)
    start: float = Field(default=500.0, ge=100.0, le=8000.0)   # band width, Hz
    stop: float = Field(default=4000.0, ge=100.0, le=8000.0)


PropertySweep = Annotated[
    Union[PitchSweep, SpeedSweep, NoiseSweep, TimeMaskSweep, FreqMaskSweep],
    Field(discriminator="property"),
]


class LinguisticAcousticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_ids: list[str] = Field(min_length=1, max_length=25)
    model: str
    task: Literal["transcription", "classification", "auto"] = "auto"
    sweeps: list[PropertySweep] = Field(min_length=1, max_length=5)
    reference_transcript: str | None = Field(default=None, max_length=20_000)
    language: str | None = "en"
    normalize_loudness: bool = True
    include_lexical_control: bool = True

    @model_validator(mode="after")
    def validate_grid(self) -> "LinguisticAcousticRequest":
        seen: set[str] = set()
        total = 1 + (1 if self.include_lexical_control else 0)
        for sweep in self.sweeps:
            if sweep.property in seen:
                raise ValueError(f"Duplicate sweep for property '{sweep.property}'")
            seen.add(sweep.property)
            if sweep.start == sweep.stop:
                raise ValueError(f"'{sweep.property}' sweep has a zero-width range")
            repeats = sweep.repeats if sweep.property in STOCHASTIC_PROPERTIES else 1
            total += sweep.steps * repeats
        total *= len(self.audio_ids)
        if total > MAX_GRID_VARIANTS:
            raise ValueError(
                f"Grid expands to {total} variants; the limit is {MAX_GRID_VARIANTS}. "
                f"Reduce steps, repeats, or the number of audio items."
            )
        return self


class LinguisticAcousticAccepted(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.queued
    status_url: str
    result_url: str
    estimated_variants: int
    estimated_seconds: float
    poll_after_ms: int = 1000
