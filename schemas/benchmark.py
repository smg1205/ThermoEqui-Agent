"""Pydantic schemas for model-agnostic benchmark cases and reports."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from schemas.domain import RunStatus, TaskManifest


class BenchmarkReferencePoint(BaseModel):
    temperature_K: float | None = Field(default=None, gt=0)
    pressure_kPa: float | None = Field(default=None, gt=0)
    vapor_fraction: float | None = Field(default=None, ge=0, le=1)
    liquid_composition: list[float] | None = None
    vapor_composition: list[float] | None = None

    @field_validator("liquid_composition", "vapor_composition")
    @classmethod
    def _bounded_composition(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and any(not 0.0 <= item <= 1.0 for item in value):
            raise ValueError("composition values must be mole fractions in [0, 1]")
        return value


class BenchmarkReference(BaseModel):
    points: list[BenchmarkReferencePoint] = Field(min_length=1)


class BenchmarkTolerances(BaseModel):
    temperature_K: float = Field(default=0.5, gt=0)
    pressure_kPa: float = Field(default=10.0, gt=0)
    composition: float = Field(default=0.01, gt=0)
    vapor_fraction: float = Field(default=0.01, gt=0)
    residual: float = Field(default=2e-6, gt=0)


class BenchmarkSource(BaseModel):
    title: str = Field(min_length=1)
    identifier: str = Field(min_length=1)
    kind: Literal["experimental", "literature", "simulation", "software_reference"]
    notes: str | None = None


class BenchmarkCase(BaseModel):
    case_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    task: TaskManifest
    reference: BenchmarkReference
    tolerances: BenchmarkTolerances = Field(default_factory=BenchmarkTolerances)
    source: BenchmarkSource
    applicable_models: list[str] = Field(default_factory=list)
    applicable_families: list[str] = Field(default_factory=list)
    notes: str | None = None


class BenchmarkMetric(BaseModel):
    field: str
    observed: float | None = None
    expected: float | None = None
    error: float | None = None
    tolerance: float
    passed: bool


class BenchmarkCaseResult(BaseModel):
    case_id: str
    model_name: str
    passed: bool
    metrics: list[BenchmarkMetric] = Field(default_factory=list)
    validation_status: RunStatus | None = None
    message: str | None = None


class BenchmarkSuiteReport(BaseModel):
    results: list[BenchmarkCaseResult] = Field(default_factory=list)
    passed: bool
    total: int
    passed_count: int
