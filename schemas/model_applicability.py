"""Schemas for model applicability filtering based on the static model catalog."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from schemas.domain import TaskManifest


class ModelAllowanceRequest(BaseModel):
    model_name: str
    calculation_type: str
    equilibrium_type: str
    available_parameters: set[str] = Field(default_factory=set)


class ModelAllowanceResult(BaseModel):
    allowed: bool
    reason: str


class ModelApplicabilityRequest(BaseModel):
    task: TaskManifest
    production_only: bool = True
    available_parameter_models: set[str] = Field(default_factory=set)


class ModelApplicabilityResult(BaseModel):
    model_name: str
    decision: Literal["keep", "exclude"]
    reasons: list[str] = Field(default_factory=list)


class ModelApplicabilityReport(BaseModel):
    results: list[ModelApplicabilityResult] = Field(default_factory=list)
