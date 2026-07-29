"""Pydantic schemas for static model catalog metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ImplementationStatus = Literal["available", "contract_only", "planned"]


class ModelCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    aliases: list[str] = Field(default_factory=list)
    family: str
    backend: str
    implementation_status: ImplementationStatus
    production_ready: bool
    supported_equilibrium_types: list[Literal["VLE", "LLE", "FLASH"]] = Field(default_factory=list)
    supported_calculation_types: list[str] = Field(default_factory=list)
    excluded_systems: list[str] = Field(default_factory=list)
    pressure_regime: list[str] = Field(default_factory=list)
    requires_binary_parameters: bool
    parameter_requirements: list[str] = Field(default_factory=list)
    validation_requirements: list[str] = Field(default_factory=list)
    scope_notes: list[str] = Field(default_factory=list)
    optional_dependency: str | None = None
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("name must not be empty after trimming whitespace")
        return normalized

    @field_validator("aliases")
    @classmethod
    def deduplicate_aliases(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        deduplicated: list[str] = []
        for item in value:
            if item not in seen:
                seen.add(item)
                deduplicated.append(item)
        return deduplicated

    @model_validator(mode="after")
    def validate_consistency(self) -> ModelCatalogEntry:
        if self.name in self.aliases:
            raise ValueError("name must not appear in aliases")
        if self.production_ready and self.implementation_status != "available":
            raise ValueError("production_ready=true requires implementation_status=available")
        if self.production_ready and self.backend == "contract_only":
            raise ValueError("production_ready=true is invalid when backend=contract_only")
        return self
