"""Loader and seeder for reviewed production parameter sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from database.session import Repository
from schemas.domain import ParameterSet


def production_parameter_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "knowledge" / "parameters"


def load_production_parameter_sets() -> list[ParameterSet]:
    """Load every production YAML record, rejecting test fixtures and duplicates."""
    directory = production_parameter_directory()
    parameter_sets: list[ParameterSet] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is None:
            raise ValueError(f"Parameter file {path} is empty; expected a YAML list.")
        if not isinstance(loaded, list):
            raise ValueError(f"Parameter file {path} must contain a top-level YAML list.")
        for index, item in enumerate(loaded):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: entry {index} must be a YAML mapping.")
            parameter_set = ParameterSet.model_validate(item)
            if parameter_set.source_type == "test_fixture":
                raise ValueError(f"{path}: test_fixture parameter sets cannot enter the production repository.")
            if parameter_set.parameter_set_id in seen:
                raise ValueError(f"Duplicate parameter_set_id {parameter_set.parameter_set_id!r} in production files.")
            seen.add(parameter_set.parameter_set_id)
            parameter_sets.append(parameter_set)
    return parameter_sets


@dataclass(frozen=True)
class SeedResult:
    added: int
    updated: int
    unchanged: int
    removed: int = 0

    @property
    def total(self) -> int:
        return self.added + self.updated + self.unchanged


def seed_production_parameters(repository: Repository) -> SeedResult:
    """Idempotently upsert every reviewed production parameter set into the database."""
    added = 0
    updated = 0
    unchanged = 0
    production_sets = load_production_parameter_sets()
    for parameter_set in production_sets:
        outcome = repository.upsert_parameter_set(parameter_set)
        if outcome == "added":
            added += 1
        elif outcome == "updated":
            updated += 1
        else:
            unchanged += 1
    removed = repository.delete_duplicate_parameter_sets(production_sets)
    return SeedResult(added=added, updated=updated, unchanged=unchanged, removed=removed)
