"""Load static thermodynamic model catalog metadata from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from schemas.model_catalog import ModelCatalogEntry


def catalog_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "knowledge" / "model_catalog"


def load_model_catalog() -> dict[str, ModelCatalogEntry]:
    entries: dict[str, ModelCatalogEntry] = {}
    directory = catalog_directory()
    for path in sorted(directory.glob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is None:
            raise ValueError(f"Model catalog file {path} is empty; expected a YAML mapping.")
        if not isinstance(loaded, dict):
            raise ValueError(f"Model catalog file {path} must contain a top-level YAML mapping.")
        entry = ModelCatalogEntry.model_validate(loaded)
        if entry.name in entries:
            raise ValueError(f"Duplicate model catalog entry name {entry.name!r} found in {path}.")
        entries[entry.name] = entry
    return entries


def get_model_catalog_entry(name: str) -> ModelCatalogEntry | None:
    return load_model_catalog().get(name)
