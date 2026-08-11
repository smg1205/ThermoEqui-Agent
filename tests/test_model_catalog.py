from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

from thermo_engine.model_catalog import get_model_catalog_entry, load_model_catalog

EXPECTED_NAMES = {
    "Ideal/Raoult",
    "Peng-Robinson",
    "Phasepy/Peng-Robinson",
    "Clapeyron/Peng-Robinson",
    "SRK",
    "Wilson",
    "NRTL",
    "UNIQUAC",
}


def test_real_model_catalog_yaml_files_all_load() -> None:
    catalog = load_model_catalog()

    assert len(catalog) == 8
    assert set(catalog) == EXPECTED_NAMES


def test_get_model_catalog_entry_returns_expected_entry() -> None:
    entry = get_model_catalog_entry("Peng-Robinson")

    assert entry is not None
    assert entry.name == "Peng-Robinson"
    assert entry.backend == "thermo"
    assert entry.implementation_status == "available"


def test_get_model_catalog_entry_returns_none_for_unknown_name() -> None:
    assert get_model_catalog_entry("does-not-exist") is None


def test_production_ready_entries_must_be_available() -> None:
    catalog = load_model_catalog()

    for entry in catalog.values():
        if entry.production_ready:
            assert entry.implementation_status == "available"


def test_contract_only_entries_must_not_be_production_ready() -> None:
    catalog = load_model_catalog()

    for entry in catalog.values():
        if entry.implementation_status == "contract_only":
            assert entry.production_ready is False


def test_optional_external_backends_are_not_production_ready() -> None:
    catalog = load_model_catalog()

    assert catalog["Phasepy/Peng-Robinson"].production_ready is False
    assert catalog["Clapeyron/Peng-Robinson"].production_ready is False


def test_activity_coefficient_backends_are_available_but_not_production_ready() -> None:
    catalog = load_model_catalog()

    for name in ("NRTL", "UNIQUAC", "Wilson"):
        assert catalog[name].implementation_status == "available"
        assert catalog[name].production_ready is False


def test_ideal_raoult_does_not_claim_lle_support() -> None:
    ideal = load_model_catalog()["Ideal/Raoult"]

    assert "LLE" not in ideal.supported_equilibrium_types
    assert "lle" not in ideal.supported_calculation_types


def test_activity_coefficient_catalog_entries_do_not_claim_executable_lle_support() -> None:
    catalog = load_model_catalog()

    for name in ("NRTL", "UNIQUAC", "Wilson"):
        assert "LLE" not in catalog[name].supported_equilibrium_types
        assert "lle" not in catalog[name].supported_calculation_types


def test_empty_yaml_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "empty.yaml").write_text("", encoding="utf-8")
        monkeypatch.setattr("thermo_engine.model_catalog.catalog_directory", lambda: temp_path)

        with pytest.raises(ValueError, match="is empty"):
            load_model_catalog()


def test_top_level_list_yaml_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "list.yaml").write_text("- not\n- a\n- mapping\n", encoding="utf-8")
        monkeypatch.setattr("thermo_engine.model_catalog.catalog_directory", lambda: temp_path)

        with pytest.raises(ValueError, match="top-level YAML mapping"):
            load_model_catalog()


def test_duplicate_name_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    first = """\
name: Duplicate
aliases: []
family: activity_coefficient
backend: internal
implementation_status: available
production_ready: true
supported_equilibrium_types: [VLE]
supported_calculation_types: [bubble_point]
excluded_systems: []
pressure_regime: [low]
requires_binary_parameters: false
parameter_requirements: []
validation_requirements: [composition_balance]
scope_notes: []
optional_dependency:
source_refs: []
"""
    second = """\
name: Duplicate
aliases: []
family: cubic_eos
backend: thermo
implementation_status: available
production_ready: true
supported_equilibrium_types: [FLASH]
supported_calculation_types: [tp_flash]
excluded_systems: []
pressure_regime: [moderate]
requires_binary_parameters: true
parameter_requirements: []
validation_requirements: [equilibrium_residual]
scope_notes: []
optional_dependency:
source_refs: []
"""
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "a.yaml").write_text(first, encoding="utf-8")
        (temp_path / "b.yaml").write_text(second, encoding="utf-8")
        monkeypatch.setattr("thermo_engine.model_catalog.catalog_directory", lambda: temp_path)

        with pytest.raises(ValueError, match="Duplicate model catalog entry name"):
            load_model_catalog()


def test_unknown_field_raises_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    content = """\
name: Test Entry
aliases: []
family: activity_coefficient
backend: internal
implementation_status: available
production_ready: true
supported_equilibrium_types: [VLE]
supported_calculation_types: [bubble_point]
excluded_systems: []
pressure_regime: [low]
requires_binary_parameters: false
parameter_requirements: []
validation_requirements: [composition_balance]
scope_notes: []
optional_dependency:
source_refs: []
unexpected_field: should_fail
"""
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "invalid.yaml").write_text(content, encoding="utf-8")
        monkeypatch.setattr("thermo_engine.model_catalog.catalog_directory", lambda: temp_path)

        with pytest.raises(ValidationError):
            load_model_catalog()
