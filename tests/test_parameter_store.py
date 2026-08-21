"""Contract tests for the reviewed production parameter store and seed tool."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from database.session import Repository, initialize_database
from schemas.domain import ParameterSet
from thermo_engine.parameter_store import (
    load_production_parameter_sets,
    seed_production_parameters,
)


def test_production_parameter_files_validate_and_cover_activity_coefficient_models() -> None:
    parameter_sets = load_production_parameter_sets()

    assert len(parameter_sets) >= 10
    assert {item.model_name for item in parameter_sets} == {"NRTL", "UNIQUAC", "Wilson"}
    assert all(item.source_type != "test_fixture" for item in parameter_sets)
    assert all(item.quality_level for item in parameter_sets)
    assert len({item.parameter_set_id for item in parameter_sets}) == len(parameter_sets)


def test_production_parameter_sets_round_trip_through_repository() -> None:
    engine = create_engine("sqlite:///:memory:")
    initialize_database(engine)
    repository = Repository(engine)

    first = seed_production_parameters(repository)
    second = seed_production_parameters(repository)

    assert first.total >= 10
    assert first.added == first.total
    assert second.added == 0
    assert second.unchanged == first.total
    assert second.removed == 0
    results = repository.search_parameter_sets("NRTL", ["ethanol", "water"])
    assert len(results) == 1
    assert results[0].parameter_set_id == "chemsep-nrtl-ethanol-water"


def test_seed_removes_stale_duplicate_parameter_sets() -> None:
    engine = create_engine("sqlite:///:memory:")
    initialize_database(engine)
    repository = Repository(engine)
    seed_production_parameters(repository)

    stale = ParameterSet(
        model_name="NRTL",
        component_order=["ethanol", "water"],
        parameters={"tau12": 0.3, "tau21": 0.1, "alpha": 0.3},
        parameter_form="NRTL",
        units={"tau12": "dimensionless", "tau21": "dimensionless", "alpha": "dimensionless"},
        equilibrium_types=["VLE"],
        source_type="literature",
        source_title="Stale legacy source",
        source_identifier="legacy-registry:stale",
        quality_level="legacy-reviewed-import",
    )
    repository.add_parameter_set(stale)

    result = seed_production_parameters(repository)

    assert result.removed == 1
    results = repository.search_parameter_sets("NRTL", ["ethanol", "water"])
    assert len(results) == 1
    assert results[0].parameter_set_id == "chemsep-nrtl-ethanol-water"


def test_production_loader_rejects_test_fixture_records(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    (tmp_path / "bad.yaml").write_text(
        "- parameter_set_id: bad-fixture\n"
        "  model_name: NRTL\n"
        "  component_order: [a, b]\n"
        "  parameters: {tau12: 1.0, tau21: 1.0, alpha: 0.3}\n"
        "  parameter_form: NRTL\n"
        "  units: {tau12: dimensionless, tau21: dimensionless, alpha: dimensionless}\n"
        "  equilibrium_types: [VLE]\n"
        "  source_type: test_fixture\n"
        "  quality_level: synthetic\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("thermo_engine.parameter_store.production_parameter_directory", lambda: tmp_path)

    with pytest.raises(ValueError, match="cannot enter the production repository"):
        load_production_parameter_sets()


def test_upsert_refreshes_changed_production_record() -> None:
    engine = create_engine("sqlite:///:memory:")
    initialize_database(engine)
    repository = Repository(engine)
    parameter_set = ParameterSet(
        model_name="NRTL",
        component_order=["ethanol", "water"],
        parameters={"tau12": 0.3, "tau21": 0.1, "alpha": 0.3},
        parameter_form="NRTL",
        units={"tau12": "dimensionless", "tau21": "dimensionless", "alpha": "dimensionless"},
        equilibrium_types=["VLE"],
        source_type="literature",
        source_title="Source title",
        source_identifier="source-id",
        quality_level="reviewed",
    )

    assert repository.upsert_parameter_set(parameter_set) == "added"
    changed = parameter_set.model_copy(update={"quality_level": "reviewed-updated"})
    assert repository.upsert_parameter_set(changed) == "updated"
    assert repository.upsert_parameter_set(changed) == "unchanged"
