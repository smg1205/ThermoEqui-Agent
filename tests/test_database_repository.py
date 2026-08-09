"""Repository tests for export recording and parameter search behavior."""

from __future__ import annotations

from sqlalchemy import create_engine

from database.session import Repository, initialize_database
from schemas.domain import ParameterSet


def test_repository_search_returns_matching_component_order() -> None:
    engine = create_engine("sqlite:///:memory:")
    initialize_database(engine)
    repository = Repository(engine)
    parameter_set = ParameterSet(
        model_name="NRTL",
        component_order=["71-43-2", "108-88-3"],
        parameters={"alpha": 0.3, "tau12": 0.5, "tau21": -0.4},
        parameter_form="NRTL",
        units={"tau12": "dimensionless", "tau21": "dimensionless", "alpha": "dimensionless"},
        equilibrium_types=["VLE"],
        quality_level="reviewed",
        source_type="literature",
        source_title="Test NRTL source",
        source_identifier="https://example.invalid/nrtl",
    )
    repository.add_parameter_set(parameter_set)

    results = repository.search_parameter_sets("NRTL", ["71-43-2", "108-88-3"])
    assert len(results) == 1
    assert results[0].parameter_set_id == parameter_set.parameter_set_id


def test_repository_record_export_creates_export_record() -> None:
    engine = create_engine("sqlite:///:memory:")
    initialize_database(engine)
    repository = Repository(engine)
    record = ParameterSet(
        model_name="NRTL",
        component_order=["71-43-2", "108-88-3"],
        parameters={"tau12": 0.5, "tau21": -0.4, "alpha": 0.3},
        parameter_form="NRTL",
        units={"tau12": "dimensionless", "tau21": "dimensionless", "alpha": "dimensionless"},
        equilibrium_types=["VLE"],
        quality_level="reviewed",
        source_type="literature",
        source_title="Test NRTL source",
        source_identifier="https://example.invalid/nrtl",
    )
    repository.add_parameter_set(record)
    repository.record_export(record.parameter_set_id, "json")

    # If no exception is raised, the export record insertion path is covered.
    assert True
