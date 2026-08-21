"""Behavioral tests for parameter sets carried on TaskManifest."""

from __future__ import annotations

import pytest

from schemas.domain import ComponentIdentity, ParameterSet, TaskManifest, ThermodynamicConditions
from thermo_engine import calculate_equilibrium
from thermo_engine.errors import ThermoEquiError
from thermo_engine.service import resolve_backend

ETHANOL = ComponentIdentity(component_id="ethanol", name="ethanol", cas_number="64-17-5")
WATER = ComponentIdentity(component_id="water", name="water", cas_number="7732-18-5")


def nrtl_parameter_set() -> ParameterSet:
    return ParameterSet(
        model_name="NRTL",
        component_order=["ethanol", "water"],
        parameters={"tau12": 0.3, "tau21": 0.1, "alpha": 0.3},
        parameter_form="NRTL",
        units={"tau12": "dimensionless", "tau21": "dimensionless", "alpha": "dimensionless"},
        equilibrium_types=["VLE"],
        source_type="user_supplied",
        quality_level="test-only",
        notes="Synthetic values for parameter-pipeline tests; not engineering evidence.",
    )


def task_with_parameters() -> TaskManifest:
    return TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[ETHANOL, WATER],
        conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
        model_name="NRTL",
        parameters=[nrtl_parameter_set()],
    )


def test_request_parameter_set_drives_activity_coefficient_backend() -> None:
    result = calculate_equilibrium(task_with_parameters())

    assert result.model_name == "NRTL"
    assert result.temperature_K == pytest.approx(354.0, abs=15.0)
    assert len(result.points) == 1
    assert all(0.0 <= value <= 1.0 for value in result.points[0].vapor_composition)


def test_parameter_sources_include_parameter_set_identity() -> None:
    parameter_set = nrtl_parameter_set()
    task = task_with_parameters().model_copy(update={"parameters": [parameter_set]})
    sources = resolve_backend(task).parameter_sources(task)

    assert len(sources) == 1
    assert sources[0]["parameter_set_id"] == parameter_set.parameter_set_id
    assert sources[0]["source_title"] == "user-supplied"


def test_mismatched_component_order_fails_as_missing_parameters() -> None:
    parameter_set = nrtl_parameter_set().model_copy(update={"component_order": ["water", "ethanol"]})
    task = task_with_parameters().model_copy(update={"parameters": [parameter_set]})

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "missing_parameters"


def test_unsupported_parameter_form_is_rejected() -> None:
    parameter_set = nrtl_parameter_set().model_copy(update={"parameter_form": "unknown-form"})
    task = task_with_parameters().model_copy(update={"parameters": [parameter_set]})

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "missing_parameters"
