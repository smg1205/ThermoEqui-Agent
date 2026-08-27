"""Behavioral tests for the RK pilot backend and its parameter contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.router import recommend_models
from schemas.domain import (
    ComponentIdentity,
    ParameterSet,
    TaskManifest,
    ThermodynamicConditions,
)
from thermo_engine import calculate_equilibrium, validate_equilibrium_result
from thermo_engine.errors import ThermoEquiError
from thermo_engine.parameters import has_rk_kij, parameter_set_to_backend_params
from thermo_engine.service import resolve_backend

METHANE = ComponentIdentity(component_id="methane", name="Methane", cas_number="74-82-8")
ETHANE = ComponentIdentity(component_id="ethane", name="Ethane", cas_number="74-84-0")
PROPANE = ComponentIdentity(component_id="propane", name="Propane", cas_number="74-98-6")


def _fixture_parameter_set(name: str) -> ParameterSet:
    path = Path(__file__).parent / "fixtures" / name
    return ParameterSet.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _flash_task(
    parameter_set: ParameterSet,
    *,
    components: list[ComponentIdentity] | None = None,
) -> TaskManifest:
    return TaskManifest(
        equilibrium_type="FLASH",
        calculation_type="tp_flash",
        components=components or [METHANE, ETHANE],
        conditions=ThermodynamicConditions(
            temperature_K=150.0,
            pressure_kPa=530.0,
            feed_composition=[0.8, 0.2],
        ),
        model_name="RK",
        parameters=[parameter_set],
    )


def test_rk_tp_flash_crosses_public_validation_gate() -> None:
    task = _flash_task(_fixture_parameter_set("rk_kij_methane_ethane.json"))

    result = calculate_equilibrium(task)
    report = validate_equilibrium_result(result)

    assert result.model_name == "RK"
    assert result.backend_version.startswith("thermo/")
    assert result.phase_state == "two_phase"
    assert report.material_balance.passed
    assert report.equilibrium_residual.passed
    assert report.convergence.passed


def test_rk_requires_explicit_kij_parameter_set() -> None:
    task = _flash_task(_fixture_parameter_set("rk_kij_methane_ethane.json")).model_copy(update={"parameters": []})

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "missing_parameters"
    assert captured.value.detail.details["model"] == "RK"
    assert captured.value.detail.details["missing_pairs"] == [["74-82-8", "74-84-0"]]


def test_rk_rejects_incomplete_parameter_set() -> None:
    incomplete = ParameterSet(
        model_name="RK",
        component_order=["74-82-8", "74-84-0"],
        parameters={"a": 0.1},
        parameter_form="RK kij",
        units={"a": "dimensionless"},
        equilibrium_types=["VLE", "FLASH"],
        source_type="user_supplied",
        quality_level="unreviewed-test-input",
    )

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(_flash_task(incomplete))

    assert captured.value.detail.failure_type == "missing_parameters"
    assert "does not contain a complete kij entry" in captured.value.detail.message


def test_rk_pilot_is_binary_only() -> None:
    task = _flash_task(
        _fixture_parameter_set("rk_kij_methane_ethane.json"),
        components=[METHANE, ETHANE, PROPANE],
    )

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "unsupported_model"
    assert "binary-only" in captured.value.detail.message


def test_rk_lle_is_rejected() -> None:
    task = TaskManifest(
        equilibrium_type="LLE",
        calculation_type="lle",
        components=[METHANE, ETHANE],
        conditions=ThermodynamicConditions(temperature_K=150.0),
        model_name="RK",
        parameters=[_fixture_parameter_set("rk_kij_methane_ethane.json")],
    )

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "unsupported_model"
    assert "does not implement lle" in captured.value.detail.message


def test_rk_parameter_sources_include_parameter_set_identity() -> None:
    parameter_set = _fixture_parameter_set("rk_kij_methane_ethane.json")
    task = _flash_task(parameter_set)

    sources = resolve_backend(task).parameter_sources(task)

    parameter_source = next(
        source for source in sources if source.get("property") == "RK binary interaction parameter kij"
    )
    assert parameter_source["parameter_set_id"] == parameter_set.parameter_set_id
    assert parameter_source["source_type"] == "test_fixture"
    assert parameter_source["parameter_values"] == '{"kij": 0.03}'


def test_rk_enters_score_based_selection_when_parameters_are_available() -> None:
    parameter_set = _fixture_parameter_set("rk_kij_methane_ethane.json")
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[METHANE, ETHANE],
        conditions=ThermodynamicConditions(pressure_kPa=800.0, liquid_composition=[0.5, 0.5]),
        parameters=[parameter_set],
    )

    recommendations = recommend_models(task, available_parameter_models={"RK"})
    rk = next(item for item in recommendations if item.model_name == "RK")

    assert rk.executable is True


def test_rk_is_not_advertised_for_multicomponent_tasks() -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="isobaric_vle",
        components=[METHANE, ETHANE, PROPANE],
        conditions=ThermodynamicConditions(pressure_kPa=800.0),
    )

    recommendations = recommend_models(task, available_parameter_models={"RK"})
    rk = next(item for item in recommendations if item.model_name == "RK")

    assert rk.executable is False
    assert any("binary-only" in exclusion for exclusion in rk.exclusions)


def test_rk_parameter_helpers_accept_complete_binary_set() -> None:
    parameter_set = _fixture_parameter_set("rk_kij_methane_ethane.json")

    assert has_rk_kij([parameter_set], [METHANE, ETHANE]) is True
    assert parameter_set_to_backend_params(parameter_set, "RK", ["74-82-8", "74-84-0"]) == {"kij": 0.03}


def test_methane_propane_pilot_system_crosses_validation_gate() -> None:
    parameter_set = _fixture_parameter_set("rk_kij_methane_propane.json")
    task = TaskManifest(
        equilibrium_type="FLASH",
        calculation_type="tp_flash",
        components=[METHANE, PROPANE],
        conditions=ThermodynamicConditions(
            temperature_K=180.0,
            pressure_kPa=1200.0,
            feed_composition=[0.6, 0.4],
        ),
        model_name="RK",
        parameters=[parameter_set],
    )

    result = calculate_equilibrium(task)
    report = validate_equilibrium_result(result)

    assert result.model_name == "RK"
    assert report.composition_balance.passed
    assert report.material_balance.passed
    assert report.equilibrium_residual.passed
    assert report.convergence.passed


def test_rk_isobaric_vle_curve_includes_pure_endpoints() -> None:
    parameter_set = _fixture_parameter_set("rk_kij_methane_ethane.json")
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="isobaric_vle",
        components=[METHANE, ETHANE],
        conditions=ThermodynamicConditions(pressure_kPa=530.0),
        model_name="RK",
        parameters=[parameter_set],
        points=5,
    )

    result = calculate_equilibrium(task)
    report = validate_equilibrium_result(result)

    assert len(result.points) == 5
    assert result.points[0].liquid_composition == pytest.approx([0.0, 1.0])
    assert result.points[-1].liquid_composition == pytest.approx([1.0, 0.0])
    assert report.equilibrium_residual.passed
    assert report.composition_balance.passed
