"""Behavioral tests for the SRK pilot backend and its parameter contract."""

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
from thermo_engine.parameters import has_srk_kij, parameter_set_to_backend_params
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
        model_name="SRK",
        parameters=[parameter_set],
    )


def test_srk_tp_flash_crosses_public_validation_gate() -> None:
    task = _flash_task(_fixture_parameter_set("srk_kij_methane_ethane.json"))

    result = calculate_equilibrium(task)
    report = validate_equilibrium_result(result)

    assert result.model_name == "SRK"
    assert result.backend_version.startswith("thermo/")
    assert result.phase_state == "two_phase"
    assert result.vapor_fraction == pytest.approx(0.637, abs=0.01)
    assert report.material_balance.passed
    assert report.equilibrium_residual.passed
    assert report.convergence.passed


def test_srk_bubble_and_dew_points_are_self_consistent() -> None:
    parameter_set = _fixture_parameter_set("srk_kij_methane_ethane.json")
    bubble_task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[METHANE, ETHANE],
        conditions=ThermodynamicConditions(
            pressure_kPa=530.0,
            liquid_composition=[0.5, 0.5],
        ),
        model_name="SRK",
        parameters=[parameter_set],
    )
    bubble = calculate_equilibrium(bubble_task)
    dew = calculate_equilibrium(
        bubble_task.model_copy(
            update={
                "calculation_type": "dew_point",
                "conditions": ThermodynamicConditions(
                    pressure_kPa=530.0,
                    vapor_composition=bubble.points[0].vapor_composition,
                ),
            }
        )
    )

    assert dew.temperature_K == pytest.approx(bubble.temperature_K, abs=1e-4)
    assert dew.points[0].liquid_composition == pytest.approx(
        bubble.points[0].liquid_composition,
        abs=1e-4,
    )


def test_srk_requires_explicit_kij_parameter_set() -> None:
    task = _flash_task(_fixture_parameter_set("srk_kij_methane_ethane.json")).model_copy(update={"parameters": []})

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "missing_parameters"
    assert captured.value.detail.details["model"] == "SRK"
    assert captured.value.detail.details["missing_pairs"] == [["74-82-8", "74-84-0"]]


def test_srk_rejects_incomplete_parameter_set() -> None:
    incomplete = ParameterSet(
        model_name="SRK",
        component_order=["74-82-8", "74-84-0"],
        parameters={"a": 0.1},
        parameter_form="SRK kij",
        units={"a": "dimensionless"},
        equilibrium_types=["VLE", "FLASH"],
        source_type="user_supplied",
        quality_level="unreviewed-test-input",
    )
    task = _flash_task(incomplete)

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "missing_parameters"
    assert "does not contain a complete kij entry" in captured.value.detail.message


def test_srk_rejects_component_order_mismatch() -> None:
    parameter_set = _fixture_parameter_set("srk_kij_methane_ethane.json").model_copy(
        update={"component_order": ["74-84-0", "74-82-8"]}
    )
    task = _flash_task(parameter_set)

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "missing_parameters"


def test_srk_pilot_is_binary_only() -> None:
    task = _flash_task(
        _fixture_parameter_set("srk_kij_methane_ethane.json"),
        components=[METHANE, ETHANE, PROPANE],
    )

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "unsupported_model"
    assert "binary-only" in captured.value.detail.message


def test_srk_lle_is_rejected() -> None:
    task = TaskManifest(
        equilibrium_type="LLE",
        calculation_type="lle",
        components=[METHANE, ETHANE],
        conditions=ThermodynamicConditions(temperature_K=150.0),
        model_name="SRK",
        parameters=[_fixture_parameter_set("srk_kij_methane_ethane.json")],
    )

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "unsupported_model"
    assert "does not implement lle" in captured.value.detail.message


def test_srk_parameter_sources_include_parameter_set_identity() -> None:
    parameter_set = _fixture_parameter_set("srk_kij_methane_ethane.json")
    task = _flash_task(parameter_set)

    sources = resolve_backend(task).parameter_sources(task)

    parameter_source = next(
        source for source in sources if source.get("property") == "SRK binary interaction parameter kij"
    )
    assert parameter_source["parameter_set_id"] == parameter_set.parameter_set_id
    assert parameter_source["source_type"] == "test_fixture"
    assert parameter_source["parameter_values"] == '{"kij": 0.0026}'


def test_srk_enters_score_based_selection_when_parameters_are_available() -> None:
    parameter_set = _fixture_parameter_set("srk_kij_methane_ethane.json")
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[METHANE, ETHANE],
        conditions=ThermodynamicConditions(pressure_kPa=800.0, liquid_composition=[0.5, 0.5]),
        parameters=[parameter_set],
    )

    recommendations = recommend_models(task, available_parameter_models={"SRK"})
    srk = next(item for item in recommendations if item.model_name == "SRK")

    assert srk.executable is True


def test_srk_is_not_advertised_for_multicomponent_tasks() -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="isobaric_vle",
        components=[METHANE, ETHANE, PROPANE],
        conditions=ThermodynamicConditions(pressure_kPa=800.0),
    )

    recommendations = recommend_models(task, available_parameter_models={"SRK"})
    srk = next(item for item in recommendations if item.model_name == "SRK")

    assert srk.executable is False
    assert any("binary-only" in exclusion for exclusion in srk.exclusions)


def test_srk_parameter_helpers_accept_complete_binary_set() -> None:
    parameter_set = _fixture_parameter_set("srk_kij_methane_ethane.json")

    assert has_srk_kij([parameter_set], [METHANE, ETHANE]) is True
    assert parameter_set_to_backend_params(parameter_set, "SRK", ["74-82-8", "74-84-0"]) == {"kij": 0.0026}


def test_methane_propane_pilot_system_crosses_validation_gate() -> None:
    parameter_set = _fixture_parameter_set("srk_kij_methane_propane.json")
    task = TaskManifest(
        equilibrium_type="FLASH",
        calculation_type="tp_flash",
        components=[METHANE, PROPANE],
        conditions=ThermodynamicConditions(
            temperature_K=180.0,
            pressure_kPa=1200.0,
            feed_composition=[0.6, 0.4],
        ),
        model_name="SRK",
        parameters=[parameter_set],
    )

    result = calculate_equilibrium(task)
    report = validate_equilibrium_result(result)

    assert result.model_name == "SRK"
    assert report.composition_balance.passed
    assert report.material_balance.passed
    assert report.equilibrium_residual.passed
    assert report.convergence.passed


def test_srk_isothermal_vle_curve_includes_pure_endpoints() -> None:
    parameter_set = _fixture_parameter_set("srk_kij_methane_ethane.json")
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="isothermal_vle",
        components=[METHANE, ETHANE],
        conditions=ThermodynamicConditions(temperature_K=150.0),
        model_name="SRK",
        parameters=[parameter_set],
        points=5,
    )

    result = calculate_equilibrium(task)
    report = validate_equilibrium_result(result)

    assert len(result.points) == 5
    assert result.points[0].liquid_composition == pytest.approx([0.0, 1.0])
    assert result.points[-1].liquid_composition == pytest.approx([1.0, 0.0])
    assert result.points[0].pressure_kPa == pytest.approx(9.1925, abs=0.01)
    assert result.points[-1].pressure_kPa == pytest.approx(1051.15, abs=1.0)
    assert report.equilibrium_residual.passed
    assert report.composition_balance.passed


def test_srk_isobaric_vle_curve_includes_pure_endpoints() -> None:
    parameter_set = _fixture_parameter_set("srk_kij_methane_ethane.json")
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="isobaric_vle",
        components=[METHANE, ETHANE],
        conditions=ThermodynamicConditions(pressure_kPa=530.0),
        model_name="SRK",
        parameters=[parameter_set],
        points=5,
    )

    result = calculate_equilibrium(task)
    report = validate_equilibrium_result(result)

    assert len(result.points) == 5
    assert result.points[0].liquid_composition == pytest.approx([0.0, 1.0])
    assert result.points[-1].liquid_composition == pytest.approx([1.0, 0.0])
    assert result.points[0].temperature_K == pytest.approx(221.94, abs=0.05)
    assert result.points[-1].temperature_K == pytest.approx(136.37, abs=0.05)
    assert report.equilibrium_residual.passed
    assert report.composition_balance.passed
