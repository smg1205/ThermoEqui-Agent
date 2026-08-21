"""Behavioral tests for the UNIFAC predictive pilot backend."""

from __future__ import annotations

import pytest

from agent.router import recommend_models
from schemas.domain import (
    ComponentIdentity,
    TaskManifest,
    ThermodynamicConditions,
)
from thermo_engine import calculate_equilibrium, validate_equilibrium_result
from thermo_engine.errors import ThermoEquiError
from thermo_engine.service import resolve_backend
from thermo_engine.unifac_backend import has_unifac_group_assignments

ETHANOL = ComponentIdentity(component_id="ethanol", name="Ethanol", cas_number="64-17-5")
WATER = ComponentIdentity(component_id="water", name="Water", cas_number="7732-18-5")
BENZENE = ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2")
METHANOL = ComponentIdentity(component_id="methanol", name="Methanol", cas_number="67-56-1")
METHANE = ComponentIdentity(component_id="methane", name="Methane", cas_number="74-82-8")
ETHANE = ComponentIdentity(component_id="ethane", name="Ethane", cas_number="74-84-0")


def _bubble_task(
    components: list[ComponentIdentity] | None = None,
    composition: list[float] | None = None,
) -> TaskManifest:
    return TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=components or [ETHANOL, WATER],
        conditions=ThermodynamicConditions(
            pressure_kPa=101.325,
            liquid_composition=composition or [0.5, 0.5],
        ),
        model_name="UNIFAC",
    )


def test_unifac_bubble_point_crosses_public_validation_gate_with_pilot_warning() -> None:
    task = _bubble_task()

    result = calculate_equilibrium(task)
    report = validate_equilibrium_result(result)

    assert result.model_name == "UNIFAC"
    assert result.backend_version.startswith("thermo/")
    assert result.temperature_K == pytest.approx(353.05, abs=0.2)
    assert report.equilibrium_residual.passed
    assert report.convergence.passed
    assert report.overall_status == "warning"
    assert any("predictive pilot" in warning for warning in result.warnings)


def test_unifac_bubble_and_dew_points_are_self_consistent() -> None:
    bubble = calculate_equilibrium(_bubble_task())
    dew_task = _bubble_task().model_copy(
        update={
            "calculation_type": "dew_point",
            "conditions": ThermodynamicConditions(
                pressure_kPa=101.325,
                vapor_composition=bubble.points[0].vapor_composition,
            ),
        }
    )

    dew = calculate_equilibrium(dew_task)

    assert dew.temperature_K == pytest.approx(bubble.temperature_K, abs=1e-4)
    assert dew.points[0].liquid_composition == pytest.approx(
        bubble.points[0].liquid_composition,
        abs=1e-4,
    )


def test_unifac_supports_multicomponent_bubble_point() -> None:
    task = _bubble_task(
        components=[ETHANOL, WATER, METHANOL],
        composition=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    )

    result = calculate_equilibrium(task)
    report = validate_equilibrium_result(result)

    assert len(result.points[0].liquid_composition) == 3
    assert report.composition_balance.passed
    assert report.equilibrium_residual.passed


def test_unifac_isobaric_vle_curve_returns_binary_points() -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="isobaric_vle",
        components=[ETHANOL, BENZENE],
        conditions=ThermodynamicConditions(pressure_kPa=53.33),
        model_name="UNIFAC",
        points=5,
    )

    result = calculate_equilibrium(task)

    assert len(result.points) == 5
    assert all(point.equilibrium_residual == pytest.approx(0.0, abs=1e-8) for point in result.points)
    assert result.warnings


def test_unifac_missing_group_assignment_is_structured_failure() -> None:
    task = _bubble_task(components=[METHANE, ETHANE])

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "missing_parameters"
    assert "group assignment" in captured.value.detail.message
    assert captured.value.detail.details["model"] == "UNIFAC"


def test_unifac_requires_cas_number() -> None:
    task = _bubble_task(
        components=[
            ComponentIdentity(component_id="ethanol", name="Ethanol"),
            WATER,
        ]
    )

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "missing_parameters"
    assert "CAS number" in captured.value.detail.message


def test_unifac_parameter_sources_declare_group_and_interaction_tables() -> None:
    task = _bubble_task()

    sources = resolve_backend(task).parameter_sources(task)

    assert any(source.get("property") == "UNIFAC subgroup assignment" for source in sources)
    assert any(source.get("property") == "Original UNIFAC interaction parameters" for source in sources)
    assert all(source["quality_level"] for source in sources)


def test_unifac_is_executable_without_binary_parameters_and_gets_pilot_penalty() -> None:
    task = _bubble_task(components=[ETHANOL, BENZENE])

    recommendations = recommend_models(task)
    unifac = next(item for item in recommendations if item.model_name == "UNIFAC")

    assert unifac.executable is True
    assert unifac.breakdown.numerical_risk_penalty == 12.0


def test_unifac_lle_is_rejected() -> None:
    task = TaskManifest(
        equilibrium_type="LLE",
        calculation_type="lle",
        components=[ETHANOL, WATER],
        conditions=ThermodynamicConditions(temperature_K=298.15),
        model_name="UNIFAC",
    )

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "unsupported_model"


def test_unifac_group_assignment_helper_covers_organic_components_only() -> None:
    assert has_unifac_group_assignments([ETHANOL, WATER, BENZENE]) is True
    assert has_unifac_group_assignments([METHANE]) is False
