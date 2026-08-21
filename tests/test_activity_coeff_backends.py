"""Numerical sanity and validation-gate coverage for activity-coefficient backends."""

from __future__ import annotations

import math

import pytest

from schemas.domain import ComponentIdentity, ParameterSet, TaskManifest, ThermodynamicConditions
from thermo_engine import calculate_equilibrium, validate_equilibrium_result
from thermo_engine.parameter_store import load_production_parameter_sets
from thermo_engine.service import resolve_backend

ETHANOL = ComponentIdentity(component_id="ethanol", name="ethanol", cas_number="64-17-5")
WATER = ComponentIdentity(component_id="water", name="water", cas_number="7732-18-5")


def production_parameter_set(model_name: str) -> ParameterSet:
    return next(
        item
        for item in load_production_parameter_sets()
        if item.model_name == model_name and item.component_order == ["ethanol", "water"]
    )


@pytest.mark.parametrize("model_name", ["NRTL", "UNIQUAC", "Wilson"])
def test_activity_coefficient_bubble_points_cross_validation_gate(model_name: str) -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[ETHANOL, WATER],
        conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
        model_name=model_name,
        parameters=[production_parameter_set(model_name)],
    )

    result = calculate_equilibrium(task)
    report = validate_equilibrium_result(result)

    assert result.model_name == model_name
    assert 300.0 < result.temperature_K < 400.0
    assert math.isfinite(result.residual)
    assert report.overall_status in {"passed", "warning"}
    assert report.equilibrium_residual.passed
    assert report.convergence.passed
    assert all(
        0.0 <= value <= 1.0 for value in result.points[0].liquid_composition + result.points[0].vapor_composition
    )


@pytest.mark.parametrize("model_name", ["NRTL", "UNIQUAC", "Wilson"])
def test_activity_coefficient_parameter_sources_are_reported(model_name: str) -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[ETHANOL, WATER],
        conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
        model_name=model_name,
        parameters=[production_parameter_set(model_name)],
    )

    sources = resolve_backend(task).parameter_sources(task)

    assert sources
    assert sources[0]["source_title"]
    assert sources[0]["component"]


@pytest.mark.parametrize("model_name", ["NRTL", "UNIQUAC", "Wilson"])
def test_bubble_dew_points_are_self_consistent(model_name: str) -> None:
    bubble = calculate_equilibrium(
        TaskManifest(
            equilibrium_type="VLE",
            calculation_type="bubble_point",
            components=[ETHANOL, WATER],
            conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
            model_name=model_name,
            parameters=[production_parameter_set(model_name)],
        )
    )
    dew = calculate_equilibrium(
        TaskManifest(
            equilibrium_type="VLE",
            calculation_type="dew_point",
            components=[ETHANOL, WATER],
            conditions=ThermodynamicConditions(
                pressure_kPa=101.325,
                vapor_composition=bubble.points[0].vapor_composition,
            ),
            model_name=model_name,
            parameters=[production_parameter_set(model_name)],
        )
    )

    assert dew.temperature_K == pytest.approx(bubble.temperature_K, abs=1e-6)
    assert dew.points[0].liquid_composition == pytest.approx(
        bubble.points[0].liquid_composition,
        abs=1e-7,
    )
