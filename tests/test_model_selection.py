"""Behavioral tests for score-based model selection and execution fallback."""

from __future__ import annotations

import pytest

from agent.executor import execute_task
from agent.router import rank_executable_models
from schemas.domain import ComponentIdentity, FailureType, TaskManifest, ThermodynamicConditions
from thermo_engine.errors import ThermoEquiError

BENZENE = ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2")
TOLUENE = ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3")
ETHANOL = ComponentIdentity(component_id="ethanol", name="ethanol", cas_number="64-17-5")
WATER = ComponentIdentity(component_id="water", name="water", cas_number="7732-18-5")
METHANE = ComponentIdentity(component_id="methane", name="Methane", cas_number="74-82-8")
ETHANE = ComponentIdentity(component_id="ethane", name="Ethane", cas_number="74-84-0")
NITROGEN = ComponentIdentity(component_id="nitrogen", name="Nitrogen", cas_number="7727-37-9")


def test_auto_selection_prefers_ideal_for_local_registry_system() -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="isobaric_vle",
        components=[BENZENE, TOLUENE],
        conditions=ThermodynamicConditions(pressure_kPa=101.325),
        points=5,
    )

    envelope = execute_task(task)

    assert envelope.result.model_name == "Ideal/Raoult"
    assert envelope.model_recommendations[0].model_name == "Ideal/Raoult"


def test_auto_selection_prefers_peng_robinson_for_high_pressure_light_gas() -> None:
    task = TaskManifest(
        equilibrium_type="FLASH",
        calculation_type="tp_flash",
        components=[METHANE, ETHANE, NITROGEN],
        conditions=ThermodynamicConditions(
            temperature_K=150.0,
            pressure_kPa=1000.0,
            feed_composition=[0.965, 0.018, 0.017],
        ),
    )

    envelope = execute_task(task)

    assert envelope.result.model_name == "Peng-Robinson"
    assert any("Auto-selected Peng-Robinson" in item for item in envelope.result.input_snapshot["assumptions"])


def test_auto_selection_uses_activity_coefficient_when_ideal_and_eos_unavailable() -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[ETHANOL, WATER],
        conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
    )

    envelope = execute_task(task)

    assert envelope.result.model_name == "NRTL"
    assert envelope.validation.overall_status in {"passed", "warning"}


def test_rank_executable_models_returns_only_executable_candidates() -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[ETHANOL, WATER],
        conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
    )

    candidates = rank_executable_models(task)

    assert candidates
    assert all(item.executable for item in candidates)
    assert candidates[0].model_name == "NRTL"


def test_auto_selection_raises_structured_error_when_nothing_is_executable() -> None:
    task = TaskManifest(
        equilibrium_type="LLE",
        calculation_type="lle",
        components=[BENZENE, TOLUENE],
        conditions=ThermodynamicConditions(temperature_K=298.15),
    )

    with pytest.raises(ThermoEquiError) as captured:
        execute_task(task)

    assert captured.value.detail.failure_type == FailureType.PARAMETER_OUT_OF_DOMAIN
    assert "No executable model" in captured.value.detail.message
