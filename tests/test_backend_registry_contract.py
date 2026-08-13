"""Registry-wide contracts for capability declarations, sources, and validation gates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from agent.router import load_model_cards
from schemas.domain import ComponentIdentity, ParameterSet, TaskManifest, ThermodynamicConditions
from thermo_engine import calculate_equilibrium, validate_equilibrium_result
from thermo_engine.errors import ThermoEquiError
from thermo_engine.model_catalog import load_model_catalog
from thermo_engine.parameter_store import load_production_parameter_sets
from thermo_engine.registry import DEFAULT_BACKEND_REGISTRY
from thermo_engine.service import resolve_backend

BENZENE = ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2")
TOLUENE = ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3")
METHANE = ComponentIdentity(component_id="methane", name="Methane", cas_number="74-82-8")
ETHANE = ComponentIdentity(component_id="ethane", name="Ethane", cas_number="74-84-0")
ETHANOL = ComponentIdentity(component_id="ethanol", name="ethanol", cas_number="64-17-5")
WATER = ComponentIdentity(component_id="water", name="water", cas_number="7732-18-5")


def _activity_task(model_name: str, parameters: list[ParameterSet] | None = None) -> TaskManifest:
    return TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[ETHANOL, WATER],
        conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
        model_name=model_name,
        parameters=parameters or [],
    )


def _flash_task(model_name: str, parameters: list[ParameterSet] | None = None) -> TaskManifest:
    return TaskManifest(
        equilibrium_type="FLASH",
        calculation_type="tp_flash",
        components=[METHANE, ETHANE],
        conditions=ThermodynamicConditions(
            temperature_K=150.0,
            pressure_kPa=530.0,
            feed_composition=[0.8, 0.2],
        ),
        model_name=model_name,
        parameters=parameters or [],
    )


def canonical_task(model_name: str, parameters: list[ParameterSet] | None = None) -> TaskManifest:
    if model_name in {"NRTL", "UNIQUAC", "Wilson"}:
        return _activity_task(model_name, parameters)
    if model_name in {"Peng-Robinson", "Phasepy/Peng-Robinson", "Clapeyron/Peng-Robinson"}:
        return _flash_task(model_name, parameters)
    if model_name == "SRK":
        return _flash_task(model_name, parameters)
    return TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[BENZENE, TOLUENE],
        conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
        model_name=model_name,
        parameters=parameters or [],
    )


def production_activity_parameter_set(model_name: str) -> ParameterSet:
    return next(
        item
        for item in load_production_parameter_sets()
        if item.model_name == model_name and item.component_order == ["ethanol", "water"]
    )


def srk_fixture_parameter_set() -> ParameterSet:
    path = Path(__file__).parent / "fixtures" / "srk_kij_methane_ethane.json"
    return ParameterSet.model_validate(json.loads(path.read_text(encoding="utf-8")))


def test_registry_capability_declarations_match_catalog_and_cards() -> None:
    catalog = load_model_catalog()
    cards = {card.model_name: card for card in load_model_cards()}

    for registration in DEFAULT_BACKEND_REGISTRY.registrations:
        entry = catalog.get(registration.canonical_name)
        card = cards.get(registration.canonical_name)
        assert entry is not None, registration.canonical_name
        assert card is not None, registration.canonical_name
        assert set(registration.supported_calculations) == set(entry.supported_calculation_types)
        assert set(entry.supported_equilibrium_types) == set(card.supported_tasks)
        assert registration.canonical_name == entry.name == card.model_name


def test_every_registered_backend_implements_the_full_protocol_seam() -> None:
    protocol_methods = (
        "parameter_sources",
        "bubble_point",
        "dew_point",
        "isobaric_vle",
        "isothermal_vle",
        "tp_flash",
        "phase_stability",
        "azeotrope",
        "lle",
    )
    for registration in DEFAULT_BACKEND_REGISTRY.registrations:
        backend = registration.factory()
        for method in protocol_methods:
            assert callable(getattr(backend, method, None)), (registration.canonical_name, method)


def test_parameter_sources_are_structured_or_raise_structured_failures() -> None:
    for registration in DEFAULT_BACKEND_REGISTRY.registrations:
        task = canonical_task(registration.canonical_name)
        try:
            backend = resolve_backend(task)
            sources = backend.parameter_sources(task)
        except ThermoEquiError as error:
            assert error.detail.failure_type in {"missing_parameters", "unsupported_model"}
            continue
        assert isinstance(sources, list)
        assert sources
        assert all(isinstance(source, dict) for source in sources)
        assert all("source_title" in source or "component" in source for source in sources)


@pytest.mark.parametrize(
    ("model_name", "parameters"),
    [
        ("Ideal/Raoult", None),
        ("Peng-Robinson", None),
        ("NRTL", "production"),
        ("UNIQUAC", "production"),
        ("Wilson", "production"),
        ("SRK", "fixture"),
    ],
)
def test_available_backends_cross_the_public_validation_gate(model_name: str, parameters: str | None) -> None:
    parameter_sets: list[ParameterSet] = []
    if parameters == "production":
        parameter_sets = [production_activity_parameter_set(model_name)]
    elif parameters == "fixture":
        parameter_sets = [srk_fixture_parameter_set()]

    result = calculate_equilibrium(canonical_task(model_name, parameter_sets))
    report = validate_equilibrium_result(result)

    assert result.model_name == model_name
    assert report.overall_status in {"passed", "warning"}
    assert report.equilibrium_residual.passed
    assert report.convergence.passed


def test_required_parameter_models_fail_structurally_without_parameters() -> None:
    catalog = load_model_catalog()
    for registration in DEFAULT_BACKEND_REGISTRY.registrations:
        entry = catalog[registration.canonical_name]
        if not entry.requires_binary_parameters:
            continue
        if registration.canonical_name in {"NRTL", "UNIQUAC", "Wilson"}:
            task = _activity_task(registration.canonical_name)
        elif registration.canonical_name == "SRK":
            task = _flash_task(registration.canonical_name)
        else:
            task = TaskManifest(
                equilibrium_type="FLASH",
                calculation_type="tp_flash",
                components=[BENZENE, TOLUENE],
                conditions=ThermodynamicConditions(
                    temperature_K=365.0,
                    pressure_kPa=101.325,
                    feed_composition=[0.5, 0.5],
                ),
                model_name=registration.canonical_name,
            )

        with pytest.raises(ThermoEquiError) as captured:
            calculate_equilibrium(task)

        assert captured.value.detail.failure_type in {
            "missing_parameters",
            "unsupported_model",
            "parameter_out_of_domain",
        }


@pytest.mark.skipif(
    importlib.util.find_spec("phasepy") is None,
    reason="Phasepy optional dependency is absent",
)
def test_phasepy_crosses_validation_gate_when_installed() -> None:
    result = calculate_equilibrium(canonical_task("Phasepy/Peng-Robinson"))
    report = validate_equilibrium_result(result)
    assert report.overall_status in {"passed", "warning"}
    assert report.equilibrium_residual.passed


@pytest.mark.skipif(
    importlib.util.find_spec("pyclapeyron") is None,
    reason="Clapeyron optional dependency is absent",
)
def test_clapeyron_crosses_validation_gate_when_installed() -> None:
    result = calculate_equilibrium(canonical_task("Clapeyron/Peng-Robinson"))
    report = validate_equilibrium_result(result)
    assert report.overall_status in {"passed", "warning"}
    assert report.equilibrium_residual.passed
