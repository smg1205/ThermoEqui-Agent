"""Tests for model card loading and deterministic recommendation logic."""

from __future__ import annotations

from agent.router import load_model_cards, recommend_models
from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions


def test_load_model_cards_contains_expected_models() -> None:
    cards = load_model_cards()
    assert {card.model_name for card in cards} == {
        "Ideal/Raoult",
        "Peng-Robinson",
        "Phasepy/Peng-Robinson",
        "Clapeyron/Peng-Robinson",
        "SRK",
        "RK",
        "UNIFAC",
        "Wilson",
        "NRTL",
        "UNIQUAC",
        "PGSSI",
        "GHGEAT",
    }


def test_recommend_models_marks_peng_robinson_executable_when_parameters_available() -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="isobaric_vle",
        components=[
            ComponentIdentity(component_id="methane", name="Methane", cas_number="74-82-8"),
            ComponentIdentity(component_id="ethane", name="Ethane", cas_number="74-84-0"),
        ],
        conditions=ThermodynamicConditions(pressure_kPa=800.0),
        points=11,
    )
    recommendations = recommend_models(task, available_parameter_models={"Peng-Robinson"})
    pr = next(item for item in recommendations if item.model_name == "Peng-Robinson")
    assert pr.executable
    nrtl = next(item for item in recommendations if item.model_name == "NRTL")
    assert not nrtl.executable


def test_recommend_models_excludes_wilson_for_lle() -> None:
    task = TaskManifest(
        equilibrium_type="LLE",
        calculation_type="lle",
        components=[
            ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2"),
            ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3"),
        ],
        conditions=ThermodynamicConditions(temperature_K=298.15),
        points=21,
    )
    recommendations = recommend_models(task)
    wilson = next(item for item in recommendations if item.model_name == "Wilson")
    assert any("hard-excluded for LLE" in exclusion for exclusion in wilson.exclusions)


def test_recommend_models_applies_model_applicability_hard_filters_for_lle() -> None:
    task = TaskManifest(
        equilibrium_type="LLE",
        calculation_type="lle",
        components=[
            ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2"),
            ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3"),
        ],
        conditions=ThermodynamicConditions(temperature_K=298.15),
        points=21,
    )

    recommendations = recommend_models(task, available_parameter_models={"NRTL"})

    nrtl = next(item for item in recommendations if item.model_name == "NRTL")
    wilson = next(item for item in recommendations if item.model_name == "Wilson")
    peng_robinson = next(item for item in recommendations if item.model_name == "Peng-Robinson")

    assert nrtl.executable is False
    assert any("does not provide executable LLE support" in exclusion for exclusion in nrtl.exclusions)
    assert wilson.executable is False
    assert any("Wilson explicitly rejects LLE" in exclusion for exclusion in wilson.exclusions)
    assert peng_robinson.executable is False


def test_recommend_models_keeps_nrtl_executable_for_vle_with_parameters() -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[
            ComponentIdentity(component_id="ethanol", name="ethanol", cas_number="64-17-5"),
            ComponentIdentity(component_id="water", name="water", cas_number="7732-18-5"),
        ],
        conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
        points=11,
    )

    recommendations = recommend_models(task, available_parameter_models={"NRTL"})

    nrtl = next(item for item in recommendations if item.model_name == "NRTL")
    assert nrtl.executable is True
    assert not any("Rejected:" in exclusion for exclusion in nrtl.exclusions)


def test_recommend_models_reports_when_no_candidates_are_executable() -> None:
    task = TaskManifest(
        equilibrium_type="LLE",
        calculation_type="lle",
        components=[
            ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2"),
            ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3"),
        ],
        conditions=ThermodynamicConditions(temperature_K=298.15),
        points=21,
    )

    recommendations = recommend_models(task)

    assert recommendations
    assert all(item.executable is False for item in recommendations)
    assert all(item.exclusions for item in recommendations)


def test_non_production_model_gets_numerical_risk_penalty() -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[
            ComponentIdentity(component_id="ethanol", name="ethanol", cas_number="64-17-5"),
            ComponentIdentity(component_id="water", name="water", cas_number="7732-18-5"),
        ],
        conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
        points=11,
    )
    recommendations = recommend_models(task, available_parameter_models={"SRK"})
    srk = next(item for item in recommendations if item.model_name == "SRK")
    assert srk.breakdown.numerical_risk_penalty == 12.0
