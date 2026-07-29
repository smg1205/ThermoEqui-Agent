"""Behavior tests at the public thermodynamic Python API seam."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions
from thermo_engine import calculate_equilibrium, validate_equilibrium_result
from thermo_engine.errors import ThermoEquiError
from thermo_engine.parameters import reverse_binary_parameter_direction
from thermo_engine.units import normalize_composition, pressure_to_kpa, temperature_to_kelvin

BENZENE = ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2")
TOLUENE = ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3")


def manifest(calculation_type: str, conditions: ThermodynamicConditions, points: int = 11) -> TaskManifest:
    return TaskManifest(
        equilibrium_type="FLASH" if calculation_type == "tp_flash" else "VLE",
        calculation_type=calculation_type,
        components=[BENZENE, TOLUENE],
        conditions=conditions,
        model_name="Ideal/Raoult",
        points=points,
    )


def test_units_and_composition_are_normalized_explicitly() -> None:
    assert pressure_to_kpa(1.0, "atm") == pytest.approx(101.325)
    assert temperature_to_kelvin(25.0, "C") == pytest.approx(298.15)
    assert normalize_composition([2.0, 3.0]) == pytest.approx([0.4, 0.6])


def test_bubble_point_has_expected_endpoint_and_equilibrium_invariant() -> None:
    result = calculate_equilibrium(
        manifest(
            "bubble_point",
            ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[1.0, 0.0]),
        )
    )
    assert result.temperature_K == pytest.approx(353.25, abs=0.25)
    assert result.points[0].vapor_composition == pytest.approx([1.0, 0.0], abs=1e-9)
    assert result.residual < 1e-7


def test_dew_and_bubble_points_are_self_consistent() -> None:
    bubble = calculate_equilibrium(
        manifest(
            "bubble_point",
            ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.4, 0.6]),
        )
    )
    dew = calculate_equilibrium(
        manifest(
            "dew_point",
            ThermodynamicConditions(
                pressure_kPa=101.325,
                vapor_composition=bubble.points[0].vapor_composition,
            ),
        )
    )
    assert dew.temperature_K == pytest.approx(bubble.temperature_K, abs=1e-6)
    assert dew.points[0].liquid_composition == pytest.approx([0.4, 0.6], abs=1e-7)


def test_binary_isobaric_curve_has_pure_endpoints_and_valid_points() -> None:
    result = calculate_equilibrium(manifest("isobaric_vle", ThermodynamicConditions(pressure_kPa=101.325), points=9))
    report = validate_equilibrium_result(result)
    assert len(result.points) == 9
    assert result.points[0].liquid_composition == pytest.approx([0.0, 1.0])
    assert result.points[-1].liquid_composition == pytest.approx([1.0, 0.0])
    assert report.overall_status in {"passed", "warning"}
    assert report.maximum_equilibrium_residual < 1e-6


def test_tp_flash_closes_material_balance() -> None:
    result = calculate_equilibrium(
        manifest(
            "tp_flash",
            ThermodynamicConditions(
                temperature_K=365.0,
                pressure_kPa=101.325,
                feed_composition=[0.5, 0.5],
            ),
        )
    )
    report = validate_equilibrium_result(result)
    assert result.vapor_fraction is not None
    assert 0.0 <= result.vapor_fraction <= 1.0
    assert report.material_balance.passed
    assert report.convergence.passed


def test_missing_nonideal_parameters_never_produce_a_result() -> None:
    task = manifest(
        "bubble_point",
        ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
    ).model_copy(update={"model_name": "NRTL"})
    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)
    assert captured.value.detail.failure_type == "missing_parameters"


def test_ideal_benzene_toluene_has_no_internal_azeotrope() -> None:
    result = calculate_equilibrium(manifest("azeotrope", ThermodynamicConditions(pressure_kPa=101.325), points=51))
    assert result.points == []
    assert any("No internal azeotrope" in warning for warning in result.warnings)


def test_directional_parameters_reverse_explicitly() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures" / "synthetic_nrtl.json").read_text(encoding="utf-8"))
    reversed_parameters = reverse_binary_parameter_direction(fixture["parameters"], [("tau12", "tau21")])
    assert reversed_parameters == {"tau12": -0.4, "tau21": 1.2, "alpha": 0.3}


def test_repeated_solver_runs_are_numerically_consistent() -> None:
    task = manifest(
        "bubble_point",
        ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.4, 0.6]),
    )
    temperatures = [calculate_equilibrium(task).temperature_K for _ in range(3)]
    assert temperatures == pytest.approx([temperatures[0]] * 3, abs=1e-12)


def test_lle_contract_refuses_inapplicable_ideal_model() -> None:
    task = manifest("lle", ThermodynamicConditions(temperature_K=298.15)).model_copy(update={"equilibrium_type": "LLE"})
    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)
    assert captured.value.detail.failure_type == "unsupported_model"


def test_mass_fraction_is_rejected_instead_of_misread_as_mole_fraction() -> None:
    task = manifest(
        "bubble_point",
        ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
    ).model_copy(update={"composition_basis": "mass_fraction"})
    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)
    assert captured.value.detail.failure_type == "unsupported_model"


def test_ideal_model_is_blocked_outside_configured_pressure_regime() -> None:
    task = manifest(
        "tp_flash",
        ThermodynamicConditions(temperature_K=365.0, pressure_kPa=600.0, feed_composition=[0.5, 0.5]),
    )
    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)
    assert captured.value.detail.failure_type == "parameter_out_of_domain"


def test_peng_robinson_tp_flash_matches_upstream_reference_case() -> None:
    task = TaskManifest(
        equilibrium_type="FLASH",
        calculation_type="tp_flash",
        components=[
            ComponentIdentity(component_id="methane", name="Methane", cas_number="74-82-8"),
            ComponentIdentity(component_id="ethane", name="Ethane", cas_number="74-84-0"),
            ComponentIdentity(component_id="nitrogen", name="Nitrogen", cas_number="7727-37-9"),
        ],
        conditions=ThermodynamicConditions(
            temperature_K=110.0,
            pressure_kPa=100.0,
            feed_composition=[0.965, 0.018, 0.017],
        ),
        model_name="Peng-Robinson",
    )

    result = calculate_equilibrium(task)
    report = validate_equilibrium_result(result)

    assert result.model_name == "Peng-Robinson"
    assert result.backend_version.startswith("thermo/")
    assert result.vapor_fraction == pytest.approx(0.0890325, abs=2e-4)
    assert result.phases[0].composition == pytest.approx([0.974400, 0.019757, 0.005843], abs=2e-4)
    assert result.phases[1].composition == pytest.approx([0.868821, 0.000025766, 0.131154], abs=2e-4)
    assert 1e-10 < result.residual < 1e-6
    assert report.material_balance.passed
    assert report.equilibrium_residual.passed


def test_peng_robinson_requires_reviewed_binary_parameters() -> None:
    task = manifest(
        "tp_flash",
        ThermodynamicConditions(
            temperature_K=365.0,
            pressure_kPa=101.325,
            feed_composition=[0.5, 0.5],
        ),
    ).model_copy(update={"model_name": "Peng-Robinson"})

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "missing_parameters"
    assert captured.value.detail.details["parameter_table"] == "ChemSep PR"
    assert captured.value.detail.details["missing_pairs"] == [["71-43-2", "108-88-3"]]


def test_peng_robinson_rejects_strongly_associating_system() -> None:
    task = TaskManifest(
        equilibrium_type="FLASH",
        calculation_type="tp_flash",
        components=[
            ComponentIdentity(component_id="water", name="Water", cas_number="7732-18-5"),
            ComponentIdentity(component_id="methanol", name="Methanol", cas_number="67-56-1"),
        ],
        conditions=ThermodynamicConditions(
            temperature_K=450.0,
            pressure_kPa=1000.0,
            feed_composition=[0.5, 0.5],
        ),
        model_name="Peng-Robinson",
    )

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "parameter_out_of_domain"
    assert captured.value.detail.details["inapplicable_components"] == ["Water", "Methanol"]


def test_automatic_routing_rejects_high_pressure_associating_system() -> None:
    task = TaskManifest(
        equilibrium_type="FLASH",
        calculation_type="tp_flash",
        components=[
            ComponentIdentity(component_id="water", name="Water", cas_number="7732-18-5"),
            ComponentIdentity(component_id="methanol", name="Methanol", cas_number="67-56-1"),
        ],
        conditions=ThermodynamicConditions(
            temperature_K=450.0,
            pressure_kPa=1000.0,
            feed_composition=[0.5, 0.5],
        ),
        model_name=None,
    )

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "parameter_out_of_domain"


def test_unset_model_routes_high_pressure_flash_to_peng_robinson() -> None:
    task = TaskManifest(
        equilibrium_type="FLASH",
        calculation_type="tp_flash",
        components=[
            ComponentIdentity(component_id="methane", name="Methane", cas_number="74-82-8"),
            ComponentIdentity(component_id="ethane", name="Ethane", cas_number="74-84-0"),
            ComponentIdentity(component_id="nitrogen", name="Nitrogen", cas_number="7727-37-9"),
        ],
        conditions=ThermodynamicConditions(
            temperature_K=150.0,
            pressure_kPa=1000.0,
            feed_composition=[0.965, 0.018, 0.017],
        ),
        model_name=None,
    )

    result = calculate_equilibrium(task)
    report = validate_equilibrium_result(result)

    assert result.model_name == "Peng-Robinson"
    assert result.input_snapshot["model_name"] == "Peng-Robinson"
    assert any("high-pressure" in item for item in result.input_snapshot["assumptions"])
    assert report.overall_status in {"passed", "warning"}


def test_unset_model_routes_unregistered_light_gases_to_peng_robinson_at_low_pressure() -> None:
    task = TaskManifest(
        equilibrium_type="FLASH",
        calculation_type="tp_flash",
        components=[
            ComponentIdentity(component_id="methane", name="Methane", cas_number="74-82-8"),
            ComponentIdentity(component_id="ethane", name="Ethane", cas_number="74-84-0"),
            ComponentIdentity(component_id="nitrogen", name="Nitrogen", cas_number="7727-37-9"),
        ],
        conditions=ThermodynamicConditions(
            temperature_K=110.0,
            pressure_kPa=100.0,
            feed_composition=[0.965, 0.018, 0.017],
        ),
        model_name=None,
    )

    result = calculate_equilibrium(task)

    assert result.model_name == "Peng-Robinson"
    assert any("outside the local Ideal registry" in item for item in result.input_snapshot["assumptions"])


def test_single_phase_peng_robinson_flash_passes_material_balance() -> None:
    task = TaskManifest(
        equilibrium_type="FLASH",
        calculation_type="tp_flash",
        components=[
            ComponentIdentity(component_id="methane", name="Methane", cas_number="74-82-8"),
            ComponentIdentity(component_id="ethane", name="Ethane", cas_number="74-84-0"),
            ComponentIdentity(component_id="nitrogen", name="Nitrogen", cas_number="7727-37-9"),
        ],
        conditions=ThermodynamicConditions(
            temperature_K=110.0,
            pressure_kPa=1000.0,
            feed_composition=[0.965, 0.018, 0.017],
        ),
        model_name="Peng-Robinson",
    )

    result = calculate_equilibrium(task)
    report = validate_equilibrium_result(result)

    assert len(result.phases) == 1
    assert result.phases[0].fraction == pytest.approx(1.0)
    assert result.phases[0].composition == pytest.approx([0.965, 0.018, 0.017])
    assert report.composition_balance.passed
    assert report.material_balance.passed
    assert report.convergence.passed


@pytest.mark.parametrize(
    ("calculation_type", "conditions"),
    [
        (
            "bubble_point",
            ThermodynamicConditions(pressure_kPa=100.0, liquid_composition=[0.5, 0.5]),
        ),
        (
            "dew_point",
            ThermodynamicConditions(pressure_kPa=100.0, vapor_composition=[0.5, 0.5]),
        ),
        ("isobaric_vle", ThermodynamicConditions(pressure_kPa=100.0)),
        ("isothermal_vle", ThermodynamicConditions(temperature_K=150.0)),
        (
            "phase_stability",
            ThermodynamicConditions(
                temperature_K=150.0,
                pressure_kPa=100.0,
                feed_composition=[0.5, 0.5],
            ),
        ),
        ("azeotrope", ThermodynamicConditions(pressure_kPa=100.0)),
    ],
)
def test_peng_robinson_advertised_operations_cross_public_validation_gate(
    calculation_type: str,
    conditions: ThermodynamicConditions,
) -> None:
    task = TaskManifest(
        equilibrium_type="FLASH" if calculation_type == "phase_stability" else "VLE",
        calculation_type=calculation_type,
        components=[
            ComponentIdentity(component_id="methane", name="Methane", cas_number="74-82-8"),
            ComponentIdentity(component_id="ethane", name="Ethane", cas_number="74-84-0"),
        ],
        conditions=conditions,
        model_name="Peng-Robinson",
        points=5,
    )

    result = calculate_equilibrium(task)
    report = validate_equilibrium_result(result)

    assert result.backend_version.startswith("thermo/")
    assert result.converged
    assert report.overall_status in {"passed", "warning"}


def test_structured_polymer_task_is_rejected_by_shared_service_guard() -> None:
    task = manifest(
        "bubble_point",
        ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
    ).model_copy(
        update={
            "components": [
                ComponentIdentity(component_id="polymer-a", name="Polymer A"),
                ComponentIdentity(component_id="solvent-b", name="Solvent B"),
            ]
        }
    )
    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)
    assert captured.value.detail.failure_type == "unsupported_model"


def test_original_question_cannot_be_hidden_by_an_incorrect_manifest() -> None:
    task = manifest(
        "bubble_point",
        ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
    ).model_copy(update={"original_question": "Calculate polymer solution VLE"})

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "unsupported_model"


def test_flowsheet_design_in_original_question_is_rejected() -> None:
    task = manifest(
        "bubble_point",
        ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
    ).model_copy(update={"original_question": "Calculate VLE and design a flowsheet"})

    with pytest.raises(ThermoEquiError) as captured:
        calculate_equilibrium(task)

    assert captured.value.detail.failure_type == "unsupported_model"
