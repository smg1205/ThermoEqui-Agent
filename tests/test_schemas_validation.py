"""Contract and independent validation behavior tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from schemas.domain import CalculationResult, ComponentIdentity, ParameterSet, ThermodynamicConditions
from thermo_engine.validation import validate_result


def parameter_payload() -> dict[str, object]:
    return json.loads((Path(__file__).parent / "fixtures" / "user_supplied_parameter.json").read_text(encoding="utf-8"))


def test_manifest_conditions_reject_unbalanced_composition() -> None:
    try:
        ThermodynamicConditions(feed_composition=[0.2, 0.2])
    except ValueError as error:
        assert "sum to one" in str(error)
    else:
        raise AssertionError("Unbalanced composition was accepted")


def test_nonconverged_result_cannot_be_validated_as_success() -> None:
    result = CalculationResult(
        task_id="task",
        calculation_type="tp_flash",
        input_snapshot={"conditions": {"feed_composition": [0.5, 0.5]}},
        model_name="Ideal/Raoult",
        converged=False,
        residual=1.0,
        iterations=100,
        warnings=[],
        backend_version="test",
    )
    report = validate_result(result)
    assert report.overall_status == "failed"
    assert not report.convergence.passed


def test_component_schema_keeps_canonical_identity() -> None:
    component = ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2")
    assert component.model_dump()["cas_number"] == "71-43-2"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"component_order": ["a", "A"]}, "must be unique"),
        ({"parameters": {"p12": float("nan"), "p21": -0.2}}, "must be finite"),
        ({"units": {"p12": "dimensionless"}}, "exactly one entry"),
        ({"temperature_range_K": [380.0, 280.0]}, "ascending order"),
        (
            {
                "source_type": "literature",
                "source_title": None,
                "source_identifier": None,
            },
            "require source_title",
        ),
    ],
)
def test_parameter_schema_rejects_unsafe_or_untraceable_values(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ParameterSet.model_validate({**parameter_payload(), **updates})


def test_validator_fails_non_positive_result_conditions() -> None:
    result = CalculationResult(
        task_id="task",
        calculation_type="bubble_point",
        input_snapshot={},
        model_name="Ideal/Raoult",
        temperature_K=-1.0,
        pressure_kPa=101.325,
        converged=True,
        residual=0.0,
        iterations=1,
        backend_version="test",
    )
    report = validate_result(result)
    assert report.overall_status == "failed"
    assert not report.parameter_applicability.passed


def test_flash_without_phase_outputs_cannot_pass_material_or_stability_checks() -> None:
    result = CalculationResult(
        task_id="task",
        calculation_type="tp_flash",
        input_snapshot={
            "conditions": {
                "temperature_K": 365.0,
                "pressure_kPa": 101.325,
                "feed_composition": [0.5, 0.5],
            }
        },
        model_name="Ideal/Raoult",
        temperature_K=365.0,
        pressure_kPa=101.325,
        converged=True,
        residual=0.0,
        iterations=1,
        backend_version="test",
    )
    report = validate_result(result)
    assert report.overall_status == "failed"
    assert not report.material_balance.passed
    assert not report.phase_stability.passed
