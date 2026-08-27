"""Tests for validation warning behavior when results are numerically valid but flagged."""

from __future__ import annotations

from schemas.domain import CalculationResult, EquilibriumPoint, PhaseResult
from thermo_engine.validation import validate_result


def test_validate_result_returns_warning_when_warnings_exist() -> None:
    result = CalculationResult(
        task_id="task",
        calculation_type="bubble_point",
        input_snapshot={"conditions": {"pressure_kPa": 101.325, "liquid_composition": [1.0, 0.0]}},
        model_name="Ideal/Raoult",
        points=[
            EquilibriumPoint(
                temperature_K=353.25,
                pressure_kPa=101.325,
                liquid_composition=[1.0, 0.0],
                vapor_composition=[1.0, 0.0],
                equilibrium_residual=1e-9,
            )
        ],
        phases=[PhaseResult(phase="liquid", fraction=1.0, composition=[1.0, 0.0])],
        temperature_K=353.25,
        pressure_kPa=101.325,
        vapor_fraction=0.0,
        phase_state="liquid",
        converged=True,
        residual=1e-9,
        iterations=1,
        warnings=["Correlation extrapolated outside reviewed range."],
        backend_version="internal-ideal-raoult/0.1.0",
    )

    report = validate_result(result)

    assert report.overall_status == "warning"
    assert report.parameter_applicability.passed is False
    assert report.equilibrium_residual.passed
    assert report.material_balance.passed
    assert report.convergence.passed
    assert report.warnings
