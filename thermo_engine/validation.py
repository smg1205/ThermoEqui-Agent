"""Independent result validation gate."""

from __future__ import annotations

import numpy as np

from schemas.domain import CalculationResult, CheckResult, ValidationReport

COMPOSITION_TOLERANCE = 1e-8
EQUILIBRIUM_TOLERANCE = 2e-6
MATERIAL_TOLERANCE = 1e-7


def _composition_error(result: CalculationResult) -> float:
    compositions = [
        *(point.liquid_composition for point in result.points),
        *(point.vapor_composition for point in result.points),
        *(phase.composition for phase in result.phases),
    ]
    composition_error = (
        0.0
        if not compositions
        else max(
            max(abs(sum(values) - 1.0), max((max(0.0, -x, x - 1.0) for x in values), default=0.0))
            for values in compositions
        )
    )
    if not result.phases:
        return composition_error
    phase_fraction_error = max(
        abs(sum(phase.fraction for phase in result.phases) - 1.0),
        max(max(0.0, -phase.fraction, phase.fraction - 1.0) for phase in result.phases),
    )
    return max(composition_error, phase_fraction_error)


def _material_error(result: CalculationResult) -> float:
    is_flash = result.calculation_type in {"tp_flash", "phase_stability"}
    if not result.phases or result.vapor_fraction is None:
        return 1.0 if is_flash else 0.0
    if not is_flash:
        return 0.0
    snapshot = result.input_snapshot
    conditions = snapshot.get("conditions", {})
    feed = conditions.get("feed_composition")
    if feed is None:
        return 1.0
    reconstructed = np.zeros(len(feed))
    for phase in result.phases:
        reconstructed += phase.fraction * np.asarray(phase.composition)
    return float(np.max(np.abs(reconstructed - np.asarray(feed))))


def validate_result(result: CalculationResult) -> ValidationReport:
    composition_error = _composition_error(result)
    material_error = _material_error(result)
    residuals = [point.equilibrium_residual for point in result.points] or [result.residual]
    maximum_residual = max(residuals)
    mean_residual = float(np.mean(residuals))
    extrapolated = any("extrapolated" in warning.lower() for warning in result.warnings)
    temperatures = [point.temperature_K for point in result.points]
    pressures = [point.pressure_kPa for point in result.points]
    if result.temperature_K is not None:
        temperatures.append(result.temperature_K)
    if result.pressure_kPa is not None:
        pressures.append(result.pressure_kPa)
    if result.calculation_type == "infinite_dilution_activity":
        temperatures.extend(point.temperature_K for point in result.gamma_infinity)
    conditions_valid = all(value > 0 for value in [*temperatures, *pressures])
    has_conditions = bool(temperatures or pressures)
    point_based = result.calculation_type in {
        "bubble_point",
        "dew_point",
        "isobaric_vle",
        "isothermal_vle",
    }
    composition_complete = not point_based or bool(result.points)
    is_flash = result.calculation_type in {"tp_flash", "phase_stability"}
    if is_flash:
        composition_complete = 1 <= len(result.phases) <= 2
    is_gamma_infinity = result.calculation_type == "infinite_dilution_activity"
    if is_gamma_infinity:
        composition_complete = bool(result.gamma_infinity)
    composition = CheckResult(
        passed=composition_complete and composition_error <= COMPOSITION_TOLERANCE,
        metric=composition_error,
        tolerance=COMPOSITION_TOLERANCE,
        message="All required phase compositions are present, bounded, and normalized."
        if composition_complete and composition_error <= COMPOSITION_TOLERANCE
        else "A phase composition is unbounded or not normalized.",
    )
    material = CheckResult(
        passed=material_error <= MATERIAL_TOLERANCE,
        metric=material_error,
        tolerance=MATERIAL_TOLERANCE,
        message="Material balance passed."
        if material_error <= MATERIAL_TOLERANCE
        else "Material balance residual exceeds tolerance.",
    )
    if is_gamma_infinity:
        gamma_values = [point.gamma_infinity for point in result.gamma_infinity]
        gamma_finite = all(np.isfinite(value) and value > 0 for value in gamma_values)
        equilibrium = CheckResult(
            passed=gamma_finite,
            metric=float(max(gamma_values)) if gamma_values else 1.0,
            tolerance=None,
            message="All predicted gamma-infinity values are positive and finite."
            if gamma_finite
            else "A predicted gamma-infinity value is non-positive or non-finite.",
        )
        convergence = CheckResult(
            passed=result.converged and result.failure is None,
            metric=1.0 if result.converged else 0.0,
            tolerance=1.0,
            message="PGSSI prediction completed." if result.converged else "PGSSI prediction did not complete.",
        )
        applicability = CheckResult(
            passed=conditions_valid and has_conditions,
            metric=0.0 if conditions_valid and has_conditions else 1.0,
            tolerance=0.0,
            message=(
                "PGSSI predicted gamma-infinity at a positive temperature."
                if conditions_valid and has_conditions
                else "Temperature is non-positive or missing."
            ),
        )
    else:
        equilibrium = CheckResult(
            passed=maximum_residual <= EQUILIBRIUM_TOLERANCE,
            metric=maximum_residual,
            tolerance=EQUILIBRIUM_TOLERANCE,
            message="Equilibrium residual passed."
            if maximum_residual <= EQUILIBRIUM_TOLERANCE
            else "Equilibrium residual exceeds tolerance.",
        )
        convergence = CheckResult(
            passed=result.converged and result.failure is None,
            metric=1.0 if result.converged else 0.0,
            tolerance=1.0,
            message="Solver converged." if result.converged else "Solver did not converge.",
        )
        applicability = CheckResult(
            passed=not extrapolated and conditions_valid and has_conditions,
            metric=0.0 if not extrapolated and conditions_valid and has_conditions else 1.0,
            tolerance=0.0,
            message=(
                "Pure-property correlations are within their stated ranges and T/P are positive."
                if not extrapolated and conditions_valid and has_conditions
                else "Temperature/pressure is non-positive."
                if not conditions_valid or not has_conditions
                else "At least one pure-property correlation was extrapolated."
            ),
        )
    required_passed = all(check.passed for check in (composition, material, equilibrium, convergence))
    stability_required = is_flash
    stability_warning = "Full tangent-plane stability analysis was not performed."
    if not required_passed or not conditions_valid or not has_conditions:
        status = "failed"
        action = "Do not use this result; correct inputs/model or numerical convergence and rerun."
    elif extrapolated or result.warnings or stability_required:
        status = "warning"
        action = "Review warnings and applicability before engineering use."
    else:
        status = "passed"
        action = None
    return ValidationReport(
        overall_status=status,
        composition_balance=composition,
        material_balance=material,
        equilibrium_residual=equilibrium,
        convergence=convergence,
        parameter_applicability=applicability,
        phase_stability=(
            CheckResult(
                passed=False,
                message="Basic phase-state classification completed; tangent-plane analysis was not performed.",
            )
            if stability_required
            else None
        ),
        warnings=[*result.warnings, *([stability_warning] if stability_required else [])],
        recommended_action=action,
        maximum_equilibrium_residual=maximum_residual,
        mean_equilibrium_residual=mean_residual,
        solver_converged=result.converged,
    )
