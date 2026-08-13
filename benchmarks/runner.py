"""Run benchmark cases against any registered thermodynamic backend."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from benchmarks.loader import load_benchmark_cases
from schemas.benchmark import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkMetric,
    BenchmarkReference,
    BenchmarkReferencePoint,
    BenchmarkSuiteReport,
    BenchmarkTolerances,
)
from schemas.domain import CalculationResult
from thermo_engine import calculate_equilibrium, validate_equilibrium_result
from thermo_engine.errors import ThermoEquiError


def _models_for_case(case: BenchmarkCase) -> list[str]:
    models: list[str] = []
    if case.task.model_name is not None:
        models.append(case.task.model_name)
    models.extend(case.applicable_models)
    return list(dict.fromkeys(models))


def _bounded_composition(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    return [min(1.0, max(0.0, value)) for value in values]


def _extract_result_point(result: CalculationResult, index: int) -> BenchmarkReferencePoint:
    if result.points:
        point = result.points[index]
        return BenchmarkReferencePoint(
            temperature_K=point.temperature_K,
            pressure_kPa=point.pressure_kPa,
            liquid_composition=_bounded_composition(point.liquid_composition),
            vapor_composition=_bounded_composition(point.vapor_composition),
        )
    if result.calculation_type in {"tp_flash", "phase_stability"}:
        liquid = next((phase for phase in result.phases if phase.phase == "liquid"), None)
        vapor = next((phase for phase in result.phases if phase.phase == "vapor"), None)
        return BenchmarkReferencePoint(
            temperature_K=result.temperature_K,
            pressure_kPa=result.pressure_kPa,
            vapor_fraction=result.vapor_fraction,
            liquid_composition=_bounded_composition(liquid.composition if liquid is not None else None),
            vapor_composition=_bounded_composition(vapor.composition if vapor is not None else None),
        )
    raise ValueError(f"Cannot extract a benchmark point for {result.calculation_type!r}.")


def _scalar_metric(
    field: str,
    observed: float | None,
    expected: float | None,
    tolerance: float,
) -> BenchmarkMetric | None:
    if observed is None or expected is None:
        return None
    error = abs(observed - expected)
    return BenchmarkMetric(
        field=field,
        observed=observed,
        expected=expected,
        error=error,
        tolerance=tolerance,
        passed=isfinite(error) and error <= tolerance,
    )


def _composition_metrics(
    field: str,
    observed: list[float] | None,
    expected: list[float] | None,
    tolerance: float,
) -> list[BenchmarkMetric]:
    if observed is None or expected is None:
        return []
    if len(observed) != len(expected):
        return [
            BenchmarkMetric(
                field=field,
                observed=float(len(observed)),
                expected=float(len(expected)),
                error=1e9,
                tolerance=tolerance,
                passed=False,
            )
        ]
    metrics: list[BenchmarkMetric] = []
    for index, (observed_value, expected_value) in enumerate(zip(observed, expected, strict=True)):
        error = abs(observed_value - expected_value)
        metrics.append(
            BenchmarkMetric(
                field=f"{field}[{index}]",
                observed=observed_value,
                expected=expected_value,
                error=error,
                tolerance=tolerance,
                passed=isfinite(error) and error <= tolerance,
            )
        )
    return metrics


def _is_dataset_case(case: BenchmarkCase) -> bool:
    """A multi-point isobaric/isothermal VLE dataset runs one point calculation per reference row."""
    return case.task.calculation_type in {"bubble_point", "dew_point"} and len(case.reference.points) > 1


def _compare_observed_points(
    observed_points: list[BenchmarkReferencePoint],
    reference: BenchmarkReference,
    tolerances: BenchmarkTolerances,
) -> list[BenchmarkMetric]:
    if len(observed_points) != len(reference.points):
        return [
            BenchmarkMetric(
                field="points",
                observed=float(len(observed_points)),
                expected=float(len(reference.points)),
                error=float(abs(len(observed_points) - len(reference.points))),
                tolerance=1.0,
                passed=False,
            )
        ]

    metrics: list[BenchmarkMetric] = []
    for index, (observed, expected) in enumerate(zip(observed_points, reference.points, strict=True)):
        prefix = f"point[{index}]"
        for field, observed_value, expected_value, tolerance in (
            (f"{prefix}.temperature_K", observed.temperature_K, expected.temperature_K, tolerances.temperature_K),
            (f"{prefix}.pressure_kPa", observed.pressure_kPa, expected.pressure_kPa, tolerances.pressure_kPa),
            (f"{prefix}.vapor_fraction", observed.vapor_fraction, expected.vapor_fraction, tolerances.vapor_fraction),
        ):
            metric = _scalar_metric(field, observed_value, expected_value, tolerance)
            if metric is not None:
                metrics.append(metric)
        metrics.extend(
            _composition_metrics(
                f"{prefix}.liquid_composition",
                observed.liquid_composition,
                expected.liquid_composition,
                tolerances.composition,
            )
        )
        metrics.extend(
            _composition_metrics(
                f"{prefix}.vapor_composition",
                observed.vapor_composition,
                expected.vapor_composition,
                tolerances.composition,
            )
        )
    return metrics


def _compare_result(
    result: CalculationResult,
    reference: BenchmarkReference,
    tolerances: BenchmarkTolerances,
) -> list[BenchmarkMetric]:
    if result.points:
        observed_points = [_extract_result_point(result, index) for index in range(len(result.points))]
    elif result.calculation_type in {"tp_flash", "phase_stability"}:
        observed_points = [_extract_result_point(result, 0)]
    else:
        observed_points = []

    metrics = _compare_observed_points(observed_points, reference, tolerances)

    if isfinite(result.residual):
        metrics.append(
            BenchmarkMetric(
                field="residual",
                observed=result.residual,
                expected=0.0,
                error=result.residual,
                tolerance=tolerances.residual,
                passed=result.residual <= tolerances.residual,
            )
        )
    return metrics


def _worst_status(statuses: list[str]) -> str:
    if "failed" in statuses:
        return "failed"
    if "warning" in statuses:
        return "warning"
    return "passed"


def _run_dataset_case(case: BenchmarkCase, model_name: str) -> tuple[list[BenchmarkReferencePoint], str, list[str]]:
    """Run one point calculation per experimental reference row."""
    observed_points: list[BenchmarkReferencePoint] = []
    statuses: list[str] = []
    messages: list[str] = []
    for index, reference_point in enumerate(case.reference.points):
        if case.task.calculation_type == "bubble_point":
            liquid_composition = reference_point.liquid_composition
            if liquid_composition is None:
                raise ValueError(f"Benchmark {case.case_id!r} bubble point {index} requires liquid_composition.")
            conditions = case.task.conditions.model_copy(update={"liquid_composition": liquid_composition})
        else:
            vapor_composition = reference_point.vapor_composition
            if vapor_composition is None:
                raise ValueError(f"Benchmark {case.case_id!r} dew point {index} requires vapor_composition.")
            conditions = case.task.conditions.model_copy(update={"vapor_composition": vapor_composition})
        task = case.task.model_copy(
            update={
                "model_name": model_name,
                "conditions": conditions,
                "original_question": f"Benchmark {case.case_id} point {index}",
            }
        )
        result = calculate_equilibrium(task)
        validation = validate_equilibrium_result(result)
        statuses.append(validation.overall_status)
        observed_points.append(_extract_result_point(result, 0))
        if validation.overall_status == "failed":
            messages.append(
                f"point[{index}] validation failed: {validation.recommended_action or validation.overall_status}"
            )
    return observed_points, _worst_status(statuses), messages


def run_benchmark_case(case: BenchmarkCase, model_name: str) -> BenchmarkCaseResult:
    task = case.task.model_copy(
        update={
            "model_name": model_name,
            "original_question": f"Benchmark {case.case_id}",
        }
    )
    try:
        if _is_dataset_case(case):
            observed_points, validation_status, dataset_messages = _run_dataset_case(case, model_name)
            metrics = _compare_observed_points(observed_points, case.reference, case.tolerances)
        else:
            result = calculate_equilibrium(task)
            validation = validate_equilibrium_result(result)
            validation_status = validation.overall_status
            metrics = _compare_result(result, case.reference, case.tolerances)
            dataset_messages = []
    except ThermoEquiError as error:
        return BenchmarkCaseResult(
            case_id=case.case_id,
            model_name=model_name,
            passed=False,
            metrics=[],
            validation_status="failed",
            message=error.detail.message,
        )

    failed_metrics = [metric.field for metric in metrics if not metric.passed]
    passed = validation_status != "failed" and not failed_metrics
    if passed:
        message = "All compared benchmark metrics and validation passed."
    elif failed_metrics:
        message = f"Failed metrics: {', '.join(failed_metrics)}."
    elif dataset_messages:
        message = "; ".join(dataset_messages)
    else:
        message = f"Validation status: {validation_status}."
    return BenchmarkCaseResult(
        case_id=case.case_id,
        model_name=model_name,
        passed=passed,
        metrics=metrics,
        validation_status=validation_status,
        message=message,
    )


def run_all_benchmarks(cases: Sequence[BenchmarkCase] | None = None) -> BenchmarkSuiteReport:
    selected = list(cases) if cases is not None else load_benchmark_cases()
    results: list[BenchmarkCaseResult] = []
    for case in selected:
        models = _models_for_case(case)
        if not models:
            raise ValueError(f"Benchmark case {case.case_id!r} must declare task.model_name or applicable_models.")
        for model_name in models:
            results.append(run_benchmark_case(case, model_name))
    passed = all(result.passed for result in results)
    return BenchmarkSuiteReport(
        results=results,
        passed=passed,
        total=len(results),
        passed_count=sum(result.passed for result in results),
    )
