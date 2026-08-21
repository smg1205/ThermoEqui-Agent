"""Contract tests for the model-agnostic benchmark framework."""

from __future__ import annotations

import pytest

from benchmarks.loader import load_benchmark_case, load_benchmark_cases
from benchmarks.runner import run_all_benchmarks, run_benchmark_case
from schemas.benchmark import (
    BenchmarkCase,
    BenchmarkReference,
    BenchmarkReferencePoint,
    BenchmarkSource,
)
from schemas.domain import ComponentIdentity, TaskManifest, ThermodynamicConditions


def test_benchmark_cases_load_with_traceable_sources() -> None:
    cases = load_benchmark_cases()

    assert cases
    assert all(case.source.identifier for case in cases)
    assert all(case.reference.points for case in cases)


def test_load_benchmark_case_by_id() -> None:
    case = load_benchmark_case("pr-methane-ethane-nitrogen-flash-110k")

    assert case.task.model_name == "Peng-Robinson"
    assert case.source.kind == "software_reference"


def test_software_reference_case_passes_runner() -> None:
    report = run_all_benchmarks()

    assert report.total >= 1
    assert report.passed_count == report.total
    assert report.passed is True
    peng_robinson_results = [result for result in report.results if result.model_name == "Peng-Robinson"]
    assert peng_robinson_results
    assert peng_robinson_results[0].passed is True


def test_run_benchmark_case_reports_metrics() -> None:
    case = load_benchmark_case("pr-methane-ethane-nitrogen-flash-110k")

    result = run_benchmark_case(case, "Peng-Robinson")

    assert result.metrics
    assert any(metric.field == "residual" for metric in result.metrics)
    assert result.validation_status in {"passed", "warning"}


def test_experimental_benchmark_cases_are_traceable_and_pass() -> None:
    cases = [case for case in load_benchmark_cases() if case.source.kind == "experimental"]

    assert len(cases) >= 4
    assert all(case.source.identifier.startswith("https://") for case in cases)
    assert all(case.task.model_name for case in cases)
    assert all(len(case.reference.points) > 1 for case in cases)

    report = run_all_benchmarks(cases)

    assert report.passed is True
    assert report.passed_count == report.total


def test_experimental_dataset_case_compares_every_reference_point() -> None:
    case = load_benchmark_case("nrtl-ethanol-benzene-isobaric-53kpa")

    result = run_benchmark_case(case, "NRTL")

    assert result.passed is True
    point_metrics = [metric for metric in result.metrics if metric.field.startswith("point[")]
    assert len(point_metrics) >= len(case.reference.points) * 3


def test_case_requires_model_or_applicable_models() -> None:
    case = BenchmarkCase(
        case_id="missing-model",
        title="Missing model declaration",
        task=TaskManifest(
            equilibrium_type="VLE",
            calculation_type="bubble_point",
            components=[ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2")],
            conditions=ThermodynamicConditions(
                pressure_kPa=101.325,
                liquid_composition=[1.0],
            ),
        ),
        reference=BenchmarkReference(points=[BenchmarkReferencePoint(temperature_K=353.0)]),
        source=BenchmarkSource(
            title="Contract test source",
            identifier="contract-test",
            kind="software_reference",
        ),
    )

    with pytest.raises(ValueError, match="must declare"):
        run_all_benchmarks([case])
