"""Load benchmark cases from the shared knowledge directory."""

from __future__ import annotations

from pathlib import Path

import yaml

from schemas.benchmark import BenchmarkCase


def benchmark_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "knowledge" / "benchmarks"


def load_benchmark_cases() -> list[BenchmarkCase]:
    directory = benchmark_directory()
    if not directory.exists():
        raise FileNotFoundError(f"Benchmark directory does not exist: {directory}")
    cases: list[BenchmarkCase] = []
    for path in sorted(directory.glob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Benchmark file {path} must contain a YAML mapping.")
        case = BenchmarkCase.model_validate(loaded)
        if any(existing.case_id == case.case_id for existing in cases):
            raise ValueError(f"Duplicate benchmark case_id {case.case_id!r} in {path}.")
        cases.append(case)
    return cases


def load_benchmark_case(case_id: str) -> BenchmarkCase:
    for case in load_benchmark_cases():
        if case.case_id == case_id:
            return case
    raise KeyError(f"No benchmark case with case_id {case_id!r}.")
