"""Behavioral tests for deterministic multi-model comparison."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import apps.api.main as api_module
from agent.comparison import compare_models
from database.session import Repository, initialize_database
from schemas.domain import (
    ComponentIdentity,
    FailureType,
    ModelRecommendation,
    ScoreBreakdown,
    TaskManifest,
    ThermodynamicConditions,
)
from thermo_engine.errors import ThermoEquiError

BENZENE = ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2")
TOLUENE = ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3")


def _bubble_task() -> TaskManifest:
    return TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[BENZENE, TOLUENE],
        conditions=ThermodynamicConditions(
            pressure_kPa=101.325,
            liquid_composition=[0.5, 0.5],
        ),
        points=5,
    )


def test_compare_models_runs_every_executable_backend_and_aggregates_counts() -> None:
    response = compare_models(_bubble_task())

    assert len(response.entries) >= 2
    assert any(entry.model_name == "Ideal/Raoult" for entry in response.entries)
    assert any(entry.model_name == "UNIFAC" for entry in response.entries)
    assert response.executed_count == len(response.entries)
    assert response.failed_count == 0
    assert response.passed_count + response.warning_count == response.executed_count
    assert "对比" in response.summary


def test_compare_entries_cross_the_public_validation_gate() -> None:
    response = compare_models(_bubble_task())

    for entry in response.entries:
        assert entry.result is not None
        assert entry.validation is not None
        assert entry.failure is None
        assert entry.validation.overall_status in {"passed", "warning"}
        assert entry.result.model_name == entry.model_name


def test_compare_models_reports_structured_failure_instead_of_aborting(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_candidate = ModelRecommendation(
        model_name="Ideal/Raoult",
        score=50.0,
        executable=True,
        reasons=["Test candidate."],
        breakdown=ScoreBreakdown(
            phase_support_score=30.0,
            system_match_score=10.0,
            condition_match_score=10.0,
            parameter_availability_score=0.0,
            evidence_quality_score=0.0,
            extrapolation_penalty=0.0,
            numerical_risk_penalty=0.0,
        ),
    )

    def fake_rank(task, available_parameter_models=None):
        del task, available_parameter_models
        return [fake_candidate]

    monkeypatch.setattr("agent.comparison.rank_executable_models", fake_rank)

    def fail_resolve(task):
        raise ThermoEquiError(
            FailureType.NUMERICAL_NONCONVERGENCE,
            "Solver failed for comparison.",
            "Review conditions.",
        )

    monkeypatch.setattr("agent.comparison.resolve_backend", fail_resolve)

    response = compare_models(_bubble_task())

    assert len(response.entries) == 1
    assert response.entries[0].failure is not None
    assert response.entries[0].failure.failure_type == FailureType.NUMERICAL_NONCONVERGENCE
    assert response.failed_count == 1


def api_client() -> TestClient:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_database(engine)
    api_module.repository = Repository(engine)

    @asynccontextmanager
    async def no_op_lifespan(_):  # type: ignore[no-untyped-def]
        yield

    api_module.app.router.lifespan_context = no_op_lifespan
    return TestClient(api_module.app)


def test_compare_endpoint_returns_model_comparison_response() -> None:
    with api_client() as client:
        response = client.post(
            "/api/calculations/compare",
            json=_bubble_task().model_dump(mode="json"),
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["executed_count"] > 0
        assert any(entry["model_name"] == "Ideal/Raoult" for entry in payload["entries"])
        assert any(entry["model_name"] == "UNIFAC" for entry in payload["entries"])
