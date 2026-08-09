"""API tests for parameter set creation, search, and model recommendation endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import apps.api.main as api_module
from database.session import Repository, initialize_database
from schemas.domain import ParameterSet, TaskManifest, ThermodynamicConditions, ComponentIdentity


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


def test_create_parameter_set_and_search_by_components() -> None:
    with api_client() as client:
        parameter_set = ParameterSet(
            model_name="NRTL",
            component_order=["71-43-2", "108-88-3"],
            parameters={"tau12": 0.5, "tau21": -0.4, "alpha": 0.3},
            parameter_form="NRTL",
            units={"tau12": "dimensionless", "tau21": "dimensionless", "alpha": "dimensionless"},
            equilibrium_types=["VLE"],
            source_type="literature",
            source_title="Test NRTL source",
            source_identifier="https://example.invalid/nrtl",
            quality_level="reviewed",
        )
        response = client.post("/api/parameters", json=parameter_set.model_dump(mode="json"))
        assert response.status_code == 201
        payload = response.json()
        assert payload["model_name"] == "NRTL"

        search = client.get(
            "/api/parameters/search?model_name=NRTL&components=71-43-2&components=108-88-3"
        )
        assert search.status_code == 200
        results = search.json()
        assert len(results) == 1
        assert results[0]["model_name"] == "NRTL"
        assert results[0]["component_order"] == ["71-43-2", "108-88-3"]


def test_model_recommendations_endpoint_returns_model_cards_for_task() -> None:
    with api_client() as client:
        task = TaskManifest(
            equilibrium_type="VLE",
            calculation_type="isobaric_vle",
            components=[
                ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2"),
                ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3"),
            ],
            conditions=ThermodynamicConditions(pressure_kPa=101.325),
            points=11,
        )
        response = client.post("/api/models/recommend", json=task.model_dump(mode="json"))
        assert response.status_code == 200
        recommendations = response.json()
        assert any(rec["model_name"] == "Ideal/Raoult" for rec in recommendations)
        assert any("executable" in rec for rec in recommendations)
