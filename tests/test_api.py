"""HTTP contract tests through the FastAPI application seam."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import apps.api.main as api_module
from agent.orchestrator import ConversationOrchestrator
from agent.providers import DeepSeekProvider
from database.models import EvidenceRecordRow, TaskRow
from database.session import Repository, initialize_database


def parameter_payload() -> dict[str, object]:
    return json.loads((Path(__file__).parent / "fixtures" / "user_supplied_parameter.json").read_text(encoding="utf-8"))


def client() -> TestClient:
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


def test_health_and_error_responses_have_request_id() -> None:
    with client() as test_client:
        response = test_client.get("/health", headers={"X-Request-ID": "request-test"})
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "request-test"
        bad = test_client.post(
            "/api/calculations/isobaric-vle",
            json={
                "equilibrium_type": "VLE",
                "calculation_type": "isobaric_vle",
                "components": [
                    {"component_id": "benzene", "name": "Benzene"},
                    {"component_id": "toluene", "name": "Toluene"},
                ],
                "conditions": {},
            },
        )
        assert bad.status_code == 422
        assert bad.json()["error"]["code"] == "missing_data"


def test_chat_persists_real_run_and_exports_json_and_csv() -> None:
    with client() as test_client:
        response = test_client.post("/api/chat", json={"message": "计算苯-甲苯在101.325 kPa下的T-x-y曲线"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["calculation"]["validation"]["overall_status"] in {"passed", "warning"}
        assert [step["phase"] for step in payload["execution_steps"]] == [
            "plan",
            "execute",
            "validate",
            "respond",
        ]
        assert payload["execution_steps"][1]["tool_name"] == "phase_equilibrium"
        run_id = payload["calculation"]["result"]["run_id"]
        run = test_client.get(f"/api/runs/{run_id}")
        assert run.status_code == 200
        assert run.json()["input_snapshot"]["conditions"]["pressure_kPa"] == 101.325
        history = test_client.get("/api/runs?limit=1&offset=0")
        assert history.status_code == 200
        assert history.json()["total"] == 1
        assert history.json()["items"][0]["run_id"] == run_id
        assert "result" not in history.json()["items"][0]
        assert "input_snapshot" not in history.json()["items"][0]
        run_status = payload["calculation"]["validation"]["overall_status"]
        filtered_history = test_client.get(f"/api/runs?status={run_status}")
        assert filtered_history.status_code == 200
        assert filtered_history.json()["total"] == 1
        exported_json = test_client.get(f"/api/runs/{run_id}/export?format=json")
        exported_csv = test_client.get(f"/api/runs/{run_id}/export?format=csv")
        assert exported_json.status_code == 200
        assert exported_csv.status_code == 200
        assert "temperature_K" in exported_csv.text
        with Session(api_module.repository.engine) as session:
            evidence_count = session.scalar(
                select(func.count()).select_from(EvidenceRecordRow).where(EvidenceRecordRow.run_id == run_id)
            )
            task_row = session.get(TaskRow, payload["task"]["task_id"])
        assert evidence_count is not None and evidence_count > 0
        assert task_row is not None
        assert task_row.conversation_id == payload["conversation_id"]


def test_run_can_be_exported_as_a_dwsim_flowsheet(monkeypatch: pytest.MonkeyPatch) -> None:
    def write_dwsim_file(run: object, destination: Path) -> Path:
        destination.write_bytes(b"dwsim-flowsheet")
        return destination

    monkeypatch.setattr(api_module, "export_dwsim_flowsheet", write_dwsim_file)
    with client() as test_client:
        response = test_client.post(
            "/api/calculations/isobaric-vle",
            json={
                "equilibrium_type": "VLE",
                "calculation_type": "isobaric_vle",
                "components": [
                    {"component_id": "benzene", "name": "Benzene"},
                    {"component_id": "toluene", "name": "Toluene"},
                ],
                "conditions": {"pressure_kPa": 101.325},
                "model_name": "Ideal/Raoult",
                "points": 3,
            },
        )
        assert response.status_code == 200
        run_id = response.json()["result"]["run_id"]
        exported = test_client.get(f"/api/runs/{run_id}/export?format=dwsim")
    assert exported.status_code == 200
    assert exported.content == b"dwsim-flowsheet"
    assert exported.headers["content-disposition"].endswith(f'filename="{run_id}.dwxmz"')


def test_direct_calculation_persists_its_task_manifest() -> None:
    task = {
        "equilibrium_type": "VLE",
        "calculation_type": "isobaric_vle",
        "components": [
            {"component_id": "benzene", "name": "Benzene"},
            {"component_id": "toluene", "name": "Toluene"},
        ],
        "conditions": {"pressure_kPa": 101.325},
        "model_name": "Ideal/Raoult",
        "points": 3,
    }

    with client() as test_client:
        response = test_client.post("/api/calculations/isobaric-vle", json=task)
        assert response.status_code == 200
        result = response.json()["result"]
        with Session(api_module.repository.engine) as session:
            task_row = session.get(TaskRow, result["task_id"])
            assert task_row is not None
            assert task_row.conversation_id is None
            assert task_row.manifest == result["input_snapshot"]


def test_openapi_contains_all_required_routes() -> None:
    expected = {
        "/api/chat",
        "/api/tasks/parse",
        "/api/models/recommend",
        "/api/models",
        "/api/parameters",
        "/api/parameters/search",
        "/api/calculations/bubble-point",
        "/api/calculations/dew-point",
        "/api/calculations/isobaric-vle",
        "/api/calculations/isothermal-vle",
        "/api/calculations/tp-flash",
        "/api/calculations/azeotrope",
        "/api/calculations/lle",
        "/api/validation",
        "/api/runs",
        "/api/runs/{run_id}",
        "/api/runs/{run_id}/export",
        "/health",
    }
    assert expected <= set(api_module.app.openapi()["paths"])


def test_parameter_create_search_and_duplicate_conflict_are_structured() -> None:
    payload = parameter_payload()
    with client() as test_client:
        created = test_client.post("/api/parameters", json=payload)
        assert created.status_code == 201

        searched = test_client.get(
            "/api/parameters/search",
            params=[
                ("model_name", payload["model_name"]),
                ("components", "test-component-a"),
                ("components", "test-component-b"),
            ],
        )
        assert searched.status_code == 200
        assert [item["parameter_set_id"] for item in searched.json()] == [payload["parameter_set_id"]]

        duplicate = test_client.post(
            "/api/parameters",
            json=payload,
            headers={"X-Request-ID": "duplicate-parameter-request"},
        )
        assert duplicate.status_code == 409
        assert duplicate.json()["error"]["code"] == "duplicate_parameter_set"
        assert duplicate.json()["error"]["request_id"] == "duplicate-parameter-request"


def test_parameter_api_rejects_missing_evidence_and_test_fixtures() -> None:
    missing_evidence = {
        **parameter_payload(),
        "source_type": "literature",
        "source_title": None,
        "source_identifier": None,
    }
    fixture = json.loads((Path(__file__).parent / "fixtures" / "synthetic_nrtl.json").read_text(encoding="utf-8"))

    with client() as test_client:
        invalid = test_client.post("/api/parameters", json=missing_evidence)
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "invalid_input"

        unsafe = test_client.post("/api/parameters", json=fixture)
        assert unsafe.status_code == 422
        assert unsafe.json()["error"]["code"] == "invalid_input"


def test_unexpected_server_error_is_sanitized_and_has_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_persistence(*_: object, **__: object) -> None:
        raise RuntimeError("sensitive-database-detail")

    with client() as test_client:
        monkeypatch.setattr(api_module.repository, "save_chat_and_run", fail_persistence)
        response = test_client.post(
            "/api/chat",
            json={"message": "解释 NRTL"},
            headers={"X-Request-ID": "unexpected-error-request"},
        )

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "unexpected-error-request"
    assert response.json()["error"]["code"] == "internal_server_error"
    assert response.json()["error"]["request_id"] == "unexpected-error-request"
    assert "sensitive-database-detail" not in response.text


def test_peng_robinson_api_returns_thermo_and_chemsep_provenance() -> None:
    task = {
        "equilibrium_type": "FLASH",
        "calculation_type": "tp_flash",
        "components": [
            {"component_id": "methane", "name": "Methane", "cas_number": "74-82-8"},
            {"component_id": "ethane", "name": "Ethane", "cas_number": "74-84-0"},
            {"component_id": "nitrogen", "name": "Nitrogen", "cas_number": "7727-37-9"},
        ],
        "conditions": {
            "temperature_K": 110.0,
            "pressure_kPa": 100.0,
            "feed_composition": [0.965, 0.018, 0.017],
        },
        "model_name": "Peng-Robinson",
    }

    with client() as test_client:
        response = test_client.post("/api/calculations/tp-flash", json=task)

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["backend_version"].startswith("thermo/")
    assert payload["result"]["parameter_set_id"].startswith("chemsep-pr:")
    assert payload["validation"]["material_balance"]["passed"]
    assert any(source["source_title"] == "ChemSep PR" for source in payload["parameter_sources"])
    assert any(source["source_title"] == "CalebBell/thermo" for source in payload["parameter_sources"])
    interaction_source = next(
        source for source in payload["parameter_sources"] if source["source_title"] == "ChemSep PR"
    )
    assert '"74-82-8"' in interaction_source["component_order"]
    assert interaction_source["parameter_set_id"] == payload["result"]["parameter_set_id"]
    recommendation = next(item for item in payload["model_recommendations"] if item["model_name"] == "Peng-Robinson")
    assert recommendation["executable"]


def test_deepseek_chat_normalizes_tp_flash_alias_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        "EQUILIBRIUM_CALCULATION",
        json.dumps(
            {
                "equilibrium_type": "FLASH",
                "calculation_type": "TP-FLASH",
                "components": [
                    {"component_id": "methane", "name": "Methane", "cas_number": "74-82-8"},
                    {"component_id": "ethane", "name": "Ethane", "cas_number": "74-84-0"},
                    {"component_id": "nitrogen", "name": "Nitrogen", "cas_number": "7727-37-9"},
                ],
                "conditions": {
                    "temperature_K": 110.0,
                    "pressure_kPa": 100.0,
                    "feed_composition": [0.965, 0.018, 0.017],
                },
                "model_name": "Peng-Robinson",
            }
        ),
        json.dumps({"tool_name": "phase_equilibrium"}),
        "The deterministic result and validation payload were received.",
    ]
    request_count = 0

    async def respond(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        content = responses[request_count]
        request_count += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )
    monkeypatch.setattr(api_module, "orchestrator", ConversationOrchestrator(provider))

    with client() as test_client:
        response = test_client.post(
            "/api/chat",
            json={
                "message": (
                    "Calculate methane, ethane and nitrogen TP Flash at 110 K and 100 kPa "
                    "with composition 0.965, 0.018, 0.017."
                )
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"]["calculation_type"] == "tp_flash"
    assert payload["calculation"]["result"]["backend_version"].startswith("thermo/")
    assert payload["calculation"]["validation"]["material_balance"]["passed"]


def test_backend_comparison_question_is_not_misclassified_as_calculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        "EQUILIBRIUM_CALCULATION",
        "Thermo, Phasepy, and Clapeyron use different deterministic backend implementations.",
    ]
    request_count = 0

    async def respond(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        content = responses[request_count]
        request_count += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )
    monkeypatch.setattr(api_module, "orchestrator", ConversationOrchestrator(provider))

    with client() as test_client:
        response = test_client.post(
            "/api/chat",
            json={"message": "thermo、Phasepy 和 Clapeyron.jl 三个计算后端有什么区别？"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "MODEL_SELECTION_QA"
    assert payload["task"] is None
    assert "缺少可识别的组分" not in payload["answer"]
    assert request_count == 2


def test_deepseek_chat_recovers_explicit_feed_composition_from_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        "EQUILIBRIUM_CALCULATION",
        json.dumps(
            {
                "equilibrium_type": "FLASH",
                "calculation_type": "TP_FLASH",
                "components": [
                    {"component_id": "methane", "name": "Methane", "cas_number": "74-82-8"},
                    {"component_id": "ethane", "name": "Ethane", "cas_number": "74-84-0"},
                    {"component_id": "nitrogen", "name": "Nitrogen", "cas_number": "7727-37-9"},
                ],
                "conditions": {
                    "temperature_K": 110.0,
                    "pressure_kPa": 100.0,
                },
                "model_name": "Peng-Robinson",
            }
        ),
        json.dumps({"tool_name": "phase_equilibrium"}),
        "The deterministic result and validation payload were received.",
    ]
    request_count = 0

    async def respond(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        content = responses[request_count]
        request_count += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )
    monkeypatch.setattr(api_module, "orchestrator", ConversationOrchestrator(provider))

    with client() as test_client:
        response = test_client.post(
            "/api/chat",
            json={
                "message": (
                    "使用 Peng-Robinson 计算甲烷、乙烷和氮气的 TP Flash：温度 110 K，压力 100 kPa，"
                    "摩尔组成为 0.965、0.018、0.017。"
                )
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"]["conditions"]["feed_composition"] == [0.965, 0.018, 0.017]
    assert payload["calculation"] is not None
    assert payload["calculation"]["validation"]["material_balance"]["passed"]
    assert request_count == 4


def test_request_validation_and_not_found_use_unified_error_shape() -> None:
    with client() as test_client:
        invalid = test_client.post(
            "/api/models/recommend",
            json={
                "equilibrium_type": "VLE",
                "calculation_type": "isobaric_vle",
                "components": [{"component_id": "benzene", "name": "Benzene"}],
                "conditions": {"pressure_kPa": -1},
            },
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "request_validation_error"
        invalid_history = test_client.get("/api/runs?limit=0")
        assert invalid_history.status_code == 422
        assert invalid_history.json()["error"]["code"] == "request_validation_error"
        invalid_status = test_client.get("/api/runs?status=unknown")
        assert invalid_status.status_code == 422
        assert invalid_status.json()["error"]["code"] == "request_validation_error"
        missing = test_client.get("/api/runs/not-found")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "http_404"


def test_deepseek_failure_returns_sanitized_fallback_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def reject(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "server-secret-detail"}})

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(reject),
    )
    monkeypatch.setattr(api_module, "orchestrator", ConversationOrchestrator(provider))

    with client() as test_client:
        response = test_client.post("/api/chat", json={"message": "解释 NRTL"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["statements"]
    assert payload["statements"][0]["category"] == "Warning"
    assert "test-key" not in response.text
    assert "server-secret-detail" not in response.text


def test_invalid_deepseek_intent_falls_back_to_deterministic_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        "NONE",
        "NRTL 是液相活度系数模型；Peng-Robinson 是立方状态方程。",
    ]
    request_count = 0

    async def respond(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        content = responses[request_count]
        request_count += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )
    monkeypatch.setattr(api_module, "orchestrator", ConversationOrchestrator(provider))

    with client() as test_client:
        response = test_client.post("/api/chat", json={"message": "NRTL和Peng-Robinson有什么区别？"})

    assert response.status_code == 200
    assert response.json()["intent"] == "MODEL_SELECTION_QA"
    assert request_count == 2


def test_unparseable_deepseek_task_returns_sanitized_fallback_response(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        "EQUILIBRIUM_CALCULATION",
        "not-a-task-manifest",
        "still-not-a-task-manifest",
    ]
    request_count = 0

    async def respond(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        content = responses[request_count]
        request_count += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )
    monkeypatch.setattr(api_module, "orchestrator", ConversationOrchestrator(provider))

    with client() as test_client:
        response = test_client.post("/api/chat", json={"message": "计算苯-甲苯的T-x-y曲线"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["statements"]
    assert payload["statements"][0]["category"] == "Warning"
    assert "not-a-task-manifest" not in response.text


def test_malformed_deepseek_envelope_returns_sanitized_fallback_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "upstream-private-content"})

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )
    monkeypatch.setattr(api_module, "orchestrator", ConversationOrchestrator(provider))

    with client() as test_client:
        response = test_client.post("/api/chat", json={"message": "解释 NRTL"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["statements"]
    assert payload["statements"][0]["category"] == "Warning"
    assert "upstream-private-content" not in response.text


def test_infinite_dilution_activity_endpoint_fails_structurally_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PGSSI_CHECKPOINT", raising=False)

    with client() as test_client:
        response = test_client.post(
            "/api/calculations/infinite-dilution-activity",
            json={
                "equilibrium_type": "VLE",
                "calculation_type": "infinite_dilution_activity",
                "components": [
                    {
                        "component_id": "ethanol",
                        "name": "ethanol",
                        "cas_number": "64-17-5",
                        "smiles": "CCO",
                        "aliases": [],
                    },
                    {
                        "component_id": "water",
                        "name": "water",
                        "cas_number": "7732-18-5",
                        "smiles": "O",
                        "aliases": [],
                    },
                ],
                "conditions": {"temperature_K": 298.15},
                "composition_basis": "mole_fraction",
                "requested_outputs": ["table", "validation"],
                "validation_requirements": ["composition_balance", "equilibrium_residual", "convergence"],
                "assumptions": [],
                "points": 21,
                "parameters": [],
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "missing_parameters"
    assert "checkpoint" in response.json()["error"]["message"].casefold()
