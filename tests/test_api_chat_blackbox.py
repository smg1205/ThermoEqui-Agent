"""Black-box tests for the /api/chat endpoint.

This file contains end-to-end HTTP-style tests for the chat interface
that exercises the model classification, task generation, calculation
execution, and unsupported-task handling behavior.

Supported query categories and example messages:

- Equilibrium calculation:
  "计算苯-甲苯在101.325 kPa下的T-x-y曲线"
  "使用 Peng-Robinson 计算甲烷、乙烷和氮气的 TP Flash：温度 110 K，压力 100 kPa，摩尔组成为 0.965、0.018、0.017。"

- Model comparison / concept QA:
  "NRTL和Peng-Robinson有什么区别？"
  "解释 NRTL"

- Incomplete condition handling:
  "计算苯-甲苯的泡点"

- Unsupported scope:
  "计算氯化钠溶液的VLE"
  "请设计一个精馏塔"

- Follow-up conversation / task correction:
  first send a calculation query, then send a follow-up query in the same
  conversation_id to test context continuity.

The tests are written as black-box scenarios and document the test goals,
request payloads, and expected response behavior.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import apps.api.main as api_module
from database.session import Repository, initialize_database


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


def test_chat_calculation_query_triggers_equilibrium_execution() -> None:
    """测试目的：验证计算类型查询会触发实际相平衡计算并返回验证结果。"""
    request = {
        "message": "计算苯-甲苯在101.325 kPa下的T-x-y曲线",
    }

    with client() as test_client:
        response = test_client.post("/api/chat", json=request, headers={"X-Request-ID": "test-chat-1"})

    assert response.status_code == 200
    payload = response.json()

    assert payload["conversation_id"]
    assert payload["intent"] == "EQUILIBRIUM_CALCULATION"
    assert payload["calculation"] is not None
    assert payload["calculation"]["validation"]["overall_status"] in {"passed", "warning"}
    assert response.headers["X-Request-ID"] == "test-chat-1"
    assert [step["phase"] for step in payload["execution_steps"]] == ["plan", "execute", "validate", "respond"]
    assert payload["execution_steps"][1]["tool_name"] == "phase_equilibrium"


def test_chat_model_comparison_question_returns_qa_intent() -> None:
    """测试目的：验证模型比较类问句不会误触相平衡计算。"""
    request = {
        "message": "NRTL和Peng-Robinson有什么区别？",
    }

    with client() as test_client:
        response = test_client.post("/api/chat", json=request)

    assert response.status_code == 200
    payload = response.json()

    assert payload["intent"] == "MODEL_SELECTION_QA"
    assert payload["calculation"] is None
    assert "缺少可识别的组分" not in payload["answer"]
    assert payload["statements"]


def test_local_ipv4_frontend_origin_is_allowed() -> None:
    with client() as test_client:
        response = test_client.options(
            "/api/chat",
            headers={
                "Origin": "http://127.0.0.1:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"


def test_local_frontend_on_an_alternate_port_is_allowed() -> None:
    with client() as test_client:
        response = test_client.options(
            "/api/chat",
            headers={
                "Origin": "http://localhost:3001",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3001"


def test_chat_incomplete_task_returns_warning_without_execution() -> None:
    """测试目的：验证条件不足时，接口返回缺失提示并且不执行计算。"""
    request = {
        "message": "计算苯-甲苯的泡点",
    }

    with client() as test_client:
        response = test_client.post("/api/chat", json=request)

    assert response.status_code == 200
    payload = response.json()

    assert payload["calculation"] is None
    assert payload["task"] is not None
    assert "pressure_kPa" in payload["answer"] or "缺少" in payload["answer"]
    assert any(statement["category"] == "Warning" for statement in payload["statements"])


def test_chat_unsupported_electrolyte_task_is_rejected() -> None:
    """测试目的：验证电解质等超范围任务被正确识别为 unsupported task。"""
    request = {
        "message": "计算氯化钠溶液的VLE",
    }

    with client() as test_client:
        response = test_client.post("/api/chat", json=request)

    assert response.status_code == 200
    payload = response.json()

    assert payload["intent"] == "UNSUPPORTED_TASK"
    assert payload["calculation"] is None
    assert "不支持" in payload["answer"] or "当前版本不支持" in payload["answer"]
    assert payload["statements"]


def test_chat_explicit_feed_composition_is_parsed_for_tp_flash() -> None:
    """测试目的：验证显式进料组分能够从对话中提取并用于 TP Flash 计算。"""
    request = {
        "message": (
            "使用 Peng-Robinson 计算甲烷、乙烷和氮气的 TP Flash：温度 110 K，压力 100 kPa，"
            "摩尔组成为 0.965、0.018、0.017。"
        ),
    }

    with client() as test_client:
        response = test_client.post("/api/chat", json=request)

    assert response.status_code == 200
    payload = response.json()

    assert payload["intent"] == "EQUILIBRIUM_CALCULATION"
    assert payload["task"] is not None
    assert payload["task"]["conditions"]["feed_composition"] == [0.965, 0.018, 0.017]
    assert payload["calculation"] is not None
    assert payload["calculation"]["validation"]["material_balance"]["passed"] is True


def test_chat_can_follow_up_with_same_conversation_id() -> None:
    """测试目的：验证同一个 conversation_id 下的后续请求保持上下文。"""

    with client() as test_client:
        first = test_client.post(
            "/api/chat",
            json={"message": "计算苯-甲苯在101.325 kPa下的T-x-y曲线"},
            headers={"X-Request-ID": "first-request"},
        )
        assert first.status_code == 200
        first_payload = first.json()
        conversation_id = first_payload["conversation_id"]
        assert isinstance(conversation_id, str)

        second = test_client.post(
            "/api/chat",
            json={"message": "请在同一个会话中改为乙醇-丙酮", "conversation_id": conversation_id},
            headers={"X-Request-ID": "second-request"},
        )

    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["conversation_id"] == conversation_id
    assert second_payload["intent"] in {
        "EQUILIBRIUM_CALCULATION",
        "TASK_CORRECTION",
        "MODEL_SELECTION_QA",
        "CONCEPT_QA",
    }
