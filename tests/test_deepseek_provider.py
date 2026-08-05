"""Behavioral tests for the DeepSeek provider at its public LLM interface."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

from agent.orchestrator import ConversationOrchestrator, DeterministicProvider
from agent.providers import DeepSeekProvider, LLMProviderError, LLMProviderOutputError
from apps.api.main import configured_provider
from schemas.domain import Intent, TaskManifest


@pytest.mark.asyncio
async def test_deepseek_provider_classifies_intent_through_chat_completions() -> None:
    captured: dict[str, object] = {}

    async def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "MODEL_SELECTION_QA"}}]},
        )

    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-flash",
        transport=httpx.MockTransport(respond),
    )

    intent = await provider.classify_intent("NRTL 和 Peng-Robinson 有什么区别？")

    assert intent == Intent.MODEL_SELECTION_QA
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["messages"][1] == {  # type: ignore[index]
        "role": "user",
        "content": "NRTL 和 Peng-Robinson 有什么区别？",
    }
    system_prompt = payload["messages"][0]["content"]  # type: ignore[index]
    assert all(intent_value.value in system_prompt for intent_value in Intent)
    assert "Never return NONE" in system_prompt
    assert payload["stream"] is False
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == 32


@pytest.mark.asyncio
async def test_deepseek_provider_normalizes_json_fenced_intent() -> None:
    async def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{"intent":"MODEL_SELECTION_QA"}\n```',
                        }
                    }
                ]
            },
        )

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )

    intent = await provider.classify_intent("NRTL 和 Peng-Robinson 有什么区别？")

    assert intent == Intent.MODEL_SELECTION_QA


def test_api_configuration_selects_deepseek_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "configured-test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/")

    provider = configured_provider()

    assert isinstance(provider, DeepSeekProvider)
    assert provider.model == "deepseek-v4-pro"
    assert provider.base_url == "https://api.deepseek.com"


def test_deepseek_provider_uses_system_https_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent.providers.urllib.request.getproxies",
        lambda: {
            "http": "http://127.0.0.1:7890",
            "https": "http://127.0.0.1:7897",
        },
    )

    provider = DeepSeekProvider(api_key="test-key")

    assert provider.proxy_url == "http://127.0.0.1:7897"


def test_api_configuration_without_deepseek_key_falls_back_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    provider = configured_provider()

    assert isinstance(provider, DeterministicProvider)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Calculate salt VLE at 100 kPa",
        "Calculate potassium chloride VLE at 100 kPa",
    ],
)
async def test_scope_rejection_precedes_external_provider_classification(message: str) -> None:
    async def respond(_: httpx.Request) -> httpx.Response:
        raise AssertionError("Excluded electrolyte tasks must not reach DeepSeek.")

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )

    response = await ConversationOrchestrator(provider).chat(message)

    assert response.intent == "UNSUPPORTED_TASK"
    assert response.task is None
    assert response.calculation is None


@pytest.mark.asyncio
async def test_deepseek_provider_requests_json_mode_for_task_manifests() -> None:
    captured_payload: dict[str, object] = {}
    manifest = {
        "equilibrium_type": "VLE",
        "calculation_type": "T-X-Y",
        "components": [{"component_id": "benzene", "name": "Benzene"}],
        "conditions": {"pressure_kPa": 101.325},
    }

    async def respond(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(manifest)}}]},
        )

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )

    task = await provider.formulate_task("计算苯在常压下的汽液平衡")

    assert task is not None
    assert task.calculation_type == "isobaric_vle"
    assert captured_payload["response_format"] == {"type": "json_object"}
    messages = captured_payload["messages"]
    assert isinstance(messages, list)
    system_prompt = messages[0]["content"]
    assert '"calculation_type"' in system_prompt
    assert '"tp_flash"' in system_prompt
    assert '"model_name"' in system_prompt
    assert "model_name may be null" in system_prompt
    assert "Never calculate equilibrium numbers" in system_prompt


@pytest.mark.asyncio
async def test_deepseek_provider_retries_invalid_task_manifest_once() -> None:
    responses = [
        '{"calculation_type":"TP_FLASH"}',
        json.dumps(
            {
                "equilibrium_type": "FLASH",
                "calculation_type": "TP_FLASH",
                "components": [{"component_id": "methane", "name": "Methane", "cas_number": "74-82-8"}],
                "conditions": {
                    "temperature_K": 110.0,
                    "pressure_kPa": 100.0,
                    "feed_composition": [1.0],
                },
            }
        ),
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

    task = await provider.formulate_task("计算甲烷 TP Flash")

    assert task is not None
    assert task.calculation_type == "tp_flash"
    assert request_count == 2


@pytest.mark.asyncio
async def test_deepseek_provider_rejects_tool_outside_allowlist() -> None:
    async def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"tool_name":"python_shell"}'}}]},
        )

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )
    task = TaskManifest.model_validate(
        {
            "equilibrium_type": "VLE",
            "calculation_type": "isobaric_vle",
            "components": [{"component_id": "benzene", "name": "Benzene"}],
            "conditions": {"pressure_kPa": 101.325},
        }
    )

    with pytest.raises(LLMProviderOutputError):
        await provider.select_tool(
            "计算汽液平衡",
            task,
            [{"name": "phase_equilibrium", "description": "Deterministic phase equilibrium"}],
        )


@pytest.mark.asyncio
async def test_deepseek_orchestration_rejects_invented_extra_component() -> None:
    responses = [
        "EQUILIBRIUM_CALCULATION",
        json.dumps(
            {
                "equilibrium_type": "FLASH",
                "calculation_type": "TP_FLASH",
                "components": [
                    {"component_id": "methane", "name": "Methane", "cas_number": "74-82-8"},
                    {"component_id": "ethane", "name": "Ethane", "cas_number": "74-84-0"},
                ],
                "conditions": {
                    "temperature_K": 150.0,
                    "pressure_kPa": 100.0,
                    "feed_composition": [0.5, 0.5],
                },
            }
        ),
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

    with pytest.raises(LLMProviderOutputError):
        await ConversationOrchestrator(provider).chat("Calculate methane TP Flash at 150 K and 100 kPa.")


@pytest.mark.asyncio
async def test_deepseek_followup_cannot_silently_replace_inherited_components() -> None:
    first_task = {
        "equilibrium_type": "FLASH",
        "calculation_type": "TP_FLASH",
        "components": [
            {"component_id": "methane", "name": "Methane", "cas_number": "74-82-8"},
            {"component_id": "ethane", "name": "Ethane", "cas_number": "74-84-0"},
        ],
        "conditions": {
            "temperature_K": 150.0,
            "pressure_kPa": 100.0,
            "feed_composition": [0.5, 0.5],
        },
        "model_name": "Peng-Robinson",
    }
    changed_task = {
        **first_task,
        "components": [
            {"component_id": "methane", "name": "Methane", "cas_number": "74-82-8"},
            {"component_id": "propane", "name": "Propane", "cas_number": "74-98-6"},
        ],
        "conditions": {
            **first_task["conditions"],
            "pressure_kPa": 200.0,
        },
    }
    responses = [
        "EQUILIBRIUM_CALCULATION",
        json.dumps(first_task),
        json.dumps({"tool_name": "phase_equilibrium"}),
        "The deterministic result was received.",
        "TASK_CORRECTION",
        json.dumps(changed_task),
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
    orchestrator = ConversationOrchestrator(provider)
    first = await orchestrator.chat("Calculate methane and ethane TP Flash at 150 K and 100 kPa.")

    with pytest.raises(LLMProviderOutputError):
        await orchestrator.chat("Change pressure to 200 kPa.", first.conversation_id)


@pytest.mark.asyncio
async def test_deepseek_new_task_requires_component_identity_evidence() -> None:
    responses = [
        "EQUILIBRIUM_CALCULATION",
        json.dumps(
            {
                "equilibrium_type": "FLASH",
                "calculation_type": "TP_FLASH",
                "components": [{"component_id": "methane", "name": "Methane", "cas_number": "74-82-8"}],
                "conditions": {
                    "temperature_K": 150.0,
                    "pressure_kPa": 100.0,
                    "feed_composition": [1.0],
                },
            }
        ),
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

    with pytest.raises(LLMProviderOutputError):
        await ConversationOrchestrator(provider).chat("Calculate a TP Flash at 150 K and 100 kPa.")


@pytest.mark.asyncio
async def test_deepseek_component_name_cannot_carry_another_compounds_cas() -> None:
    responses = [
        "EQUILIBRIUM_CALCULATION",
        json.dumps(
            {
                "equilibrium_type": "VLE",
                "calculation_type": "BUBBLE_POINT",
                "components": [{"component_id": "acetone", "name": "Acetone", "cas_number": "71-43-2"}],
                "conditions": {
                    "pressure_kPa": 101.325,
                    "liquid_composition": [1.0],
                },
                "model_name": "Peng-Robinson",
            }
        ),
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

    with pytest.raises(LLMProviderOutputError):
        await ConversationOrchestrator(provider).chat("Calculate the acetone bubble point at 101.325 kPa.")


@pytest.mark.asyncio
async def test_deepseek_cannot_omit_external_component_from_mixed_system() -> None:
    responses = [
        "EQUILIBRIUM_CALCULATION",
        json.dumps(
            {
                "equilibrium_type": "VLE",
                "calculation_type": "BUBBLE_POINT",
                "components": [{"component_id": "benzene", "name": "Benzene", "cas_number": "71-43-2"}],
                "conditions": {
                    "pressure_kPa": 101.325,
                    "liquid_composition": [1.0],
                },
                "model_name": "Ideal/Raoult",
            }
        ),
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

    with pytest.raises(LLMProviderOutputError):
        await ConversationOrchestrator(provider).chat("Calculate the acetone and benzene bubble point at 101.325 kPa.")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Calculate acetone TP Flash at 300 K and 100 kPa without ethanol.",
        "Calculate acetone TP Flash at 300 K and 100 kPa without any ethanol.",
        "Calculate acetone TP Flash at 300 K and 100 kPa without adding ethanol.",
        "Calculate acetone TP Flash at 300 K and 100 kPa excluding any ethanol.",
        "Calculate acetone TP Flash at 300 K and 100 kPa; do not include ethanol.",
        "Calculate acetone TP Flash at 300 K and 100 kPa; do not add ethanol.",
        "Calculate acetone TP Flash at 300 K and 100 kPa; do not use ethanol.",
        "Calculate acetone TP Flash at 300 K and 100 kPa with no added ethanol.",
        "Calculate acetone TP Flash at 300 K and 100 kPa without any trace of ethanol.",
        "Calculate acetone TP Flash at 300 K and 100 kPa free of ethanol.",
        "Calculate acetone TP Flash at 300 K and 100 kPa; for example, ethanol.",
        "Calculate acetone TP Flash at 300 K and 100 kPa; ethanol, for example.",
        "Calculate acetone TP Flash at 300 K and 100 kPa; ethanol should be excluded.",
        "Calculate acetone TP Flash at 300 K and 100 kPa; ethanol-free.",
        "Compare acetone and ethanol in a TP Flash at 300 K and 100 kPa.",
    ],
)
async def test_deepseek_requires_clarification_for_ambiguous_component_role(message: str) -> None:
    responses = [
        "EQUILIBRIUM_CALCULATION",
        json.dumps(
            {
                "equilibrium_type": "FLASH",
                "calculation_type": "TP_FLASH",
                "components": [
                    {"component_id": "acetone", "name": "Acetone", "cas_number": "67-64-1"},
                    {"component_id": "ethanol", "name": "Ethanol", "cas_number": "64-17-5"},
                ],
                "conditions": {
                    "temperature_K": 300.0,
                    "pressure_kPa": 100.0,
                    "feed_composition": [0.5, 0.5],
                },
                "model_name": "Peng-Robinson",
            }
        ),
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

    with pytest.raises(LLMProviderOutputError):
        await ConversationOrchestrator(provider).chat(message)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("literal", "canonical_name", "cas_number"),
    [
        ("2-propanol", "Isopropanol", "67-63-0"),
        ("isopropyl alcohol", "Isopropanol", "67-63-0"),
        ("ethyl alcohol", "Ethanol", "64-17-5"),
        ("n-hexane", "Hexane", "110-54-3"),
        ("R134a", "Norflurane", "811-97-2"),
    ],
)
async def test_deepseek_accepts_database_verified_component_aliases(
    literal: str,
    canonical_name: str,
    cas_number: str,
) -> None:
    responses = [
        "EQUILIBRIUM_CALCULATION",
        json.dumps(
            {
                "equilibrium_type": "VLE",
                "calculation_type": "BUBBLE_POINT",
                "components": [
                    {
                        "component_id": cas_number,
                        "name": literal,
                        "cas_number": cas_number,
                    }
                ],
                "conditions": {
                    "pressure_kPa": 101.325,
                    "liquid_composition": [1.0],
                },
                "model_name": "Peng-Robinson",
            }
        ),
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

    _, task = await ConversationOrchestrator(provider).parse(f"Calculate the bubble point of {literal} at 101.325 kPa.")

    assert task is not None
    assert task.components[0].name == canonical_name
    assert task.components[0].cas_number == cas_number


@pytest.mark.asyncio
async def test_deepseek_cannot_omit_middle_component_from_comma_separated_list() -> None:
    responses = [
        "EQUILIBRIUM_CALCULATION",
        json.dumps(
            {
                "equilibrium_type": "FLASH",
                "calculation_type": "TP_FLASH",
                "components": [
                    {"component_id": "acetone", "name": "Acetone", "cas_number": "67-64-1"},
                    {
                        "component_id": "2-propanol",
                        "name": "2-Propanol",
                        "cas_number": "67-63-0",
                    },
                ],
                "conditions": {
                    "temperature_K": 300.0,
                    "pressure_kPa": 100.0,
                    "feed_composition": [0.5, 0.5],
                },
                "model_name": "Peng-Robinson",
            }
        ),
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

    with pytest.raises(LLMProviderOutputError):
        await ConversationOrchestrator(provider).parse(
            "Calculate acetone, ethanol, and 2-propanol TP Flash at 300 K and 100 kPa."
        )


@pytest.mark.asyncio
async def test_deepseek_provider_withholds_ungrounded_numbers_and_citations() -> None:
    async def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "预测汽化率为 .5，根据 NIST Chemistry WebBook。",
                        }
                    }
                ]
            },
        )

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )

    statements = await provider.answer_with_evidence("计算苯和甲苯的汽化率")

    assert statements[0].category == "Warning"
    assert ".5" not in statements[0].text
    assert "NIST" not in statements[0].text


@pytest.mark.asyncio
async def test_deepseek_provider_rejects_malformed_response_schema() -> None:
    async def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"unexpected": "shape"}]})

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )

    with pytest.raises(LLMProviderOutputError):
        await provider.classify_intent("解释 NRTL")


@pytest.mark.asyncio
async def test_deepseek_provider_rejects_empty_assistant_content() -> None:
    async def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "   "}}]},
        )

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )

    with pytest.raises(LLMProviderOutputError):
        await provider.classify_intent("解释 NRTL")


@pytest.mark.asyncio
async def test_deepseek_provider_sanitizes_remote_api_errors() -> None:
    async def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "server-secret-detail"}},
        )

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )

    with pytest.raises(LLMProviderError) as captured:
        await provider.classify_intent("解释 NRTL")

    assert captured.value.provider == "DeepSeek"
    assert captured.value.status_code == 401
    assert "test-key" not in str(captured.value)
    assert "server-secret-detail" not in str(captured.value)


def test_compose_forwards_deepseek_configuration_to_api() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    environment = compose["services"]["api"]["environment"]

    assert environment["DEEPSEEK_API_KEY"] == "${DEEPSEEK_API_KEY:-}"
    assert environment["DEEPSEEK_MODEL"] == "${DEEPSEEK_MODEL:-deepseek-v4-flash}"
    assert environment["DEEPSEEK_BASE_URL"] == "${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"


@pytest.mark.asyncio
async def test_deepseek_orchestration_uses_real_engine_and_receives_validated_result() -> None:
    task_payload = {
        "equilibrium_type": "FLASH",
        "calculation_type": "TP_FLASH",
        "components": [
            {"component_id": "nitrogen", "name": "Nitrogen", "cas_number": "7727-37-9"},
            {"component_id": "methane", "name": "Methane", "cas_number": "74-82-8"},
            {"component_id": "ethane", "name": "Ethane", "cas_number": "74-84-0"},
        ],
        "conditions": {
            "temperature_K": 110.0,
            "pressure_kPa": 100.0,
            "feed_composition": [0.017, 0.965, 0.018],
        },
        "model_name": "Peng-Robinson",
    }
    responses = [
        "EQUILIBRIUM_CALCULATION",
        json.dumps(task_payload),
        json.dumps({"tool_name": "phase_equilibrium"}),
        "计算结果已通过确定性验证。",
    ]
    requests: list[dict[str, object]] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": responses[len(requests) - 1]}}]},
        )

    provider = DeepSeekProvider(
        api_key="test-key",
        transport=httpx.MockTransport(respond),
    )

    response = await ConversationOrchestrator(provider).chat(
        "使用 Peng-Robinson 计算甲烷、乙烷和氮气的 TP Flash：110 K，100 kPa，组成 0.965、0.018、0.017。"
    )

    assert response.calculation is not None
    assert response.task is not None
    assert response.task.calculation_type == "tp_flash"
    assert [component.cas_number for component in response.task.components] == [
        "74-82-8",
        "74-84-0",
        "7727-37-9",
    ]
    assert response.task.conditions.feed_composition == [0.965, 0.018, 0.017]
    assert len(response.calculation.result.phases) == 2
    assert response.calculation.result.backend_version.startswith("thermo/")
    assert response.calculation.validation.overall_status in {"passed", "warning"}
    tool_selection_input = json.loads(requests[2]["messages"][1]["content"])  # type: ignore[index]
    assert tool_selection_input["available_tools"][0]["name"] == "phase_equilibrium"
    interpretation_input = json.loads(requests[3]["messages"][1]["content"])  # type: ignore[index]
    assert len(interpretation_input["result"]["phases"]) == 2
    assert interpretation_input["validation"]["overall_status"] in {"passed", "warning"}
