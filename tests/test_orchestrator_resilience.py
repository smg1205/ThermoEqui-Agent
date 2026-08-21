"""Resilience tests for external-provider failures during chat orchestration."""

from __future__ import annotations

import asyncio

from agent.executor import execute_task
from agent.graph_workflow import BoundedAgentGraph
from agent.orchestrator import ConversationOrchestrator
from agent.providers import LLMProviderError
from agent.tools import DEFAULT_TOOL_REGISTRY
from schemas.domain import (
    ComponentIdentity,
    EvidenceStatement,
    Intent,
    TaskManifest,
    ThermodynamicConditions,
)


class ClassifyFailingProvider:
    async def classify_intent(self, message: str) -> Intent:
        del message
        raise LLMProviderError("Test")


class InterpretFailingProvider:
    async def interpret_result(self, result: dict[str, object]) -> list[EvidenceStatement]:
        del result
        raise LLMProviderError("Test")


def test_classify_intent_falls_back_when_provider_raises() -> None:
    orchestrator = ConversationOrchestrator(provider=ClassifyFailingProvider())  # type: ignore[arg-type]

    intent = asyncio.run(orchestrator._classify_intent("计算苯/甲苯泡点"))

    assert intent == Intent.EQUILIBRIUM_CALCULATION


def test_respond_node_survives_provider_interpretation_failure() -> None:
    task = TaskManifest(
        equilibrium_type="VLE",
        calculation_type="bubble_point",
        components=[
            ComponentIdentity(component_id="benzene", name="Benzene", cas_number="71-43-2"),
            ComponentIdentity(component_id="toluene", name="Toluene", cas_number="108-88-3"),
        ],
        conditions=ThermodynamicConditions(pressure_kPa=101.325, liquid_composition=[0.5, 0.5]),
        model_name="Ideal/Raoult",
    )
    envelope = execute_task(task)
    graph = BoundedAgentGraph(InterpretFailingProvider(), DEFAULT_TOOL_REGISTRY)  # type: ignore[arg-type]

    result = asyncio.run(graph._respond({"envelope": envelope}))

    statements = result["statements"]
    assert len(statements) == 1
    assert statements[0].category == "Warning"
    assert "确定性引擎" in statements[0].text
