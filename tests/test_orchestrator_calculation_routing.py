"""Routing tests for calculation-like phrasing that the deterministic layer must catch."""

from __future__ import annotations

import asyncio

from agent.orchestrator import ConversationOrchestrator, DeterministicProvider
from schemas.domain import Intent


def test_azeotrope_search_is_routed_as_calculation() -> None:
    message = "苯/甲苯在 101.325 kPa 搜索共沸物"

    intent = asyncio.run(DeterministicProvider().classify_intent(message))

    assert intent == Intent.EQUILIBRIUM_CALCULATION


def test_azeotrope_search_survives_orchestrator_intent_reconciliation() -> None:
    message = "苯/甲苯在 101.325 kPa 搜索共沸物"
    orchestrator = ConversationOrchestrator(provider=DeterministicProvider())

    intent = asyncio.run(orchestrator._classify_intent(message))

    assert intent == Intent.EQUILIBRIUM_CALCULATION


def test_parameter_search_stays_parameter_query() -> None:
    message = "搜索 NRTL 的 alpha 参数"

    intent = asyncio.run(DeterministicProvider().classify_intent(message))

    assert intent == Intent.PARAMETER_QUERY
