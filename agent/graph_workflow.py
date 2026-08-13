"""Bounded LangGraph workflow inspired by CAi_copilot's plan/execute agent shell."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from agent.executor import TaskExecution, validate_task_execution
from agent.providers import LLMProvider, LLMProviderError, LLMProviderOutputError
from agent.tools import EngineeringToolRegistry
from schemas.domain import AgentStep, CalculationEnvelope, EvidenceStatement, TaskManifest


class CalculationWorkflowState(TypedDict, total=False):
    message: str
    task: TaskManifest
    tool_name: str
    execution: TaskExecution
    envelope: CalculationEnvelope
    statements: list[EvidenceStatement]
    steps: Annotated[list[AgentStep], operator.add]


class BoundedAgentGraph:
    """Run one allowlisted deterministic tool through plan, execute, validate, respond nodes."""

    def __init__(self, provider: LLMProvider, tools: EngineeringToolRegistry) -> None:
        self.provider = provider
        self.tools = tools
        builder = StateGraph(CalculationWorkflowState)
        builder.add_node("plan", self._plan)
        builder.add_node("execute", self._execute)
        builder.add_node("validate", self._validate)
        builder.add_node("respond", self._respond)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "execute")
        builder.add_edge("execute", "validate")
        builder.add_edge("validate", "respond")
        builder.add_edge("respond", END)
        self._graph: Any = builder.compile()

    async def _plan(self, state: CalculationWorkflowState) -> CalculationWorkflowState:
        tool_name = await self.provider.select_tool(
            state["message"],
            state["task"],
            self.tools.catalog(),
        )
        return {
            "tool_name": tool_name,
            "steps": [
                AgentStep(
                    phase="plan",
                    status="completed",
                    summary="A structured task manifest was created and an allowlisted tool was selected.",
                )
            ],
        }

    def _execute(self, state: CalculationWorkflowState) -> CalculationWorkflowState:
        tool_name = state["tool_name"]
        execution = self.tools.execute(tool_name, state["task"])
        return {
            "execution": execution,
            "steps": [
                AgentStep(
                    phase="execute",
                    status="completed",
                    summary="The registered deterministic phase-equilibrium tool completed.",
                    tool_name=tool_name,
                )
            ],
        }

    @staticmethod
    def _validate(state: CalculationWorkflowState) -> CalculationWorkflowState:
        envelope = validate_task_execution(state["execution"])
        validation = envelope.validation
        return {
            "envelope": envelope,
            "steps": [
                AgentStep(
                    phase="validate",
                    status="completed" if validation.overall_status != "failed" else "failed",
                    summary=f"Independent physical validation status: {validation.overall_status}.",
                    tool_name=state["tool_name"],
                )
            ],
        }

    async def _respond(self, state: CalculationWorkflowState) -> CalculationWorkflowState:
        envelope = state["envelope"]
        try:
            statements = await self.provider.interpret_result(
                {
                    "result": envelope.result.model_dump(mode="json"),
                    "validation": envelope.validation.model_dump(mode="json"),
                }
            )
        except (LLMProviderError, LLMProviderOutputError):
            statements = [
                EvidenceStatement(
                    category="Warning",
                    text="外部模型解读暂时不可用；结果来自确定性引擎并已通过独立验证。",
                )
            ]
        return {
            "statements": statements,
            "steps": [
                AgentStep(
                    phase="respond",
                    status="completed",
                    summary="The provider interpreted only the grounded calculation and validation payload.",
                )
            ],
        }

    async def run(
        self,
        message: str,
        task: TaskManifest,
    ) -> tuple[CalculationEnvelope, list[EvidenceStatement], list[AgentStep]]:
        state = cast(
            CalculationWorkflowState,
            await self._graph.ainvoke(
                CalculationWorkflowState(
                    message=message,
                    task=task,
                    steps=[],
                )
            ),
        )
        return state["envelope"], state["statements"], state["steps"]
