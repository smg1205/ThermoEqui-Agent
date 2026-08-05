"""Single orchestrator with deterministic parsing and tool invocation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from uuid import uuid4

from agent.graph_workflow import BoundedAgentGraph
from agent.providers import LLMProvider, LLMProviderError, LLMProviderOutputError
from agent.skill_integration import answer_with_skills
from agent.tools import DEFAULT_TOOL_REGISTRY, EngineeringToolRegistry
from schemas.domain import (
    AgentStep,
    CalculationEnvelope,
    ChatResponse,
    ComponentIdentity,
    EvidenceStatement,
    Intent,
    TaskManifest,
    ThermodynamicConditions,
)
from thermo_engine.errors import ThermoEquiError
from thermo_engine.identity import (
    has_chemical_role_evidence,
    is_electrolyte_identity,
    resolve_literal_components,
)
from thermo_engine.units import pressure_to_kpa, temperature_to_kelvin

COMPONENT_PATTERNS = (
    ("benzene", "Benzene", "71-43-2", ("苯", "benzene")),
    ("toluene", "Toluene", "108-88-3", ("甲苯", "toluene")),
    ("ethanol", "Ethanol", "64-17-5", ("乙醇", "ethanol")),
    ("acetone", "Acetone", "67-64-1", ("丙酮", "acetone")),
    ("methane", "Methane", "74-82-8", ("甲烷", "methane")),
    ("ethane", "Ethane", "74-84-0", ("乙烷", "ethane")),
    ("propane", "Propane", "74-98-6", ("丙烷", "propane")),
    ("nitrogen", "Nitrogen", "7727-37-9", ("氮气", "nitrogen", "n2")),
    ("water", "Water", "7732-18-5", ("水", "water")),
    ("methanol", "Methanol", "67-56-1", ("甲醇", "methanol")),
    ("carbon-dioxide", "Carbon dioxide", "124-38-9", ("二氧化碳", "carbon dioxide", "co2")),
)

_NUMBER_PATTERN = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_EXPLICIT_COMPOSITION_PATTERN = re.compile(
    (
        r"(?:feed[_\s-]*composition|feed\s+(?:mole|molar)\s+(?:composition|fractions?)|"
        r"mole\s+fractions?|molar\s+composition|进料(?:摩尔)?组成|摩尔组成|组成)"
        r"\s*(?:is|=|:|：|为)?\s*\[?\s*"
        rf"(?P<values>{_NUMBER_PATTERN}(?:\s*(?:,|，|、)\s*{_NUMBER_PATTERN})+)"
        r"\s*\]?"
    ),
    re.IGNORECASE,
)
_MODEL_COMPARISON_MARKERS = (
    "区别",
    "有什么不同",
    "比较",
    "对比",
    "差异",
    "difference",
    "compare",
    "comparison",
    "versus",
    "更适合",
    "更好",
    "哪个好",
    "why.*better",
)
_MODEL_TOPIC_MARKERS = (
    "模型",
    "后端",
    "状态方程",
    "活度系数",
    "backend",
    "model",
    "equation of state",
    "activity coefficient",
    "thermo",
    "phasepy",
    "clapeyron",
    "nrtl",
    "peng-robinson",
    "peng–robinson",
    "wilson",
    "uniquac",
    "raoult",
)
_PARAMETER_QUERY_MARKERS = (
    "参数",
    "parameter",
    "参数值",
    "parameter value",
    "二元参数",
    "binary parameter",
    "交互参数",
    "interaction parameter",
)
_PARAMETER_QUERY_TOPIC_MARKERS = (
    "是什么",
    "有哪些",
    "是多少",
    "是什么意思",
    "代表什么",
    "物理意义",
    "含义",
    "meaning",
    "what is",
    "what are",
    "how much",
    "参数",
    "parameter",
)


def _is_parameter_query_question(message: str) -> bool:
    """Check if a message is a pure parameter query (not a model comparison or calculation).

    Pure parameter queries are like: "NRTL的α参数是什么", "二元参数怎么获取"
    Model comparisons mentioning parameters are NOT parameter queries.
    Concept questions about parameter meaning are NOT parameter queries.
    """
    lower = message.casefold()
    has_param_marker = any(marker in lower for marker in _PARAMETER_QUERY_MARKERS)
    if not has_param_marker:
        return False
    if _is_model_comparison_question(message):
        return False
    comparison_markers = ("区别", "比较", "对比", "difference", "compare", "comparison", "versus")
    if any(marker in lower for marker in comparison_markers):
        return False
    calc_markers = ("计算", "算", "求", "calc", "compute", "simulate")
    if any(marker in lower for marker in calc_markers):
        return False
    # If asking about physical meaning/interpretation, it's a concept QA
    meaning_markers = ("物理意义", "是什么意思", "含义", "代表什么", "meaning", "怎么理解", "是什么")
    if any(marker in lower for marker in meaning_markers):
        return False
    # If asking about Antoine parameters, it's a data/concept query
    if "antoine" in lower:
        return False
    return True


_CONCEPT_QA_KEYWORDS = re.compile(
    r"(分析|解释|原理|概念|意义|为什么|是什么|介绍|阐述|理解|讲解|说明|讨论|"
    r"对比|比较|区别|联系|特点|特征|应用|用途|案例|例子|请问|如何|怎样|怎么|"
    r"判断|判断.*是否|是否会|if.*occurs|how.*to.*judge|why|what|concept|explain|describe)"
)


def _is_concept_question(message: str) -> bool:
    """Check if a message is a concept Q&A rather than a calculation request."""
    return _CONCEPT_QA_KEYWORDS.search(message) is not None


def _mentioned_components(message: str) -> list[ComponentIdentity]:
    if _is_concept_question(message):
        return []
    lower = message.casefold()
    candidates: list[tuple[int, int, int, ComponentIdentity]] = []
    for component_id, name, cas_number, aliases in COMPONENT_PATTERNS:
        component = ComponentIdentity(
            component_id=component_id,
            name=name,
            cas_number=cas_number,
            aliases=list(aliases),
        )
        for alias in aliases:
            escaped = re.escape(alias)
            pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])" if alias.isascii() else escaped
            for match in re.finditer(pattern, lower):
                candidates.append(
                    (
                        match.start(),
                        -len(alias),
                        match.end(),
                        component,
                    )
                )
    selected: list[tuple[int, int, ComponentIdentity]] = []
    seen_components: set[str] = set()
    for start, _, end, component in sorted(candidates, key=lambda item: (item[0], item[1])):
        if component.component_id in seen_components:
            continue
        if any(start < selected_end and end > selected_start for selected_start, selected_end, _ in selected):
            continue
        selected.append(
            (
                start,
                end,
                component,
            )
        )
        seen_components.add(component.component_id)
    return [component for _, _, component in selected]


def _explicit_composition(message: str) -> list[float] | None:
    match = _EXPLICIT_COMPOSITION_PATTERN.search(message)
    if match is None:
        return None
    return [float(value) for value in re.findall(_NUMBER_PATTERN, match.group("values"), flags=re.IGNORECASE)]


def _is_model_comparison_question(message: str) -> bool:
    lower = message.casefold()
    return any(marker in lower for marker in _MODEL_COMPARISON_MARKERS) and any(
        marker in lower for marker in _MODEL_TOPIC_MARKERS
    )


def _token_span(message: str, token: str) -> tuple[int, int] | None:
    normalized = token.casefold()
    escaped = re.escape(normalized)
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])" if normalized.isascii() else escaped
    match = re.search(pattern, message.casefold())
    return (match.start(), match.end()) if match is not None else None


def _component_role_is_ambiguous(message: str, start: int, end: int) -> bool:
    lower = message.casefold()
    if re.search(r"\b(?:compare|compared|versus|vs)\b|\bdifference\s+between\b", lower) or any(
        marker in lower for marker in ("比较", "对比", "区别")
    ):
        return True
    prefix = lower[max(0, start - 64) : start]
    suffix = lower[end : min(len(lower), end + 64)]
    prefix_is_ambiguous = re.search(
        r"(?:without|excluding|exclude|except|do\s+not\s+(?:include|add|use)|"
        r"not\s+(?:include|add|use)|with\s+no|no|instead\s+of|rather\s+than|"
        r"but\s+not|free\s+of|such\s+as|for\s+example|example\s+of|e\.g\.|"
        r"不含|排除|不要|不包括|例如|比如)"
        r"(?:[\s,]+(?:any|added?|adding|using|including|trace|of|the|component|compound)){0,4}"
        r"[\s,;:]*$",
        prefix,
    )
    suffix_is_ambiguous = re.match(
        r"(?:\s*-\s*free\b|\s+(?:should\s+be\s+)?(?:excluded|omitted|removed)\b|"
        r"\s*,?\s*(?:for\s+example|e\.g\.)\b)",
        suffix,
    )
    return prefix_is_ambiguous is not None or suffix_is_ambiguous is not None


def _requested_components(message: str) -> list[ComponentIdentity]:
    by_cas: dict[str, tuple[int, ComponentIdentity]] = {}
    for component in _mentioned_components(message):
        if component.cas_number is None:
            continue
        spans = [
            span
            for token in (component.name, *component.aliases, component.cas_number)
            if (span := _token_span(message, token)) is not None
        ]
        if spans:
            position, end = min(spans)
            if _component_role_is_ambiguous(message, position, end):
                raise LLMProviderOutputError("The component role is ambiguous and requires clarification.")
            if not has_chemical_role_evidence(message, position, end):
                continue
            by_cas[component.cas_number] = (position, component)
    for position, component in resolve_literal_components(message):
        literal = component.aliases[0] if component.aliases else component.name
        if _component_role_is_ambiguous(message, position, position + len(literal)):
            raise LLMProviderOutputError("The component role is ambiguous and requires clarification.")
        if component.cas_number is not None:
            by_cas.setdefault(component.cas_number, (position, component))
    return [component for _, component in sorted(by_cas.values(), key=lambda item: item[0])]


def _has_positive_scope_marker(message: str, markers: tuple[str, ...]) -> bool:
    lower = message.casefold()
    # Verbs that indicate the user wants to PERFORM an unsupported task
    _REQUEST_VERBS = re.compile(
        r"(?:请|帮我|请帮我|如何|怎么|怎样|想|需要|要求|能否|可否|"
        r"帮|请给|求|麻烦)"
        r"\s*"
        r"(?:设计|模拟|优化|计算|做|搞|进行|开展|搭建|建立|开发)"
    )
    # Verbs that can form a request when combined with topic keywords
    _REQUEST_VERBS_ALONE = re.compile(
        r"(?:设计|模拟|优化|搭建|建立|开发|计算)"
    )
    # Additional unsupported topic keywords that can appear in flexible word order
    _FLEXIBLE_EXCLUDED_TOPICS = (
        "精馏塔",
        "流程",
        "塔设计",
        "精馏设计",
    )
    for marker in markers:
        escaped = re.escape(marker.casefold())
        for match in re.finditer(escaped, lower):
            context_start = max(0, match.start() - 30)
            context_end = min(len(lower), match.end() + 10)
            context = lower[context_start:context_end]
            # Check if there's an active request verb nearby
            if _REQUEST_VERBS.search(context):
                return True
            # Also check if the marker appears as a standalone topic
            # (preceded by punctuation or start of string)
            before = lower[max(0, match.start() - 5):match.start()]
            if match.start() == 0 or re.search(r"[\s，,。.！!？?、；;：:（(]\s*$", before):
                return True
    # Check for flexible word-order combinations (e.g., "设计一个精馏塔")
    # Only when there's an explicit request prefix + verb pattern
    if _REQUEST_VERBS.search(lower):
        for topic in _FLEXIBLE_EXCLUDED_TOPICS:
            if topic in lower:
                return True
    return False


@dataclass
class ConversationState:
    task: TaskManifest | None = None
    run_ids: list[str] = field(default_factory=list)


class DeterministicProvider:
    """No-key provider for supported demonstrations and safe refusals."""

    async def classify_intent(self, message: str) -> Intent:
        lower = message.casefold()
        excluded_markers = (
            "氯化钠",
            "电解质",
            "nacl",
            "salt",
            "brine",
            "saltwater",
            "sodium chloride",
            "ionic mixture",
            "盐",
            "离子体系",
            "sle",
            "固液",
            "v lle",
            "vlle",
            "聚合物",
            "polymer",
            "水合物",
            "hydrate",
            "假组分",
            "pseudocomponent",
            "多晶",
            "polymorph",
            "复杂临界",
            "流程设计",
            "流程模拟",
            "流程优化",
            "flowsheet",
            "flowsheet design",
            "process flowsheet design",
            "design a flowsheet",
            "design the flowsheet",
            "design process",
            "liquefaction",
            "lng流程",
            "精馏塔设计",
            "full column design",
            "反应相平衡",
            "reactive equilibrium",
        )
        resolved_electrolyte = any(
            is_electrolyte_identity(component) for _, component in resolve_literal_components(message)
        )
        if _has_positive_scope_marker(lower, excluded_markers) or resolved_electrolyte:
            return Intent.UNSUPPORTED_TASK
        if any(word in lower for word in ("改为", "改成", "再算", "change", "rerun")):
            return Intent.TASK_CORRECTION
        if any(word in lower for word in ("解释结果", "结果含义", "interpret result")):
            return Intent.RESULT_INTERPRETATION
        if any(word in lower for word in ("敏感性", "sensitivity")):
            return Intent.SENSITIVITY_ANALYSIS
        if any(word in lower for word in ("工艺建议", "流程建议", "process recommendation")):
            return Intent.PROCESS_RECOMMENDATION
        if any(word in lower for word in ("查询数据", "获取数据", "查数据", "database", "data query")):
            return Intent.DATA_QUERY
        if any(word in lower for word in ("参数", "parameter")):
            if _is_parameter_query_question(message):
                return Intent.PARAMETER_QUERY
            if _is_model_comparison_question(message):
                return Intent.MODEL_SELECTION_QA
            if any(word in lower for word in ("区别", "选择", "difference", "select", "用什么")):
                return Intent.MODEL_SELECTION_QA
        if _is_model_comparison_question(message):
            return Intent.MODEL_SELECTION_QA
        if any(word in lower for word in ("区别", "选择", "difference", "select", "用什么")):
            return Intent.MODEL_SELECTION_QA
        if _is_active_calculation_request(message):
            return Intent.EQUILIBRIUM_CALCULATION
        return Intent.CONCEPT_QA


_CALCULATION_REQUEST_VERBS = ("计算", "算", "求", "calc", "compute", "simulate", "flash", "求算", "算出", "推算")
_NON_REQUEST_CALCULATION_PREFIXES = (
    "模型计算", "方程计算", "经计算", "通过计算", "由计算", "用计算",
    "计算得到", "计算得出", "计算结果", "计算显示", "计算表明",
    "经过计算", "理论计算", "模拟计算",
)


def _is_active_calculation_request(message: str) -> bool:
    """Detect active calculation requests vs passive descriptions of calculations.

    Active: "计算苯-甲苯气液平衡", "帮我求算", "calc the VLE"
    Passive: "模型计算得到", "经计算表明", "计算结果显示"
    """
    lower = message.casefold()
    for prefix in _NON_REQUEST_CALCULATION_PREFIXES:
        if prefix.casefold() in lower:
            return False
    if any(verb.casefold() in lower for verb in _CALCULATION_REQUEST_VERBS):
        return True
    return False

    async def formulate_task(self, message: str, previous: TaskManifest | None = None) -> TaskManifest | None:
        lower = message.casefold()
        component_list = _requested_components(message)
        pressure, pressure_assumption = self._pressure(message)
        temperature = self._temperature(message)
        if previous and any(word in lower for word in ("改为", "改成", "再算", "change", "rerun")):
            conditions = previous.conditions.model_copy(
                update={
                    **({"pressure_kPa": pressure} if pressure is not None else {}),
                    **({"temperature_K": temperature} if temperature is not None else {}),
                }
            )
            assumptions = [*previous.assumptions]
            if pressure_assumption and pressure_assumption not in assumptions:
                assumptions.append(pressure_assumption)
            return previous.model_copy(
                update={
                    "task_id": str(uuid4()),
                    "conditions": conditions,
                    "assumptions": assumptions,
                    "original_question": message,
                }
            )
        if not component_list:
            return None
        calculation_type = self._calculation_type(lower)
        equilibrium_type = "FLASH" if calculation_type == "tp_flash" else "LLE" if calculation_type == "lle" else "VLE"
        assumptions = [pressure_assumption] if pressure_assumption else []
        conditions = ThermodynamicConditions(temperature_K=temperature, pressure_kPa=pressure)
        return TaskManifest(
            equilibrium_type=equilibrium_type,
            calculation_type=calculation_type,
            components=component_list,
            conditions=conditions,
            requested_outputs=["chart", "table", "validation", "json", "csv"],
            assumptions=assumptions,
            model_name=(
                "Ideal/Raoult"
                if calculation_type != "lle" and {c.component_id for c in component_list} == {"benzene", "toluene"}
                else None
            ),
            original_question=message,
        )

    async def answer_with_evidence(self, message: str, strict: bool = False) -> list[EvidenceStatement]:
        if "nrtl" in message.casefold() and ("peng" in message.casefold() or "pr" in message.casefold()):
            text = (
                "NRTL 是液相活度系数模型，适合低到中压下的非理想液相 VLE/LLE，通常需要有来源的二元交互参数；"
                "Peng–Robinson 是立方状态方程，直接描述相逸度，常用于中高压烃类 VLE/Flash。两者不能仅凭名称互换。"
            )
        else:
            text = "当前离线知识库可解释模型与验证原则；数值问题必须交给确定性热力学工具。"
        return [EvidenceStatement(category="Knowledge", text=text)]

    async def select_tool(
        self,
        message: str,
        task: TaskManifest,
        available_tools: list[dict[str, str]],
    ) -> str:
        del message, task
        allowed_names = {tool["name"] for tool in available_tools}
        if "phase_equilibrium" not in allowed_names:
            raise LLMProviderOutputError("No deterministic phase-equilibrium tool is available.")
        return "phase_equilibrium"

    async def interpret_result(self, result: dict[str, object]) -> list[EvidenceStatement]:
        status = result.get("validation_status", "unknown")
        return [
            EvidenceStatement(
                category="Calculation",
                text=f"数值来自确定性热力学后端；物理验证状态为 {status}。",
            )
        ]

    @staticmethod
    def _pressure(message: str) -> tuple[float | None, str | None]:
        if "常压" in message or "atmospheric" in message.casefold():
            return 101.325, "“常压”规范化为 101.325 kPa。"
        match = re.search(r"(\d+(?:\.\d+)?)\s*(kpa|mpa|bar|atm)", message, re.IGNORECASE)
        if not match:
            return None, None
        unit = match.group(2).casefold()
        if unit == "kpa":
            return pressure_to_kpa(float(match.group(1)), "kPa"), None
        if unit == "mpa":
            return pressure_to_kpa(float(match.group(1)), "MPa"), None
        if unit == "bar":
            return pressure_to_kpa(float(match.group(1)), "bar"), None
        return pressure_to_kpa(float(match.group(1)), "atm"), None

    @staticmethod
    def _temperature(message: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(k|℃|°c|c)\b", message, re.IGNORECASE)
        if not match:
            return None
        if match.group(2).casefold() == "k":
            return temperature_to_kelvin(float(match.group(1)), "K")
        return temperature_to_kelvin(float(match.group(1)), "C")

    @staticmethod
    def _calculation_type(lower: str) -> str:
        if "lle" in lower or "液液" in lower or "liquid-liquid" in lower:
            return "lle"
        if "p-x-y" in lower or "pxy" in lower or "等温" in lower:
            return "isothermal_vle"
        if "flash" in lower:
            return "tp_flash"
        if "泡点" in lower or "bubble" in lower:
            return "bubble_point"
        if "露点" in lower or "dew" in lower:
            return "dew_point"
        if "共沸" in lower or "azeotrope" in lower:
            return "azeotrope"
        return "isobaric_vle"
                                                   # 改动7.30
def _build_calculation_summary(           
    envelope: CalculationEnvelope,
    components: list[ComponentIdentity],
) -> str:
    """从计算结果构造可读摘要，替代硬编码的"计算完成"。"""
    result = envelope.result
    comps = " / ".join(c.name for c in components)
    parts: list[str] = [f"模型：{result.model_name}"]


    if result.temperature_K is not None:
        parts.append(f"T={result.temperature_K:.2f} K")
    if result.pressure_kPa is not None:
        parts.append(f"P={result.pressure_kPa:.4f} kPa")

    if result.calculation_type == "tp_flash" and result.phases:
        for p in result.phases:
            if p.fraction < 1e-10:
                continue
            c_str = ", ".join(f"{x:.4f}" for x in p.composition)
            parts.append(f"{p.phase}相({p.fraction*100:.1f}%)：({c_str})")
        if result.vapor_fraction is not None:
            parts.append(f"汽化分率 β={result.vapor_fraction:.4f}")
    elif result.points:
        parts.append(f"数据点：{len(result.points)} 个")

    parts.append(f"验证：{envelope.validation.overall_status}")

    if result.warnings:
        first = result.warnings[0]
        parts.append(f"⚠ {first[:80]}{'…' if len(first) > 80 else ''}")

    return "计算完成。\n" + "\n".join(parts)
class ConversationOrchestrator:
    def __init__(
        self,
        provider: LLMProvider | None = None,
        tools: EngineeringToolRegistry | None = None,
    ) -> None:
        self.provider = provider or DeterministicProvider()
        self.tools = tools or DEFAULT_TOOL_REGISTRY
        self.graph = BoundedAgentGraph(self.provider, self.tools)
        self.states: dict[str, ConversationState] = {}

    async def parse(self, message: str, conversation_id: str | None = None) -> tuple[Intent, TaskManifest | None]:
        intent = await self._classify_intent(message)
        state = self.states.get(conversation_id or "")
        task = await self.provider.formulate_task(message, state.task if state else None)
        if task is not None:
            task = self._prepare_task(
                message,
                task,
                previous_task=state.task if state is not None and intent == Intent.TASK_CORRECTION else None,
            )
        return intent, task

    async def chat(self, message: str, conversation_id: str | None = None) -> ChatResponse:
        conversation_id = conversation_id or str(uuid4())
        state = self.states.setdefault(conversation_id, ConversationState())
        intent = await self._classify_intent(message)
        if intent == Intent.UNSUPPORTED_TASK:
            return ChatResponse(
                conversation_id=conversation_id,
                intent=intent,
                answer=(
                    "当前版本不支持电解质、聚合物、水合物、假组分、多晶型、SLE、VLLE、"
                    "反应相平衡或完整精馏塔设计；未执行不适用的普通分子模型。"
                ),
                statements=[EvidenceStatement(category="Warning", text="任务超出 0.1 支持边界。")],
            )
        if intent == Intent.SENSITIVITY_ANALYSIS:
            return ChatResponse(
                conversation_id=conversation_id,
                intent=intent,
                answer="已识别敏感性分析意图；该自动工作流列入 Phase 4，当前未执行数值扫描。",
                statements=[EvidenceStatement(category="Warning", text="请逐个修改条件并保留独立运行进行比较。")],
            )
        if intent in {
            Intent.CONCEPT_QA,
            Intent.MODEL_SELECTION_QA,
            Intent.PARAMETER_QUERY,
            Intent.DATA_QUERY,
            Intent.PROCESS_RECOMMENDATION,
            Intent.RESULT_INTERPRETATION,
        }:
            strict = intent in {Intent.PARAMETER_QUERY, Intent.DATA_QUERY}   # 新增
            try:
                statements = await self.provider.answer_with_evidence(message, strict=strict)
            except (LLMProviderError, LLMProviderOutputError):
                statements = []
            if not statements or statements[0].category == "Warning":
                skill_statements = answer_with_skills(message, intent)
                if skill_statements:
                    statements = skill_statements
            return ChatResponse(
                conversation_id=conversation_id,
                intent=intent,
                answer="\n".join(item.text for item in statements),
                statements=statements,
            )
        task = await self.provider.formulate_task(message, state.task)
        if task is None:
            return ChatResponse(
                conversation_id=conversation_id,
                intent=intent,
                answer="缺少可识别的组分，尚未执行计算。",
                statements=[EvidenceStatement(category="Warning", text="需要明确组分身份。")],
            )
        task = self._prepare_task(
            message,
            task,
            previous_task=state.task if intent == Intent.TASK_CORRECTION else None,
        )
        state.task = task
        required_missing = self._missing_conditions(task)
        if required_missing:
            return ChatResponse(
                conversation_id=conversation_id,
                intent=intent,
                answer=f"已生成结构化任务，但缺少 {', '.join(required_missing)}，尚未执行计算。",
                statements=[EvidenceStatement(category="Warning", text="缺失必要计算条件。")],
                task=task,
            )
        try:
            envelope, statements, execution_steps = await self.graph.run(message, task)
            validation = envelope.validation
            state.run_ids.append(envelope.result.run_id)
            return ChatResponse(
                conversation_id=conversation_id,
                intent=intent,
                answer=_build_calculation_summary(envelope, task.components),
                statements=statements,
                execution_steps=execution_steps,
                task=task,
                calculation=envelope,
            )
        except ThermoEquiError as error:
            return ChatResponse(
                conversation_id=conversation_id,
                intent=intent,
                answer=error.detail.message,
                statements=[EvidenceStatement(category="Warning", text=error.detail.recovery_action)],
                execution_steps=[
                    AgentStep(
                        phase="plan",
                        status="completed",
                        summary="A structured task manifest was created.",
                    ),
                    AgentStep(
                        phase="execute",
                        status="failed",
                        summary=error.detail.message,
                        tool_name="phase_equilibrium",
                    ),
                ],
                task=task,
            )

    async def _classify_intent(self, message: str) -> Intent:
        deterministic_intent = await DeterministicProvider().classify_intent(message)
        if deterministic_intent == Intent.UNSUPPORTED_TASK:
            return deterministic_intent
        try:
            provider_intent = await self.provider.classify_intent(message)
        except LLMProviderOutputError:
            return deterministic_intent
        if (
            provider_intent == Intent.EQUILIBRIUM_CALCULATION
            and deterministic_intent != Intent.EQUILIBRIUM_CALCULATION
            and not _mentioned_components(message)
        ):
            return deterministic_intent
        if (
            deterministic_intent == Intent.EQUILIBRIUM_CALCULATION
            and provider_intent != Intent.EQUILIBRIUM_CALCULATION
            and _mentioned_components(message)
        ):
            return deterministic_intent

        if (
            deterministic_intent == Intent.MODEL_SELECTION_QA
            and provider_intent == Intent.EQUILIBRIUM_CALCULATION
        ):
            return deterministic_intent
        if provider_intent == Intent.UNSUPPORTED_TASK and deterministic_intent in {
            Intent.CONCEPT_QA,
            Intent.MODEL_SELECTION_QA,
            Intent.PARAMETER_QUERY,
            Intent.DATA_QUERY,
            Intent.RESULT_INTERPRETATION,
            Intent.PROCESS_RECOMMENDATION,
        }:
            return deterministic_intent
        return provider_intent

    @classmethod
    def _prepare_task(
        cls,
        message: str,
        task: TaskManifest,
        *,
        previous_task: TaskManifest | None,
    ) -> TaskManifest:
        mentioned = _requested_components(message)
        if previous_task is not None:
            previous_components = previous_task.components
            if mentioned:
                mentioned_ids = {component.cas_number for component in mentioned}
                previous_ids = {component.cas_number for component in previous_components}
                if mentioned_ids == previous_ids:
                    expected_components = mentioned
                elif mentioned_ids.issubset(previous_ids):
                    expected_components = previous_components
                elif len(mentioned) == len(task.components):
                    expected_components = mentioned
                else:
                    raise LLMProviderOutputError(
                        "External provider returned components inconsistent with the correction."
                    )
            else:
                expected_components = previous_components
        elif mentioned:
            expected_components = mentioned
        else:
            raise LLMProviderOutputError("No component identity could be resolved independently from the user message.")
        grounded = cls._align_task_components(task, expected_components)
        explicit_composition = _explicit_composition(message)
        composition_field = {
            "tp_flash": "feed_composition",
            "bubble_point": "liquid_composition",
            "dew_point": "vapor_composition",
        }.get(grounded.calculation_type)
        if explicit_composition is not None and composition_field is not None:
            if len(explicit_composition) != len(grounded.components):
                raise ValueError("The explicit composition count must match the number of resolved components.")
            condition_data = grounded.conditions.model_dump()
            condition_data[composition_field] = explicit_composition
            grounded = grounded.model_copy(
                update={"conditions": ThermodynamicConditions.model_validate(condition_data)}
            )
        return grounded.model_copy(update={"original_question": message})

    @staticmethod
    def _align_task_components(
        task: TaskManifest,
        expected_components: list[ComponentIdentity],
    ) -> TaskManifest:
        expected_by_cas = {
            component.cas_number: component for component in expected_components if component.cas_number is not None
        }
        provider_order: list[str] = []
        for provider_component in task.components:
            provider_tokens = {
                provider_component.component_id.casefold(),
                provider_component.name.casefold(),
                *(alias.casefold() for alias in provider_component.aliases),
                *([provider_component.cas_number] if provider_component.cas_number else []),
            }
            matches = [
                cas_number
                for cas_number, canonical in expected_by_cas.items()
                if cas_number in provider_tokens
                or canonical.component_id.casefold() in provider_tokens
                or canonical.name.casefold() in provider_tokens
            ]
            if len(matches) != 1:
                raise LLMProviderOutputError(
                    "External provider returned components inconsistent with the user message."
                )
            if provider_component.cas_number is not None and provider_component.cas_number != matches[0]:
                raise LLMProviderOutputError("External provider returned a component name/CAS mismatch.")
            provider_order.append(matches[0])
        expected_order = [component.cas_number for component in expected_components if component.cas_number is not None]
        if (
            len(expected_by_cas) != len(expected_components)
            or len(provider_order) != len(expected_order)
            or len(set(provider_order)) != len(provider_order)
            or set(provider_order) != set(expected_order)
        ):
            raise LLMProviderOutputError("External provider returned components inconsistent with the user message.")
        permutation = [provider_order.index(cas_number) for cas_number in expected_order]
        condition_updates: dict[str, list[float]] = {}
        for field_name in (
            "feed_composition",
            "liquid_composition",
            "vapor_composition",
        ):
            values = getattr(task.conditions, field_name)
            if values is None:
                continue
            if len(values) != len(provider_order):
                raise LLMProviderOutputError("External provider returned a composition with the wrong component count.")
            condition_updates[field_name] = [values[index] for index in permutation]
        conditions = task.conditions.model_copy(update=condition_updates)
        return task.model_copy(
            update={
                "components": expected_components,
                "conditions": conditions,
            }
        )

    @staticmethod
    def _missing_conditions(task: TaskManifest) -> list[str]:
        missing: list[str] = []
        if (
            task.calculation_type in {"isobaric_vle", "bubble_point", "dew_point", "azeotrope"}
            and task.conditions.pressure_kPa is None
        ):
            missing.append("pressure_kPa")
        if task.calculation_type in {"isothermal_vle", "tp_flash"} and task.conditions.temperature_K is None:
            missing.append("temperature_K")
        if task.calculation_type == "tp_flash" and task.conditions.feed_composition is None:
            missing.append("feed_composition")
        if task.calculation_type == "bubble_point" and task.conditions.liquid_composition is None:
            missing.append("liquid_composition")
        if task.calculation_type == "dew_point" and task.conditions.vapor_composition is None:
            missing.append("vapor_composition")
        if task.calculation_type == "lle" and task.conditions.temperature_K is None:
            missing.append("temperature_K")
        return missing
