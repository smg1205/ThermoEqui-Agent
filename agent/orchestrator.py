"""Single orchestrator with deterministic parsing and tool invocation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from uuid import uuid4

from agent.graph_workflow import BoundedAgentGraph
from agent.memory_integration import retrieve_for_calculation, retrieve_for_concept_qa, save_turn
from agent.providers import LLMProvider, LLMProviderError, LLMProviderOutputError
from agent.skill_integration import answer_with_skill_payload
from agent.tools import DEFAULT_TOOL_REGISTRY, EngineeringToolRegistry
from schemas.domain import (
    AgentStep,
    CalculationEnvelope,
    ChatResponse,
    ComponentIdentity,
    EvidenceStatement,
    Intent,
    ParameterSet,
    TaskManifest,
    ThermodynamicConditions,
)
from thermo_engine.errors import ThermoEquiError
from thermo_engine.identity import (
    has_chemical_role_evidence,
    is_electrolyte_identity,
    resolve_literal_components,
)
from thermo_engine.parameter_store import load_production_parameter_sets
from thermo_engine.units import pressure_to_kpa, temperature_to_kelvin

_MODEL_ALLOWED_FOR_AUTO = {"Wilson", "NRTL", "UNIQUAC"}

#: (component_id, name, cas_number, aliases, smiles).  The SMILES are used by
#: SMILES-grounded backends such as PGSSI; they are the standard canonical
#: forms used by the PGSSI training dataset.
COMPONENT_PATTERNS = (
    ("benzene", "Benzene", "71-43-2", ("苯", "benzene"), "c1ccccc1"),
    ("toluene", "Toluene", "108-88-3", ("甲苯", "toluene"), "Cc1ccccc1"),
    ("ethanol", "Ethanol", "64-17-5", ("乙醇", "醇", "ethanol"), "CCO"),
    ("acetone", "Acetone", "67-64-1", ("丙酮", "acetone"), "CC(=O)C"),
    ("methane", "Methane", "74-82-8", ("甲烷", "methane"), "C"),
    ("ethane", "Ethane", "74-84-0", ("乙烷", "ethane"), "CC"),
    ("propane", "Propane", "74-98-6", ("丙烷", "propane"), "CCC"),
    ("nitrogen", "Nitrogen", "7727-37-9", ("氮气", "nitrogen", "n2"), "N#N"),
    ("water", "Water", "7732-18-5", ("水", "water"), "O"),
    ("methanol", "Methanol", "67-56-1", ("甲醇", "methanol"), "CO"),
    ("carbon-dioxide", "Carbon dioxide", "124-38-9", ("二氧化碳", "carbon dioxide", "co2"), "O=C=O"),
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
    for component_id, name, cas_number, aliases, smiles in COMPONENT_PATTERNS:
        component = ComponentIdentity(
            component_id=component_id,
            name=name,
            cas_number=cas_number,
            smiles=smiles,
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


_PARTIAL_COMPOSITION_PATTERN = re.compile(
    (
        r"(?:液相|气相|进料|feed|liquid|vapor)\s*组成为?\s*"
        r"(?P<value>\d+(?:\.\d*)?|\.\d+)"
    ),
    re.IGNORECASE,
)


def _partial_composition(message: str) -> list[float] | None:
    """Extract composition values when a user specifies e.g. 'x1=0.35' or '液相组成为0.35'."""
    explicit = _explicit_composition(message)
    if explicit is not None:
        return explicit
    matches = re.findall(r"组成为?\s*(\d+(?:\.\d*)?|\.\d+)", message, re.IGNORECASE)
    if matches:
        values = [float(v) for v in matches]
        if values and all(0 <= v <= 1 for v in values) and sum(values) <= 1 + 1e-8:
            return values
    x_pattern = re.findall(r"x[₀-₉0-9]*\s*[=＝]\s*(\d+(?:\.\d*)?|\.\d+)", message, re.IGNORECASE)
    if x_pattern:
        values = [float(v) for v in x_pattern]
        if values and all(0 <= v <= 1 for v in values):
            return values
    return None


def _extract_composition_values(message: str) -> list[float]:
    """Extract any 0-1 range numbers that look like composition values."""
    values = _partial_composition(message)
    if values is not None:
        return values
    numbers = re.findall(r"\d+(?:\.\d*)?|\.\d+", message)
    compo = []
    for n in numbers:
        v = float(n)
        if 0.0 <= v <= 1.0 and v not in (0.0, 1.0):
            compo.append(v)
    return compo


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
    _REQUEST_VERBS_ALONE = re.compile(r"(?:设计|模拟|优化|搭建|建立|开发|计算)")
    # Additional unsupported topic keywords that can appear in flexible word order
    # 注意："流程"不再进入此列表，因为 FLOW_DESIGN_QA 现在是支持的意图。
    # 此处仅保留 v0.1 明确排除的多单元/超范围设计词。
    _FLEXIBLE_EXCLUDED_TOPICS: tuple[str, ...] = ()
    for marker in markers:
        escaped = re.escape(marker.casefold())
        for match in re.finditer(escaped, lower):
            context_start = max(0, match.start() - 30)
            context_end = min(len(lower), match.end() + 10)
            context = lower[context_start:context_end]
            # Check if there's an active request verb nearby
            if _REQUEST_VERBS.search(context):
                return True
            # Chinese "计算聚合物" / "模拟电解质" style: an action verb directly
            # adjacent to an excluded topic is an active request.
            verb_prefix = lower[max(0, match.start() - 8) : match.start()]
            if re.search(r"(?:设计|模拟|优化|搭建|建立|开发|计算|做|搞|进行)\s*$", verb_prefix):
                return True
            # Also check if the marker appears as a standalone topic
            # (preceded by punctuation or start of string)
            before = lower[max(0, match.start() - 5) : match.start()]
            after = lower[match.end() : min(len(lower), match.end() + 24)]
            negated = (
                re.search(
                    r"(?:without|excluding|exclude|free\s+of|with\s+no|no\s+)\s*$",
                    lower[max(0, match.start() - 40) : match.start()],
                )
                is not None
                or re.match(r"(?:[a-z]+)?\s*(?:-free\b|free\b)", after) is not None
            )
            if not negated and (match.start() == 0 or re.search(r"[\s，,。.！!？?、；;：:（(]\s*$", before)):
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
    last_envelope: CalculationEnvelope | None = None


_CALCULATION_REQUEST_VERBS = (
    "计算",
    "算",
    "求",
    "判断",
    "搜索",
    "calc",
    "compute",
    "simulate",
    "classify",
    "search",
    "flash",
    "求算",
    "算出",
    "推算",
)
_NON_REQUEST_CALCULATION_PREFIXES = (
    "经计算",
    "通过计算",
    "由计算",
    "计算得到",
    "计算得出",
    "计算结果",
    "计算显示",
    "计算表明",
    "经过计算",
    "理论计算",
    "模拟计算",
)
# Passive calculation prefixes that indicate reporting rather than requesting
_NON_REQUEST_PASSIVE_PATTERNS = (
    re.compile(r"模型(?:计算|方程计算).*(?:得到|得出|结果|显示|表明|给出)"),
    re.compile(r"方程计算.*(?:得到|得出|结果|显示|表明)"),
    re.compile(r"用(?:计算|模型|方程)(?:得到|得出|结果|显示|表明)"),
)


_CONCEPT_QUESTION_WORDS_IN_CALC = (
    "多少",
    "为什么",
    "怎么",
    "怎样",
    "为何",
    "区别",
    "差异",
    "对比",
    "比较",
    "不同",
    "不一样",
    "原因",
    "解释",
    "分析",
    "合理",
    "对不对",
    "正确",
    "不对",
)

#: Property keywords whose value can only come from a deterministic backend.
#: A question asking for such a value must be routed to calculation, never to
#: an LLM free-form answer (the LLM may not fabricate numbers).
_GAMMA_INFINITY_KEYWORDS = (
    "无限稀释",
    "无限稀",
    "γ∞",
    "γinf",
    "gamma infinity",
    "gamma-infinity",
    "gamma_inf",
    "infinite dilution",
)

#: Thermo keywords used to gate the "how much is X" question form.
_THERMO_TOPIC_KEYWORDS = (
    "活度系数",
    "活度",
    "相平衡",
    "气液",
    "液液",
    "泡点",
    "露点",
    "共沸",
    "flash",
    "vle",
    "lle",
    "γ∞",
    "gamma infinity",
    "无限稀释",
)

#: Question forms that ask for a numeric value ("how much is X").
_NUMERIC_QUESTION_WORDS = ("是多少", "多少", "数值", "值是多少", "为多少")


_CALCULATION_SEARCH_VERBS = ("搜索", "查找")


def _is_active_calculation_request(message: str) -> bool:
    """Detect active calculation requests vs passive descriptions or judgment questions.

    Active: "计算苯-甲苯气液平衡", "帮我求算", "calc the VLE",
            "乙醇在水中的无限稀释活度系数是多少" (gamma-infinity value questions
            and thermo "how much" questions are routed to deterministic calculation)
    Passive: "模型计算得到", "经计算表明", "计算结果显示"
    Judgment: "可以用拉乌尔定律计算吗", "该体系能用NRTL计算吗"
    """
    lower = message.casefold()
    # Judgment questions: "...可以/能/应该/是否 ...计算吗/适用吗/可行吗" → not a request to run
    if re.search(
        r"(?:可以|能|应该|是否|适不适合|合不合适|能不能|可不可以).*(?:计算|适用|可行|使用|采用).*(?:吗|呢|？|\?)", lower
    ):
        return False
    # If the message asks whether a MODEL is suitable/applicable → concept/model-selection, not calculation
    if re.search(r"(?:可以用|能用|适用|合适|应该选|应该用|选什么).*(?:定律|模型|方程|方法)", lower) and (
        "吗" in lower or "呢" in lower or "?" in lower or "？" in lower
    ):
        return False
    # If the message is a concept question about existing results (contains concept-question words),
    # treat it as concept/interpretation, not a request to run a calculation.
    if any(word in lower for word in _CONCEPT_QUESTION_WORDS_IN_CALC):
        return False
    for prefix in _NON_REQUEST_CALCULATION_PREFIXES:
        if prefix.casefold() in lower:
            return False
    if any(keyword in lower for keyword in _GAMMA_INFINITY_KEYWORDS):
        return True
    for pattern in _NON_REQUEST_PASSIVE_PATTERNS:
        if pattern.search(message):
            return False
    if any(verb.casefold() in lower for verb in _CALCULATION_REQUEST_VERBS):
        return True
    if any(verb.casefold() in lower for verb in _CALCULATION_SEARCH_VERBS):
        return True
    if any(word in lower for word in _NUMERIC_QUESTION_WORDS) and any(
        topic.casefold() in lower for topic in _THERMO_TOPIC_KEYWORDS
    ):
        return True
    return False


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
            "liquefaction",
            "lng流程",
            "反应相平衡",
            "reactive equilibrium",
        )
        resolved_electrolyte = any(
            is_electrolyte_identity(component) for _, component in resolve_literal_components(message)
        )
        if _has_positive_scope_marker(lower, excluded_markers) or resolved_electrolyte:
            return Intent.UNSUPPORTED_TASK
        # 第二层防护：即使 _has_positive_scope_marker 放过，若 excluded_markers（例如"聚合物""SLE""电解质"）
        # 与请求动词（帮我/请/设计/模拟 等）同时出现，仍视为超范围请求。
        _REQUEST_VERBS_FOR_EXCLUSION = re.compile(
            r"(?:请|帮我|请帮我|如何|怎么|怎样|想|需要|帮|求|麻烦|是否|能不能|能否|可否)"
        )
        _DESIGN_ACTION_FOR_EXCLUSION = re.compile(r"(?:设计|模拟|优化|搭建|建立|开发|计算)")
        if _REQUEST_VERBS_FOR_EXCLUSION.search(lower) or _DESIGN_ACTION_FOR_EXCLUSION.search(lower):
            _DOMAIN_EXCLUDE_TOPICS = (
                "电解质",
                "离子体系",
                "nacl",
                "salt",
                "brine",
                "sodium chloride",
                "saltwater",
                "ionic mixture",
                "sle",
                "固液",
                "vlle",
                "v lle",
                "聚合物",
                "polymer",
                "水合物",
                "hydrate",
                "假组分",
                "pseudocomponent",
                "多晶",
                "polymorph",
                "反应相平衡",
                "reactive equilibrium",
                "liquefaction",
                "液化",
            )
            if any(topic in lower for topic in _DOMAIN_EXCLUDE_TOPICS):
                return Intent.UNSUPPORTED_TASK
        _TASK_CORRECTION_STRONG = (
            "改为",
            "改成",
            "换成",
            "再算",
            "change",
            "rerun",
            "沿用",
            "同样条件",
            "一样条件",
            "以上条件",
            "上面条件",
            "前面的条件",
            "之前的条件",
            "用同样的",
            "按之前的",
            "照之前的",
            "仍用",
            "仍然用",
            "保持",
            "同样的条件",
            "一样的条件",
        )
        _TASK_CORRECTION_WEAK = (
            "同样",
            "一样",
            "刚才",
            "前面的",
            "之前的",
            "上次的",
            "上面的",
            "刚才我们",
            "刚才我",
            "之前我们",
            "之前我",
        )
        if any(word in lower for word in _TASK_CORRECTION_STRONG):
            return Intent.TASK_CORRECTION
        weak_hit = any(word in lower for word in _TASK_CORRECTION_WEAK)
        _CONCEPT_COMPARE_WORDS = (
            "多少",
            "为什么",
            "对比",
            "比较",
            "差值",
            "差了",
            "差异",
            "原因",
            "合理",
            "趋势",
            "区别",
            "联系",
            "特点",
            "特征",
            "分析",
            "解释",
            "原理",
            "影响",
            "结论",
            "对不对",
            "正确",
            "不对",
            "不一样",
            "相同吗",
            "一致吗",
        )
        concept_hit = any(word in lower for word in _CONCEPT_COMPARE_WORDS)
        if weak_hit and not concept_hit:
            return Intent.TASK_CORRECTION
        if any(word in lower for word in ("解释结果", "结果含义", "interpret result")):
            return Intent.RESULT_INTERPRETATION
        if any(word in lower for word in ("敏感性", "sensitivity")):
            return Intent.SENSITIVITY_ANALYSIS
        if any(word in lower for word in ("工艺建议", "流程建议", "process recommendation")):
            return Intent.PROCESS_RECOMMENDATION
        if any(
            phrase in lower
            for phrase in (
                "设计流程",
                "流程设计",
                "设计一条",
                "分离流程",
                "回收流程",
                "提纯流程",
                "工艺流程",
                "design flow",
                "design process",
                "design a flow",
                "需要几个塔",
                "几塔流程",
            )
        ):
            return Intent.FLOW_DESIGN_QA
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

    async def formulate_task(self, message: str, previous: TaskManifest | None = None) -> TaskManifest | None:
        lower = message.casefold()
        component_list = _requested_components(message)
        pressure, pressure_assumption = self._pressure(message)
        temperature = self._temperature(message)
        temperature_span = self._temperature_span(message)
        _CORRECTION_OR_CONTINUATION_STRONG = (
            "改为",
            "改成",
            "换成",
            "再算",
            "change",
            "rerun",
            "沿用",
            "同样条件",
            "一样条件",
            "以上条件",
            "上面条件",
            "前面的条件",
            "之前的条件",
            "用同样的",
            "按之前的",
            "照之前的",
            "仍用",
            "仍然用",
            "保持",
            "同样的条件",
            "一样的条件",
        )
        _CORRECTION_OR_CONTINUATION_WEAK = (
            "同样",
            "一样",
            "刚才",
            "前面的",
            "之前的",
            "上次的",
            "上面的",
            "刚才我们",
            "刚才我",
            "之前我们",
            "之前我",
        )
        _CONCEPT_COMPARE_WORDS = (
            "多少",
            "为什么",
            "对比",
            "比较",
            "差值",
            "差了",
            "差异",
            "原因",
            "合理",
            "趋势",
            "区别",
            "联系",
            "特点",
            "特征",
            "分析",
            "解释",
            "原理",
            "影响",
            "结论",
            "对不对",
            "正确",
            "不对",
            "不一样",
            "相同吗",
            "一致吗",
        )
        has_strong_marker = any(word in lower for word in _CORRECTION_OR_CONTINUATION_STRONG)
        has_weak_marker = any(word in lower for word in _CORRECTION_OR_CONTINUATION_WEAK)
        has_concept = any(word in lower for word in _CONCEPT_COMPARE_WORDS)
        is_explicit_continuation = has_strong_marker or (has_weak_marker and not has_concept)
        composition_values = _extract_composition_values(message)
        is_implicit_continuation = (
            previous is not None
            and len(composition_values) >= 1
            and (pressure is not None or temperature is not None or component_list)
        )
        if previous and (is_explicit_continuation or is_implicit_continuation):
            updated_components = component_list or previous.components
            components_changed = {
                component.cas_number or component.component_id for component in updated_components
            } != {component.cas_number or component.component_id for component in previous.components}
            updates = {
                "task_id": str(uuid4()),
                "conditions": previous.conditions.model_copy(
                    update={
                        **({"pressure_kPa": pressure} if pressure is not None else {}),
                        **({"temperature_K": temperature} if temperature is not None else {}),
                    }
                ),
                "assumptions": [*previous.assumptions],
                "original_question": message,
                "model_name": None if components_changed else previous.model_name,
            }
            if pressure_assumption and pressure_assumption not in updates["assumptions"]:
                updates["assumptions"].append(pressure_assumption)
            if component_list:
                updates["components"] = component_list
            return previous.model_copy(update=updates)
        if not component_list:
            return None
        calculation_type = self._calculation_type(lower)
        equilibrium_type = "FLASH" if calculation_type in {"tp_flash", "phase_stability"} else "LLE"
        if calculation_type not in {"tp_flash", "phase_stability", "lle"}:
            equilibrium_type = "VLE"
        assumptions = [pressure_assumption] if pressure_assumption else []
        conditions = ThermodynamicConditions(
            temperature_K=temperature,
            temperature_span_K=temperature_span,
            pressure_kPa=pressure,
        )
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

    async def answer_with_evidence(
        self,
        message: str,
        strict: bool = False,
        grounded_numbers: set[str] | None = None,
        *,
        intent_label: str | None = None,
    ) -> list[EvidenceStatement]:
        del strict, grounded_numbers, intent_label
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
        # 排除 kPa/mpa/bar/atm 等压力单位里的 k/c 误匹配
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(k(?![pab])|℃|°c|c(?![mabd]))",
            message,
            re.IGNORECASE,
        )
        if not match:
            return None
        unit = match.group(2).casefold()
        if unit == "k":
            return temperature_to_kelvin(float(match.group(1)), "K")
        return temperature_to_kelvin(float(match.group(1)), "C")

    @staticmethod
    def _temperature_span(message: str) -> tuple[float, float] | None:
        """Extract a temperature range (e.g. '280到360K', '280-360 K', '20到80°C')."""
        # 数字 + 分隔词(到/至/-/~) + 数字 + 单位；排除 kPa 等压力单位的误匹配
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:到|至|－|—|-|~|～)\s*(\d+(?:\.\d+)?)\s*(k(?![pab])|℃|°c|c(?![mabd]))",
            message,
            re.IGNORECASE,
        )
        if not match:
            return None
        unit = match.group(3).casefold()
        if unit == "k":
            low = temperature_to_kelvin(float(match.group(1)), "K")
            high = temperature_to_kelvin(float(match.group(2)), "K")
        else:
            low = temperature_to_kelvin(float(match.group(1)), "C")
            high = temperature_to_kelvin(float(match.group(2)), "C")
        if low >= high:
            return None
        return (low, high)

    @staticmethod
    def _calculation_type(lower: str) -> str:
        if "相态" in lower or "phase classification" in lower or "phase state" in lower:
            return "phase_stability"
        if "lle" in lower or "液液" in lower or "liquid-liquid" in lower:
            return "lle"
        if (
            "无限稀释" in lower
            or "γ∞" in lower
            or "γinf" in lower
            or "gamma infinity" in lower
            or "gamma-infinity" in lower
            or "无限稀" in lower
            or "活度系数" in lower
            or "gamma_inf" in lower
        ):
            return "infinite_dilution_activity"
        if "p-x-y" in lower or "pxy" in lower or "等温" in lower:
            return "isothermal_vle"
        if "flash" in lower or "闪蒸" in lower:
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
            parts.append(f"{p.phase}相({p.fraction * 100:.1f}%)：({c_str})")
        if result.vapor_fraction is not None:
            parts.append(f"汽化分率 β={result.vapor_fraction:.4f}")
    elif result.points:
        parts.append(f"数据点：{len(result.points)} 个")
    elif result.gamma_infinity:
        parts.append(f"γ∞ 预测：{len(result.gamma_infinity)} 个方向")
        for point in result.gamma_infinity:
            solute = components[point.solute_index].name
            solvent = components[point.solvent_index].name
            parts.append(
                f"γ∞({solute}→{solvent}) = {point.gamma_infinity:.4f} "
                f"(ln γ∞ = {point.ln_gamma_infinity:.4f} @ {point.temperature_K:.2f} K)"
            )

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

    async def parse(
        self,
        message: str,
        conversation_id: str | None = None,
        parameter_sets: list[ParameterSet] | None = None,
    ) -> tuple[Intent, TaskManifest | None]:
        intent = await self._classify_intent(message)
        state = self.states.get(conversation_id or "")
        if intent == Intent.EQUILIBRIUM_CALCULATION and state is not None and state.task is not None:
            composition_values = _extract_composition_values(message)
            if composition_values and len(composition_values) >= 1:
                lower = message.casefold()
                _CONCEPT_COMPARE_WORDS = (
                    "多少",
                    "为什么",
                    "对比",
                    "比较",
                    "差值",
                    "差了",
                    "差异",
                    "原因",
                    "合理",
                    "趋势",
                    "区别",
                    "联系",
                    "特点",
                    "特征",
                    "分析",
                    "解释",
                    "原理",
                    "影响",
                    "结论",
                    "对不对",
                    "正确",
                    "不对",
                    "不一样",
                    "相同吗",
                    "一致吗",
                )
                has_concept = any(word in lower for word in _CONCEPT_COMPARE_WORDS)
                if has_concept:
                    pass
                else:
                    has_context_ref = any(
                        word in lower for word in ("刚才", "之前", "前面", "上次", "仍用", "仍然", "保持", "现在")
                    )
                    has_no_new_components = not _requested_components(message)
                    if has_context_ref or has_no_new_components:
                        intent = Intent.TASK_CORRECTION
        task = await self.provider.formulate_task(message, state.task if state else None)
        if task is not None:
            if parameter_sets:
                task = self._merge_parameter_sets(task, parameter_sets)
            task = self._prepare_task(
                message,
                task,
                previous_task=state.task if state is not None and intent == Intent.TASK_CORRECTION else None,
            )
        return intent, task

    async def chat(
        self,
        message: str,
        conversation_id: str | None = None,
        parameter_sets: list[ParameterSet] | None = None,
    ) -> ChatResponse:
        conversation_id = conversation_id or str(uuid4())
        state = self.states.setdefault(conversation_id, ConversationState())
        intent = await self._classify_intent(message)
        if intent == Intent.EQUILIBRIUM_CALCULATION and state.task is not None:
            composition_values = _extract_composition_values(message)
            if composition_values and len(composition_values) >= 1:
                lower = message.casefold()
                _CONCEPT_COMPARE_WORDS = (
                    "多少",
                    "为什么",
                    "对比",
                    "比较",
                    "差值",
                    "差了",
                    "差异",
                    "原因",
                    "合理",
                    "趋势",
                    "区别",
                    "联系",
                    "特点",
                    "特征",
                    "分析",
                    "解释",
                    "原理",
                    "影响",
                    "结论",
                )
                has_concept = any(word in lower for word in _CONCEPT_COMPARE_WORDS)
                if not has_concept:
                    has_context_ref = any(
                        word in lower for word in ("刚才", "之前", "前面", "上次", "仍用", "仍然", "保持", "现在")
                    )
                    has_no_new_components = not _requested_components(message)
                    if has_context_ref or has_no_new_components:
                        intent = Intent.TASK_CORRECTION
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
            Intent.FLOW_DESIGN_QA,
        }:
            # Retrieve conversation memory and inject as context prefix
            memory_prefix, grounded_numbers = retrieve_for_concept_qa(conversation_id, message)
            effective_message = f"{memory_prefix}{message}" if memory_prefix else message
            strict = intent in {Intent.PARAMETER_QUERY, Intent.DATA_QUERY}
            statements: list[EvidenceStatement] = []
            # For FLOW_DESIGN_QA, prefer the dedicated skill which produces
            # a well-typed FlowDesignDraft over generic LLM free text. The
            # structured draft is carried on ChatResponse.flow_design so the
            # frontend and downstream exporters do not need to parse prose.
            flow_design_payload = None
            if intent == Intent.FLOW_DESIGN_QA:
                skill_answer = answer_with_skill_payload(effective_message, intent)
                statements = skill_answer.statements
                flow_design_payload = skill_answer.flow_design
            if not statements:
                try:
                    statements = await self.provider.answer_with_evidence(
                        effective_message,
                        strict=strict,
                        grounded_numbers=grounded_numbers,
                        intent_label=intent.value,
                    )
                except (LLMProviderError, LLMProviderOutputError):
                    fallback_answer = answer_with_skill_payload(effective_message, intent)
                    statements = fallback_answer.statements
                    if fallback_answer.flow_design and flow_design_payload is None:
                        flow_design_payload = fallback_answer.flow_design
                    if not statements:
                        statements = [
                            EvidenceStatement(
                                category="Warning",
                                text="外部模型暂时不可用；请稍后重试或改用确定性计算接口。",
                            )
                        ]
            if not statements or statements[0].category == "Warning":
                fallback_answer = answer_with_skill_payload(effective_message, intent)
                if fallback_answer.statements:
                    statements = fallback_answer.statements
                    if fallback_answer.flow_design and flow_design_payload is None:
                        flow_design_payload = fallback_answer.flow_design
            answer_text = "\n".join(item.text for item in statements)
            # Save this turn to conversation memory
            save_turn(conversation_id, message, answer_text, intent)
            return ChatResponse(
                conversation_id=conversation_id,
                intent=intent,
                answer=answer_text,
                statements=statements,
                flow_design=flow_design_payload,
            )
        task = None
        try:
            task = await self.provider.formulate_task(message, state.task)
        except (LLMProviderError, LLMProviderOutputError, ValueError) as error:
            if state.task:
                try:
                    task = await DeterministicProvider().formulate_task(message, state.task)
                except Exception:
                    task = None
            if task is None:
                return ChatResponse(
                    conversation_id=conversation_id,
                    intent=intent,
                    answer=f"已识别计算意图，但未能构建结构化任务：{error}",
                    statements=[
                        EvidenceStatement(category="Warning", text="请补充明确的组分、条件或沿用之前的有效计算任务。")
                    ],
                )
        if task is None:
            return ChatResponse(
                conversation_id=conversation_id,
                intent=intent,
                answer="缺少可识别的组分，尚未执行计算。",
                statements=[EvidenceStatement(category="Warning", text="需要明确组分身份。")],
            )
        if parameter_sets:
            task = self._merge_parameter_sets(task, parameter_sets)
        try:
            task = self._prepare_task(
                message,
                task,
                previous_task=state.task if intent == Intent.TASK_CORRECTION else None,
            )
        except (LLMProviderOutputError, ValueError) as error:
            return ChatResponse(
                conversation_id=conversation_id,
                intent=intent,
                answer=f"已识别计算意图，但未能构建结构化任务：{error}",
                statements=[EvidenceStatement(category="Warning", text="请明确组分与条件后重试。")],
            )
        state.task = task
        # Auto-populate parameters from production YAML when task has no explicit parameter_sets
        auto_params = self._auto_lookup_parameters(task)
        if auto_params:
            task = self._merge_parameter_sets(task, auto_params)
            state.task = task
        # Try to infer missing temperature from previous calculation results
        if task.conditions.temperature_K is None:
            inferred_temp = self._infer_temperature_from_context(message, task, state.last_envelope)
            if inferred_temp is not None:
                task = task.model_copy(
                    update={"conditions": task.conditions.model_copy(update={"temperature_K": inferred_temp})}
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
            state.run_ids.append(envelope.result.run_id)
            state.last_envelope = envelope
            answer_text = _build_calculation_summary(envelope, task.components)
            # Append historical calculation reference if available
            calc_ref = retrieve_for_calculation(conversation_id, message)
            if calc_ref:
                answer_text += calc_ref
            # Save this turn to conversation memory
            component_names = [c.name for c in task.components]
            task_summary = f"{task.calculation_type}, {task.equilibrium_type}, {task.model_name or 'auto'}"
            save_turn(
                conversation_id,
                message,
                answer_text,
                intent,
                components=component_names,
                task_summary=task_summary,
            )
            return ChatResponse(
                conversation_id=conversation_id,
                intent=intent,
                answer=answer_text,
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
        except (LLMProviderError, LLMProviderOutputError):
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

        if deterministic_intent == Intent.MODEL_SELECTION_QA and provider_intent in {
            Intent.EQUILIBRIUM_CALCULATION,
            Intent.CONCEPT_QA,
        }:
            return deterministic_intent
        # Specialized intents require explicit trigger keywords; if the deterministic
        # classifier did not fire them and the LLM hallucinated one, defer to
        # deterministic for concept/compare/interpretation flows.
        _SPECIALIZED_INTENTS_REQUIRING_TRIGGER = {
            Intent.SENSITIVITY_ANALYSIS,
            Intent.PROCESS_RECOMMENDATION,
            Intent.RESULT_INTERPRETATION,
            Intent.DATA_QUERY,
            Intent.PARAMETER_QUERY,
            Intent.FLOW_DESIGN_QA,
        }
        if (
            provider_intent in _SPECIALIZED_INTENTS_REQUIRING_TRIGGER
            and deterministic_intent not in _SPECIALIZED_INTENTS_REQUIRING_TRIGGER
            and deterministic_intent != Intent.UNSUPPORTED_TASK
        ):
            return deterministic_intent
        # Conversely: if the deterministic classifier fired a specialized intent
        # (trigger keywords like 流程设计/敏感性/参数查询 were explicitly
        # detected in the message) while the LLM collapsed it into a generic
        # CONCEPT_QA or MODEL_SELECTION_QA, trust the keyword-based result so
        # the matching skill (e.g. ProcessFlowDesignSkill) takes priority.
        if deterministic_intent in _SPECIALIZED_INTENTS_REQUIRING_TRIGGER and provider_intent in {
            Intent.CONCEPT_QA,
            Intent.MODEL_SELECTION_QA,
        }:
            return deterministic_intent
        if provider_intent == Intent.UNSUPPORTED_TASK and deterministic_intent in {
            Intent.CONCEPT_QA,
            Intent.MODEL_SELECTION_QA,
            Intent.PARAMETER_QUERY,
            Intent.DATA_QUERY,
            Intent.RESULT_INTERPRETATION,
            Intent.PROCESS_RECOMMENDATION,
            Intent.FLOW_DESIGN_QA,
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
        if explicit_composition is None:
            partial_composition = _partial_composition(message)
            if partial_composition is not None and len(partial_composition) < len(grounded.components):
                n = len(grounded.components)
                if len(partial_composition) == 1 and n == 2:
                    explicit_composition = [partial_composition[0], 1.0 - partial_composition[0]]
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
    def _auto_lookup_parameters(
        task: TaskManifest,
    ) -> list[ParameterSet]:
        """Auto-populate parameters from production YAML when task has no explicit parameter_sets.

        Supports Wilson, NRTL, and UNIQUAC models for binary systems.
        Matches by component name (case-insensitive) in forward or reverse order.
        Returns a list of ParameterSet objects (empty if no match found).
        """
        model_name = task.model_name
        if not model_name or model_name not in _MODEL_ALLOWED_FOR_AUTO:
            return []
        if len(task.components) != 2:
            return []
        if task.parameters:
            return []
        component_names = [c.name.casefold() for c in task.components]
        try:
            all_production_sets = load_production_parameter_sets()
        except Exception:
            return []
        for param_set in all_production_sets:
            if param_set.model_name.casefold() != model_name.casefold():
                continue
            order_lower = [c.casefold() for c in param_set.component_order]
            if order_lower == component_names or list(reversed(order_lower)) == component_names:
                return [param_set]
        return []

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
                    f"External provider returned components inconsistent with the user message. provider_tokens={provider_tokens} expected_by_cas={expected_by_cas}"
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
    def _infer_temperature_from_context(
        message: str,
        task: TaskManifest,
        last_envelope: CalculationEnvelope | None,
    ) -> float | None:
        """Infer a missing temperature from the previous calculation result.

        When the user says e.g. "在x1=0.3的温度下进行闪蒸" without an explicit
        temperature value, look up the previously calculated temperature for
        that composition from the last envelope.
        """
        if last_envelope is None:
            return None
        result = last_envelope.result
        if result.temperature_K is None and not result.points:
            return None
        # If the message references a specific composition, try to match a point
        composition_values = _extract_composition_values(message)
        if composition_values:
            ref_comp = composition_values[0]
            # Look through equilibrium points for a matching liquid composition
            for point in result.points:
                if point.liquid_composition and len(point.liquid_composition) > 0:
                    if abs(point.liquid_composition[0] - ref_comp) < 1e-6:
                        return point.temperature_K
            # If no points matched but result has a single temperature, check
            # if the reference composition matches the task's liquid_composition
            if result.temperature_K is not None:
                task_liq = task.conditions.liquid_composition
                if task_liq and len(task_liq) > 0 and abs(task_liq[0] - ref_comp) < 1e-6:
                    return result.temperature_K
        # No composition reference, but the message clearly implies using
        # the previous result's temperature (e.g. "在该温度下", "在上述温度下")
        lower = message.casefold()
        if any(phrase in lower for phrase in ("该温度", "上述温度", "刚才的温度", "之前的温度", "那个温度")):
            if result.temperature_K is not None:
                return result.temperature_K
            if result.points:
                return result.points[0].temperature_K
        return None

    @staticmethod
    def _missing_conditions(task: TaskManifest) -> list[str]:
        missing: list[str] = []
        if task.calculation_type == "isobaric_vle" and task.conditions.pressure_kPa is None:
            missing.append("pressure_kPa")
        if task.calculation_type in {"bubble_point", "dew_point", "azeotrope"}:
            if task.conditions.temperature_K is None and task.conditions.pressure_kPa is None:
                missing.append("temperature_K or pressure_kPa")
        if task.calculation_type in {"isothermal_vle", "tp_flash"} and task.conditions.temperature_K is None:
            missing.append("temperature_K")
        if task.calculation_type == "phase_stability":
            if task.conditions.temperature_K is None:
                missing.append("temperature_K")
            if task.conditions.pressure_kPa is None:
                missing.append("pressure_kPa")
            if task.conditions.feed_composition is None:
                missing.append("feed_composition")
        if task.calculation_type == "tp_flash" and task.conditions.feed_composition is None:
            missing.append("feed_composition")
        if task.calculation_type == "bubble_point" and task.conditions.liquid_composition is None:
            missing.append("liquid_composition")
        if task.calculation_type == "dew_point" and task.conditions.vapor_composition is None:
            missing.append("vapor_composition")
        if task.calculation_type == "lle" and task.conditions.temperature_K is None:
            missing.append("temperature_K")
        return missing

    @staticmethod
    def _merge_parameter_sets(
        task: TaskManifest,
        parameter_sets: list[ParameterSet],
    ) -> TaskManifest:
        seen = {parameter_set.parameter_set_id for parameter_set in task.parameters}
        return task.model_copy(
            update={
                "parameters": [
                    *task.parameters,
                    *(parameter_set for parameter_set in parameter_sets if parameter_set.parameter_set_id not in seen),
                ]
            }
        )
