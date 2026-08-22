"""Central integration module for RAG, knowledge graph, and skill-based answering.
Provides lazy initialization and a clean interface for the orchestrator.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from schemas.domain import EvidenceStatement, FlowDesignDraft, Intent

logger = logging.getLogger(__name__)

# Lazy init state
_rag_initialized = False
_kg_initialized = False
_skill_registry = None
_init_lock = threading.Lock()


def _ensure_rag_index() -> None:
    global _rag_initialized
    if _rag_initialized:
        return
    with _init_lock:
        if _rag_initialized:
            return
        try:
            from rag.retriever import KnowledgeRetriever

            retriever = KnowledgeRetriever()
            retriever.build_index()
            logger.info("RAG index built: %d vectors", retriever.vector_store.size)
        except Exception as exc:
            logger.warning("RAG index build skipped: %s", exc)
        _rag_initialized = True


def _ensure_knowledge_graph() -> None:
    global _kg_initialized
    if _kg_initialized:
        return
    with _init_lock:
        if _kg_initialized:
            return
        try:
            from knowledge_graph.kg_builder import build_graph_from_kb

            graph = build_graph_from_kb()
            logger.info("KG built: %d nodes, %d rels", len(graph.nodes), len(graph.relationships))
        except Exception as exc:
            logger.warning("KG build skipped: %s", exc)
        _kg_initialized = True


def _init_in_background() -> None:
    def _run():
        _ensure_rag_index()
        _ensure_knowledge_graph()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logger.info("RAG/KG background init started.")


def initialize_all() -> None:
    _init_in_background()


def get_skill_registry():
    global _skill_registry
    if _skill_registry is not None:
        return _skill_registry
    from skills.knowledge_base.skill_registry import DEFAULT_SKILL_REGISTRY

    _skill_registry = DEFAULT_SKILL_REGISTRY
    return _skill_registry


_KB_NO_ANSWER_MARKERS = (
    "未包含",
    "未找到",
    "未提供",
    "无法提供",
    "无法回答",
    "不包含",
    "未识别",
    "未识别到",
    "无法识别",
    "not contain",
    "not found",
    "cannot answer",
    "unable to answer",
    "no relevant",
    "no information",
    "not related",
    "unrelated",
)

_THERMO_KEYWORDS = (
    "相平衡",
    "气液",
    "液液",
    "vle",
    "分离流程",
    "精馏塔",
    "精馏",
    "流程设计",
    "工艺流",
    "提纯流程",
    "分离方法",
    "需要几个塔",
    "单元操作",
    "进料预热器",
    "塔顶冷凝",
    "塔顶冷凝器",
    "塔釜再沸",
    "乙醇",
    "甲醇",
    "水",
    "苯",
    "甲苯",
    "丙酮",
    "乙苯",
    "异丙醇",
    "乙酸",
    "泡点",
    "露点",
    "vle",
    "lle",
    "flash",
    "泡点",
    "露点",
    "共沸",
    "热力学",
    "活度",
    "逸度",
    "相图",
    " Antoine",
    "antoine",
    "nrtl",
    "wilson",
    "uniquac",
    "peng",
    "raoult",
    "ideal",
    "状态方程",
    "活度系数",
    "二元",
    "组分",
    "进料",
    "phase equilibrium",
    "vapor-liquid",
    "liquid-liquid",
    "bubble point",
    "dew point",
    "azeotrope",
    "thermodynamic",
    "activity coefficient",
    "fugacity",
    "equation of state",
    "binary interaction",
    "benzene",
    "toluene",
    "ethanol",
    "acetone",
    "methanol",
    "温度",
    "压力",
    "组成",
    "摩尔",
    "kPa",
    "kPa",
)


def _question_is_thermo_related(question: str) -> bool:
    lower = question.casefold()
    return any(kw.casefold() in lower for kw in _THERMO_KEYWORDS)


def _kb_answer_is_relevant(answer: str) -> bool:
    lower = answer.casefold()
    if any(marker.casefold() in lower for marker in _KB_NO_ANSWER_MARKERS):
        return False
    if len(answer.strip()) < 30:
        return False
    return True


@dataclass(frozen=True)
class SkillAnswer:
    """Result produced by a knowledge-skill invocation.

    ``statements`` is rendered for direct human consumption. When the skill
    emitted a structured flow-design draft (``flow_design``), callers should
    attach it to the outer response so downstream exporters (DWSIM etc.) can
    consume a well-typed schema instead of re-parsing free text.
    """

    statements: list[EvidenceStatement]
    flow_design: FlowDesignDraft | None = None


def _extract_flow_design(result: object) -> FlowDesignDraft | None:
    """Best-effort extraction of FlowDesignDraft out of a skill's metadata dict."""
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    payload = metadata.get("flow_design_json")
    if payload is None:
        return None
    try:
        if isinstance(payload, FlowDesignDraft):
            return payload
        return FlowDesignDraft.model_validate(payload)
    except Exception:
        logger.warning("skill returned invalid flow_design_json; dropped from response")
        return None


def answer_with_skill_payload(question: str, intent: Intent) -> SkillAnswer:
    if not _question_is_thermo_related(question):
        return SkillAnswer(statements=[])

    registry = get_skill_registry()
    intent_to_skill = {
        Intent.CONCEPT_QA: ("knowledge_qa", "知识问答"),
        Intent.MODEL_SELECTION_QA: ("model_recommendation", "模型推荐"),
        Intent.PARAMETER_QUERY: ("parameter_query", "参数查询"),
        Intent.DATA_QUERY: ("knowledge_qa", "数据查询"),
        Intent.RESULT_INTERPRETATION: ("knowledge_qa", "结果解读"),
        Intent.FLOW_DESIGN_QA: ("process_flow_design", "流程设计"),
    }
    skill_name, label = intent_to_skill.get(intent, ("knowledge_qa", "知识问答"))
    flow_design: FlowDesignDraft | None = None
    try:
        result = registry.execute_skill(skill_name, question)
        if result and result.answer and result.confidence >= 0.6 and _kb_answer_is_relevant(result.answer):
            text = result.answer
            if result.sources:
                sources = "\u3001".join(result.sources[:3])
                text += f"\n\n[\u6765\u6e90: {sources}]"
            flow_design = _extract_flow_design(result)
            return SkillAnswer(
                statements=[EvidenceStatement(category="Knowledge", text=text)],
                flow_design=flow_design,
            )
    except Exception:
        pass
    if skill_name != "model_recommendation":
        try:
            result = registry.execute_skill("model_recommendation", question)
            if result and result.answer and result.confidence >= 0.6 and _kb_answer_is_relevant(result.answer):
                return SkillAnswer(statements=[EvidenceStatement(category="Knowledge", text=result.answer)])
        except Exception:
            pass
    return SkillAnswer(statements=[])


def answer_with_skills(question: str, intent: Intent) -> list[EvidenceStatement]:
    """Backwards-compatible wrapper around :func:`answer_with_skill_payload`.

    Preserves the legacy ``list[EvidenceStatement]`` return type for existing
    call sites that do not yet consume :class:`SkillAnswer.flow_design`.
    """
    return answer_with_skill_payload(question, intent).statements

