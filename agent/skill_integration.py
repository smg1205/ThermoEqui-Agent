"""Central integration module for RAG, knowledge graph, and skill-based answering.
Provides lazy initialization and a clean interface for the orchestrator.
"""
from __future__ import annotations
import logging
import threading
from typing import Optional
from schemas.domain import Intent, EvidenceStatement
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


def answer_with_skills(question: str, intent: Intent) -> list[EvidenceStatement]:
    registry = get_skill_registry()
    intent_str = intent.value if isinstance(intent, Intent) else str(intent)
    intent_to_skill = {
        Intent.CONCEPT_QA: ("knowledge_qa", "知识问答"),
        Intent.MODEL_SELECTION_QA: ("model_recommendation", "模型推荐"),
        Intent.PARAMETER_QUERY: ("parameter_query", "参数查询"),
        Intent.DATA_QUERY: ("knowledge_qa", "数据查询"),
        Intent.RESULT_INTERPRETATION: ("knowledge_qa", "结果解读"),
    }
    skill_name, label = intent_to_skill.get(intent, ("knowledge_qa", "知识问答"))
    try:
        result = registry.execute_skill(skill_name, question)
        if result and result.answer and result.confidence > 0:
            text = result.answer
            if result.sources:
                sources = "\u3001".join(result.sources[:3])
                text += f"\n\n[\u6765\u6e90: {sources}]"
            return [EvidenceStatement(category="Knowledge", text=text)]
    except Exception:
        pass
    if skill_name != "model_recommendation":
        try:
            result = registry.execute_skill("model_recommendation", question)
            if result and result.answer and result.confidence > 0:
                return [EvidenceStatement(category="Knowledge", text=result.answer)]
        except Exception:
            pass
    return []
