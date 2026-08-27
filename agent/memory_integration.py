"""Integration layer: retrieve conversation memory and inject into responses."""

from __future__ import annotations

import logging
import re

from agent.conversation_memory import ConversationMemory, MemoryRecord
from schemas.domain import Intent

logger = logging.getLogger(__name__)

# Lazy singleton
_memory_instance: ConversationMemory | None = None


def get_memory() -> ConversationMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory()
    return _memory_instance


def build_context_prefix(memories: list[MemoryRecord]) -> str:
    """Build a context prefix from retrieved memories for LLM prompt injection."""
    if not memories:
        return ""
    lines = ["之前的对话历史（供参考）："]
    for i, mem in enumerate(memories, 1):
        lines.append(f"{i}. 问：{mem.question}")
        # Truncate long answers to keep prompt concise
        answer_preview = mem.answer[:200] + "…" if len(mem.answer) > 200 else mem.answer
        lines.append(f"   答：{answer_preview}")
    lines.append("")
    return "\n".join(lines)


_NUMBER_IN_TEXT = re.compile(r"\d+(?:\.\d+)?")


def extract_grounded_numbers(memories: list[MemoryRecord]) -> set[str]:
    """Extract numeric tokens from memory answers that have been validated by thermo_engine.

    These numbers are grounded in prior calculation results and can be safely
    referenced by the LLM during concept/model-comparison Q&A.
    """
    grounded: set[str] = set()
    for mem in memories:
        if mem.task_summary:  # Calculation memories = validated by thermo_engine
            for num in _NUMBER_IN_TEXT.findall(mem.answer):
                grounded.add(num)
    return grounded


def retrieve_for_concept_qa(session_id: str, message: str) -> tuple[str, set[str]]:
    """Retrieve memories for concept Q&A.

    Returns (context_prefix, grounded_number_strings).
    """
    memory = get_memory()
    memories = memory.retrieve(session_id, message, top_k=3)
    prefix = build_context_prefix(memories)
    grounded = extract_grounded_numbers(memories)
    return prefix, grounded


def build_calc_reference(memories: list[MemoryRecord]) -> str:
    """Build a reference suffix for calculation tasks citing prior results."""
    calc_memories = [m for m in memories if m.task_summary]
    if not calc_memories:
        return ""
    lines = ["\n\n[历史计算参考]"]
    for mem in calc_memories[:2]:
        lines.append(f"- 您之前问过「{mem.question}」：{mem.task_summary}")
    return "\n".join(lines)


def retrieve_for_calculation(session_id: str, message: str) -> str:
    """Retrieve memories and return reference suffix for calculation tasks."""
    memory = get_memory()
    memories = memory.retrieve(session_id, message, top_k=3)
    return build_calc_reference(memories)


def save_turn(
    session_id: str,
    question: str,
    answer: str,
    intent: Intent,
    components: list[str] | None = None,
    task_summary: str | None = None,
) -> None:
    """Save a conversation turn to memory."""
    try:
        memory = get_memory()
        memory.save(
            session_id=session_id,
            question=question,
            answer=answer,
            intent=intent,
            components=components,
            task_summary=task_summary,
        )
    except Exception as exc:
        logger.warning("Failed to save conversation memory: %s", exc)
