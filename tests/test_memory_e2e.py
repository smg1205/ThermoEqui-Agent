"""End-to-end test: simulate multi-turn conversation memory usage.

Run: python -m pytest tests/test_memory_e2e.py -v
     python tests/test_memory_e2e.py  (standalone demo)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.conversation_memory import ConversationMemory
from agent.memory_integration import (
    build_calc_reference,
    build_context_prefix,
    retrieve_for_calculation,
    retrieve_for_concept_qa,
    save_turn,
)
from schemas.domain import Intent


def run_demo() -> None:
    """Simulate a realistic multi-turn thermodynamics conversation."""
    db_path = Path(__file__).resolve().parent / "data" / "demo_memory.db"
    if db_path.exists():
        db_path.unlink()

    print("=" * 60)
    print("  Conversation Memory E2E Demo")
    print("=" * 60)

    memory = ConversationMemory(db_path=db_path)
    session_id = "demo-session-001"

    # ── Turn 1: Concept Q&A ──────────────────────────────────────
    print("\n[Turn 1] User: 什么是活度系数？")
    save_turn(
        session_id,
        "什么是活度系数？",
        "活度系数是描述实际溶液偏离理想溶液程度的参数，用γ表示。γ=1为理想溶液。",
        Intent.CONCEPT_QA,
    )
    prefix = retrieve_for_concept_qa(session_id, "它和逸度系数有什么区别？")
    print(f"  → Memory injected: {'YES' if prefix else 'NO'}")
    if prefix:
        print(f"  → Context:\n{prefix}")

    # ── Turn 2: Follow-up referencing Turn 1 ──────────────────────
    print("\n[Turn 2] User: 它和逸度系数有什么区别？")
    save_turn(
        session_id,
        "它和逸度系数有什么区别？",
        "活度系数用于液相，逸度系数用于气相。两者都描述偏离理想的程度。",
        Intent.CONCEPT_QA,
    )
    prefix = retrieve_for_concept_qa(session_id, "活度系数的公式是什么？")
    print(f"  → Memory injected: {'YES' if prefix else 'NO'}")
    if prefix:
        print(f"  → Context:\n{prefix}")

    # ── Turn 3: Calculation task ─────────────────────────────────
    print("\n[Turn 3] User: 计算苯-甲苯在101.3kPa下的泡点")
    save_turn(
        session_id,
        "计算苯-甲苯在101.3kPa下的泡点",
        "T_bubble = 384.15K，模型：Ideal/Raoult",
        Intent.EQUILIBRIUM_CALCULATION,
        components=["benzene", "toluene"],
        task_summary="bubble_point, benzene-toluene, Ideal/Raoult, 101.3kPa",
    )
    ref = retrieve_for_calculation(session_id, "再算一下乙醇-水的泡点")
    print(f"  → Calc reference: {'YES' if ref else 'NO'}")
    if ref:
        print(f"  → Reference:\n{ref}")

    # ── Turn 4: New calculation, should reference Turn 3 ─────────
    print("\n[Turn 4] User: 再算一下乙醇-水的泡点")
    save_turn(
        session_id,
        "再算一下乙醇-水的泡点",
        "T_bubble = 351.5K，乙醇-水形成共沸物",
        Intent.EQUILIBRIUM_CALCULATION,
        components=["ethanol", "water"],
        task_summary="bubble_point, ethanol-water, azeotrope at 101.3kPa",
    )
    ref = retrieve_for_calculation(session_id, "那甲烷-乙烷呢？")
    print(f"  → Calc reference: {'YES' if ref else 'NO'}")
    if ref:
        print(f"  → Reference:\n{ref}")

    # ── Turn 5: Model selection Q&A ─────────────────────────────
    print("\n[Turn 5] User: NRTL和Wilson模型有什么区别？")
    save_turn(
        session_id,
        "NRTL和Wilson模型有什么区别？",
        "NRTL可处理LLE，Wilson不能。NRTL三参数，Wilson两参数。",
        Intent.MODEL_SELECTION_QA,
    )
    prefix = retrieve_for_concept_qa(session_id, "那应该用哪个模型？")
    print(f"  → Memory injected: {'YES' if prefix else 'NO'}")
    if prefix:
        print(f"  → Context:\n{prefix}")

    # ── Turn 6: Follow-up model selection ────────────────────────
    print("\n[Turn 6] User: 那应该用哪个模型？")
    save_turn(
        session_id,
        "那应该用哪个模型？",
        "部分互溶体系选NRTL，完全互溶可选Wilson或NRTL。",
        Intent.MODEL_SELECTION_QA,
    )

    # ── Cross-session test ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Cross-Session Isolation Test")
    print("=" * 60)
    other_session = retrieve_for_concept_qa("other-session", "活度系数")
    print(f"\n  Retrieving '活度系数' in other-session: {'FOUND' if other_session else 'NOT FOUND (expected)'}")
    assert not other_session, "Cross-session leak detected!"

    same_session = retrieve_for_concept_qa(session_id, "活度系数")
    print(f"  Retrieving '活度系数' in demo-session-001: {'FOUND' if same_session else 'NOT FOUND'}")
    assert same_session, "Same-session memory miss!"

    # ── Build context prefix test ────────────────────────────────
    print("\n" + "=" * 60)
    print("  Context Prefix Builder Test")
    print("=" * 60)
    memories = memory.retrieve(session_id, "活度系数", top_k=2)
    prefix = build_context_prefix(memories)
    print(f"\n  Retrieved {len(memories)} memories")
    print(f"  Built prefix:\n{prefix}")

    # ── Build calc reference test ────────────────────────────────
    print("\n" + "=" * 60)
    print("  Calculation Reference Builder Test")
    print("=" * 60)
    memories = memory.retrieve(session_id, "泡点", top_k=3)
    ref = build_calc_reference(memories)
    print(f"\n  Retrieved {len(memories)} memories")
    print(f"  Built reference:\n{ref if ref else '  (empty — no calculation records)'}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
    print(f"\n  Database: {db_path}")
    print(f"  Total records in demo-session-001: {_count_records(db_path, session_id)}")

    # Cleanup
    db_path.unlink(missing_ok=True)
    print("  (demo database cleaned up)")


def _count_records(db_path: Path, session_id: str) -> int:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    count = conn.execute(
        "SELECT COUNT(*) FROM conversation_memory WHERE session_id = ?",
        (session_id,),
    ).fetchone()[0]
    conn.close()
    return count


if __name__ == "__main__":
    run_demo()
