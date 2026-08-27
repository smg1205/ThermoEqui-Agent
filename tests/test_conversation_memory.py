"""Tests for conversation memory hybrid retrieval accuracy."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.conversation_memory import ConversationMemory


@pytest.fixture
def memory(tmp_path: Path) -> ConversationMemory:
    """Fresh in-memory-like SQLite DB per test."""
    db_path = tmp_path / "test_memory.db"
    return ConversationMemory(db_path=db_path)


@pytest.fixture
def populated_memory(memory: ConversationMemory) -> ConversationMemory:
    """Pre-populated with realistic thermodynamics Q&A pairs."""
    test_data = [
        {
            "session_id": "session-1",
            "question": "什么是活度系数？",
            "answer": (
                "活度系数是描述实际溶液偏离理想溶液程度的参数，用γ表示。γ=1为理想溶液，γ>1为正偏差，γ<1为负偏差。"
            ),
            "intent": "CONCEPT_QA",
            "components": [],
            "task_summary": None,
        },
        {
            "session_id": "session-1",
            "question": "NRTL和Wilson模型有什么区别？",
            "answer": "NRTL可处理部分互溶体系（LLE），Wilson不能。NRTL有三参数(τ12,τ21,α)，Wilson有两参数(Λ12,Λ21)。",
            "intent": "MODEL_SELECTION_QA",
            "components": [],
            "task_summary": None,
        },
        {
            "session_id": "session-1",
            "question": "计算苯-甲苯体系在101.3kPa下的泡点温度",
            "answer": "计算完成。模型：Ideal/Raoult。T=384.15K，P=101.3kPa。",
            "intent": "EQUILIBRIUM_CALCULATION",
            "components": ["benzene", "toluene"],
            "task_summary": "bubble_point, VLE, Ideal/Raoult",
        },
        {
            "session_id": "session-1",
            "question": "乙醇-水体系在常压下会形成共沸物吗？",
            "answer": "乙醇-水在101.3kPa下形成正共沸物，共沸温度约351.5K，乙醇摩尔分数约0.894。",
            "intent": "CONCEPT_QA",
            "components": ["ethanol", "water"],
            "task_summary": None,
        },
        {
            "session_id": "session-1",
            "question": "Peng-Robinson状态方程适用于什么条件？",
            "answer": "PR方程适用于中高压烃类体系的VLE和Flash计算，特别是非极性或弱极性流体。",
            "intent": "CONCEPT_QA",
            "components": [],
            "task_summary": None,
        },
        {
            "session_id": "session-1",
            "question": "帮我计算甲烷-乙烷-氮气的TP Flash",
            "answer": "计算完成。模型：Peng-Robinson。Flash温度=200K，压力=5000kPa。",
            "intent": "EQUILIBRIUM_CALCULATION",
            "components": ["methane", "ethane", "nitrogen"],
            "task_summary": "tp_flash, FLASH, Peng-Robinson",
        },
        {
            "session_id": "session-1",
            "question": "什么是逸度？",
            "answer": "逸度是修正后的有效压力，描述真实气体偏离理想气体的程度。f=φP，φ为逸度系数。",
            "intent": "CONCEPT_QA",
            "components": [],
            "task_summary": None,
        },
        {
            "session_id": "session-1",
            "question": "如何选择合适的活度系数模型？",
            "answer": "低中压极性体系选NRTL或UNIQUAC；高压非极性体系选状态方程；理想体系选Raoult定律。",
            "intent": "MODEL_SELECTION_QA",
            "components": [],
            "task_summary": None,
        },
    ]

    for item in test_data:
        memory.save(
            session_id=item["session_id"],
            question=item["question"],
            answer=item["answer"],
            intent=item["intent"],
            components=item["components"],
            task_summary=item["task_summary"],
        )
    return memory


class TestHybridRetrieval:
    """Test hybrid retrieval accuracy with realistic thermodynamics queries."""

    def test_semantic_match_activity_coefficient(self, populated_memory: ConversationMemory) -> None:
        """Query about γ should retrieve the activity coefficient Q&A first."""
        results = populated_memory.retrieve("session-1", "活度系数γ是什么意思", top_k=3)
        assert len(results) > 0
        assert "活度系数" in results[0].question

    def test_semantic_match_model_difference(self, populated_memory: ConversationMemory) -> None:
        """Query about NRTL vs Wilson should retrieve the model comparison Q&A."""
        results = populated_memory.retrieve("session-1", "NRTL比Wilson好在哪", top_k=3)
        assert len(results) > 0
        assert "NRTL" in results[0].question or "Wilson" in results[0].question

    def test_keyword_match_benzene_toluene(self, populated_memory: ConversationMemory) -> None:
        """Query with explicit component names should boost keyword match."""
        results = populated_memory.retrieve("session-1", "苯-甲苯的泡点计算结果", top_k=3)
        assert len(results) > 0
        # The benzene-toluene calculation should be in top results
        top_questions = [r.question for r in results[:2]]
        assert any("苯" in q or "benzene" in q.lower() for q in top_questions)

    def test_keyword_match_ethanol_water(self, populated_memory: ConversationMemory) -> None:
        """Query about ethanol-water azeotrope should find the right Q&A."""
        results = populated_memory.retrieve("session-1", "乙醇和水会形成共沸物吗", top_k=3)
        assert len(results) > 0
        top_questions = [r.question for r in results[:2]]
        assert any("乙醇" in q or "ethanol" in q.lower() for q in top_questions)

    def test_calculation_task_retrieval(self, populated_memory: ConversationMemory) -> None:
        """Query about TP Flash should retrieve the methane-ethane-nitrogen calculation."""
        results = populated_memory.retrieve("session-1", "甲烷乙烷的闪蒸计算", top_k=3)
        assert len(results) > 0
        # Should find the flash calculation record
        top_questions = [r.question for r in results[:3]]
        assert any("Flash" in q or "flash" in q.lower() for q in top_questions)

    def test_peng_robinson_retrieval(self, populated_memory: ConversationMemory) -> None:
        """Query about PR equation should retrieve the PR concept Q&A."""
        results = populated_memory.retrieve("session-1", "PR状态方程适用范围", top_k=3)
        assert len(results) > 0
        top_questions = [r.question for r in results[:3]]
        assert any("Peng-Robinson" in q or "PR" in q for q in top_questions)

    def test_fugacity_retrieval(self, populated_memory: ConversationMemory) -> None:
        """Query about fugacity should retrieve the fugacity Q&A."""
        results = populated_memory.retrieve("session-1", "逸度系数怎么理解", top_k=3)
        assert len(results) > 0
        assert "逸度" in results[0].question

    def test_model_selection_retrieval(self, populated_memory: ConversationMemory) -> None:
        """Query about model selection should retrieve the model selection Q&A."""
        results = populated_memory.retrieve("session-1", "应该用什么模型来计算", top_k=3)
        assert len(results) > 0
        # Should find one of the model selection Q&As
        top_questions = [r.question for r in results[:3]]
        assert any("模型" in q and ("选择" in q or "区别" in q) for q in top_questions)

    def test_cross_session_isolation(self, populated_memory: ConversationMemory) -> None:
        """Memory from session-1 should not be retrievable in session-2."""
        results = populated_memory.retrieve("session-2", "活度系数是什么", top_k=3)
        assert len(results) == 0

    def test_empty_session(self, memory: ConversationMemory) -> None:
        """Empty session should return no results."""
        results = memory.retrieve("empty-session", "任何问题", top_k=3)
        assert len(results) == 0

    def test_top_k_limit(self, populated_memory: ConversationMemory) -> None:
        """Should respect top_k limit."""
        results = populated_memory.retrieve("session-1", "热力学", top_k=2)
        assert len(results) <= 2

    def test_similarity_ordering(self, populated_memory: ConversationMemory) -> None:
        """Results should be ordered by combined similarity (descending)."""
        results = populated_memory.retrieve("session-1", "NRTL Wilson 区别", top_k=5)
        assert len(results) >= 2
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity

    def test_task_summary_preserved(self, populated_memory: ConversationMemory) -> None:
        """Calculation records should preserve task_summary for reference."""
        results = populated_memory.retrieve("session-1", "苯甲苯泡点", top_k=3)
        calc_records = [r for r in results if r.task_summary]
        assert len(calc_records) > 0
        assert "bubble_point" in calc_records[0].task_summary

    def test_components_preserved(self, populated_memory: ConversationMemory) -> None:
        """Component list should be preserved for calculation records."""
        results = populated_memory.retrieve("session-1", "甲烷乙烷闪蒸", top_k=3)
        calc_records = [r for r in results if r.components]
        assert len(calc_records) > 0
        assert "methane" in calc_records[0].components

    def test_clear_session(self, populated_memory: ConversationMemory) -> None:
        """Clearing a session should remove all its records."""
        populated_memory.clear_session("session-1")
        results = populated_memory.retrieve("session-1", "活度系数", top_k=3)
        assert len(results) == 0
