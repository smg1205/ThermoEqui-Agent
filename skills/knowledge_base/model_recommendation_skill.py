"""Model recommendation skill using knowledge graph."""

from __future__ import annotations

from knowledge_graph.query_engine import GraphQueryEngine
from .skill_base import KnowledgeSkill, SkillResult
from .llm_client import LLMClient


class ModelRecommendationSkill(KnowledgeSkill):
    def __init__(self, query_engine: GraphQueryEngine | None = None, llm: LLMClient | None = None) -> None:
        super().__init__(llm=llm)
        self.query_engine = query_engine or GraphQueryEngine()

    def name(self) -> str:
        return "model_recommendation"

    def description(self) -> str:
        return "Recommend suitable thermodynamic models based on task and system type."

    def execute(self, query: str, **kwargs) -> SkillResult:
        try:
            # 1. 先尝试从图谱中提取实体并查询
            result = self.query_engine.query(query)
            if result.answer:
                return SkillResult(
                    answer=result.answer,
                    sources=["knowledge_graph"],
                    confidence=0.6,
                    metadata={
                        "nodes_found": len(result.nodes),
                        "relationships_found": len(result.relationships),
                    },
                )

            # 2. 若没有直接答案，尝试关键词检测任务
            detected_task = self._detect_task(query)
            if detected_task:
                models = self.query_engine.find_models_for_task(detected_task)
                if models:
                    answer_text = f"支持 {detected_task} 的推荐模型：{', '.join(models)}"
                    # 若有LLM，可润色
                    if self._llm_available and self._llm is not None:
                        context = f"检测到任务: {detected_task}\n推荐模型: {', '.join(models)}"
                        answer_text = self._synthesize_with_llm(
                            query=query,
                            context=context,
                            system_prompt="请根据检测到的任务和推荐模型，给出自然简洁的推荐理由。"
                        )
                    return SkillResult(
                        answer=answer_text,
                        sources=["knowledge_graph"],
                        confidence=0.7,
                        metadata={"task": detected_task, "models": models, "llm_used": self._llm_available},
                    )
        except Exception as e:
            return SkillResult(
                answer=f"模型推荐失败: {e}",
                sources=[],
                confidence=0.0,
            )

        return SkillResult(
            answer="无法从知识图谱中获取模型推荐信息。",
            sources=[],
            confidence=0.0,
        )

    def _detect_task(self, query: str) -> str | None:
        """检测查询中的任务关键词（中英文）"""
        task_mapping = {
            "VLE": ["vle", "vapor liquid equilibrium", "气液平衡", "汽液平衡"],
            "LLE": ["lle", "liquid liquid equilibrium", "液液平衡"],
            "Flash": ["flash", "闪蒸"],
            "bubble": ["bubble point", "泡点"],
            "dew": ["dew point", "露点"],
            "distillation": ["distillation", "蒸馏", "精馏"],
        }
        lower_query = query.lower()
        for task, keywords in task_mapping.items():
            if any(kw.lower() in lower_query for kw in keywords):
                return task
        return None

    def supports_intent(self, intent: str) -> bool:
        return intent in {"MODEL_SELECTION_QA", "EQUILIBRIUM_CALCULATION"}