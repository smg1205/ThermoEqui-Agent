"""Skill registry for managing knowledge skills."""

from __future__ import annotations

from schemas.domain import Intent

from .graph_query_skill import GraphQuerySkill
from .knowledge_qa_skill import KnowledgeQASkill
from .model_recommendation_skill import ModelRecommendationSkill
from .parameter_query_skill import ParameterQuerySkill
from .skill_base import KnowledgeSkill, SkillResult


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, KnowledgeSkill] = {}

    def register(self, skill: KnowledgeSkill) -> None:
        self._skills[skill.name()] = skill

    def get_skill(self, name: str) -> KnowledgeSkill | None:
        return self._skills.get(name)

    def list_skills(self) -> list[dict[str, str]]:
        return [{"name": skill.name(), "description": skill.description()} for skill in self._skills.values()]

    def execute_skill(self, name: str, query: str, **kwargs) -> SkillResult:
        skill = self.get_skill(name)
        if skill is None:
            return SkillResult(
                answer=f"Skill {name} not found.",
                sources=[],
                confidence=0.0,
            )
        return skill.execute(query, **kwargs)

    def recommend_skill(self, intent: Intent | str) -> list[KnowledgeSkill]:
        """根据意图推荐技能"""
        if isinstance(intent, str):
            try:
                intent = Intent(intent)
            except ValueError:
                return []
        return [skill for skill in self._skills.values() if skill.supports_intent(intent.value)]


# 创建默认注册表并注册所有技能
DEFAULT_SKILL_REGISTRY = SkillRegistry()
DEFAULT_SKILL_REGISTRY.register(KnowledgeQASkill())
DEFAULT_SKILL_REGISTRY.register(ModelRecommendationSkill())
DEFAULT_SKILL_REGISTRY.register(GraphQuerySkill())
DEFAULT_SKILL_REGISTRY.register(ParameterQuerySkill())
