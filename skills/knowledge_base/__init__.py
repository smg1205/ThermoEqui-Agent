"""Knowledge base skills for RAG and knowledge graph queries."""

from .graph_query_skill import GraphQuerySkill
from .knowledge_qa_skill import KnowledgeQASkill
from .llm_client import LLMClient, MockLLMClient, ProjectProviderClient, get_default_llm
from .model_recommendation_skill import ModelRecommendationSkill
from .parameter_query_skill import ParameterQuerySkill
from .skill_base import KnowledgeSkill, SkillResult
from .skill_registry import DEFAULT_SKILL_REGISTRY, Intent, SkillRegistry

__all__ = [
    "KnowledgeQASkill",
    "ModelRecommendationSkill",
    "GraphQuerySkill",
    "ParameterQuerySkill",
    "KnowledgeSkill",
    "SkillResult",
    "SkillRegistry",
    "Intent",
    "DEFAULT_SKILL_REGISTRY",
    "LLMClient",
    "ProjectProviderClient",
    "MockLLMClient",
    "get_default_llm",
]

# 简单的自测
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Skills 模块测试")
    print("=" * 60)

    print("已注册的技能:")
    for s in DEFAULT_SKILL_REGISTRY.list_skills():
        print(f"  - {s['name']}: {s['description']}")

    print("\n尝试执行 knowledge_qa 技能（查询：什么是热力学平衡）...")
    result = DEFAULT_SKILL_REGISTRY.execute_skill("knowledge_qa", "什么是热力学平衡")
    print(f"回答: {result.answer[:200] if result.answer else '无'}...")
    print(f"置信度: {result.confidence}")
    print("=" * 60)
