"""测试 Skills 模块（在 skills/knowledge_base/ 目录下直接运行）"""

import sys
from pathlib import Path

# 当前文件在 skills/knowledge_base/ 下，向上三级到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"项目根目录: {PROJECT_ROOT}")

from skills.knowledge_base.knowledge_qa_skill import KnowledgeQASkill
from skills.knowledge_base.graph_query_skill import GraphQuerySkill
from skills.knowledge_base.parameter_query_skill import ParameterQuerySkill
from skills.knowledge_base.llm_client import ZhipuClient


def test_skill(skill, query: str, skill_name: str, expected_keywords: list[str] = None):
    """通用测试函数，可检查预期关键词"""
    print(f"\n{'='*60}")
    print(f"测试 {skill_name}")
    print(f"查询: {query}")
    print("-" * 60)

    result = skill.execute(query)

    print(f"\n✅ 答案:\n{result.answer}")
    print(f"\n📊 置信度: {result.confidence}")
    print(f"📂 来源: {result.sources}")
    print(f"📋 元数据: {result.metadata}")

    # 关键词检查（如果提供了预期关键词）
    if expected_keywords:
        found = [kw for kw in expected_keywords if kw.lower() in result.answer.lower()]
        if found:
            print(f"✅ 检测到预期关键词: {found}")
        else:
            print(f"⚠️ 未检测到预期关键词: {expected_keywords}")
            print("   建议检查答案是否符合预期")


def main():
    print("=" * 60)
    print("测试 Skills 模块（全部技能）- 基于知识库内容")
    print("=" * 60)

    # 初始化 LLM
    try:
        llm = ZhipuClient()
        print("✅ LLM 客户端初始化成功（智谱）")
    except Exception as e:
        print(f"⚠️ LLM 初始化失败: {e}")
        llm = None

    # 创建技能
    knowledge_qa = KnowledgeQASkill(llm=llm)
    graph_query = GraphQuerySkill(llm=llm)
    parameter_query = ParameterQuerySkill(llm=llm)

    # ========== 测试用例（基于实际知识库内容） ==========
    test_cases = [
        # Knowledge QA 测试（RAG 检索）
        (
            knowledge_qa,
            "Wilson 模型的特点是什么？",
            "Knowledge QA",
            ["局部组成", "local composition", "二元参数", "LLE"]
        ),
        (
            knowledge_qa,
            "Peng-Robinson 方程的关键参数有哪些？",
            "Knowledge QA",
            ["偏心因子", "acentric", "临界温度", "critical temperature"]
        ),
        (
            knowledge_qa,
            "什么是 TP Flash？",
            "Knowledge QA",
            ["闪蒸", "vapor fraction", "Rachford-Rice"]
        ),

        # Graph Query 测试（知识图谱）
        (
            graph_query,
            "SRK 模型不适用于什么体系？",
            "Graph Query",
            ["electrolyte", "电解质"]
        ),
        (
            graph_query,
            "Wilson 模型支持什么任务？",
            "Graph Query",
            ["VLE"]
        ),
        (
            graph_query,
            "NRTL 模型排除什么体系？",
            "Graph Query",
            ["electrolyte"]
        ),

        # Parameter Query 测试（参数提取）
        (
            parameter_query,
            "Wilson 模型需要二元参数吗？",
            "Parameter Query",
            ["true", "是", "需要"]
        ),
        (
            parameter_query,
            "PR 模型的压力范围是什么？",
            "Parameter Query",
            ["moderate", "high", "中压", "高压"]
        ),
        (
            parameter_query,
            "NRTL 的实现状态是什么？",
            "Parameter Query",
            ["contract_only"]
        ),
    ]

    for skill, query, name, keywords in test_cases:
        test_skill(skill, query, name, keywords)

    print("\n" + "=" * 60)
    print("🎉 所有技能测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
    from knowledge_graph.graph import KnowledgeGraph

    kg = KnowledgeGraph.load()
    print("图谱节点:", list(kg.nodes.keys()))
    print("图谱关系:", len(kg.relationships))
