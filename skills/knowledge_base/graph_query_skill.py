"""Knowledge graph query skill with CoT, Few-shot, and direct graph lookup."""

from __future__ import annotations

from knowledge_graph.query_engine import GraphQueryEngine
from knowledge_graph.graph import KnowledgeGraph
from .skill_base import KnowledgeSkill, SkillResult
from .llm_client import LLMClient


class GraphQuerySkill(KnowledgeSkill):
    """知识图谱查询技能：实体关系查询 + 多跳推理 + 直接图谱查找"""

    def __init__(
        self,
        query_engine: GraphQueryEngine | None = None,
        llm: LLMClient | None = None,
        temperature: float = 0.2,
    ) -> None:
        super().__init__(llm=llm, temperature=temperature)
        # ========== 关键修复：加载已保存的图谱 ==========
        if query_engine is None:
            kg = KnowledgeGraph.load()  # 从 data/knowledge_graph.json 加载
            self.query_engine = GraphQueryEngine(graph=kg)
        else:
            self.query_engine = query_engine

    def name(self) -> str:
        return "graph_query"

    def description(self) -> str:
        return (
            "Query the knowledge graph for entity relationships, model information, "
            "and multi-hop reasoning about thermodynamic models."
        )

    def _direct_graph_lookup(self, query: str) -> tuple[list[dict], list[dict]]:
        """直接从图谱查找，绕过实体提取器"""
        query_lower = query.lower()

        model_mappings = {
            "srk": ["soave-redlich-kwong", "srk"],
            "pr": ["peng-robinson", "pr"],
            "nrtl": ["nrtl"],
            "wilson": ["wilson"],
            "uniquac": ["uniquac"],
            "ideal": ["ideal/raoult", "ideal", "raoult"],
            "raoult": ["ideal/raoult", "ideal", "raoult"],
        }

        matched_nodes = []
        matched_ids = set()

        for model_key, search_terms in model_mappings.items():
            if model_key in query_lower or any(term in query_lower for term in search_terms):
                for node_id, node in self.query_engine.graph.nodes.items():
                    node_id_lower = node_id.lower()
                    node_label_lower = node.label.lower()
                    for term in search_terms:
                        if (term in node_id_lower or
                            term in node_label_lower or
                            node_id_lower.endswith(f":{term}") or
                            node_label_lower == term):
                            if node.id not in matched_ids:
                                matched_nodes.append(node)
                                matched_ids.add(node.id)
                            break

        if not matched_nodes:
            return [], []

        nodes = []
        relationships = []
        visited_ids = set()

        for node in matched_nodes:
            nodes.append({
                "id": node.id,
                "label": node.label,
                "type": node.type,
                "attributes": node.attributes,
            })
            visited_ids.add(node.id)

            for neighbor, rel in self.query_engine.graph.neighbors(node.id):
                if neighbor.id not in visited_ids:
                    nodes.append({
                        "id": neighbor.id,
                        "label": neighbor.label,
                        "type": neighbor.type,
                        "attributes": neighbor.attributes,
                    })
                    visited_ids.add(neighbor.id)
                relationships.append({
                    "source": node.id,
                    "target": neighbor.id,
                    "type": rel.type.value,
                    "attributes": rel.attributes,
                })

        return nodes, relationships

    def _generate_answer_from_nodes(
        self,
        nodes: list[dict],
        relationships: list[dict],
        query: str,
    ) -> str:
        model_nodes = [n for n in nodes if n["type"] == "model"]
        task_nodes = [n for n in nodes if n["type"] == "task"]
        system_nodes = [n for n in nodes if n["type"] == "system_type"]

        answer_parts = []

        if any(kw in query.lower() for kw in ["不适用", "不适用于", "排除", "except", "exclude"]):
            if model_nodes and system_nodes:
                for model in model_nodes:
                    excluded = []
                    for rel in relationships:
                        if rel["source"] == model["id"] and rel["type"] == "excludes":
                            for sys_node in system_nodes:
                                if sys_node["id"] == rel["target"]:
                                    excluded.append(sys_node["label"])
                    if excluded:
                        answer_parts.append(f"{model['label']} 不适用于: {', '.join(excluded)}")
                    else:
                        answer_parts.append(f"{model['label']} 无不适用体系")
                return "\n".join(answer_parts) if answer_parts else "未找到排除关系。"

        if any(kw in query.lower() for kw in ["支持", "适用", "任务", "计算", "support", "task"]):
            if model_nodes and task_nodes:
                for model in model_nodes:
                    supported = []
                    for rel in relationships:
                        if rel["source"] == model["id"] and rel["type"] == "supports_task":
                            for task in task_nodes:
                                if task["id"] == rel["target"]:
                                    supported.append(task["label"])
                    if supported:
                        answer_parts.append(f"{model['label']} 支持的任务: {', '.join(supported)}")
                    else:
                        answer_parts.append(f"{model['label']} 暂无支持的任务")
                return "\n".join(answer_parts) if answer_parts else "未找到任务关系。"

        if nodes:
            labels = [f"{n['label']} ({n['type']})" for n in nodes[:5]]
            return f"识别到相关实体: {', '.join(labels)}"

        return "知识图谱中未找到相关实体。"

    def execute(self, query: str, **kwargs) -> SkillResult:
        try:
            result = self.query_engine.query(query)

            if not result.nodes:
                direct_nodes, direct_rels = self._direct_graph_lookup(query)
                if direct_nodes:
                    answer = self._generate_answer_from_nodes(direct_nodes, direct_rels, query)

                    if self._llm_available and self._llm is not None:
                        context_parts = []
                        node_info = [f"- {n['label']} ({n['type']})" for n in direct_nodes]
                        rel_info = [f"- {r['type']}: {r['source']} -> {r['target']}" for r in direct_rels]
                        if node_info:
                            context_parts.append("实体:\n" + "\n".join(node_info))
                        if rel_info:
                            context_parts.append("关系:\n" + "\n".join(rel_info))
                        combined_context = "\n\n".join(context_parts)

                        answer = self._synthesize_with_llm(
                            query=query,
                            context=combined_context,
                            system_prompt=(
                                "你是热力学知识图谱分析专家。请根据图谱查询结果回答用户问题。\n"
                                "直接回答用户的问题，不要重复列出所有实体。\n"
                                "如果用户问的是'不适用于什么体系'，重点回答排除项。\n"
                                "如果用户问的是'支持什么任务'，重点回答支持的任务。"
                            ),
                            temperature=self._temperature,
                        )

                    return SkillResult(
                        answer=answer,
                        sources=["knowledge_graph"],
                        confidence=0.7,
                        metadata={
                            "nodes_found": len(direct_nodes),
                            "relationships_found": len(direct_rels),
                            "llm_used": self._llm_available,
                            "method": "direct_lookup",
                        },
                    )

            if result.nodes:
                node_info = [f"- {n['label']} ({n['type']})" for n in result.nodes]
                rel_info = [f"- {r['type']}: {r['source']} -> {r['target']}" for r in result.relationships]

                context_parts = []
                if node_info:
                    context_parts.append("识别到的实体：\n" + "\n".join(node_info))
                if rel_info:
                    context_parts.append("关系：\n" + "\n".join(rel_info))
                if result.answer:
                    context_parts.append(f"原始回答：\n{result.answer}")

                combined_context = "\n\n".join(context_parts)

                answer = self._synthesize_with_llm(
                    query=query,
                    context=combined_context,
                    system_prompt=(
                        "你是热力学知识图谱分析专家。请根据图谱查询结果回答用户问题。\n"
                        "直接回答用户的问题，不要重复列出所有实体。\n"
                        "如果用户问的是'不适用于什么体系'，重点回答排除项。\n"
                        "如果用户问的是'支持什么任务'，重点回答支持的任务。"
                    ),
                    temperature=self._temperature,
                )

                return SkillResult(
                    answer=answer,
                    sources=["knowledge_graph"],
                    confidence=0.6,
                    metadata={
                        "nodes_found": len(result.nodes),
                        "relationships_found": len(result.relationships),
                        "llm_used": self._llm_available,
                        "method": "entity_extractor",
                    },
                )

            return SkillResult(
                answer="🔍 知识图谱中未找到相关实体。请尝试使用完整的模型名称（如 'SRK' 或 'Peng-Robinson'）。",
                sources=[],
                confidence=0.0,
            )

        except Exception as e:
            return SkillResult(
                answer=f"❌ 知识图谱查询失败: {e}",
                sources=[],
                confidence=0.0,
            )

    def supports_intent(self, intent: str) -> bool:
        return intent in {"CONCEPT_QA", "KNOWLEDGE_QA", "MODEL_SELECTION_QA"}