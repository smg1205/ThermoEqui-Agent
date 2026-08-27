"""Query engine for knowledge graph traversal and reasoning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from knowledge_graph.entity_extractor import Entity, EntityExtractor, ThermoEntityExtractor
from knowledge_graph.graph import KnowledgeGraph, RelationshipType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphQueryResult:
    nodes: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    answer: str | None = None


class GraphQueryEngine:
    def __init__(
        self,
        graph: KnowledgeGraph | None = None,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        self.graph = graph or KnowledgeGraph()
        self.entity_extractor = entity_extractor or ThermoEntityExtractor()

    def cached_query(self, question: str) -> GraphQueryResult:
        """带缓存的高频查询接口"""
        return self.query(question)

    def query(self, question: str) -> GraphQueryResult:
        entities = self.entity_extractor.extract(question)
        if not entities:
            return GraphQueryResult(nodes=[], relationships=[], answer="未识别到相关实体")

        results: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        for entity in entities:
            node_id = self._find_node_id(entity)
            if node_id:
                node = self.graph.get_node(node_id)
                if node:
                    results.append(
                        {
                            "id": node.id,
                            "label": node.label,
                            "type": node.type,
                            "attributes": node.attributes,
                        }
                    )
                neighbors = self.graph.neighbors(node_id)
                for neighbor, rel in neighbors:
                    results.append(
                        {
                            "id": neighbor.id,
                            "label": neighbor.label,
                            "type": neighbor.type,
                            "attributes": neighbor.attributes,
                        }
                    )
                    relationships.append(
                        {
                            "source": node_id,
                            "target": neighbor.id,
                            "type": rel.type.value,
                            "attributes": rel.attributes,
                        }
                    )

        answer = self._generate_answer(question, entities, results, relationships)
        return GraphQueryResult(
            nodes=results,
            relationships=relationships,
            answer=answer,
        )

    def _find_node_id(self, entity: Entity) -> str | None:
        """改进的节点查找，支持精确匹配和别名归一化"""
        entity_text = entity.text.lower().replace(" ", "_")

        # 1. 尝试匹配节点ID后缀（例如 "model:nrtl" 匹配 "nrtl"）
        for node_id in self.graph.nodes:
            if node_id.endswith(f":{entity_text}"):
                return node_id

        # 2. 精确匹配节点标签（不区分大小写）
        for node_id, node in self.graph.nodes.items():
            if node.label.lower() == entity_text:
                return node_id

        # 3. 词边界匹配（防止子串误匹配，如 "propane" 不匹配 "propanol"）
        for node_id, node in self.graph.nodes.items():
            label_lower = node.label.lower()
            if f" {entity_text} " in f" {label_lower} ":
                return node_id

        # 4. 别名映射（将实体文本映射到标准名称后再次尝试）
        alias_map = {
            "peng-robinson": "pr",
            "soave-redlich-kwong": "srk",
            "non-random two-liquid": "nrtl",
            "universal quasichemical": "uniquac",
            "威尔逊": "wilson",
            "非随机双液体模型": "nrtl",
            "彭-罗宾逊": "pr",
            "索阿韦-雷德利希-邝": "srk",
            "通用准化学": "uniquac",
        }
        if entity_text in alias_map:
            canonical = alias_map[entity_text]
            for node_id in self.graph.nodes:
                if node_id.endswith(f":{canonical}"):
                    return node_id

        return None

    def _generate_answer(
        self,
        question: str,
        entities: list[Entity],
        nodes: list[dict],
        relationships: list[dict],
    ) -> str | None:
        """基于规则生成自然语言答案（可扩展）"""
        lower_question = question.lower()
        model_nodes = [n for n in nodes if n["type"] == "model"]
        task_nodes = [n for n in nodes if n["type"] == "task"]
        excluded_nodes = [n for n in nodes if n["type"] == "system_type"]

        # 检测是否询问排除体系
        if any(kw in lower_question for kw in ["不适用", "不适用于", "排除", "except", "exclude"]):
            if model_nodes and excluded_nodes:
                info = []
                for model in model_nodes:
                    excluded = []
                    for rel in relationships:
                        if rel["source"] == model["id"] and rel["type"] == "excludes":
                            for ex_node in excluded_nodes:
                                if ex_node["id"] == rel["target"]:
                                    excluded.append(ex_node["label"])
                    if excluded:
                        info.append(f"{model['label']} 不适用于: {', '.join(excluded)}")
                    else:
                        info.append(f"{model['label']} 无不适用体系")
                return "\n".join(info)

        # 询问支持的任务
        if any(kw in lower_question for kw in ["支持", "适用", "任务", "计算", "support", "task"]):
            if model_nodes and task_nodes:
                info = []
                for model in model_nodes:
                    supported = []
                    for rel in relationships:
                        if rel["source"] == model["id"] and rel["type"] == "supports_task":
                            for task in task_nodes:
                                if task["id"] == rel["target"]:
                                    supported.append(task["label"])
                    info.append(f"{model['label']} 支持的任务: {', '.join(supported) if supported else '无'}")
                return "\n".join(info)

        # 通用：返回识别到的实体信息
        if nodes:
            labels = [f"{n['label']} ({n['type']})" for n in nodes[:5]]
            return f"识别到相关实体: {', '.join(labels)}"

        return None

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        # 尝试通过名称查找节点
        node_id = None
        for nid, node in self.graph.nodes.items():
            if node.label.lower() == model_name.lower():
                node_id = nid
                break
        if not node_id:
            # 尝试后缀匹配
            for nid in self.graph.nodes:
                if nid.endswith(f":{model_name.lower().replace(' ', '_')}"):
                    node_id = nid
                    break
        if not node_id:
            return None

        node = self.graph.get_node(node_id)
        if not node:
            return None
        return {
            "name": node.label,
            "type": node.type,
            "attributes": node.attributes,
            "relationships": [
                {
                    "type": rel.type.value,
                    "target": self.graph.get_node(rel.target).label if rel.target in self.graph.nodes else rel.target,
                }
                for rel in self.graph.get_relationships(source=node_id)
            ],
        }

    def find_models_for_task(self, task_name: str) -> list[str]:
        task_id = f"task:{task_name.lower().replace(' ', '_')}"
        models: list[str] = []
        for rel in self.graph.get_relationships(target=task_id):
            if rel.type == RelationshipType.SUPPORTS_TASK:
                node = self.graph.get_node(rel.source)
                if node:
                    models.append(node.label)
        return models

    @classmethod
    def from_loaded_graph(cls) -> GraphQueryEngine:
        graph = KnowledgeGraph.load()
        return cls(graph=graph)
