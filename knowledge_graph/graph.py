"""Knowledge graph implementation with nodes and relationships."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

GRAPH_STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "knowledge_graph.json"


class RelationshipType(StrEnum):
    USES_MODEL = "uses_model"
    HAS_PROPERTY = "has_property"
    SUPPORTS_TASK = "supports_task"
    REQUIRES_PARAMETER = "requires_parameter"
    HAS_RELATIONSHIP = "has_relationship"
    BELONGS_TO = "belongs_to"
    DESCRIBES = "describes"
    APPLIES_TO = "applies_to"
    # ========== P0/P1修复：新增明确的排除关系 ==========
    EXCLUDES = "excludes"


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    type: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Relationship:
    source: str
    target: str
    type: RelationshipType
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    relationships: list[Relationship] = field(default_factory=list)

    # ========== P0修复：生成带类型前缀的唯一ID ==========
    @staticmethod
    def _make_id(prefix: str, name: str) -> str:
        """生成带命名空间前缀的节点ID，彻底避免ID冲突"""
        return f"{prefix}:{name.lower().replace(' ', '_')}"

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_relationship(self, relationship: Relationship) -> None:
        if relationship.source not in self.nodes:
            raise ValueError(f"Source node {relationship.source} not found")
        if relationship.target not in self.nodes:
            raise ValueError(f"Target node {relationship.target} not found")
        self.relationships.append(relationship)

    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def get_relationships(self, source: str | None = None, target: str | None = None) -> list[Relationship]:
        result = []
        for rel in self.relationships:
            if source is not None and rel.source != source:
                continue
            if target is not None and rel.target != target:
                continue
            result.append(rel)
        return result

    def neighbors(self, node_id: str) -> list[tuple[Node, Relationship]]:
        neighbors: list[tuple[Node, Relationship]] = []
        for rel in self.relationships:
            if rel.source == node_id and rel.target in self.nodes:
                neighbors.append((self.nodes[rel.target], rel))
            elif rel.target == node_id and rel.source in self.nodes:
                neighbors.append((self.nodes[rel.source], rel))
        return neighbors

    def save(self, path: Path | None = None) -> None:
        save_path = path or GRAPH_STORE_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "type": node.type,
                    "attributes": node.attributes,
                }
                for node in self.nodes.values()
            ],
            "relationships": [
                {
                    "source": rel.source,
                    "target": rel.target,
                    "type": rel.type.value,
                    "attributes": rel.attributes,
                }
                for rel in self.relationships
            ],
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path | None = None) -> KnowledgeGraph:
        load_path = path or GRAPH_STORE_PATH
        if not load_path.exists():
            return cls()
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = cls()
        for node_data in data.get("nodes", []):
            graph.add_node(
                Node(
                    id=node_data["id"],
                    label=node_data["label"],
                    type=node_data["type"],
                    attributes=node_data.get("attributes", {}),
                )
            )
        for rel_data in data.get("relationships", []):
            try:
                rel_type = RelationshipType(rel_data["type"])
            except ValueError:
                continue
            graph.add_relationship(
                Relationship(
                    source=rel_data["source"],
                    target=rel_data["target"],
                    type=rel_type,
                    attributes=rel_data.get("attributes", {}),
                )
            )
        return graph

    def build_from_model_cards(self, model_cards: list[dict[str, Any]]) -> None:
        for card in model_cards:
            # ========== P0修复：使用前缀ID ==========
            model_id = self._make_id("model", card["model_name"])
            self.add_node(
                Node(
                    id=model_id,
                    label=card["model_name"],
                    type="model",
                    attributes={
                        "family": card.get("family"),
                        "requires_binary_parameters": card.get("requires_binary_parameters"),
                        "implementation_status": card.get("implementation_status"),
                        "pressure_regime": card.get("pressure_regime", []),
                    },
                )
            )
            for task in card.get("supported_tasks", []):
                task_id = self._make_id("task", task)
                if task_id not in self.nodes:
                    self.add_node(
                        Node(
                            id=task_id,
                            label=task,
                            type="task",
                        )
                    )
                self.add_relationship(
                    Relationship(
                        source=model_id,
                        target=task_id,
                        type=RelationshipType.SUPPORTS_TASK,
                    )
                )
            for excluded in card.get("excluded_systems", []):
                # ========== P0/P1修复：使用前缀ID + 正确的 EXCLUDES 关系 ==========
                excluded_id = self._make_id("system_type", excluded)
                if excluded_id not in self.nodes:
                    self.add_node(
                        Node(
                            id=excluded_id,
                            label=excluded,
                            type="system_type",
                        )
                    )
                self.add_relationship(
                    Relationship(
                        source=model_id,
                        target=excluded_id,
                        type=RelationshipType.EXCLUDES,  # 不再是 BELONGS_TO
                        attributes={"reason": "excluded"},
                    )
                )
    # ... 保留原有代码不变，在类末尾增加：
    def export_graphml(self, path: Path) -> None:
        """导出为 GraphML 格式，便于在可视化工具中查看"""
        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        root = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
        # 定义属性
        keys = [
            ("label", "string"),
            ("type", "string"),
            ("family", "string"),
            ("implementation_status", "string"),
            ("pressure_regime", "string"),
        ]
        for key_name, key_type in keys:
            key = ET.SubElement(root, "key", id=key_name, for_="node", attr_name=key_name, attr_type=key_type)

        graph_elem = ET.SubElement(root, "graph", id="G", edgedefault="directed")

        # 添加节点
        for node in self.nodes.values():
            node_elem = ET.SubElement(graph_elem, "node", id=node.id)
            for attr_name in ["label", "type"]:
                data = ET.SubElement(node_elem, "data", key=attr_name)
                data.text = getattr(node, attr_name, "")
            # attributes 字典中的字段
            if node.attributes:
                family = node.attributes.get("family", "")
                if family:
                    data = ET.SubElement(node_elem, "data", key="family")
                    data.text = family
                status = node.attributes.get("implementation_status", "")
                if status:
                    data = ET.SubElement(node_elem, "data", key="implementation_status")
                    data.text = status
                pressure = node.attributes.get("pressure_regime", [])
                if pressure:
                    data = ET.SubElement(node_elem, "data", key="pressure_regime")
                    data.text = ", ".join(pressure)

        # 添加边
        for rel in self.relationships:
            edge = ET.SubElement(graph_elem, "edge", source=rel.source, target=rel.target)
            # 可以添加 label 属性
            label = ET.SubElement(edge, "data", key="label")
            label.text = rel.type.value

        # 格式化输出
        xml_str = ET.tostring(root, encoding="utf-8")
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")
        with open(path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)