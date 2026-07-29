"""Graph builder that integrates with existing RAG knowledge base (YAML rules)."""

import logging
import yaml
from pathlib import Path
from typing import Any

from knowledge_graph.graph import KnowledgeGraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge"

logger = logging.getLogger(__name__)


class GraphBuilder:
    """从知识库 YAML 文件自动构建图谱"""

    def __init__(self):
        self.graph = KnowledgeGraph()

    def build_from_knowledge_base(self) -> KnowledgeGraph:
        """扫描 knowledge/ 下所有 YAML，抽取模型卡片并构建图谱"""
        yaml_files = list(KNOWLEDGE_ROOT.rglob("*.yaml")) + list(KNOWLEDGE_ROOT.rglob("*.yml"))
        if not yaml_files:
            logger.warning(f"在 {KNOWLEDGE_ROOT} 下未找到任何 YAML 文件")
            return self.graph

        model_cards = []
        for yaml_path in yaml_files:
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                cards = self._extract_model_cards(data, source=str(yaml_path.relative_to(KNOWLEDGE_ROOT)))
                if cards:
                    model_cards.extend(cards)
                    logger.info(f"从 {yaml_path.name} 抽取了 {len(cards)} 个模型卡片")
            except Exception as e:
                logger.error(f"解析 {yaml_path.name} 失败: {e}", exc_info=True)
                continue

        if not model_cards:
            logger.warning("未从知识库中提取到任何模型卡片，请检查 YAML 格式")
            return self.graph

        self.graph.build_from_model_cards(model_cards)
        return self.graph

    def _extract_model_cards(self, data: dict, source: str) -> list[dict]:
        """支持三种格式：单卡片、模型列表、规则列表"""
        cards = []

        # 格式1：单卡片 {"model_name": "...", ...}
        if "model_name" in data or "name" in data or "model" in data:
            return [self._normalize_card(data, source)]

        # 格式2：模型列表 {"models": [{"model_name": "..."}, ...]}
        for key in ["models", "cards", "model_list"]:
            if key in data and isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict) and ("model_name" in item or "name" in item or "model" in item):
                        cards.append(self._normalize_card(item, source))
                return cards

        # 格式3：规则列表 {"rules": [{"model": "..."}, ...]}
        if "rules" in data and isinstance(data["rules"], list):
            for rule in data["rules"]:
                if isinstance(rule, dict) and ("model" in rule or "model_name" in rule or "name" in rule):
                    cards.append(self._normalize_card(rule, source))
            return cards

        return cards

    def _normalize_card(self, raw: dict, source: str) -> dict:
        return {
            "model_name": raw.get("model_name") or raw.get("name") or raw.get("model", "unknown"),
            "family": raw.get("family", "unknown"),
            "implementation_status": raw.get("implementation_status") or raw.get("status", "unknown"),
            "pressure_regime": raw.get("pressure_regime") or raw.get("pressure", []),
            "supported_tasks": raw.get("supported_tasks") or raw.get("tasks") or raw.get("applicable", []),
            "excluded_systems": raw.get("excluded_systems") or raw.get("exclude") or raw.get("not_applicable", []),
            "_source": source,
        }


def build_graph_from_kb() -> KnowledgeGraph:
    """便捷函数：一键从知识库构建图谱"""
    builder = GraphBuilder()
    graph = builder.build_from_knowledge_base()
    graph.save()
    logger.info(f"图谱保存成功！节点数: {len(graph.nodes)}, 关系数: {len(graph.relationships)}")
    return graph