"""Document loading utilities for knowledge base files."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

# 允许通过环境变量 KNOWLEDGE_ROOT 指定知识库根目录
KNOWLEDGE_ROOT = Path(os.getenv("KNOWLEDGE_ROOT", Path(__file__).resolve().parents[1] / "knowledge"))

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    content: str
    source: str
    category: str
    title: str | None = None
    metadata: dict[str, str] | None = None


class DocumentLoader(Protocol):
    def load(self, path: Path) -> list[Document]: ...


class MarkdownLoader:
    @staticmethod
    def _extract_title(content: str) -> str | None:
        match = re.match(r"^#\s+(.+)$", content, re.MULTILINE)
        return match.group(1).strip() if match else None

    def load(self, path: Path) -> list[Document]:
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            logger.warning(f"跳过无法读取的文件 {path}: {e}")
            return []
        category = path.parent.name
        title = self._extract_title(content)
        return [
            Document(
                content=content,
                source=str(path.relative_to(KNOWLEDGE_ROOT)),
                category=category,
                title=title,
                metadata={"file_type": "markdown"},
            )
        ]


class YamlLoader:
    @staticmethod
    def _format_yaml_content(data: dict) -> str:
        """将结构化 YAML 数据转换为自然语言描述，保留所有字段信息"""
        lines = []
        field_mappings = {
            "model_name": "模型名称",
            "family": "模型家族",
            "description": "描述",
            "type": "类型",
            "name": "名称",
            "implementation_status": "实现状态",
            "requires_binary_parameters": "需要二元参数",
            "pressure_regime": "压力范围",
            "supported_tasks": "支持的任务",
            "excluded_systems": "不适用体系",
            "validation_requirements": "验证要求",
            "parameters": "参数",
        }
        
        for key, display_name in field_mappings.items():
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    value_str = "、".join(str(v) for v in value)
                elif isinstance(value, bool):
                    value_str = "是" if value else "否"
                elif isinstance(value, dict):
                    value_str = "，".join(f"{k}={v}" for k, v in value.items())
                else:
                    value_str = str(value)
                lines.append(f"{display_name}：{value_str}")
        
        if not lines:
            return yaml.dump(data, default_flow_style=False, allow_unicode=True)
        
        return "；".join(lines)

    def load(self, path: Path) -> list[Document]:
        try:
            raw = path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw)
        except (yaml.YAMLError, UnicodeDecodeError, OSError) as e:
            logger.warning(f"跳过无法解析的 YAML 文件 {path}: {e}")
            return []
        category = path.parent.name
        title = data.get("model_name") or data.get("name") or path.stem
        content = self._format_yaml_content(data)
        return [
            Document(
                content=content,
                source=str(path.relative_to(KNOWLEDGE_ROOT)),
                category=category,
                title=str(title),
                metadata={"file_type": "yaml"},
            )
        ]


class KnowledgeLoader:
    _loaders: dict[str, DocumentLoader] = {
        ".md": MarkdownLoader(),
        ".yaml": YamlLoader(),
        ".yml": YamlLoader(),
    }

    @classmethod
    def load_all(cls) -> list[Document]:
        documents: list[Document] = []
        for path in sorted(KNOWLEDGE_ROOT.rglob("*")):
            if path.is_file() and path.suffix in cls._loaders:
                loader = cls._loaders[path.suffix]
                documents.extend(loader.load(path))
        return documents

    @classmethod
    def load_category(cls, category: str) -> list[Document]:
        category_path = KNOWLEDGE_ROOT / category
        if not category_path.exists():
            return []
        documents: list[Document] = []
        for path in sorted(category_path.rglob("*")):
            if path.is_file() and path.suffix in cls._loaders:
                loader = cls._loaders[path.suffix]
                documents.extend(loader.load(path))
        return documents