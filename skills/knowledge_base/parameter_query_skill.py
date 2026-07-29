"""Parameter query skill for thermodynamic model parameters with CoT and Few-shot."""

from __future__ import annotations

import logging
import re

import yaml
from rag.retriever import KnowledgeRetriever, RetrievedChunk
from .skill_base import KnowledgeSkill, SkillResult
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class ParameterQuerySkill(KnowledgeSkill):
    """参数查询技能：从知识库提取模型参数信息"""

    MODEL_NAME_MAP = {
        "srk": "Soave-Redlich-Kwong",
        "pr": "Peng-Robinson",
        "nrtl": "NRTL",
        "wilson": "Wilson",
        "uniquac": "UNIQUAC",
        "ideal": "Ideal/Raoult",
        "raoult": "Ideal/Raoult",
    }

    def __init__(
        self,
        retriever: KnowledgeRetriever | None = None,
        llm: LLMClient | None = None,
        temperature: float = 0.1,
    ) -> None:
        super().__init__(llm=llm, temperature=temperature)
        self._retriever = retriever
        self._retriever_loaded = False

    def _get_retriever(self) -> KnowledgeRetriever:
        if not self._retriever_loaded:
            if self._retriever is None:
                from rag.retriever import DEFAULT_RETRIEVER
                self._retriever = DEFAULT_RETRIEVER
            self._retriever_loaded = True
        return self._retriever

    def name(self) -> str:
        return "parameter_query"

    def description(self) -> str:
        return (
            "Extract and summarize thermodynamic model parameters, constants, "
            "and application ranges from the knowledge base."
        )

    def _extract_model_name_from_query(self, query: str) -> tuple[str | None, str | None]:
        query_lower = query.lower()
        for short_name, full_name in self.MODEL_NAME_MAP.items():
            if short_name in query_lower or full_name.lower() in query_lower:
                return short_name, full_name
        return None, None

    def _parse_yaml_card(self, content: str) -> dict | None:
        """解析卡片内容（支持原始 YAML 和格式化文本两种格式）"""
        # 尝试原始 YAML 解析
        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict) and ("model_name" in data or "name" in data):
                return data
        except yaml.YAMLError:
            pass
        
        # 尝试解析格式化文本："模型名称：xxx；模型家族：xxx；..."
        if "：" in content or ":" in content:
            result = {}
            # 按分号分割字段
            parts = re.split(r'[;；]', content)
            field_mappings = {
                "模型名称": "model_name",
                "模型家族": "family",
                "实现状态": "implementation_status",
                "需要二元参数": "requires_binary_parameters",
                "压力范围": "pressure_regime",
                "支持的任务": "supported_tasks",
                "不适用体系": "excluded_systems",
                "验证要求": "validation_requirements",
                "描述": "description",
                "类型": "type",
                "名称": "name",
                "参数": "parameters",
            }
            
            for part in parts:
                part = part.strip()
                for cn_name, en_name in field_mappings.items():
                    if part.startswith(cn_name + "：") or part.startswith(cn_name + ":"):
                        value = part[len(cn_name) + 1:].strip() if cn_name + "：" in part or cn_name + ":" in part else ""
                        # 转换值
                        if value in ["是", "true", "True"]:
                            result[en_name] = True
                        elif value in ["否", "false", "False"]:
                            result[en_name] = False
                        elif "、" in value:
                            result[en_name] = [v.strip() for v in value.split("、")]
                        else:
                            result[en_name] = value
                        break
            
            if result and ("model_name" in result or "name" in result):
                return result
        
        return None

    def _format_card_answer(self, card_data: dict, query: str) -> str:
        """将解析的卡片数据格式化为答案"""
        model_name = card_data.get("model_name") or card_data.get("name") or "未知模型"
        
        # 检查用户是否询问特定字段
        specific_field = self._detect_specific_field(query)
        
        if specific_field:
            value = card_data.get(specific_field)
            if value is not None:
                field_display = self._get_field_display_name(specific_field)
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                elif isinstance(value, bool):
                    value_str = "是" if value else "否"
                else:
                    value_str = str(value)
                return f"**{model_name}** 的{field_display}: {value_str}"
            else:
                field_display = self._get_field_display_name(specific_field)
                return f"⚠️ 未找到 {model_name} 的{field_display}"
        
        # 返回完整参数信息
        lines = [f"**【{model_name}】**"]
        
        field_mappings = [
            ("family", "模型家族"),
            ("supported_tasks", "支持的任务"),
            ("excluded_systems", "不适用体系"),
            ("requires_binary_parameters", "需要二元参数"),
            ("pressure_regime", "压力范围"),
            ("implementation_status", "实现状态"),
        ]
        
        for field_key, display_name in field_mappings:
            value = card_data.get(field_key)
            if value is None:
                lines.append(f"- {display_name}: ⚠️ 未找到")
            elif isinstance(value, list):
                lines.append(f"- {display_name}: {', '.join(str(v) for v in value)}")
            elif isinstance(value, bool):
                lines.append(f"- {display_name}: {'是' if value else '否'}")
            else:
                lines.append(f"- {display_name}: {value}")
        
        return "\n".join(lines)

    def _detect_specific_field(self, query: str) -> str | None:
        """检测用户是否询问特定字段"""
        field_keywords = {
            "family": ["模型家族", "family", "类型", "type"],
            "supported_tasks": ["支持的任务", "supported_tasks", "task", "任务", "适用"],
            "excluded_systems": ["不适用体系", "excluded", "排除", "不适用于", "不能用于"],
            "requires_binary_parameters": ["二元参数", "binary", "需要参数", "参数"],
            "pressure_regime": ["压力范围", "pressure", "压力", "pressure_regime"],
            "implementation_status": ["实现状态", "implementation_status", "状态", "status"],
        }
        
        query_lower = query.lower()
        for field, keywords in field_keywords.items():
            for kw in keywords:
                if kw.lower() in query_lower:
                    return field
        return None

    def _get_field_display_name(self, field_key: str) -> str:
        """获取字段的中文显示名"""
        field_names = {
            "family": "模型家族",
            "supported_tasks": "支持的任务",
            "excluded_systems": "不适用体系",
            "requires_binary_parameters": "需要二元参数",
            "pressure_regime": "压力范围",
            "implementation_status": "实现状态",
        }
        return field_names.get(field_key, field_key)

    def _force_retrieve_model_card(self, short_name: str, full_name: str) -> list[RetrievedChunk]:
        """强制加载指定模型的卡片"""
        from rag.loader import KnowledgeLoader

        all_docs = KnowledgeLoader.load_all()

        target_filenames = [
            f"{full_name.lower().replace(' ', '-')}.yaml",
            f"{short_name.lower()}.yaml",
        ]

        model_card_docs = [d for d in all_docs if "model_cards" in d.source.lower().replace("\\", "/")]

        target_docs = []

        for doc in model_card_docs:
            source_lower = doc.source.lower().replace("\\", "/")
            for fname in target_filenames:
                if source_lower.endswith(f"/{fname}") or source_lower.endswith(f"\\{fname}"):
                    target_docs.append(doc)
                    break
            if target_docs:
                break

        if not target_docs:
            patterns = [
                rf"model_name\s*[：:]\s*{re.escape(full_name)}",
                rf"模型名称\s*[：:]\s*{re.escape(full_name)}",
                rf"model_name\s*[：:]\s*{re.escape(short_name)}",
            ]
            for doc in model_card_docs:
                for pat in patterns:
                    if re.search(pat, doc.content, re.IGNORECASE):
                        target_docs.append(doc)
                        break
                if target_docs:
                    break

        if not target_docs:
            for doc in all_docs:
                source_lower = doc.source.lower().replace("\\", "/")
                for fname in target_filenames:
                    if source_lower.endswith(f"/{fname}") or source_lower.endswith(f"\\{fname}"):
                        target_docs.append(doc)
                        break
                if target_docs:
                    break

        if not target_docs:
            logger.warning(f"未找到模型卡片: {full_name} (简称: {short_name})")
            return []

        result = []
        for doc in target_docs:
            chunk = RetrievedChunk(
                content=doc.content,
                source=doc.source,
                category="model_cards",
                similarity=0.05
            )
            result.append(chunk)

        return result

    def execute(self, query: str, **kwargs) -> SkillResult:
        try:
            retriever = self._get_retriever()
            chunks = retriever.retrieve_with_keywords(query, top_k=5)
        except Exception as e:
            return SkillResult(
                answer=f"参数查询失败: {e}",
                sources=[],
                confidence=0.0,
            )

        short_name, full_name = self._extract_model_name_from_query(query)

        if short_name and full_name:
            # 强制加载模型卡片
            card_chunks = self._force_retrieve_model_card(short_name, full_name)
            
            if card_chunks:
                # 直接解析 YAML 卡片
                for chunk in card_chunks:
                    card_data = self._parse_yaml_card(chunk.content)
                    if card_data:
                        answer = self._format_card_answer(card_data, query)
                        return SkillResult(
                            answer=answer,
                            sources=[chunk.source],
                            confidence=0.9,
                            metadata={
                                "method": "yaml_parse",
                                "detected_model_short": short_name,
                                "detected_model_full": full_name,
                            },
                        )

        # 降级方案：从检索内容中尝试解析
        if chunks:
            for chunk in chunks:
                if chunk.category == "model_cards":
                    card_data = self._parse_yaml_card(chunk.content)
                    if card_data:
                        answer = self._format_card_answer(card_data, query)
                        return SkillResult(
                            answer=answer,
                            sources=[chunk.source],
                            confidence=0.8,
                            metadata={
                                "method": "yaml_parse_from_retrieval",
                            },
                        )

        # 最终降级：使用 LLM 处理非结构化内容
        if not chunks:
            return SkillResult(
                answer="知识库中未找到相关参数信息。",
                sources=[],
                confidence=0.0,
            )

        card_chunks_retrieved = [c for c in chunks if c.category == "model_cards"]
        other_chunks = [c for c in chunks if c.category != "model_cards"]

        final_chunks = card_chunks_retrieved + other_chunks
        final_chunks = final_chunks[:8]

        context_parts = []
        sources = []
        for chunk in final_chunks:
            source_label = chunk.source.replace("\\", "/")
            context_parts.append(f"[来源: {source_label}]\n{chunk.content}")
            sources.append(chunk.source)

        combined_context = "\n\n---\n\n".join(context_parts)

        system_prompt = f"""你是 ThermoEqui-Agent 的热力学模型参数专家。

## 重要说明
知识库中的模型卡片以 YAML 格式存储，字段名可能是英文或中文。

## 任务
从上下文中找到对应模型的卡片，提取用户询问的参数。

## 输出格式
直接回答用户的问题，不需要重复用户的问题。

用户正在询问关于 **{full_name or '未知模型'}** 模型的参数信息。
"""

        answer = self._synthesize_with_llm(
            query=query,
            context=combined_context,
            system_prompt=system_prompt,
            temperature=self._temperature,
        )

        return SkillResult(
            answer=answer,
            sources=list(set(sources)),
            confidence=0.5,
            metadata={
                "chunks_retrieved": len(chunks),
                "sources": sources,
                "llm_used": self._llm_available,
                "temperature": self._temperature,
                "detected_model_short": short_name,
                "detected_model_full": full_name,
            },
        )

    def supports_intent(self, intent: str) -> bool:
        return intent in {"CONCEPT_QA", "MODEL_SELECTION_QA"}
