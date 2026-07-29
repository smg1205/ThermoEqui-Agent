"""Base class for knowledge skills with LLM support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from .llm_client import LLMClient, get_default_llm


@dataclass(frozen=True)
class SkillResult:
    """技能执行结果"""
    answer: str
    sources: list[str]
    confidence: float
    metadata: dict[str, Any] | None = None


class KnowledgeSkill(ABC):
    """知识技能抽象基类，支持可插拔 LLM"""

    def __init__(self, llm: Optional[LLMClient] = None, temperature: float = 0.3) -> None:
        self._llm = llm or get_default_llm()
        self._llm_available = self._llm is not None
        self._temperature = temperature  # 允许每个技能设置不同的温度

    @abstractmethod
    def name(self) -> str:
        """技能唯一名称"""
        ...

    @abstractmethod
    def description(self) -> str:
        """技能描述（供 LLM 或用户理解）"""
        ...

    @abstractmethod
    def execute(self, query: str, **kwargs) -> SkillResult:
        """执行技能，返回结构化结果"""
        ...

    def supports_intent(self, intent: str) -> bool:
        """检查技能是否支持特定意图（默认不支持）"""
        return False

    def _synthesize_with_llm(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        使用 LLM 合成答案，若不可用则降级为简单拼接

        Args:
            query: 用户问题
            context: 知识库上下文
            system_prompt: 系统提示词（如不传则使用通用优化模板）
            temperature: 温度参数，如不传则使用实例化时的值
        """
        if not self._llm_available or self._llm is None:
            return f"📚 根据知识库内容：\n\n{context}"

        # 使用传入的 temperature 或实例化时的默认值
        temp = temperature if temperature is not None else self._temperature

        # 通用优化系统提示词（如果外部没有传入）
        if system_prompt is None:
            system_prompt = (
                "你是 ThermoEqui-Agent 的热力学专家助手。\n\n"
                "## 任务\n"
                "根据下方知识库上下文，准确回答用户的问题。\n\n"
                "## 约束\n"
                "1. 只基于提供的上下文回答，不添加外部知识\n"
                "2. 如果上下文不充分，请明确说明「⚠️ 当前知识库信息有限，以下答案可能不完整」\n"
                "3. 回答长度控制在 150-300 字\n"
                "4. 涉及多个要点时用编号（1. 2. 3.）列出\n"
                "5. 如果涉及模型推荐，简要说明推荐理由\n"
                "6. 涉及参数数值时，保留小数点后 3 位\n\n"
                "## 输出格式\n"
                "直接回答问题，不需要重复用户的问题。\n"
                "如果引用来源，在句末标注 [来源: 文件名]"
            )

        user_prompt = f"""## 知识库上下文
---
{context}
---

## 用户问题
{query}

## 回答
请基于上述知识库上下文回答用户问题："""

        try:
            return self._llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temp,
            )
        except Exception as e:
            # LLM 调用失败时降级
            return f"📚 根据知识库内容：\n\n{context}\n\n（注：LLM合成失败，已降级为原始内容）"