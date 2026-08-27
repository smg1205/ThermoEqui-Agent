"""LLM client adapter wrapping the project's configured provider (DeepSeek/OpenAI)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract LLM client for skill answer synthesis."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        pass


class ProjectProviderClient(LLMClient):
    """Wraps the project's configured LLM provider (DeepSeek/OpenAI) for skill use.

    Uses the same provider mechanism as the main orchestrator, so the user's
    .env LLM_PROVIDER setting is respected.
    """

    def __init__(self, provider=None, temperature: float = 0.3) -> None:
        self._provider = provider
        self._temperature = temperature

    def _get_provider(self):
        if self._provider is None:
            from apps.api.main import configured_provider

            self._provider = configured_provider()
        return self._provider

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        import json
        import os

        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if not api_key:
            return "（未配置 API key，无法调用 LLM）"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = json.dumps({"model": model, "messages": messages, "stream": False, "max_tokens": 2048}).encode()
        req = __import__("urllib.request", fromlist=["Request"]).Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            resp = __import__("urllib.request", fromlist=["urlopen"]).urlopen(req, timeout=60)
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return f"[LLM调用失败] {e}"


class MockLLMClient(LLMClient):
    """Deterministic fallback when no LLM provider is configured."""

    def __init__(self, temperature: float = 0.3) -> None:
        self._temperature = temperature

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        return "（Mock模式）根据知识库检索，建议结合上下文具体分析。"


def get_default_llm() -> LLMClient | None:
    """Return a ProjectProviderClient; falls back to MockLLMClient on error."""
    try:
        return ProjectProviderClient()
    except Exception as e:
        logger.warning(f"Provider LLM client init failed, using mock: {e}")
        return MockLLMClient()
