"""Isolated LLM provider interfaces and implementations."""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

from schemas.domain import EvidenceStatement, Intent, TaskManifest


class LLMProvider(Protocol):
    async def classify_intent(self, message: str) -> Intent: ...

    async def formulate_task(self, message: str, previous: TaskManifest | None = None) -> TaskManifest | None: ...

    async def select_tool(
        self,
        message: str,
        task: TaskManifest,
        available_tools: list[dict[str, str]],
    ) -> str: ...

    async def answer_with_evidence(
        self,
        message: str,
        strict: bool = False,
        grounded_numbers: set[str] | None = None,
        *,
        intent_label: str | None = None,
    ) -> list[EvidenceStatement]: ...

    async def interpret_result(self, result: dict[str, object]) -> list[EvidenceStatement]: ...


class LLMProviderError(RuntimeError):
    """Sanitized external-provider failure safe for API responses and logs."""

    def __init__(self, provider: str, status_code: int | None = None) -> None:
        self.provider = provider
        self.status_code = status_code
        detail = f" with status {status_code}" if status_code is not None else ""
        super().__init__(f"{provider} API request failed{detail}.")


class LLMProviderOutputError(RuntimeError):
    """External provider returned content that violates the expected contract."""


_NUMERIC_TOKEN = re.compile(r"(?<![\w.])[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?(?![\w.])")
_EXTERNAL_REFERENCE = re.compile(
    r"https?://|doi\s*:|10\.\d{4,9}/|NIST|Chemistry WebBook|according to|"
    r"\bsource\b|\bcitation\b|\breference\b|\bbibliography\b|\bet al\.?|文献|来源|参考|研究表明|数据库",
    re.IGNORECASE,
)
_WITHHELD_TEXT = "外部模型输出因包含未经确定性工具证实的数值或引用而被扣留。"
_INTENT_VALUES = tuple(intent.value for intent in Intent)
_INTENT_INSTRUCTIONS = (
    "Classify the user message as exactly one ThermoEqui intent. "
    f"Allowed values: {', '.join(_INTENT_VALUES)}. "
    "Return only the enum value with no prose, Markdown, or JSON. Never return NONE."
)
_FENCED_VALUE = re.compile(r"^```(?:json|text)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)


class _OpenAIContentPart(BaseModel):
    type: str
    text: str | None = None


class _OpenAIOutput(BaseModel):
    content: list[_OpenAIContentPart] = Field(default_factory=list)


class _OpenAIResponse(BaseModel):
    output: list[_OpenAIOutput] = Field(default_factory=list)


class _DeepSeekMessage(BaseModel):
    content: str


class _DeepSeekChoice(BaseModel):
    message: _DeepSeekMessage


class _DeepSeekChatCompletion(BaseModel):
    choices: list[_DeepSeekChoice] = Field(min_length=1)


class _ToolSelection(BaseModel):
    tool_name: str


def _system_proxy_url() -> str | None:
    proxies = urllib.request.getproxies()
    return proxies.get("https") or proxies.get("http")


_THERMO_QA_KEYWORDS: frozenset[str] = frozenset(
    {
        "相平衡",
        "气液",
        "液液",
        "vle",
        "lle",
        "flash",
        "泡点",
        "露点",
        "共沸",
        "热力学",
        "活度",
        "逸度",
        "相图",
        "antoine",
        "nrtl",
        "wilson",
        "uniquac",
        "peng",
        "raoult",
        "ideal",
        "benzene",
        "toluene",
        "ethanol",
        "acetone",
        "methanol",
        "苯",
        "甲苯",
        "乙醇",
        "丙酮",
        "甲醇",
        "phase equilibrium",
        "bubble point",
        "dew point",
        "azeotrope",
        "thermodynamic",
        "activity coefficient",
        "fugacity",
        "equation of state",
        "binary interaction",
    }
)

_CALCULATION_KEYWORDS: frozenset[str] = frozenset(
    {
        "calc",
        "compute",
        "simulate",
        "求算",
        "算出",
        "推算",
        "T-x-y",
        "P-x-y",
    }
)
_CALCULATION_REQUEST_PATTERN = re.compile(
    r"(?:请|帮我|请帮我|试|试着)?\s*(?:计算(?:一下|一下下)?|算一算|求解|试算|求(?:解|算)?|推算)"
)
_NON_REQUEST_CALCULATION_PREFIX = re.compile(
    r"(?:模型|方程|公式|理论|方法|系统|体系|过程)的?\s*(?:计算|算)|"
    r"(?:经过|经|通过|由|用)\s*(?:模型|理论|公式)?\s*(?:计算|推算)|"
    r"(?:计算|推算)(?:得到|得出|显示|表明|结果|可知|可见)"
)

#: Property keywords whose value can only come from a deterministic backend.
_GAMMA_INFINITY_KEYWORDS: frozenset[str] = frozenset(
    {
        "无限稀释",
        "无限稀",
        "γ∞",
        "γinf",
        "gamma infinity",
        "gamma-infinity",
        "gamma_inf",
        "infinite dilution",
    }
)

#: "how much is X" question forms that request a numeric value.
_NUMERIC_QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "是多少",
        "多少",
        "数值",
        "值是多少",
        "为多少",
    }
)


def _is_thermo_question(question: str) -> bool:
    lower = question.casefold()
    return any(kw.casefold() in lower for kw in _THERMO_QA_KEYWORDS)


_JUDGMENT_QUESTION_PATTERN = re.compile(
    r"(?:可以|能|应该|是否|适不适合|合不合适|能不能|可不可以)"
    r".*(?:计算|适用|可行|使用|采用)"
    r".*(?:吗|呢|？|\?)"
)
_MODEL_SUITABILITY_PATTERN = re.compile(
    r"(?:可以用|能用|适用|合适|应该选|应该用|选什么)"
    r".*(?:定律|模型|方程|方法)"
)


def _is_calculation_question(question: str) -> bool:
    lower = question.casefold()
    if any(kw.casefold() in lower for kw in _GAMMA_INFINITY_KEYWORDS):
        return True
    if _JUDGMENT_QUESTION_PATTERN.search(lower):
        return False
    if _MODEL_SUITABILITY_PATTERN.search(lower) and ("吗" in lower or "呢" in lower or "?" in lower or "？" in lower):
        return False
    if any(kw.casefold() in lower for kw in _CALCULATION_KEYWORDS):
        return True
    if _CALCULATION_REQUEST_PATTERN.search(question):
        if not _NON_REQUEST_CALCULATION_PREFIX.search(question):
            return True
    if any(word in lower for word in _NUMERIC_QUESTION_WORDS) and _is_thermo_question(question):
        return True
    return False


def _contains_ungrounded_claim(
    text: str,
    check_numbers: bool = True,
    grounded_numbers: set[str] | None = None,
) -> bool:
    """Only check external references for calculation questions.

    Concept Q&A (e.g. 'what is activity coefficient') should allow
    mentions of Lewis, IUPAC, textbooks, etc. These are legitimate
    knowledge references, not fabricated data. Only calculation tasks
    must be strict: any number or citation not from thermo_engine is
    potentially fabricated.

    When grounded_numbers is provided, those specific numeric strings
    (which come from prior validated calculation results via memory)
    are considered safe and will not trigger the ungrounded check.
    """
    grounded = grounded_numbers or set()
    if _EXTERNAL_REFERENCE.search(text):
        return True
    if not check_numbers:
        return False
    for match in _NUMERIC_TOKEN.finditer(text):
        value = match.group(0)
        if value in grounded:
            continue
        stripped = value.rstrip(".")
        if stripped in grounded:
            continue
        return True
    return False


def _normalize_intent_value(raw_value: str) -> str:
    value = raw_value.strip()
    fenced = _FENCED_VALUE.fullmatch(value)
    if fenced:
        value = fenced.group(1).strip()
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        decoded = value
    if isinstance(decoded, dict):
        decoded = decoded.get("intent")
    if not isinstance(decoded, str):
        raise LLMProviderOutputError("External provider returned an invalid intent value.")
    match = re.fullmatch(r"(?:intent\s*[:=]\s*)?['\"]?([A-Za-z_]+)['\"]?", decoded.strip(), re.IGNORECASE)
    if match is None:
        raise LLMProviderOutputError("External provider returned an invalid intent value.")
    return match.group(1).upper()


_TASK_MANIFEST_LIST_DEFAULTS: dict[str, list[str]] = {
    "requested_outputs": ["table", "validation"],
    "validation_requirements": ["composition_balance", "equilibrium_residual", "convergence"],
    "assumptions": [],
    "parameters": [],
}


def _normalize_task_manifest_payload(value: str) -> str:
    """Repair common LLM JSON habits without changing the public schema."""
    raw = value.strip()
    fenced = _FENCED_VALUE.fullmatch(raw)
    if fenced is not None:
        raw = fenced.group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return value
    if not isinstance(payload, dict):
        return value
    for key, default in _TASK_MANIFEST_LIST_DEFAULTS.items():
        if not isinstance(payload.get(key), list):
            payload[key] = default
    parameters = payload.get("parameters")
    if parameters not in (None, []):
        payload["assumptions"] = [
            *payload["assumptions"],
            (
                "External provider-supplied parameters were discarded; "
                "reviewed parameter sets must be created through the parameter API."
            ),
        ]
    payload["parameters"] = []
    if payload.get("composition_basis") is None:
        payload["composition_basis"] = "mole_fraction"
    if payload.get("points") is None:
        payload["points"] = 21
    if "task_id" in payload and payload["task_id"] is None:
        del payload["task_id"]
    components = payload.get("components")
    if isinstance(components, list):
        for component in components:
            if isinstance(component, dict) and component.get("aliases") is None:
                component["aliases"] = []
    return json.dumps(payload, ensure_ascii=False)


class ConstrainedLLMProvider:
    """Shared orchestration behavior; subclasses implement only their HTTP transport."""

    async def _request(
        self,
        instructions: str,
        message: str,
        *,
        json_mode: bool = False,
        max_tokens: int = 1024,
    ) -> str:
        raise NotImplementedError

    async def classify_intent(self, message: str) -> Intent:
        value = await self._request(
            _INTENT_INSTRUCTIONS,
            message,
            max_tokens=32,
        )
        try:
            return Intent(_normalize_intent_value(value))
        except ValueError:
            raise LLMProviderOutputError("External provider returned an invalid intent value.") from None

    async def formulate_task(self, message: str, previous: TaskManifest | None = None) -> TaskManifest | None:
        context = previous.model_dump_json() if previous else "null"
        schema = json.dumps(
            TaskManifest.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        instructions = (
            "Return only a TaskManifest JSON object or null. Follow the supplied JSON Schema exactly. "
            "model_name may be null so the deterministic router can select an applicable model. "
            "Never invent components, conditions, parameters, data, or citations. "
            "Never populate the parameters field; leave it as [] because parameter sets are managed "
            "separately through the parameter repository. "
            "Do not emit null for optional fields; omit them or use the schema defaults. "
            "Return plain JSON, not Markdown fenced code. "
            "Never calculate equilibrium numbers; deterministic tools do that after validation. "
            f"TaskManifest JSON Schema: {schema}. Previous manifest: {context}"
        )
        validation_fields: list[str] = []
        for attempt in range(2):
            retry_note = (
                ""
                if attempt == 0
                else " The previous response failed validation at these fields: "
                f"{', '.join(validation_fields)}. Return a complete corrected object."
            )
            value = await self._request(
                instructions + retry_note,
                message,
                json_mode=True,
                max_tokens=2048,
            )
            normalized_value = _normalize_task_manifest_payload(value)
            if normalized_value.strip() == "null":
                return None
            try:
                return TaskManifest.model_validate_json(normalized_value)
            except ValidationError as error:
                validation_fields = sorted(
                    {
                        ".".join(str(part) for part in issue["loc"])
                        for issue in error.errors(include_url=False, include_input=False)
                    }
                )
        raise LLMProviderOutputError("External provider returned an invalid task manifest.")

    async def answer_with_evidence(
        self,
        message: str,
        strict: bool = False,
        grounded_numbers: set[str] | None = None,
        *,
        intent_label: str | None = None,
    ) -> list[EvidenceStatement]:
        value = await self._request(
            "Answer concise thermodynamics knowledge questions without fabricating numerical data or citations. "
            "Do not cite any source. Prefix every paragraph with Knowledge:, Inference:, or Warning:.",
            message,
        )
        if strict and _contains_ungrounded_claim(value, grounded_numbers=grounded_numbers):
            return [EvidenceStatement(category="Warning", text=_WITHHELD_TEXT)]
        # Treat truly-calculation intents (EQUILIBRIUM_CALCULATION, TASK_CORRECTION)
        # as strict (check_numbers on ungrounded tokens).
        # Concept/compare/interpret intents (CONCEPT_QA, MODEL_SELECTION_QA,
        # RESULT_INTERPRETATION) are not calculation requests - they interpret
        # prior results, so derived numbers (differences, ratios) are acceptable.
        strict_calc_intents = {"EQUILIBRIUM_CALCULATION", "TASK_CORRECTION", "FLASH_CALCULATION"}
        relaxed_intents = {
            "CONCEPT_QA",
            "MODEL_SELECTION_QA",
            "RESULT_INTERPRETATION",
            "PROCESS_RECOMMENDATION",
            "SENSITIVITY_ANALYSIS",
            "FLOW_DESIGN_QA",
        }
        if intent_label and intent_label.upper() in strict_calc_intents:
            check_numbers = True
        elif intent_label and intent_label.upper() in relaxed_intents:
            check_numbers = False
        else:
            check_numbers = _is_thermo_question(message) and _is_calculation_question(message)
        if check_numbers:
            if _contains_ungrounded_claim(value, check_numbers=True, grounded_numbers=grounded_numbers):
                return [EvidenceStatement(category="Warning", text=_WITHHELD_TEXT)]
        if _contains_ungrounded_claim(value, grounded_numbers=grounded_numbers):
            value += "\n\n（以上涉及数值未经确定性引擎验证，请以计算结果为准。）"
        return [EvidenceStatement(category="Knowledge", text=value)]

    async def select_tool(
        self,
        message: str,
        task: TaskManifest,
        available_tools: list[dict[str, str]],
    ) -> str:
        allowed_names = {tool["name"] for tool in available_tools}
        selection_schema = json.dumps(
            _ToolSelection.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        planning_input = json.dumps(
            {
                "user_message": message,
                "task": task.model_dump(mode="json"),
                "available_tools": available_tools,
            },
            ensure_ascii=False,
        )
        value = await self._request(
            "Select exactly one allowlisted engineering tool for this task. "
            "Return only the JSON object; do not provide reasoning or calculate any result. "
            f"Selection JSON Schema: {selection_schema}",
            planning_input,
            json_mode=True,
            max_tokens=64,
        )
        try:
            selection = _ToolSelection.model_validate_json(value)
        except ValidationError:
            raise LLMProviderOutputError("External provider returned an invalid tool selection.") from None
        if selection.tool_name not in allowed_names:
            raise LLMProviderOutputError("External provider selected a tool outside the allowlist.")
        return selection.tool_name

    async def interpret_result(self, result: dict[str, object]) -> list[EvidenceStatement]:
        grounded_source = json.dumps(result, ensure_ascii=False)
        value = await self._request(
            "Interpret only the supplied tool JSON. Preserve failures and warnings; do not add numbers or citations.",
            grounded_source,
        )
        if _contains_ungrounded_claim(value):
            return [EvidenceStatement(category="Warning", text=_WITHHELD_TEXT)]
        return [EvidenceStatement(category="Inference", text=value)]


class OpenAIProvider(ConstrainedLLMProvider):
    """Optional provider using the official Responses API through a single adapter."""

    def __init__(self, api_key: str, model: str = "gpt-5-mini", timeout_seconds: float = 30.0) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def _request(
        self,
        instructions: str,
        message: str,
        *,
        json_mode: bool = False,
        max_tokens: int = 1024,
    ) -> str:
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": message,
            "max_output_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            try:
                response = await client.post("https://api.openai.com/v1/responses", json=payload, headers=headers)
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                raise LLMProviderError("OpenAI", error.response.status_code) from None
            except httpx.RequestError:
                raise LLMProviderError("OpenAI") from None
            try:
                completion = _OpenAIResponse.model_validate(response.json())
            except (json.JSONDecodeError, ValidationError):
                raise LLMProviderOutputError("External provider returned an invalid response envelope.") from None
        texts = [
            part.text
            for output in completion.output
            for part in output.content
            if part.type == "output_text" and part.text is not None
        ]
        content = "".join(texts)
        if not content.strip():
            raise LLMProviderOutputError("External provider returned empty assistant content.")
        return content


class DeepSeekProvider(ConstrainedLLMProvider):
    """DeepSeek Chat Completions adapter with the same constrained LLM role."""

    async def interpret_result(self, result: dict[str, object]) -> list[EvidenceStatement]:
        """Override: do not filter numbers since they come from the deterministic engine."""
        grounded_source = json.dumps(result, ensure_ascii=False)
        value = await self._request(
            "Interpret the tool result in natural language. "
            "Summarise the equilibrium phases, key compositions, and any warnings. "
            "Do not add external citations.",
            grounded_source,
        )
        return [EvidenceStatement(category="Inference", text=value)]

    async def answer_with_evidence(
        self,
        message: str,
        strict: bool = False,
        grounded_numbers: set[str] | None = None,
        *,
        intent_label: str | None = None,
    ) -> list[EvidenceStatement]:
        """Override: answer both thermodynamics knowledge and general questions."""
        value = await self._request(
            "Answer the user's question concisely and helpfully. "
            "If the question is about thermodynamics or phase equilibrium, answer with domain expertise. "
            "If the question is general, answer naturally. "
            "Do not cite external sources. Keep answers informative.",
            message,
        )
        if strict and _contains_ungrounded_claim(value, grounded_numbers=grounded_numbers):
            return [EvidenceStatement(category="Warning", text=_WITHHELD_TEXT)]
        # Use intent-based relaxed/strict check if available; fall back to message heuristics.
        strict_calc_intents = {"EQUILIBRIUM_CALCULATION", "TASK_CORRECTION", "FLASH_CALCULATION"}
        relaxed_intents = {
            "CONCEPT_QA",
            "MODEL_SELECTION_QA",
            "RESULT_INTERPRETATION",
            "PROCESS_RECOMMENDATION",
            "SENSITIVITY_ANALYSIS",
            "FLOW_DESIGN_QA",
        }
        if intent_label and intent_label.upper() in strict_calc_intents:
            check_numbers = True
        elif intent_label and intent_label.upper() in relaxed_intents:
            check_numbers = False
        else:
            check_numbers = _is_thermo_question(message) and _is_calculation_question(message)
        if _is_thermo_question(message) and check_numbers:
            if _contains_ungrounded_claim(value, check_numbers=True, grounded_numbers=grounded_numbers):
                return [EvidenceStatement(category="Warning", text=_WITHHELD_TEXT)]
        if _contains_ungrounded_claim(value, grounded_numbers=grounded_numbers):
            value += "\n\n（以上涉及数值未经确定性引擎验证，请以计算结果为准。）"
        return [EvidenceStatement(category="Knowledge", text=value)]

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        proxy_url: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("DeepSeek API key is required")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.proxy_url = proxy_url if proxy_url is not None else _system_proxy_url()

    async def _request(
        self,
        instructions: str,
        message: str,
        *,
        json_mode: bool = False,
        max_tokens: int = 1024,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": message},
            ],
            "stream": False,
            "thinking": {"type": "disabled"},
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.proxy_url is not None and self.transport is None:
            client = httpx.AsyncClient(timeout=self.timeout_seconds, proxy=self.proxy_url)
        else:
            client = httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport)
        async with client:
            try:
                response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                raise LLMProviderError("DeepSeek", error.response.status_code) from None
            except httpx.RequestError:
                raise LLMProviderError("DeepSeek") from None
            try:
                completion = _DeepSeekChatCompletion.model_validate(response.json())
            except (json.JSONDecodeError, ValidationError):
                raise LLMProviderOutputError("External provider returned an invalid response envelope.") from None
        content = completion.choices[0].message.content
        if not content.strip():
            raise LLMProviderOutputError("External provider returned empty assistant content.")
        return content
