"""Process flow design skill: LLM generates a structured flow design draft."""

from __future__ import annotations

import json
from typing import Any

from schemas.domain import FlowDesignDraft, FlowFeed, FlowParameter, FlowProductSpec, FlowUnitOperation, Intent

from .llm_client import LLMClient
from .skill_base import KnowledgeSkill, SkillResult

_SYSTEM_PROMPT = """你是 ThermoEqui-Agent 的化工流程设计专家助手。

## 任务
根据用户的需求描述，设计一个初步的化工分离/处理流程，输出结构化的流程定义 JSON。

## 约束
1. 只输出流程结构建议，不直接生成精确的工程数值（塔板数、回流比等给建议值并标注 needs_validation）
2. 所有设计参数必须标注 "source": "LLM_SUGGESTED"，表示非确定性计算结果
3. 如果信息不足，在 assumptions 字段中列出假设条件
4. 单元操作类型只能从以下白名单中选择：
   - preheater（进料预热器）
   - distillation_column（精馏塔）
   - flash_drum（闪蒸罐）
   - condenser（冷凝器）
   - reboiler（再沸器）
   - heat_exchanger（换热器）
   - mixer（混合器）
   - splitter（分流器）
5. 在 notes 字段中标注"本流程为初步设计建议，精确参数需经过后端热力学计算验证"

## 输出格式
输出严格的 JSON，格式如下：
```json
{
  "flow_name": "流程名称",
  "feed": {
    "components": ["组分1", "组分2"],
    "composition_mole": [0.x, 0.x],
    "temperature_K": 数字或null,
    "pressure_kPa": 数字或null,
    "flow_rate_mol_s": 数字或null
  },
  "unit_operations": [
    {
      "id": "U1",
      "type": "白名单中的类型",
      "name": "设备名称",
      "input_stream": "输入流股ID",
      "output_streams": {"产物名": "输出流股ID"},
      "conditions": {
        "参数名": {"value": 数字, "source": "LLM_SUGGESTED", "needs_validation": true}
      }
    }
  ],
  "thermodynamic_model": "推荐的热力学模型或null",
  "product_specs": [
    {"stream": "流股ID", "spec": "纯度要求描述"}
  ],
  "assumptions": ["假设条件1", "假设条件2"],
  "notes": ["本流程为初步设计建议，精确参数需经过后端热力学计算验证"]
}
```
"""


class ProcessFlowDesignSkill(KnowledgeSkill):
    """Skill that uses LLM to generate a structured process flow design draft."""

    def __init__(self, llm: LLMClient | None = None, temperature: float = 0.4) -> None:
        super().__init__(llm=llm, temperature=temperature)

    def name(self) -> str:
        return "process_flow_design"

    def description(self) -> str:
        return "根据用户需求设计初步化工流程，输出结构化流程定义 JSON（建议值，非确定性计算结果）。"

    def supports_intent(self, intent: str) -> bool:
        return intent == Intent.FLOW_DESIGN_QA.value

    def execute(self, query: str, **kwargs: Any) -> SkillResult:
        """Generate a flow design draft from user query.

        Returns SkillResult with:
        - answer: natural language summary for the user
        - metadata["flow_design_json"]: FlowDesignDraft JSON-serializable dict
        """
        # Extract basic info (components) from the query even without LLM
        detected_components = self._extract_components(query)

        def _rule_fallback(reason: str) -> SkillResult:
            draft = self._build_rule_based_flow(query, detected_components)
            summary = self._build_summary(draft)
            fallback_hint = (
                f"\n\n> ℹ️ 说明：{reason}，以上为基于二元物系经典精馏流程的规则模板。"
                "配置好 LLM 后可获得更贴合问题的定制化流程建议。"
            )
            return SkillResult(
                answer=summary + fallback_hint,
                sources=["rule-based-template"],
                confidence=0.7,
                metadata={
                    "flow_design_json": draft.model_dump(mode="json"),
                    "fallback": "rule_based_binary_distillation",
                },
            )

        if not self._llm_available or self._llm is None:
            return _rule_fallback("当前 DeepSeek API 不可用")

        user_prompt = f"""## 用户需求
{query}

## 输出
请输出严格的 JSON 格式的流程设计建议。不要输出 JSON 以外的内容。"""

        try:
            raw_output = self._llm.generate(
                prompt=user_prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=self._temperature,
            )
        except Exception:
            return _rule_fallback("LLM 调用失败")

        # If LLM output is clearly an error (no API key / unavailable) use rule fallback
        _LLM_ERROR_MARKERS = (
            "未配置 api key",
            "无法调用 llm",
            "[llm调用失败]",
            "api key",
        )
        lower_out = raw_output.casefold()
        if any(m in lower_out for m in _LLM_ERROR_MARKERS) and self._extract_json(raw_output) is None:
            return _rule_fallback("未配置 API key 或 LLM 暂时不可用")

        # 尝试从输出中提取 JSON，并用 FlowDesignDraft 强校验
        raw_flow = self._extract_json(raw_output)
        draft: FlowDesignDraft | None = None
        if raw_flow is not None:
            try:
                draft = FlowDesignDraft.model_validate(raw_flow)
            except Exception:
                # LLM returned JSON that does not match the schema → fall back
                draft = None
        if draft is None:
            return _rule_fallback("LLM 输出格式异常")

        summary = self._build_summary(draft)
        return SkillResult(
            answer=summary,
            sources=["llm"],
            confidence=0.7,
            metadata={"flow_design_json": draft.model_dump(mode="json")},
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extract a JSON object from LLM text output."""
        # Try direct parse
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # Try extracting from ```json ... ``` block
        start = text.find("```json")
        if start != -1:
            start = text.find("\n", start)
            if start != -1:
                start += 1
                end = text.find("```", start)
                if end != -1:
                    try:
                        obj = json.loads(text[start:end].strip())
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
        # Try finding first { and last }
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                obj = json.loads(text[first : last + 1])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _build_summary(flow: FlowDesignDraft) -> str:
        """Build a natural language summary from the flow design draft."""
        lines: list[str] = []
        lines.append(f"## 流程设计建议：{flow.flow_name}\n")

        feed = flow.feed
        if feed.components:
            lines.append(f"**进料组分**：{', '.join(feed.components)}")
            if feed.composition_mole:
                comp_str = ", ".join(f"{c}: {v}" for c, v in zip(feed.components, feed.composition_mole, strict=False))
                lines.append(f"**进料组成（摩尔分数）**：{comp_str}")
            if feed.temperature_K:
                lines.append(f"**进料温度**：{feed.temperature_K} K")
            if feed.pressure_kPa:
                lines.append(f"**进料压力**：{feed.pressure_kPa} kPa")

        if flow.unit_operations:
            lines.append("\n**单元操作序列**：")
            for i, unit in enumerate(flow.unit_operations, 1):
                lines.append(f"  {i}. {unit.name}（{unit.type}）")

        if flow.thermodynamic_model:
            lines.append(f"\n**推荐热力学模型**：{flow.thermodynamic_model}")

        if flow.notes:
            lines.append("\n**注意事项**：")
            for note in flow.notes:
                lines.append(f"  - {note}")

        lines.append("\n> ⚠️ 以上为初步流程设计建议，所有参数均为 LLM 建议值，精确参数需经过后端热力学计算验证。")
        return "\n".join(lines)

    @staticmethod
    def _extract_components(query: str) -> list[str]:
        """Very lightweight component extractor for rule-based fallback.

        Returns a list of component names (Chinese or English) found in the query.
        This is intentionally minimal and does NOT bind to authoritative identities;
        it only produces a rule-based fallback when LLM is unavailable.
        """
        components: list[str] = []
        # Common binary pairs in test/demo queries
        aliases: list[tuple[str, str]] = [
            ("乙醇", "ethanol"),
            ("甲醇", "methanol"),
            ("水", "water"),
            ("苯", "benzene"),
            ("甲苯", "toluene"),
            ("丙酮", "acetone"),
            ("乙酸", "acetic acid"),
            ("乙苯", "ethylbenzene"),
            ("异丙醇", "isopropanol"),
        ]
        lower = query.casefold()
        for zh, en in aliases:
            if zh in query or en in lower:
                components.append(zh)
        # Remove duplicates while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for comp in components:
            if comp not in seen:
                seen.add(comp)
                unique.append(comp)
        return unique

    @staticmethod
    def _build_rule_based_flow(query: str, components: list[str]) -> FlowDesignDraft:
        """Generate a rule-based binary distillation flow template.

        Used as deterministic fallback when LLM is not available. All parameters
        are marked as RULE_SUGGESTED / needs_validation so the consumer treats
        them as non-final just like any LLM output.
        """
        names: list[str] = components if len(components) >= 2 else ["组分A", "组分B"]
        comp_names_display = "与".join(names[:2])
        flow_name = f"{comp_names_display} 常压二元精馏分离流程（规则模板）"

        n = len(names)
        # Equimolar assumption as placeholder
        composition = [1.0 / n] * n
        is_alcohol_water_pair = any(c in names for c in ("乙醇", "甲醇")) and "水" in names

        feed = FlowFeed(
            components=names,
            composition_mole=composition,
            temperature_K=298.15,
            pressure_kPa=101.325,
            flow_rate_mol_s=1.0,
            assumption="进料 T/P 未指定，采用 25°C / 101.325 kPa 常温常压假设",
        )
        unit_operations: list[FlowUnitOperation] = [
            FlowUnitOperation(
                id="U1",
                type="preheater",
                name="进料预热器",
                input_stream="FEED",
                output_streams={"outlet": "S1"},
                conditions={
                    "outlet_temperature_K": FlowParameter(
                        value=350.0,
                        source="RULE_SUGGESTED",
                        note="典型泡点附近温度，需根据实际物系调整",
                    )
                },
            ),
            FlowUnitOperation(
                id="U2",
                type="distillation_column",
                name="常压精馏塔",
                input_stream="S1",
                output_streams={"distillate": "S2", "bottoms": "S3"},
                conditions={
                    "theoretical_trays": FlowParameter(
                        value=20,
                        source="RULE_SUGGESTED",
                        note="二元物系典型初值，需根据实际分离要求通过 Fenske/Underwood/Gilliland 重算",
                    ),
                    "feed_tray": FlowParameter(
                        value=10,
                        source="RULE_SUGGESTED",
                        note="默认进料板位于中段",
                    ),
                    "reflux_ratio": FlowParameter(
                        value=1.5,
                        source="RULE_SUGGESTED",
                        note="常见 1.2~2.0 × Rmin 的经验初值",
                    ),
                    "operating_pressure_kPa": FlowParameter(value=101.325, source="RULE_SUGGESTED"),
                    "top_mole_recovery_percent": FlowParameter(value=95.0, source="RULE_SUGGESTED"),
                },
            ),
            FlowUnitOperation(
                id="U3",
                type="condenser",
                name="塔顶全凝器",
                input_stream="S2",
                output_streams={"product": "DIST_PRODUCT", "reflux": "S2_REFLUX"},
                conditions={
                    "outlet_temperature_K": FlowParameter(value=351.5, source="RULE_SUGGESTED"),
                },
            ),
            FlowUnitOperation(
                id="U4",
                type="reboiler",
                name="塔釜再沸器",
                input_stream="S3_REBOIL",
                output_streams={"bottom_product": "BOT_PRODUCT", "vapor_return": "S3_VAPOR"},
                conditions={"note": FlowParameter(value="与精馏塔塔釜物理集成，蒸汽加热", source="RULE_SUGGESTED")},
            ),
        ]
        product_specs = [
            FlowProductSpec(
                stream="DIST_PRODUCT",
                spec="轻组分（低沸点）纯度 ≥ 95% 摩尔分数（默认目标，需根据用户实际要求修正）",
            ),
            FlowProductSpec(
                stream="BOT_PRODUCT",
                spec="重组分（高沸点）纯度 ≥ 99% 摩尔分数（默认目标，需根据用户实际要求修正）",
            ),
        ]
        return FlowDesignDraft(
            flow_name=flow_name,
            flow_type="binary_distillation",
            feed=feed,
            unit_operations=unit_operations,
            streams_connectivity_note=(
                "预热器(S1) → 精馏塔；塔顶(S2) → 全凝器 → 部分回流 + DIST_PRODUCT；"
                "塔釜→再沸器→BOT_PRODUCT。具体的热流股与再沸器连接由后端根据 DWSIM 习惯完善。"
            ),
            thermodynamic_model="Wilson" if is_alcohol_water_pair else "NRTL",
            model_recommendation_note=(
                "醇-水体系常用 Wilson；一般非理想体系常用 NRTL。需根据模型适用域与 VLE 数据做最终选型。"
            ),
            product_specs=product_specs,
            assumptions=[
                "常压操作（101.325 kPa），若物系热敏或共沸需改为真空或加压",
                "进料为常温液体",
                "塔顶全凝",
                "回流比和塔板数为经验初值，需经过严格计算验证",
            ],
            notes=[
                "本流程为二元物系经典精馏的规则化模板。",
                "对于共沸物系（如乙醇-水 ≥ 95.6%），需额外设计萃取精馏、变压精馏或膜分离等后处理单元。",
                "所有参数来源标记为 RULE_SUGGESTED，必须由后端热力学 / 精馏计算模块复核。",
                "参数适用性范围与当前 0.1 版本对齐：仅覆盖非电解质二元 VLE 背景下的初步流程结构建议。",
            ],
        )
