"""Entity extraction utilities for thermodynamic domain."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

# ========== 扩展模型名称集合（包含别名） ==========
MODEL_NAMES = {
    "ideal/raoult",
    "ideal",
    "raoult",
    "wilson",
    "nrtl",
    "uniquac",
    "peng-robinson",
    "pr",
    "srk",
    "soave-redlich-kwong",
    "non-random two-liquid",
    "universal quasichemical",
    # 中文别名
    "威尔逊",
    "非随机双液体模型",
    "通用准化学模型",
}

# ========== 别名映射（用于扩展匹配） ==========
MODEL_ALIAS_MAP = {
    "nrtl": ["nrtl", "non-random two-liquid", "非随机双液体模型"],
    "pr": ["pr", "peng-robinson", "彭-罗宾逊"],
    "srk": ["srk", "soave-redlich-kwong", "索阿韦-雷德利希-邝"],
    "wilson": ["wilson", "威尔逊"],
    "uniquac": ["uniquac", "universal quasichemical", "通用准化学"],
}

COMPONENT_NAMES = {
    ("benzene", "苯"),
    ("toluene", "甲苯"),
    ("methane", "甲烷"),
    ("ethane", "乙烷"),
    ("propane", "丙烷"),
    ("water", "水"),
    ("methanol", "甲醇"),
    ("ethanol", "乙醇"),
    ("carbon dioxide", "二氧化碳", "co2"),
    ("nitrogen", "氮气", "n2"),
    ("hydrogen", "氢气", "h2"),
}

CALCULATION_TYPES = {
    "bubble point",
    "bubble_point",
    "dew point",
    "dew_point",
    "isobaric vle",
    "isobaric_vle",
    "isothermal vle",
    "isothermal_vle",
    "tp flash",
    "tp_flash",
    "azeotrope",
    "lle",
    "liquid liquid equilibrium",
}

PROPERTY_NAMES = {
    "vapor pressure",
    "vapor_pressure",
    "antoinette",
    "antoine",
    "boiling point",
    "boiling_point",
    "melting point",
    "melting_point",
    "critical temperature",
    "critical_temperature",
    "critical pressure",
    "critical_pressure",
    "enthalpy",
    "entropy",
    "density",
    "viscosity",
}

CONCEPT_NAMES = {
    "vle",
    "vapor liquid equilibrium",
    "相平衡",
    "热力学",
    "thermodynamics",
    "equilibrium",
    "phase equilibrium",
    "activity coefficient",
    "活度系数",
    "equation of state",
    "状态方程",
    "fugacity",
    "逸度",
    "raoult's law",
    "拉乌尔定律",
    "henry's law",
    "亨利定律",
}


class EntityType(StrEnum):
    MODEL = "model"
    COMPONENT = "component"
    CALCULATION_TYPE = "calculation_type"
    PROPERTY = "property"
    CONCEPT = "concept"


@dataclass(frozen=True)
class Entity:
    text: str
    type: EntityType
    start: int
    end: int


class EntityExtractor(Protocol):
    def extract(self, text: str) -> list[Entity]: ...


class ThermoEntityExtractor:
    def extract(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        lower_text = text.lower()
        entities.extend(self._extract_models(lower_text))
        entities.extend(self._extract_components(lower_text))
        entities.extend(self._extract_calculation_types(lower_text))
        entities.extend(self._extract_properties(lower_text))
        entities.extend(self._extract_concepts(lower_text))

        # 去重
        seen = set()
        unique_entities = []
        for e in sorted(entities, key=lambda x: x.start):
            key = (e.text, e.type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)
        return unique_entities

    def _extract_models(self, text: str) -> list[Entity]:
        """提取模型名称，支持别名扩展"""
        entities = []
        # 1. 直接精确匹配
        entities.extend(self._extract_keywords(text, MODEL_NAMES, EntityType.MODEL))

        # 2. 别名扩展匹配（例如 "peng-robinson" 出现在文本中时，我们也可标记为 "pr"）
        # 但为了保持实体文本一致，我们只添加原始出现的文本，但可能无法映射到图节点。
        # 更好做法：在查询引擎中做别名归一化，这里先保留。
        # 为简化，我们只匹配别名集合中定义的原始文本。
        # 但为了支持更多召回，我们额外匹配一些常见变体
        alias_variants = set()
        for aliases in MODEL_ALIAS_MAP.values():
            alias_variants.update(aliases)
        alias_variants.difference_update(MODEL_NAMES)  # 移除已在主集合中的
        if alias_variants:
            entities.extend(self._extract_keywords(text, alias_variants, EntityType.MODEL))
        return entities

    def _extract_components(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        for aliases in COMPONENT_NAMES:
            for alias in aliases:
                pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
                for match in re.finditer(pattern, text):
                    entities.append(
                        Entity(
                            text=match.group(0),
                            type=EntityType.COMPONENT,
                            start=match.start(),
                            end=match.end(),
                        )
                    )
        return entities

    def _extract_calculation_types(self, text: str) -> list[Entity]:
        return self._extract_keywords(text, CALCULATION_TYPES, EntityType.CALCULATION_TYPE)

    def _extract_properties(self, text: str) -> list[Entity]:
        return self._extract_keywords(text, PROPERTY_NAMES, EntityType.PROPERTY)

    def _extract_concepts(self, text: str) -> list[Entity]:
        return self._extract_keywords(text, CONCEPT_NAMES, EntityType.CONCEPT)

    @staticmethod
    def _extract_keywords(text: str, keywords: set[str], entity_type: EntityType) -> list[Entity]:
        entities: list[Entity] = []
        for keyword in keywords:
            pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
            for match in re.finditer(pattern, text):
                entities.append(
                    Entity(
                        text=match.group(0),
                        type=entity_type,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return entities