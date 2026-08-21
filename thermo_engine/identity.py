"""Independent component-identity resolution for Agent task grounding."""

from __future__ import annotations

import re
from functools import lru_cache

from chemicals.elements import simple_formula_parser
from chemicals.identifiers import CAS_to_int, search_chemical
from thermo import ChemicalConstantsPackage

from schemas.domain import ComponentIdentity

_NON_COMPONENT_TERMS = {
    "and",
    "at",
    "bubble",
    "calculate",
    "calculation",
    "composition",
    "dew",
    "equilibrium",
    "for",
    "flash",
    "fraction",
    "from",
    "in",
    "include",
    "into",
    "liquid",
    "model",
    "mole",
    "molar",
    "phase",
    "point",
    "pressure",
    "result",
    "temperature",
    "the",
    "to",
    "use",
    "using",
    "vapor",
    "with",
}
_CHINESE_ALIAS_MAP: dict[str, str] = {
    "水": "water",
    "甲烷": "methane",
    "乙烷": "ethane",
    "丙烷": "propane",
    "丁烷": "butane",
    "异丁烷": "isobutane",
    "氮气": "nitrogen",
    "氧气": "oxygen",
    "氢气": "hydrogen",
    "空气": "air",
    "苯": "benzene",
    "甲苯": "toluene",
    "二甲苯": "xylene",
    "邻二甲苯": "o-xylene",
    "间二甲苯": "m-xylene",
    "对二甲苯": "p-xylene",
    "乙苯": "ethylbenzene",
    "苯乙烯": "styrene",
    "乙醇": "ethanol",
    "甲醇": "methanol",
    "正丁醇": "n-butanol",
    "异丁醇": "isobutanol",
    "正丙醇": "n-propanol",
    "异丙醇": "isopropanol",
    "正戊醇": "n-pentanol",
    "正己烷": "n-hexane",
    "正庚烷": "n-heptane",
    "正辛烷": "n-octane",
    "环己烷": "cyclohexane",
    "环戊烷": "cyclopentane",
    "丙酮": "acetone",
    "丁酮": "2-butanone",
    "甲基异丁基酮": "methyl isobutyl ketone",
    "乙酸乙酯": "ethyl acetate",
    "乙酸甲酯": "methyl acetate",
    "乙酸": "acetic acid",
    "氯仿": "chloroform",
    "四氯化碳": "carbon tetrachloride",
    "二氯甲烷": "dichloromethane",
    "三氯乙烯": "trichloroethylene",
    "四氯乙烯": "tetrachloroethylene",
    "乙醚": "diethyl ether",
    "甲基叔丁基醚": "methyl tert-butyl ether",
    "丙烯腈": "acrylonitrile",
    "苯胺": "aniline",
    "苯酚": "phenol",
    "呋喃": "furan",
    "噻吩": "thiophene",
    "吡啶": "pyridine",
    "吡咯": "pyrrole",
    "吗啉": "morpholine",
    "哌嗪": "piperazine",
    "二甲基甲酰胺": "dimethylformamide",
    "二甲基亚砜": "dimethyl sulfoxide",
    "N甲基吡咯烷酮": "n-methyl-2-pyrrolidone",
    "碳酸二甲酯": "dimethyl carbonate",
    "碳酸乙酯": "diethyl carbonate",
    "甲酸": "formic acid",
    "丙酸": "propionic acid",
    "丁酸": "butyric acid",
    "丙烯酸": "acrylic acid",
    "氨水": "ammonia",
    "液氨": "ammonia",
    "二氧化硫": "sulfur dioxide",
    "二氧化碳": "carbon dioxide",
    "一氧化碳": "carbon monoxide",
    "硫化氢": "hydrogen sulfide",
    "氯化氢": "hydrogen chloride",
    "溴化氢": "hydrogen bromide",
    "氟化氢": "hydrogen fluoride",
    "硝酸": "nitric acid",
    "硫酸": "sulfuric acid",
    "盐酸": "hydrochloric acid",
    "磷酸": "phosphoric acid",
    "氢氧化钠": "sodium hydroxide",
    "氢氧化钾": "potassium hydroxide",
    "氯化钠": "sodium chloride",
    "氯化钾": "potassium chloride",
    "碳酸钠": "sodium carbonate",
    "碳酸氢钠": "sodium bicarbonate",
    "硫酸钠": "sodium sulfate",
    "硫酸铜": "copper sulfate",
    "氯化钙": "calcium chloride",
    "硝酸铵": "ammonium nitrate",
    "尿素": "urea",
    "甘油": "glycerol",
    "葡萄糖": "glucose",
    "蔗糖": "sucrose",
    "果糖": "fructose",
}
_CHINESE_NAME_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
_CAS_PATTERN = re.compile(r"(?<!\d)(\d{2,7}-\d{2}-\d)(?!\d)")
_NAME_PATTERN = re.compile(r"(?<![A-Za-z0-9])((?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)(?![A-Za-z0-9])")
_MAX_NAME_WORDS = 4
_CURATED_UNAMBIGUOUS_ALIASES = {
    "2 propanol": "67-63-0",
    "isopropyl alcohol": "67-63-0",
    "ethyl alcohol": "64-17-5",
    "n hexane": "110-54-3",
    "r134a": "811-97-2",
}
_ELECTROLYTE_NAME_PATTERN = re.compile(
    r"^(?:ammonium\b|hydrochloric acid$|hydrobromic acid$|hydroiodic acid$|"
    r"hydrofluoric acid$|sulfuric acid$|nitric acid$|phosphoric acid$|perchloric acid$)",
    re.IGNORECASE,
)
_CURATED_ELECTROLYTE_CAS = {
    "67-48-1",  # choline chloride
    "75-57-0",  # tetramethylammonium chloride
    "127-08-2",  # potassium acetate
    "127-09-3",  # sodium acetate
    "1066-33-7",  # ammonium bicarbonate
    "1310-58-3",  # potassium hydroxide
    "1310-73-2",  # sodium hydroxide
    "144-55-8",  # sodium bicarbonate
    "1643-19-2",  # tetrabutylammonium bromide
    "7447-40-7",  # potassium chloride
    "7647-01-0",  # hydrochloric acid
    "7647-14-5",  # sodium chloride
    "7664-93-9",  # sulfuric acid
    "7705-08-0",  # ferric chloride
    "7758-98-7",  # copper sulfate
    "12265-14-4",  # phosphonium chloride
}


@lru_cache(maxsize=512)
def resolve_external_component(identifier: str) -> ComponentIdentity | None:
    """Resolve one literal name or CAS through the deterministic property database."""
    try:
        constants, _ = ChemicalConstantsPackage.from_IDs([identifier])
    except (ValueError, LookupError, TypeError):
        return None
    if len(constants.CASs) != 1 or len(constants.names) != 1:
        return None
    cas_number = str(constants.CASs[0])
    name = str(constants.names[0]).title()
    is_cas_format = bool(re.match(r"^\d+-\d+-\d+$", identifier))
    component_id = identifier if not is_cas_format else cas_number
    return ComponentIdentity(
        component_id=component_id,
        name=name,
        cas_number=cas_number,
        aliases=[identifier],
    )


def _canonical_name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _is_verified_alias(literal: str, resolved: ComponentIdentity) -> bool:
    if resolved.cas_number is None:
        return False
    try:
        metadata = search_chemical(literal)
    except (ValueError, LookupError, TypeError):
        return False
    if metadata.CAS != CAS_to_int(resolved.cas_number):
        return False
    literal_key = _canonical_name_key(literal)
    canonical_keys = {
        _canonical_name_key(name) for name in (metadata.common_name, metadata.iupac_name, resolved.name) if name
    }
    formula_matches = False
    if re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", literal):
        try:
            formula_matches = simple_formula_parser(literal) == simple_formula_parser(metadata.formula)
        except (ValueError, TypeError):
            formula_matches = False
    return (
        literal_key in canonical_keys
        or _CURATED_UNAMBIGUOUS_ALIASES.get(literal_key) == resolved.cas_number
        or formula_matches
    )


def _has_negated_chemical_role(message: str, start: int) -> bool:
    prefix = message[max(0, start - 64) : start].casefold()
    return (
        re.search(
            r"(?:without|excluding|exclude|free\s+of|with\s+no|do\s+not\s+(?:include|add|use)|"
            r"不含|排除|不要|不包括)[^,;.!?。！？]{0,40}$",
            prefix,
        )
        is not None
    )


def _has_chinese_chemical_role_evidence(message: str, start: int, end: int) -> bool:
    """Check if a Chinese component alias appears in a chemical context.

    More lenient than the English check: Chinese aliases come from a curated map,
    so they need less contextual evidence. Check for separators, adjacent components,
    or chemical keywords nearby.
    """
    prefix = message[max(0, start - 32) : start]
    suffix = message[end : min(len(message), end + 32)]
    separators = {"-", "—", "/", "+", "、", "和", "与", "及", "）", ")"}
    if prefix and prefix[-1] in separators:
        return True
    if suffix and suffix[0] in separators:
        return True
    chemical_keywords = (
        "体系",
        "物系",
        "溶液",
        "混合物",
        "平衡",
        "相",
        "气液",
        "液液",
        "计算",
        "模拟",
        "闪蒸",
        "泡点",
        "露点",
        "曲线",
        "组成",
        "浓度",
    )
    combined = prefix + suffix
    for kw in chemical_keywords:
        if kw in combined:
            return True
    return False


def has_chemical_role_evidence(message: str, start: int, end: int) -> bool:
    prefix = message[max(0, start - 64) : start].casefold()
    suffix = message[end : min(len(message), end + 48)].casefold()
    if _has_negated_chemical_role(message, start):
        return False
    if re.match(r"\s+(?:time|loss(?:es)?|usage|standard|effects?|record(?:ed|s)?)\b", suffix):
        return False
    before = re.search(
        r"(?:(?:calculate|compute|simulate|evaluate)\s+(?:(?:a|the)\s+)?"
        r"(?:(?:(?:salt|brine|saltwater)[-\s]?free|non[-\s]?ionic)\s+)?|"
        r"(?:bubble\s+point|dew\s+point|(?:tp\s+)?flash|vle|azeotrope|equilibrium|"
        r"mixture|system|components?)\s+(?:of|for)\s+(?:the\s+)?|"
        r"(?:use|using)\s+[a-z0-9-]+\s+for\s+(?:the\s+)?|"
        r"(?:mixture|system|components?)\s+(?:containing|contains|with)\s+(?:the\s+)?|"
        r"计算|模拟|物系|组分|含|体系|如|包括|例如|对于)\s*$",
        prefix,
    )
    after = re.match(
        r"\s*(?:and\b|with\b|at\b|in\b|(?:tp\s*)?flash\b|bubble\b|dew\b|vle\b|"
        r"azeotrope\b|equilibrium\b|composition\b|t-x-y|p-x-y|"
        r"和|与|、|在|的|常压|汽液|液液|泡点|露点|相平衡|曲线|闪蒸|"
        r"体系|物系|混合物|系统|组分|中|里|内|"
        r"[-/—+]|[)）]|及)",
        suffix,
    )
    return before is not None or after is not None


def is_electrolyte_identity(component: ComponentIdentity) -> bool:
    """Return whether a resolved component is a supported-scope electrolyte salt."""
    cas_number = component.cas_number
    resolved_name = component.name
    if cas_number is None:
        resolved = resolve_external_component(component.name)
        if resolved is None or resolved.cas_number is None:
            return _ELECTROLYTE_NAME_PATTERN.search(component.name.strip()) is not None
        cas_number = resolved.cas_number
        resolved_name = resolved.name
    try:
        metadata = search_chemical(cas_number)
    except (ValueError, LookupError, TypeError):
        return False
    is_registered = cas_number in _CURATED_ELECTROLYTE_CAS
    is_charged = bool(metadata.charge)
    is_curated_electrolyte = any(
        _ELECTROLYTE_NAME_PATTERN.search(name.strip()) is not None for name in (component.name, resolved_name)
    )
    return is_registered or is_charged or is_curated_electrolyte


def _is_component_list_separator(value: str) -> bool:
    return re.fullmatch(r"\s*(?:(?:,\s*)?(?:and|with)\b|,|/|\+)\s*", value.casefold()) is not None


def _has_scoped_list_role_evidence(message: str, start: int, end: int) -> bool:
    prefix = message[max(0, start - 80) : start].casefold()
    suffix = message[end:].casefold()
    scoped_prefix = re.search(
        r"(?:(?:calculate|compute|simulate|evaluate)\s+(?:(?:a|the)\s+)?"
        r"(?:(?:tp\s+)?flash|vle|bubble\s+point|dew\s+point|azeotrope|equilibrium)?\s*"
        r"(?:for|of)?|(?:use|using)\s+[a-z0-9-]+\s+for)\s*$",
        prefix,
    )
    complete_suffix = re.fullmatch(r"\s*(?:[.!?。！？]\s*)?", suffix)
    return scoped_prefix is not None and complete_suffix is not None


_NON_COMPONENT_CJK_TERMS = frozenset({"水果", "水平", "水分", "水泥", "水晶", "水产", "水电", "水文"})

_CONCEPT_QA_KEYWORDS = re.compile(
    r"(分析|解释|原理|概念|意义|为什么|是什么|介绍|阐述|理解|讲解|说明|讨论|"
    r"对比|比较|区别|联系|特点|特征|应用|用途|案例|例子)"
)


def _is_concept_question(message: str) -> bool:
    """Check if a message is a concept Q&A rather than a calculation request."""
    return _CONCEPT_QA_KEYWORDS.search(message) is not None


def _find_chinese_component_aliases(message: str) -> list[tuple[int, int, str]]:
    """Find Chinese component names in a message and return (start, end, english_name) tuples.

    Handles both standalone names (e.g. "水") and joined names (e.g. "水-正丁醇").
    Also finds names embedded in longer Chinese spans (e.g. "如水" contains "水").

    Skips component identification for concept questions.
    """
    if _is_concept_question(message):
        return []
    results: list[tuple[int, int, str]] = []
    seen_positions: set[tuple[int, int]] = set()
    for chinese_alias, english_name in _CHINESE_ALIAS_MAP.items():
        alias_len = len(chinese_alias)
        idx = 0
        while True:
            idx = message.find(chinese_alias, idx)
            if idx < 0:
                break
            end = idx + alias_len
            if (idx, end) not in seen_positions:
                if not _is_part_of_non_component_term(message, idx, end):
                    seen_positions.add((idx, end))
                    results.append((idx, end, english_name))
            idx += 1
    results.sort(key=lambda x: x[0])
    return results


def _is_part_of_non_component_term(message: str, start: int, end: int) -> bool:
    """Check if a matched component alias is part of a non-component Chinese term."""
    for term in _NON_COMPONENT_CJK_TERMS:
        term_start = 0
        while True:
            term_start = message.find(term, term_start)
            if term_start < 0:
                break
            term_end = term_start + len(term)
            if start >= term_start and end <= term_end:
                return True
            term_start += 1
    return False


def resolve_literal_components(message: str) -> list[tuple[int, ComponentIdentity]]:
    """Resolve literal identities using longest verified-name spans and CAS numbers."""
    candidates: list[tuple[int, int, str, bool, bool]] = [
        (match.start(1), match.end(1), match.group(1), True, False) for match in _CAS_PATTERN.finditer(message)
    ]
    for start, end, english_name in _find_chinese_component_aliases(message):
        candidates.append((start, end, english_name, True, True))
    name_matches = list(_NAME_PATTERN.finditer(message))
    for start_index, first in enumerate(name_matches):
        if first.group(1).casefold() in _NON_COMPONENT_TERMS:
            continue
        for end_index in range(start_index, min(len(name_matches), start_index + _MAX_NAME_WORDS)):
            last = name_matches[end_index]
            words = [match.group(1) for match in name_matches[start_index : end_index + 1]]
            if any(word.casefold() in _NON_COMPONENT_TERMS for word in words):
                break
            if end_index > start_index:
                gap = message[name_matches[end_index - 1].end(1) : last.start(1)]
                if not gap.isspace():
                    break
            literal = " ".join(words)
            if len(literal) >= 3:
                candidates.append((first.start(1), last.end(1), literal, False, False))

    verified_spans: list[tuple[int, int, ComponentIdentity, bool, bool]] = []
    for start, end, literal, is_cas, is_chinese in candidates:
        resolved = resolve_external_component(literal)
        if resolved is None or resolved.cas_number is None:
            continue
        if _has_negated_chemical_role(message, start):
            continue
        if not is_cas and not _is_verified_alias(literal, resolved):
            continue
        verified_spans.append((start, end, resolved, is_cas, is_chinese))

    selected: list[tuple[int, int, ComponentIdentity, bool, bool]] = []
    for start, end, resolved, is_cas, is_chinese in sorted(
        verified_spans,
        key=lambda item: (-(item[1] - item[0]), item[0]),
    ):
        if any(start < selected_end and end > selected_start for selected_start, selected_end, _, _, _ in selected):
            continue
        selected.append((start, end, resolved, is_cas, is_chinese))

    selected.sort(key=lambda item: item[0])
    grounded_indexes: set[int] = set()
    for index, (start, end, _, _is_cas, is_chinese) in enumerate(selected):
        if is_chinese:
            if _has_chinese_chemical_role_evidence(message, start, end):
                grounded_indexes.add(index)
        else:
            if has_chemical_role_evidence(message, start, end):
                grounded_indexes.add(index)

    group_start = 0
    for index in range(len(selected)):
        group_ends = index == len(selected) - 1 or not _is_component_list_separator(
            message[selected[index][1] : selected[index + 1][0]]
        )
        if not group_ends:
            continue
        endpoints_are_grounded = group_start in grounded_indexes and index in grounded_indexes
        group_is_scoped = _has_scoped_list_role_evidence(
            message,
            selected[group_start][0],
            selected[index][1],
        )
        if index > group_start and (endpoints_are_grounded or group_is_scoped):
            grounded_indexes.update(range(group_start, index + 1))
        group_start = index + 1

    resolved_by_cas: dict[str, tuple[int, ComponentIdentity]] = {}
    for index, (position, _, resolved, _, _) in enumerate(selected):
        if index not in grounded_indexes:
            continue
        assert resolved.cas_number is not None
        current = resolved_by_cas.get(resolved.cas_number)
        if current is None or position < current[0]:
            resolved_by_cas[resolved.cas_number] = (position, resolved)
    return sorted(resolved_by_cas.values(), key=lambda item: item[0])
