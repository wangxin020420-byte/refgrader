import json
import math
import re
from collections import Counter
from copy import deepcopy


BLANK_VALUES = {"未书写", "字迹模糊", ""}
LOW_QUALITY_VALUES = {
    "是", "有", "已书写", "存在", "有书写", "对", "正确",
    "有提取标注", "有计算过程", "有标注",
}
BLANK_EXTRACTION_VALUES = {"未书写", "未作答", "空白", "无内容", ""}
PERCEPTION_FAILURE_VALUES = {"字迹模糊", "无法识别", "识别失败", "不清晰"}
JUDGEMENT_ANSWER_VALUES = {
    "是", "否", "对", "错", "正确", "错误", "命中", "未命中",
    "可以", "不可以", "能", "不能", "可", "不可", "有", "无",
    "存在", "不存在", "发生", "不发生",
}
GENERIC_PROCESS_VALUES = {"已书写", "有书写", "有提取标注", "有计算过程", "有标注", "有过程"}

NUMERIC_TYPES = {"direct_numeric", "derived_numeric", "numeric"}
METHOD_TYPES = {"formula", "method"}
VISUAL_TYPES = {"sequence", "table_entry", "diagram_ocr"}
TRUSTED_METADATA_THRESHOLD = 0.80
A3WA_RISK_WEIGHTS = {
    "extract": 0.35,
    "score": 0.30,
    "semantic": 0.20,
    "blank": 0.15,
    "overcredit": 0.0,
}
A3WA_LOSS_PARAMS = {
    "lambda1": 5.0,
    "lambda2": 1.0,
    "mu1": 3.0,
    "mu2": 7.0,
    "m": 0.5,
}

SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-",
})


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def normalized_risk_weights(weights=None):
    """Return non-negative A3WA risk weights normalized to sum to 1."""
    base = dict(A3WA_RISK_WEIGHTS)
    if isinstance(weights, dict):
        base.update(weights)
    cleaned = {}
    for key in ("extract", "score", "semantic", "blank", "overcredit"):
        cleaned[key] = max(safe_float(base.get(key, 0.0), 0.0), 0.0)
    total = sum(cleaned.values())
    if total <= 1e-12:
        return dict(A3WA_RISK_WEIGHTS)
    return {key: value / total for key, value in cleaned.items()}


def parse_json_maybe(value, default=None):
    if default is None:
        default = {}
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def rubric_points_map(rubrics_data):
    if not isinstance(rubrics_data, list):
        return {}, 1.0
    points_map = {}
    points_values = []
    for item in rubrics_data:
        item_id = str(item.get("id", ""))
        points = safe_float(item.get("points", 0), 0.0)
        if item_id:
            points_map[item_id] = points
        if points > 0:
            points_values.append(points)
    fallback = sum(points_values) / len(points_values) if points_values else 1.0
    return points_map, fallback


def _normalized_text(text):
    return str(text or "").replace("−", "-").replace("×", "x").replace("÷", "/")


def _canonical_unit(unit):
    raw = str(unit or "").strip()
    raw = raw.replace("µ", "μ")
    lower = raw.lower()
    aliases = {
        "秒": "s", "sec": "s", "second": "s", "seconds": "s",
        "毫秒": "ms", "微秒": "us", "μs": "us", "us": "us", "纳秒": "ns",
        "hz": "hz", "khz": "khz", "mhz": "mhz", "ghz": "ghz",
        "位": "bit", "bit": "bit", "bits": "bit", "b": "bit",
        "字节": "byte", "byte": "byte", "bytes": "byte",
        "kb": "kb", "kib": "kb", "mb": "mb", "mib": "mb",
    }
    return aliases.get(raw, aliases.get(lower, lower))


def _unit_dimension(unit):
    unit = _canonical_unit(unit)
    if unit in {"s", "ms", "us", "ns"}:
        return "time"
    if unit in {"hz", "khz", "mhz", "ghz"}:
        return "frequency"
    if unit in {"bit", "byte", "kb", "mb"}:
        return "data"
    return ""


def _unit_factor_to_base(unit):
    unit = _canonical_unit(unit)
    factors = {
        "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9,
        "hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9,
        "bit": 1.0, "byte": 8.0, "kb": 8.0 * 1024, "mb": 8.0 * 1024 * 1024,
    }
    return factors.get(unit)


def _convert_unit(value, source_unit, target_unit):
    source_unit = _canonical_unit(source_unit)
    target_unit = _canonical_unit(target_unit)
    if not target_unit or not source_unit:
        return value
    if _unit_dimension(source_unit) != _unit_dimension(target_unit):
        return None
    source_factor = _unit_factor_to_base(source_unit)
    target_factor = _unit_factor_to_base(target_unit)
    if source_factor is None or target_factor is None or target_factor == 0:
        return None
    return value * source_factor / target_factor


def _unit_pattern():
    return r"(GHz|MHz|KHz|kHz|Hz|μs|µs|us|ms|ns|s|位|bits?|bytes?|字节|KiB|KB|MiB|MB|B|b)"


def infer_unit(text):
    matches = list(re.finditer(_unit_pattern(), str(text or ""), re.I))
    if not matches:
        return ""
    return _canonical_unit(matches[-1].group(1))


def _convert_number_token(number_text, suffix=""):
    value = safe_float(str(number_text).replace(",", ""), None)
    if value is None:
        return None
    if suffix == "万":
        return value * 10000
    if suffix == "亿":
        return value * 100000000
    return value


def extract_numeric_candidates(text, target_unit=None):
    """Return numeric candidates without assuming a question-specific unit system."""
    raw = _normalized_text(text)
    target_unit = _canonical_unit(target_unit)
    candidates = []
    unit_pat = _unit_pattern()

    # Scientific notation variants: 4x10^9, 4x10⁹, 10⁵.
    sci_text = raw.translate(SUPERSCRIPT_MAP)
    for match in re.finditer(rf"(-?\d+(?:\.\d+)?)\s*x\s*10\s*\^?\s*(-?\d+)\s*{unit_pat}?", sci_text, re.I):
        base = safe_float(match.group(1), None)
        exp = safe_float(match.group(2), None)
        if base is not None and exp is not None and abs(exp) <= 32:
            value = base * (10 ** int(exp))
            unit = match.group(3) if len(match.groups()) >= 3 else ""
            converted = _convert_unit(value, unit, target_unit) if unit and target_unit else value
            if converted is not None:
                candidates.append(converted)
            candidates.append(base)

    for match in re.finditer(rf"\b10\s*\^?\s*(-?\d+)\s*{unit_pat}?", sci_text, re.I):
        exp = safe_float(match.group(1), None)
        if exp is not None and abs(exp) <= 32:
            value = 10 ** int(exp)
            unit = match.group(2) if len(match.groups()) >= 2 else ""
            converted = _convert_unit(value, unit, target_unit) if unit and target_unit else value
            if converted is not None:
                candidates.append(converted)

    # Power notation commonly used in CS answers, e.g. 2^9.
    for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*\^\s*(-?\d+)", sci_text):
        base = safe_float(match.group(1), None)
        exp = safe_float(match.group(2), None)
        if base is not None and exp is not None and abs(exp) <= 16:
            candidates.append(base ** int(exp))

    # Plain numbers, with Chinese large-number suffixes handled locally.
    for match in re.finditer(rf"(-?\d+(?:,\d{{3}})*(?:\.\d+)?|-?\d+(?:\.\d+)?)(万|亿)?\s*{unit_pat}?", raw, re.I):
        value = _convert_number_token(match.group(1), match.group(2) or "")
        if value is not None:
            unit_group_index = 3
            unit = match.group(unit_group_index) if len(match.groups()) >= unit_group_index else ""
            converted = _convert_unit(value, unit, target_unit) if unit and target_unit else value
            if converted is not None:
                candidates.append(converted)

    unique = []
    for value in candidates:
        if abs(value) <= 1e100 and math.isfinite(float(value)) and not any(abs(value - seen) <= 1e-9 for seen in unique):
            unique.append(value)
    return unique


def infer_expected_number(rubric_item):
    if isinstance(rubric_item, dict) and rubric_item.get("expected") is not None:
        return safe_float(rubric_item.get("expected"), None)
    text = rubric_item.get("item", "") if isinstance(rubric_item, dict) else str(rubric_item)
    raw = _normalized_text(text)
    matches = list(re.finditer(r"(-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)(万|亿)?", raw))
    if not matches:
        return None
    last = matches[-1]
    return _convert_number_token(last.group(1), last.group(2) or "")


def prepare_rubrics_for_calibration(rubrics_data):
    """Conservatively enrich plain rubrics with machine-checkable metadata.

    The generated metadata is only trusted when it passes simple deterministic
    quality gates. This keeps the calibration automated without adding
    question-specific code paths.
    """
    if not isinstance(rubrics_data, list):
        return rubrics_data

    prepared = []
    for raw_item in rubrics_data:
        item = deepcopy(raw_item)
        existing_metadata = any(
            key in item
            for key in ("answer_type", "role", "expected", "unit", "formula", "expected_formula", "depends_on")
        )
        meta = classify_rubric_item(item)
        item.setdefault("answer_type", meta["answer_type"])
        item.setdefault("role", meta["role"])

        confidence = safe_float(item.get("metadata_confidence", 0.0), 0.0)
        if existing_metadata:
            confidence = max(confidence, 1.0)
            item.setdefault("metadata_source", "explicit")
            item.setdefault("metadata_hard_enabled", True)
        else:
            expected = infer_expected_number(item)
            unit = infer_unit(item.get("item", ""))
            if expected is not None and item["answer_type"] in NUMERIC_TYPES:
                item["expected"] = expected
                if unit:
                    item["unit"] = unit
                item["metadata_source"] = "auto"
                item["metadata_hard_enabled"] = False
                confidence = 0.85 if unit or abs(expected) >= 1 else 0.80
            elif item["answer_type"] in VISUAL_TYPES:
                item["metadata_source"] = "auto"
                item["metadata_hard_enabled"] = False
                confidence = 0.80
            else:
                item["metadata_source"] = "auto_low_confidence"
                item["metadata_hard_enabled"] = False
                confidence = 0.0

        item["metadata_confidence"] = round(confidence, 4)
        prepared.append(item)
    return prepared


def classify_rubric_item(item):
    """Infer a generic item type. Explicit metadata wins over text heuristics."""
    if not isinstance(item, dict):
        return {"answer_type": "unknown", "role": "unknown", "visual_complexity": False}

    text = str(item.get("item", ""))
    explicit_type = str(item.get("answer_type", "")).strip()
    explicit_role = str(item.get("role", "")).strip()

    if explicit_type:
        answer_type = explicit_type
    elif any(word in text for word in ("公式", "方法", "表达式", "推导式", "算法")):
        answer_type = "formula"
    elif any(word in text for word in ("二进制", "十六进制", "序列", "串", "编码", "状态序列")):
        answer_type = "sequence"
    elif any(word in text for word in ("表", "矩阵", "分析表", "页表", "状态表")):
        answer_type = "table_entry"
    elif infer_expected_number(item) is not None:
        if any(word in text for word in ("识别", "读出", "给出", "指出")):
            answer_type = "direct_numeric"
        else:
            answer_type = "derived_numeric"
    elif any(word in text for word in ("判断", "结论", "命中", "能否", "是否")):
        answer_type = "judgement"
    else:
        answer_type = "concept_keyword"

    if explicit_role:
        role = explicit_role
    elif answer_type in METHOD_TYPES:
        role = "method"
    elif any(word in text for word in ("最终", "总", "最多", "结果", "结论", "容量", "时间")):
        role = "final"
    elif answer_type == "direct_numeric":
        role = "parameter"
    elif answer_type == "derived_numeric":
        role = "intermediate"
    else:
        role = "unknown"

    visual_complexity = bool(item.get("visual_complexity", False)) or answer_type in VISUAL_TYPES
    return {
        "answer_type": answer_type,
        "role": role,
        "visual_complexity": visual_complexity,
    }


def is_blank_extraction(value):
    return str(value).strip() in BLANK_EXTRACTION_VALUES


def is_perception_failure(value):
    return str(value).strip() in PERCEPTION_FAILURE_VALUES


def is_low_quality_extraction(value, rubric_item=None):
    """Whether an extracted fact is too generic for its rubric item.

    Judgement items can legitimately be answered by short labels such as
    "yes/no" or "hit/miss". Formula, numeric, and process items need concrete
    content rather than generic phrases such as "has calculation process".
    """
    text = str(value).strip()
    if is_blank_extraction(text) or is_perception_failure(text):
        return False

    meta = classify_rubric_item(rubric_item or {})
    answer_type = meta.get("answer_type", "unknown")
    role = meta.get("role", "unknown")

    if answer_type == "judgement" and text in JUDGEMENT_ANSWER_VALUES:
        return False

    if text in GENERIC_PROCESS_VALUES:
        return True

    if text in LOW_QUALITY_VALUES:
        if answer_type == "judgement":
            return False
        if role == "final" and answer_type in ("judgement", "concept_keyword"):
            return False
        return True

    return False


def compute_extraction_quality_counts(facts_dict, rubrics_data):
    """Count blank, perception-failure, and rubric-aware low-quality facts."""
    facts_dict = facts_dict if isinstance(facts_dict, dict) else {}
    rubrics_data = rubrics_data if isinstance(rubrics_data, list) else []
    blank_count = 0
    perception_fail_count = 0
    low_quality_count = 0

    if rubrics_data:
        for item in rubrics_data:
            item_id = str(item.get("id", ""))
            value = facts_dict.get(item_id, "") if item_id else ""
            if is_blank_extraction(value):
                blank_count += 1
            elif is_perception_failure(value):
                perception_fail_count += 1
            elif is_low_quality_extraction(value, item):
                low_quality_count += 1
        total = len(rubrics_data)
    else:
        for value in facts_dict.values():
            if is_blank_extraction(value):
                blank_count += 1
            elif is_perception_failure(value):
                perception_fail_count += 1
            elif is_low_quality_extraction(value, None):
                low_quality_count += 1
        total = len(facts_dict)

    return {
        "blank_count": blank_count,
        "perception_fail_count": perception_fail_count,
        "low_quality_count": low_quality_count,
        "total_items": total,
    }


def _numeric_value_matches(fact_value, expected, tolerance=0.10, target_unit=None):
    if expected is None:
        return None
    candidates = extract_numeric_candidates(fact_value, target_unit=target_unit)
    if not candidates:
        return False
    denom = max(abs(expected), 1e-9)
    return any(abs(candidate - expected) / denom <= tolerance for candidate in candidates)


def _formula_is_supported(fact_value, rubric_item):
    text = _normalized_text(fact_value)
    if str(fact_value).strip() in BLANK_VALUES or str(fact_value).strip() in LOW_QUALITY_VALUES:
        return False

    has_operator = any(op in text for op in ("=", "+", "-", "*", "/", "x", "^"))
    if not has_operator:
        return False

    expected_formula = str(rubric_item.get("expected_formula", "") or rubric_item.get("formula", ""))
    if expected_formula:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]{2,}", expected_formula)
        meaningful = [token.lower() for token in tokens if len(token) >= 2]
        if meaningful:
            lower_text = text.lower()
            matched = sum(1 for token in meaningful if token in lower_text)
            return matched >= max(1, math.ceil(len(meaningful) * 0.5))

    # Without metadata, only check that the answer is formula-like. This is
    # intentionally conservative to avoid hard-coding question-specific formulas.
    return True


def _detail_by_item(strict_cots):
    per_item = {}
    for cot in strict_cots or []:
        for detail in cot.get("details", []):
            item_id = str(detail.get("id", ""))
            if not item_id:
                continue
            per_item.setdefault(item_id, []).append(detail)
    return per_item


def _majority_category(details):
    categories = [str(d.get("error_category", "")) for d in details]
    return Counter(categories).most_common(1)[0][0] if categories else ""


def build_post_grading_calibration(
    facts_dict,
    rubrics_data,
    strict_cots,
    avg_model_score,
    max_score,
    blank_rate,
    risk_profile=None,
):
    """Build generic, question-agnostic calibration signals and score bounds."""
    facts_dict = facts_dict if isinstance(facts_dict, dict) else {}
    rubrics_data = rubrics_data if isinstance(rubrics_data, list) else []
    max_score = max(safe_float(max_score, 0.0), 1.0)
    avg_model_score = safe_float(avg_model_score, 0.0)
    points_map, fallback_points = rubric_points_map(rubrics_data)
    details_by_item = _detail_by_item(strict_cots)

    unsupported_match_points = 0.0
    verified_method_final_points = 0.0
    method_final_points = 0.0
    direct_points = 0.0
    direct_awarded_points = 0.0
    result_points = 0.0
    result_awarded_points = 0.0
    result_strong_points = 0.0
    method_evidence_points = 0.0
    method_evidence_total = 0.0
    fatal_points = 0.0
    partial_or_format_points = 0.0
    visual_items = 0
    rule_hits = []
    metadata_items = 0
    explicit_chain_items = 0

    for item in rubrics_data:
        item_id = str(item.get("id", ""))
        points = points_map.get(item_id, fallback_points)
        meta = classify_rubric_item(item)
        answer_type = meta["answer_type"]
        role = meta["role"]
        fact_value = facts_dict.get(item_id, "")
        details = details_by_item.get(item_id, [])
        majority_category = _majority_category(details)
        has_structured_metadata = (
            safe_float(item.get("metadata_confidence", 0.0), 0.0) >= TRUSTED_METADATA_THRESHOLD
        )
        hard_metadata_enabled = has_structured_metadata and bool(item.get("metadata_hard_enabled", False))
        if has_structured_metadata:
            metadata_items += 1
        if hard_metadata_enabled and any(
            key in item for key in ("formula", "expected_formula", "depends_on")
        ):
            explicit_chain_items += 1

        if meta["visual_complexity"]:
            visual_items += 1

        if answer_type == "direct_numeric" or role == "parameter":
            direct_points += points
            if majority_category in ("MATCH", "FORMAT_MINOR", "PARTIAL_MATCH"):
                direct_awarded_points += points

        is_result_item = role == "final" or (answer_type in NUMERIC_TYPES and role != "parameter")
        is_method_evidence_item = (
            role in ("method", "intermediate", "final")
            or answer_type in METHOD_TYPES
            or answer_type == "derived_numeric"
        )
        if is_result_item:
            result_points += points
            if majority_category in ("MATCH", "FORMAT_MINOR", "PARTIAL_MATCH"):
                result_awarded_points += points
            if majority_category in ("MATCH", "FORMAT_MINOR"):
                result_strong_points += points
        if is_method_evidence_item:
            method_evidence_total += points
            if majority_category in ("MATCH", "FORMAT_MINOR", "PARTIAL_MATCH"):
                method_evidence_points += points
        if majority_category == "SEMANTIC_FATAL":
            fatal_points += points
        if majority_category in ("PARTIAL_MATCH", "FORMAT_MINOR"):
            partial_or_format_points += points

        if role in ("method", "intermediate", "final") or answer_type in METHOD_TYPES or answer_type == "derived_numeric":
            method_final_points += points

        if answer_type in NUMERIC_TYPES and hard_metadata_enabled:
            expected = infer_expected_number(item)
            numeric_match = _numeric_value_matches(fact_value, expected, target_unit=item.get("unit"))
            if numeric_match is True and role in ("method", "intermediate", "final"):
                verified_method_final_points += points
            if majority_category == "MATCH" and numeric_match is False and role in ("method", "intermediate", "final"):
                unsupported_match_points += points

        elif answer_type in METHOD_TYPES and hard_metadata_enabled:
            formula_supported = _formula_is_supported(fact_value, item)
            if formula_supported and role in ("method", "intermediate", "final"):
                verified_method_final_points += points
            if majority_category == "MATCH" and not formula_supported:
                unsupported_match_points += points

    unsupported_ratio = unsupported_match_points / max_score
    method_final_ratio = verified_method_final_points / max(method_final_points, 1e-9)
    direct_points_ratio = direct_points / max_score
    direct_awarded_ratio = direct_awarded_points / max(direct_points, 1e-9)
    result_correctness_signal = result_awarded_points / max(result_points, 1e-9) if result_points > 0 else direct_awarded_ratio
    result_strong_signal = result_strong_points / max(result_points, 1e-9) if result_points > 0 else direct_awarded_ratio
    method_evidence_signal = method_evidence_points / max(method_evidence_total, 1e-9) if method_evidence_total > 0 else 0.0
    partial_or_format_ratio = partial_or_format_points / max_score
    fatal_ratio = fatal_points / max_score
    avg_ratio = clamp01(avg_model_score / max_score)
    process_evidence = max(method_evidence_signal, partial_or_format_ratio, 0.5 * direct_awarded_ratio)
    score_gap_to_high_band = clamp01((0.85 - avg_ratio) / 0.85)
    bare_answer_risk = clamp01(result_correctness_signal * (1.0 - max(method_evidence_signal, partial_or_format_ratio)))
    lenient_undercredit_signal = clamp01(
        result_correctness_signal
        * process_evidence
        * score_gap_to_high_band
        * (1.0 - 0.35 * clamp01(fatal_ratio))
    )
    metadata_coverage = metadata_items / max(len(rubrics_data), 1)
    explicit_chain_coverage = explicit_chain_items / max(len(rubrics_data), 1)
    has_chain_structure = explicit_chain_coverage >= 0.30 and method_final_points >= max_score * 0.25 and direct_points > 0
    core_anchor_failed = (
        has_chain_structure
        and direct_awarded_ratio >= 0.50
        and method_final_ratio <= 0.15
        and avg_model_score >= max_score * 0.25
    )
    unsupported_high_score_risk = clamp01(
        avg_ratio
        * max(
            unsupported_ratio,
            bare_answer_risk if avg_ratio >= 0.70 else 0.0,
            fatal_ratio if avg_ratio >= 0.80 else 0.0,
            1.0 if core_anchor_failed else 0.0,
        )
    )

    visual_blank_review = bool(visual_items) and blank_rate >= 0.40
    if unsupported_ratio >= 0.15:
        rule_hits.append("unsupported_match_guard")
    if core_anchor_failed:
        rule_hits.append("core_anchor_guard")
    if visual_blank_review:
        rule_hits.append("visual_blank_review")
    if lenient_undercredit_signal >= 0.08:
        rule_hits.append("lenient_undercredit_review")
    if unsupported_high_score_risk >= 0.25:
        rule_hits.append("unsupported_high_score_review")

    lower_bound = 0.0
    upper_bound = max_score
    if unsupported_ratio >= 0.15:
        upper_bound = min(upper_bound, avg_model_score)
    if unsupported_ratio >= 0.25:
        upper_bound = min(upper_bound, max(avg_model_score * 0.85, max_score * 0.25))
    if core_anchor_failed:
        parameter_cap = direct_points * 0.30
        verified_cap = verified_method_final_points
        upper_bound = min(upper_bound, max(parameter_cap + verified_cap, max_score * 0.20))

    boundary_domain = (
        unsupported_ratio >= 0.15
        or core_anchor_failed
        or lenient_undercredit_signal >= 0.08
        or unsupported_high_score_risk >= 0.25
    )
    reject_domain = visual_blank_review

    return {
        "unsupported_match_points": round(unsupported_match_points, 4),
        "unsupported_match_points_ratio": round(unsupported_ratio, 4),
        "method_final_points": round(method_final_points, 4),
        "verified_method_final_points": round(verified_method_final_points, 4),
        "method_final_verified_ratio": round(method_final_ratio, 4),
        "direct_points_ratio": round(direct_points_ratio, 4),
        "direct_awarded_ratio": round(direct_awarded_ratio, 4),
        "result_correctness_signal": round(result_correctness_signal, 4),
        "result_strong_signal": round(result_strong_signal, 4),
        "method_evidence_signal": round(method_evidence_signal, 4),
        "partial_or_format_points_ratio": round(partial_or_format_ratio, 4),
        "bare_answer_risk": round(bare_answer_risk, 4),
        "lenient_undercredit_signal": round(lenient_undercredit_signal, 4),
        "unsupported_high_score_risk": round(unsupported_high_score_risk, 4),
        "metadata_coverage": round(metadata_coverage, 4),
        "explicit_chain_coverage": round(explicit_chain_coverage, 4),
        "core_anchor_failed": core_anchor_failed,
        "visual_blank_review": visual_blank_review,
        "boundary_domain": boundary_domain,
        "reject_domain": reject_domain,
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "rule_hits": rule_hits,
    }


def clamp01(value):
    return max(0.0, min(1.0, safe_float(value, 0.0)))


def compute_a3wa_thresholds(lambda1=5.0, lambda2=1.0, mu1=3.0, mu2=7.0, m=0.5):
    """Compute asymmetric three-way thresholds from A3WA loss parameters."""
    lambda1 = max(safe_float(lambda1, 5.0), 1e-9)
    lambda2 = max(safe_float(lambda2, 1.0), 1e-9)
    mu1 = max(safe_float(mu1, 3.0), 1e-9)
    mu2 = max(safe_float(mu2, 7.0), 1e-9)
    m = clamp01(m)
    alpha = (lambda1 + lambda2 * m) / (lambda1 + lambda2)
    beta = (mu2 * m) / (mu1 + mu2)
    if beta >= alpha:
        midpoint = (alpha + beta) / 2.0
        alpha = min(1.0, midpoint + 0.05)
        beta = max(0.0, midpoint - 0.05)
    return alpha, beta


def discrete_a3wa_information_loss(confidences, lambda1=5.0, lambda2=1.0, mu1=3.0, mu2=7.0, m=0.5):
    """Discrete version of A3WA information loss over confidence values."""
    alpha, beta = compute_a3wa_thresholds(lambda1, lambda2, mu1, mu2, m)
    total = 0.0
    for confidence in confidences or []:
        value = clamp01(confidence)
        if value >= alpha:
            total += lambda1 * (1.0 - value)
        elif value >= m:
            total += lambda2 * (value - m)
        elif value <= beta:
            total += mu1 * value
        else:
            total += mu2 * (m - value)
    return total


def optimize_a3wa_m(confidences, lambda1=5.0, lambda2=1.0, mu1=3.0, mu2=7.0, grid=None):
    """Search m on a fixed grid. Intended for offline/batch analysis."""
    if grid is None:
        grid = [i / 100 for i in range(10, 91, 5)]
    best_m = A3WA_LOSS_PARAMS["m"]
    best_loss = None
    for candidate in grid:
        loss = discrete_a3wa_information_loss(confidences, lambda1, lambda2, mu1, mu2, candidate)
        if best_loss is None or loss < best_loss:
            best_m = candidate
            best_loss = loss
    alpha, beta = compute_a3wa_thresholds(lambda1, lambda2, mu1, mu2, best_m)
    return {
        "m": round(best_m, 4),
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
        "information_loss": round(best_loss or 0.0, 6),
    }


def build_a3wa_decision(
    model_scores,
    avg_model_score,
    std_dev,
    max_score,
    blank_rate,
    low_quality_rate,
    perception_failure_rate,
    extraction_quality,
    fatal_points_ratio,
    high_blank_high_score=False,
    post_calibration=None,
    weights=None,
    loss_params=None,
):
    """Build A3WA-inspired route decision from generic risk components."""
    model_scores = [safe_float(s, 0.0) for s in (model_scores or [])]
    max_score = max(safe_float(max_score, 0.0), 1.0)
    avg_model_score = safe_float(avg_model_score, 0.0)
    std_dev = safe_float(std_dev, 0.0)
    post_calibration = post_calibration or {}
    weights = normalized_risk_weights(weights)
    params = dict(A3WA_LOSS_PARAMS if loss_params is None else loss_params)

    score_spread = max(model_scores) - min(model_scores) if model_scores else 0.0
    normalized_std = clamp01(std_dev / max_score)
    score_spread_norm = clamp01(score_spread / max_score)

    u_extract = clamp01(0.5 * safe_float(low_quality_rate, 0.0) + 0.5 * safe_float(perception_failure_rate, 0.0))
    u_score = clamp01(0.5 * normalized_std + 0.5 * score_spread_norm)
    u_semantic = clamp01(fatal_points_ratio)
    u_blank = clamp01(blank_rate)

    unsupported_ratio = safe_float(post_calibration.get("unsupported_match_points_ratio", 0.0), 0.0)
    if unsupported_ratio >= 0.15:
        u_semantic = max(u_semantic, min(1.0, 0.50 + unsupported_ratio))
    if post_calibration.get("core_anchor_failed", False):
        u_semantic = max(u_semantic, 0.60)
    if avg_model_score <= 0.80 * max_score and u_blank <= 0.50:
        u_score = min(1.0, u_score + 0.10)
    avg_ratio = clamp01(avg_model_score / max_score)
    core_anchor_score = 1.0 if post_calibration.get("core_anchor_failed", False) else 0.0
    high_blank_score = u_blank if avg_ratio >= 0.70 else 0.0
    lenient_undercredit = clamp01(post_calibration.get("lenient_undercredit_signal", 0.0))
    unsupported_high_score = clamp01(post_calibration.get("unsupported_high_score_risk", 0.0))
    u_overcredit = clamp01(
        avg_ratio * max(
            u_semantic,
            unsupported_ratio,
            core_anchor_score,
            high_blank_score,
        )
    )

    risk = (
        safe_float(weights.get("extract", 0.35), 0.35) * u_extract
        + safe_float(weights.get("score", 0.30), 0.30) * u_score
        + safe_float(weights.get("semantic", 0.20), 0.20) * u_semantic
        + safe_float(weights.get("blank", 0.15), 0.15) * u_blank
        + safe_float(weights.get("overcredit", 0.0), 0.0) * u_overcredit
    )
    risk = clamp01(risk)
    confidence = 1.0 - risk

    alpha, beta = compute_a3wa_thresholds(
        lambda1=params.get("lambda1", 5.0),
        lambda2=params.get("lambda2", 1.0),
        mu1=params.get("mu1", 3.0),
        mu2=params.get("mu2", 7.0),
        m=params.get("m", 0.5),
    )

    hard_neg_reasons = []
    if extraction_quality == "failed":
        hard_neg_reasons.append("extraction_failed")
    if score_spread >= max(2.0, max_score * 0.35):
        hard_neg_reasons.append("large_score_spread")
    if u_semantic >= 0.75 and lenient_undercredit < 0.10:
        hard_neg_reasons.append("semantic_risk_too_high")
    if high_blank_high_score and u_extract >= 0.50:
        hard_neg_reasons.append("high_blank_high_score")
    if post_calibration.get("reject_domain", False):
        hard_neg_reasons.extend(post_calibration.get("rule_hits", ["post_calibration_reject"]))

    if hard_neg_reasons:
        route = "NEG"
        reason = "hard_neg:" + ",".join(hard_neg_reasons)
    elif confidence >= alpha:
        if lenient_undercredit >= 0.08 and avg_ratio <= 0.85:
            route = "BND"
            reason = "high_confidence_lenient_undercredit_review"
        elif unsupported_high_score >= 0.25:
            route = "BND"
            reason = "high_confidence_unsupported_high_score_review"
        else:
            route = "POS"
            reason = "confidence_ge_alpha"
    elif confidence <= beta:
        route = "NEG"
        reason = "confidence_le_beta"
    else:
        route = "BND"
        reason = "beta_lt_confidence_lt_alpha"

    return {
        "route": route,
        "reason": reason,
        "hard_neg_reasons": hard_neg_reasons,
        "risk": round(risk, 6),
        "confidence": round(confidence, 6),
        "mu": round(confidence, 6),
        "alpha": round(alpha, 6),
        "beta": round(beta, 6),
        "m": round(safe_float(params.get("m", 0.5), 0.5), 6),
        "lambda1": safe_float(params.get("lambda1", 5.0), 5.0),
        "lambda2": safe_float(params.get("lambda2", 1.0), 1.0),
        "mu1": safe_float(params.get("mu1", 3.0), 3.0),
        "mu2": safe_float(params.get("mu2", 7.0), 7.0),
        "risk_components": {
            "U_extract": round(u_extract, 6),
            "U_score": round(u_score, 6),
            "U_semantic": round(u_semantic, 6),
            "U_blank": round(u_blank, 6),
            "U_overcredit": round(u_overcredit, 6),
            "L_lenient_undercredit": round(lenient_undercredit, 6),
            "U_unsupported_high_score": round(unsupported_high_score, 6),
            "normalized_std": round(normalized_std, 6),
            "score_spread_norm": round(score_spread_norm, 6),
        },
        "score_spread": round(score_spread, 6),
    }


def a3wa_dynamic_bounds(
    avg_model_score,
    max_score,
    a3wa_decision,
    risk_profile=None,
    post_calibration=None,
    gamma=0.30,
    min_delta_ratio=0.10,
):
    """Direction-aware score bounds for BND arbitration."""
    avg_model_score = safe_float(avg_model_score, 0.0)
    max_score = max(safe_float(max_score, 0.0), 1.0)
    a3wa_decision = a3wa_decision or {}
    risk_profile = risk_profile or {}
    post_calibration = post_calibration or {}

    confidence = safe_float(a3wa_decision.get("confidence", a3wa_decision.get("mu", 0.5)), 0.5)
    alpha = safe_float(a3wa_decision.get("alpha", 0.917), 0.917)
    beta = safe_float(a3wa_decision.get("beta", 0.35), 0.35)
    denom = max(alpha - beta, 1e-9)
    review_strength = clamp01((alpha - confidence) / denom)
    delta = max(
        safe_float(min_delta_ratio, 0.10) * max_score,
        safe_float(gamma, 0.30) * review_strength * max_score,
    )

    signals = build_boundary_direction_signals(
        avg_model_score=avg_model_score,
        max_score=max_score,
        a3wa_decision=a3wa_decision,
        risk_profile=risk_profile,
        post_calibration=post_calibration,
    )

    small_margin = max(0.05 * max_score, 0.5)
    large_margin = max(delta, 0.20 * max_score, 1.5)

    lower_margin = large_margin if signals["over_score_risk"] else small_margin
    upper_margin = large_margin if signals["under_score_risk"] else small_margin

    lower_bound = max(0.0, avg_model_score - lower_margin)
    upper_bound = min(max_score, avg_model_score + upper_margin)

    if signals["strong_over_score_risk"]:
        upper_bound = min(upper_bound, avg_model_score)

    post_upper = safe_float(post_calibration.get("upper_bound", max_score), max_score)
    post_lower = safe_float(post_calibration.get("lower_bound", 0.0), 0.0)
    if post_upper < upper_bound:
        upper_bound = post_upper
    lower_bound = max(lower_bound, post_lower)
    if lower_bound > upper_bound:
        lower_bound = 0.0 if upper_bound < avg_model_score else max(0.0, min(avg_model_score, upper_bound))

    return lower_bound, upper_bound, {
        "review_strength": round(review_strength, 6),
        "delta": round(delta, 6),
        "over_score_guard": signals["strong_over_score_risk"],
        "direction_signals": signals,
    }


def build_boundary_direction_signals(
    avg_model_score,
    max_score,
    a3wa_decision=None,
    risk_profile=None,
    post_calibration=None,
):
    """Infer whether a boundary sample has evidence for lowering or raising."""
    avg_model_score = safe_float(avg_model_score, 0.0)
    max_score = max(safe_float(max_score, 0.0), 1.0)
    a3wa_decision = a3wa_decision or {}
    risk_profile = risk_profile or {}
    post_calibration = post_calibration or {}
    risk_features = risk_profile.get("risk_features", {}) if isinstance(risk_profile, dict) else {}

    def risk_value(key, default=0.0):
        if isinstance(risk_profile, dict) and key in risk_profile:
            return risk_profile.get(key, default)
        if isinstance(risk_features, dict):
            return risk_features.get(key, default)
        return default

    avg_ratio = clamp01(avg_model_score / max_score)
    fatal_ratio = clamp01(risk_value("fatal_points_ratio", 0.0))
    perception_risk = clamp01(risk_value("perception_risk", 0.0))
    blank_rate = clamp01(risk_value("blank_rate", 0.0))
    low_quality_rate = clamp01(risk_value("low_quality_rate", 0.0))
    perception_failure_rate = clamp01(risk_value("perception_failure_rate", 0.0))
    uncertainty_index = clamp01(risk_value("uncertainty_index", risk_value("std_ratio", 0.0)))
    spread_ratio = clamp01(risk_value("spread_ratio", 0.0))
    if not spread_ratio and a3wa_decision:
        spread_ratio = clamp01(safe_float(a3wa_decision.get("score_spread", 0.0), 0.0) / max_score)
    partial_ratio = clamp01(risk_value("partial_match_points_ratio", 0.0))
    format_ratio = clamp01(risk_value("format_minor_points_ratio", 0.0))
    unsupported_ratio = clamp01(post_calibration.get("unsupported_match_points_ratio", 0.0))
    lenient_undercredit = clamp01(post_calibration.get("lenient_undercredit_signal", risk_value("lenient_undercredit_signal", 0.0)))
    unsupported_high_score = clamp01(post_calibration.get("unsupported_high_score_risk", risk_value("unsupported_high_score_risk", 0.0)))
    result_correctness = clamp01(post_calibration.get("result_correctness_signal", risk_value("result_correctness_signal", 0.0)))
    method_evidence = clamp01(post_calibration.get("method_evidence_signal", risk_value("method_evidence_signal", 0.0)))
    bare_answer_risk = clamp01(post_calibration.get("bare_answer_risk", risk_value("bare_answer_risk", 0.0)))
    post_upper = safe_float(post_calibration.get("upper_bound", max_score), max_score)
    extraction_risk = clamp01(0.5 * low_quality_rate + 0.5 * perception_failure_rate)

    over_reasons = []
    if unsupported_high_score >= 0.25:
        over_reasons.append("unsupported_high_score")
    if bool(risk_value("high_blank_high_score", False)) and avg_ratio >= 0.80 and unsupported_high_score >= 0.20:
        over_reasons.append("high_blank_high_score")
    if fatal_ratio >= 0.70 and avg_ratio >= 0.75 and unsupported_high_score >= 0.20:
        over_reasons.append("fatal_points_high")
    if unsupported_ratio >= 0.15:
        over_reasons.append("unsupported_match")
    if bool(post_calibration.get("core_anchor_failed", False)):
        over_reasons.append("core_anchor_failed")
    if blank_rate >= 0.50 and avg_ratio >= 0.80 and unsupported_high_score >= 0.20:
        over_reasons.append("blank_high_score")
    if perception_risk >= 0.66 and avg_ratio >= 0.60:
        over_reasons.append("perception_high_score")
    if post_upper < avg_model_score - max(0.02 * max_score, 0.25):
        over_reasons.append("post_upper_cap")

    strong_over_reasons = []
    if unsupported_high_score >= 0.40:
        strong_over_reasons.append("unsupported_high_score_high")
    if bool(risk_value("high_blank_high_score", False)) and avg_ratio >= 0.85 and unsupported_high_score >= 0.30:
        strong_over_reasons.append("high_blank_high_score")
    if fatal_ratio >= 0.80 and unsupported_high_score >= 0.30:
        strong_over_reasons.append("fatal_points_very_high")
    if unsupported_ratio >= 0.25:
        strong_over_reasons.append("unsupported_match_high")
    if bool(post_calibration.get("core_anchor_failed", False)):
        strong_over_reasons.append("core_anchor_failed")
    if post_upper < avg_model_score - max(0.10 * max_score, 0.75):
        strong_over_reasons.append("strict_post_upper_cap")

    under_reasons = []
    if lenient_undercredit >= 0.08:
        under_reasons.append("lenient_undercredit")
    if extraction_risk >= 0.20 and blank_rate <= 0.60 and avg_ratio <= 0.75:
        under_reasons.append("extraction_uncertain_nonblank")
    if uncertainty_index >= 0.10 or spread_ratio >= 0.15:
        under_reasons.append("score_disagreement")
    if format_ratio >= 0.15:
        under_reasons.append("format_minor_mass")
    if partial_ratio >= 0.20 and avg_ratio <= 0.65 and fatal_ratio <= 0.60:
        under_reasons.append("partial_match_mass")

    strong_over_score_risk = bool(strong_over_reasons)
    over_score_risk = bool(over_reasons)
    under_score_risk = bool(under_reasons) and not strong_over_score_risk

    return {
        "over_score_risk": over_score_risk,
        "under_score_risk": under_score_risk,
        "strong_over_score_risk": strong_over_score_risk,
        "over_reasons": over_reasons,
        "under_reasons": under_reasons,
        "strong_over_reasons": strong_over_reasons,
        "avg_ratio": round(avg_ratio, 6),
        "fatal_points_ratio": round(fatal_ratio, 6),
        "perception_risk": round(perception_risk, 6),
        "blank_rate": round(blank_rate, 6),
        "extraction_risk": round(extraction_risk, 6),
        "uncertainty_index": round(uncertainty_index, 6),
        "spread_ratio": round(spread_ratio, 6),
        "partial_match_points_ratio": round(partial_ratio, 6),
        "format_minor_points_ratio": round(format_ratio, 6),
        "unsupported_match_points_ratio": round(unsupported_ratio, 6),
        "lenient_undercredit_signal": round(lenient_undercredit, 6),
        "unsupported_high_score_risk": round(unsupported_high_score, 6),
        "result_correctness_signal": round(result_correctness, 6),
        "method_evidence_signal": round(method_evidence, 6),
        "bare_answer_risk": round(bare_answer_risk, 6),
    }


def apply_boundary_no_harm_gate(
    avg_model_score,
    candidate_score,
    max_score,
    a3wa_decision=None,
    risk_profile=None,
    post_calibration=None,
    lower_bound=None,
    upper_bound=None,
):
    """Accept a BND correction only when its direction has supporting evidence."""
    avg_model_score = safe_float(avg_model_score, 0.0)
    candidate_score = safe_float(candidate_score, avg_model_score)
    max_score = max(safe_float(max_score, 0.0), 1.0)

    signals = build_boundary_direction_signals(
        avg_model_score=avg_model_score,
        max_score=max_score,
        a3wa_decision=a3wa_decision,
        risk_profile=risk_profile,
        post_calibration=post_calibration,
    )

    if lower_bound is None or upper_bound is None:
        lower_bound, upper_bound, _ = a3wa_dynamic_bounds(
            avg_model_score=avg_model_score,
            max_score=max_score,
            a3wa_decision=a3wa_decision,
            risk_profile=risk_profile,
            post_calibration=post_calibration,
        )

    lower_bound = safe_float(lower_bound, 0.0)
    upper_bound = safe_float(upper_bound, max_score)
    baseline = max(0.0, min(max_score, avg_model_score))
    raw_candidate = max(0.0, min(max_score, candidate_score))
    bounded_candidate = max(lower_bound, min(upper_bound, raw_candidate))
    delta = bounded_candidate - baseline
    trivial_margin = max(0.02 * max_score, 0.25)

    accepted = False
    action = "keep_baseline"
    gate_reason = "no_directional_evidence"
    final_score = baseline

    if abs(delta) <= trivial_margin:
        action = "keep_minor_change"
        gate_reason = "minor_change_without_directional_evidence"
    elif delta < 0:
        if signals["over_score_risk"]:
            accepted = True
            action = "accept_lower"
            gate_reason = "over_score_evidence:" + ",".join(signals["over_reasons"])
            final_score = bounded_candidate
        else:
            action = "reject_lower"
    elif delta > 0:
        if signals["strong_over_score_risk"]:
            action = "reject_raise"
            gate_reason = "strong_over_score_evidence:" + ",".join(signals["strong_over_reasons"])
        elif signals["under_score_risk"]:
            accepted = True
            action = "accept_raise"
            gate_reason = "under_score_evidence:" + ",".join(signals["under_reasons"])
            final_score = bounded_candidate
        else:
            action = "reject_raise"

    return {
        "final_score": round(final_score, 4),
        "baseline_score": round(baseline, 4),
        "raw_candidate_score": round(raw_candidate, 4),
        "bounded_candidate_score": round(bounded_candidate, 4),
        "delta_from_baseline": round(final_score - baseline, 4),
        "accepted": accepted,
        "action": action,
        "gate_reason": gate_reason,
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "direction_signals": signals,
    }


def apply_boundary_action_policy(
    avg_model_score,
    candidate_score,
    max_score,
    a3wa_decision=None,
    risk_profile=None,
    post_calibration=None,
):
    """Validation-calibratable BND action policy with model average as baseline."""
    avg_model_score = safe_float(avg_model_score, 0.0)
    candidate_score = safe_float(candidate_score, avg_model_score)
    max_score = max(safe_float(max_score, 0.0), 1.0)
    a3wa_decision = a3wa_decision or {}
    risk_profile = risk_profile or {}
    post_calibration = post_calibration or {}

    baseline = max(0.0, min(max_score, avg_model_score))
    raw_candidate = max(0.0, min(max_score, candidate_score))
    delta = raw_candidate - baseline
    components = a3wa_decision.get("risk_components", {}) or {}
    signals = build_boundary_direction_signals(
        avg_model_score=baseline,
        max_score=max_score,
        a3wa_decision=a3wa_decision,
        risk_profile=risk_profile,
        post_calibration=post_calibration,
    )

    avg_ratio = clamp01(baseline / max_score)
    fatal = clamp01(components.get("U_semantic", signals.get("fatal_points_ratio", 0.0)))
    overcredit = clamp01(components.get("U_overcredit", 0.0))
    blank = clamp01(components.get("U_blank", signals.get("blank_rate", 0.0)))
    score_risk = clamp01(components.get("U_score", signals.get("uncertainty_index", 0.0)))
    extract = clamp01(components.get("U_extract", signals.get("extraction_risk", 0.0)))
    lenient_undercredit = clamp01(
        post_calibration.get(
            "lenient_undercredit_signal",
            components.get("L_lenient_undercredit", signals.get("lenient_undercredit_signal", 0.0)),
        )
    )
    unsupported_high_score = clamp01(
        post_calibration.get(
            "unsupported_high_score_risk",
            components.get("U_unsupported_high_score", signals.get("unsupported_high_score_risk", 0.0)),
        )
    )
    result_correctness = clamp01(post_calibration.get("result_correctness_signal", signals.get("result_correctness_signal", 0.0)))
    result_strong = clamp01(post_calibration.get("result_strong_signal", result_correctness))
    method_evidence = clamp01(post_calibration.get("method_evidence_signal", signals.get("method_evidence_signal", 0.0)))
    bare_answer_risk = clamp01(post_calibration.get("bare_answer_risk", signals.get("bare_answer_risk", 0.0)))
    direct_points_ratio = clamp01(post_calibration.get("direct_points_ratio", 0.0))
    direct_awarded_ratio = clamp01(post_calibration.get("direct_awarded_ratio", 1.0))
    minor_margin = max(0.03 * max_score, 0.3)
    small_margin = max(0.07 * max_score, 0.7)
    large_margin = max(0.15 * max_score, 1.5)

    final_score = baseline
    accepted = False
    action = "keep_baseline"
    gate_reason = "no_profitable_action_evidence"
    parameter_dense_weak_final = (
        direct_points_ratio >= 0.30
        and direct_awarded_ratio < 0.70
        and result_strong < 0.65
    )
    lenient_raise_ready = (
        lenient_undercredit >= 0.08
        and result_correctness >= 0.65
        and (method_evidence >= 0.10 or signals.get("partial_match_points_ratio", 0.0) >= 0.15)
        and bare_answer_risk < 0.90
        and unsupported_high_score < 0.35
        and avg_ratio <= 0.90
        and not parameter_dense_weak_final
    )
    strong_lenient_raise = lenient_raise_ready and lenient_undercredit >= 0.16 and avg_ratio <= 0.75

    if abs(delta) <= minor_margin:
        if strong_lenient_raise:
            high_band_gap = max(0.0, 0.85 * max_score - baseline)
            final_score = min(max_score, baseline + min(small_margin, high_band_gap))
            accepted = final_score > baseline
            action = "auto_small_raise" if accepted else "keep_minor_change"
            gate_reason = "lenient_undercredit_without_agent_delta" if accepted else "minor_candidate_delta"
        else:
            action = "keep_minor_change"
            gate_reason = "minor_candidate_delta"
    elif delta < 0:
        strong_lower = unsupported_high_score >= 0.40 and avg_ratio >= 0.70
        supported_lower = (
            avg_ratio >= 0.75
            and unsupported_high_score >= 0.25
            and (fatal >= 0.60 or blank >= 0.60 or overcredit >= 0.25 or bare_answer_risk >= 0.50)
        )
        if strong_lower:
            final_score = max(0.0, baseline - min(abs(delta), large_margin))
            accepted = True
            action = "large_lower"
            gate_reason = "strong_unsupported_high_score_evidence"
        elif supported_lower:
            final_score = max(0.0, baseline - min(abs(delta), small_margin))
            accepted = True
            action = "small_lower"
            gate_reason = "supported_unsupported_high_score_evidence"
        else:
            action = "reject_lower"
            gate_reason = "lower_without_sufficient_unsupported_high_score_evidence"
    elif delta > 0:
        supported_raise = (
            lenient_raise_ready
            or (
                avg_ratio <= 0.70
                and (score_risk >= 0.12 or extract >= 0.20)
                and unsupported_high_score < 0.25
            )
        )
        if supported_raise:
            margin = large_margin if strong_lenient_raise else small_margin
            final_score = min(max_score, baseline + min(delta, margin))
            accepted = True
            action = "medium_raise" if strong_lenient_raise else "small_raise"
            gate_reason = "supported_lenient_undercredit_evidence" if lenient_raise_ready else "supported_undercredit_evidence"
        else:
            action = "reject_raise"
            gate_reason = "raise_without_sufficient_undercredit_evidence"

    return {
        "final_score": round(final_score, 4),
        "baseline_score": round(baseline, 4),
        "raw_candidate_score": round(raw_candidate, 4),
        "bounded_candidate_score": round(final_score, 4),
        "delta_from_baseline": round(final_score - baseline, 4),
        "accepted": accepted,
        "action": action,
        "gate_reason": gate_reason,
        "lower_bound": round(max(0.0, baseline - large_margin), 4),
        "upper_bound": round(min(max_score, baseline + large_margin), 4),
        "direction_signals": signals,
    }


def calibrated_bounds(avg_model_score, max_score, risk_profile, post_calibration):
    avg_model_score = safe_float(avg_model_score, 0.0)
    max_score = max(safe_float(max_score, 0.0), 1.0)
    risk_profile = risk_profile or {}
    post_calibration = post_calibration or {}

    fatal_points_ratio = safe_float(risk_profile.get("fatal_points_ratio", 0.0), 0.0)
    perception_risk = safe_float(risk_profile.get("perception_risk", 0.0), 0.0)
    high_blank_high_score = bool(risk_profile.get("high_blank_high_score", False))

    has_over_score_signal = (
        high_blank_high_score
        or fatal_points_ratio >= 0.30
        or perception_risk >= 0.33
        or safe_float(post_calibration.get("unsupported_match_points_ratio", 0.0), 0.0) >= 0.15
        or bool(post_calibration.get("core_anchor_failed", False))
    )
    strong_over_score_signal = (
        high_blank_high_score
        or fatal_points_ratio >= 0.50
        or perception_risk >= 0.66
        or safe_float(post_calibration.get("unsupported_match_points_ratio", 0.0), 0.0) >= 0.25
        or bool(post_calibration.get("core_anchor_failed", False))
    )

    raise_cap = max_score * (0.15 if has_over_score_signal else 0.30)
    lower_cap = max_score * 0.15
    lower_bound = max(0.0, avg_model_score - lower_cap)
    upper_bound = min(max_score, avg_model_score + raise_cap)

    if strong_over_score_signal:
        upper_bound = min(upper_bound, avg_model_score)
    post_upper_bound = safe_float(post_calibration.get("upper_bound", max_score), max_score)
    if post_upper_bound <= lower_bound and post_upper_bound < avg_model_score:
        lower_bound = 0.0
    upper_bound = min(upper_bound, post_upper_bound)
    lower_bound = max(lower_bound, safe_float(post_calibration.get("lower_bound", 0.0), 0.0))
    if lower_bound > upper_bound:
        lower_bound = 0.0

    return lower_bound, upper_bound, {
        "has_over_score_signal": has_over_score_signal,
        "strong_over_score_signal": strong_over_score_signal,
    }
