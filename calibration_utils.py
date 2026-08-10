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

NUMERIC_TYPES = {"direct_numeric", "derived_numeric", "numeric", "base_number"}
METHOD_TYPES = {"formula", "method"}
VISUAL_TYPES = {"sequence", "relation", "diagram_relation", "table_entry", "structured_fields", "diagram_ocr"}
STRUCTURED_TYPES = {"base_number", "bit_vector", "sequence", "set", "relation", "diagram_relation", "table_entry", "structured_fields", "diagram_ocr"}
TRUSTED_METADATA_THRESHOLD = 0.80
A3WA_RISK_WEIGHTS = {
    "extract": 0.40,
    "score": 0.25,
    "semantic": 0.20,
    "blank": 0.00,
    "overcredit": 0.15,
}
A3WA_LOSS_PARAMS = {
    "lambda1": 5.0,
    "lambda2": 2.0,
    "mu1": 2.0,
    "mu2": 5.0,
    "m": 0.4,
}

SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
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


def normalized_primary_risk_weights(weights=None):
    """Map legacy routing weights onto the three paper-facing risks."""
    legacy = normalized_risk_weights(weights)
    primary = {
        "U_E": legacy["extract"] + legacy["blank"],
        "U_S": legacy["score"],
        "U_R": legacy["semantic"] + legacy["overcredit"],
    }
    total = sum(primary.values())
    if total <= 1e-12:
        return {"U_E": 1.0 / 3.0, "U_S": 1.0 / 3.0, "U_R": 1.0 / 3.0}
    return {key: value / total for key, value in primary.items()}


def sigmoid(value):
    value = max(-40.0, min(40.0, safe_float(value, 0.0)))
    return 1.0 / (1.0 + math.exp(-value))


def conformal_score_interval(
    center_score,
    max_score,
    score_spread_norm,
    config=None,
):
    """Build a locally scaled split-conformal score interval.

    The quantile is fitted only on validation residuals. The interval is an
    uncertainty signal, not a score correction.
    """
    center_score = safe_float(center_score, 0.0)
    max_score = max(safe_float(max_score, 0.0), 1.0)
    score_spread_norm = clamp01(score_spread_norm)
    config = config if isinstance(config, dict) else {}
    enabled = bool(config.get("enabled", False))
    scale_floor = max(safe_float(config.get("scale_floor", 0.05), 0.05), 1e-6)
    quantile = max(safe_float(config.get("nonconformity_quantile", 0.0), 0.0), 0.0)
    tolerance_ratio = max(safe_float(config.get("safe_tolerance_ratio", 0.10), 0.10), 1e-6)
    local_scale = max(score_spread_norm, scale_floor)
    half_width_ratio = quantile * local_scale if enabled else 0.0
    half_width = half_width_ratio * max_score
    lower = max(0.0, center_score - half_width)
    upper = min(max_score, center_score + half_width)
    stability_risk = clamp01(half_width_ratio / tolerance_ratio) if enabled else None
    return {
        "enabled": enabled,
        "coverage": safe_float(config.get("coverage", 0.90), 0.90),
        "lower": round(lower, 6),
        "upper": round(upper, 6),
        "half_width": round(half_width, 6),
        "half_width_ratio": round(half_width_ratio, 6),
        "local_scale": round(local_scale, 6),
        "stability_risk": None if stability_risk is None else round(stability_risk, 6),
    }


def calibrated_a3wa_membership(u_e, u_s, u_r, weights=None, model=None):
    """Return monotonic membership in the safe-auto-grading fuzzy set."""
    risks = {
        "U_E": clamp01(u_e),
        "U_S": clamp01(u_s),
        "U_R": clamp01(u_r),
    }
    if isinstance(model, dict) and model.get("type") == "monotonic_logistic":
        coefficients = model.get("coefficients", {})
        intercept = safe_float(model.get("intercept", 0.0), 0.0)
        linear = intercept
        for key, risk in risks.items():
            coefficient = max(safe_float(coefficients.get(key, 0.0), 0.0), 0.0)
            linear -= coefficient * risk
        mu = sigmoid(linear)
        return {
            "mu": round(mu, 6),
            "risk": round(1.0 - mu, 6),
            "source": "validation_monotonic_logistic",
        }

    primary_weights = normalized_primary_risk_weights(weights)
    risk = sum(primary_weights[key] * risks[key] for key in risks)
    return {
        "mu": round(1.0 - clamp01(risk), 6),
        "risk": round(clamp01(risk), 6),
        "source": "weighted_primary_risks",
    }


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


def infer_canonicalization(item, meta=None):
    """Infer a generic evidence-normalization operator for a rubric item."""
    if not isinstance(item, dict):
        return "none"
    if item.get("canonicalization"):
        return str(item.get("canonicalization"))
    meta = meta or classify_rubric_item(item)
    answer_type = str(meta.get("answer_type", "unknown"))
    text = str(item.get("item", ""))
    norm = _normalized_text(text)
    if answer_type == "base_number":
        if re.search(r"\b[0-9A-Fa-f]+\s*[Hh]\b", norm) or re.search(r"\b[01]{6,}\b", norm):
            return "base_number"
        return "numeric_representation"
    if answer_type == "bit_vector":
        return "bit_vector"
    if answer_type == "sequence":
        return "sequence"
    if answer_type == "set":
        return "set"
    if answer_type in ("relation", "diagram_relation"):
        return "graph_relation"
    if answer_type == "table_entry":
        return "table"
    if answer_type in NUMERIC_TYPES:
        return "numeric"
    if answer_type in METHOD_TYPES:
        return "formula"
    return "semantic_text"


def infer_evidence_source(item, meta=None):
    """Infer whether an item normally needs text, formula, table, or diagram evidence."""
    if not isinstance(item, dict):
        return "text"
    if item.get("evidence_source"):
        return str(item.get("evidence_source"))
    meta = meta or classify_rubric_item(item)
    answer_type = str(meta.get("answer_type", "unknown"))
    if answer_type in ("relation", "diagram_relation", "diagram_ocr"):
        return "diagram"
    if answer_type == "table_entry":
        return "table"
    if answer_type in METHOD_TYPES:
        return "formula"
    return "text"


def infer_score_layer(item, meta=None):
    """Classify rubric credit into core/support/auxiliary layers for BND arbitration."""
    if not isinstance(item, dict):
        return "support"
    explicit = str(item.get("score_layer", "")).strip().lower()
    if explicit in {"core", "support", "auxiliary"}:
        return explicit

    meta = meta or classify_rubric_item(item)
    answer_type = str(meta.get("answer_type", "unknown"))
    role = str(meta.get("role", "unknown"))
    text = _normalized_text(
        " ".join(
            str(item.get(key, ""))
            for key in ("item", "source_text", "parent_official_item", "standard_answer_text")
        )
    ).lower()

    auxiliary_markers = (
        "单位", "格式", "符号", "名称", "标注", "说明", "写出单位",
        "unit", "format", "notation", "label",
    )
    if role == "final" or answer_type in {
        "judgement",
        "relation",
        "diagram_relation",
        "bit_vector",
        "base_number",
    }:
        return "core"
    if role in {"method", "intermediate", "parameter"} or answer_type in {
        "formula",
        "derived_numeric",
        "sequence",
        "table_entry",
    }:
        return "support"
    if any(marker in text for marker in auxiliary_markers):
        return "auxiliary"
    return "support"


def _normalized_text(text):
    return (
        str(text or "")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("×", "x")
        .replace("＊", "*")
        .replace("·", "*")
        .replace("÷", "/")
        .replace("／", "/")
        .replace("µ", "μ")
    )


def _canonical_unit(unit):
    raw = str(unit or "").strip()
    raw = raw.replace("µ", "μ")
    lower = raw.lower()
    aliases = {
        "秒": "s", "sec": "s", "second": "s", "seconds": "s",
        "毫秒": "ms", "微秒": "us", "μs": "us", "us": "us",
        "microsecond": "us", "microseconds": "us", "纳秒": "ns",
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

    # Scientific notation variants: 4x10^9, 4×10⁹, 10⁵, 5/3×10⁻⁵.
    sci_text = raw.translate(SUPERSCRIPT_MAP)
    for match in re.finditer(
        rf"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*[x\*]?\s*10\s*\^?\s*(-?\d+)\s*{unit_pat}?",
        sci_text,
        re.I,
    ):
        numerator = safe_float(match.group(1), None)
        denominator = safe_float(match.group(2), None)
        exp = safe_float(match.group(3), None)
        if (
            numerator is not None
            and denominator not in (None, 0)
            and exp is not None
            and abs(exp) <= 32
        ):
            value = numerator / denominator * (10 ** int(exp))
            unit = match.group(4) if len(match.groups()) >= 4 else ""
            converted = _convert_unit(value, unit, target_unit) if unit and target_unit else value
            if converted is not None:
                candidates.append(converted)

    for match in re.finditer(rf"(-?\d+(?:\.\d+)?)\s*[x\*]\s*10\s*\^?\s*(-?\d+)\s*{unit_pat}?", sci_text, re.I):
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

    for match in re.finditer(rf"(-?\d+(?:\.\d+)?)\s*[eE]\s*([+-]?\d+)\s*{unit_pat}?", sci_text, re.I):
        base = safe_float(match.group(1), None)
        exp = safe_float(match.group(2), None)
        if base is not None and exp is not None and abs(exp) <= 32:
            value = base * (10 ** int(exp))
            unit = match.group(3) if len(match.groups()) >= 3 else ""
            converted = _convert_unit(value, unit, target_unit) if unit and target_unit else value
            if converted is not None:
                candidates.append(converted)

    # Power notation commonly used in CS answers, e.g. 2^9.
    for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*\^\s*(-?\d+)", sci_text):
        base = safe_float(match.group(1), None)
        exp = safe_float(match.group(2), None)
        if base is not None and exp is not None and abs(exp) <= 16:
            candidates.append(base ** int(exp))

    plain_text = re.sub(
        rf"-?\d+(?:\.\d+)?\s*/\s*-?\d+(?:\.\d+)?\s*[x\*]?\s*10\s*\^?\s*-?\d+\s*{unit_pat}?",
        " ",
        sci_text,
        flags=re.I,
    )
    plain_text = re.sub(
        rf"-?\d+(?:\.\d+)?\s*[x\*]\s*10\s*\^?\s*-?\d+\s*{unit_pat}?",
        " ",
        plain_text,
        flags=re.I,
    )
    plain_text = re.sub(rf"\b10\s*\^?\s*-?\d+\s*{unit_pat}?", " ", plain_text, flags=re.I)
    plain_text = re.sub(rf"-?\d+(?:\.\d+)?\s*[eE]\s*[+-]?\d+\s*{unit_pat}?", " ", plain_text, flags=re.I)

    # Plain numbers, with Chinese large-number suffixes handled locally.
    for match in re.finditer(rf"(-?\d+(?:,\d{{3}})*(?:\.\d+)?|-?\d+(?:\.\d+)?)(万|亿)?\s*{unit_pat}?", plain_text, re.I):
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
    if isinstance(rubric_item, dict):
        text_candidates = [
            rubric_item.get("standard_answer_text", ""),
            rubric_item.get("item", ""),
            rubric_item.get("source_text", ""),
            rubric_item.get("parent_official_item", ""),
        ]
    else:
        text_candidates = [str(rubric_item)]

    for text in text_candidates:
        raw = _normalized_text(text)
        if not raw.strip():
            continue
        if re.search(r"(?:[x\*]\s*10|10\s*\^|[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]|\d\s*[eE]\s*[+-]?\d)", str(text)):
            candidates = extract_numeric_candidates(raw, target_unit=infer_unit(raw))
            if candidates:
                return candidates[0]
        matches = list(re.finditer(r"(-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)(万|亿)?", raw))
        if matches:
            last = matches[-1]
            return _convert_number_token(last.group(1), last.group(2) or "")
    return None


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
            for key in (
                "answer_type", "role", "expected", "unit", "formula",
                "expected_formula", "depends_on", "canonicalization",
                "evidence_source", "dependency_group", "score_layer",
            )
        )
        meta = classify_rubric_item(item)
        item.setdefault("answer_type", meta["answer_type"])
        item.setdefault("role", meta["role"])
        item.setdefault("canonicalization", infer_canonicalization(item, meta))
        item.setdefault("evidence_source", infer_evidence_source(item, meta))
        item.setdefault("score_layer", infer_score_layer(item, meta))
        item.setdefault("source_text", item.get("item", ""))
        item.setdefault("parent_official_item", item.get("parent_id", ""))

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
            elif item["answer_type"] in ("bit_vector", "set"):
                item["metadata_source"] = "auto"
                item["metadata_hard_enabled"] = False
                confidence = 0.75
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
    norm_text = _normalized_text(text)
    explicit_type = str(item.get("answer_type", "")).strip()
    explicit_role = str(item.get("role", "")).strip()

    if explicit_type:
        answer_type = explicit_type
    elif re.search(r"\b[0-9A-Fa-f]+\s*[Hh]\b", norm_text):
        answer_type = "base_number"
    elif re.search(r"(?:\b[01]\s*){4,}", norm_text):
        answer_type = "bit_vector"
    elif re.search(r"\b[A-Za-z]\s*(?:->|=>|>|<|\.|-)\s*[A-Za-z]", norm_text):
        answer_type = "sequence"
    elif re.search(r"[A-Za-z](?:\s*[,/]\s*[A-Za-z]){1,}", norm_text):
        answer_type = "set"
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
    elif answer_type in ("base_number", "bit_vector", "sequence"):
        role = "final"
    elif answer_type in ("relation", "diagram_relation"):
        role = "intermediate"
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


def infer_rubric_task_profile(rubrics_data, max_score=None):
    """Infer task-level scoring structure from rubric item metadata.

    This avoids using a question id or a raw full-score threshold to decide
    whether lenient calculation/derivation grading is appropriate.
    """
    rubrics_data = rubrics_data if isinstance(rubrics_data, list) else []
    points_values = [max(0.0, safe_float(item.get("points", 0.0), 0.0)) for item in rubrics_data]
    total_points = sum(points_values)
    if total_points <= 1e-9:
        total_points = max(safe_float(max_score, 0.0), 1.0)

    type_counts = Counter()
    type_points = Counter()
    positive_points = [points for points in points_values if points > 0]
    item_count = len(positive_points)
    max_item_points = max(positive_points) if positive_points else 0.0
    max_item_points_ratio = clamp01(max_item_points / total_points)
    small_item_count = sum(1 for points in positive_points if points <= max(2.0, 0.12 * total_points))
    small_item_ratio = clamp01(small_item_count / max(item_count, 1))
    process_points = 0.0
    result_points = 0.0
    parameter_points = 0.0
    numeric_formula_points = 0.0
    visual_or_sequence_points = 0.0
    concept_judgement_points = 0.0
    result_sufficiency_points = 0.0
    explicit_metadata_points = 0.0

    for item, points in zip(rubrics_data, points_values):
        meta = classify_rubric_item(item)
        answer_type = str(meta.get("answer_type", "unknown"))
        role = str(meta.get("role", "unknown"))
        scoring_role = str(item.get("scoring_role", "")).strip().lower()
        task_semantics = str(item.get("task_semantics", "")).strip().lower()
        type_counts[answer_type] += 1
        type_points[answer_type] += points

        if safe_float(item.get("metadata_confidence", 0.0), 0.0) >= TRUSTED_METADATA_THRESHOLD:
            explicit_metadata_points += points

        is_parameter = role == "parameter" or answer_type == "direct_numeric"
        is_result = role == "final" or scoring_role == "final"
        is_process = (
            role in ("method", "intermediate")
            or scoring_role in ("support_process", "core_process")
            or answer_type in METHOD_TYPES
            or answer_type in ("derived_numeric", "sequence", "table_entry")
        )
        is_numeric_formula = answer_type in NUMERIC_TYPES or answer_type in METHOD_TYPES
        is_visual_or_sequence = answer_type in VISUAL_TYPES or answer_type in ("sequence", "table_entry")
        is_concept_judgement = answer_type in ("concept_keyword", "judgement")

        if is_parameter:
            parameter_points += points
        if is_result:
            result_points += points
        if is_process:
            process_points += points
        if is_numeric_formula:
            numeric_formula_points += points
        if is_visual_or_sequence:
            visual_or_sequence_points += points
        if is_concept_judgement:
            concept_judgement_points += points
        if (
            task_semantics == "result_sufficient"
            or item.get("scoring_policy") == "final_sufficient_partial_credit"
        ):
            result_sufficiency_points += points

    process_ratio = clamp01(process_points / total_points)
    result_ratio = clamp01(result_points / total_points)
    parameter_ratio = clamp01(parameter_points / total_points)
    numeric_formula_ratio = clamp01(numeric_formula_points / total_points)
    visual_sequence_ratio = clamp01(visual_or_sequence_points / total_points)
    concept_judgement_ratio = clamp01(concept_judgement_points / total_points)
    metadata_ratio = clamp01(explicit_metadata_points / total_points)
    result_sufficiency_ratio = clamp01(result_sufficiency_points / total_points)
    fragmented_rubric = item_count >= 10 and max_item_points_ratio <= 0.15 and small_item_ratio >= 0.70
    concentrated_result_weight = max_item_points_ratio >= 0.25 or result_ratio >= 0.45

    calculation_or_derivation = (
        numeric_formula_ratio >= 0.45
        and process_ratio >= 0.25
        and concept_judgement_ratio < 0.70
    )
    algorithmic_or_mapping = (
        visual_sequence_ratio >= 0.25
        or (
            process_ratio >= 0.35
            and numeric_formula_ratio >= 0.25
            and concept_judgement_ratio < 0.60
        )
    )
    final_answer_weight_high = (
        result_sufficiency_ratio >= 0.50 or result_ratio >= 0.45
    )
    concept_dominant = concept_judgement_ratio >= 0.60 and numeric_formula_ratio < 0.35
    complex_derivation_task = (
        not concept_dominant
        and (
            calculation_or_derivation
            or algorithmic_or_mapping
            or (process_ratio >= 0.50 and numeric_formula_ratio >= 0.35)
        )
    )
    upper_consensus_eligible = (
        complex_derivation_task
        and process_ratio >= 0.60
        and numeric_formula_ratio >= 0.45
        and concept_judgement_ratio < 0.60
    )

    if complex_derivation_task and visual_sequence_ratio >= 0.25:
        task_type = "algorithmic_or_mapping"
    elif complex_derivation_task:
        task_type = "calculation_derivation"
    elif concept_dominant:
        task_type = "concept_or_judgement"
    else:
        task_type = "mixed_or_unknown"

    return {
        "task_type": task_type,
        "complex_derivation_task": bool(complex_derivation_task),
        "upper_consensus_eligible": bool(upper_consensus_eligible),
        "calculation_or_derivation": bool(calculation_or_derivation),
        "algorithmic_or_mapping": bool(algorithmic_or_mapping),
        "final_answer_weight_high": bool(final_answer_weight_high),
        "concept_dominant": bool(concept_dominant),
        "process_points_ratio": round(process_ratio, 6),
        "result_points_ratio": round(result_ratio, 6),
        "result_sufficiency_ratio": round(result_sufficiency_ratio, 6),
        "parameter_points_ratio": round(parameter_ratio, 6),
        "numeric_formula_points_ratio": round(numeric_formula_ratio, 6),
        "visual_sequence_points_ratio": round(visual_sequence_ratio, 6),
        "concept_judgement_points_ratio": round(concept_judgement_ratio, 6),
        "metadata_points_ratio": round(metadata_ratio, 6),
        "max_item_points_ratio": round(max_item_points_ratio, 6),
        "small_item_ratio": round(small_item_ratio, 6),
        "fragmented_rubric": bool(fragmented_rubric),
        "concentrated_result_weight": bool(concentrated_result_weight),
        "answer_type_counts": dict(type_counts),
        "answer_type_points": {key: round(value, 4) for key, value in type_points.items()},
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


def _has_formula_structure(text):
    text = _normalized_text(text)
    if any(op in text for op in ("=", "+", "-", "*", "/", "x", "^", "->", "=>", ":=")):
        return True
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", text):
        return True
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*[<>]=?", text):
        return True
    return False


def _has_sequence_structure(text):
    text = str(text or "").strip()
    compact = re.sub(r"\s+", "", text)
    if len(re.findall(r"[01]", compact)) >= 4:
        return True
    if len(re.findall(r"[0-9A-Fa-f]", compact)) >= 4 and re.search(r"[HhBb]$", compact):
        return True
    separators = (" ", ",", "，", "->", "=>", "-", "|", "/", ";", "；")
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text)
    return len(tokens) >= 2 and any(sep in text for sep in separators)


def _has_table_mapping_structure(text):
    text = str(text or "").strip()
    if any(sep in text for sep in ("=", ":", "：", "->", "=>", ",", "，", ";", "；", "|", "\t")):
        return True
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text)
    return len(tokens) >= 3


def is_structure_missing_extraction(value, rubric_item=None):
    """Whether an extracted fact lacks the structure required by its rubric type.

    This is a lightweight support check, not grading. It is designed for
    computer-science exam answers such as numbers, formulas, sequences, tables,
    mappings, and short judgements.
    """
    text = str(value).strip()
    if not text:
        return True
    if is_blank_extraction(text) or is_perception_failure(text):
        return False
    if is_low_quality_extraction(text, rubric_item):
        return False

    meta = classify_rubric_item(rubric_item or {})
    answer_type = meta.get("answer_type", "unknown")
    role = meta.get("role", "unknown")

    if answer_type == "judgement":
        return False
    if answer_type == "concept_keyword" and role not in ("method", "intermediate", "final"):
        return False
    if answer_type == "bit_vector":
        return len(re.findall(r"[01]", re.sub(r"\s+", "", text))) < 4
    if answer_type == "relation":
        return not _has_sequence_structure(text)
    if answer_type == "set":
        return len(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text)) < 2
    if answer_type in NUMERIC_TYPES:
        return not bool(extract_numeric_candidates(text))
    if answer_type in METHOD_TYPES:
        return not (_has_formula_structure(text) or len(text) >= 6)
    if answer_type == "sequence":
        return not _has_sequence_structure(text)
    if answer_type in ("table_entry", "diagram_ocr"):
        return not _has_table_mapping_structure(text)
    return False


def compute_extraction_quality_counts(facts_dict, rubrics_data):
    """Count blank, perception-failure, and rubric-aware low-quality facts."""
    facts_dict = facts_dict if isinstance(facts_dict, dict) else {}
    rubrics_data = rubrics_data if isinstance(rubrics_data, list) else []
    blank_count = 0
    perception_fail_count = 0
    low_quality_count = 0
    structure_missing_count = 0
    suspicious_items = []

    if rubrics_data:
        for item in rubrics_data:
            item_id = str(item.get("id", ""))
            value = facts_dict.get(item_id, "") if item_id else ""
            if is_blank_extraction(value):
                blank_count += 1
                suspicious_items.append({"id": item_id, "reason": "blank"})
            elif is_perception_failure(value):
                perception_fail_count += 1
                suspicious_items.append({"id": item_id, "reason": "perception_failure"})
            elif is_low_quality_extraction(value, item):
                low_quality_count += 1
                suspicious_items.append({"id": item_id, "reason": "low_quality"})
            elif is_structure_missing_extraction(value, item):
                structure_missing_count += 1
                suspicious_items.append({"id": item_id, "reason": "structure_missing"})
        total = len(rubrics_data)
    else:
        for item_id, value in facts_dict.items():
            if is_blank_extraction(value):
                blank_count += 1
                suspicious_items.append({"id": str(item_id), "reason": "blank"})
            elif is_perception_failure(value):
                perception_fail_count += 1
                suspicious_items.append({"id": str(item_id), "reason": "perception_failure"})
            elif is_low_quality_extraction(value, None):
                low_quality_count += 1
                suspicious_items.append({"id": str(item_id), "reason": "low_quality"})
            elif is_structure_missing_extraction(value, None):
                structure_missing_count += 1
                suspicious_items.append({"id": str(item_id), "reason": "structure_missing"})
        total = len(facts_dict)

    return {
        "blank_count": blank_count,
        "perception_fail_count": perception_fail_count,
        "low_quality_count": low_quality_count,
        "structure_missing_count": structure_missing_count,
        "suspicious_items": suspicious_items,
        "total_items": total,
    }


def compute_extraction_risk_features(extraction_counts):
    """Build a rubric-agnostic extraction-risk profile from quality counts."""
    extraction_counts = extraction_counts or {}
    total = max(safe_float(extraction_counts.get("total_items", 0.0), 0.0), 1.0)
    blank_rate = clamp01(safe_float(extraction_counts.get("blank_count", 0.0), 0.0) / total)
    low_quality_rate = clamp01(safe_float(extraction_counts.get("low_quality_count", 0.0), 0.0) / total)
    perception_failure_rate = clamp01(safe_float(extraction_counts.get("perception_fail_count", 0.0), 0.0) / total)
    structure_missing_rate = clamp01(safe_float(extraction_counts.get("structure_missing_count", 0.0), 0.0) / total)
    suspicious_rate = clamp01(
        (
            safe_float(extraction_counts.get("blank_count", 0.0), 0.0)
            + safe_float(extraction_counts.get("low_quality_count", 0.0), 0.0)
            + safe_float(extraction_counts.get("perception_fail_count", 0.0), 0.0)
            + safe_float(extraction_counts.get("structure_missing_count", 0.0), 0.0)
        )
        / total
    )
    u_extract = clamp01(
        0.40 * blank_rate
        + 0.25 * low_quality_rate
        + 0.20 * perception_failure_rate
        + 0.15 * structure_missing_rate
    )
    if u_extract >= 0.50:
        extraction_quality = "failed"
    elif u_extract >= 0.20:
        extraction_quality = "low"
    else:
        extraction_quality = "high"
    return {
        "blank_rate": round(blank_rate, 6),
        "low_quality_rate": round(low_quality_rate, 6),
        "perception_failure_rate": round(perception_failure_rate, 6),
        "structure_missing_rate": round(structure_missing_rate, 6),
        "suspicious_extraction_rate": round(suspicious_rate, 6),
        "extraction_risk": round(u_extract, 6),
        "extraction_quality": extraction_quality,
    }


def _numeric_value_matches(fact_value, expected, tolerance=0.10, target_unit=None):
    if expected is None:
        return None
    candidates = extract_numeric_candidates(fact_value, target_unit=target_unit)
    if not candidates:
        return None
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


def compute_bidirectional_credit_risks(facts_dict, rubrics_data, strict_cots):
    """Compute parameter-free, item-level credit discrepancy diagnostics.

    The returned signals are diagnostic candidates only. They compare the
    allocation of rubric credit across independent probes and, where trusted
    structured metadata is available, compare that credit with deterministic
    fact support. Teacher scores are deliberately not used.
    """
    facts_dict = facts_dict if isinstance(facts_dict, dict) else {}
    rubrics_data = rubrics_data if isinstance(rubrics_data, list) else []
    strict_cots = [cot for cot in (strict_cots or []) if isinstance(cot, dict)]
    points_map, fallback_points = rubric_points_map(rubrics_data)
    details_by_item = _detail_by_item(strict_cots)

    total_points = 0.0
    allocation_under = 0.0
    allocation_over = 0.0
    allocation_range = 0.0
    deterministic_under = 0.0
    deterministic_over = 0.0
    deterministic_points = 0.0
    missing_judgements = 0.0
    item_diagnostics = []

    for item in rubrics_data:
        item_id = str(item.get("id", ""))
        points = max(points_map.get(item_id, fallback_points), 0.0)
        if not item_id or points <= 0.0:
            continue

        details = details_by_item.get(item_id, [])
        score_ratios = []
        for detail in details:
            score_ratios.append(
                clamp01(safe_float(detail.get("score_given", 0.0), 0.0) / points)
            )

        observed = len(score_ratios)
        expected = len(strict_cots)
        missing_ratio = (
            max(expected - observed, 0) / expected if expected > 0 else 0.0
        )
        missing_judgements += points * missing_ratio

        mean_credit = sum(score_ratios) / observed if observed else 0.0
        maximum_credit = max(score_ratios) if score_ratios else 0.0
        minimum_credit = min(score_ratios) if score_ratios else 0.0
        item_allocation_under = max(0.0, maximum_credit - mean_credit)
        item_allocation_over = max(0.0, mean_credit - minimum_credit)
        item_allocation_range = max(0.0, maximum_credit - minimum_credit)
        allocation_under += points * item_allocation_under
        allocation_over += points * item_allocation_over
        allocation_range += points * item_allocation_range

        meta = classify_rubric_item(item)
        answer_type = meta["answer_type"]
        fact_value = facts_dict.get(item_id, "")
        hard_metadata_enabled = (
            safe_float(item.get("metadata_confidence", 0.0), 0.0)
            >= TRUSTED_METADATA_THRESHOLD
            and bool(item.get("metadata_hard_enabled", False))
        )
        deterministic_match = None
        if hard_metadata_enabled and answer_type in NUMERIC_TYPES:
            expected_number = infer_expected_number(item)
            target_unit = item.get("unit") or infer_unit(
                item.get("standard_answer_text") or item.get("item", "")
            )
            deterministic_match = _numeric_value_matches(
                fact_value,
                expected_number,
                target_unit=target_unit,
            )
        elif hard_metadata_enabled and answer_type in METHOD_TYPES:
            deterministic_match = _formula_is_supported(fact_value, item)

        if deterministic_match is not None:
            deterministic_points += points
            if deterministic_match is True:
                deterministic_under += points * max(0.0, 1.0 - mean_credit)
            else:
                deterministic_over += points * mean_credit

        total_points += points
        item_diagnostics.append({
            "id": item_id,
            "points": round(points, 6),
            "observed_probes": observed,
            "expected_probes": expected,
            "mean_credit_ratio": round(mean_credit, 6),
            "allocation_undercredit": round(item_allocation_under, 6),
            "allocation_overcredit": round(item_allocation_over, 6),
            "allocation_range": round(item_allocation_range, 6),
            "deterministic_match": deterministic_match,
        })

    denominator = max(total_points, 1e-9)
    deterministic_denominator = max(deterministic_points, 1e-9)
    return {
        "U_R_allocation_undercredit": round(allocation_under / denominator, 6),
        "U_R_allocation_overcredit": round(allocation_over / denominator, 6),
        "U_R_allocation_disagreement": round(allocation_range / denominator, 6),
        "U_R_deterministic_undercredit": round(
            deterministic_under / deterministic_denominator, 6
        ) if deterministic_points > 0 else 0.0,
        "U_R_deterministic_overcredit": round(
            deterministic_over / deterministic_denominator, 6
        ) if deterministic_points > 0 else 0.0,
        "deterministic_coverage": round(deterministic_points / denominator, 6),
        "missing_judgement_risk": round(missing_judgements / denominator, 6),
        "fusion_status": "diagnostic_only",
        "item_diagnostics": item_diagnostics,
    }


def _majority_category(details):
    categories = [str(d.get("error_category", "")) for d in details]
    return Counter(categories).most_common(1)[0][0] if categories else ""


def compute_three_way_primary_risks(
    facts_dict,
    rubrics_data,
    strict_cots,
    model_scores,
    max_score,
    gamma=0.20,
):
    """Compress routing evidence into three paper-facing risk variables.

    U_E: evidence quality risk, whether facts were reliably extracted.
    U_S: score stability risk, whether repeated grading disagrees.
    U_R: rubric adaptation risk, whether evidence can be mapped to rubric items.
    """
    facts_dict = facts_dict if isinstance(facts_dict, dict) else {}
    rubrics_data = rubrics_data if isinstance(rubrics_data, list) else []
    strict_cots = strict_cots or []
    model_scores = [safe_float(score, None) for score in (model_scores or [])]
    model_scores = [score for score in model_scores if score is not None]
    max_score = max(safe_float(max_score, 0.0), 1.0)
    points_map, fallback_points = rubric_points_map(rubrics_data)
    details_by_item = _detail_by_item(strict_cots)

    evidence_quality_sum = 0.0
    adaptation_sum = 0.0
    total_points = 0.0
    item_risks = []

    for item in rubrics_data:
        item_id = str(item.get("id", ""))
        points = max(points_map.get(item_id, fallback_points), 0.0)
        if points <= 0:
            continue
        value = facts_dict.get(item_id, "") if item_id else ""
        details = details_by_item.get(item_id, [])
        majority_category = _majority_category(details)

        if is_blank_extraction(value) or is_perception_failure(value):
            q_i = 0.0
        elif is_low_quality_extraction(value, item) or is_structure_missing_extraction(value, item):
            q_i = 0.5
        else:
            q_i = 1.0

        if q_i <= 0.0:
            a_i = 0.0
        elif not details:
            a_i = 0.5 * q_i
        elif majority_category in ("MATCH", "PARTIAL_MATCH", "FORMAT_MINOR", "SEMANTIC_FATAL", "BLANK"):
            majority_count = sum(
                1 for detail in details
                if str(detail.get("error_category", "")) == majority_category
            )
            category_consistency = majority_count / max(len(details), 1)
            a_i = q_i * category_consistency
        else:
            a_i = 0.5 * q_i

        evidence_quality_sum += points * q_i
        adaptation_sum += points * a_i
        total_points += points
        item_risks.append({
            "id": item_id,
            "points": round(points, 4),
            "evidence_quality": round(q_i, 4),
            "rubric_adaptation": round(a_i, 4),
            "majority_category": majority_category,
            "answer_type": str(item.get("answer_type", classify_rubric_item(item).get("answer_type", "unknown"))),
            "canonicalization": str(item.get("canonicalization", infer_canonicalization(item))),
        })

    if total_points <= 1e-9:
        total_points = max_score
    u_e = clamp01(1.0 - evidence_quality_sum / max(total_points, 1e-9))

    if len(model_scores) >= 2:
        spread = max(model_scores) - min(model_scores)
    else:
        spread = 0.0
    u_s = clamp01(spread / max(gamma * max_score, 1e-9))
    u_r = clamp01(1.0 - adaptation_sum / max(total_points, 1e-9))
    risk = clamp01((u_e + u_s + u_r) / 3.0)
    return {
        "U_E": round(u_e, 6),
        "U_S": round(u_s, 6),
        "U_R": round(u_r, 6),
        "risk": round(risk, 6),
        "mu": round(1.0 - risk, 6),
        "score_spread": round(spread, 6),
        "gamma": round(safe_float(gamma, 0.20), 6),
        "item_risks": item_risks,
    }


def fuse_rubric_mapping_risk(
    consensus_risk,
    post_calibration=None,
    include_undercredit=False,
):
    """Fuse rubric-consensus and evidence-conflict risks with a max t-conorm.

    Both inputs describe membership in the same fuzzy risk set: an unreliable
    mapping from extracted evidence to rubric credit. The maximum operator is
    monotone, idempotent, bounded, and introduces no fitted fusion parameter.
    """
    post_calibration = (
        post_calibration if isinstance(post_calibration, dict) else {}
    )
    consensus = clamp01(consensus_risk)
    unsupported = clamp01(
        post_calibration.get(
            "effective_unsupported_match_points_ratio",
            post_calibration.get("unsupported_match_points_ratio", 0.0),
        )
    )
    contradiction = clamp01(
        post_calibration.get("core_contradiction_ratio", 0.0)
    )
    anchor_failure = (
        1.0 if post_calibration.get("core_anchor_failed", False) else 0.0
    )
    evidence = max(unsupported, contradiction, anchor_failure)
    undercredit = (
        clamp01(post_calibration.get("lenient_undercredit_signal", 0.0))
        if include_undercredit
        else 0.0
    )
    return {
        "U_R_consensus": round(consensus, 6),
        "U_R_evidence": round(evidence, 6),
        "U_R_undercredit": round(undercredit, 6),
        "U_R": round(max(consensus, evidence, undercredit), 6),
        "fusion": "max_t_conorm",
    }


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
    task_profile = infer_rubric_task_profile(rubrics_data, max_score=max_score)
    extraction_counts = compute_extraction_quality_counts(facts_dict, rubrics_data)
    extraction_risk = compute_extraction_risk_features(extraction_counts)
    score_history = [
        safe_float(cot.get("total_score"), None)
        for cot in strict_cots
        if isinstance(cot, dict) and cot.get("total_score") is not None
    ]
    score_history = [score for score in score_history if score is not None]
    primary_risks = compute_three_way_primary_risks(
        facts_dict=facts_dict,
        rubrics_data=rubrics_data,
        strict_cots=strict_cots,
        model_scores=score_history,
        max_score=max_score,
    )

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
    item_judgement_summaries = []
    visual_items = 0
    rule_hits = []
    metadata_items = 0
    explicit_chain_items = 0
    layer_points = Counter()
    layer_supported_points = Counter()
    layer_strong_points = Counter()
    layer_partial_points = Counter()
    layer_unsupported_points = Counter()
    layer_confirmed_unsupported_points = Counter()

    for item in rubrics_data:
        item_id = str(item.get("id", ""))
        points = points_map.get(item_id, fallback_points)
        meta = classify_rubric_item(item)
        answer_type = meta["answer_type"]
        role = meta["role"]
        score_layer = infer_score_layer(item, meta)
        fact_value = facts_dict.get(item_id, "")
        details = details_by_item.get(item_id, [])
        majority_category = _majority_category(details)
        layer_points[score_layer] += points
        if majority_category in ("MATCH", "FORMAT_MINOR", "PARTIAL_MATCH"):
            layer_supported_points[score_layer] += points
        if majority_category in ("MATCH", "FORMAT_MINOR"):
            layer_strong_points[score_layer] += points
        if majority_category == "PARTIAL_MATCH":
            layer_partial_points[score_layer] += points
        if majority_category in ("SEMANTIC_FATAL", "BLANK", "INSUFFICIENT_INFO"):
            layer_unsupported_points[score_layer] += points
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
        item_judgement_summaries.append({
            "points": points,
            "role": role,
            "answer_type": answer_type,
            "score_layer": score_layer,
            "majority_category": majority_category,
        })
        if majority_category in ("PARTIAL_MATCH", "FORMAT_MINOR"):
            partial_or_format_points += points

        if role in ("method", "intermediate", "final") or answer_type in METHOD_TYPES or answer_type == "derived_numeric":
            method_final_points += points

        if answer_type in NUMERIC_TYPES and hard_metadata_enabled:
            expected = infer_expected_number(item)
            target_unit = item.get("unit") or infer_unit(
                item.get("standard_answer_text") or item.get("item", "")
            )
            numeric_match = _numeric_value_matches(fact_value, expected, target_unit=target_unit)
            if numeric_match is True and role in ("method", "intermediate", "final"):
                verified_method_final_points += points
            if majority_category == "MATCH" and numeric_match is False and role in ("method", "intermediate", "final"):
                unsupported_match_points += points
                layer_confirmed_unsupported_points[score_layer] += points

        elif answer_type in METHOD_TYPES and hard_metadata_enabled:
            formula_supported = _formula_is_supported(fact_value, item)
            if formula_supported and role in ("method", "intermediate", "final"):
                verified_method_final_points += points
            if majority_category == "MATCH" and not formula_supported:
                unsupported_match_points += points
                layer_confirmed_unsupported_points[score_layer] += points

    result_any_strong = result_strong_points > 0
    for summary in item_judgement_summaries:
        if summary["majority_category"] != "SEMANTIC_FATAL":
            continue
        process_like = (
            summary["role"] in ("method", "intermediate")
            or summary["answer_type"] in METHOD_TYPES
            or summary["answer_type"] == "derived_numeric"
        )
        if result_any_strong and process_like:
            continue
        fatal_points += summary["points"]

    unsupported_ratio = unsupported_match_points / max_score
    core_points = layer_points["core"]
    support_points = layer_points["support"]
    auxiliary_points = layer_points["auxiliary"]
    core_support_signal = (
        (layer_strong_points["core"] + 0.5 * layer_partial_points["core"])
        / max(core_points, 1e-9)
        if core_points > 0 else 0.0
    )
    support_signal = (
        (layer_strong_points["support"] + 0.5 * layer_partial_points["support"])
        / max(support_points, 1e-9)
        if support_points > 0 else 0.0
    )
    auxiliary_signal = (
        (layer_strong_points["auxiliary"] + 0.5 * layer_partial_points["auxiliary"])
        / max(auxiliary_points, 1e-9)
        if auxiliary_points > 0 else 0.0
    )
    core_unsupported_ratio = layer_unsupported_points["core"] / max(core_points, 1e-9) if core_points > 0 else 0.0
    support_unsupported_ratio = layer_unsupported_points["support"] / max(support_points, 1e-9) if support_points > 0 else 0.0
    auxiliary_unsupported_ratio = (
        layer_unsupported_points["auxiliary"] / max(auxiliary_points, 1e-9)
        if auxiliary_points > 0 else 0.0
    )
    core_contradiction_ratio = (
        layer_confirmed_unsupported_points["core"] / max(core_points, 1e-9)
        if core_points > 0 else 0.0
    )
    support_contradiction_ratio = (
        layer_confirmed_unsupported_points["support"] / max(support_points, 1e-9)
        if support_points > 0 else 0.0
    )
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
        and core_support_signal < 0.65
    )
    confirmed_core_over_score = (
        core_contradiction_ratio >= 0.25
        or (core_unsupported_ratio >= 0.50 and core_support_signal < 0.50)
        or core_anchor_failed
    )
    evidence_supported_answer = (
        core_support_signal >= 0.70
        and result_strong_signal >= 0.65
        and method_evidence_signal >= 0.60
    )
    effective_unsupported_ratio = unsupported_ratio
    if evidence_supported_answer and core_contradiction_ratio < 0.15:
        effective_unsupported_ratio = min(effective_unsupported_ratio, support_contradiction_ratio)
    unsupported_high_score_risk = clamp01(
        avg_ratio
        * max(
            effective_unsupported_ratio,
            bare_answer_risk if avg_ratio >= 0.70 else 0.0,
            fatal_ratio if avg_ratio >= 0.80 else 0.0,
            1.0 if core_anchor_failed else 0.0,
        )
    )
    weak_result_high_score_review = (
        avg_ratio >= 0.65
        and result_strong_signal <= 0.50
        and unsupported_high_score_risk >= 0.10
    )
    stable_undercredit_review = (
        avg_ratio <= 0.60
        and result_correctness_signal >= 0.40
        and method_evidence_signal >= 0.40
        and lenient_undercredit_signal >= 0.04
        and unsupported_high_score_risk < 0.20
    )
    result_anchored_undercredit_review = (
        task_profile["concentrated_result_weight"]
        and avg_ratio <= 0.60
        and result_strong_signal >= 0.50
        and result_correctness_signal >= 0.50
        and fatal_ratio <= 0.35
        and unsupported_high_score_risk < 0.20
        and bare_answer_risk < 0.40
    )
    direct_only_high_score_risk = (
        direct_points_ratio >= 0.30
        and direct_awarded_ratio >= 0.80
        and result_strong_signal <= 0.50
        and avg_ratio >= 0.50
        and method_evidence_signal <= 0.65
    )
    structure_missing_review = (
        extraction_risk["structure_missing_rate"] >= 0.20
        or extraction_risk["suspicious_extraction_rate"] >= 0.35
    )

    visual_blank_review = bool(visual_items) and blank_rate >= 0.40
    if unsupported_ratio >= 0.15:
        rule_hits.append("unsupported_match_review")
    if effective_unsupported_ratio >= 0.15 and not evidence_supported_answer:
        rule_hits.append("unsupported_match_guard")
    if core_anchor_failed:
        rule_hits.append("core_anchor_guard")
    if visual_blank_review:
        rule_hits.append("visual_blank_review")
    if lenient_undercredit_signal >= 0.08:
        rule_hits.append("lenient_undercredit_review")
    if unsupported_high_score_risk >= 0.25:
        rule_hits.append("unsupported_high_score_review")
    if weak_result_high_score_review:
        rule_hits.append("weak_result_high_score_review")
    if stable_undercredit_review:
        rule_hits.append("stable_undercredit_review")
    if result_anchored_undercredit_review:
        rule_hits.append("result_anchored_undercredit_review")
    if direct_only_high_score_risk:
        rule_hits.append("direct_only_high_score_risk")
    if structure_missing_review:
        rule_hits.append("structure_missing_review")

    lower_bound = 0.0
    upper_bound = max_score
    if effective_unsupported_ratio >= 0.15 and not evidence_supported_answer:
        upper_bound = min(upper_bound, avg_model_score)
    if effective_unsupported_ratio >= 0.25 and not evidence_supported_answer:
        upper_bound = min(upper_bound, max(avg_model_score * 0.85, max_score * 0.25))
    if core_anchor_failed:
        parameter_cap = direct_points * 0.30
        verified_cap = verified_method_final_points
        upper_bound = min(upper_bound, max(parameter_cap + verified_cap, max_score * 0.20))

    boundary_domain = (
        unsupported_ratio >= 0.15
        or effective_unsupported_ratio >= 0.15
        or confirmed_core_over_score
        or core_anchor_failed
        or lenient_undercredit_signal >= 0.08
        or unsupported_high_score_risk >= 0.25
        or weak_result_high_score_review
        or stable_undercredit_review
        or result_anchored_undercredit_review
        or direct_only_high_score_risk
        or structure_missing_review
    )
    extraction_retry_review = (
        visual_blank_review
        or extraction_risk["extraction_quality"] == "failed"
        or blank_rate >= 0.40
        or (
            avg_ratio <= 0.30
            and lenient_undercredit_signal < 0.05
            and result_correctness_signal < 0.30
            and unsupported_high_score_risk < 0.20
            and not direct_only_high_score_risk
        )
    )
    if extraction_retry_review:
        boundary_domain = True
        reject_domain = False
        if "extraction_retry_review" not in rule_hits:
            rule_hits.append("extraction_retry_review")
    else:
        reject_domain = False

    mapping_risk = fuse_rubric_mapping_risk(
        primary_risks.get("U_R", 0.0),
        {
            "effective_unsupported_match_points_ratio": effective_unsupported_ratio,
            "core_contradiction_ratio": core_contradiction_ratio,
            "core_anchor_failed": core_anchor_failed,
        },
    )
    primary_risks.update(mapping_risk)
    primary_risk = clamp01(
        (
            safe_float(primary_risks.get("U_E"), 0.0)
            + safe_float(primary_risks.get("U_S"), 0.0)
            + safe_float(primary_risks.get("U_R"), 0.0)
        )
        / 3.0
    )
    primary_risks["risk"] = round(primary_risk, 6)
    primary_risks["mu"] = round(1.0 - primary_risk, 6)

    return {
        "unsupported_match_points": round(unsupported_match_points, 4),
        "unsupported_match_points_ratio": round(unsupported_ratio, 4),
        "effective_unsupported_match_points_ratio": round(effective_unsupported_ratio, 4),
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
        "core_support_signal": round(core_support_signal, 4),
        "support_signal": round(support_signal, 4),
        "auxiliary_signal": round(auxiliary_signal, 4),
        "core_unsupported_ratio": round(core_unsupported_ratio, 4),
        "support_unsupported_ratio": round(support_unsupported_ratio, 4),
        "auxiliary_unsupported_ratio": round(auxiliary_unsupported_ratio, 4),
        "core_contradiction_ratio": round(core_contradiction_ratio, 4),
        "support_contradiction_ratio": round(support_contradiction_ratio, 4),
        "confirmed_core_over_score": confirmed_core_over_score,
        "evidence_supported_answer": evidence_supported_answer,
        "layer_points": {key: round(layer_points[key], 4) for key in ("core", "support", "auxiliary")},
        "layer_supported_points": {
            key: round(layer_supported_points[key], 4) for key in ("core", "support", "auxiliary")
        },
        "layer_confirmed_unsupported_points": {
            key: round(layer_confirmed_unsupported_points[key], 4)
            for key in ("core", "support", "auxiliary")
        },
        "metadata_coverage": round(metadata_coverage, 4),
        "explicit_chain_coverage": round(explicit_chain_coverage, 4),
        "task_type": task_profile["task_type"],
        "complex_derivation_task": task_profile["complex_derivation_task"],
        "upper_consensus_eligible": task_profile["upper_consensus_eligible"],
        "calculation_or_derivation": task_profile["calculation_or_derivation"],
        "algorithmic_or_mapping": task_profile["algorithmic_or_mapping"],
        "final_answer_weight_high": task_profile["final_answer_weight_high"],
        "rubric_task_profile": task_profile,
        "core_anchor_failed": core_anchor_failed,
        "weak_result_high_score_review": weak_result_high_score_review,
        "stable_undercredit_review": stable_undercredit_review,
        "result_anchored_undercredit_review": result_anchored_undercredit_review,
        "direct_only_high_score_risk": direct_only_high_score_risk,
        "extraction_retry_review": extraction_retry_review,
        "structure_missing_review": structure_missing_review,
        "structure_missing_rate": extraction_risk["structure_missing_rate"],
        "suspicious_extraction_rate": extraction_risk["suspicious_extraction_rate"],
        "extraction_risk": extraction_risk["extraction_risk"],
        "extraction_quality": extraction_risk["extraction_quality"],
        "rubric_item_points": {
            key: round(value, 4) for key, value in points_map.items()
        },
        "primary_risks": primary_risks,
        "three_way_primary_risks": primary_risks,
        "visual_blank_review": visual_blank_review,
        "boundary_domain": boundary_domain,
        "reject_domain": reject_domain,
        "fatal_ratio": round(fatal_ratio, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "rule_hits": rule_hits,
    }


def clamp01(value):
    return max(0.0, min(1.0, safe_float(value, 0.0)))


def select_baseline_score(model_scores, model_avg_score, max_score, post_calibration=None, risk_profile=None):
    """Select the score trusted by the 3WD framework before route/BND handling.

    The default is the ordinary three-run average. For complex derivation or
    calculation tasks, a guarded upper-consensus score can be used when the
    answer has enough result/process evidence and low over-credit risk.
    """
    scores = [safe_float(score, None) for score in (model_scores or [])]
    scores = [score for score in scores if score is not None]
    max_score = max(safe_float(max_score, 0.0), 1.0)
    model_avg_score = safe_float(model_avg_score, 0.0)
    post_calibration = post_calibration or {}
    risk_profile = risk_profile or {}
    risk_features = risk_profile.get("risk_features", {}) if isinstance(risk_profile, dict) else {}

    if scores:
        score_min = min(scores)
        score_max = max(scores)
        sorted_scores = sorted(scores)
        score_median = sorted_scores[len(sorted_scores) // 2]
    else:
        score_min = score_max = score_median = model_avg_score

    score_spread = max(0.0, score_max - score_min)
    score_spread_ratio = clamp01(score_spread / max_score)
    selected_score = model_avg_score
    baseline_policy = "model_avg"
    baseline_source = "model_avg_score"

    result_correctness = clamp01(post_calibration.get("result_correctness_signal", 0.0))
    result_strong = clamp01(post_calibration.get("result_strong_signal", result_correctness))
    method_evidence = clamp01(post_calibration.get("method_evidence_signal", 0.0))
    lenient_undercredit = clamp01(post_calibration.get("lenient_undercredit_signal", 0.0))
    unsupported_high_score = clamp01(post_calibration.get("unsupported_high_score_risk", 0.0))
    bare_answer_risk = clamp01(post_calibration.get("bare_answer_risk", 0.0))
    extraction_risk = clamp01(post_calibration.get("extraction_risk", risk_features.get("extraction_risk", 0.0)))
    structure_missing_rate = clamp01(
        post_calibration.get("structure_missing_rate", risk_features.get("structure_missing_rate", 0.0))
    )
    suspicious_extraction_rate = clamp01(
        post_calibration.get("suspicious_extraction_rate", risk_features.get("suspicious_extraction_rate", 0.0))
    )
    weak_result_high_score = bool(post_calibration.get("weak_result_high_score_review", False))
    direct_only_high_score = bool(post_calibration.get("direct_only_high_score_risk", False))
    fatal_ratio = clamp01(
        risk_profile.get(
            "fatal_points_ratio",
            risk_features.get("fatal_points_ratio", 0.0),
        )
    )

    task_profile = post_calibration.get("rubric_task_profile", {})
    if not isinstance(task_profile, dict):
        task_profile = {}
    complex_derivation_task = bool(
        post_calibration.get(
            "complex_derivation_task",
            task_profile.get("complex_derivation_task", False),
        )
    )
    final_answer_weight_high = bool(
        post_calibration.get(
            "final_answer_weight_high",
            task_profile.get("final_answer_weight_high", False),
        )
    )
    upper_consensus_eligible = bool(
        post_calibration.get(
            "upper_consensus_eligible",
            task_profile.get("upper_consensus_eligible", False),
        )
    )
    concentrated_result_weight = bool(task_profile.get("concentrated_result_weight", False))
    model_avg_ratio = clamp01(model_avg_score / max_score)
    high_over_unsafe = (
        unsupported_high_score >= 0.08
        or fatal_ratio >= 0.65
        or bare_answer_risk >= 0.20
        or extraction_risk >= 0.22
        or structure_missing_rate >= 0.18
        or suspicious_extraction_rate >= 0.30
        or score_spread_ratio >= 0.12
        or weak_result_high_score
        or direct_only_high_score
        or result_strong < 0.60
        or method_evidence < 0.55
        or (model_avg_ratio >= 0.65 and (result_strong < 0.75 or method_evidence < 0.65))
    )
    strict_upper_consensus_ready = (
        upper_consensus_eligible
        and not high_over_unsafe
        and final_answer_weight_high
        and result_correctness >= 0.65
        and result_strong >= 0.65
        and method_evidence >= 0.55
        and lenient_undercredit >= 0.08
    )
    low_score_undercredit_upper_candidate = (
        upper_consensus_eligible
        and final_answer_weight_high
        and model_avg_ratio <= 0.60
        and lenient_undercredit >= 0.075
        and result_correctness >= 0.45
        and method_evidence >= 0.45
        and unsupported_high_score < 0.05
        and bare_answer_risk < 0.30
        and extraction_risk < 0.25
        and not weak_result_high_score
        and not direct_only_high_score
    )
    low_score_undercredit_upper_ready = (
        low_score_undercredit_upper_candidate
        and concentrated_result_weight
        and model_avg_ratio <= 0.65
        and score_spread_ratio <= 0.20
    )
    upper_consensus_ready = strict_upper_consensus_ready
    if low_score_undercredit_upper_ready:
        upper_consensus_ready = True

    if upper_consensus_ready and score_max > model_avg_score:
        upper_candidate = score_max
        high_score_weak_evidence = (
            upper_candidate / max_score >= 0.75
            and result_strong < 0.60
            and method_evidence < 0.55
        )
        explicit_over_guard = unsupported_high_score >= 0.10 or bare_answer_risk >= 0.30
        if high_score_weak_evidence or explicit_over_guard:
            if result_strong >= 0.75 and method_evidence >= 0.65:
                raise_cap = max(2.0, 0.18 * max_score)
            elif result_strong >= 0.60 or method_evidence >= 0.60:
                raise_cap = max(1.5, 0.12 * max_score)
            else:
                raise_cap = max(1.0, 0.08 * max_score)
            upper_candidate = min(score_max, model_avg_score + raise_cap)
        selected_score = upper_candidate
        if high_score_weak_evidence:
            selected_score = max(model_avg_score, min(score_median, selected_score))
        baseline_policy = "upper_consensus_strict"
        baseline_source = "guarded_max_of_three" if selected_score == score_max else "capped_max_of_three"
        if low_score_undercredit_upper_ready and not strict_upper_consensus_ready:
            baseline_policy = "upper_consensus_undercredit"

    selected_score = max(0.0, min(max_score, selected_score))
    selected_ratio = clamp01(selected_score / max_score)
    high_score_safety_review = (
        selected_ratio >= 0.75
        and (
            unsupported_high_score >= 0.10
            or result_strong < 0.65
            or score_spread_ratio >= 0.10
            or bool(post_calibration.get("direct_only_high_score_risk", False))
        )
    )

    return {
        "selected_baseline_score": round(selected_score, 4),
        "baseline_policy": baseline_policy,
        "baseline_score_source": baseline_source,
        "baseline_selection_signals": {
            "complex_derivation_task": complex_derivation_task,
            "upper_consensus_eligible": upper_consensus_eligible,
            "task_type": post_calibration.get("task_type", task_profile.get("task_type", "mixed_or_unknown")),
            "final_answer_weight_high": final_answer_weight_high,
            "concentrated_result_weight": concentrated_result_weight,
            "upper_consensus_ready": upper_consensus_ready,
            "low_score_undercredit_upper_candidate": low_score_undercredit_upper_candidate,
            "high_over_unsafe": high_over_unsafe,
            "high_score_safety_review": high_score_safety_review,
            "extraction_risk": round(extraction_risk, 6),
            "structure_missing_rate": round(structure_missing_rate, 6),
            "suspicious_extraction_rate": round(suspicious_extraction_rate, 6),
            "score_history_min": round(score_min, 4),
            "score_history_max": round(score_max, 4),
            "score_history_median": round(score_median, 4),
            "raise_cap": round(max(0.0, selected_score - model_avg_score), 4),
            "score_spread_ratio": round(score_spread_ratio, 6),
            "selected_ratio": round(selected_ratio, 6),
        },
    }


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
    structure_missing_rate=0.0,
    extraction_risk=None,
    high_blank_high_score=False,
    post_calibration=None,
    weights=None,
    loss_params=None,
    membership_model=None,
    score_uncertainty=None,
    directional_undercredit_risk=False,
):
    """Build an evidence-calibrated A3WA route decision.

    U_E, U_S and U_R describe evidence, score-stability and rubric-mapping
    uncertainty. A validation-fitted monotonic model may map them to membership
    in the safe-auto-grading fuzzy set. Without that model, a weighted linear
    mapping is used for backward compatibility.
    """
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
    score_interval = conformal_score_interval(
        center_score=avg_model_score,
        max_score=max_score,
        score_spread_norm=score_spread_norm,
        config=score_uncertainty,
    )

    if extraction_risk is None:
        u_extract = clamp01(
            0.40 * safe_float(blank_rate, 0.0)
            + 0.25 * safe_float(low_quality_rate, 0.0)
            + 0.20 * safe_float(perception_failure_rate, 0.0)
            + 0.15 * safe_float(structure_missing_rate, 0.0)
        )
    else:
        u_extract = clamp01(extraction_risk)
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
    lenient_undercredit = clamp01(post_calibration.get("lenient_undercredit_signal", 0.0))
    unsupported_high_score = clamp01(post_calibration.get("unsupported_high_score_risk", 0.0))
    result_correctness = clamp01(post_calibration.get("result_correctness_signal", 0.0))
    result_strong = clamp01(post_calibration.get("result_strong_signal", 0.0))
    method_evidence = clamp01(post_calibration.get("method_evidence_signal", 0.0))
    bare_answer_risk = clamp01(post_calibration.get("bare_answer_risk", 0.0))
    weak_result_high_score = bool(post_calibration.get("weak_result_high_score_review", False))
    stable_undercredit = bool(post_calibration.get("stable_undercredit_review", False))
    direct_only_high_score = bool(post_calibration.get("direct_only_high_score_risk", False))
    high_score_safety = bool(post_calibration.get("high_score_safety_review", False))
    structure_missing_review = bool(post_calibration.get("structure_missing_review", False))
    extraction_retry_review = bool(post_calibration.get("extraction_retry_review", False))
    low_score_nonblank_review = avg_ratio <= 0.30 and u_extract < 0.40 and u_blank <= 0.35
    core_anchor_score = 1.0 if post_calibration.get("core_anchor_failed", False) else 0.0
    high_blank_score = u_extract if avg_ratio >= 0.70 and u_blank >= 0.45 else 0.0
    high_score_weak_evidence = 0.0
    if avg_ratio >= 0.65 and (result_strong < 0.60 or method_evidence < 0.50):
        high_score_weak_evidence = 0.45
    u_overcredit = clamp01(
        avg_ratio * max(
            u_semantic,
            unsupported_ratio,
            unsupported_high_score,
            bare_answer_risk,
            core_anchor_score,
            high_blank_score,
            0.80 if direct_only_high_score else 0.0,
            0.70 if weak_result_high_score else 0.0,
            0.60 if high_score_safety else 0.0,
            high_score_weak_evidence,
        )
    )

    primary_risks = post_calibration.get("primary_risks") or post_calibration.get("three_way_primary_risks")
    if isinstance(primary_risks, dict):
        u_e = clamp01(primary_risks.get("U_E", u_extract))
        u_s = clamp01(primary_risks.get("U_S", u_score))
        mapping_risk = fuse_rubric_mapping_risk(
            primary_risks.get(
                "U_R_consensus", primary_risks.get("U_R", 0.0)
            ),
            post_calibration,
            include_undercredit=directional_undercredit_risk,
        )
        u_r_consensus = mapping_risk["U_R_consensus"]
        u_r_evidence = mapping_risk["U_R_evidence"]
        u_r_undercredit = mapping_risk["U_R_undercredit"]
        u_r = mapping_risk["U_R"]
    else:
        u_e = u_extract
        u_s = u_score
        u_r_undercredit = (
            lenient_undercredit if directional_undercredit_risk else 0.0
        )
        u_r = max(u_semantic, u_overcredit, u_r_undercredit)
        u_r_consensus = u_r
        u_r_evidence = 0.0
    membership = calibrated_a3wa_membership(
        u_e=u_e,
        u_s=u_s,
        u_r=u_r,
        weights=weights,
        model=membership_model,
    )
    risk = membership["risk"]
    confidence = membership["mu"]

    alpha, beta = compute_a3wa_thresholds(
        lambda1=params.get("lambda1", 5.0),
        lambda2=params.get("lambda2", 1.0),
        mu1=params.get("mu1", 3.0),
        mu2=params.get("mu2", 7.0),
        m=params.get("m", 0.5),
    )

    hard_neg_reasons = []
    no_usable_evidence = (
        extraction_quality == "failed"
        and u_e >= 0.95
        and result_strong < 0.30
        and method_evidence < 0.30
    )
    if no_usable_evidence:
        hard_neg_reasons.append("no_usable_evidence")

    if hard_neg_reasons:
        route = "NEG"
        reason = "hard_neg:" + ",".join(hard_neg_reasons)
    elif confidence >= alpha:
        route = "POS"
        reason = "membership_ge_alpha"
    elif confidence <= beta:
        route = "NEG"
        reason = "membership_le_beta"
    else:
        route = "BND"
        reason = "beta_lt_membership_lt_alpha"

    review_signals = []
    if extraction_retry_review:
        review_signals.append("extraction_retry_review")
    if low_score_nonblank_review:
        review_signals.append("low_score_nonblank_review")
    if structure_missing_review:
        review_signals.append("structure_missing_review")
    if unsupported_high_score >= 0.25:
        review_signals.append("unsupported_high_score_review")
    if lenient_undercredit >= 0.08:
        review_signals.append("possible_undercredit")
    if (
        score_interval["enabled"]
        and score_interval["stability_risk"] is not None
        and score_interval["stability_risk"] >= 1.0
    ):
        review_signals.append("conformal_interval_exceeds_safe_tolerance")

    return {
        "route": route,
        "reason": reason,
        "hard_neg_reasons": hard_neg_reasons,
        "review_signals": review_signals,
        "risk": round(risk, 6),
        "confidence": round(confidence, 6),
        "mu": round(confidence, 6),
        "membership_source": membership["source"],
        "score_interval": score_interval,
        "alpha": round(alpha, 6),
        "beta": round(beta, 6),
        "m": round(safe_float(params.get("m", 0.5), 0.5), 6),
        "lambda1": safe_float(params.get("lambda1", 5.0), 5.0),
        "lambda2": safe_float(params.get("lambda2", 1.0), 1.0),
        "mu1": safe_float(params.get("mu1", 3.0), 3.0),
        "mu2": safe_float(params.get("mu2", 7.0), 7.0),
        "risk_components": {
            "U_E": round(u_e, 6),
            "U_S": round(u_s, 6),
            "U_R": round(u_r, 6),
            "U_R_consensus": round(u_r_consensus, 6),
            "U_R_evidence": round(u_r_evidence, 6),
            "U_R_undercredit": round(u_r_undercredit, 6),
            "U_extract": round(u_extract, 6),
            "U_score": round(u_score, 6),
            "U_semantic": round(u_semantic, 6),
            "U_blank": round(u_blank, 6),
            "U_overcredit": round(u_overcredit, 6),
            "U_structure_missing": round(safe_float(structure_missing_rate, 0.0), 6),
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
    structure_missing_rate = clamp01(risk_value("structure_missing_rate", 0.0))
    extraction_risk_profile = risk_value("extraction_risk", None)
    uncertainty_index = clamp01(risk_value("uncertainty_index", risk_value("std_ratio", 0.0)))
    spread_ratio = clamp01(risk_value("spread_ratio", 0.0))
    if not spread_ratio and a3wa_decision:
        spread_ratio = clamp01(safe_float(a3wa_decision.get("score_spread", 0.0), 0.0) / max_score)
    partial_ratio = clamp01(risk_value("partial_match_points_ratio", 0.0))
    format_ratio = clamp01(risk_value("format_minor_points_ratio", 0.0))
    raw_unsupported_ratio = clamp01(post_calibration.get("unsupported_match_points_ratio", 0.0))
    unsupported_ratio = clamp01(
        post_calibration.get("effective_unsupported_match_points_ratio", raw_unsupported_ratio)
    )
    core_support_signal = clamp01(post_calibration.get("core_support_signal", 0.0))
    core_contradiction_ratio = clamp01(post_calibration.get("core_contradiction_ratio", 0.0))
    lenient_undercredit = clamp01(post_calibration.get("lenient_undercredit_signal", risk_value("lenient_undercredit_signal", 0.0)))
    unsupported_high_score = clamp01(post_calibration.get("unsupported_high_score_risk", risk_value("unsupported_high_score_risk", 0.0)))
    result_correctness = clamp01(post_calibration.get("result_correctness_signal", risk_value("result_correctness_signal", 0.0)))
    result_strong = clamp01(post_calibration.get("result_strong_signal", result_correctness))
    method_evidence = clamp01(post_calibration.get("method_evidence_signal", risk_value("method_evidence_signal", 0.0)))
    bare_answer_risk = clamp01(post_calibration.get("bare_answer_risk", risk_value("bare_answer_risk", 0.0)))
    post_upper = safe_float(post_calibration.get("upper_bound", max_score), max_score)
    if extraction_risk_profile is None:
        extraction_risk = clamp01(
            0.40 * blank_rate
            + 0.25 * low_quality_rate
            + 0.20 * perception_failure_rate
            + 0.15 * structure_missing_rate
        )
    else:
        extraction_risk = clamp01(extraction_risk_profile)
    weak_result_high_score = bool(post_calibration.get("weak_result_high_score_review", False))
    stable_undercredit = bool(post_calibration.get("stable_undercredit_review", False))
    direct_only_high_score = bool(post_calibration.get("direct_only_high_score_risk", False))
    high_score_safety = bool(post_calibration.get("high_score_safety_review", False))
    partial_or_format_evidence = max(partial_ratio, format_ratio)
    high_score_weak_evidence = 1.0 if avg_ratio >= 0.65 and (result_strong < 0.60 or method_evidence < 0.50) else 0.0
    raise_evidence_score = clamp01(
        0.42 * result_correctness
        + 0.32 * method_evidence
        + 0.18 * lenient_undercredit
        + 0.08 * partial_or_format_evidence
        - 0.30 * unsupported_high_score
        - 0.20 * bare_answer_risk
        - 0.15 * extraction_risk
        - (0.18 if direct_only_high_score else 0.0)
    )
    if avg_ratio >= 0.65:
        raise_evidence_score = clamp01(raise_evidence_score - 0.35 * (avg_ratio - 0.65))
    lower_evidence_score = clamp01(
        0.38 * unsupported_high_score
        + 0.24 * bare_answer_risk
        + 0.18 * high_score_weak_evidence
        + 0.10 * fatal_ratio
        + 0.10 * max(extraction_risk, structure_missing_rate)
        + (0.20 if direct_only_high_score else 0.0)
        + (0.16 if weak_result_high_score else 0.0)
        + (0.12 if high_score_safety else 0.0)
    )

    over_reasons = []
    if weak_result_high_score:
        over_reasons.append("weak_result_high_score")
    if direct_only_high_score:
        over_reasons.append("direct_only_high_score")
    if high_score_safety:
        over_reasons.append("high_score_safety_review")
    if unsupported_high_score >= 0.10 and avg_ratio >= 0.70:
        over_reasons.append("unsupported_high_score")
    if bool(risk_value("high_blank_high_score", False)) and avg_ratio >= 0.80 and unsupported_high_score >= 0.20:
        over_reasons.append("high_blank_high_score")
    if fatal_ratio >= 0.70 and avg_ratio >= 0.75 and unsupported_high_score >= 0.20:
        over_reasons.append("fatal_points_high")
    if unsupported_ratio >= 0.15 and (core_support_signal < 0.70 or core_contradiction_ratio >= 0.15):
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
    if unsupported_ratio >= 0.25 and (core_support_signal < 0.70 or core_contradiction_ratio >= 0.15):
        strong_over_reasons.append("unsupported_match_high")
    if bool(post_calibration.get("core_anchor_failed", False)):
        strong_over_reasons.append("core_anchor_failed")
    if post_upper < avg_model_score - max(0.10 * max_score, 0.75):
        strong_over_reasons.append("strict_post_upper_cap")

    under_reasons = []
    if stable_undercredit:
        under_reasons.append("stable_undercredit")
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

    strong_over_score_risk = bool(strong_over_reasons) or lower_evidence_score >= 0.65
    over_score_risk = bool(over_reasons) or lower_evidence_score >= 0.45
    under_score_risk = (
        (bool(under_reasons) or raise_evidence_score >= 0.55)
        and raise_evidence_score >= 0.50
        and lower_evidence_score < 0.45
        and not strong_over_score_risk
    )

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
        "structure_missing_rate": round(structure_missing_rate, 6),
        "uncertainty_index": round(uncertainty_index, 6),
        "spread_ratio": round(spread_ratio, 6),
        "partial_match_points_ratio": round(partial_ratio, 6),
        "format_minor_points_ratio": round(format_ratio, 6),
        "unsupported_match_points_ratio": round(raw_unsupported_ratio, 6),
        "effective_unsupported_match_points_ratio": round(unsupported_ratio, 6),
        "core_support_signal": round(core_support_signal, 6),
        "core_contradiction_ratio": round(core_contradiction_ratio, 6),
        "lenient_undercredit_signal": round(lenient_undercredit, 6),
        "unsupported_high_score_risk": round(unsupported_high_score, 6),
        "result_correctness_signal": round(result_correctness, 6),
        "result_strong_signal": round(result_strong, 6),
        "method_evidence_signal": round(method_evidence, 6),
        "bare_answer_risk": round(bare_answer_risk, 6),
        "raise_evidence_score": round(raise_evidence_score, 6),
        "lower_evidence_score": round(lower_evidence_score, 6),
        "weak_result_high_score_review": weak_result_high_score,
        "stable_undercredit_review": stable_undercredit,
        "direct_only_high_score_risk": direct_only_high_score,
        "high_score_safety_review": high_score_safety,
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


BOUNDARY_RAISE_REASON_TYPES = {
    "lenient_process_credit",
    "propagated_error",
    "format_minor",
    "valid_alternative",
    "calculation_trace",
    "process_credit",
    "near_correct_final",
}
BOUNDARY_LOWER_REASON_TYPES = {
    "direct_only",
    "unsupported_final",
    "wrong_core_result",
    "unsupported_match",
    "bare_answer",
    "contradiction",
    "severe_extraction_absence",
}


def summarize_boundary_agent_evidence(agent_evidence, max_score, item_points=None):
    """Summarize structured BND missed/over credit items without trusting free-form totals."""
    max_score = max(safe_float(max_score, 0.0), 1.0)
    item_points = item_points if isinstance(item_points, dict) else {}
    if not isinstance(agent_evidence, dict):
        return {
            "has_agent_evidence": False,
            "missed_points": 0.0,
            "over_points": 0.0,
            "allowed_missed_points": 0.0,
            "allowed_over_points": 0.0,
            "missed_count": 0,
            "over_count": 0,
            "missed_reason_types": [],
            "over_reason_types": [],
            "confidence": 0.0,
        }

    if "allowed_missed_points" in agent_evidence and "allowed_over_points" in agent_evidence:
        return {
            "has_agent_evidence": bool(agent_evidence.get("has_agent_evidence", True)),
            "missed_points": round(max(0.0, safe_float(agent_evidence.get("missed_points", 0.0), 0.0)), 4),
            "over_points": round(max(0.0, safe_float(agent_evidence.get("over_points", 0.0), 0.0)), 4),
            "allowed_missed_points": round(max(0.0, safe_float(agent_evidence.get("allowed_missed_points", 0.0), 0.0)), 4),
            "allowed_over_points": round(max(0.0, safe_float(agent_evidence.get("allowed_over_points", 0.0), 0.0)), 4),
            "missed_count": int(safe_float(agent_evidence.get("missed_count", 0), 0)),
            "over_count": int(safe_float(agent_evidence.get("over_count", 0), 0)),
            "missed_reason_types": list(agent_evidence.get("missed_reason_types", [])),
            "over_reason_types": list(agent_evidence.get("over_reason_types", [])),
            "confidence": round(clamp01(agent_evidence.get("confidence", 0.0)), 4),
        }

    def collect(items, allowed_types, lower=False):
        total = 0.0
        allowed_total = 0.0
        count = 0
        reason_types = []
        credited_by_item = {}
        if not isinstance(items, list):
            return total, allowed_total, count, reason_types
        for item in items:
            if not isinstance(item, dict):
                continue
            points = max(0.0, min(max_score, safe_float(item.get("points", 0.0), 0.0)))
            if points <= 0:
                continue
            evidence = str(item.get("evidence", "")).strip()
            reason_type = str(item.get("reason_type", "")).strip().lower()
            score_layer = str(item.get("score_layer", "")).strip().lower()
            evidence_status = str(item.get("evidence_status", "")).strip().lower()
            item_id = str(item.get("id", "")).strip()
            count += 1
            total += points
            if reason_type:
                reason_types.append(reason_type)
            if not evidence or reason_type not in allowed_types:
                continue
            allowed_statuses = {"explicit", "derived_from_canonical_context"}
            if lower:
                allowed_statuses = {"explicit", "contradiction"}
                if reason_type in {"bare_answer", "severe_extraction_absence"}:
                    allowed_statuses.add("absent")
            if evidence_status not in allowed_statuses:
                continue
            if score_layer == "auxiliary" and lower and reason_type not in {"contradiction", "severe_extraction_absence"}:
                continue
            if score_layer == "auxiliary":
                points *= 0.5
            if score_layer == "support" and lower:
                points *= 0.75
            if item_id and item_id in item_points:
                points = min(points, max(0.0, safe_float(item_points[item_id], 0.0)))
            if item_id:
                remaining = max(0.0, safe_float(item_points.get(item_id, max_score), max_score) - credited_by_item.get(item_id, 0.0))
                points = min(points, remaining)
                credited_by_item[item_id] = credited_by_item.get(item_id, 0.0) + points
            if points > 0:
                allowed_total += points
        return total, allowed_total, count, sorted(set(reason_types))

    missed_points, allowed_missed, missed_count, missed_types = collect(
        agent_evidence.get("missed_credit_items"),
        BOUNDARY_RAISE_REASON_TYPES,
    )
    over_points, allowed_over, over_count, over_types = collect(
        agent_evidence.get("over_credit_items"),
        BOUNDARY_LOWER_REASON_TYPES,
        lower=True,
    )
    return {
        "has_agent_evidence": True,
        "missed_points": round(missed_points, 4),
        "over_points": round(over_points, 4),
        "allowed_missed_points": round(allowed_missed, 4),
        "allowed_over_points": round(allowed_over, 4),
        "missed_count": missed_count,
        "over_count": over_count,
        "missed_reason_types": missed_types,
        "over_reason_types": over_types,
        "confidence": round(clamp01(agent_evidence.get("confidence", 0.0)), 4),
    }


def apply_boundary_action_policy(
    avg_model_score,
    candidate_score,
    max_score,
    a3wa_decision=None,
    risk_profile=None,
    post_calibration=None,
    agent_evidence=None,
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
    agent_summary = summarize_boundary_agent_evidence(agent_evidence, max_score)

    avg_ratio = clamp01(baseline / max_score)
    fatal = clamp01(components.get("U_semantic", signals.get("fatal_points_ratio", 0.0)))
    overcredit = clamp01(components.get("U_overcredit", 0.0))
    blank = clamp01(components.get("U_blank", signals.get("blank_rate", 0.0)))
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
    direct_only_high_score = bool(post_calibration.get("direct_only_high_score_risk", False))
    high_score_safety = bool(post_calibration.get("high_score_safety_review", False))
    structure_missing_review = bool(post_calibration.get("structure_missing_review", False))
    final_answer_weight_high = bool(post_calibration.get("final_answer_weight_high", False))
    task_profile = post_calibration.get("rubric_task_profile", {})
    if not isinstance(task_profile, dict):
        task_profile = {}
    fragmented_rubric = bool(task_profile.get("fragmented_rubric", False))
    concentrated_result_weight = bool(task_profile.get("concentrated_result_weight", False))
    score_history_max = safe_float(post_calibration.get("score_history_max", baseline), baseline)
    score_history_median = safe_float(post_calibration.get("score_history_median", baseline), baseline)
    partial_or_format_evidence = max(
        clamp01(signals.get("partial_match_points_ratio", 0.0)),
        clamp01(signals.get("format_minor_points_ratio", 0.0)),
    )
    raise_evidence_score = clamp01(signals.get("raise_evidence_score", 0.0))
    lower_evidence_score = clamp01(signals.get("lower_evidence_score", 0.0))
    raise_has_agent_evidence = agent_summary["allowed_missed_points"] > 0
    lower_has_agent_evidence = agent_summary["allowed_over_points"] > 0
    agent_raise_ratio = clamp01(agent_summary["allowed_missed_points"] / max_score)
    agent_over_ratio = clamp01(agent_summary["allowed_over_points"] / max_score)
    core_support_signal = clamp01(post_calibration.get("core_support_signal", result_strong))
    core_unsupported_ratio = clamp01(post_calibration.get("core_unsupported_ratio", 0.0))
    core_contradiction_ratio = clamp01(
        post_calibration.get("core_contradiction_ratio", core_unsupported_ratio)
    )
    confirmed_core_over_score = bool(post_calibration.get("confirmed_core_over_score", False)) or (
        core_contradiction_ratio >= 0.25
        and core_support_signal < 0.65
    )
    lower_direction_permission = (
        confirmed_core_over_score
        or (
            lower_has_agent_evidence
            and (
                agent_over_ratio >= 0.03
                or core_contradiction_ratio >= 0.15
                or core_support_signal < 0.55
            )
        )
        or (
            direct_only_high_score
            and core_support_signal < 0.55
            and core_contradiction_ratio >= 0.10
        )
    )
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
    short_calc_stable_undercredit = (
        max_score <= 10.0
        and concentrated_result_weight
        and bool(post_calibration.get("stable_undercredit_review", False))
        and avg_ratio <= 0.65
        and result_correctness >= 0.50
        and result_strong >= 0.50
        and method_evidence >= 0.30
        and unsupported_high_score < 0.12
        and bare_answer_risk < 0.40
    )
    result_anchored_raise = (
        bool(post_calibration.get("result_anchored_undercredit_review", False))
        and concentrated_result_weight
        and avg_ratio <= 0.60
        and result_strong >= 0.50
        and result_correctness >= 0.50
        and fatal <= 0.35
        and unsupported_high_score < 0.20
        and bare_answer_risk < 0.40
        and not direct_only_high_score
    )
    if concentrated_result_weight and avg_ratio <= 0.55:
        raise_evidence_floor = 0.34
    elif concentrated_result_weight:
        raise_evidence_floor = 0.40
    else:
        raise_evidence_floor = 0.52
    lenient_raise_ready = (
        lenient_undercredit >= 0.08
        and result_correctness >= 0.50
        and method_evidence >= 0.35
        and result_strong >= 0.35
        and raise_evidence_score >= raise_evidence_floor
        and lower_evidence_score < 0.45
        and bare_answer_risk < 0.35
        and unsupported_high_score < 0.25
        and avg_ratio <= 0.90
        and (
            lenient_undercredit >= 0.10
            or result_correctness >= 0.60
            or avg_ratio <= 0.55
            or partial_or_format_evidence >= 0.05
        )
        and not (structure_missing_review and avg_ratio >= 0.55 and result_correctness < 0.60)
        and not parameter_dense_weak_final
        and not direct_only_high_score
        and not (fragmented_rubric and not raise_has_agent_evidence)
    )
    strong_lenient_raise = (
        lenient_raise_ready
        and lenient_undercredit >= 0.16
        and raise_evidence_score >= 0.62
        and result_strong >= 0.45
        and avg_ratio <= 0.70
    )
    # Generic high-band guard: once the baseline is already in the middle/high
    # score band, raising without explicit agent evidence is more likely to
    # create over-credit. This is evidence-based, not question-specific.
    high_band_non_agent_raise_guard = (
        avg_ratio >= 0.55
        and not raise_has_agent_evidence
        and (
            result_strong < 0.70
            or method_evidence < 0.70
            or bare_answer_risk >= 0.18
            or unsupported_high_score >= 0.08
        )
    )
    very_high_band_raise_guard = (
        avg_ratio >= 0.70
        and not raise_has_agent_evidence
        and not strong_lenient_raise
    )
    non_agent_raise_allowed = (
        (avg_ratio <= 0.55 and not fragmented_rubric)
        or (
            concentrated_result_weight
            and avg_ratio <= 0.72
            and lenient_raise_ready
            and result_strong >= 0.55
            and method_evidence >= 0.45
            and lenient_undercredit >= 0.08
            and lower_evidence_score < 0.45
        )
    )
    if short_calc_stable_undercredit or result_anchored_raise:
        high_band_non_agent_raise_guard = False
        very_high_band_raise_guard = False
        non_agent_raise_allowed = True
    elif high_band_non_agent_raise_guard or very_high_band_raise_guard:
        non_agent_raise_allowed = False
    raise_direction_permission = raise_has_agent_evidence or non_agent_raise_allowed
    agent_supported_raise = (
        raise_has_agent_evidence
        and agent_raise_ratio >= 0.03
        and lower_evidence_score < 0.45
        and unsupported_high_score < 0.25
        and bare_answer_risk < 0.45
        and not direct_only_high_score
    )
    under_direction_ready = (
        lenient_raise_ready
        or result_anchored_raise
        or agent_supported_raise
        or (
            signals.get("under_score_risk", False)
            and raise_evidence_score >= 0.55
            and lower_evidence_score < 0.45
            and result_correctness >= 0.50
            and method_evidence >= 0.35
            and unsupported_high_score < 0.25
            and bare_answer_risk < 0.45
            and not (structure_missing_review and avg_ratio >= 0.55 and result_correctness < 0.60)
            and not direct_only_high_score
        )
    )
    over_direction_ready = (
        lower_evidence_score >= 0.45
        or (lower_has_agent_evidence and agent_over_ratio >= 0.03)
        or unsupported_high_score >= 0.10
        or bare_answer_risk >= 0.30
        or direct_only_high_score
        or high_score_safety
        or signals.get("weak_result_high_score_review", False)
    )
    strong_over_direction = (
        lower_evidence_score >= 0.65
        or (lower_has_agent_evidence and agent_over_ratio >= 0.10)
        or unsupported_high_score >= 0.25
        or direct_only_high_score
        or signals.get("weak_result_high_score_review", False)
        or (bare_answer_risk >= 0.35 and avg_ratio >= 0.70)
        or (high_score_safety and result_strong < 0.60 and method_evidence < 0.55)
    )
    short_answer_no_evidence_lower_guard = (
        max_score <= 10.0
        and final_answer_weight_high
        and not lower_has_agent_evidence
        and not confirmed_core_over_score
        and unsupported_high_score < 0.25
        and bare_answer_risk < 0.35
    )
    short_answer_positive_evidence_lower_guard = (
        max_score <= 10.0
        and final_answer_weight_high
        and result_strong >= 0.70
        and method_evidence >= 0.75
        and result_correctness >= 0.70
        and core_support_signal >= 0.70
        and bare_answer_risk < 0.20
        and avg_ratio <= 0.85
        and not direct_only_high_score
        and not lower_has_agent_evidence
        and not confirmed_core_over_score
    )

    if abs(delta) <= minor_margin:
        strong_positive_evidence = (
            result_strong >= 0.65
            and method_evidence >= 0.70
            and core_support_signal >= 0.65
            and lower_evidence_score < 0.45
            and unsupported_high_score < 0.20
            and bare_answer_risk < 0.25
            and not direct_only_high_score
            and not lower_has_agent_evidence
            and not confirmed_core_over_score
            and not signals.get("weak_result_high_score_review", False)
        )
        if (
            over_direction_ready
            and lower_direction_permission
            and avg_ratio >= 0.55
            and not strong_positive_evidence
            and not short_answer_no_evidence_lower_guard
            and not short_answer_positive_evidence_lower_guard
            and (strong_over_direction or (lower_has_agent_evidence and not under_direction_ready))
        ):
            lower_margin = small_margin
            if strong_over_direction:
                lower_margin = large_margin
            if lower_has_agent_evidence:
                lower_margin = min(lower_margin, agent_summary["allowed_over_points"])
            history_target = score_history_median if score_history_median < baseline else baseline - lower_margin
            final_score = max(0.0, min(baseline - min(lower_margin, max(0.05 * max_score, 0.5)), history_target))
            accepted = final_score < baseline
            action = "auto_small_lower" if accepted else "keep_minor_change"
            gate_reason = "directional_overcredit_signal" if accepted else "minor_candidate_delta"
        elif under_direction_ready and raise_direction_permission:
            high_band_gap = max(0.0, 0.90 * max_score - baseline)
            history_gap = max(0.0, score_history_max - baseline)
            fallback_gap = max(0.0, min(small_margin, 0.90 * max_score - baseline))
            allowed_gap = fallback_gap
            if raise_has_agent_evidence:
                allowed_gap = max(fallback_gap, min(history_gap, small_margin))
            if result_anchored_raise:
                allowed_gap = max(allowed_gap, min(history_gap, large_margin))
            margin = large_margin if strong_lenient_raise else small_margin
            if result_anchored_raise:
                margin = max(margin, small_margin)
            final_score = min(max_score, baseline + min(margin, high_band_gap, allowed_gap))
            accepted = final_score > baseline
            action = "auto_medium_raise" if strong_lenient_raise and accepted else ("auto_small_raise" if accepted else "keep_minor_change")
            gate_reason = "directional_undercredit_signal" if accepted else "minor_candidate_delta"
        else:
            action = "keep_minor_change"
            gate_reason = "minor_candidate_delta"
    elif delta < 0:
        has_over_item = lower_direction_permission
        strong_positive_evidence = (
            result_strong >= 0.65
            and method_evidence >= 0.70
            and core_support_signal >= 0.65
            and lower_evidence_score < 0.45
            and unsupported_high_score < 0.20
            and bare_answer_risk < 0.25
            and not direct_only_high_score
            and not lower_has_agent_evidence
            and not confirmed_core_over_score
            and not signals.get("weak_result_high_score_review", False)
        )
        strong_lower = (
            has_over_item
            and not strong_positive_evidence
            and not short_answer_no_evidence_lower_guard
            and not short_answer_positive_evidence_lower_guard
            and avg_ratio >= 0.70
            and strong_over_direction
        )
        supported_lower = (
            has_over_item
            and not strong_positive_evidence
            and not short_answer_no_evidence_lower_guard
            and not short_answer_positive_evidence_lower_guard
            and avg_ratio >= 0.55
            and (strong_over_direction or not under_direction_ready)
            and (
                unsupported_high_score >= 0.10
                or agent_over_ratio >= 0.03
                or direct_only_high_score
                or signals.get("weak_result_high_score_review", False)
                or high_score_safety
                or bare_answer_risk >= 0.30
            )
            and (
                fatal >= 0.20
                or blank >= 0.60
                or overcredit >= 0.10
                or bare_answer_risk >= 0.15
                or result_strong < 0.70
                or method_evidence < 0.55
            )
        )
        if strong_lower:
            final_score = max(0.0, baseline - min(abs(delta), large_margin))
            if score_history_median < baseline:
                final_score = min(final_score, score_history_median)
            accepted = True
            action = "large_lower"
            gate_reason = "strong_unsupported_high_score_evidence"
        elif supported_lower:
            final_score = max(0.0, baseline - min(abs(delta), small_margin))
            if score_history_median < baseline:
                final_score = min(final_score, score_history_median)
            accepted = True
            action = "small_lower"
            gate_reason = "supported_unsupported_high_score_evidence"
        else:
            action = "reject_lower"
            gate_reason = "lower_without_sufficient_unsupported_high_score_evidence"
    elif delta > 0:
        supported_raise = (
            (lenient_raise_ready or result_anchored_raise)
            and raise_direction_permission
            and (raise_has_agent_evidence or under_direction_ready)
        )
        if supported_raise:
            high_band_gap = max(0.0, 0.90 * max_score - baseline)
            history_gap = max(0.0, score_history_max - baseline)
            fallback_gap = max(0.0, min(small_margin, 0.90 * max_score - baseline))
            allowed_gap = fallback_gap
            if raise_has_agent_evidence:
                allowed_gap = max(fallback_gap, min(history_gap, small_margin))
            if result_anchored_raise:
                allowed_gap = max(allowed_gap, min(history_gap, large_margin))
            margin = large_margin if strong_lenient_raise else small_margin
            if result_anchored_raise:
                margin = max(margin, small_margin)
            final_score = min(max_score, baseline + min(delta, margin, high_band_gap, allowed_gap))
            accepted = final_score > baseline
            action = "medium_raise" if strong_lenient_raise else "small_raise"
            if not accepted:
                action = "keep_minor_change"
                gate_reason = "no_upper_consensus_margin"
            else:
                gate_reason = "supported_lenient_undercredit_evidence"
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
        "agent_evidence_summary": agent_summary,
    }


def apply_structured_boundary_action_policy(
    avg_model_score,
    candidate_score,
    max_score,
    post_calibration=None,
    agent_evidence=None,
    config=None,
):
    """Apply a BND action only when structured item evidence supports it.

    This is the second action in sequential 3WD. The arbitrator proposes a
    score, while validated item-level evidence determines whether and how much
    of that proposal can be accepted.
    """
    baseline = max(0.0, min(safe_float(max_score, 1.0), safe_float(avg_model_score, 0.0)))
    max_score = max(safe_float(max_score, 0.0), 1.0)
    candidate = max(0.0, min(max_score, safe_float(candidate_score, baseline)))
    post_calibration = post_calibration if isinstance(post_calibration, dict) else {}
    config = config if isinstance(config, dict) else {}
    item_points = post_calibration.get("rubric_item_points", {})
    summary = summarize_boundary_agent_evidence(
        agent_evidence,
        max_score,
        item_points=item_points,
    )
    confidence = summary["confidence"]
    min_confidence = clamp01(config.get("min_evidence_confidence", 0.60))
    auto_keep_confidence = clamp01(config.get("auto_keep_confidence", 0.80))
    max_adjustment_ratio = max(
        0.0,
        safe_float(config.get("max_adjustment_ratio", 0.20), 0.20),
    )
    max_adjustment = max_adjustment_ratio * max_score
    minimum_change = max(0.02 * max_score, 0.10)
    allow_raise = bool(config.get("allow_raise", True))
    allow_lower = bool(config.get("allow_lower", True))

    missed = summary["allowed_missed_points"]
    over = summary["allowed_over_points"]
    evidence_delta = missed - over
    proposed_delta = candidate - baseline
    accepted_delta = 0.0
    action = "keep_baseline"
    reason = "no_structured_directional_evidence"

    direction_agrees = (
        evidence_delta * proposed_delta > 0
        or (abs(proposed_delta) <= minimum_change and abs(evidence_delta) > minimum_change)
    )
    if confidence < min_confidence:
        reason = "structured_evidence_confidence_below_threshold"
    elif abs(evidence_delta) <= minimum_change:
        reason = "structured_evidence_supports_no_material_change"
    elif not direction_agrees:
        reason = "candidate_direction_conflicts_with_structured_evidence"
    else:
        proposed_magnitude = abs(proposed_delta)
        if proposed_magnitude <= minimum_change:
            proposed_magnitude = abs(evidence_delta)
        magnitude = min(proposed_magnitude, abs(evidence_delta), max_adjustment)
        if magnitude > minimum_change:
            accepted_delta = magnitude if evidence_delta > 0 else -magnitude
            action = "accept_structured_raise" if accepted_delta > 0 else "accept_structured_lower"
            reason = "validated_item_evidence"

    if action == "accept_structured_raise" and not allow_raise:
        accepted_delta = 0.0
        action = "keep_baseline"
        reason = "raise_action_disabled_by_validation_gate"
    elif action == "accept_structured_lower" and not allow_lower:
        accepted_delta = 0.0
        action = "keep_baseline"
        reason = "lower_action_disabled_by_validation_gate"

    final_score = max(0.0, min(max_score, baseline + accepted_delta))
    accepted = abs(final_score - baseline) > 1e-9
    if accepted:
        sequential_outcome = "auto_adjusted"
    elif confidence >= auto_keep_confidence and missed <= minimum_change and over <= minimum_change:
        sequential_outcome = "auto_kept_after_review"
    else:
        sequential_outcome = "defer_human"

    return {
        "policy_version": "structured_sequential_v1",
        "final_score": round(final_score, 4),
        "baseline_score": round(baseline, 4),
        "raw_candidate_score": round(candidate, 4),
        "bounded_candidate_score": round(final_score, 4),
        "delta_from_baseline": round(final_score - baseline, 4),
        "accepted": accepted,
        "action": action,
        "gate_reason": reason,
        "sequential_outcome": sequential_outcome,
        "requires_human_review": sequential_outcome == "defer_human",
        "agent_evidence_summary": summary,
    }


def route_score_band(score, max_score):
    """Return a stable low/mid/high score band for validation calibration."""
    max_score = max(safe_float(max_score, 0.0), 1.0)
    ratio = clamp01(safe_float(score, 0.0) / max_score)
    if ratio < 0.35:
        return "low"
    if ratio < 0.70:
        return "mid"
    return "high"


def _score_calibration_table(config):
    if not isinstance(config, dict):
        return {}
    score_config = config.get("score_calibration", config)
    if not isinstance(score_config, dict) or not score_config.get("enabled", False):
        return {}
    table = score_config.get("table", {})
    return table if isinstance(table, dict) else {}


def _score_calibration_value(config, key, default=None):
    score_config = config.get("score_calibration", config) if isinstance(config, dict) else {}
    if not isinstance(score_config, dict):
        return default
    return score_config.get(key, default)


def _score_calibration_diagnostics(config):
    score_config = config.get("score_calibration", config) if isinstance(config, dict) else {}
    if not isinstance(score_config, dict):
        return {}
    diagnostics = score_config.get("diagnostics", {})
    return diagnostics if isinstance(diagnostics, dict) else {}


def apply_route_score_calibration(
    score,
    max_score,
    question_id,
    route,
    post_calibration=None,
    config=None,
):
    """Apply validation-learned route/score-band correction with evidence guards.

    The calibration table is produced on validation data. Runtime application is
    intentionally additive and conservative: positive corrections are blocked
    when core contradiction is explicit, while negative corrections require
    stronger over-credit evidence to avoid worsening systematic under-scoring.
    """
    score = safe_float(score, 0.0)
    max_score = max(safe_float(max_score, 0.0), 1.0)
    question_id = str(question_id or "")
    route = str(route or "UNKNOWN")
    post_calibration = post_calibration or {}
    config = config or {}
    table = _score_calibration_table(config)
    band = route_score_band(score, max_score)
    result = {
        "enabled": bool(table),
        "applied": False,
        "reason": "no_score_calibration",
        "score_before": round(score, 4),
        "score_after": round(max(0.0, min(max_score, score)), 4),
        "correction": 0.0,
        "lookup_key": "",
        "score_band": band,
    }
    if not table:
        return result

    lookup_candidates = [
        ("question_route_band", f"{question_id}|{route}|{band}"),
        ("question_route", f"{question_id}|{route}"),
        ("question", question_id),
        ("route", route),
        ("global", "*"),
    ]
    selected_entry = None
    selected_key = ""
    for group, key in lookup_candidates:
        entry = table.get(group, {}).get(key) if isinstance(table.get(group), dict) else None
        if isinstance(entry, dict):
            selected_entry = entry
            selected_key = f"{group}:{key}"
            break
    if not selected_entry:
        result["reason"] = "no_matching_calibration_cell"
        return result

    raw_correction = safe_float(selected_entry.get("correction", 0.0), 0.0)
    max_points = safe_float(_score_calibration_value(config, "max_correction_points", 2.0), 2.0)
    max_ratio = safe_float(_score_calibration_value(config, "max_correction_ratio", 0.12), 0.12)
    cap = max(0.0, min(max_points, max_ratio * max_score))
    correction = max(-cap, min(cap, raw_correction))

    selected_group = selected_key.split(":", 1)[0] if selected_key else ""
    if selected_group in {"route", "global"} and abs(correction) >= 1e-9:
        diagnostics = _score_calibration_diagnostics(config)
        local_candidates = [
            ("question_route", f"{question_id}|{route}"),
            ("question", question_id),
        ]
        minimum_count = int(
            safe_float(
                _score_calibration_value(config, "direction_guard_min_count", 3),
                3,
            )
        )
        floor_ratio = safe_float(
            _score_calibration_value(config, "direction_guard_floor_ratio", 0.02),
            0.02,
        )
        direction_floor = max(0.10, floor_ratio * max_score)
        for local_group, local_key in local_candidates:
            group_entries = diagnostics.get(local_group, {})
            local_entry = (
                group_entries.get(local_key)
                if isinstance(group_entries, dict)
                else None
            )
            if not isinstance(local_entry, dict):
                continue
            local_n = int(safe_float(local_entry.get("n", 0), 0))
            local_mean = safe_float(local_entry.get("mean_residual", 0.0), 0.0)
            if (
                local_n >= max(1, minimum_count)
                and abs(local_mean) >= direction_floor
                and local_mean * correction < 0.0
            ):
                result.update({
                    "reason": "cross_question_direction_conflict",
                    "lookup_key": selected_key,
                    "correction": round(correction, 4),
                    "local_diagnostic_key": f"{local_group}:{local_key}",
                    "local_diagnostic_n": local_n,
                    "local_mean_residual": round(local_mean, 4),
                })
                return result

    core_support = clamp01(post_calibration.get("core_support_signal", 0.0))
    support_signal = clamp01(post_calibration.get("support_signal", 0.0))
    core_contradiction = clamp01(post_calibration.get("core_contradiction_ratio", 0.0))
    unsupported_high = clamp01(post_calibration.get("unsupported_high_score_risk", 0.0))
    bare_answer_risk = clamp01(post_calibration.get("bare_answer_risk", 0.0))
    confirmed_core_over = bool(post_calibration.get("confirmed_core_over_score", False))
    evidence_supported = bool(post_calibration.get("evidence_supported_answer", False))
    direct_only_high = bool(post_calibration.get("direct_only_high_score_risk", False))

    if abs(correction) < 1e-9:
        result.update({
            "reason": "zero_correction",
            "lookup_key": selected_key,
        })
        return result

    if correction > 0:
        if confirmed_core_over or core_contradiction >= 0.25 or unsupported_high >= 0.35 or direct_only_high:
            result.update({
                "reason": "raise_blocked_by_core_overcredit_evidence",
                "lookup_key": selected_key,
                "correction": round(correction, 4),
            })
            return result
        if bare_answer_risk >= 0.45 and core_support < 0.55 and support_signal < 0.55:
            result.update({
                "reason": "raise_blocked_by_weak_evidence",
                "lookup_key": selected_key,
                "correction": round(correction, 4),
            })
            return result
    else:
        lower_allowed = (
            confirmed_core_over
            or core_contradiction >= 0.20
            or unsupported_high >= 0.20
            or direct_only_high
            or (core_support < 0.45 and not evidence_supported and bare_answer_risk >= 0.25)
        )
        if not lower_allowed:
            result.update({
                "reason": "lower_blocked_without_core_overcredit_evidence",
                "lookup_key": selected_key,
                "correction": round(correction, 4),
            })
            return result
        if core_support >= 0.70 and unsupported_high < 0.25 and core_contradiction < 0.20:
            result.update({
                "reason": "lower_blocked_by_core_support",
                "lookup_key": selected_key,
                "correction": round(correction, 4),
            })
            return result

    adjusted = round(max(0.0, min(max_score, score + correction)), 4)
    result.update({
        "applied": adjusted != round(max(0.0, min(max_score, score)), 4),
        "reason": "validation_route_score_band_correction",
        "score_after": adjusted,
        "correction": round(adjusted - score, 4),
        "lookup_key": selected_key,
        "cell_n": selected_entry.get("n"),
        "cell_mean_residual": selected_entry.get("mean_residual"),
    })
    return result


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
        or safe_float(
            post_calibration.get(
                "effective_unsupported_match_points_ratio",
                post_calibration.get("unsupported_match_points_ratio", 0.0),
            ),
            0.0,
        ) >= 0.15
        or bool(post_calibration.get("core_anchor_failed", False))
    )
    strong_over_score_signal = (
        high_blank_high_score
        or fatal_points_ratio >= 0.50
        or perception_risk >= 0.66
        or safe_float(
            post_calibration.get(
                "effective_unsupported_match_points_ratio",
                post_calibration.get("unsupported_match_points_ratio", 0.0),
            ),
            0.0,
        ) >= 0.25
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
