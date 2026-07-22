"""Semantic contracts and validation for rubric refinement."""

from __future__ import annotations

import re
import math
import unicodedata
from copy import deepcopy
from typing import Any


RUBRIC_SEMANTIC_CONTRACT_VERSION = 6

# A coarse immutable baseline may remain active only when a structurally valid
# refinement was evaluated on paired teacher scores and failed the deployment
# non-inferiority gate. This is a model-selection result, not a diagnostic
# fallback and not a general exemption from semantic validation.
SEMANTIC_MODE_CALIBRATED_BASELINE = "calibrated_noninferior_baseline_selected"

# High-value composite parents are too coarse to grade reliably as one opaque
# item. Two children are enough to expose partial credit without over-fragmenting
# the official rubric. Truly atomic outcomes remain exempt.
HIGH_VALUE_SPLIT_THRESHOLD = 4.0
MIN_HIGH_VALUE_SCORING_CHILDREN = 2
MIN_COMPLEX_PROCESS_CHILDREN = 3

SCORING_POLICY_STRICT_ATOMIC = "strict_atomic"
SCORING_POLICY_ADDITIVE = "additive_split"
SCORING_POLICY_HIERARCHICAL = "final_sufficient_partial_credit"
SCORING_POLICY_ROLE_WEIGHTED = "role_weighted_additive"
SUPPORTED_SCORING_POLICIES = {
    SCORING_POLICY_STRICT_ATOMIC,
    SCORING_POLICY_ADDITIVE,
    SCORING_POLICY_HIERARCHICAL,
    SCORING_POLICY_ROLE_WEIGHTED,
}

TASK_SEMANTICS_STRICT_ATOMIC = "strict_atomic"
TASK_SEMANTICS_RESULT_SUFFICIENT = "result_sufficient"
TASK_SEMANTICS_ORTHOGONAL = "orthogonal_additive"
TASK_SEMANTICS_COMPONENT = "component_additive"
TASK_SEMANTICS_PROCESS_DOMINANT = "process_dominant"
SUPPORTED_TASK_SEMANTICS = {
    TASK_SEMANTICS_STRICT_ATOMIC,
    TASK_SEMANTICS_RESULT_SUFFICIENT,
    TASK_SEMANTICS_ORTHOGONAL,
    TASK_SEMANTICS_COMPONENT,
    TASK_SEMANTICS_PROCESS_DOMINANT,
}

SCORING_ROLE_SUPPORT = "support_process"
SCORING_ROLE_CORE = "core_process"
SCORING_ROLE_FINAL = "final"
SCORING_ROLE_COMPONENT = "component"
SUPPORTED_SCORING_ROLES = {
    SCORING_ROLE_SUPPORT,
    SCORING_ROLE_CORE,
    SCORING_ROLE_FINAL,
    SCORING_ROLE_COMPONENT,
}


def _normalized_answer_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).upper()


def immutable_answer_literals(value: Any) -> set[str]:
    """Extract strong base-number facts that a refinement must not invent.

    These literals are deliberately narrow. Binary and hexadecimal anchors are
    common in computer-science rubrics and can be compared deterministically;
    ordinary decimal values are left to the existing semantic/canonical checks
    because splitting a derivation may legitimately expose intermediate values.
    """
    text = _normalized_answer_text(value)
    literals: set[str] = set()
    binary_pattern = re.compile(
        r"(?<![0-9A-F])(?:[01]{4}(?:[\s_]+[01]{4})+|[01]{4,})\s*B?(?![0-9A-F])"
    )
    for match in binary_pattern.finditer(text):
        bits = re.sub(r"[\s_B]", "", match.group(0))
        if len(bits) >= 4:
            literals.add(f"BIN:{bits}")

    hex_pattern = re.compile(
        r"(?<![0-9A-F])(?:0X[0-9A-F]+|[0-9][0-9A-F]*H)(?![0-9A-F])"
    )
    for match in hex_pattern.finditer(text):
        token = match.group(0)
        digits = token[2:] if token.startswith("0X") else token[:-1]
        if digits:
            literals.add(f"HEX:{int(digits, 16):X}")
    return literals


def answer_conclusion(value: Any) -> str:
    """Return the final explicit binary judgement in an answer, if present."""
    text = _normalized_answer_text(value)
    matches = list(
        re.finditer(
            r"未命中|不命中|命中|不成立|成立|不正确|错误|正确",
            text,
        )
    )
    if not matches:
        return ""
    token = matches[-1].group(0)
    if token in {"未命中", "不命中"}:
        return "cache_miss"
    if token == "命中":
        return "cache_hit"
    if token == "不成立":
        return "false"
    if token == "成立":
        return "true"
    if token in {"不正确", "错误"}:
        return "incorrect"
    return "correct"


def assess_candidate_replay(
    records: list[dict[str, Any]],
    max_score: float,
    *,
    minimum_coverage: float = 0.80,
    mae_margin_ratio: float = 0.02,
    severe_regression_ratio: float = 0.20,
) -> dict[str, Any]:
    """Apply a teacher-score non-inferiority gate on rubric-calibration data."""
    expected = len(records)
    paired = [
        record
        for record in records
        if record.get("baseline_score") is not None
        and record.get("candidate_score") is not None
        and record.get("teacher_score") is not None
    ]
    required = min(expected, max(1, math.ceil(expected * minimum_coverage)))
    max_score = max(float(max_score or 0.0), 1.0)
    margin = max(0.10, mae_margin_ratio * max_score)
    severe_margin = max(0.50, severe_regression_ratio * max_score)

    report = {
        "method": "paired_teacher_score_noninferiority",
        "expected": expected,
        "paired": len(paired),
        "required": required,
        "coverage": round(len(paired) / expected, 6) if expected else 0.0,
        "mae_margin": round(margin, 6),
        "severe_regression_margin": round(severe_margin, 6),
        "accepted": False,
        "reason": "insufficient_replay_coverage",
    }
    if expected == 0 or len(paired) < required:
        return report

    baseline_errors = [
        abs(float(item["baseline_score"]) - float(item["teacher_score"]))
        for item in paired
    ]
    candidate_errors = [
        abs(float(item["candidate_score"]) - float(item["teacher_score"]))
        for item in paired
    ]
    baseline_mae = sum(baseline_errors) / len(baseline_errors)
    candidate_mae = sum(candidate_errors) / len(candidate_errors)
    regressions = [
        candidate - baseline
        for baseline, candidate in zip(baseline_errors, candidate_errors)
    ]
    severe_regressions = sum(value > severe_margin for value in regressions)
    accepted = (
        candidate_mae <= baseline_mae + margin
        and severe_regressions == 0
    )
    report.update({
        "baseline_mae": round(baseline_mae, 6),
        "candidate_mae": round(candidate_mae, 6),
        "mae_delta": round(candidate_mae - baseline_mae, 6),
        "improved": sum(value < -1e-9 for value in regressions),
        "unchanged": sum(abs(value) <= 1e-9 for value in regressions),
        "worsened": sum(value > 1e-9 for value in regressions),
        "severe_regressions": severe_regressions,
        "accepted": accepted,
        "reason": "accepted_noninferior" if accepted else (
            "severe_sample_regression"
            if severe_regressions
            else "candidate_mae_exceeds_margin"
        ),
    })
    return report


def candidate_replay_supports_baseline_selection(report: dict[str, Any]) -> bool:
    """Return whether paired replay formally supports retaining the baseline."""
    if report.get("method") != "paired_teacher_score_noninferiority":
        return False
    if report.get("accepted") is not False:
        return False
    if report.get("reason") not in {
        "severe_sample_regression",
        "candidate_mae_exceeds_margin",
    }:
        return False
    try:
        expected = int(report.get("expected", 0) or 0)
        paired = int(report.get("paired", 0) or 0)
        required = int(report.get("required", 0) or 0)
        baseline_mae = float(report["baseline_mae"])
        candidate_mae = float(report["candidate_mae"])
        margin = float(report["mae_margin"])
        severe_regressions = int(report.get("severe_regressions", 0) or 0)
    except (KeyError, TypeError, ValueError):
        return False
    if expected <= 0 or required <= 0 or paired < required:
        return False
    if not all(math.isfinite(value) for value in (baseline_mae, candidate_mae, margin)):
        return False
    if report["reason"] == "severe_sample_regression":
        return severe_regressions > 0
    return (
        severe_regressions == 0
        and candidate_mae > baseline_mae + margin - 1e-9
    )


def manifest_allows_unchanged_baseline(manifest: dict[str, Any]) -> bool:
    """Validate evidence for the deployable calibrated-baseline mode."""
    return bool(
        manifest.get("semantic_validation_mode")
        == SEMANTIC_MODE_CALIBRATED_BASELINE
        and manifest.get("selected_variant") == "baseline"
        and manifest.get("decomposition_deferred") is True
        and candidate_replay_supports_baseline_selection(
            manifest.get("candidate_replay") or {}
        )
    )


ATOMIC_OUTCOME_PATTERNS = (
    r"最终结果",
    r"得出.*结果",
    r"写出.*(?:排列顺序|屏蔽字)",
    r"明确指出.*位数",
    r"得出正确的.*格式",
    r"操作数内容",
    r"总长度",
    r"总容量[（(]?(?:位|字节)",
)

COMPOUND_REQUIREMENT_PATTERNS = (
    r"并说明",
    r"及判断",
    r"并判断",
    r"分别",
    r"各级",
    r"顺序图",
    r"轨迹图",
    r"画出",
)

ORTHOGONAL_REQUIREMENT_PATTERNS = (
    r"及判断",
    r"并判断",
    r"分别",
    r"各级",
    r"计算.+及.+",
    r"计算.+和.+",
)

COMPONENT_REQUIREMENT_PATTERNS = (
    r"顺序图",
    r"轨迹图",
    r"画出",
    r"数据通路",
    r"表格",
    r"(?:多个|各|[0-9一二三四五六七八九十]+个).*(?:组|部分|阶段).*位数",
)

PROCESS_REQUIREMENT_PATTERNS = (
    r"并说明",
    r"说明理由",
    r"证明",
    r"推导",
    r"映射",
    r"比较",
    r"计算",
)

COMPLEX_PROCESS_PATTERNS = (
    r"并说明",
    r"说明理由",
    r"证明",
    r"推导",
    r"映射",
    r"比较",
    r"命中",
)


def _item_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key, "") or "")
        for key in (
            "item",
            "source_text",
            "parent_official_item",
        )
    )


def infer_parent_task_semantics(item: dict[str, Any]) -> str:
    """Classify what evidence an official parent item actually rewards.

    Explicit metadata always wins.  The lexical fallback is intentionally
    conservative: only prompts that request reasoning are classified as
    process dominant, while independently checkable outputs remain additive.
    """
    explicit = str(item.get("task_semantics", "")).strip().lower()
    if explicit in SUPPORTED_TASK_SEMANTICS:
        return explicit

    policy = str(item.get("scoring_policy", "")).strip().lower()
    if policy == SCORING_POLICY_HIERARCHICAL:
        return TASK_SEMANTICS_RESULT_SUFFICIENT
    if policy == SCORING_POLICY_STRICT_ATOMIC:
        return TASK_SEMANTICS_STRICT_ATOMIC
    if policy == SCORING_POLICY_ROLE_WEIGHTED:
        return TASK_SEMANTICS_PROCESS_DOMINANT

    text = _item_text(item)
    if any(re.search(pattern, text) for pattern in COMPONENT_REQUIREMENT_PATTERNS):
        return TASK_SEMANTICS_COMPONENT
    if any(re.search(pattern, text) for pattern in ORTHOGONAL_REQUIREMENT_PATTERNS):
        return TASK_SEMANTICS_ORTHOGONAL
    if any(re.search(pattern, text) for pattern in PROCESS_REQUIREMENT_PATTERNS):
        return TASK_SEMANTICS_PROCESS_DOMINANT
    if is_atomic_outcome_item(item):
        return TASK_SEMANTICS_STRICT_ATOMIC
    return TASK_SEMANTICS_ORTHOGONAL


def infer_process_complexity(item: dict[str, Any]) -> str:
    explicit = str(item.get("process_complexity", "")).strip().lower()
    if explicit in {"short", "complex"}:
        return explicit
    text = _item_text(item)
    hits = sum(bool(re.search(pattern, text)) for pattern in COMPLEX_PROCESS_PATTERNS)
    return "complex" if hits >= 2 or "并说明" in text else "short"


def is_atomic_outcome_item(item: dict[str, Any]) -> bool:
    """Return whether full credit is anchored to one final outcome."""
    explicit = str(item.get("split_policy", "")).strip().lower()
    if explicit == "preserve_atomic":
        return True
    if explicit == "allow_semantic_split":
        return False

    text = _item_text(item)
    if any(re.search(pattern, text) for pattern in COMPOUND_REQUIREMENT_PATTERNS):
        return False
    return any(re.search(pattern, text) for pattern in ATOMIC_OUTCOME_PATTERNS)


def has_machine_readable_decomposition_anchor(item: dict[str, Any]) -> bool:
    """Return whether a text optimizer can safely derive scoring children.

    A reference image alone is evidence for grading, but its path is not a
    machine-readable answer specification for the text-only rubric refiner.
    Such parents remain auditable coarse items until structured reference facts
    are supplied by the dataset or a separate visual-reference extraction step.
    """
    if str(item.get("standard_answer_text", "") or "").strip():
        return True
    canonicalization = item.get("canonicalization")
    if isinstance(canonicalization, dict):
        fields = canonicalization.get("fields")
        if isinstance(fields, list) and fields:
            return True
    return False


def prepare_rubric_semantic_contract(
    rubric: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach stable parent and split-policy metadata without changing points."""
    contracted: list[dict[str, Any]] = []
    for index, raw_item in enumerate(rubric):
        item = deepcopy(raw_item)
        item_id = str(item.get("id") or f"item_{index + 1}")
        item["id"] = item_id
        item.setdefault("parent_id", item_id)
        item["semantic_contract_version"] = RUBRIC_SEMANTIC_CONTRACT_VERSION
        item.setdefault("parent_points", float(item.get("points", 0) or 0))
        parent_points = float(item.get("parent_points", 0) or 0)
        task_semantics = infer_parent_task_semantics(item)
        item["task_semantics"] = task_semantics
        scoring_policy = str(item.get("scoring_policy", "")).strip().lower()
        if scoring_policy not in SUPPORTED_SCORING_POLICIES:
            scoring_policy = {
                TASK_SEMANTICS_STRICT_ATOMIC: SCORING_POLICY_STRICT_ATOMIC,
                TASK_SEMANTICS_RESULT_SUFFICIENT: SCORING_POLICY_HIERARCHICAL,
                TASK_SEMANTICS_PROCESS_DOMINANT: (
                    SCORING_POLICY_ROLE_WEIGHTED
                    if parent_points >= HIGH_VALUE_SPLIT_THRESHOLD
                    else SCORING_POLICY_STRICT_ATOMIC
                ),
                TASK_SEMANTICS_ORTHOGONAL: SCORING_POLICY_ADDITIVE,
                TASK_SEMANTICS_COMPONENT: SCORING_POLICY_ADDITIVE,
            }[task_semantics]
        item["scoring_policy"] = scoring_policy

        if scoring_policy == SCORING_POLICY_STRICT_ATOMIC:
            item["split_policy"] = "preserve_atomic"
            item.setdefault("weighting_policy", "preserve_parent")
            item["decomposition_required"] = False
            item.pop("minimum_scoring_children", None)
            if parent_points >= HIGH_VALUE_SPLIT_THRESHOLD:
                item["decomposition_exemption"] = "strict_atomic_single_outcome"
        else:
            item["split_policy"] = "allow_semantic_split"
            default_weighting = {
                SCORING_POLICY_HIERARCHICAL: "preserve_parent",
                SCORING_POLICY_ADDITIVE: "equal_atomic",
                SCORING_POLICY_ROLE_WEIGHTED: "role_constrained",
            }[scoring_policy]
            item["weighting_policy"] = default_weighting
            item.pop("decomposition_exemption", None)
            decomposition_requested = bool(
                scoring_policy in {
                    SCORING_POLICY_HIERARCHICAL,
                    SCORING_POLICY_ROLE_WEIGHTED,
                }
                or parent_points >= HIGH_VALUE_SPLIT_THRESHOLD
            )
            decomposition_anchor = has_machine_readable_decomposition_anchor(item)
            item["decomposition_required"] = bool(
                decomposition_requested and decomposition_anchor
            )
            if decomposition_requested and not decomposition_anchor:
                item["decomposition_exemption"] = (
                    "insufficient_machine_readable_reference_anchor"
                )
            else:
                item.pop("decomposition_exemption", None)
            if item["decomposition_required"]:
                minimum = MIN_HIGH_VALUE_SCORING_CHILDREN
                if scoring_policy == SCORING_POLICY_ROLE_WEIGHTED:
                    complexity = infer_process_complexity(item)
                    item["process_complexity"] = complexity
                    minimum = (
                        MIN_COMPLEX_PROCESS_CHILDREN
                        if complexity == "complex"
                        else MIN_HIGH_VALUE_SCORING_CHILDREN
                    )
                item["minimum_scoring_children"] = max(
                    minimum,
                    int(item.get("minimum_scoring_children", 0) or 0),
                )
            else:
                item.pop("minimum_scoring_children", None)

        if scoring_policy in {
            SCORING_POLICY_STRICT_ATOMIC,
            SCORING_POLICY_HIERARCHICAL,
        }:
            item.setdefault("full_credit_anchor", item.get("standard_answer_text", ""))
        if scoring_policy == SCORING_POLICY_HIERARCHICAL:
            item.setdefault("full_credit_policy", "final_answer_sufficient")
            item.setdefault(
                "fallback_cap",
                max(
                    0.0,
                    float(item.get("parent_points", item.get("points", 0)) or 0)
                    - float(item.get("points", 0) or 0)
                    if item.get("full_credit_trigger")
                    else float(item.get("fallback_cap", 0) or 0),
                ),
            )
            item["full_credit_trigger"] = bool(item.get("full_credit_trigger", False))
        if scoring_policy == SCORING_POLICY_ROLE_WEIGHTED:
            complexity = infer_process_complexity(item)
            item["process_complexity"] = complexity
            item["dependency_mode"] = str(
                item.get("dependency_mode", "independent")
            ).strip().lower()
            if complexity == "complex":
                item["minimum_process_ratio"] = 0.80
                item["minimum_core_process_ratio"] = 0.50
                item["maximum_final_ratio"] = 0.20
            else:
                item["minimum_process_ratio"] = 0.65
                item["minimum_core_process_ratio"] = 0.0
                item["maximum_final_ratio"] = 0.35
        contracted.append(item)
    return contracted


def high_value_split_targets(
    rubric: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return high-value composite parents whose decomposition is incomplete."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in prepare_rubric_semantic_contract(rubric):
        parent_id = str(item.get("parent_id") or item.get("id", ""))
        groups.setdefault(parent_id, []).append(item)

    targets = []
    for parent_id, items in groups.items():
        first = items[0]
        parent_points = float(first.get("parent_points", 0) or 0)
        if (
            parent_points < HIGH_VALUE_SPLIT_THRESHOLD
            or not first.get("decomposition_required")
            or first.get("scoring_policy") not in {
                SCORING_POLICY_ADDITIVE,
                SCORING_POLICY_ROLE_WEIGHTED,
            }
        ):
            continue
        minimum_children = int(
            first.get(
                "minimum_scoring_children",
                MIN_HIGH_VALUE_SCORING_CHILDREN,
            )
        )
        scoring_children = sum(
            float(item.get("points", 0) or 0) > 0 for item in items
        )
        if scoring_children >= minimum_children:
            continue
        targets.append(
            {
                "parent_id": parent_id,
                "parent_points": parent_points,
                "current_scoring_children": scoring_children,
                "minimum_scoring_children": minimum_children,
                "weighting_policy": first.get(
                    "weighting_policy", "equal_atomic"
                ),
            }
        )
    return targets


def project_refined_candidate_to_contract(
    original_rubric: list[dict[str, Any]],
    candidate_rubric: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project an LLM proposal onto immutable parent-level constraints.

    The model may propose changes only inside parents that are both splittable
    and backed by machine-readable reference facts. Atomic parents and parents
    exempt from automatic decomposition are copied byte-for-byte at the rubric
    item level. Equal-atomic children are normalized to conserve the official
    parent score, preventing aggregate-parent plus child double counting.
    """
    original = prepare_rubric_semantic_contract(original_rubric)
    original_groups: dict[str, list[dict[str, Any]]] = {}
    parent_order: list[str] = []
    for item in original:
        parent_id = str(item.get("parent_id") or item["id"])
        if parent_id not in original_groups:
            original_groups[parent_id] = []
            parent_order.append(parent_id)
        original_groups[parent_id].append(item)

    candidate_groups: dict[str, list[dict[str, Any]]] = {}
    for raw_item in candidate_rubric or []:
        item = deepcopy(raw_item)
        item_id = str(item.get("id", "") or "")
        parent_id = str(item.get("parent_id", "") or "")
        if not parent_id and item_id in original_groups:
            parent_id = item_id
        if parent_id in original_groups:
            candidate_groups.setdefault(parent_id, []).append(item)

    projected: list[dict[str, Any]] = []
    inherited_keys = (
        "parent_points",
        "split_policy",
        "weighting_policy",
        "task_semantics",
        "scoring_policy",
        "process_complexity",
        "minimum_process_ratio",
        "minimum_core_process_ratio",
        "maximum_final_ratio",
        "dependency_mode",
        "full_credit_anchor",
        "full_credit_policy",
        "fallback_cap",
        "decomposition_required",
        "minimum_scoring_children",
        "decomposition_exemption",
    )
    for parent_id in parent_order:
        source_items = original_groups[parent_id]
        spec = source_items[0]
        immutable_parent = bool(
            spec.get("split_policy") == "preserve_atomic"
            or spec.get("decomposition_exemption")
            == "insufficient_machine_readable_reference_anchor"
        )
        proposed = candidate_groups.get(parent_id, [])
        if immutable_parent or not proposed:
            projected.extend(deepcopy(source_items))
            continue

        positive = [
            item for item in proposed if float(item.get("points", 0) or 0) > 0
        ]
        if len(positive) > 1:
            without_aggregate = [
                item for item in proposed if str(item.get("id", "")) != parent_id
            ]
            if any(float(item.get("points", 0) or 0) > 0 for item in without_aggregate):
                proposed = without_aggregate

        for item in proposed:
            item["parent_id"] = parent_id
            for key in inherited_keys:
                if key in spec:
                    item[key] = deepcopy(spec[key])
                else:
                    item.pop(key, None)

        scoring_children = [
            item for item in proposed if float(item.get("points", 0) or 0) > 0
        ]
        if spec.get("weighting_policy") == "equal_atomic" and scoring_children:
            child_points = float(spec.get("parent_points", 0) or 0) / len(scoring_children)
            for item in scoring_children:
                item["points"] = child_points
        projected.extend(proposed)

    return prepare_rubric_semantic_contract(projected)


def rubric_scoring_signature(rubric: list[dict[str, Any]]) -> tuple:
    """Return scoring-relevant content while ignoring generated audit metadata."""
    rows = []
    for item in prepare_rubric_semantic_contract(rubric):
        rows.append(
            (
                str(item.get("parent_id") or ""),
                str(item.get("id") or ""),
                str(item.get("item") or "").strip(),
                round(float(item.get("points", 0) or 0), 8),
                str(item.get("standard_answer_text") or "").strip(),
                str(item.get("scoring_policy") or ""),
                str(item.get("task_semantics") or ""),
                str(item.get("scoring_role") or ""),
                str(item.get("weighting_policy") or ""),
                str(item.get("dependency_mode") or ""),
                bool(item.get("full_credit_trigger", False)),
            )
        )
    return tuple(sorted(rows))


def rubric_structure_signature(rubric: list[dict[str, Any]]) -> tuple:
    """Return parent-to-positive-child cardinality for structural comparison."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in prepare_rubric_semantic_contract(rubric):
        parent_id = str(item.get("parent_id") or item.get("id", ""))
        groups.setdefault(parent_id, []).append(item)
    return tuple(
        sorted(
            (
                parent_id,
                str(items[0].get("scoring_policy") or ""),
                sum(float(item.get("points", 0) or 0) > 0 for item in items),
            )
            for parent_id, items in groups.items()
        )
    )


def validate_refined_rubric(
    original_rubric: list[dict[str, Any]],
    refined_rubric: list[dict[str, Any]],
    total_score: float,
    tolerance: float = 1e-6,
    *,
    allow_unchanged_baseline: bool = False,
) -> tuple[bool, list[str]]:
    """Validate score conservation, traceability, and policy structure.

    ``allow_unchanged_baseline`` is reserved for the non-inferiority fallback:
    an immutable baseline may remain coarse when every refined candidate was
    rejected, but only when its complete scoring signature and structure are
    unchanged. A changed candidate therefore cannot bypass this contract.
    """
    preserving_unchanged_baseline = bool(
        allow_unchanged_baseline
        and rubric_scoring_signature(original_rubric)
        == rubric_scoring_signature(refined_rubric)
        and rubric_structure_signature(original_rubric)
        == rubric_structure_signature(refined_rubric)
    )
    originals_list = prepare_rubric_semantic_contract(original_rubric)
    original_groups: dict[str, list[dict[str, Any]]] = {}
    for item in originals_list:
        parent_id = str(item.get("parent_id") or item["id"])
        original_groups.setdefault(parent_id, []).append(item)

    parent_specs: dict[str, dict[str, Any]] = {}
    for parent_id, items in original_groups.items():
        first = items[0]
        parent_points = float(first.get("parent_points", 0) or 0)
        if parent_points <= 0:
            parent_points = sum(float(item.get("points", 0) or 0) for item in items)
        parent_specs[parent_id] = {
            "parent_points": parent_points,
            "split_policy": first.get("split_policy", "allow_semantic_split"),
            "weighting_policy": first.get("weighting_policy", "equal_atomic"),
            "scoring_policy": first.get("scoring_policy", SCORING_POLICY_ADDITIVE),
            "task_semantics": first.get("task_semantics", ""),
            "process_complexity": first.get("process_complexity", ""),
            "dependency_mode": first.get("dependency_mode", "independent"),
            "minimum_process_ratio": float(first.get("minimum_process_ratio", 0) or 0),
            "minimum_core_process_ratio": float(
                first.get("minimum_core_process_ratio", 0) or 0
            ),
            "maximum_final_ratio": float(first.get("maximum_final_ratio", 1) or 1),
            "full_credit_anchor": first.get("full_credit_anchor", ""),
            "fallback_cap": float(first.get("fallback_cap", 0) or 0),
            "decomposition_required": bool(
                first.get("decomposition_required", False)
            ),
            "minimum_scoring_children": int(
                first.get("minimum_scoring_children", 0) or 0
            ),
            "items": items,
            "immutable_answer_literals": sorted({
                literal
                for original_item in items
                for literal in immutable_answer_literals(
                    original_item.get("standard_answer_text", "")
                )
            }),
            "answer_conclusion": answer_conclusion(
                " ".join(
                    str(original_item.get("standard_answer_text", "") or "")
                    for original_item in items
                )
            ),
        }
    errors: list[str] = []
    seen_ids: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in parent_specs}

    for item in refined_rubric:
        item_id = str(item.get("id", "") or "")
        if not item_id:
            errors.append("refined item is missing id")
        elif item_id in seen_ids:
            errors.append(f"duplicate refined item id: {item_id}")
        seen_ids.add(item_id)

        try:
            points = float(item.get("points", 0) or 0)
        except (TypeError, ValueError):
            points = -1.0
        if points < 0:
            errors.append(f"invalid points for {item_id or '<missing>'}")

        parent_id = str(item.get("parent_id", "") or "")
        if not parent_id and item_id in parent_specs:
            parent_id = item_id
        if parent_id not in parent_specs:
            errors.append(
                f"refined item {item_id or '<missing>'} has unknown parent_id {parent_id!r}"
            )
            continue
        item["parent_id"] = parent_id
        spec = parent_specs[parent_id]
        item["parent_points"] = spec["parent_points"]
        item["split_policy"] = spec["split_policy"]
        item["weighting_policy"] = spec["weighting_policy"]
        item["scoring_policy"] = spec["scoring_policy"]
        item["task_semantics"] = spec["task_semantics"]
        if spec["scoring_policy"] == SCORING_POLICY_ROLE_WEIGHTED:
            item["process_complexity"] = spec["process_complexity"]
            item["dependency_mode"] = spec["dependency_mode"]
            item["minimum_process_ratio"] = spec["minimum_process_ratio"]
            item["minimum_core_process_ratio"] = spec[
                "minimum_core_process_ratio"
            ]
            item["maximum_final_ratio"] = spec["maximum_final_ratio"]
        if spec["full_credit_anchor"]:
            item["full_credit_anchor"] = spec["full_credit_anchor"]
        if spec["scoring_policy"] == SCORING_POLICY_HIERARCHICAL:
            item["fallback_cap"] = spec["fallback_cap"]
        grouped[parent_id].append(item)

    refined_total = sum(float(item.get("points", 0) or 0) for item in refined_rubric)
    if abs(refined_total - float(total_score)) > tolerance:
        errors.append(f"total score changed: {refined_total} != {total_score}")

    for parent_id, spec in parent_specs.items():
        children = grouped[parent_id]
        child_total = sum(float(item.get("points", 0) or 0) for item in children)
        original_points = spec["parent_points"]
        if abs(child_total - original_points) > tolerance:
            errors.append(
                f"parent {parent_id} score changed: {child_total} != {original_points}"
            )

        scoring_children = [
            item for item in children if float(item.get("points", 0) or 0) > tolerance
        ]
        allowed_literals = set(spec.get("immutable_answer_literals", []))
        for child in scoring_children:
            child_id = str(child.get("id", "") or "<missing>")
            introduced_literals = (
                immutable_answer_literals(child.get("standard_answer_text", ""))
                - allowed_literals
            )
            if introduced_literals:
                errors.append(
                    f"child {child_id} introduced unsupported answer literals: "
                    + ", ".join(sorted(introduced_literals))
                )
            if str(child.get("scoring_role", "")).strip().lower() == SCORING_ROLE_FINAL:
                original_conclusion = str(spec.get("answer_conclusion", "") or "")
                child_conclusion = answer_conclusion(
                    child.get("standard_answer_text", "")
                )
                if (
                    original_conclusion
                    and child_conclusion
                    and child_conclusion != original_conclusion
                ):
                    errors.append(
                        f"final child {child_id} changed parent conclusion: "
                        f"{child_conclusion} != {original_conclusion}"
                    )
        if preserving_unchanged_baseline:
            # Score conservation, IDs, parent traceability, and immutable
            # answer literals were checked above. Structural decomposition is
            # an optimization objective, so an unchanged baseline may defer it
            # when the calibrated candidate is demonstrably worse.
            continue
        if spec["scoring_policy"] == SCORING_POLICY_ADDITIVE:
            minimum_children = int(spec.get("minimum_scoring_children", 0) or 0)
            if spec.get("decomposition_required") and minimum_children < 2:
                minimum_children = MIN_HIGH_VALUE_SCORING_CHILDREN
            if minimum_children and len(scoring_children) < minimum_children:
                errors.append(
                    f"additive parent {parent_id} must contain at least "
                    f"{minimum_children} scoring items, got {len(scoring_children)}"
                )
            if len(scoring_children) > 1:
                for child in scoring_children:
                    child_id = str(child.get("id", "") or "<missing>")
                    if not str(child.get("item", "") or "").strip():
                        errors.append(
                            f"additive child {child_id} has no objective scoring item"
                        )
                    if not str(child.get("standard_answer_text", "") or "").strip():
                        errors.append(
                            f"additive child {child_id} has no standard answer"
                        )
        if spec["scoring_policy"] == SCORING_POLICY_ROLE_WEIGHTED:
            minimum_children = int(spec.get("minimum_scoring_children", 0) or 0)
            if len(scoring_children) < minimum_children:
                errors.append(
                    f"role-weighted parent {parent_id} must contain at least "
                    f"{minimum_children} scoring items, got {len(scoring_children)}"
                )
            roles: dict[str, list[dict[str, Any]]] = {}
            for child in scoring_children:
                child_id = str(child.get("id", "") or "<missing>")
                role = str(child.get("scoring_role", "")).strip().lower()
                if role not in SUPPORTED_SCORING_ROLES:
                    errors.append(
                        f"role-weighted child {child_id} has invalid scoring_role {role!r}"
                    )
                    continue
                roles.setdefault(role, []).append(child)
                if not str(child.get("item", "") or "").strip():
                    errors.append(
                        f"role-weighted child {child_id} has no objective scoring item"
                    )
                if not str(child.get("standard_answer_text", "") or "").strip():
                    errors.append(
                        f"role-weighted child {child_id} has no standard answer"
                    )

            if len(roles.get(SCORING_ROLE_FINAL, [])) != 1:
                errors.append(
                    f"role-weighted parent {parent_id} must have exactly one final child"
                )
            if not roles.get(SCORING_ROLE_CORE):
                errors.append(
                    f"role-weighted parent {parent_id} must have a core-process child"
                )
            if (
                spec.get("process_complexity") == "complex"
                and not roles.get(SCORING_ROLE_SUPPORT)
            ):
                errors.append(
                    f"complex role-weighted parent {parent_id} must have a support-process child"
                )

            final_points = sum(
                float(item.get("points", 0) or 0)
                for item in roles.get(SCORING_ROLE_FINAL, [])
            )
            process_points = sum(
                float(item.get("points", 0) or 0)
                for role in (SCORING_ROLE_SUPPORT, SCORING_ROLE_CORE)
                for item in roles.get(role, [])
            )
            core_points = sum(
                float(item.get("points", 0) or 0)
                for item in roles.get(SCORING_ROLE_CORE, [])
            )
            denominator = max(original_points, tolerance)
            if final_points / denominator > spec["maximum_final_ratio"] + tolerance:
                errors.append(
                    f"role-weighted parent {parent_id} gives too much weight to the final answer"
                )
            if process_points / denominator + tolerance < spec["minimum_process_ratio"]:
                errors.append(
                    f"role-weighted parent {parent_id} gives insufficient process weight"
                )
            if (
                core_points / denominator + tolerance
                < spec["minimum_core_process_ratio"]
            ):
                errors.append(
                    f"role-weighted parent {parent_id} gives insufficient core-process weight"
                )
            if spec["dependency_mode"] not in {"independent", "evidence_required"}:
                errors.append(
                    f"role-weighted parent {parent_id} has invalid dependency_mode"
                )
        if (
            spec["weighting_policy"] == "equal_atomic"
            and len(scoring_children) > 1
        ):
            child_points = [float(item.get("points", 0) or 0) for item in scoring_children]
            if max(child_points) - min(child_points) > tolerance:
                errors.append(f"parent {parent_id} must use equal atomic weights")

        if spec["scoring_policy"] == SCORING_POLICY_HIERARCHICAL:
            if len(scoring_children) < 2:
                errors.append(
                    f"hierarchical parent {parent_id} must contain a final trigger "
                    "and at least one partial-credit item"
                )
                continue
            triggers = [item for item in scoring_children if item.get("full_credit_trigger")]
            if len(triggers) != 1:
                errors.append(
                    f"hierarchical parent {parent_id} must have exactly one "
                    f"full_credit_trigger, got {len(triggers)}"
                )
                continue
            trigger = triggers[0]
            anchor = str(spec.get("full_credit_anchor", "") or "").strip()
            trigger_answer = str(trigger.get("standard_answer_text", "") or "").strip()
            if anchor and trigger_answer != anchor:
                errors.append(
                    f"hierarchical parent {parent_id} changed its full-credit answer"
                )
            fallback_points = sum(
                float(item.get("points", 0) or 0)
                for item in scoring_children
                if not item.get("full_credit_trigger")
            )
            fallback_cap = float(spec.get("fallback_cap", 0) or 0)
            if fallback_cap <= tolerance:
                errors.append(f"hierarchical parent {parent_id} has no fallback_cap")
            elif abs(fallback_points - fallback_cap) > tolerance:
                errors.append(
                    f"hierarchical parent {parent_id} fallback points changed: "
                    f"{fallback_points} != {fallback_cap}"
                )
            continue

        if spec["split_policy"] != "preserve_atomic":
            continue
        if len(scoring_children) != 1:
            errors.append(
                f"atomic parent {parent_id} must remain one scoring item, got {len(scoring_children)}"
            )
            continue
        child = scoring_children[0]
        original = spec["items"][0]
        if str(child.get("item", "") or "").strip() != str(original.get("item", "") or "").strip():
            errors.append(f"atomic parent {parent_id} changed its scoring meaning")
        original_answer = str(original.get("standard_answer_text", "") or "").strip()
        child_answer = str(child.get("standard_answer_text", "") or "").strip()
        if original_answer and child_answer != original_answer:
            errors.append(f"atomic parent {parent_id} changed its full-credit answer")

    return not errors, errors


def apply_hierarchical_scoring_policy(
    grading_result: dict[str, Any],
    rubric: list[dict[str, Any]],
    canonical_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply final-answer-full-credit and partial-process fallback deterministically.

    Hierarchical rubric children still sum to their parent score, so the rest of
    the grading pipeline can remain item based. A canonical match on the sole
    ``full_credit_trigger`` grants the complete parent score. Otherwise only
    non-trigger children contribute, capped by the declared fallback allowance.
    """
    if not isinstance(grading_result, dict) or not isinstance(rubric, list):
        return grading_result
    details = grading_result.get("details")
    if not isinstance(details, list):
        return grading_result

    prepared = prepare_rubric_semantic_contract(rubric)
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in prepared:
        if item.get("scoring_policy") != SCORING_POLICY_HIERARCHICAL:
            continue
        parent_id = str(item.get("parent_id") or item.get("id", ""))
        groups.setdefault(parent_id, []).append(item)
    if not groups:
        return grading_result

    canonical_comparisons = {
        str(item.get("id", "")): (item.get("comparison") or {})
        for item in (canonical_context or {}).get("items", [])
        if isinstance(item, dict)
    }
    details_by_id = {
        str(detail.get("id", "")): detail
        for detail in details
        if isinstance(detail, dict)
    }

    applied = []
    for parent_id, items in groups.items():
        triggers = [item for item in items if item.get("full_credit_trigger")]
        if len(triggers) != 1:
            continue
        trigger = triggers[0]
        trigger_id = str(trigger.get("id", ""))
        trigger_detail = details_by_id.get(trigger_id, {})
        canonical_comparison = canonical_comparisons.get(trigger_id, {})
        canonical_match = canonical_comparison.get("match")
        canonical_status = str(canonical_comparison.get("status", ""))
        if canonical_match is None and canonical_status != "student_blank":
            trigger_match = str(trigger_detail.get("error_category", "")).upper() == "MATCH"
            trigger_match_source = "semantic_grader"
        else:
            # A deterministic mismatch or blank must not be overturned by a model label.
            trigger_match = canonical_match is True
            trigger_match_source = "canonicalizer"

        parent_points = float(trigger.get("parent_points", 0) or 0)
        fallback_cap = float(trigger.get("fallback_cap", 0) or 0)
        group_ids = {str(item.get("id", "")) for item in items}
        previous_group_total = 0.0
        for detail in details:
            if not isinstance(detail, dict) or str(detail.get("id", "")) not in group_ids:
                continue
            try:
                previous_group_total += float(detail.get("score_given", 0) or 0)
            except (TypeError, ValueError):
                pass

        if trigger_match:
            effective_score = parent_points
            for item in items:
                item_id = str(item.get("id", ""))
                detail = details_by_id.get(item_id)
                if detail is None:
                    detail = {"id": item_id}
                    details.append(detail)
                    details_by_id[item_id] = detail
                detail["hierarchical_full_credit"] = True
                if item_id == trigger_id:
                    detail["score_given"] = float(item.get("points", 0) or 0)
                    detail["error_category"] = "MATCH"
                    detail["hierarchical_parent_score"] = parent_points
                else:
                    # Preserve the observed process evidence. The final answer
                    # waives this requirement; it does not prove the process was written.
                    detail["dependency_status"] = "not_required_due_to_final_answer"
                    detail["credit_requirement"] = "waived_by_final_answer"
                    detail["reason"] = (
                        "最终答案满足该父项的充分满分条件；"
                        "过程项不作为满分前置要求。"
                    )
            mode = "final_answer_full_credit"
        else:
            if trigger_detail:
                trigger_detail["score_given"] = 0.0
                trigger_detail["hierarchical_full_credit"] = False
            support_total = 0.0
            for item in items:
                if item.get("full_credit_trigger"):
                    continue
                item_id = str(item.get("id", ""))
                detail = details_by_id.get(item_id, {})
                try:
                    awarded = float(detail.get("score_given", 0) or 0)
                except (TypeError, ValueError):
                    awarded = 0.0
                support_total += min(
                    max(awarded, 0.0),
                    float(item.get("points", 0) or 0),
                )
            effective_score = min(support_total, fallback_cap)
            mode = "partial_process_fallback"

        applied.append(
            {
                "parent_id": parent_id,
                "mode": mode,
                "trigger_match_source": trigger_match_source,
                "previous_score": round(previous_group_total, 6),
                "effective_score": round(effective_score, 6),
                "parent_points": parent_points,
                "fallback_cap": fallback_cap,
            }
        )

    if not applied:
        return grading_result

    hierarchical_ids = {
        str(item.get("id", ""))
        for items in groups.values()
        for item in items
    }
    non_hierarchical_total = 0.0
    for detail in details:
        if not isinstance(detail, dict) or str(detail.get("id", "")) in hierarchical_ids:
            continue
        try:
            non_hierarchical_total += float(detail.get("score_given", 0) or 0)
        except (TypeError, ValueError):
            pass
    grading_result["total_score"] = round(
        non_hierarchical_total + sum(item["effective_score"] for item in applied),
        6,
    )
    grading_result["hierarchical_scoring"] = applied
    return grading_result


def apply_role_weighted_scoring_policy(
    grading_result: dict[str, Any],
    rubric: list[dict[str, Any]],
) -> dict[str, Any]:
    """Enforce role weights and optional evidence dependencies deterministically.

    The semantic grader may award partial credit inside each child, but it may
    not exceed the validated child weight.  A final conclusion remains an
    independently scorable, low-weight item unless the official question
    explicitly declares ``dependency_mode=evidence_required``.
    """
    if not isinstance(grading_result, dict) or not isinstance(rubric, list):
        return grading_result
    details = grading_result.get("details")
    if not isinstance(details, list):
        return grading_result

    prepared = prepare_rubric_semantic_contract(rubric)
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in prepared:
        if item.get("scoring_policy") != SCORING_POLICY_ROLE_WEIGHTED:
            continue
        parent_id = str(item.get("parent_id") or item.get("id", ""))
        groups.setdefault(parent_id, []).append(item)
    if not groups:
        return grading_result

    details_by_id = {
        str(detail.get("id", "")): detail
        for detail in details
        if isinstance(detail, dict)
    }
    applied = []
    role_weighted_ids: set[str] = set()

    for parent_id, items in groups.items():
        parent_total = 0.0
        child_scores: dict[str, float] = {}
        for item in items:
            item_id = str(item.get("id", ""))
            role_weighted_ids.add(item_id)
            detail = details_by_id.get(item_id)
            if detail is None:
                detail = {"id": item_id, "score_given": 0.0}
                details.append(detail)
                details_by_id[item_id] = detail
            try:
                raw_score = float(detail.get("score_given", 0) or 0)
            except (TypeError, ValueError):
                raw_score = 0.0
            item_points = float(item.get("points", 0) or 0)
            score = min(max(raw_score, 0.0), item_points)
            detail["score_given"] = score
            detail["scoring_role"] = item.get("scoring_role")
            child_scores[item_id] = score

        dependency_mode = str(
            items[0].get("dependency_mode", "independent")
        ).strip().lower()
        dependency_applied = False
        if dependency_mode == "evidence_required":
            core_evidence = any(
                child_scores.get(str(item.get("id", "")), 0.0) > 0
                for item in items
                if item.get("scoring_role") == SCORING_ROLE_CORE
            )
            if not core_evidence:
                for item in items:
                    if item.get("scoring_role") != SCORING_ROLE_FINAL:
                        continue
                    item_id = str(item.get("id", ""))
                    child_scores[item_id] = 0.0
                    detail = details_by_id[item_id]
                    detail["score_given"] = 0.0
                    detail["dependency_status"] = "blocked_without_core_evidence"
                dependency_applied = True

        parent_total = sum(child_scores.values())
        applied.append(
            {
                "parent_id": parent_id,
                "effective_score": round(parent_total, 6),
                "parent_points": float(items[0].get("parent_points", 0) or 0),
                "process_complexity": items[0].get("process_complexity"),
                "dependency_mode": dependency_mode,
                "dependency_applied": dependency_applied,
            }
        )

    non_role_total = 0.0
    for detail in details:
        if not isinstance(detail, dict):
            continue
        if str(detail.get("id", "")) in role_weighted_ids:
            continue
        try:
            non_role_total += float(detail.get("score_given", 0) or 0)
        except (TypeError, ValueError):
            pass
    grading_result["total_score"] = round(
        non_role_total + sum(item["effective_score"] for item in applied),
        6,
    )
    grading_result["role_weighted_scoring"] = applied
    return grading_result


def project_rubric_for_risk(
    rubric: list[dict[str, Any]],
    strict_cots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project hierarchical parents onto evidence that is required for routing.

    When a strict majority of probes grants a hierarchical parent full credit
    from its final trigger, omitted support steps are optional rather than
    extraction failures. The trigger represents the complete parent score for
    risk weighting. Without majority agreement, the original child structure
    remains unchanged so missing process evidence can still raise risk.
    """
    prepared = prepare_rubric_semantic_contract(rubric)
    if len(strict_cots) < 2:
        return prepared

    full_credit_counts: dict[str, int] = {}
    for cot in strict_cots:
        if not isinstance(cot, dict):
            continue
        seen_parents = set()
        for audit in cot.get("hierarchical_scoring", []):
            if not isinstance(audit, dict):
                continue
            parent_id = str(audit.get("parent_id", ""))
            if (
                parent_id
                and audit.get("mode") == "final_answer_full_credit"
                and parent_id not in seen_parents
            ):
                full_credit_counts[parent_id] = full_credit_counts.get(parent_id, 0) + 1
                seen_parents.add(parent_id)

    required_votes = len(strict_cots) // 2 + 1
    full_credit_parents = {
        parent_id
        for parent_id, count in full_credit_counts.items()
        if count >= required_votes
    }
    if not full_credit_parents:
        return prepared

    projected = []
    for item in prepared:
        parent_id = str(item.get("parent_id") or item.get("id", ""))
        if parent_id not in full_credit_parents:
            projected.append(item)
            continue
        if not item.get("full_credit_trigger"):
            continue
        trigger = deepcopy(item)
        trigger["points"] = float(trigger.get("parent_points", 0) or 0)
        trigger["score_layer"] = "core"
        trigger["role"] = "final"
        trigger["risk_projection"] = "hierarchical_final_trigger"
        projected.append(trigger)
    return projected


def has_deterministic_hierarchical_full_credit(
    rubric: list[dict[str, Any]],
    strict_cots: list[dict[str, Any]],
) -> bool:
    """Return whether deterministic final-answer rules cover the full rubric.

    This is deliberately stricter than the risk projection. Every positive-point
    rubric item must belong to a hierarchical parent, and every parent must have
    a strict-majority full-credit decision sourced from a canonicalizer. The
    minimum of two successful probes prevents a single model call from creating
    a score lock.
    """
    prepared = prepare_rubric_semantic_contract(rubric)
    successful_probes = [cot for cot in strict_cots if isinstance(cot, dict)]
    if len(successful_probes) < 2:
        return False

    hierarchical_parents = {
        str(item.get("parent_id") or item.get("id", ""))
        for item in prepared
        if float(item.get("points", 0) or 0) > 0
        and item.get("scoring_policy") == SCORING_POLICY_HIERARCHICAL
    }
    if not hierarchical_parents:
        return False
    if any(
        float(item.get("points", 0) or 0) > 0
        and item.get("scoring_policy") != SCORING_POLICY_HIERARCHICAL
        for item in prepared
    ):
        return False

    canonical_full_counts = {parent_id: 0 for parent_id in hierarchical_parents}
    for cot in successful_probes:
        seen_parents = set()
        for audit in cot.get("hierarchical_scoring", []):
            if not isinstance(audit, dict):
                continue
            parent_id = str(audit.get("parent_id", ""))
            if (
                parent_id in canonical_full_counts
                and parent_id not in seen_parents
                and audit.get("mode") == "final_answer_full_credit"
                and audit.get("trigger_match_source") == "canonicalizer"
            ):
                canonical_full_counts[parent_id] += 1
                seen_parents.add(parent_id)

    required_votes = len(successful_probes) // 2 + 1
    return all(
        canonical_full_counts[parent_id] >= required_votes
        for parent_id in hierarchical_parents
    )
