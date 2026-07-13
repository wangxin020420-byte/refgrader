"""Semantic contracts and validation for rubric refinement."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


RUBRIC_SEMANTIC_CONTRACT_VERSION = 2

SCORING_POLICY_STRICT_ATOMIC = "strict_atomic"
SCORING_POLICY_ADDITIVE = "additive_split"
SCORING_POLICY_HIERARCHICAL = "final_sufficient_partial_credit"
SUPPORTED_SCORING_POLICIES = {
    SCORING_POLICY_STRICT_ATOMIC,
    SCORING_POLICY_ADDITIVE,
    SCORING_POLICY_HIERARCHICAL,
}


ATOMIC_OUTCOME_PATTERNS = (
    r"最终结果",
    r"正确计算出",
    r"得出.*结果",
    r"写出.*(?:排列顺序|屏蔽字)",
    r"明确指出.*位数",
    r"得出正确的.*格式",
    r"操作数内容",
    r"运行时间",
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


def _item_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key, "") or "")
        for key in (
            "item",
            "source_text",
            "parent_official_item",
        )
    )


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
        scoring_policy = str(item.get("scoring_policy", "")).strip().lower()
        if scoring_policy not in SUPPORTED_SCORING_POLICIES:
            scoring_policy = (
                SCORING_POLICY_STRICT_ATOMIC
                if is_atomic_outcome_item(item)
                else SCORING_POLICY_ADDITIVE
            )
        item["scoring_policy"] = scoring_policy

        if scoring_policy == SCORING_POLICY_STRICT_ATOMIC:
            item["split_policy"] = "preserve_atomic"
            item.setdefault("weighting_policy", "preserve_parent")
        else:
            item["split_policy"] = "allow_semantic_split"
            item.setdefault(
                "weighting_policy",
                "preserve_parent"
                if scoring_policy == SCORING_POLICY_HIERARCHICAL
                else "equal_atomic",
            )

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
        contracted.append(item)
    return contracted


def validate_refined_rubric(
    original_rubric: list[dict[str, Any]],
    refined_rubric: list[dict[str, Any]],
    total_score: float,
    tolerance: float = 1e-6,
) -> tuple[bool, list[str]]:
    """Validate score conservation, traceability, and atomic full-credit anchors."""
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
            "full_credit_anchor": first.get("full_credit_anchor", ""),
            "fallback_cap": float(first.get("fallback_cap", 0) or 0),
            "items": items,
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
