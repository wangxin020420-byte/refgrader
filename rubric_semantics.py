"""Semantic contracts and validation for rubric refinement."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


RUBRIC_SEMANTIC_CONTRACT_VERSION = 1


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
        item.setdefault("semantic_contract_version", RUBRIC_SEMANTIC_CONTRACT_VERSION)
        item.setdefault("parent_points", float(item.get("points", 0) or 0))
        item.setdefault(
            "split_policy",
            "preserve_atomic"
            if is_atomic_outcome_item(item)
            else "allow_semantic_split",
        )
        item.setdefault(
            "weighting_policy",
            "preserve_parent"
            if item["split_policy"] == "preserve_atomic"
            else "equal_atomic",
        )
        if item["split_policy"] == "preserve_atomic":
            item.setdefault("full_credit_anchor", item.get("standard_answer_text", ""))
        contracted.append(item)
    return contracted


def _resolve_parent_id(
    item: dict[str, Any],
    originals: dict[str, dict[str, Any]],
) -> str:
    parent_id = str(item.get("parent_id", "") or "")
    if parent_id:
        return parent_id
    item_id = str(item.get("id", "") or "")
    if item_id in originals:
        return item_id
    parent_text = str(item.get("parent_official_item", "") or "")
    for original_id, original in originals.items():
        if parent_text and parent_text in {
            str(original.get("item", "") or ""),
            str(original.get("parent_official_item", "") or ""),
        }:
            return original_id
    return ""


def validate_refined_rubric(
    original_rubric: list[dict[str, Any]],
    refined_rubric: list[dict[str, Any]],
    total_score: float,
    tolerance: float = 1e-6,
) -> tuple[bool, list[str]]:
    """Validate score conservation, traceability, and atomic full-credit anchors."""
    originals_list = prepare_rubric_semantic_contract(original_rubric)
    originals = {str(item["id"]): item for item in originals_list}
    errors: list[str] = []
    seen_ids: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in originals}

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

        parent_id = _resolve_parent_id(item, originals)
        if parent_id not in originals:
            errors.append(
                f"refined item {item_id or '<missing>'} has unknown parent_id {parent_id!r}"
            )
            continue
        item["parent_id"] = parent_id
        item["parent_points"] = float(originals[parent_id].get("points", 0) or 0)
        item["split_policy"] = originals[parent_id].get(
            "split_policy", "allow_semantic_split"
        )
        item["weighting_policy"] = originals[parent_id].get(
            "weighting_policy", "equal_atomic"
        )
        grouped[parent_id].append(item)

    refined_total = sum(float(item.get("points", 0) or 0) for item in refined_rubric)
    if abs(refined_total - float(total_score)) > tolerance:
        errors.append(f"total score changed: {refined_total} != {total_score}")

    for parent_id, original in originals.items():
        children = grouped[parent_id]
        child_total = sum(float(item.get("points", 0) or 0) for item in children)
        original_points = float(original.get("points", 0) or 0)
        if abs(child_total - original_points) > tolerance:
            errors.append(
                f"parent {parent_id} score changed: {child_total} != {original_points}"
            )

        scoring_children = [
            item for item in children if float(item.get("points", 0) or 0) > tolerance
        ]
        if (
            original.get("weighting_policy") == "equal_atomic"
            and len(scoring_children) > 1
        ):
            child_points = [float(item.get("points", 0) or 0) for item in scoring_children]
            if max(child_points) - min(child_points) > tolerance:
                errors.append(f"parent {parent_id} must use equal atomic weights")

        if original.get("split_policy") != "preserve_atomic":
            continue
        if len(scoring_children) != 1:
            errors.append(
                f"atomic parent {parent_id} must remain one scoring item, got {len(scoring_children)}"
            )
            continue
        child = scoring_children[0]
        if str(child.get("item", "") or "").strip() != str(
            original.get("item", "") or ""
        ).strip():
            errors.append(f"atomic parent {parent_id} changed its scoring meaning")
        original_answer = str(original.get("standard_answer_text", "") or "").strip()
        child_answer = str(child.get("standard_answer_text", "") or "").strip()
        if original_answer and child_answer != original_answer:
            errors.append(f"atomic parent {parent_id} changed its full-credit answer")

    return not errors, errors
