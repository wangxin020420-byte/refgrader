"""Canonical answer normalization layer.

This module converts heterogeneous student/reference expressions into typed,
comparable structures before semantic grading. It is intentionally conservative:
when a value cannot be normalized safely, it returns an ``unknown`` comparison
rather than forcing a score.
"""

from __future__ import annotations

import json
import re
from typing import Any


BLANK_MARKERS = (
    "未书写",
    "字迹模糊",
    "需要查看图像",
    "图中未明确显示",
    "表格结构不清晰",
)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _is_blankish(text: str) -> bool:
    stripped = _as_text(text).strip()
    return not stripped or any(marker in stripped for marker in BLANK_MARKERS)


def _parse_json_maybe(payload: Any) -> Any:
    if not isinstance(payload, str):
        return payload
    try:
        return json.loads(payload)
    except Exception:
        return payload


def _rubric_type(item: dict[str, Any]) -> str:
    text = " ".join(
        _as_text(item.get(key, ""))
        for key in ("answer_type", "canonicalization", "evidence_source", "item")
    ).lower()
    if "structured_fields" in text or "structured fields" in text:
        return "structured_fields"
    if "bit_vector" in text or "bit vector" in text:
        return "bit_vector"
    if "sequence" in text or "顺序" in text or "序列" in text:
        return "sequence"
    if "relation" in text or "diagram" in text or "graph" in text or "关系" in text:
        return "relation"
    if "table" in text or "grid" in text or "matrix" in text or "表" in text:
        return "table_grid"
    return _as_text(item.get("answer_type", "text")).strip() or "text"


def _extract_universe(item: dict[str, Any], standard_text: str, student_text: str) -> list[str]:
    canonicalization = item.get("canonicalization")
    if isinstance(canonicalization, dict):
        labels = canonicalization.get("label_universe") or canonicalization.get("labels")
        if isinstance(labels, list) and labels:
            return [str(label).strip().upper() for label in labels if str(label).strip()]

    text = " ".join(
        [
            _as_text(item.get("item", "")),
            _as_text(item.get("source_text", "")),
            _as_text(canonicalization),
            standard_text,
            student_text,
        ]
    )
    explicit = re.search(r"\b([A-Z](?:\s*[,/>→\-]\s*[A-Z]){2,})\b", text.upper())
    if explicit:
        labels = re.findall(r"[A-Z]", explicit.group(1))
        if 2 <= len(labels) <= 12:
            deduped = []
            for label in labels:
                if label not in deduped:
                    deduped.append(label)
            return deduped

    binary_match = re.search(r"\b[01]{2,16}\b", standard_text)
    if binary_match:
        length = len(binary_match.group(0))
        if length <= 26:
            return [chr(ord("A") + index) for index in range(length)]

    labels = sorted(set(re.findall(r"\b[A-Z]\b", text.upper())))
    if 2 <= len(labels) <= 12:
        return labels
    return []


def _extract_item_label(item: dict[str, Any], standard_text: str, student_text: str) -> str | None:
    joined = " ".join(
        [
            _as_text(item.get("item", "")),
            _as_text(item.get("source_text", "")),
            standard_text,
            student_text,
        ]
    ).upper()
    colon_label = re.search(r"\b([A-Z])\s*[:：]", joined)
    if colon_label:
        return colon_label.group(1)
    level_label = re.search(r"\b([A-Z])\s*(?:级|LEVEL|INTERRUPT)", joined)
    if level_label:
        return level_label.group(1)
    return None


def _normalize_bit_vector(
    value: Any,
    *,
    item: dict[str, Any],
    standard_bits: str | None = None,
    universe: list[str] | None = None,
    item_label: str | None = None,
) -> dict[str, Any]:
    text = _as_text(value).strip().upper()
    if _is_blankish(text):
        return {"type": "bit_vector", "status": "blank", "raw": _as_text(value)}

    universe = universe or _extract_universe(item, _as_text(item.get("standard_answer_text", "")), text)
    if not universe and standard_bits:
        universe = [chr(ord("A") + index) for index in range(len(standard_bits))]
    width = len(universe) if universe else (len(standard_bits) if standard_bits else 0)

    payload = text
    colon_value = re.search(r"[:：]\s*([A-Z0-9{},，、\s]+)", payload)
    if colon_value:
        payload = colon_value.group(1).strip()

    binary = re.search(r"\b[01]{2,32}\b", payload)
    if binary and (not width or len(binary.group(0)) == width):
        bits = binary.group(0)
        return {
            "type": "bit_vector",
            "status": "ok",
            "raw": _as_text(value),
            "bits": bits,
            "universe": universe,
            "source_form": "binary",
        }

    zero_literal = payload.strip() in {"0", "00", "NONE", "NULL", "无"}
    if universe and zero_literal:
        label_set = set()
        if (
            item_label
            and item_label in universe
            and standard_bits
            and standard_bits[universe.index(item_label)] == "1"
        ):
            label_set.add(item_label)
        bits = "".join("1" if label in label_set else "0" for label in universe)
        return {
            "type": "bit_vector",
            "status": "ok",
            "raw": _as_text(value),
            "bits": bits,
            "universe": universe,
            "source_form": "zero_or_empty_label_set",
            "labels": sorted(label_set),
        }

    hex_match = re.search(r"\b(?:0X)?([0-9A-F]+)H?\b", payload)
    if hex_match and re.search(r"[0-9]", hex_match.group(1)):
        number = int(hex_match.group(1), 16)
        bits = bin(number)[2:]
        if width:
            bits = bits.zfill(width)[-width:]
        return {
            "type": "bit_vector",
            "status": "ok",
            "raw": _as_text(value),
            "bits": bits,
            "universe": universe,
            "source_form": "hex_or_number",
        }

    if universe:
        label_set = set(re.findall(r"\b[A-Z]\b", payload))
        compact = re.sub(r"[^A-Z]", "", payload)
        if not label_set and 1 <= len(compact) <= len(universe):
            label_set = {char for char in compact if char in universe}
        if label_set:
            # Common shorthand in mask-vector answers: "B:0" means no other
            # label is listed, but the current row/level itself remains active.
            if (
                item_label
                and item_label in universe
                and standard_bits
                and standard_bits[universe.index(item_label)] == "1"
            ):
                label_set.add(item_label)
            bits = "".join("1" if label in label_set else "0" for label in universe)
            return {
                "type": "bit_vector",
                "status": "ok",
                "raw": _as_text(value),
                "bits": bits,
                "universe": universe,
                "source_form": "label_set",
                "labels": sorted(label_set),
            }

    return {"type": "bit_vector", "status": "unknown", "raw": _as_text(value)}


def _normalize_sequence(value: Any) -> dict[str, Any]:
    text = _as_text(value).strip()
    if _is_blankish(text):
        return {"type": "sequence", "status": "blank", "raw": _as_text(value)}

    upper = text.upper()
    # Prefer explicit arrow/comparison separated sequences.
    parts = re.split(r"(?:->|=>|→|>|≫|,|，|、|\s+)", upper)
    tokens = [part for part in parts if re.fullmatch(r"[A-Z][A-Z0-9_]*|[0-9]+", part)]
    if len(tokens) >= 2:
        return {"type": "sequence", "status": "ok", "raw": _as_text(value), "items": tokens}

    compact = re.sub(r"[^A-Z0-9]", "", upper)
    if 2 <= len(compact) <= 16 and compact.isalpha():
        return {"type": "sequence", "status": "ok", "raw": _as_text(value), "items": list(compact)}

    return {"type": "sequence", "status": "unknown", "raw": _as_text(value)}


def _normalize_relation(value: Any) -> dict[str, Any]:
    text = _as_text(value).strip()
    if _is_blankish(text):
        return {"type": "relation", "status": "blank", "raw": _as_text(value)}

    path = None
    path_match = re.search(r"(?:完整可见路径|observed_execution_path|path)[：:]\s*([A-Z0-9_\-?>→,，、\s]+)", text, re.I)
    if path_match:
        path = _normalize_sequence(path_match.group(1)).get("items")
    if not path:
        seq = _normalize_sequence(text)
        if seq.get("status") == "ok" and len(seq.get("items", [])) >= 3:
            path = seq.get("items")

    edges = []
    if path and len(path) >= 2:
        edges = [[path[index], path[index + 1]] for index in range(len(path) - 1)]

    return {
        "type": "relation",
        "status": "ok" if path or edges else "text_only",
        "raw": _as_text(value),
        "path": path,
        "edges": edges,
    }


def _normalize_table(value: Any) -> dict[str, Any]:
    parsed = _parse_json_maybe(value)
    if isinstance(parsed, dict) and "rows" in parsed:
        return {"type": "table_grid", "status": "ok", "raw": _as_text(value), "rows": parsed.get("rows")}
    text = _as_text(value).strip()
    if _is_blankish(text):
        return {"type": "table_grid", "status": "blank", "raw": _as_text(value)}
    rows = [
        [cell.strip() for cell in re.split(r"\s*\|\s*|\t+", line) if cell.strip()]
        for line in text.splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row]
    return {
        "type": "table_grid",
        "status": "ok" if rows else "text_only",
        "raw": _as_text(value),
        "rows": rows,
    }


def _structured_schema(item: dict[str, Any]) -> dict[str, Any]:
    canonicalization = item.get("canonicalization")
    if not isinstance(canonicalization, dict):
        return {}
    fields = canonicalization.get("fields")
    if not isinstance(fields, list) or not fields:
        return {}
    return {
        "ordered": bool(canonicalization.get("ordered", False)),
        "allow_positional": bool(canonicalization.get("allow_positional", False)),
        "fields": [field for field in fields if isinstance(field, dict)],
    }


def _normalize_structured_value(value: Any) -> str:
    text = _as_text(value).strip().upper()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?:位|BITS?|BIT)$", "", text, flags=re.I)
    return text.strip("。.;；,，:：")


def _normalize_structured_fields(
    value: Any,
    item: dict[str, Any],
) -> dict[str, Any]:
    schema = _structured_schema(item)
    text = _as_text(value).strip()
    if _is_blankish(text):
        return {"type": "structured_fields", "status": "blank", "raw": text}
    if not schema:
        return {
            "type": "structured_fields",
            "status": "unknown",
            "raw": text,
            "reason": "missing_structured_field_schema",
        }

    parsed = _parse_json_maybe(value)
    source_dict = parsed if isinstance(parsed, dict) else {}
    extracted: dict[str, str] = {}
    positions: dict[str, int] = {}
    raw_segments = [
        segment.strip()
        for segment in re.split(r"[|｜;；\n]+", text)
        if segment.strip()
    ]

    for field in schema["fields"]:
        name = _as_text(field.get("name", "")).strip()
        if not name:
            continue
        aliases = [name] + [
            _as_text(alias).strip()
            for alias in field.get("aliases", [])
            if _as_text(alias).strip()
        ]
        for key, raw in source_dict.items():
            if _as_text(key).strip().lower() in {alias.lower() for alias in aliases}:
                extracted[name] = _normalize_structured_value(raw)
                positions[name] = len(positions)
                break
        if name in extracted:
            continue
        for index, segment in enumerate(raw_segments):
            for alias in sorted(aliases, key=len, reverse=True):
                match = re.search(
                    rf"{re.escape(alias)}\s*(?:[:：=为是]\s*)?(.+)$",
                    segment,
                    re.I,
                )
                if match:
                    extracted[name] = _normalize_structured_value(match.group(1))
                    positions[name] = index
                    break
            if name in extracted:
                break

    if schema["allow_positional"] and raw_segments:
        missing = [
            field for field in schema["fields"]
            if _as_text(field.get("name", "")).strip() not in extracted
        ]
        if len(missing) == len(schema["fields"]) and len(raw_segments) == len(missing):
            for index, (field, segment) in enumerate(zip(missing, raw_segments)):
                name = _as_text(field.get("name", "")).strip()
                extracted[name] = _normalize_structured_value(segment)
                positions[name] = index

    return {
        "type": "structured_fields",
        "status": "ok" if extracted else "unknown",
        "raw": text,
        "fields": extracted,
        "positions": positions,
        "schema": schema,
    }


def _normalize_base_number(value: Any, item: dict[str, Any]) -> dict[str, Any]:
    text = _as_text(value).strip()
    if _is_blankish(text):
        return {"type": "base_number", "status": "blank", "raw": text}

    compact = re.sub(r"\s+", "", text).upper()
    probes = (
        (r"(?<![0-9A-F])(?:0X)?([0-9A-F]+)H(?![0-9A-Z])", 16, "hex"),
        (r"([01]+)₂", 2, "binary"),
        (r"(?<![0-9A-Z])([01]+)(?:_?2|B)(?![0-9A-Z])", 2, "binary"),
        (r"([0-9]+)₁₀", 10, "decimal"),
        (r"(?<![0-9A-Z])([0-9]+)(?:_?10|D)(?![0-9A-Z])", 10, "decimal"),
    )
    explicit_values = []
    explicit_tokens = []
    explicit_bases = []
    for pattern, base, label in probes:
        for match in re.finditer(pattern, compact):
            try:
                numeric_value = int(match.group(1), base)
            except ValueError:
                continue
            if numeric_value not in explicit_values:
                explicit_values.append(numeric_value)
            explicit_tokens.append(match.group(0))
            if label not in explicit_bases:
                explicit_bases.append(label)

    if explicit_values:
        if len(explicit_values) > 1:
            return {
                "type": "base_number",
                "status": "ambiguous",
                "raw": text,
                "values": explicit_values,
                "bases": explicit_bases,
                "tokens": explicit_tokens,
            }
        numeric_value = explicit_values[0]
        return {
            "type": "base_number",
            "status": "ok",
            "raw": text,
            "value": numeric_value,
            "values": [numeric_value],
            "base": explicit_bases[0],
            "bases": explicit_bases,
            "token": explicit_tokens[0],
            "tokens": explicit_tokens,
        }

    canonicalization = item.get("canonicalization")
    implicit_bases = []
    if isinstance(canonicalization, dict):
        raw_bases = canonicalization.get("implicit_bases") or []
        if isinstance(raw_bases, (int, str)):
            raw_bases = [raw_bases]
        for raw_base in raw_bases:
            try:
                base = int(raw_base)
            except (TypeError, ValueError):
                continue
            if base in {2, 8, 10, 16} and base not in implicit_bases:
                implicit_bases.append(base)
    bare = re.fullmatch(r"[0-9A-F]+", compact)
    if bare and implicit_bases:
        values = []
        for base in implicit_bases:
            try:
                candidate = int(bare.group(0), base)
            except ValueError:
                continue
            if candidate not in values:
                values.append(candidate)
        if values:
            return {
                "type": "base_number",
                "status": "ok",
                "raw": text,
                "value": values[0],
                "values": values,
                "base": "implicit",
                "implicit_bases": implicit_bases,
                "token": bare.group(0),
            }

    # A bare number without an explicit rubric policy is intentionally not
    # assigned a base. It remains a semantic grading decision.
    return {"type": "base_number", "status": "unknown", "raw": text}


def _normalize_by_type(value: Any, item: dict[str, Any], answer_type: str, *, standard_bits: str | None = None) -> dict[str, Any]:
    standard_text = _as_text(item.get("standard_answer_text", ""))
    student_text = _as_text(value)
    if answer_type == "bit_vector":
        universe = _extract_universe(item, standard_text, student_text)
        item_label = _extract_item_label(item, standard_text, student_text)
        return _normalize_bit_vector(
            value,
            item=item,
            standard_bits=standard_bits,
            universe=universe,
            item_label=item_label,
        )
    if answer_type == "structured_fields":
        return _normalize_structured_fields(value, item)
    if answer_type == "base_number":
        return _normalize_base_number(value, item)
    if answer_type == "sequence":
        return _normalize_sequence(value)
    if answer_type == "relation":
        return _normalize_relation(value)
    if answer_type == "table_grid":
        return _normalize_table(value)
    return {"type": answer_type or "text", "status": "raw", "raw": _as_text(value)}


def _compare_normalized(student_norm: dict[str, Any], standard_norm: dict[str, Any]) -> dict[str, Any]:
    st = student_norm.get("status")
    rt = standard_norm.get("status")
    if st in {"blank", "unknown"} or rt in {"blank", "unknown"}:
        return {
            "status": "unknown" if st != "blank" else "student_blank",
            "match": None,
            "reason": f"student_status={st}, standard_status={rt}",
        }
    if st == "ambiguous" or rt == "ambiguous":
        return {
            "status": "ambiguous",
            "match": None,
            "student_values": student_norm.get("values", []),
            "standard_values": standard_norm.get("values", []),
            "reason": f"student_status={st}, standard_status={rt}",
        }

    answer_type = standard_norm.get("type") or student_norm.get("type")
    if answer_type == "bit_vector":
        match = student_norm.get("bits") == standard_norm.get("bits")
        return {
            "status": "match" if match else "mismatch",
            "match": match,
            "student_bits": student_norm.get("bits"),
            "standard_bits": standard_norm.get("bits"),
        }
    if answer_type == "structured_fields":
        schema = standard_norm.get("schema") or student_norm.get("schema") or {}
        fields = schema.get("fields", [])
        student_fields = student_norm.get("fields", {})
        standard_fields = standard_norm.get("fields", {})
        required_names = [
            _as_text(field.get("name", "")).strip()
            for field in fields
            if field.get("required", True) and _as_text(field.get("name", "")).strip()
        ]
        matched = [
            name for name in required_names
            if name in student_fields
            and name in standard_fields
            and student_fields[name] == standard_fields[name]
        ]
        missing = [name for name in required_names if name not in student_fields]
        mismatched = [
            name for name in required_names
            if name in student_fields
            and name in standard_fields
            and student_fields[name] != standard_fields[name]
        ]
        order_match = True
        if schema.get("ordered"):
            observed = [
                name for name, _ in sorted(
                    student_norm.get("positions", {}).items(),
                    key=lambda pair: pair[1],
                )
                if name in required_names
            ]
            expected_order = [name for name in required_names if name in student_fields]
            order_match = observed == expected_order
        match = (
            bool(required_names)
            and len(matched) == len(required_names)
            and not missing
            and not mismatched
            and order_match
        )
        return {
            "status": "match" if match else "partial_or_mismatch",
            "match": match,
            "field_match_ratio": round(len(matched) / max(len(required_names), 1), 4),
            "matched_fields": matched,
            "missing_fields": missing,
            "mismatched_fields": mismatched,
            "order_match": order_match,
            "student_fields": student_fields,
            "standard_fields": standard_fields,
        }
    if answer_type == "base_number":
        student_values = set(student_norm.get("values") or [student_norm.get("value")])
        standard_values = set(standard_norm.get("values") or [standard_norm.get("value")])
        student_values.discard(None)
        standard_values.discard(None)
        match = bool(student_values & standard_values)
        return {
            "status": "match" if match else "mismatch",
            "match": match,
            "student_value": student_norm.get("value"),
            "standard_value": standard_norm.get("value"),
            "student_values": sorted(student_values),
            "standard_values": sorted(standard_values),
            "student_base": student_norm.get("base"),
            "standard_base": standard_norm.get("base"),
        }
    if answer_type == "sequence":
        student_items = student_norm.get("items")
        standard_items = standard_norm.get("items")
        match = student_items == standard_items
        common_edges_student = set(zip(student_items or [], (student_items or [])[1:]))
        common_edges_standard = set(zip(standard_items or [], (standard_items or [])[1:]))
        edge_overlap = len(common_edges_student & common_edges_standard)
        edge_total = max(len(common_edges_standard), 1)
        return {
            "status": "match" if match else "partial_or_mismatch",
            "match": match,
            "student_items": student_items,
            "standard_items": standard_items,
            "edge_overlap_ratio": round(edge_overlap / edge_total, 4),
        }
    if answer_type == "relation":
        student_edges = {tuple(edge) for edge in student_norm.get("edges", [])}
        standard_edges = {tuple(edge) for edge in standard_norm.get("edges", [])}
        if not standard_edges:
            return {"status": "reference_text_only", "match": None}
        overlap = len(student_edges & standard_edges)
        return {
            "status": "match" if overlap == len(standard_edges) else "partial_or_mismatch",
            "match": overlap == len(standard_edges),
            "edge_overlap_ratio": round(overlap / max(len(standard_edges), 1), 4),
            "student_edges": [list(edge) for edge in sorted(student_edges)],
            "standard_edges": [list(edge) for edge in sorted(standard_edges)],
        }
    return {"status": "not_comparable", "match": None}


def build_canonical_grading_context(student_facts: Any, rubrics_json: Any) -> dict[str, Any]:
    """Build a per-rubric canonical comparison report.

    The returned object is safe to pass into prompts and result JSON. It does
    not assign scores; it supplies deterministic equivalence evidence.
    """
    facts = _parse_json_maybe(student_facts)
    rubrics = _parse_json_maybe(rubrics_json)
    if not isinstance(facts, dict) or not isinstance(rubrics, list):
        return {"schema_version": 1, "available": False, "items": []}

    items = []
    summary = {"match": 0, "mismatch": 0, "unknown": 0, "not_comparable": 0}
    for item in rubrics:
        if not isinstance(item, dict):
            continue
        item_id = _as_text(item.get("id", "")).strip()
        if not item_id:
            continue
        answer_type = _rubric_type(item)
        raw_student = facts.get(item_id, "")
        raw_standard = item.get("standard_answer_text", "")

        standard_probe = _normalize_by_type(raw_standard, item, answer_type)
        standard_bits = standard_probe.get("bits") if answer_type == "bit_vector" else None
        student_norm = _normalize_by_type(
            raw_student,
            item,
            answer_type,
            standard_bits=standard_bits,
        )
        standard_norm = standard_probe
        comparison = _compare_normalized(student_norm, standard_norm)

        status = comparison.get("status")
        if comparison.get("match") is True:
            summary["match"] += 1
        elif comparison.get("match") is False:
            summary["mismatch"] += 1
        elif status in {"unknown", "student_blank"}:
            summary["unknown"] += 1
        else:
            summary["not_comparable"] += 1

        items.append(
            {
                "id": item_id,
                "answer_type": answer_type,
                "points": item.get("points"),
                "student_raw": _as_text(raw_student),
                "standard_raw": _as_text(raw_standard),
                "student_normalized": student_norm,
                "standard_normalized": standard_norm,
                "comparison": comparison,
            }
        )

    return {
        "schema_version": 1,
        "available": True,
        "summary": summary,
        "items": items,
    }
