"""Development-only score interval projection; never releases or changes a grade.

The caller must independently establish evidence provenance and the rubric
meaning. Numeric intervals are assumptions, not teacher-score residuals or
statistical confidence intervals. Unknown items have the full [0, limit] range.
"""
from math import isfinite


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def propose_evidence_projection(*, core_score, item_limits, constraints):
    """Return a diagnostic candidate; never authorize score changes or POS release.

    Each item has a required state. A 'supported' interval requires a rubric
    rule reference and evidence reference. 'unknown' has no narrower bounds.
    Reference strings do not themselves verify either source.
    """
    if not isinstance(item_limits, dict) or not item_limits:
        raise ValueError("Nonempty item limits are required")
    limits = {}
    for item, value in item_limits.items():
        if not isinstance(item, str) or not item:
            raise ValueError("Invalid item ID")
        limit = _number(value, "item limit")
        if limit <= 0:
            raise ValueError("Item limits must be positive")
        limits[item] = limit
    if not isinstance(constraints, dict) or set(constraints) != set(limits):
        raise ValueError("Constraints must cover exactly the rubric items")
    core = _number(core_score, "core score")
    total = sum(limits.values())
    if not isfinite(total) or not 0 <= core <= total:
        raise ValueError("Core score must lie within the rubric total")
    lower = upper = 0.0
    unresolved = []
    for item, cap in limits.items():
        record = constraints[item]
        if not isinstance(record, dict):
            raise ValueError("Invalid constraint")
        if record == {"state": "unknown"}:
            lo, hi = 0.0, cap
            unresolved.append(item)
        elif set(record) == {"state", "lower", "upper", "rubric_ref", "evidence_ref"} and record["state"] == "supported":
            lo, hi = _number(record["lower"], "lower"), _number(record["upper"], "upper")
            if not 0 <= lo <= hi <= cap:
                raise ValueError("Constraint outside item limits or inconsistent")
            for key in ("rubric_ref", "evidence_ref"):
                if not isinstance(record[key], str) or not record[key].strip():
                    raise ValueError("Rule and evidence references are required")
        else:
            raise ValueError("Unsupported or malformed constraint")
        lower += lo
        upper += hi
    proposal = min(upper, max(lower, core))
    return {"core_score": core, "score_interval": {"lower": lower, "upper": upper},
            "proposed_score": proposal, "proposed_delta": proposal - core,
            "unresolved_items": sorted(unresolved), "score_change_authorized": False,
            "requires_human_review": True, "scope": "diagnostic_only"}
