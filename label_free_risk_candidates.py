"""Experimental, label-free risk alternatives for offline comparisons.

Not connected to grading or evidence extraction. Inputs must already be audited;
in particular, a model's 'missing' status does not establish a blank answer.
"""
from math import isfinite

from label_free_a3wa import equal_weight_membership


def _validated_limits(item_limits):
    if not isinstance(item_limits, dict) or not item_limits:
        raise ValueError("Non-empty item limits are required")
    limits = {}
    for item, limit in item_limits.items():
        if not isinstance(item, str) or not item or isinstance(limit, bool):
            raise ValueError("Invalid item or limit")
        try:
            value = float(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid item limit") from exc
        if not isfinite(value) or value <= 0:
            raise ValueError("Item limits must be positive and finite")
        limits[item] = value
    return limits


def evidence_availability_risk(evidence, item_limits):
    """Compute an experimental risk from explicit, auditable evidence states.

    A model's raw missing/unclear is not accepted as verified absence. The
    caller must establish the check and record its source and description.
    A low evidence risk alone is never authorization to issue a grade.
    """
    limits = _validated_limits(item_limits)
    if not isinstance(evidence, dict) or set(evidence) != set(limits):
        raise ValueError("Evidence must cover exactly the rubric items")
    allowed = {"readable", "verified_absent", "unreadable", "unlocated", "conflict"}
    unresolved = {"unreadable", "unlocated", "conflict"}
    risk = 0.0
    for item, limit in limits.items():
        entry = evidence[item]
        if not isinstance(entry, dict) or set(entry) != {"state", "source", "description"}:
            raise ValueError("Each evidence item needs state, source and description")
        state = entry["state"]
        if not isinstance(state, str) or state not in allowed:
            raise ValueError("Unrecognized or unverified evidence state")
        for key in ("source", "description"):
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise ValueError("Evidence provenance and description are required")
        if state in unresolved:
            risk += limit
    return risk / sum(limits.values())


def item_score_range_risk(probes, item_limits):
    """Sum per-item score ranges divided by the sum of item limits."""
    limits = _validated_limits(item_limits)
    if not isinstance(probes, list) or len(probes) < 2:
        raise ValueError("At least two probes are required")
    scores = {item: [] for item in limits}
    for probe in probes:
        if not isinstance(probe, dict) or set(probe) != set(limits):
            raise ValueError("Each probe must contain exactly the rubric items")
        for item, limit in limits.items():
            entry = probe[item]
            if not isinstance(entry, dict) or "score" not in entry or isinstance(entry["score"], bool):
                raise ValueError("Missing or invalid item score")
            try:
                score = float(entry["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid item score") from exc
            if not isfinite(score) or not 0 <= score <= limit:
                raise ValueError("Item score outside rubric limits")
            scores[item].append(score)
    return sum(max(values) - min(values) for values in scores.values()) / sum(limits.values())


def candidate_memberships(risks, *, valid_probe_count):
    """Compare existing equal mean with 1-max; neither is correctness probability."""
    mean = equal_weight_membership(risks, valid_probe_count=valid_probe_count)
    return {"equal_mean": mean, "noncompensatory_max": 1.0 - max(float(v) for v in risks.values())}
