"""Conservative actions for A3WA label-free routing.

This is an isolated policy baseline. It is not wired to model scoring, and it
does not certify the origin or accuracy of risk inputs. BND always defers until
an independently tested evidence-verification gate is available.
"""
from math import isfinite

from label_free_a3wa import equal_weight_membership, route_membership
from label_free_risk_candidates import candidate_memberships


def decide_action(*, baseline_score, max_score, risks, valid_probe_count, m, loss_parameters, fusion="equal_mean"):
    """Accept POS; retain provisional scores for BND and NEG human review."""
    if isinstance(baseline_score, bool) or isinstance(max_score, bool):
        raise ValueError("Scores must be finite numbers")
    try:
        baseline_score = float(baseline_score)
        max_score = float(max_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("Scores must be finite numbers") from exc
    if not isfinite(max_score) or max_score <= 0 or not isfinite(baseline_score):
        raise ValueError("Scores must be finite and max_score must be positive")
    if not 0 <= baseline_score <= max_score:
        raise ValueError("Baseline score outside allowed rubric range")
    if fusion == "equal_mean":
        membership = equal_weight_membership(risks, valid_probe_count=valid_probe_count)
    elif fusion == "noncompensatory_max":
        membership = candidate_memberships(risks, valid_probe_count=valid_probe_count)[fusion]
    else:
        raise ValueError("Unsupported risk fusion")
    decision = route_membership(membership, m=m, loss_parameters=loss_parameters)
    auto = decision["route"] == "POS"
    return {
        **decision,
        "provisional_score": baseline_score,
        "automatic_score": baseline_score if auto else None,
        "requires_human_review": not auto,
        "action": "accept_baseline" if auto else "defer_human",
        "policy": "conservative_bnd_defer_v1" if fusion == "equal_mean" else "candidate_max_bnd_defer_v1",
    }
