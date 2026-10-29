"""Link BND evidence checks to a conservative, non-authorizing projection.

Model-extracted text is not an independent check of image fidelity or rubric
relevance. All constraints stay unknown until a future audited verifier exists.
"""
from label_free_boundary import assess_boundary_evidence
from label_free_evidence_projection import propose_evidence_projection


def diagnose_bnd(*, decision, observation, rubric):
    if not isinstance(decision, dict) or decision.get("route") != "BND":
        raise ValueError("BND decision required")
    if not isinstance(rubric, list) or not rubric:
        raise ValueError("Nonempty rubric required")
    limits = {}
    for item in rubric:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError("Invalid rubric item")
        if item["id"] in limits:
            raise ValueError("Duplicate rubric item")
        limits[item["id"]] = item.get("points")
    if not isinstance(observation, dict) or not isinstance(observation.get("facts"), dict):
        raise ValueError("Evidence facts are required")
    facts = observation["facts"]
    if set(facts) != set(limits):
        raise ValueError("Evidence/rubric item mismatch")
    checks = []
    for item_id, fact in facts.items():
        if not isinstance(fact, dict) or set(fact) != {"status", "evidence"}:
            raise ValueError("Invalid evidence fact")
        if fact["status"] not in {"present", "missing", "unclear"} or not isinstance(fact["evidence"], str):
            raise ValueError("Invalid evidence status or text")
        if fact["status"] == "present":
            if not fact["evidence"].strip():
                raise ValueError("Present evidence must not be empty")
            checked = assess_boundary_evidence(answer_text=fact["evidence"], rubric=rubric,
                claim={"item_id": item_id, "start": 0, "end": len(fact["evidence"])})
            checks.append({"source": "model_extracted_item_text", **checked})
        else:
            checks.append({"item_id": item_id, "source": "model_extraction", "status": fact["status"],
                           "canonical_match": None, "criterion_relevance_verified": False})
    constraints = {item_id: {"state": "unknown"} for item_id in limits}
    projection = propose_evidence_projection(core_score=decision["provisional_score"],
        item_limits=limits, constraints=constraints)
    return {"scope": "diagnostic_only", "checks": checks, "projection": projection,
            "score_change_authorized": False, "requires_human_review": True}
