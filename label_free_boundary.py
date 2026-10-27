"""Traceable, teacher-score-free evidence checks for BND diagnostics.

Exact location and canonical equality do not establish that the excerpt is the
correct criterion evidence or justify a score change. This check never changes
scores; the text, rubric and their provenance still require independent audit.
"""
from canonicalizers import build_canonical_grading_context


def assess_boundary_evidence(*, answer_text, rubric, claim):
    if not isinstance(answer_text, str) or not isinstance(rubric, list):
        raise ValueError("Answer text and rubric are required")
    if not isinstance(claim, dict) or set(claim) != {"item_id", "start", "end"}:
        raise ValueError("Claim must contain exactly item_id, start and end")
    item_id, start, end = claim["item_id"], claim["start"], claim["end"]
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("Invalid rubric item ID")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)):
        raise ValueError("Evidence offsets must be integers")
    if not 0 <= start < end <= len(answer_text):
        raise ValueError("Evidence offsets are outside the answer")
    excerpt = answer_text[start:end]
    if not excerpt.strip():
        raise ValueError("Evidence excerpt must not be blank")
    items = [item for item in rubric if isinstance(item, dict) and item.get("id") == item_id]
    if len(items) != 1:
        raise ValueError("Evidence must reference exactly one rubric item")
    canonical = build_canonical_grading_context({item_id: excerpt}, items)
    comparison = canonical["items"][0]["comparison"]
    matched = comparison.get("match")
    if matched is not True and matched is not False:
        matched = None
    return {
        "item_id": item_id,
        "span": {"start": start, "end": end},
        "span_within_answer": True,
        "canonical_status": comparison.get("status", "unknown"),
        "canonical_match": matched,
        "criterion_relevance_verified": False,
        "input_content_audited": False,
        "score_change_authorized": False,
        "requires_human_review": True,
    }
