"""Diagnostic comparison of explicitly transcribed ordered bit-width fields.

Both inputs are caller-provided observations, not independently verified
transcriptions. This module never decides rubric relevance, assigns points or
authorizes a score change. Missing subgroup details remain unknown.
"""


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate(record):
    keys = {"group_total", "subfields", "following_fields", "source_ref"}
    if not isinstance(record, dict) or set(record) != keys:
        raise ValueError("Exactly the declared field observations are required")
    total = _positive_integer(record["group_total"], "group_total")
    if not isinstance(record["source_ref"], str) or not record["source_ref"].strip():
        raise ValueError("A source reference is required")
    subfields = record["subfields"]
    if subfields is not None:
        if not isinstance(subfields, list) or not subfields:
            raise ValueError("subfields must be a nonempty list or null")
        subfields = [_positive_integer(x, "subfield") for x in subfields]
    following = record["following_fields"]
    if not isinstance(following, list) or not following:
        raise ValueError("following_fields must be a nonempty list")
    following = [_positive_integer(x, "following field") for x in following]
    return total, subfields, following


def compare_partitioned_fields(*, student, reference):
    """Compare numeric structure while retaining unresolved subgroup detail."""
    st, ss, sf = _validate(student)
    rt, rs, rf = _validate(reference)
    return {
        "scope": "diagnostic_only",
        "student_source_ref": student["source_ref"],
        "reference_source_ref": reference["source_ref"],
        "group_total_match": st == rt,
        "subfields_match": None if ss is None or rs is None else ss == rs,
        "student_subfields_sum_matches_total": None if ss is None else sum(ss) == st,
        "reference_subfields_sum_matches_total": None if rs is None else sum(rs) == rt,
        "following_fields_match": sf == rf,
        "student_source_independently_verified": False,
        "criterion_relevance_verified": False,
        "score_change_authorized": False,
        "requires_human_review": True,
    }
