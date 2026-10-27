"""Strict metadata boundary for the experimental target-label-free protocol.

This module validates structure, not the visual or semantic absence of teacher
marks. Image pixels, transcriptions, data provenance and parameter provenance
must be audited separately before any scoring run.
"""

import json
import math
import re
from pathlib import Path


ALLOWED_ANSWER_FIELDS = frozenset({
    "answer_id", "question_id", "raw_text", "isimagine", "source_file",
    "source_image", "source_line", "student_image", "subject",
    "subject_name", "visual_placeholder_detected",
})


def load_unlabeled_metadata(path):
    """Read exactly one metadata file, rejecting unexpected fields and IDs.

    Unknown fields are rejected rather than silently stripped, so adding a
    teacher-score column cannot accidentally create an apparently safe input.
    No teacher database or review policy is opened by this function.
    """
    answers = {}
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid metadata JSON on line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Metadata line {line_number} must be an object")
            unexpected = set(record) - ALLOWED_ANSWER_FIELDS
            if unexpected:
                raise ValueError(
                    f"Unexpected metadata fields on line {line_number}: "
                    + ", ".join(sorted(unexpected))
                )
            answer_id = record.get("answer_id")
            question_id = record.get("question_id")
            if not isinstance(answer_id, str) or not answer_id.strip():
                raise ValueError(f"Missing answer_id on line {line_number}")
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError(f"Missing question_id on line {line_number}")
            if answer_id in answers:
                raise ValueError(f"Duplicate answer_id on line {line_number}")
            if "raw_text" in record and not isinstance(record["raw_text"], str):
                raise ValueError(f"raw_text must be text on line {line_number}")
            answers[answer_id] = record
    return answers


def load_unlabeled_question(root, question_id, split):
    """Load initial rubric and the exact unreviewed ID split for one question.

    Paths for active optimized rubrics, teacher labels, review decisions and
    calibration configurations are never used. This is an input boundary only:
    it does not establish that text, images or upstream selection lack labels.
    """
    root = Path(root).resolve(strict=True)
    if not isinstance(question_id, str) or not re.fullmatch(r"[A-Za-z0-9_]+", question_id):
        raise ValueError("Invalid question ID")
    if split not in {"calibration", "validation", "test"}:
        raise ValueError("Unsupported answer split")
    data = json.loads((root / "exam_database.json").read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Question database must be a list")
    matches = [item for item in data if isinstance(item, dict) and item.get("question_id") == question_id]
    if len(matches) != 1:
        raise ValueError("Question ID must occur exactly once")
    question = matches[0]
    group = question.get("rubric_group") or question_id.split("_", 1)[0]
    if not isinstance(group, str) or not re.fullmatch(r"[A-Za-z0-9_]+", group):
        raise ValueError("Invalid rubric group")
    rubric_path = root / "rubrics" / "initial" / group / f"{question_id}_rubric_standard.json"
    if rubric_path.is_symlink() or not rubric_path.resolve().is_relative_to(root):
        raise ValueError("Unsafe initial rubric path")
    rubric = json.loads(rubric_path.read_text(encoding="utf-8-sig"))
    if not isinstance(rubric, list) or not rubric:
        raise ValueError("Initial rubric must contain scoring items")
    try:
        points = [float(item["points"]) for item in rubric]
        total = float(question["total_score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid rubric score values") from exc
    if not math.isfinite(total) or total <= 0 or any(not math.isfinite(p) or p < 0 for p in points):
        raise ValueError("Invalid rubric score range")
    if abs(sum(points) - total) > 1e-6:
        raise ValueError("Initial rubric total does not match question total")
    metadata = load_unlabeled_metadata(root / "answer_metadata.jsonl")
    path = root / "splits" / "by_question" / f"{question_id}.json"
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Unsafe split path")
    splits = json.loads(path.read_text(encoding="utf-8-sig"))
    if splits.get("question_id") != question_id:
        raise ValueError("Split question mismatch")
    groups = [splits.get(name) for name in ("calibration", "validation", "test")]
    if any(not isinstance(group_ids, list) for group_ids in groups):
        raise ValueError("Missing answer split")
    ids = [answer_id for group_ids in groups for answer_id in group_ids]
    if any(not isinstance(answer_id, str) for answer_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Invalid or overlapping answer splits")
    available = {answer_id for answer_id, item in metadata.items() if item["question_id"] == question_id}
    if set(ids) != available:
        raise ValueError("Split answers do not match question metadata")
    return {
        "question_id": question_id,
        "question_text": question.get("question_text", ""),
        "reference_text": question.get("ref_text", ""),
        "total_score": total,
        "initial_rubric": rubric,
        "answer_split": split,
        "answers": [metadata[answer_id] for answer_id in splits[split]],
        "teacher_labels_loaded": False,
        "content_provenance_audited": False,
    }
