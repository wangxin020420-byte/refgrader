"""Strict metadata boundary for the experimental target-label-free protocol.

This module validates structure, not the visual or semantic absence of teacher
marks. Image pixels, transcriptions, data provenance and parameter provenance
must be audited separately before any scoring run.
"""

import json
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
