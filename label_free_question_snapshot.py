"""Read-only adapter for an unapproved candidate question snapshot.

The adapter supplies original item-level data and all item-linked images. It
does not certify provenance, access answer records or enable formal testing.
"""
from collections import defaultdict
import json
from pathlib import Path

from scripts.verify_label_free_question_snapshot import verify


def load_candidate_question(snapshot_dir, question_id):
    root = Path(snapshot_dir).resolve(strict=True)
    report = verify(root)
    snapshot = json.loads((root / "snapshot.json").read_text(encoding="utf-8-sig"))
    if question_id in snapshot["reference_with_teacher_word_questions"]:
        raise ValueError("Question reference contains unresolved teacher-scoring wording")
    matches = [q for q in snapshot["questions"] if q["question_id"] == question_id]
    if len(matches) != 1:
        raise ValueError("Question must exist exactly once in candidate snapshot")
    q = matches[0]
    if len(q["question_images"]) > 1:
        raise ValueError("Multiple question images are not supported by the model interface")
    rubric = [{"id": item["id"], "item": item["description"], "points": item["points"],
               "standard_answer_text": item["reference_text"]} for item in q["rubric"]]
    grouped = defaultdict(list)
    for item in q["rubric"]:
        if item["reference_image"]:
            grouped[item["reference_image"]["path"]].append(item["id"])
    refs = [{"path": root / path, "item_ids": ids} for path, ids in sorted(grouped.items())]
    question_image = root / q["question_images"][0]["path"] if q["question_images"] else None
    context = {"question_id": question_id, "question_text": q["question_text"],
               "reference_text": "\n".join(item["standard_answer_text"] for item in rubric),
               "total_score": q["max_score"], "initial_rubric": rubric}
    return {"context": context, "question_image": question_image, "reference_images": refs,
            "snapshot_sha256_normalized": report["snapshot_sha256_normalized"],
            "source_provenance_audited": False, "approved_for_formal_experiment": False}
