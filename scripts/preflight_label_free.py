"""Read-only label-free input preflight. This does not execute scoring.

Passing the structural checks is not proof of label-free visual/text content,
parameter provenance, independent test data, or readiness for a live experiment.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from label_free_inputs import load_unlabeled_question
from scripts.audit_label_free_splits import audit_splits
from scripts.label_free_input_inventory import build_inventory


def preflight(root):
    root = Path(root)
    inventory = build_inventory(root)
    split_audit = audit_splits(root)
    if split_audit["mismatched_question_ids"]:
        raise ValueError("Input splits differ from the expected ID-based split")
    database = json.loads((root / "exam_database.json").read_text(encoding="utf-8-sig"))
    if not isinstance(database, list):
        raise ValueError("Question database must be a list")
    questions = [item.get("question_id") for item in database if isinstance(item, dict)]
    if len(questions) != len(database) or not questions or len(set(questions)) != len(questions):
        raise ValueError("Question IDs must be unique and present")
    counts = {name: 0 for name in ("calibration", "validation", "test")}
    for question_id in questions:
        for split in counts:
            item = load_unlabeled_question(root, question_id, split)
            counts[split] += len(item["answers"])
    if counts != split_audit["counts"] or len(questions) != split_audit["questions"]:
        raise ValueError("Question and metadata split totals differ")
    return {
        "schema": "label_free_preflight_v1",
        "structural_checks_passed": True,
        "question_count": len(questions),
        "answer_counts": counts,
        "input_groups": inventory["groups"],
        "teacher_score_file_opened": False,
        "live_model_calls": 0,
        "checks_pending": [
            "student-image teacher annotations",
            "transcription and other input content provenance",
            "upstream data selection and fresh test holdout",
            "label-free scoring integration and frozen parameter provenance",
            "isolated OCR and facts caches",
        ],
        "ready_for_live_scoring": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default="data/csbench")
    args = parser.parse_args(argv)
    try:
        result = preflight(args.prepared_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
