#!/usr/bin/env python3
"""Audit the outer question split and per-question answer splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


OUTER_SPLITS = ("train", "validation", "test")
ANSWER_SPLITS = ("calibration", "validation", "test")


def score_band(value: float) -> str:
    if value <= 5:
        return "0-5"
    if value <= 10:
        return "6-10"
    if value <= 20:
        return "11-20"
    return ">20"


def audit(prepared_dir: Path) -> dict:
    database_path = prepared_dir / "exam_database.json"
    questions = json.loads(database_path.read_text(encoding="utf-8-sig"))
    by_id = {str(item["question_id"]): item for item in questions}
    metadata_records = []
    with (prepared_dir / "answer_metadata.jsonl").open(
        "r", encoding="utf-8-sig"
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                record = json.loads(line)
                record["_line_number"] = line_number
                metadata_records.append(record)
    metadata_counts = Counter(
        str(record.get("answer_id", "")) for record in metadata_records
    )
    metadata_by_id = {
        str(record["answer_id"]): record
        for record in metadata_records
        if record.get("answer_id")
    }

    outer_membership: dict[str, str] = {}
    outer_summary = {}
    errors = []
    for split in OUTER_SPLITS:
        path = prepared_dir / "splits" / f"{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        ids = [str(value) for value in payload.get("question_ids", [])]
        duplicates = sorted(
            qid for qid, count in Counter(ids).items() if count > 1
        )
        unknown = sorted(set(ids) - set(by_id))
        overlap = sorted(qid for qid in ids if qid in outer_membership)
        if duplicates:
            errors.append(f"{split}: duplicate question IDs: {duplicates}")
        if unknown:
            errors.append(f"{split}: unknown question IDs: {unknown}")
        if overlap:
            errors.append(f"{split}: outer split overlap: {overlap}")
        for qid in ids:
            outer_membership[qid] = split

        selected = [by_id[qid] for qid in ids if qid in by_id]
        outer_summary[split] = {
            "question_count": len(ids),
            "answer_count": sum(int(item.get("sample_count", 0)) for item in selected),
            "subjects": dict(sorted(Counter(
                str(item.get("subject") or "unknown") for item in selected
            ).items())),
            "score_bands": dict(sorted(Counter(
                score_band(float(item.get("total_score", 0))) for item in selected
            ).items())),
            "visual_questions": sum(
                bool(item.get("requires_visual_evidence")) for item in selected
            ),
            "question_ids": ids,
        }

    missing_outer = sorted(set(by_id) - set(outer_membership))
    if missing_outer:
        errors.append(f"questions missing from outer split: {missing_outer}")

    matrix = {
        split: {answer_split: 0 for answer_split in ANSWER_SPLITS}
        for split in OUTER_SPLITS
    }
    per_question = {}
    all_answer_ids = set()
    answer_membership = {}
    duplicate_metadata_ids = sorted(
        answer_id
        for answer_id, count in metadata_counts.items()
        if answer_id and count > 1
    )
    if duplicate_metadata_ids:
        errors.append(
            f"answer_metadata.jsonl duplicate answer IDs: {duplicate_metadata_ids}"
        )
    for qid, question in sorted(by_id.items()):
        split_path = prepared_dir / "splits" / "by_question" / f"{qid}.json"
        payload = json.loads(split_path.read_text(encoding="utf-8-sig"))
        seen = set()
        counts = {}
        for answer_split in ANSWER_SPLITS:
            ids = [str(value) for value in payload.get(answer_split, [])]
            duplicate_ids = sorted(
                value for value, count in Counter(ids).items() if count > 1
            )
            overlap = sorted(set(ids) & seen)
            if duplicate_ids:
                errors.append(
                    f"{qid}/{answer_split}: duplicate answer IDs: {duplicate_ids}"
                )
            if overlap:
                errors.append(f"{qid}: answer split overlap: {overlap}")
            for answer_id in ids:
                metadata = metadata_by_id.get(answer_id)
                if metadata is None:
                    errors.append(
                        f"{qid}/{answer_split}: unknown answer ID: {answer_id}"
                    )
                elif str(metadata.get("question_id")) != qid:
                    errors.append(
                        f"{qid}/{answer_split}: {answer_id} belongs to "
                        f"{metadata.get('question_id')} in answer metadata"
                    )
                previous = answer_membership.get(answer_id)
                current = f"{qid}/{answer_split}"
                if previous and previous != current:
                    errors.append(
                        f"answer ID assigned more than once: {answer_id} "
                        f"in {previous} and {current}"
                    )
                answer_membership[answer_id] = current
            seen.update(ids)
            all_answer_ids.update(ids)
            counts[answer_split] = len(ids)
            outer = outer_membership.get(qid)
            if outer:
                matrix[outer][answer_split] += len(ids)
        expected = int(question.get("sample_count", 0))
        if len(seen) != expected:
            errors.append(
                f"{qid}: answer split total {len(seen)} != sample_count {expected}"
            )
        per_question[qid] = counts

    missing_answer_ids = sorted(set(metadata_by_id) - all_answer_ids)
    if missing_answer_ids:
        errors.append(
            f"answers missing from per-question splits: {missing_answer_ids}"
        )

    report = {
        "status": "passed" if not errors else "failed",
        "prepared_dir": str(prepared_dir.resolve()),
        "question_count": len(by_id),
        "answer_count": len(all_answer_ids),
        "answer_metadata_count": len(metadata_records),
        "outer_question_splits": outer_summary,
        "outer_by_answer_split_matrix": matrix,
        "per_question_answer_splits": per_question,
        "errors": errors,
        "interpretation": {
            "outer_train": (
                "Questions available for method development and model/rubric "
                "selection; it is not the final held-out benchmark."
            ),
            "answer_calibration": (
                "Answers used for rubric variance refinement inside one question."
            ),
            "answer_validation": (
                "Independent answers used to fit A3WA thresholds, uncertainty, "
                "BND action gates, and optional residual calibration."
            ),
            "answer_test": (
                "Held-out answers used only for final reporting after all gates pass."
            ),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", default="data/csbench")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(Path(args.prepared_dir))
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(destination)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
