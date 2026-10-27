"""Audit existing CSBench answer-ID splits without opening teacher scores.

This checks consistency with the repository's deterministic ID-hash split.
It cannot prove the upstream answer selection was independent of labels, nor
that the already inspected test questions form a new independent holdout.
"""
import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from label_free_inputs import load_unlabeled_metadata
from scripts.prepare_csbench import split_answer_ids


def audit_splits(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    seed = manifest["split_seed"]
    calibration_ratio = manifest["calibration_ratio"]
    validation_ratio = manifest["validation_ratio"]
    minimum = manifest["minimum_calibration_size"]
    if not isinstance(seed, str) or not seed:
        raise ValueError("Invalid split seed")
    metadata = load_unlabeled_metadata(root / "answer_metadata.jsonl")
    by_question = defaultdict(list)
    for answer_id, record in metadata.items():
        by_question[record["question_id"]].append(answer_id)
    if not by_question:
        raise ValueError("No metadata answers found")
    names = ("calibration", "validation", "test")
    counts = {name: 0 for name in names}
    mismatches = []
    for question_id, ids in sorted(by_question.items()):
        if not question_id.replace("_", "").isalnum():
            raise ValueError("Invalid question ID")
        path = root / "splits" / "by_question" / f"{question_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        expected = split_answer_ids(ids, f"{seed}:{question_id}", calibration_ratio, validation_ratio, minimum)
        if payload.get("question_id") != question_id or payload.get("split_seed") != seed:
            mismatches.append(question_id)
            continue
        if any(payload.get(name) != expected[name] for name in names):
            mismatches.append(question_id)
            continue
        for name in names:
            counts[name] += len(expected[name])
    return {
        "algorithm": "sha256_seed_question_answer_id",
        "questions": len(by_question),
        "answers": len(metadata),
        "counts": counts,
        "matching_questions": len(by_question) - len(mismatches),
        "mismatched_question_ids": mismatches,
        "teacher_scores_opened": False,
        "upstream_label_independence_verified": False,
        "fresh_independent_holdout_verified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default="data/csbench")
    args = parser.parse_args(argv)
    try:
        report = audit_splits(args.prepared_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Split audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["mismatched_question_ids"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
