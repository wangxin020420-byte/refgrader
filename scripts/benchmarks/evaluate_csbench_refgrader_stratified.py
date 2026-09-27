"""Evaluate a RefGrader private-test run with the SAS-Bench comparison metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmarks.evaluate_csbench_sasbench_baseline import evaluate


TASK_FILE_PATTERN = re.compile(r"^\d+_.+\.jsonl$", flags=re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_scores(compare_csv: Path, score_column: str) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    with compare_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"student_id", "teacher", score_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"compare.csv is missing columns: {sorted(missing)}")
        for row in reader:
            answer_id = str(row["student_id"]).strip()
            if not answer_id or answer_id in scores:
                raise ValueError(f"Missing or duplicate student_id: {answer_id!r}")
            teacher = float(row["teacher"])
            prediction = float(row[score_column])
            if not math.isfinite(teacher) or not math.isfinite(prediction):
                raise ValueError(f"Non-finite score for {answer_id}")
            scores[answer_id] = {"teacher": teacher, "prediction": prediction}
    return scores


def materialize_predictions(
    compare_csv: Path,
    source_dir: Path,
    output_dir: Path,
    score_column: str,
    expected_records: int,
) -> Path:
    scores = load_scores(compare_csv, score_column)
    if len(scores) != expected_records:
        raise ValueError(
            f"compare.csv count mismatch: expected={expected_records}, actual={len(scores)}"
        )
    prediction_dir = output_dir / "materialized_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    found_ids: set[str] = set()
    task_count = 0
    for source_path in sorted(source_dir.glob("*.jsonl")):
        if not TASK_FILE_PATTERN.fullmatch(source_path.name):
            continue
        predictions = []
        for source_row in read_jsonl(source_path):
            answer_id = str(source_row.get("id") or "").strip()
            score = scores.get(answer_id)
            if score is None:
                continue
            if answer_id in found_ids:
                raise ValueError(f"Duplicate source answer id: {answer_id}")
            teacher = float(source_row["manual_label"])
            total = float(source_row["total"])
            if not math.isclose(teacher, score["teacher"], abs_tol=1e-9):
                raise ValueError(f"Teacher score mismatch for {answer_id}")
            prediction = score["prediction"]
            if not 0.0 <= prediction <= total:
                raise ValueError(
                    f"Prediction outside [0, total] for {answer_id}: {prediction}/{total}"
                )
            row = dict(source_row)
            row["pred_label"] = prediction
            predictions.append(row)
            found_ids.add(answer_id)
        if not predictions:
            continue
        task_count += 1
        target = prediction_dir / f"{source_path.stem}_prediction.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    missing_ids = sorted(set(scores) - found_ids)
    if missing_ids:
        raise ValueError(
            f"Source data is missing {len(missing_ids)} answers; first={missing_ids[:5]}"
        )
    if task_count == 0:
        raise ValueError("No task predictions were materialized")
    return prediction_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-csv", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--score-column", default="final_calibrated_score")
    args = parser.parse_args()
    prediction_dir = materialize_predictions(
        args.compare_csv.resolve(),
        args.source_dir.resolve(),
        args.output_dir.resolve(),
        args.score_column,
        args.expected_records,
    )
    summary = evaluate(prediction_dir, args.output_dir, args.expected_records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
