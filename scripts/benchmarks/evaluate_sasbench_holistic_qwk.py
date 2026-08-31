"""Evaluate SAS-Bench holistic scores with the official discrete QWK protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.metrics import cohen_kappa_score
except ModuleNotFoundError:
    def cohen_kappa_score(y_true, y_pred, weights=None):
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        if not labels or len(y_true) == 0:
            return 0.0
        index = {label: i for i, label in enumerate(labels)}
        observed = np.zeros((len(labels), len(labels)), dtype=float)
        for true_value, pred_value in zip(y_true, y_pred):
            observed[index[true_value], index[pred_value]] += 1.0
        observed /= len(y_true)
        expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
        if weights == "quadratic":
            denominator = max((len(labels) - 1) ** 2, 1)
            weight_matrix = np.fromfunction(
                lambda i, j: ((i - j) ** 2) / denominator,
                observed.shape,
                dtype=float,
            )
        else:
            weight_matrix = np.not_equal.outer(labels, labels).astype(float)
        expected_disagreement = float(np.sum(expected * weight_matrix))
        if expected_disagreement == 0.0:
            return 1.0 if np.array_equal(y_true, y_pred) else 0.0
        return 1.0 - float(np.sum(observed * weight_matrix)) / expected_disagreement


METHOD_COLUMNS = {
    "single": "single_first_score",
    "avg": "model_avg_score",
    "selected": "selected_baseline_score",
    "3wd_core": "three_way_core_score",
    "3wd": "final_calibrated_score",
}


def _task_order(task: str) -> tuple[int, str]:
    prefix = task.split("_", 1)[0]
    return (int(prefix) if prefix.isdigit() else 10**9, task)


def _official_integer(value: Any, total: Any) -> int:
    """Match the official scripts, which convert scores with int()."""
    numeric = float(value)
    maximum = int(float(total))
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite score: {value!r}")
    return min(max(int(numeric), 0), maximum)


def load_question_metadata(database_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(database_path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        records = payload.get("questions", payload)
        if isinstance(records, dict):
            records = list(records.values())
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError("Runtime exam database must contain a question list")
    result = {}
    for record in records:
        question_id = str(record.get("question_id", "")).strip()
        source_task = str(record.get("source_task", "")).strip()
        if not question_id or not source_task:
            raise ValueError("Runtime question is missing question_id or source_task")
        result[question_id] = {
            "source_task": source_task,
            "total_score": record.get("total_score"),
        }
    return result


def evaluate_holistic_qwk(
    compare_path: Path,
    database_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question_metadata = load_question_metadata(database_path)
    grouped: dict[tuple[str, str], tuple[list[int], list[int]]] = defaultdict(
        lambda: ([], [])
    )
    seen_ids: set[str] = set()
    with compare_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"question", "student_id", "teacher", *METHOD_COLUMNS.values()}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"compare.csv is missing columns: {sorted(missing)}")
        for row in reader:
            question_id = row["question"]
            metadata = question_metadata.get(question_id)
            if metadata is None:
                raise ValueError(f"Unknown question in compare.csv: {question_id}")
            answer_id = row["student_id"]
            if answer_id in seen_ids:
                raise ValueError(f"Duplicate answer in compare.csv: {answer_id}")
            seen_ids.add(answer_id)
            task = metadata["source_task"]
            total = metadata["total_score"]
            teacher = _official_integer(row["teacher"], total)
            for method, column in METHOD_COLUMNS.items():
                true_scores, predicted_scores = grouped[(task, method)]
                true_scores.append(teacher)
                predicted_scores.append(_official_integer(row[column], total))

    rows = []
    for (task, method), (true_scores, predicted_scores) in sorted(
        grouped.items(), key=lambda item: (_task_order(item[0][0]), item[0][1])
    ):
        qwk = float(
            cohen_kappa_score(true_scores, predicted_scores, weights="quadratic")
        )
        rows.append(
            {
                "task": task,
                "method": method,
                "n": len(true_scores),
                "qwk": qwk if math.isfinite(qwk) else None,
                "qwk_percent": round(qwk * 100.0, 4) if math.isfinite(qwk) else None,
            }
        )

    summary_methods = {}
    for method in METHOD_COLUMNS:
        method_rows = [row for row in rows if row["method"] == method]
        valid = [row for row in method_rows if row["qwk"] is not None]
        total_n = sum(row["n"] for row in valid)
        summary_methods[method] = {
            "task_count": len(method_rows),
            "valid_task_count": len(valid),
            "macro_qwk": (
                sum(row["qwk"] for row in valid) / len(valid) if valid else None
            ),
            "answer_weighted_qwk": (
                sum(row["qwk"] * row["n"] for row in valid) / total_n
                if total_n
                else None
            ),
        }
    summary = {
        "record_count": len(seen_ids),
        "task_count": len({task for task, _ in grouped}),
        "score_discretization": "official_int_then_clip_to_question_total",
        "methods": summary_methods,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-csv", type=Path, required=True)
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-records", type=int)
    args = parser.parse_args()

    rows, summary = evaluate_holistic_qwk(
        args.compare_csv.resolve(), args.database_path.resolve()
    )
    if args.expected_records is not None and summary["record_count"] != args.expected_records:
        raise ValueError(
            "Holistic QWK record count mismatch: "
            f"expected={args.expected_records}, actual={summary['record_count']}"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "holistic_qwk_by_task.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["task", "method", "n", "qwk", "qwk_percent"]
        )
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "holistic_qwk_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Task QWK: {csv_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
