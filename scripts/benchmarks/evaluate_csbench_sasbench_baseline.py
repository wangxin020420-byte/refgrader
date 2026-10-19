"""Evaluate SAS-Bench-method predictions on the private CSBench test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_level(value: float) -> int:
    """Encode the private dataset's half-point score quantum for QWK."""
    return int(round(float(value) * 2.0))


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = np.asarray([float(row["manual_label"]) for row in rows], dtype=float)
    pred = np.asarray([float(row["pred_label"]) for row in rows], dtype=float)
    errors = pred - truth
    true_levels = [score_level(value) for value in truth]
    pred_levels = [score_level(value) for value in pred]
    if len(set(true_levels) | set(pred_levels)) < 2:
        qwk = None
    else:
        raw_qwk = float(
            cohen_kappa_score(true_levels, pred_levels, weights="quadratic")
        )
        qwk = raw_qwk if math.isfinite(raw_qwk) else None
    if len(rows) > 1 and np.std(truth) > 0 and np.std(pred) > 0:
        pearson = float(np.corrcoef(truth, pred)[0, 1])
    else:
        pearson = None
    return {
        "n": len(rows),
        "QWK": qwk,
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "Pearson": pearson,
        "TAR2": float(np.mean(np.abs(errors) <= 2.0)),
        "SER2": float(np.mean(np.abs(errors) > 2.0)),
        "bias": float(np.mean(errors)),
    }


def qwk_aggregate(task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_qwk = [row for row in task_rows if row["QWK"] is not None]
    if not valid_qwk:
        return {"macro_QWK": None, "answer_weighted_QWK": None}
    return {
        "macro_QWK": float(np.mean([row["QWK"] for row in valid_qwk])),
        "answer_weighted_QWK": float(
            np.average(
                [row["QWK"] for row in valid_qwk],
                weights=[row["n"] for row in valid_qwk],
            )
        ),
    }


def infer_response_type(row: dict[str, Any]) -> str:
    declared = row.get("response_type")
    if declared in {"pure_text", "visual_dependent", "blank"}:
        return str(declared)
    steps = row.get("steps") or []
    raw_text = "".join(str(step.get("response") or "") for step in steps).strip()
    if not raw_text:
        return "blank"
    if bool(row.get("source_isimagine")) or bool(
        row.get("source_visual_placeholder_detected")
    ):
        return "visual_dependent"
    return "pure_text"


def evaluate(prediction_dir: Path, output_dir: Path, expected_records: int) -> dict[str, Any]:
    prediction_dir = prediction_dir.resolve()
    output_dir = output_dir.resolve()
    files = sorted(prediction_dir.glob("*_prediction.jsonl"))
    if not files:
        raise FileNotFoundError(f"No prediction files in {prediction_dir}")
    task_rows = []
    all_rows = []
    seen_ids = set()
    for path in files:
        rows = read_jsonl(path)
        for row in rows:
            answer_id = str(row.get("id") or "")
            if not answer_id or answer_id in seen_ids:
                raise ValueError(f"Missing or duplicate answer id: {answer_id!r}")
            seen_ids.add(answer_id)
            pred = float(row["pred_label"])
            total = float(row["total"])
            if not math.isfinite(pred) or not 0 <= pred <= total:
                raise ValueError(f"Prediction outside [0, total] for {answer_id}")
        item = {"task": path.name.removesuffix("_prediction.jsonl"), **metrics(rows)}
        task_rows.append(item)
        all_rows.extend(rows)
    if len(all_rows) != expected_records:
        raise ValueError(
            f"Prediction count mismatch: expected={expected_records}, actual={len(all_rows)}"
        )
    qwk_summary = qwk_aggregate(task_rows)
    response_type_rows = []
    response_type_summary = {}
    for response_name in ("pure_text", "visual_dependent", "blank"):
        grouped_rows = [
            row for row in all_rows if infer_response_type(row) == response_name
        ]
        grouped_task_rows = []
        for path in files:
            rows = [
                row
                for row in read_jsonl(path)
                if infer_response_type(row) == response_name
            ]
            if not rows:
                continue
            item = {
                "response_type": response_name,
                "task": path.name.removesuffix("_prediction.jsonl"),
                **metrics(rows),
            }
            grouped_task_rows.append(item)
            response_type_rows.append(item)
        aggregate = qwk_aggregate(grouped_task_rows)
        response_type_summary[response_name] = {
            "record_count": len(grouped_rows),
            "task_count": len(grouped_task_rows),
            **aggregate,
            "global_holistic_metrics": metrics(grouped_rows) if grouped_rows else None,
        }
    summary = {
        "record_count": len(all_rows),
        "task_count": len(task_rows),
        "qwk_score_quantum": 0.5,
        **qwk_summary,
        "global_holistic_metrics": metrics(all_rows),
        "response_type_metrics": response_type_summary,
        "CCS": None,
        "ECS": None,
        "fine_grained_metrics_reason": "Private data lacks human step-score and error-type labels.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics_by_question.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    if response_type_rows:
        with (output_dir / "metrics_by_question_and_response_type.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(response_type_rows[0]))
            writer.writeheader()
            writer.writerows(response_type_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    args = parser.parse_args()
    summary = evaluate(args.prediction_dir, args.output_dir, args.expected_records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
