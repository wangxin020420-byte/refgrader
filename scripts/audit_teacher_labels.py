"""Screen teacher labels for human review without changing the dataset.

This command compares raw teacher scores with an existing grading checkpoint.
It produces candidates only.  A large disagreement is not treated as proof
that the teacher label is wrong because extraction, rubric mapping, or model
grading can also be responsible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SCORE_KEY = "three_way_core_score"
MODEL_SCORE_KEYS = (
    "model_avg_score",
    "selected_baseline_score",
    "three_way_core_score",
    "final_calibrated_score",
)


def normalize_question_id(value: str) -> str:
    question_id = value.strip().upper()
    if not question_id:
        raise argparse.ArgumentTypeError("Question ID cannot be empty.")
    return question_id


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def robust_limit(
    residuals: list[float],
    *,
    max_score: float,
    sigma: float,
    minimum_ratio: float,
    minimum_points: float,
) -> float:
    magnitudes = [abs(value) for value in residuals]
    if not magnitudes:
        return max(minimum_points, max_score * minimum_ratio)
    center = statistics.median(magnitudes)
    deviations = [abs(value - center) for value in magnitudes]
    mad = statistics.median(deviations)
    robust_spread = 1.4826 * mad
    return max(
        minimum_points,
        max_score * minimum_ratio,
        center + sigma * robust_spread,
    )


def load_teacher_scores(path: Path) -> dict[str, dict[str, float]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Teacher scores must be a JSON object: {path}")
    return payload


def load_question_scores(path: Path) -> dict[str, float]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Exam database must be a JSON array: {path}")
    return {
        str(row["question_id"]).upper(): float(row["total_score"])
        for row in payload
        if isinstance(row, dict)
        and row.get("question_id")
        and row.get("total_score") is not None
    }


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            answer_id = str(row.get("answer_id") or "").strip()
            if answer_id:
                records[answer_id] = row
    return records


def checkpoint_paths(
    questions: list[str],
    results_dir: Path,
) -> dict[str, Path]:
    paths = {}
    for question_id in questions:
        path = results_dir / f"{question_id}_grading_checkpoint.json"
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        paths[question_id] = path
    return paths


def model_scores(row: dict[str, Any]) -> dict[str, float]:
    scores = {}
    for key in MODEL_SCORE_KEYS:
        value = numeric(row.get(key))
        if value is not None:
            scores[key] = value
    return scores


def risk_value(row: dict[str, Any], name: str) -> float:
    profile = row.get("risk_features") or {}
    value = numeric(profile.get(name))
    if value is None:
        value = numeric((profile.get("primary_risks") or {}).get(name))
    return value or 0.0


def classify_confounds(
    row: dict[str, Any],
    *,
    max_score: float,
    scores: dict[str, float],
) -> tuple[list[str], bool]:
    reasons = []
    extraction_risk = numeric(row.get("extraction_risk")) or 0.0
    extraction_quality = str(row.get("extraction_quality") or "").lower()
    if extraction_risk >= 0.30 or extraction_quality in {"low", "failed"}:
        reasons.append("extraction_risk")

    rubric_risk = risk_value(row, "U_R")
    if rubric_risk >= 0.40:
        reasons.append("rubric_mapping_risk")

    average = scores.get("model_avg_score")
    core = scores.get("three_way_core_score")
    if average is not None and core is not None:
        if abs(average - core) > max(0.5, 0.10 * max_score):
            reasons.append("a3wa_changed_score_materially")

    std_dev = numeric(row.get("std_dev"))
    if std_dev is not None and std_dev > 0.10 * max_score:
        reasons.append("unstable_model_judgements")

    high_confidence = not reasons
    return reasons, high_confidence


def analyze_question(
    question_id: str,
    path: Path,
    *,
    teacher_scores: dict[str, dict[str, float]],
    metadata: dict[str, dict[str, Any]],
    max_score: float,
    score_key: str,
    sigma: float,
    minimum_ratio: float,
    minimum_points: float,
    severe_ratio: float,
    severe_points: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Checkpoint must be a JSON array: {path}")

    comparable = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        answer_id = str(
            row.get("student_id") or row.get("answer_id") or ""
        ).strip()
        raw_teacher = numeric(
            (teacher_scores.get(answer_id) or {}).get(question_id)
        )
        scores = model_scores(row)
        fallback_keys = (
            score_key,
            "selected_baseline_score",
            "model_avg_score",
            "final_calibrated_score",
        )
        reference_key = next(
            (key for key in fallback_keys if key in scores),
            None,
        )
        reference = scores.get(reference_key) if reference_key else None
        if raw_teacher is None or reference is None:
            continue
        comparable.append(
            (
                row,
                answer_id,
                raw_teacher,
                reference,
                reference_key,
                scores,
            )
        )

    residuals = [
        teacher - reference
        for _, _, teacher, reference, _, _ in comparable
    ]
    raw_threshold = robust_limit(
        residuals,
        max_score=max_score,
        sigma=sigma,
        minimum_ratio=minimum_ratio,
        minimum_points=minimum_points,
    )
    severe_threshold = max(severe_points, max_score * severe_ratio)
    # A shifted residual distribution is itself an audit signal. Do not let
    # that shift raise the candidate threshold above the severe-error bound.
    threshold = min(raw_threshold, severe_threshold)
    median_residual = statistics.median(residuals) if residuals else 0.0
    bias_threshold = max(minimum_points, max_score * minimum_ratio)
    systematic_bias = abs(median_residual) >= bias_threshold

    centered_residuals = [
        residual - median_residual for residual in residuals
    ]
    raw_outlier_threshold = robust_limit(
        centered_residuals,
        max_score=max_score,
        sigma=sigma,
        minimum_ratio=minimum_ratio,
        minimum_points=minimum_points,
    )
    outlier_threshold = min(raw_outlier_threshold, severe_threshold)

    candidates = []
    for row, answer_id, teacher, reference, reference_key, scores in comparable:
        residual = teacher - reference
        centered_residual = residual - median_residual
        candidate_reasons = []
        if abs(residual) >= threshold:
            candidate_reasons.append("absolute_disagreement")
        if abs(centered_residual) >= outlier_threshold:
            candidate_reasons.append("within_question_outlier")
        if not candidate_reasons:
            continue
        confounds, high_confidence = classify_confounds(
            row,
            max_score=max_score,
            scores=scores,
        )
        severe = abs(residual) >= severe_threshold
        priority = "P1" if severe and high_confidence else "P2"
        direction = (
            "possible_teacher_over_score"
            if residual > 0
            else "possible_teacher_under_score"
        )
        meta = metadata.get(answer_id) or {}
        candidates.append(
            {
                "question_id": question_id,
                "answer_id": answer_id,
                "review_priority": priority,
                "candidate_type": direction,
                "teacher_score": round(teacher, 6),
                "reference_score_key": reference_key,
                "reference_score": round(reference, 6),
                "teacher_minus_reference": round(residual, 6),
                "absolute_difference": round(abs(residual), 6),
                "candidate_threshold": round(threshold, 6),
                "centered_residual": round(centered_residual, 6),
                "outlier_threshold": round(outlier_threshold, 6),
                "candidate_reasons": "|".join(candidate_reasons),
                "question_median_residual": round(median_residual, 6),
                "question_systematic_bias": systematic_bias,
                "max_score": max_score,
                "model_avg_score": scores.get("model_avg_score"),
                "selected_baseline_score": scores.get(
                    "selected_baseline_score"
                ),
                "three_way_core_score": scores.get(
                    "three_way_core_score"
                ),
                "final_calibrated_score": scores.get(
                    "final_calibrated_score"
                ),
                "route": row.get("3wd_route"),
                "std_dev": row.get("std_dev"),
                "extraction_quality": row.get("extraction_quality"),
                "U_E": risk_value(row, "U_E"),
                "U_S": risk_value(row, "U_S"),
                "U_R": risk_value(row, "U_R"),
                "confounds": "|".join(confounds),
                "human_decision": "",
                "corrected_score": "",
                "review_note": "",
                "raw_text": meta.get("raw_text"),
                "student_image": meta.get("student_image"),
            }
        )

    candidates.sort(
        key=lambda row: (
            row["review_priority"] != "P1",
            -row["absolute_difference"],
            row["answer_id"],
        )
    )
    summary = {
        "question_id": question_id,
        "checkpoint": str(path),
        "records": len(payload),
        "comparable": len(comparable),
        "candidate_count": len(candidates),
        "p1_count": sum(
            row["review_priority"] == "P1" for row in candidates
        ),
        "threshold": round(threshold, 6),
        "raw_robust_threshold": round(raw_threshold, 6),
        "severe_threshold": round(severe_threshold, 6),
        "outlier_threshold": round(outlier_threshold, 6),
        "raw_outlier_threshold": round(raw_outlier_threshold, 6),
        "median_residual": round(median_residual, 6),
        "bias_threshold": round(bias_threshold, 6),
        "systematic_bias": systematic_bias,
        "score_key": score_key,
    }
    return candidates, summary


def write_outputs(
    output_dir: Path,
    candidates: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    fields = list(candidates[0]) if candidates else [
        "question_id",
        "answer_id",
        "review_priority",
        "candidate_type",
        "teacher_score",
        "reference_score_key",
        "reference_score",
        "teacher_minus_reference",
        "absolute_difference",
        "candidate_threshold",
        "human_decision",
        "corrected_score",
        "review_note",
    ]
    with (output_dir / "teacher_label_candidates.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    with (output_dir / "teacher_label_candidates.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in candidates:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate teacher-label review candidates from checkpoints."
    )
    parser.add_argument("questions", nargs="+", type=normalize_question_id)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/csbench"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--score-key",
        choices=MODEL_SCORE_KEYS,
        default=DEFAULT_SCORE_KEY,
    )
    parser.add_argument("--robust-sigma", type=float, default=3.0)
    parser.add_argument("--minimum-ratio", type=float, default=0.15)
    parser.add_argument("--minimum-points", type=float, default=1.0)
    parser.add_argument("--severe-ratio", type=float, default=0.25)
    parser.add_argument("--severe-points", type=float, default=2.0)
    args = parser.parse_args()

    questions = list(dict.fromkeys(args.questions))
    prepared = args.prepared_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    teacher_scores = load_teacher_scores(prepared / "teacher_scores.json")
    question_scores = load_question_scores(prepared / "exam_database.json")
    metadata = load_metadata(prepared / "answer_metadata.jsonl")
    checkpoints = checkpoint_paths(questions, results_dir)

    all_candidates = []
    question_summaries = []
    for question_id in questions:
        if question_id not in question_scores:
            raise KeyError(f"Question not found in database: {question_id}")
        candidates, summary = analyze_question(
            question_id,
            checkpoints[question_id],
            teacher_scores=teacher_scores,
            metadata=metadata,
            max_score=question_scores[question_id],
            score_key=args.score_key,
            sigma=args.robust_sigma,
            minimum_ratio=args.minimum_ratio,
            minimum_points=args.minimum_points,
            severe_ratio=args.severe_ratio,
            severe_points=args.severe_points,
        )
        all_candidates.extend(candidates)
        question_summaries.append(summary)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else prepared
        / "quality_control"
        / "reports"
        / f"teacher_label_audit_{tag}"
    )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "human_review_candidates_only",
        "results_dir": str(results_dir),
        "score_key": args.score_key,
        "candidate_count": len(all_candidates),
        "questions": question_summaries,
        "warning": (
            "A candidate is not automatically a noisy teacher label. "
            "Human review is required before policy activation."
        ),
    }
    write_outputs(output_dir, all_candidates, summary)

    print(f"Teacher-label audit: {len(all_candidates)} candidates")
    for row in all_candidates:
        print(
            f"{row['review_priority']} {row['question_id']} "
            f"{row['answer_id']}: teacher={row['teacher_score']}, "
            f"{row['reference_score_key']}={row['reference_score']}, "
            f"diff={row['teacher_minus_reference']:+.2f}, "
            f"confounds={row['confounds'] or 'none'}"
        )
    print(f"Review files: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
