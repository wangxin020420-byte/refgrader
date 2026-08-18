from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from benchmark_datasets.contract import (
    audit_prepared_benchmark,
    load_json,
    sha256_file,
    write_json,
)


PAPER_REFERENCE_RESULTS = {
    "citation": (
        "Mohler, Bunescu, and Mihalcea. Learning to Grade Short Answer "
        "Questions using Semantic Similarity Measures and Dependency Graph "
        "Alignments. ACL 2011."
    ),
    "reported_question_count": 80,
    "answer_count": 2273,
    "fold_count": 12,
    "average_grade_baseline_rmse": 1.097,
    "best_pearson": 0.518,
    "best_rmse": 0.978,
    "best_median_per_question_rmse": 0.862,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def _source_unit(source_question_id: str) -> int:
    try:
        unit = int(str(source_question_id).split(".", 1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid Mohler source question ID: {source_question_id!r}"
        ) from exc
    if not 1 <= unit <= 12:
        raise ValueError(
            f"Mohler source unit must be in [1, 12], got {unit}"
        )
    return unit


def _normalized_exclusions(
    questions: list[dict[str, Any]],
    excluded_question_ids: Iterable[str],
) -> set[str]:
    by_source = {
        str(item["source_question_id"]): str(item["question_id"])
        for item in questions
    }
    by_normalized = {value: value for value in by_source.values()}
    resolved: set[str] = set()
    unknown: list[str] = []
    for raw in excluded_question_ids:
        value = str(raw).strip()
        question_id = by_source.get(value) or by_normalized.get(value)
        if question_id:
            resolved.add(question_id)
        else:
            unknown.append(value)
    if unknown:
        raise ValueError(f"Unknown Mohler question exclusions: {sorted(unknown)}")
    return resolved


def _partition_integrity(
    all_answer_ids: set[str],
    fold_answers: dict[str, set[str]],
    *,
    fold_id: str,
) -> None:
    names = ("train", "calibration", "test")
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = fold_answers[left] & fold_answers[right]
            if overlap:
                raise ValueError(
                    f"{fold_id} has {left}/{right} leakage: "
                    f"{sorted(overlap)[:5]}"
                )
    union = set().union(*(fold_answers[name] for name in names))
    if union != all_answer_ids:
        raise ValueError(
            f"{fold_id} does not partition all answers: "
            f"expected={len(all_answer_ids)}, actual={len(union)}"
        )


def build_mohler_acl2011_protocol(
    prepared_dir: str | Path,
    *,
    excluded_question_ids: Iterable[str] = (),
    require_paper_question_count: bool = False,
) -> dict[str, Any]:
    root = Path(prepared_dir).expanduser().resolve()
    audit = audit_prepared_benchmark(root)
    manifest = load_json(root / "manifest.json")
    if manifest.get("adapter") != "mohler":
        raise ValueError(
            "The Mohler ACL 2011 protocol requires a prepared Mohler dataset."
        )

    questions = _read_jsonl(root / "normalized" / "questions.jsonl")
    answers = _read_jsonl(root / "normalized" / "answers.jsonl")
    excluded = _normalized_exclusions(questions, excluded_question_ids)
    questions = [
        item for item in questions if str(item["question_id"]) not in excluded
    ]
    included_question_ids = {str(item["question_id"]) for item in questions}
    answers = [
        item for item in answers if str(item["question_id"]) in included_question_ids
    ]
    if not questions or not answers:
        raise ValueError("The Mohler protocol cannot be built from empty data.")

    answer_ids = [str(item["answer_id"]) for item in answers]
    if len(answer_ids) != len(set(answer_ids)):
        raise ValueError("Mohler answer IDs are not unique.")

    questions_by_unit: dict[int, list[str]] = defaultdict(list)
    answers_by_unit: dict[int, set[str]] = defaultdict(set)
    for item in questions:
        questions_by_unit[_source_unit(str(item["source_question_id"]))].append(
            str(item["question_id"])
        )
    for item in answers:
        answers_by_unit[_source_unit(str(item["source_question_id"]))].add(
            str(item["answer_id"])
        )
    if set(questions_by_unit) != set(range(1, 13)):
        raise ValueError(
            "Mohler ACL 2011 requires all 12 assignment/exam source units."
        )

    all_answer_ids = set(answer_ids)
    test_coverage: Counter[str] = Counter()
    folds: list[dict[str, Any]] = []
    for test_unit in range(1, 13):
        calibration_unit = (test_unit % 12) + 1
        train_units = [
            unit
            for unit in range(1, 13)
            if unit not in {test_unit, calibration_unit}
        ]
        fold_answers = {
            "train": set().union(*(answers_by_unit[unit] for unit in train_units)),
            "calibration": set(answers_by_unit[calibration_unit]),
            "test": set(answers_by_unit[test_unit]),
        }
        fold_id = f"fold_{test_unit:02d}"
        _partition_integrity(all_answer_ids, fold_answers, fold_id=fold_id)
        test_coverage.update(fold_answers["test"])
        folds.append(
            {
                "fold_id": fold_id,
                "test_unit": test_unit,
                "test_unit_type": (
                    "assignment" if test_unit <= 10 else "exam"
                ),
                "calibration_unit": calibration_unit,
                "train_units": train_units,
                "train_question_ids": sorted(
                    question_id
                    for unit in train_units
                    for question_id in questions_by_unit[unit]
                ),
                "calibration_question_ids": sorted(
                    questions_by_unit[calibration_unit]
                ),
                "test_question_ids": sorted(questions_by_unit[test_unit]),
                "train_answer_count": len(fold_answers["train"]),
                "calibration_answer_count": len(fold_answers["calibration"]),
                "test_answer_count": len(fold_answers["test"]),
            }
        )

    invalid_coverage = [
        answer_id
        for answer_id in all_answer_ids
        if test_coverage[answer_id] != 1
    ]
    if invalid_coverage:
        raise ValueError(
            "Every Mohler answer must be tested exactly once; invalid IDs: "
            f"{sorted(invalid_coverage)[:5]}"
        )

    question_count = len(questions)
    paper_count_compatible = (
        question_count == PAPER_REFERENCE_RESULTS["reported_question_count"]
    )
    if require_paper_question_count and not paper_count_compatible:
        raise ValueError(
            "ACL 2011 reports 80 questions, but this protocol contains "
            f"{question_count}. Supply a documented question exclusion before "
            "claiming a paper-count-compatible reproduction."
        )

    source_units = []
    for unit in range(1, 13):
        source_units.append(
            {
                "unit": unit,
                "type": "assignment" if unit <= 10 else "exam",
                "question_ids": sorted(questions_by_unit[unit]),
                "question_count": len(questions_by_unit[unit]),
                "answer_count": len(answers_by_unit[unit]),
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": (
            "mohler_acl2011_paper_count_v1"
            if paper_count_compatible
            else f"mohler_acl2011_archive{question_count}_v1"
        ),
        "protocol_family": "mohler_acl2011_12fold",
        "dataset_id": str(manifest["dataset_id"]),
        "dataset_manifest_sha256": sha256_file(root / "manifest.json"),
        "prepared_content_sha256": audit.get("prepared_content_sha256"),
        "question_count": question_count,
        "answer_count": len(answers),
        "excluded_question_ids": sorted(excluded),
        "gold_score_policy": "official_two_rater_average_normalized_0_5",
        "fold_policy": {
            "test": "one_complete_assignment_or_exam",
            "calibration": "next_source_unit_cyclically",
            "train": "remaining_ten_source_units",
            "test_coverage": "every_included_answer_exactly_once",
        },
        "paper_compatibility": {
            "paper_reported_question_count": PAPER_REFERENCE_RESULTS[
                "reported_question_count"
            ],
            "question_count_compatible": paper_count_compatible,
            "direct_paper_comparison_authorized": False,
            "status": (
                "question_count_compatible_but_exclusion_requires_citation"
                if paper_count_compatible
                else "archive_complete_81_question_protocol"
            ),
            "note": (
                "The distributed archive contains 81 included questions while "
                "the ACL 2011 paper reports 80. Published numbers are reference "
                "targets, not directly comparable until the discrepancy is "
                "resolved and documented."
            ),
        },
        "paper_reference_results": PAPER_REFERENCE_RESULTS,
        "source_units": source_units,
        "folds": folds,
        "integrity": {
            "fold_count": len(folds),
            "partition_disjoint_per_fold": True,
            "all_answers_tested_once": True,
        },
    }


def write_protocol(protocol: dict[str, Any], output: str | Path) -> Path:
    target = Path(output).expanduser().resolve()
    write_json(target, protocol)
    return target


def _pearson(actual: list[float], predicted: list[float]) -> float | None:
    if len(actual) < 2:
        return None
    mean_actual = statistics.fmean(actual)
    mean_predicted = statistics.fmean(predicted)
    numerator = sum(
        (left - mean_actual) * (right - mean_predicted)
        for left, right in zip(actual, predicted)
    )
    denominator = math.sqrt(
        sum((value - mean_actual) ** 2 for value in actual)
        * sum((value - mean_predicted) ** 2 for value in predicted)
    )
    return numerator / denominator if denominator > 0 else None


def _score_metrics(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Metric vectors must be non-empty and equally sized.")
    errors = [estimate - truth for truth, estimate in zip(actual, predicted)]
    return {
        "n": len(actual),
        "MAE": statistics.fmean(abs(value) for value in errors),
        "RMSE": math.sqrt(statistics.fmean(value * value for value in errors)),
        "Pearson": _pearson(actual, predicted),
        "SER2": statistics.fmean(abs(value) > 2.0 for value in errors),
        "bias": statistics.fmean(errors),
    }


def evaluate_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    score_fields: Iterable[str],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No prediction rows were provided.")
    per_question_rows: list[dict[str, Any]] = []
    global_metrics: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    for score_field in score_fields:
        actual = [float(row["teacher_score"]) for row in rows]
        predicted = [float(row[score_field]) for row in rows]
        metrics = _score_metrics(actual, predicted)
        question_rmses: list[float] = []
        for question_id in sorted(grouped):
            question_rows = grouped[question_id]
            question_metrics = _score_metrics(
                [float(row["teacher_score"]) for row in question_rows],
                [float(row[score_field]) for row in question_rows],
            )
            question_rmses.append(float(question_metrics["RMSE"]))
            per_question_rows.append(
                {
                    "score_type": score_field,
                    "question_id": question_id,
                    **question_metrics,
                }
            )
        metrics["median_per_question_RMSE"] = statistics.median(question_rmses)
        global_metrics[score_field] = metrics
    return {
        "global": global_metrics,
        "per_question": per_question_rows,
    }


def _baseline_text(
    answer: dict[str, Any],
    question: dict[str, Any],
) -> str:
    return (
        f"question: {question.get('question_text', '')}\n"
        f"reference: {question.get('reference_answer', '')}\n"
        f"student: {answer.get('raw_text', '')}"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def run_mohler_acl2011_baselines(
    prepared_dir: str | Path,
    protocol: dict[str, Any],
    output_dir: str | Path,
    *,
    random_state: int = 2011,
) -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.isotonic import IsotonicRegression
        from sklearn.svm import LinearSVR
    except ImportError as exc:
        raise RuntimeError(
            "Mohler baselines require the locked scikit-learn dependency."
        ) from exc

    root = Path(prepared_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    questions = {
        str(item["question_id"]): item
        for item in _read_jsonl(root / "normalized" / "questions.jsonl")
    }
    excluded = set(protocol.get("excluded_question_ids") or [])
    answers = [
        item
        for item in _read_jsonl(root / "normalized" / "answers.jsonl")
        if str(item["question_id"]) not in excluded
    ]
    answers_by_unit: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for answer in answers:
        answers_by_unit[_source_unit(str(answer["source_question_id"]))].append(
            answer
        )

    predictions: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold in protocol["folds"]:
        train = [
            answer
            for unit in fold["train_units"]
            for answer in answers_by_unit[int(unit)]
        ]
        calibration = answers_by_unit[int(fold["calibration_unit"])]
        test = answers_by_unit[int(fold["test_unit"])]
        train_y = [float(item["actual_score"]) for item in train]
        calibration_y = [float(item["actual_score"]) for item in calibration]
        test_y = [float(item["actual_score"]) for item in test]
        train_mean = statistics.fmean(train_y)

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
        )
        train_matrix = vectorizer.fit_transform(
            _baseline_text(item, questions[str(item["question_id"])])
            for item in train
        )
        calibration_matrix = vectorizer.transform(
            _baseline_text(item, questions[str(item["question_id"])])
            for item in calibration
        )
        test_matrix = vectorizer.transform(
            _baseline_text(item, questions[str(item["question_id"])])
            for item in test
        )
        model = LinearSVR(
            C=1.0,
            epsilon=0.1,
            max_iter=20000,
            random_state=random_state,
        )
        model.fit(train_matrix, train_y)
        calibration_raw = [
            min(5.0, max(0.0, float(value)))
            for value in model.predict(calibration_matrix)
        ]
        test_raw = [
            min(5.0, max(0.0, float(value)))
            for value in model.predict(test_matrix)
        ]
        if len(set(round(value, 8) for value in calibration_raw)) >= 2:
            isotonic = IsotonicRegression(
                y_min=0.0,
                y_max=5.0,
                out_of_bounds="clip",
            )
            isotonic.fit(calibration_raw, calibration_y)
            test_calibrated = [float(value) for value in isotonic.predict(test_raw)]
            calibration_status = "fitted"
        else:
            test_calibrated = list(test_raw)
            calibration_status = "skipped_constant_predictions"

        for answer, teacher, raw_score, calibrated_score in zip(
            test,
            test_y,
            test_raw,
            test_calibrated,
        ):
            predictions.append(
                {
                    "fold_id": fold["fold_id"],
                    "test_unit": fold["test_unit"],
                    "question_id": str(answer["question_id"]),
                    "student_id": str(answer["answer_id"]),
                    "teacher_score": teacher,
                    "train_mean": train_mean,
                    "linear_tfidf_svr": raw_score,
                    "linear_tfidf_svr_isotonic": calibrated_score,
                }
            )
        fold_summaries.append(
            {
                "fold_id": fold["fold_id"],
                "train_n": len(train),
                "calibration_n": len(calibration),
                "test_n": len(test),
                "isotonic_status": calibration_status,
            }
        )

    student_ids = [str(row["student_id"]) for row in predictions]
    if len(student_ids) != int(protocol["answer_count"]):
        raise ValueError(
            "Baseline test coverage does not match the protocol: "
            f"expected={protocol['answer_count']}, actual={len(student_ids)}"
        )
    if len(student_ids) != len(set(student_ids)):
        raise ValueError("Baseline predictions contain duplicate test answers.")

    score_fields = (
        "train_mean",
        "linear_tfidf_svr",
        "linear_tfidf_svr_isotonic",
    )
    evaluation = evaluate_prediction_rows(predictions, score_fields=score_fields)
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "comparison_scope": protocol["paper_compatibility"]["status"],
        "paper_reference_results": PAPER_REFERENCE_RESULTS,
        "folds": fold_summaries,
        **evaluation,
    }
    write_protocol(protocol, output / "protocol.json")
    _write_csv(output / "predictions.csv", predictions)
    _write_csv(output / "per_question_metrics.csv", evaluation["per_question"])
    write_json(output / "summary.json", summary)
    return summary
