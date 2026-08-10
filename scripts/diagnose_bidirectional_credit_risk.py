import argparse
import csv
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration_utils import (  # noqa: E402
    compute_bidirectional_credit_risks,
    prepare_rubrics_for_calibration,
    safe_float,
)
from sample_quality import load_policy_for_data_path  # noqa: E402


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def question_id_from_path(path):
    name = Path(path).name
    for suffix in (
        "_grading_checkpoint.json",
        "_graded_results.json",
        "_rejected.json",
        "_failed.json",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def resolve_path(path, config_path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root_candidate = ROOT / candidate
    if root_candidate.exists():
        return root_candidate
    return config_path.parent / candidate


def teacher_score(database, student_id, question_id):
    record = database.get(str(student_id), {})
    if isinstance(record, dict) and question_id in record:
        return record.get(question_id)
    return None


def parse_facts(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def auc(scores, labels):
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def best_budget_point(scores, unsafe_labels, max_bnd_ratio=0.60):
    candidates = []
    for threshold in sorted(set(scores)):
        positive_indexes = [
            index for index, score in enumerate(scores) if score <= threshold
        ]
        if not positive_indexes:
            continue
        bnd_ratio = 1.0 - len(positive_indexes) / len(scores)
        if bnd_ratio > max_bnd_ratio + 1e-12:
            continue
        unsafe_count = sum(unsafe_labels[index] for index in positive_indexes)
        candidates.append({
            "threshold": threshold,
            "pos_n": len(positive_indexes),
            "bnd_ratio": bnd_ratio,
            "unsafe_pos_rate": unsafe_count / len(positive_indexes),
        })
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (item["unsafe_pos_rate"], item["bnd_ratio"]),
    )


def load_questions(config, config_path):
    database_path = resolve_path(config["database_path"], config_path)
    questions = load_json(database_path)
    return {
        str(question["question_id"]): question
        for question in questions
        if isinstance(question, dict) and question.get("question_id")
    }


def load_rubric(question, config_path):
    rubric_path = (
        question.get("optimized_rubric_path")
        or question.get("initial_rubric_path")
    )
    if not rubric_path:
        raise ValueError(f"No rubric path for {question.get('question_id')}")
    rubric = load_json(resolve_path(rubric_path, config_path))
    return prepare_rubrics_for_calibration(rubric)


def build_rows(config, config_path):
    questions = load_questions(config, config_path)
    database_path = resolve_path(config["database_path"], config_path)
    teacher_path = resolve_path(config["teacher_db"], config_path)
    teacher_db = load_json(teacher_path)
    sample_policy = load_policy_for_data_path(str(database_path))
    membership_model = config.get("membership_model") or {}
    safe_error_ratio = safe_float(
        membership_model.get("safe_error_ratio", 0.10), 0.10
    )
    safe_error_points = safe_float(
        membership_model.get("safe_error_points", 0.50), 0.50
    )
    rubric_cache = {}
    output = []

    for input_path in config.get("files", []):
        path = resolve_path(input_path, config_path)
        question_id = question_id_from_path(path)
        if question_id not in questions:
            raise KeyError(f"Question missing from database: {question_id}")
        if question_id not in rubric_cache:
            rubric_cache[question_id] = load_rubric(
                questions[question_id], config_path
            )
        rubric = rubric_cache[question_id]
        max_score = max(
            safe_float(questions[question_id].get("total_score"), 0.0),
            1.0,
        )

        for record in load_json(path):
            student_id = str(record.get("student_id", ""))
            teacher = teacher_score(teacher_db, student_id, question_id)
            if teacher is not None:
                teacher = sample_policy.effective_teacher_score(
                    question_id, student_id, teacher
                )
            baseline = record.get(
                "selected_baseline_score", record.get("model_avg_score")
            )
            if teacher is None or baseline is None:
                continue
            facts = parse_facts(record.get("facts"))
            probes = record.get("strict_cots_all") or []
            risks = compute_bidirectional_credit_risks(facts, rubric, probes)
            post = record.get("post_calibration") or {}
            primary = (
                post.get("primary_risks")
                or post.get("three_way_primary_risks")
                or {}
            )
            tolerance = max(
                safe_error_points, safe_error_ratio * max_score
            )
            residual = safe_float(teacher) - safe_float(baseline)
            output.append({
                "question_id": question_id,
                "student_id": student_id,
                "teacher_score": safe_float(teacher),
                "baseline_score": safe_float(baseline),
                "tolerance": tolerance,
                "undercredit_unsafe": residual > tolerance,
                "overcredit_unsafe": residual < -tolerance,
                "unsafe": abs(residual) > tolerance,
                "U_E": safe_float(primary.get("U_E")),
                "U_S": safe_float(primary.get("U_S")),
                "U_R": safe_float(primary.get("U_R")),
                "U_R_undercredit_existing": safe_float(
                    post.get("lenient_undercredit_signal")
                ),
                **{key: value for key, value in risks.items()
                   if key != "item_diagnostics"},
            })
    return output


def evaluate(rows):
    under_labels = [row["undercredit_unsafe"] for row in rows]
    over_labels = [row["overcredit_unsafe"] for row in rows]
    unsafe_labels = [row["unsafe"] for row in rows]
    feature_names = [
        "U_R_allocation_undercredit",
        "U_R_allocation_overcredit",
        "U_R_allocation_disagreement",
        "U_R_deterministic_undercredit",
        "U_R_deterministic_overcredit",
        "missing_judgement_risk",
    ]
    feature_report = {}
    for feature in feature_names:
        scores = [safe_float(row.get(feature)) for row in rows]
        target = under_labels if "undercredit" in feature else over_labels
        if feature in ("U_R_allocation_disagreement", "missing_judgement_risk"):
            target = unsafe_labels
        feature_report[feature] = {
            "auc": auc(scores, target),
            "nonzero": sum(score > 0.0 for score in scores),
        }

    combined_scores = [
        max(
            row["U_S"],
            row["U_R"],
            row["U_R_undercredit_existing"],
            row["U_R_allocation_undercredit"],
            row["U_R_allocation_overcredit"],
            row["U_R_deterministic_undercredit"],
            row["U_R_deterministic_overcredit"],
        )
        for row in rows
    ]
    return {
        "n": len(rows),
        "undercredit_unsafe_n": sum(under_labels),
        "overcredit_unsafe_n": sum(over_labels),
        "safe_n": len(rows) - sum(unsafe_labels),
        "feature_report": feature_report,
        "optimistic_combined": {
            "auc": auc(combined_scores, unsafe_labels),
            "best_budget_point": best_budget_point(
                combined_scores, unsafe_labels
            ),
        },
        "status": "diagnostic_only",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    rows = build_rows(config, config_path)
    report = evaluate(rows)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else config_path.parent / f"{config_path.stem}_credit_risk_diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "summary.json"
    csv_path = output_dir / "cases.csv"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    print(f"Cases: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
