from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_datasets.contract import load_json, write_json


SCORE_FIELDS = ("single", "avg", "selected", "3wd_core", "3wd")
COMPARISONS = (
    ("avg_to_3wd_core", "avg", "3wd_core"),
    ("3wd_core_to_3wd", "3wd_core", "3wd"),
    ("avg_to_3wd", "avg", "3wd"),
)
METRICS = ("MAE", "RMSE")


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "question_id",
            "student_id",
            "teacher_score",
            *SCORE_FIELDS,
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("Prediction CSV is empty.")

    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["question_id"]), str(row["student_id"]))
        if key in seen:
            raise ValueError(f"Duplicate prediction row: {key[0]}/{key[1]}")
        seen.add(key)
        for field in ("teacher_score", *SCORE_FIELDS):
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {field} for {key[0]}/{key[1]}")
            row[field] = value
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _quantize_half_point(value: float) -> int:
    clipped = min(5.0, max(0.0, float(value)))
    return int(math.floor(clipped * 2.0 + 0.5))


def quadratic_weighted_kappa(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> float | None:
    left = [_quantize_half_point(value) for value in actual]
    right = [_quantize_half_point(value) for value in predicted]
    if len(left) != len(right) or not left:
        raise ValueError("QWK vectors must be non-empty and equally sized.")

    category_count = 11
    observed = [[0.0] * category_count for _ in range(category_count)]
    actual_hist = [0.0] * category_count
    predicted_hist = [0.0] * category_count
    for truth, estimate in zip(left, right):
        observed[truth][estimate] += 1.0
        actual_hist[truth] += 1.0
        predicted_hist[estimate] += 1.0

    count = float(len(left))
    observed_cost = 0.0
    expected_cost = 0.0
    denominator = float((category_count - 1) ** 2)
    for truth in range(category_count):
        for estimate in range(category_count):
            weight = ((truth - estimate) ** 2) / denominator
            observed_cost += weight * observed[truth][estimate]
            expected = actual_hist[truth] * predicted_hist[estimate] / count
            expected_cost += weight * expected
    if expected_cost <= 0.0:
        return None
    return 1.0 - observed_cost / expected_cost


def _qwk_analysis(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)

    per_question: list[dict[str, Any]] = []
    for question_id in sorted(grouped):
        question_rows = grouped[question_id]
        item: dict[str, Any] = {
            "question_id": question_id,
            "n": len(question_rows),
        }
        actual = [float(row["teacher_score"]) for row in question_rows]
        for score_field in SCORE_FIELDS:
            item[f"{score_field}_qwk"] = quadratic_weighted_kappa(
                actual,
                [float(row[score_field]) for row in question_rows],
            )
        per_question.append(item)

    aggregate: dict[str, Any] = {
        "quantization": "clip_0_5_round_nearest_0.5_then_encode_0_10",
        "aggregation": "sample_weighted_mean_of_defined_per_question_qwk",
        "methods": {},
    }
    common_defined = [
        row
        for row in per_question
        if all(row[f"{score_field}_qwk"] is not None for score_field in SCORE_FIELDS)
    ]
    common_weight = sum(int(row["n"]) for row in common_defined)
    aggregate["common_defined_question_count"] = len(common_defined)
    aggregate["common_defined_sample_count"] = common_weight
    for score_field in SCORE_FIELDS:
        key = f"{score_field}_qwk"
        defined = [row for row in per_question if row[key] is not None]
        weight = sum(int(row["n"]) for row in defined)
        weighted = (
            sum(float(row[key]) * int(row["n"]) for row in defined) / weight
            if weight
            else None
        )
        aggregate["methods"][score_field] = {
            "sample_weighted_per_question_qwk": weighted,
            "sample_weighted_common_question_qwk": (
                sum(
                    float(row[key]) * int(row["n"])
                    for row in common_defined
                )
                / common_weight
                if common_weight
                else None
            ),
            "defined_question_count": len(defined),
            "undefined_question_count": len(per_question) - len(defined),
            "defined_sample_count": weight,
            "global_qwk": quadratic_weighted_kappa(
                [float(row["teacher_score"]) for row in rows],
                [float(row[score_field]) for row in rows],
            ),
        }
    return per_question, aggregate


def _metric(total_absolute: float, total_squared: float, count: int, name: str) -> float:
    if name == "MAE":
        return total_absolute / count
    if name == "RMSE":
        return math.sqrt(total_squared / count)
    raise ValueError(f"Unsupported metric: {name}")


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    if iterations < 100:
        raise ValueError("At least 100 bootstrap iterations are required.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    questions = sorted(grouped)

    sufficient: dict[str, dict[str, tuple[float, float, int]]] = {}
    for question_id, question_rows in grouped.items():
        sufficient[question_id] = {}
        for score_field in SCORE_FIELDS:
            errors = [
                float(row[score_field]) - float(row["teacher_score"])
                for row in question_rows
            ]
            sufficient[question_id][score_field] = (
                sum(abs(value) for value in errors),
                sum(value * value for value in errors),
                len(errors),
            )

    rng = random.Random(seed)
    samples: dict[tuple[str, str], list[float]] = {
        (comparison_id, metric): []
        for comparison_id, _, _ in COMPARISONS
        for metric in METRICS
    }
    for _ in range(iterations):
        selected = [rng.choice(questions) for _ in questions]
        for comparison_id, source, target in COMPARISONS:
            source_absolute = source_squared = 0.0
            target_absolute = target_squared = 0.0
            count = 0
            for question_id in selected:
                source_item = sufficient[question_id][source]
                target_item = sufficient[question_id][target]
                source_absolute += source_item[0]
                source_squared += source_item[1]
                target_absolute += target_item[0]
                target_squared += target_item[1]
                count += source_item[2]
            for metric in METRICS:
                gain = _metric(
                    source_absolute, source_squared, count, metric
                ) - _metric(target_absolute, target_squared, count, metric)
                samples[(comparison_id, metric)].append(gain)

    output: list[dict[str, Any]] = []
    for comparison_id, source, target in COMPARISONS:
        for metric in METRICS:
            source_absolute = sum(sufficient[q][source][0] for q in questions)
            source_squared = sum(sufficient[q][source][1] for q in questions)
            target_absolute = sum(sufficient[q][target][0] for q in questions)
            target_squared = sum(sufficient[q][target][1] for q in questions)
            count = sum(sufficient[q][source][2] for q in questions)
            observed = _metric(
                source_absolute, source_squared, count, metric
            ) - _metric(target_absolute, target_squared, count, metric)
            question_gains = []
            for question_id in questions:
                source_item = sufficient[question_id][source]
                target_item = sufficient[question_id][target]
                question_gains.append(
                    _metric(*source_item, metric)
                    - _metric(*target_item, metric)
                )
            estimates = samples[(comparison_id, metric)]
            low = _percentile(estimates, 0.025)
            high = _percentile(estimates, 0.975)
            output.append(
                {
                    "comparison": comparison_id,
                    "source_score": source,
                    "target_score": target,
                    "metric": metric,
                    "n": count,
                    "question_count": len(questions),
                    "observed_gain": observed,
                    "ci95_low": low,
                    "ci95_high": high,
                    "probability_gain_gt_zero": statistics.fmean(
                        value > 0.0 for value in estimates
                    ),
                    "significant_positive_95": low > 0.0,
                    "question_improved": sum(value > 1e-12 for value in question_gains),
                    "question_unchanged": sum(abs(value) <= 1e-12 for value in question_gains),
                    "question_worsened": sum(value < -1e-12 for value in question_gains),
                    "bootstrap_iterations": iterations,
                    "seed": seed,
                    "cluster_unit": "question_id",
                }
            )
    return output


def _paper_comparison_rows(
    protocol: dict[str, Any],
    predictions: list[dict[str, Any]],
    baseline_summary: dict[str, Any] | None,
    *,
    deployment_class: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from benchmark_datasets.protocols.mohler_acl2011 import evaluate_prediction_rows

    current = evaluate_prediction_rows(predictions, score_fields=SCORE_FIELDS)["global"]
    rows: list[dict[str, Any]] = []
    for method in SCORE_FIELDS:
        for metric in ("MAE", "RMSE", "Pearson", "SER2", "bias", "median_per_question_RMSE"):
            rows.append(
                {
                    "source": "current_refgrader",
                    "method": method,
                    "metric": metric,
                    "value": current[method].get(metric),
                    "question_count": protocol["question_count"],
                    "comparison_status": deployment_class,
                    "direct_paper_comparison": False,
                }
            )
    if baseline_summary:
        for method, metrics in baseline_summary.get("global", {}).items():
            for metric, value in metrics.items():
                if metric == "n":
                    continue
                rows.append(
                    {
                        "source": "local_reproducible_baseline",
                        "method": method,
                        "metric": metric,
                        "value": value,
                        "question_count": protocol["question_count"],
                        "comparison_status": "same_archive_protocol",
                        "direct_paper_comparison": False,
                    }
                )
    references = protocol["paper_reference_results"]
    for metric, value in (
        ("RMSE", references["average_grade_baseline_rmse"]),
        ("Pearson", references["best_pearson"]),
        ("RMSE", references["best_rmse"]),
        ("median_per_question_RMSE", references["best_median_per_question_rmse"]),
    ):
        rows.append(
            {
                "source": "acl2011_paper",
                "method": "reported_reference",
                "metric": metric,
                "value": value,
                "question_count": references["reported_question_count"],
                "comparison_status": "reference_only",
                "direct_paper_comparison": False,
            }
        )
    boundary = {
        "direct_paper_comparison_authorized": False,
        "deployment_class": deployment_class,
        "current_question_count": protocol["question_count"],
        "paper_reported_question_count": references["reported_question_count"],
        "reasons": [
            "The distributed archive contains 81 included questions while the paper reports 80.",
            "The current RefGrader run is zero-shot with a private-data A3WA configuration, while the paper trains supervised regressors inside its folds.",
            "The active A3WA configuration was used under an experimental deployment override.",
        ],
        "allowed_claim": "historical_reference_and_same_archive_baseline_comparison",
        "forbidden_claim": "exact_acl2011_reproduction_or_direct_superiority",
    }
    return rows, boundary


def _report_markdown(
    bootstrap: list[dict[str, Any]],
    qwk: dict[str, Any],
    boundary: dict[str, Any],
) -> str:
    lines = [
        "# Mohler ACL 2011 Statistical Analysis",
        "",
        f"Deployment class: `{boundary['deployment_class']}`",
        "",
        "## Paired question-cluster bootstrap",
        "",
        "| Comparison | Metric | Gain | 95% CI | P(gain > 0) | Significant |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in bootstrap:
        lines.append(
            "| {comparison} | {metric} | {observed_gain:.6f} | "
            "[{ci95_low:.6f}, {ci95_high:.6f}] | "
            "{probability_gain_gt_zero:.4f} | {significant_positive_95} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Positive gain means the target score has lower error. Questions are resampled as clusters.",
            "",
            "## Per-question QWK aggregate",
            "",
            "| Method | Common-question weighted QWK | Method-defined weighted QWK | Defined questions | Undefined questions |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method, item in qwk["methods"].items():
        common_value = item["sample_weighted_common_question_qwk"]
        method_value = item["sample_weighted_per_question_qwk"]
        common_display = "NA" if common_value is None else f"{common_value:.6f}"
        method_display = "NA" if method_value is None else f"{method_value:.6f}"
        lines.append(
            f"| {method} | {common_display} | {method_display} | "
            f"{item['defined_question_count']} | "
            f"{item['undefined_question_count']} |"
        )
    lines.extend(
        [
            "",
            "Scores are clipped to 0-5, rounded to the nearest 0.5, encoded as 0-10, and QWK is computed independently per question.",
            "",
            "## Comparison boundary",
            "",
            "Direct comparison with the ACL 2011 reported numbers is not authorized.",
        ]
    )
    lines.extend(f"- {reason}" for reason in boundary["reasons"])
    return "\n".join(lines) + "\n"


def analyze(
    *,
    predictions_path: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
    baseline_summary_path: str | Path | None = None,
    bootstrap_iterations: int = 10000,
    seed: int = 2011,
    deployment_class: str = "experimental_external_validation",
) -> dict[str, Any]:
    predictions = _read_predictions(Path(predictions_path).expanduser().resolve())
    protocol = load_json(Path(protocol_path).expanduser().resolve())
    questions = {str(row["question_id"]) for row in predictions}
    if len(predictions) != int(protocol["answer_count"]):
        raise ValueError(
            f"Prediction count mismatch: {len(predictions)} != {protocol['answer_count']}"
        )
    if len(questions) != int(protocol["question_count"]):
        raise ValueError(
            f"Question count mismatch: {len(questions)} != {protocol['question_count']}"
        )

    baseline_summary = (
        load_json(Path(baseline_summary_path).expanduser().resolve())
        if baseline_summary_path
        else None
    )
    per_question_qwk, qwk = _qwk_analysis(predictions)
    bootstrap = _cluster_bootstrap(
        predictions,
        iterations=bootstrap_iterations,
        seed=seed,
    )
    comparison, boundary = _paper_comparison_rows(
        protocol,
        predictions,
        baseline_summary,
        deployment_class=deployment_class,
    )
    result = {
        "schema_version": 1,
        "n": len(predictions),
        "question_count": len(questions),
        "bootstrap": bootstrap,
        "qwk": qwk,
        "comparison_boundary": boundary,
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "paired_question_cluster_bootstrap.csv", bootstrap)
    _write_csv(output / "per_question_qwk.csv", per_question_qwk)
    _write_csv(output / "paper_comparison.csv", comparison)
    write_json(output / "comparison_boundary.json", boundary)
    write_json(output / "summary.json", result)
    (output / "report.md").write_text(
        _report_markdown(bootstrap, qwk, boundary),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze completed Mohler ACL 2011 predictions without rerunning grading."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-summary")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2011)
    parser.add_argument(
        "--deployment-class",
        choices=("formal", "experimental_external_validation"),
        default="experimental_external_validation",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = analyze(
        predictions_path=args.predictions,
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        baseline_summary_path=args.baseline_summary,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        deployment_class=args.deployment_class,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
