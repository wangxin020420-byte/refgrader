from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCORE_TYPES = (
    "single",
    "avg",
    "selected",
    "3WD-Core",
    "3WD",
)
SCORE_COLUMNS = {
    "single": "single_first_score",
    "avg": "model_avg_score",
    "selected": "selected_baseline_score",
    "3WD-Core": "three_way_core_score",
    "3WD": "final_calibrated_score",
}
ABS_ERROR_COLUMNS = {
    "single": "single_abs_error",
    "avg": "avg_abs_error",
    "selected": "selected_abs_error",
    "3WD-Core": "core_abs_error",
    "3WD": "final_abs_error",
}
METRIC_FIELDS = ("MAE", "RMSE", "Pearson", "TAR2", "SER2", "bias")
ABLATION_COMPONENTS = {
    "three_way_core": ("avg_abs_error", "core_abs_error"),
    "validation_residual": ("core_abs_error", "final_abs_error"),
    "full_three_way": ("avg_abs_error", "final_abs_error"),
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value for {field}: {value!r}")
    return result


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _numeric_summary(values: Iterable[float]) -> dict[str, float]:
    numbers = list(values)
    if not numbers:
        raise ValueError("Cannot summarize an empty numeric sequence")
    return {
        "mean": statistics.fmean(numbers),
        "sample_std": _sample_std(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _load_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "evaluation" / "summary.json"
    compare_path = run_dir / "evaluation" / "compare.csv"
    for path in (manifest_path, summary_path, compare_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required repeat-run file is missing: {path}")

    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(
            f"Run {run_dir.name} is not complete: {manifest.get('status')!r}"
        )
    if manifest.get("split") != "test":
        raise ValueError(
            f"Run {run_dir.name} is not a test run: {manifest.get('split')!r}"
        )

    summary = _read_json(summary_path)
    global_metrics = {
        str(item.get("score_type")): item
        for item in summary.get("global", [])
        if isinstance(item, dict)
    }
    missing_score_types = [
        score_type for score_type in SCORE_TYPES if score_type not in global_metrics
    ]
    if missing_score_types:
        raise ValueError(
            f"Run {run_dir.name} is missing global score types: "
            f"{missing_score_types}"
        )

    rows: dict[tuple[str, str], dict[str, str]] = {}
    with compare_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("question", "")), str(row.get("student_id", "")))
            if not all(key):
                raise ValueError(f"Run {run_dir.name} contains an empty sample key")
            if key in rows:
                raise ValueError(f"Run {run_dir.name} contains duplicate sample {key}")
            for column in (*SCORE_COLUMNS.values(), *ABS_ERROR_COLUMNS.values()):
                _float(row.get(column), field=f"{run_dir.name}:{key}:{column}")
            rows[key] = row

    expected_n = int(global_metrics["avg"].get("n", -1))
    if expected_n != len(rows):
        raise ValueError(
            f"Run {run_dir.name} global N={expected_n} but compare.csv has "
            f"{len(rows)} rows"
        )

    dataset_snapshot = manifest.get("dataset_snapshot") or {}
    model_config = manifest.get("model_config") or {}
    return {
        "run_id": str(manifest.get("run_id") or run_dir.name),
        "run_dir": run_dir,
        "manifest": manifest,
        "summary": summary,
        "global_metrics": global_metrics,
        "rows": rows,
        "contract": {
            "dataset_id": manifest.get("dataset_id"),
            "dataset_content_sha256": dataset_snapshot.get(
                "prepared_content_sha256"
            ),
            "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
            "split": manifest.get("split"),
            "questions": list(manifest.get("questions") or []),
            "model_config": model_config,
            "a3wa_config_sha256": manifest.get("a3wa_config_sha256"),
            "a3wa_deployment_class": manifest.get("a3wa_deployment_class"),
        },
    }


def _validate_contract(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("At least two complete repeat runs are required")
    reference = runs[0]
    contract_fields = (
        "dataset_id",
        "dataset_content_sha256",
        "dataset_manifest_sha256",
        "split",
        "questions",
        "model_config",
        "a3wa_config_sha256",
        "a3wa_deployment_class",
    )
    for run in runs[1:]:
        for field in contract_fields:
            if run["contract"].get(field) != reference["contract"].get(field):
                raise ValueError(
                    f"Repeat contract mismatch for {field}: "
                    f"{reference['run_id']} != {run['run_id']}"
                )
        if set(run["rows"]) != set(reference["rows"]):
            missing = sorted(set(reference["rows"]) - set(run["rows"]))[:5]
            extra = sorted(set(run["rows"]) - set(reference["rows"]))[:5]
            raise ValueError(
                f"Repeat sample mismatch for {run['run_id']}: "
                f"missing={missing}, extra={extra}"
            )
        for key, row in run["rows"].items():
            reference_teacher = _float(
                reference["rows"][key].get("teacher"), field=f"{key}:teacher"
            )
            teacher = _float(row.get("teacher"), field=f"{key}:teacher")
            if teacher != reference_teacher:
                raise ValueError(
                    f"Teacher label mismatch for {key}: "
                    f"{reference_teacher} != {teacher}"
                )
    return reference["contract"]


def _run_metric_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run in runs:
        for score_type in SCORE_TYPES:
            metric = run["global_metrics"][score_type]
            records.append(
                {
                    "run_id": run["run_id"],
                    "score_type": score_type,
                    "n": int(metric["n"]),
                    **{field: metric.get(field) for field in METRIC_FIELDS},
                }
            )
    return records


def _aggregate_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for score_type in SCORE_TYPES:
        result[score_type] = {}
        for field in METRIC_FIELDS:
            values = [
                _float(run["global_metrics"][score_type][field], field=field)
                for run in runs
                if run["global_metrics"][score_type].get(field) is not None
            ]
            result[score_type][field] = (
                _numeric_summary(values) if values else None
            )
    return result


def _ablation_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run in runs:
        rows = list(run["rows"].values())
        for component, (baseline_column, candidate_column) in (
            ABLATION_COMPONENTS.items()
        ):
            gains = [
                _float(row[baseline_column], field=baseline_column)
                - _float(row[candidate_column], field=candidate_column)
                for row in rows
            ]
            records.append(
                {
                    "run_id": run["run_id"],
                    "component": component,
                    "n": len(gains),
                    "mean_gain": statistics.fmean(gains),
                    "improved": sum(value > 1e-12 for value in gains),
                    "unchanged": sum(abs(value) <= 1e-12 for value in gains),
                    "worsened": sum(value < -1e-12 for value in gains),
                }
            )
    return records


def _aggregate_ablation(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for component in ABLATION_COMPONENTS:
        selected = [row for row in records if row["component"] == component]
        result[component] = {
            "mean_gain": _numeric_summary(
                _float(row["mean_gain"], field="mean_gain") for row in selected
            ),
            "per_run": selected,
        }
    return result


def _route_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    per_run = []
    for run in runs:
        rows = list(run["rows"].values())
        route_counts = Counter(row.get("route", "") for row in rows)
        serious = sum(_bool(row.get("baseline_serious_error")) for row in rows)
        captured = sum(_bool(row.get("risk_captured_by_route")) for row in rows)
        pos = [row for row in rows if row.get("route") == "POS"]
        safe_pos = sum(_bool(row.get("safe_pos")) for row in pos)
        per_run.append(
            {
                "run_id": run["run_id"],
                "n": len(rows),
                "route_counts": dict(sorted(route_counts.items())),
                "route_rates": {
                    route: count / len(rows)
                    for route, count in sorted(route_counts.items())
                },
                "baseline_serious_errors": serious,
                "risk_captured": captured,
                "risk_recall": captured / serious if serious else None,
                "safe_pos": safe_pos,
                "unsafe_pos": len(pos) - safe_pos,
                "safe_pos_rate": safe_pos / len(pos) if pos else None,
            }
        )
    route_names = sorted(
        {route for item in per_run for route in item["route_counts"]}
    )
    return {
        "per_run": per_run,
        "aggregate_route_rates": {
            route: _numeric_summary(
                item["route_rates"].get(route, 0.0) for item in per_run
            )
            for route in route_names
        },
        "risk_recall": _numeric_summary(
            item["risk_recall"]
            for item in per_run
            if item["risk_recall"] is not None
        ),
        "safe_pos_rate": _numeric_summary(
            item["safe_pos_rate"]
            for item in per_run
            if item["safe_pos_rate"] is not None
        ),
    }


def _sample_stability_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run_ids = [run["run_id"] for run in runs]
    sample_keys = sorted(runs[0]["rows"])
    records: list[dict[str, Any]] = []
    for question, student_id in sample_keys:
        key = (question, student_id)
        record: dict[str, Any] = {
            "question": question,
            "student_id": student_id,
            "teacher": _float(runs[0]["rows"][key]["teacher"], field="teacher"),
        }
        for score_type in ("avg", "3WD-Core", "3WD"):
            column = SCORE_COLUMNS[score_type]
            values = [
                _float(run["rows"][key][column], field=column) for run in runs
            ]
            slug = score_type.lower().replace("-", "_")
            record[f"{slug}_mean"] = statistics.fmean(values)
            record[f"{slug}_sample_std"] = _sample_std(values)
            record[f"{slug}_min"] = min(values)
            record[f"{slug}_max"] = max(values)
            record[f"{slug}_exact"] = max(values) == min(values)
            for run_id, value in zip(run_ids, values):
                record[f"{run_id}:{slug}"] = value
        routes = [run["rows"][key].get("route", "") for run in runs]
        actions = [run["rows"][key].get("boundary_action", "") for run in runs]
        record["route_exact"] = len(set(routes)) == 1
        record["boundary_action_exact"] = len(set(actions)) == 1
        for run_id, route, action in zip(run_ids, routes, actions):
            record[f"{run_id}:route"] = route
            record[f"{run_id}:boundary_action"] = action
        records.append(record)
    return records


def _sample_stability_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for score_type in ("avg", "3WD-Core", "3WD"):
        slug = score_type.lower().replace("-", "_")
        deviations = [
            _float(row[f"{slug}_sample_std"], field=f"{slug}_sample_std")
            for row in records
        ]
        maximum = max(records, key=lambda row: row[f"{slug}_sample_std"])
        exact = sum(bool(row[f"{slug}_exact"]) for row in records)
        result[score_type] = {
            "exact_count": exact,
            "exact_rate": exact / len(records),
            "mean_sample_std": statistics.fmean(deviations),
            "max_sample_std": max(deviations),
            "max_instability_sample": {
                "question": maximum["question"],
                "student_id": maximum["student_id"],
                "teacher": maximum["teacher"],
            },
        }
    route_exact = sum(bool(row["route_exact"]) for row in records)
    action_exact = sum(bool(row["boundary_action_exact"]) for row in records)
    result["route"] = {
        "exact_count": route_exact,
        "exact_rate": route_exact / len(records),
    }
    result["boundary_action"] = {
        "exact_count": action_exact,
        "exact_rate": action_exact / len(records),
    }
    return result


def _question_stability_rows(
    runs: list[dict[str, Any]], sample_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    questions = sorted({row["question"] for row in sample_records})
    records: list[dict[str, Any]] = []
    for question in questions:
        sample_keys = [
            key for key in runs[0]["rows"] if key[0] == question
        ]
        record: dict[str, Any] = {"question": question, "n": len(sample_keys)}
        for score_type in ("avg", "3WD-Core", "3WD"):
            error_column = ABS_ERROR_COLUMNS[score_type]
            per_run_mae = [
                statistics.fmean(
                    _float(run["rows"][key][error_column], field=error_column)
                    for key in sample_keys
                )
                for run in runs
            ]
            slug = score_type.lower().replace("-", "_")
            record[f"{slug}_mae_mean"] = statistics.fmean(per_run_mae)
            record[f"{slug}_mae_sample_std"] = _sample_std(per_run_mae)
            record[f"{slug}_mae_range"] = max(per_run_mae) - min(per_run_mae)
        matching_samples = [
            row for row in sample_records if row["question"] == question
        ]
        record["final_score_mean_sample_std"] = statistics.fmean(
            row["3wd_sample_std"] for row in matching_samples
        )
        records.append(record)
    return records


def _cluster_bootstrap(
    runs: list[dict[str, Any]],
    *,
    baseline_column: str,
    candidate_column: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive")
    questions = sorted({key[0] for key in runs[0]["rows"]})
    gains_by_question: dict[str, list[float]] = {}
    for question in questions:
        gains = []
        sample_keys = [key for key in runs[0]["rows"] if key[0] == question]
        for run in runs:
            for key in sample_keys:
                row = run["rows"][key]
                gains.append(
                    _float(row[baseline_column], field=baseline_column)
                    - _float(row[candidate_column], field=candidate_column)
                )
        gains_by_question[question] = gains

    observed = statistics.fmean(
        value for gains in gains_by_question.values() for value in gains
    )
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        selected = [rng.choice(questions) for _ in questions]
        values = [
            value for question in selected for value in gains_by_question[question]
        ]
        estimates.append(statistics.fmean(values))
    estimates.sort()
    return {
        "unit": "question_cluster_with_repeated_run_measurements",
        "question_count": len(questions),
        "run_count": len(runs),
        "iterations": iterations,
        "seed": seed,
        "observed_mean_gain": observed,
        "ci95": [
            _percentile(estimates, 0.025),
            _percentile(estimates, 0.975),
        ],
        "probability_gain_positive": sum(value > 0 for value in estimates)
        / len(estimates),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(summary: dict[str, Any]) -> str:
    metrics = summary["aggregate_metrics"]
    ablation = summary["ablation"]
    routes = summary["routes"]
    stability = summary["sample_stability"]
    bootstrap = summary["cluster_bootstrap"]
    lines = [
        "# Repeated Public Benchmark Summary",
        "",
        f"Runs: {len(summary['run_ids'])}",
        f"Samples per run: {summary['sample_count']}",
        f"Deployment class: {summary['contract']['a3wa_deployment_class']}",
        "",
        "## Global metrics",
        "",
        "| Score | MAE mean | MAE SD | RMSE mean | SER mean | Bias mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for score_type in SCORE_TYPES:
        item = metrics[score_type]
        lines.append(
            f"| {score_type} | {item['MAE']['mean']:.6f} | "
            f"{item['MAE']['sample_std']:.6f} | "
            f"{item['RMSE']['mean']:.6f} | {item['SER2']['mean']:.2%} | "
            f"{item['bias']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism ablation",
            "",
            "| Component | Mean gain | SD | 95% cluster CI | P(gain > 0) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for component in ("three_way_core", "validation_residual", "full_three_way"):
        item = ablation[component]["mean_gain"]
        boot = bootstrap[component]
        lines.append(
            f"| {component} | {item['mean']:.6f} | "
            f"{item['sample_std']:.6f} | "
            f"[{boot['ci95'][0]:.6f}, {boot['ci95'][1]:.6f}] | "
            f"{boot['probability_gain_positive']:.4f} |"
        )
    route_rates = routes["aggregate_route_rates"]
    lines.extend(["", "## Routing and stability", ""])
    for route in sorted(route_rates):
        lines.append(
            f"- {route} mean rate: {route_rates[route]['mean']:.2%}"
        )
    lines.extend(
        [
            f"- Risk recall: {routes['risk_recall']['mean']:.2%}",
            f"- Safe POS rate: {routes['safe_pos_rate']['mean']:.2%}",
            f"- Final score exact across runs: "
            f"{stability['3WD']['exact_rate']:.2%}",
            f"- Route exact across runs: {stability['route']['exact_rate']:.2%}",
            f"- Boundary action exact across runs: "
            f"{stability['boundary_action']['exact_rate']:.2%}",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate_repeats(
    runs_root: str | Path,
    run_ids: list[str],
    output_dir: str | Path,
    *,
    bootstrap_iterations: int = 20000,
    seed: int = 20260808,
) -> dict[str, Any]:
    root = Path(runs_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Runs root not found: {root}")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("--run-ids contains duplicates")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Use a new report ID."
        )

    runs = [_load_run(root / run_id) for run_id in run_ids]
    contract = _validate_contract(runs)
    run_metrics = _run_metric_rows(runs)
    ablation_rows = _ablation_rows(runs)
    sample_rows = _sample_stability_rows(runs)
    question_rows = _question_stability_rows(runs, sample_rows)
    summary = {
        "schema_version": 1,
        "run_ids": [run["run_id"] for run in runs],
        "run_count": len(runs),
        "sample_count": len(runs[0]["rows"]),
        "contract": contract,
        "aggregate_metrics": _aggregate_metrics(runs),
        "ablation": _aggregate_ablation(ablation_rows),
        "routes": _route_summary(runs),
        "sample_stability": _sample_stability_summary(sample_rows),
        "cluster_bootstrap": {
            component: _cluster_bootstrap(
                runs,
                baseline_column=columns[0],
                candidate_column=columns[1],
                iterations=bootstrap_iterations,
                seed=seed + index,
            )
            for index, (component, columns) in enumerate(
                ABLATION_COMPONENTS.items()
            )
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "repeat_summary.json", summary)
    _write_csv(output / "run_metrics.csv", run_metrics)
    _write_csv(output / "ablation_by_run.csv", ablation_rows)
    _write_csv(output / "sample_stability.csv", sample_rows)
    _write_csv(output / "question_stability.csv", question_rows)
    (output / "report.md").write_text(
        _markdown_report(summary), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate complete repeated public benchmark test runs without "
            "rerunning grading or calibration."
        )
    )
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = aggregate_repeats(
        args.runs_root,
        args.run_ids,
        args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    print(f"Repeated runs: {summary['run_count']}")
    print(f"Samples per run: {summary['sample_count']}")
    print(
        "3WD-Core mean gain: "
        f"{summary['ablation']['three_way_core']['mean_gain']['mean']:.6f}"
    )
    print(
        "Final 3WD mean gain: "
        f"{summary['ablation']['full_three_way']['mean_gain']['mean']:.6f}"
    )
    print(f"Report: {Path(args.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
