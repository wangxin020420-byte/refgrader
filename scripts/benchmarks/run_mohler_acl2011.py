from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_datasets.contract import load_json, write_json
from benchmark_datasets.protocols.mohler_acl2011 import (
    build_mohler_acl2011_protocol,
    evaluate_prediction_rows,
    run_mohler_acl2011_baselines,
    write_protocol,
)
from scripts.benchmarks import run_benchmark as benchmark_runner


SCORE_FIELDS = {
    "single": "single_first_score",
    "avg": "model_avg_score",
    "selected": "selected_baseline_score",
    "3wd_core": "three_way_core_score",
    "3wd": "final_calibrated_score",
}


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _score(record: dict[str, Any], field: str) -> float:
    if field == "single_first_score":
        history = record.get("model_scores_history") or []
        value = history[0] if history else None
    else:
        value = record.get(field)
    if value is None:
        raise ValueError(
            f"Missing {field} for {record.get('student_id', '<unknown>')}"
        )
    return float(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _run_complete(run_dir: Path) -> bool:
    manifest = run_dir / "run_manifest.json"
    return (
        manifest.is_file()
        and load_json(manifest).get("status") == "complete"
        and (run_dir / "evaluation" / "summary.json").is_file()
    )


def _grading_complete(run_dir: Path) -> bool:
    manifest = run_dir / "run_manifest.json"
    return manifest.is_file() and load_json(manifest).get("status") == "complete"


def _fold_manifest(
    run_dir: Path,
    *,
    protocol: dict[str, Any],
    fold: dict[str, Any],
    variant: str,
) -> None:
    write_json(
        run_dir / "acl2011_fold_manifest.json",
        {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "variant": variant,
            "fold": fold,
            "paper_compatibility": protocol["paper_compatibility"],
        },
    )


def _publish_if_requested(
    context: dict[str, Any],
    *,
    run_id: str,
    artifacts_repo: str | None,
) -> None:
    if not artifacts_repo:
        return
    target = (
        Path(artifacts_repo).expanduser().resolve()
        / "public_benchmarks"
        / str(context["manifest"]["dataset_id"])
        / "runs"
        / run_id
    )
    if target.exists():
        print(f"Artifact already exists; leaving unchanged: {target}")
        return
    published = benchmark_runner.publish_run(
        context,
        run_id=run_id,
        artifacts_repo=artifacts_repo,
        force=False,
    )
    print(f"Published: {published}")


def run_refgrader_folds(
    prepared_dir: str | Path,
    protocol: dict[str, Any],
    *,
    tag: str,
    variant: str,
    a3wa_config: str | None,
    allow_experimental_a3wa: bool,
    start_fold: int,
    end_fold: int,
    force: bool,
    dry_run: bool,
    artifacts_repo: str | None,
) -> int:
    if variant == "zero_shot" and not a3wa_config:
        raise ValueError("--a3wa-config is required for the zero_shot variant.")
    context = benchmark_runner._prepared_context(prepared_dir)
    for fold_number in range(start_fold, end_fold + 1):
        fold = protocol["folds"][fold_number - 1]
        fold_tag = f"{tag}_{fold['fold_id']}"
        config = a3wa_config
        if variant == "fold_calibrated":
            calibration_run_id = f"{fold_tag}_calibration"
            calibration_questions = list(fold["calibration_question_ids"])
            calibration_run_dir = benchmark_runner._run_dir(
                str(context["manifest"]["dataset_id"]), calibration_run_id
            )
            config_path = (
                benchmark_runner._result_root(str(context["manifest"]["dataset_id"]))
                / "calibration"
                / f"{fold_tag}_a3wa.json"
            )
            if dry_run or force or not _grading_complete(calibration_run_dir):
                code = benchmark_runner.grade(
                    context,
                    questions=calibration_questions,
                    split="all",
                    run_id=calibration_run_id,
                    force=force,
                    a3wa_config=None,
                    dry_run=dry_run,
                    evaluate_after=False,
                )
                if code != 0:
                    return code
            else:
                print(
                    "Calibration grading already complete; skipping: "
                    f"{fold['fold_id']}"
                )
            if dry_run or force or not config_path.is_file():
                code, config_path = benchmark_runner.calibrate(
                    context,
                    questions=calibration_questions,
                    validation_run_id=calibration_run_id,
                    output=str(config_path),
                    score_calibration=True,
                    dry_run=dry_run,
                )
                if code != 0:
                    return code
            else:
                print(f"Fold calibration already exists; skipping: {config_path}")
            config = str(config_path)

        test_run_id = f"{fold_tag}_test"
        test_run_dir = benchmark_runner._run_dir(
            str(context["manifest"]["dataset_id"]), test_run_id
        )
        if not force and _run_complete(test_run_dir):
            print(f"Fold already complete; skipping: {fold['fold_id']}")
            continue
        code = benchmark_runner.grade(
            context,
            questions=list(fold["test_question_ids"]),
            split="all",
            run_id=test_run_id,
            force=force,
            a3wa_config=config,
            dry_run=dry_run,
            evaluate_after=False,
            allow_experimental_a3wa=allow_experimental_a3wa,
        )
        if code != 0:
            return code
        if dry_run:
            continue
        code = benchmark_runner.evaluate_run(
            context,
            questions=list(fold["test_question_ids"]),
            run_id=test_run_id,
            dry_run=False,
        )
        if code != 0:
            return code
        _fold_manifest(
            test_run_dir,
            protocol=protocol,
            fold=fold,
            variant=variant,
        )
        _publish_if_requested(
            context,
            run_id=test_run_id,
            artifacts_repo=artifacts_repo,
        )
    return 0


def summarize_refgrader_folds(
    prepared_dir: str | Path,
    protocol: dict[str, Any],
    *,
    tag: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    context = benchmark_runner._prepared_context(prepared_dir)
    teacher_scores = load_json(context["root"] / "teacher_scores.json")
    rows: list[dict[str, Any]] = []
    for fold in protocol["folds"]:
        run_id = f"{tag}_{fold['fold_id']}_test"
        run_dir = benchmark_runner._run_dir(
            str(context["manifest"]["dataset_id"]), run_id
        )
        if not _run_complete(run_dir):
            raise FileNotFoundError(f"Fold run is incomplete: {run_dir}")
        for question_id in fold["test_question_ids"]:
            checkpoint = load_json(
                run_dir / f"{question_id}_grading_checkpoint.json"
            )
            for record in checkpoint:
                student_id = str(record["student_id"])
                teacher = teacher_scores.get(student_id, {}).get(question_id)
                if teacher is None:
                    raise ValueError(
                        f"Missing teacher score for {question_id}/{student_id}"
                    )
                row: dict[str, Any] = {
                    "fold_id": fold["fold_id"],
                    "test_unit": fold["test_unit"],
                    "question_id": question_id,
                    "student_id": student_id,
                    "teacher_score": float(teacher),
                    "route": str(record.get("3wd_route", "")),
                }
                for label, field in SCORE_FIELDS.items():
                    row[label] = _score(record, field)
                rows.append(row)

    student_ids = [str(row["student_id"]) for row in rows]
    expected = int(protocol["answer_count"])
    if len(student_ids) != expected or len(student_ids) != len(set(student_ids)):
        raise ValueError(
            "RefGrader fold coverage is invalid: "
            f"expected={expected}, rows={len(student_ids)}, "
            f"unique={len(set(student_ids))}"
        )
    evaluation = evaluate_prediction_rows(rows, score_fields=SCORE_FIELDS)
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "comparison_scope": protocol["paper_compatibility"]["status"],
        "run_tag": tag,
        "route_counts": dict(Counter(str(row["route"]) for row in rows)),
        **evaluation,
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_protocol(protocol, output / "protocol.json")
    _write_csv(output / "predictions.csv", rows)
    _write_csv(output / "per_question_metrics.csv", evaluation["per_question"])
    write_json(output / "summary.json", summary)
    return summary


def _protocol_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return build_mohler_acl2011_protocol(
        args.prepared_dir,
        excluded_question_ids=args.exclude_question,
        require_paper_question_count=args.require_paper_question_count,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the assignment/exam-level Mohler ACL 2011 protocol."
    )
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--exclude-question", action="append", default=[])
    parser.add_argument("--require-paper-question-count", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--output")

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--output-dir", required=True)
    baseline.add_argument("--random-state", type=int, default=2011)

    refgrader = subparsers.add_parser("refgrader")
    refgrader.add_argument("--tag", required=True)
    refgrader.add_argument(
        "--variant",
        choices=["zero_shot", "fold_calibrated"],
        default="zero_shot",
    )
    refgrader.add_argument("--a3wa-config")
    refgrader.add_argument("--allow-experimental-a3wa", action="store_true")
    refgrader.add_argument("--start-fold", type=int, default=1)
    refgrader.add_argument("--end-fold", type=int, default=12)
    refgrader.add_argument("--force", action="store_true")
    refgrader.add_argument("--dry-run", action="store_true")
    refgrader.add_argument("--artifacts-repo")

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--tag", required=True)
    summarize.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    _configure_utf8_stdio()
    args = build_parser().parse_args()
    protocol = _protocol_from_args(args)
    if args.command == "audit":
        if args.output:
            target = write_protocol(protocol, args.output)
            print(f"Protocol: {target}")
        print(
            json.dumps(
                {
                    "protocol_id": protocol["protocol_id"],
                    "question_count": protocol["question_count"],
                    "answer_count": protocol["answer_count"],
                    "fold_count": protocol["integrity"]["fold_count"],
                    "all_answers_tested_once": protocol["integrity"][
                        "all_answers_tested_once"
                    ],
                    "paper_compatibility": protocol["paper_compatibility"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "baseline":
        summary = run_mohler_acl2011_baselines(
            args.prepared_dir,
            protocol,
            args.output_dir,
            random_state=args.random_state,
        )
        print(json.dumps(summary["global"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "refgrader":
        if not 1 <= args.start_fold <= args.end_fold <= 12:
            raise ValueError("Fold range must satisfy 1 <= start <= end <= 12.")
        return run_refgrader_folds(
            args.prepared_dir,
            protocol,
            tag=args.tag,
            variant=args.variant,
            a3wa_config=args.a3wa_config,
            allow_experimental_a3wa=args.allow_experimental_a3wa,
            start_fold=args.start_fold,
            end_fold=args.end_fold,
            force=args.force,
            dry_run=args.dry_run,
            artifacts_repo=args.artifacts_repo,
        )
    summary = summarize_refgrader_folds(
        args.prepared_dir,
        protocol,
        tag=args.tag,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary["global"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
