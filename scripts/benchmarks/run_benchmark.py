from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_datasets.contract import (
    audit_prepared_benchmark,
    load_json,
    sha256_file,
    write_json,
)
from model_runtime import runtime_model_config


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_utf8_stdio()


PATH_FIELDS = (
    "question_image",
    "ref_image",
    "source_rubric_path",
    "initial_rubric_path",
    "optimized_rubric_path",
    "rubric_split_path",
    "student_images_dir",
)

SPLIT_SOURCE_KEYS = {"external_holdout": "train"}


def _source_split(split: str) -> str:
    return SPLIT_SOURCE_KEYS.get(split, split)


def _display(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> int:
    print(_display(command))
    if dry_run:
        return 0
    runtime_env = os.environ.copy()
    if env:
        runtime_env.update(env)
    runtime_env["PYTHONIOENCODING"] = "utf-8"
    runtime_env["PYTHONUTF8"] = "1"
    try:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=runtime_env,
        ).returncode
    except KeyboardInterrupt:
        print(
            "Benchmark interrupted; completed checkpoints remain resumable.",
            file=sys.stderr,
        )
        return 130


def _prepared_context(prepared_dir: str | Path) -> dict[str, Any]:
    root = Path(prepared_dir).expanduser().resolve()
    audit = audit_prepared_benchmark(root)
    manifest = load_json(root / "manifest.json")
    questions = load_json(root / "exam_database.json")
    return {
        "root": root,
        "audit": audit,
        "manifest": manifest,
        "questions": questions,
        "question_ids": [str(item["question_id"]) for item in questions],
    }


def _selected_questions(
    available: list[str], requested: list[str] | None
) -> list[str]:
    if not requested:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown prepared question IDs: {unknown}")
    return list(dict.fromkeys(requested))


def _runtime_database(
    context: dict[str, Any],
    run_dir: Path,
) -> Path:
    prepared_root: Path = context["root"]
    runtime_questions = []
    for question in context["questions"]:
        resolved = dict(question)
        for field in PATH_FIELDS:
            value = resolved.get(field)
            if not value:
                continue
            path = Path(str(value))
            if not path.is_absolute():
                path = prepared_root / path
            resolved[field] = str(path.resolve())
        runtime_questions.append(resolved)
    target = run_dir / "runtime_exam_database.json"
    write_json(target, runtime_questions)
    return target


def _snapshot_run_inputs(
    context: dict[str, Any],
    run_dir: Path,
    questions: list[str],
) -> None:
    prepared_root: Path = context["root"]
    snapshot_root = run_dir / "dataset_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "audit.json"):
        shutil.copy2(prepared_root / name, snapshot_root / name)
    adapter_spec = prepared_root / "source" / "adapter_spec.json"
    if adapter_spec.is_file():
        target = snapshot_root / "source" / "adapter_spec.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapter_spec, target)

    selected = [
        question
        for question in context["questions"]
        if str(question["question_id"]) in set(questions)
    ]
    write_json(snapshot_root / "exam_database.json", selected)
    for question in selected:
        for field in (
            "source_rubric_path",
            "initial_rubric_path",
            "optimized_rubric_path",
            "rubric_split_path",
        ):
            value = question.get(field)
            if not value:
                continue
            relative = Path(str(value))
            source = relative if relative.is_absolute() else prepared_root / relative
            if not source.is_file():
                raise FileNotFoundError(
                    f"Cannot snapshot {field} for {question['question_id']}: "
                    f"{source}"
                )
            target = (
                snapshot_root / relative
                if not relative.is_absolute()
                else snapshot_root / "external_inputs" / field / source.name
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _archive_a3wa_config(
    run_dir: Path,
    a3wa_config: str | None,
    *,
    dry_run: bool,
) -> Path | None:
    if not a3wa_config:
        return None
    source = Path(a3wa_config).expanduser().resolve()
    if not source.is_file():
        if dry_run:
            return source
        raise FileNotFoundError(f"A3WA configuration not found: {source}")
    target = run_dir / "calibration" / "a3wa_config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source != target.resolve():
        shutil.copy2(source, target)
    return target.resolve()


def _result_root(dataset_id: str) -> Path:
    return PROJECT_ROOT / "results_runs" / "public_benchmarks" / dataset_id


def _run_dir(dataset_id: str, run_id: str) -> Path:
    return _result_root(dataset_id) / "runs" / run_id


def _a3wa_deployment_class(config_path: str | Path | None) -> str:
    if not config_path:
        return "none"
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        return "pending"
    payload = load_json(path)
    gate = payload.get("deployment_gate", {})
    return "formal" if gate.get("passed") is True else "experimental"


def _run_completeness(
    context: dict[str, Any],
    *,
    run_dir: Path,
    questions: list[str],
    split: str,
    limit: int | None,
) -> dict[str, dict[str, int]]:
    by_id = {
        str(question["question_id"]): question
        for question in context["questions"]
    }
    incomplete: dict[str, dict[str, int]] = {}
    for question_id in questions:
        question = by_id[question_id]
        split_path = context["root"] / question["rubric_split_path"]
        split_payload = load_json(split_path)
        source_split = _source_split(split)
        if source_split not in split_payload:
            raise ValueError(
                f"{question_id} split metadata has no {source_split} "
                f"partition required by {split}."
            )
        expected = len(split_payload[source_split])
        if limit is not None:
            expected = min(expected, limit)
        checkpoint_path = run_dir / f"{question_id}_grading_checkpoint.json"
        checkpoint = load_json(checkpoint_path) if checkpoint_path.is_file() else []
        completed = len(checkpoint) if isinstance(checkpoint, list) else 0
        if completed != expected:
            incomplete[question_id] = {
                "expected": expected,
                "completed": completed,
            }
    return incomplete


def _write_run_manifest(
    run_dir: Path,
    *,
    context: dict[str, Any],
    run_id: str,
    split: str,
    questions: list[str],
    a3wa_config: str | None,
    status: str,
    command: list[str],
) -> None:
    config = runtime_model_config()
    config_path = Path(a3wa_config).resolve() if a3wa_config else None
    config_reference = None
    if config_path:
        try:
            config_reference = str(config_path.relative_to(run_dir.resolve()))
        except ValueError:
            config_reference = str(config_path)
    write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "dataset_id": context["manifest"]["dataset_id"],
            "dataset_snapshot": context["audit"],
            "dataset_manifest_sha256": sha256_file(
                context["root"] / "manifest.json"
            ),
            "split": split,
            "source_split": _source_split(split),
            "questions": questions,
            "extraction_backend": "text_only",
            "model_config": config,
            "a3wa_config": config_reference,
            "a3wa_deployment_class": _a3wa_deployment_class(config_path),
            "a3wa_config_sha256": (
                sha256_file(config_path)
                if config_path and config_path.is_file()
                else None
            ),
            "status": status,
            "command": command,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def grade(
    context: dict[str, Any],
    *,
    questions: list[str],
    split: str,
    run_id: str,
    force: bool,
    a3wa_config: str | None,
    dry_run: bool,
    evaluate_after: bool,
    limit: int | None = None,
    allow_experimental_a3wa: bool = False,
) -> int:
    dataset_id = context["manifest"]["dataset_id"]
    run_dir = _run_dir(dataset_id, run_id)
    if force and run_dir.exists() and not dry_run:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_run_inputs(context, run_dir, questions)
    archived_a3wa_config = _archive_a3wa_config(
        run_dir,
        a3wa_config,
        dry_run=dry_run,
    )
    runtime_db = _runtime_database(context, run_dir)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main_pipeline.py"),
        "--mode",
        "FULL",
        "--questions",
        *questions,
        "--database-path",
        str(runtime_db),
        "--teacher-db",
        str(context["root"] / "teacher_scores.json"),
        "--answer-metadata",
        str(context["root"] / "answer_metadata.jsonl"),
        "--initial-rubric-dir",
        str(context["root"] / "rubrics" / "initial"),
        "--rubric-dir",
        str(context["root"] / "rubrics" / "optimized"),
        "--extraction-backend",
        "text_only",
        "--ocr-cache-dir",
        str(
            PROJECT_ROOT
            / "ocr_cache"
            / "public_benchmarks"
            / dataset_id
        ),
        "--answer-split",
        split,
        "--results-dir",
        str(run_dir),
        "--progress-file",
        str(run_dir / "progress.json"),
        "--run-id",
        run_id,
    ]
    if force:
        command.append("--force-rerun")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        command.extend(["--img-limit", str(limit)])
    env = os.environ.copy()
    env["A3WA_CALIBRATION_CONFIG"] = (
        str(archived_a3wa_config)
        if archived_a3wa_config
        else ""
    )
    if allow_experimental_a3wa:
        env["REFGRADER_ALLOW_EXPERIMENTAL_A3WA"] = "1"
    _write_run_manifest(
        run_dir,
        context=context,
        run_id=run_id,
        split=split,
        questions=questions,
        a3wa_config=(
            str(archived_a3wa_config) if archived_a3wa_config else None
        ),
        status="dry_run" if dry_run else "running",
        command=command,
    )
    code = _run(command, env=env, dry_run=dry_run)
    if code != 0 or dry_run:
        failure_status = "interrupted" if code == 130 else "failed"
        _write_run_manifest(
            run_dir,
            context=context,
            run_id=run_id,
            split=split,
            questions=questions,
            a3wa_config=(
                str(archived_a3wa_config) if archived_a3wa_config else None
            ),
            status="dry_run" if dry_run else failure_status,
            command=command,
        )
        return code
    if evaluate_after:
        incomplete = _run_completeness(
            context,
            run_dir=run_dir,
            questions=questions,
            split=split,
            limit=limit,
        )
        if incomplete:
            print(
                "Refusing evaluation for an incomplete public benchmark run: "
                + json.dumps(incomplete, ensure_ascii=False, sort_keys=True)
            )
            _write_run_manifest(
                run_dir,
                context=context,
                run_id=run_id,
                split=split,
                questions=questions,
                a3wa_config=(
                    str(archived_a3wa_config)
                    if archived_a3wa_config
                    else None
                ),
                status="incomplete",
                command=command,
            )
            return 3
        code = evaluate_run(
            context,
            questions=questions,
            run_id=run_id,
            dry_run=False,
        )
    _write_run_manifest(
        run_dir,
        context=context,
        run_id=run_id,
        split=split,
        questions=questions,
        a3wa_config=(
            str(archived_a3wa_config) if archived_a3wa_config else None
        ),
        status="complete" if code == 0 else "evaluation_failed",
        command=command,
    )
    return code


def calibrate(
    context: dict[str, Any],
    *,
    questions: list[str],
    validation_run_id: str,
    output: str | None,
    score_calibration: bool,
    dry_run: bool,
) -> tuple[int, Path]:
    dataset_id = context["manifest"]["dataset_id"]
    validation_dir = _run_dir(dataset_id, validation_run_id)
    files = [
        validation_dir / f"{question_id}_grading_checkpoint.json"
        for question_id in questions
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing and not dry_run:
        raise FileNotFoundError(
            "Validation checkpoints are missing: " + ", ".join(missing)
        )
    output_path = (
        Path(output).expanduser().resolve()
        if output
        else (
            _result_root(dataset_id)
            / "calibration"
            / f"{validation_run_id}_a3wa.json"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "calibrate_a3wa.py"),
        "--files",
        *(str(path) for path in files),
        "--teacher-db",
        str(context["root"] / "teacher_scores.json"),
        "--database-path",
        str(validation_dir / "runtime_exam_database.json"),
        "--output",
        str(output_path),
    ]
    if score_calibration:
        command.append("--score-calibration")
    else:
        command.append("--no-score-calibration")
    code = _run(command, dry_run=dry_run)
    return code, output_path


def evaluate_run(
    context: dict[str, Any],
    *,
    questions: list[str],
    run_id: str,
    dry_run: bool,
) -> int:
    run_dir = _run_dir(context["manifest"]["dataset_id"], run_id)
    evaluation_dir = run_dir / "evaluation"
    if not dry_run:
        evaluation_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluate.py"),
        "--questions",
        *questions,
        "--results-dir",
        str(run_dir),
        "--result-source",
        "checkpoint",
        "--teacher-db",
        str(context["root"] / "teacher_scores.json"),
        "--database-path",
        str(run_dir / "runtime_exam_database.json"),
        "--compare",
        "--compare-score-keys",
        "single",
        "avg",
        "selected",
        "3wd-core",
        "3wd",
        "--compare-output",
        str(evaluation_dir / "compare.csv"),
        "--summary-output",
        str(evaluation_dir / "summary.json"),
    ]
    return _run(command, dry_run=dry_run)


def publish_run(
    context: dict[str, Any],
    *,
    run_id: str,
    artifacts_repo: str,
    force: bool,
) -> Path:
    source = _run_dir(context["manifest"]["dataset_id"], run_id)
    if not source.is_dir():
        raise FileNotFoundError(f"Run directory not found: {source}")
    target = (
        Path(artifacts_repo).expanduser().resolve()
        / "public_benchmarks"
        / context["manifest"]["dataset_id"]
        / "runs"
        / run_id
    )
    if target.exists() and force:
        shutil.rmtree(target)
    if target.exists():
        raise FileExistsError(
            f"Artifact run already exists: {target}. Use --force to replace it."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible RefGrader public benchmark experiments."
    )
    parser.add_argument("--prepared-dir", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    grade_parser = subparsers.add_parser("grade")
    grade_parser.add_argument("questions", nargs="*")
    grade_parser.add_argument(
        "--split",
        choices=[
            "all",
            "calibration",
            "validation",
            "test",
            "external_holdout",
        ],
        default="test",
        help=(
            "Prepared partition to grade. external_holdout evaluates the "
            "original train partition as an unused external holdout."
        ),
    )
    grade_parser.add_argument("--run-id", required=True)
    grade_parser.add_argument("--force", action="store_true")
    grade_parser.add_argument("--a3wa-config")
    grade_parser.add_argument("--limit", type=int)
    grade_parser.add_argument("--no-evaluate", action="store_true")
    grade_parser.add_argument(
        "--allow-experimental-a3wa",
        action="store_true",
        help="Run with a failed deployment gate and label the run experimental.",
    )
    grade_parser.add_argument("--dry-run", action="store_true")

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("questions", nargs="*")
    calibrate_parser.add_argument("--validation-run-id", required=True)
    calibrate_parser.add_argument("--output")
    calibrate_parser.add_argument("--score-calibration", action="store_true")
    calibrate_parser.add_argument("--dry-run", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("questions", nargs="*")
    evaluate_parser.add_argument("--run-id", required=True)
    evaluate_parser.add_argument("--dry-run", action="store_true")

    workflow_parser = subparsers.add_parser("workflow")
    workflow_parser.add_argument("questions", nargs="*")
    workflow_parser.add_argument("--tag")
    workflow_parser.add_argument("--force", action="store_true")
    workflow_parser.add_argument("--score-calibration", action="store_true")
    workflow_parser.add_argument("--limit", type=int)
    workflow_parser.add_argument(
        "--strict-deployment-gate",
        action="store_true",
        help="Stop before test grading when the calibrated A3WA gate fails.",
    )
    workflow_parser.add_argument("--dry-run", action="store_true")

    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--run-id", required=True)
    publish_parser.add_argument("--artifacts-repo", required=True)
    publish_parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    context = _prepared_context(args.prepared_dir)
    questions = _selected_questions(
        context["question_ids"],
        getattr(args, "questions", None),
    )
    if args.command == "grade":
        return grade(
            context,
            questions=questions,
            split=args.split,
            run_id=args.run_id,
            force=args.force,
            a3wa_config=args.a3wa_config,
            dry_run=args.dry_run,
            evaluate_after=(
                not args.no_evaluate
                and args.split in {"test", "external_holdout"}
            ),
            limit=args.limit,
            allow_experimental_a3wa=args.allow_experimental_a3wa,
        )
    if args.command == "calibrate":
        code, output = calibrate(
            context,
            questions=questions,
            validation_run_id=args.validation_run_id,
            output=args.output,
            score_calibration=args.score_calibration,
            dry_run=args.dry_run,
        )
        print(f"A3WA config: {output}")
        return code
    if args.command == "evaluate":
        return evaluate_run(
            context,
            questions=questions,
            run_id=args.run_id,
            dry_run=args.dry_run,
        )
    if args.command == "workflow":
        tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        validation_run = f"{tag}_validation"
        test_run = f"{tag}_test"
        code = grade(
            context,
            questions=questions,
            split="validation",
            run_id=validation_run,
            force=args.force,
            a3wa_config=None,
            dry_run=args.dry_run,
            evaluate_after=False,
            limit=args.limit,
        )
        if code != 0:
            return code
        code, config = calibrate(
            context,
            questions=questions,
            validation_run_id=validation_run,
            output=None,
            score_calibration=args.score_calibration,
            dry_run=args.dry_run,
        )
        if code != 0:
            return code
        deployment_class = _a3wa_deployment_class(config)
        allow_experimental = deployment_class == "experimental"
        if allow_experimental:
            message = (
                "A3WA deployment gate failed; the test run will continue "
                "as an explicitly experimental benchmark."
            )
            if args.strict_deployment_gate:
                print(message + " Strict mode requested, stopping before test.")
                return 2
            print("WARNING: " + message)
        return grade(
            context,
            questions=questions,
            split="test",
            run_id=test_run,
            force=args.force,
            a3wa_config=str(config),
            dry_run=args.dry_run,
            evaluate_after=True,
            limit=args.limit,
            allow_experimental_a3wa=allow_experimental,
        )
    if args.command == "publish":
        target = publish_run(
            context,
            run_id=args.run_id,
            artifacts_repo=args.artifacts_repo,
            force=args.force,
        )
        print(f"Published: {target}")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
