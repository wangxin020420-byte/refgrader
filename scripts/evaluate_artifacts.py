"""
Evaluate CSBench grading results published in refgrader-artifacts.

This is a local convenience wrapper around evaluate.py. It stages the
per-question artifact layout into the flat results_runs layout expected by
evaluate.py, then runs the existing evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_REPO = PROJECT_ROOT.parent / "refgrader-artifacts"
DEFAULT_PREPARED_DIR = PROJECT_ROOT / "data" / "csbench"
GRADING_FILES = {
    "grading_checkpoint.json": "grading_checkpoint",
    "graded_results.json": "graded_results",
    "rejected.json": "rejected",
    "failed.json": "failed",
}


def normalize_question(question: str) -> str:
    value = question.strip().upper()
    if not value:
        raise ValueError("empty question id")
    return value


def question_slug(question: str) -> str:
    return question.lower().replace("_", "")


def latest_run_id(artifacts_repo: Path, question: str) -> str:
    run_root = artifacts_repo / "csbench" / question / "grading_runs"
    if not run_root.exists():
        raise FileNotFoundError(f"grading run directory not found: {run_root}")
    runs = [path for path in run_root.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"no grading runs found under: {run_root}")
    runs.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    return runs[0].name


def stage_artifact_results(
    *,
    artifacts_repo: Path,
    questions: list[str],
    run_id: str,
    results_dir: Path,
) -> tuple[dict[str, dict], Path | None]:
    results_dir.mkdir(parents=True, exist_ok=True)
    completion = {}
    policy_descriptors = []
    policy_sources = []
    for question in questions:
        run_dir = artifacts_repo / "csbench" / question / "grading_runs" / run_id
        grading_dir = run_dir / "grading"
        checkpoint = grading_dir / "grading_checkpoint.json"
        if not checkpoint.exists():
            raise FileNotFoundError(f"required checkpoint not found: {checkpoint}")

        for source_name, suffix in GRADING_FILES.items():
            source = grading_dir / source_name
            if not source.exists():
                continue
            target = results_dir / f"{question}_{suffix}.json"
            shutil.copy2(source, target)
        report_path = grading_dir / "completion_report.json"
        manifest_path = run_dir / "run_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if manifest_path.is_file()
            else {}
        )
        policy_descriptor = manifest.get("sample_quality_policy") or {
            "mode": "raw",
            "policy_id": "raw",
            "sha256": None,
        }
        policy_descriptors.append(policy_descriptor)
        if policy_descriptor.get("mode") == "active":
            policy_source = run_dir / "dataset" / "sample_quality_policy.json"
            if not policy_source.is_file():
                raise FileNotFoundError(
                    "Active sample-quality policy is missing from artifact: "
                    f"{policy_source}"
                )
            policy_sources.append(policy_source)
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8-sig"))
            question_report = report.get("questions", {}).get(question, {})
        elif manifest_path.is_file():
            question_report = manifest.get("completion") or {}
        else:
            question_report = {}
        completion[question] = question_report
    serialized = {
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        for value in policy_descriptors
    }
    if len(serialized) != 1:
        raise ValueError(
            "Requested artifacts used different sample-quality policies."
        )
    staged_policy = None
    descriptor = policy_descriptors[0]
    if descriptor.get("mode") == "active":
        if len(policy_sources) != len(questions):
            raise ValueError(
                "Sample-quality policy is missing from one or more artifacts."
            )
        payloads = {source.read_bytes() for source in policy_sources}
        if len(payloads) != 1:
            raise ValueError(
                "Requested artifacts contain different policy payloads."
            )
        staged_policy = results_dir / "sample_quality_policy.json"
        shutil.copy2(policy_sources[0], staged_policy)
    return completion, staged_policy


def build_evaluate_command(args: argparse.Namespace, questions: list[str], results_dir: Path) -> list[str]:
    teacher_db = args.prepared_dir / "teacher_scores.json"
    database_path = args.prepared_dir / "exam_database.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "evaluate.py"),
        "--questions",
        *questions,
        "--results-dir",
        str(results_dir),
        "--result-source",
        args.result_source,
        "--teacher-db",
        str(teacher_db),
        "--database-path",
        str(database_path),
    ]

    if args.detail:
        cmd.append("--detail")

    requested_score_keys = args.compare_score_keys or args.score_key
    auto_compare = len(requested_score_keys) > 1

    if args.compare or args.export or auto_compare:
        cmd.extend(["--compare", "--compare-score-keys", *requested_score_keys])
        if args.compare_output:
            cmd.extend(["--compare-output", str(args.compare_output)])
        if args.summary_output:
            cmd.extend(["--summary-output", str(args.summary_output)])
    elif requested_score_keys:
        cmd.extend(["--score-key", requested_score_keys[0]])

    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage CSBench results from refgrader-artifacts and evaluate them locally."
    )
    parser.add_argument("questions", nargs="+", help="Question ids, e.g. CO_1 CO_2 CO_3")
    parser.add_argument(
        "--run-id",
        help="Artifact grading run id. If omitted, uses the newest run under the first question.",
    )
    parser.add_argument(
        "--artifacts-repo",
        type=Path,
        default=DEFAULT_ARTIFACTS_REPO,
        help=f"Path to refgrader-artifacts. Default: {DEFAULT_ARTIFACTS_REPO}",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=DEFAULT_PREPARED_DIR,
        help=f"Prepared CSBench data directory. Default: {DEFAULT_PREPARED_DIR}",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Flat staging directory for evaluate.py. Default: results_runs/artifacts_<questions>_<run_id>",
    )
    parser.add_argument(
        "--result-source",
        choices=("checkpoint", "graded"),
        default="checkpoint",
        help="Evaluation source passed to evaluate.py.",
    )
    parser.add_argument(
        "--score-key",
        nargs="+",
        default=["3wd"],
        help=(
            "Score form(s): single, avg, selected, 3wd-core, 3wd. "
            "One value runs single-form evaluation; multiple values automatically run compare."
        ),
    )
    parser.add_argument("--detail", action="store_true", help="Show per-answer details.")
    parser.add_argument("--compare", action="store_true", help="Compare multiple score forms.")
    parser.add_argument(
        "--compare-score-keys",
        nargs="+",
        default=None,
        help=(
            "Score forms for --compare. If omitted, uses --score-key; "
            "if neither is customized, --compare/--export compares "
            "single avg selected 3wd-core 3wd."
        ),
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Shortcut for --compare plus CSV export.",
    )
    parser.add_argument(
        "--compare-output",
        type=Path,
        help="CSV path. Default with --export: outputs/csbench_<questions>_<run_id>_compare.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="JSON path for machine-readable evaluation metrics.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the evaluate.py command without running it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    questions = [normalize_question(question) for question in args.questions]
    args.artifacts_repo = args.artifacts_repo.resolve()
    args.prepared_dir = args.prepared_dir.resolve()

    if (args.compare or args.export) and args.compare_score_keys is None and args.score_key == ["3wd"]:
        args.compare_score_keys = ["single", "avg", "selected", "3wd-core", "3wd"]

    run_id = args.run_id or latest_run_id(args.artifacts_repo, questions[0])
    slug = "_".join(question_slug(question) for question in questions)
    results_dir = args.results_dir or PROJECT_ROOT / "results_runs" / f"artifacts_{slug}_{run_id}"
    results_dir = results_dir.resolve()

    if args.export and not args.compare_output:
        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.compare_output = output_dir / f"csbench_{slug}_{run_id}_compare.csv"
    if args.export and not args.summary_output:
        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.summary_output = output_dir / f"csbench_{slug}_{run_id}_summary.json"

    completion, staged_policy = stage_artifact_results(
        artifacts_repo=args.artifacts_repo,
        questions=questions,
        run_id=run_id,
        results_dir=results_dir,
    )
    cmd = build_evaluate_command(args, questions, results_dir)

    print(f"Staged artifacts run {run_id} to: {results_dir}", flush=True)
    for question, report in completion.items():
        status = report.get("status", "unknown")
        checkpoint = report.get("checkpoint_count", "?")
        expected = report.get("expected_count", "?")
        failed = report.get("failed_count", "?")
        print(
            f"Coverage {question}: status={status}, "
            f"checkpoint={checkpoint}/{expected}, failed={failed}",
            flush=True,
        )
        if status == "partial":
            print(
                "WARNING: metrics below describe successful checkpoint IDs "
                "only; report coverage with the evaluation.",
                flush=True,
            )
    print("Running:", " ".join(str(part) for part in cmd), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    if staged_policy:
        env["REFGRADER_SAMPLE_POLICY"] = str(staged_policy)
        env.pop("REFGRADER_SAMPLE_POLICY_MODE", None)
    else:
        env["REFGRADER_SAMPLE_POLICY_MODE"] = "raw"
        env.pop("REFGRADER_SAMPLE_POLICY", None)
    return subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
