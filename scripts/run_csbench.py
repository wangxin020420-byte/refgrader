"""Unified command-line entry point for CSBench experiments.

Examples:
    python scripts/run_csbench.py optimize CO_3
    python scripts/run_csbench.py optimize CO_2 CO_3 CO_4
    python scripts/run_csbench.py grade CO_3 --background --force
    python scripts/run_csbench.py grade CO_2 CO_3 --background --force
    python scripts/run_csbench.py evaluate CO_3 --export
    python scripts/run_csbench.py outputs CO_3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OCR_DEVICE = "cpu" if os.name == "nt" else "gpu:0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rubric_total(rubric: list[dict[str, Any]]) -> float:
    return sum(float(item.get("points", 0)) for item in rubric)


def normalize_question_id(value: str) -> str:
    return str(value).strip().upper()


def question_slug(question_id: str) -> str:
    return question_id.lower().replace("_", "")


def normalize_question_ids(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        question_id = normalize_question_id(value)
        if question_id and question_id not in seen:
            result.append(question_id)
            seen.add(question_id)
    return result


def batch_slug(contexts: list["CSBenchContext"]) -> str:
    return "_".join(
        question_slug(question_id)
        for question_id in sorted(ctx.question_id for ctx in contexts)
    )


def build_contexts(
    prepared_dir: str, question_ids: list[str]
) -> list["CSBenchContext"]:
    return [
        CSBenchContext(prepared_dir, question_id)
        for question_id in normalize_question_ids(question_ids)
    ]


def display_command(command: list[str], env_overrides: dict[str, str]) -> str:
    if os.name == "nt":
        prefix = " ".join(
            f'$env:{key}="{value}";' for key, value in env_overrides.items()
        )
        return f"{prefix} {subprocess.list2cmdline(command)}".strip()
    prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in env_overrides.items()
    )
    return f"{prefix} {shlex.join(command)}".strip()


class CSBenchContext:
    def __init__(self, prepared_dir: str, question_id: str | None = None):
        self.root = (PROJECT_ROOT / prepared_dir).resolve()
        self.database = self.root / "exam_database.json"
        self.teacher_db = self.root / "teacher_scores.json"
        self.answer_metadata = self.root / "answer_metadata.jsonl"
        self.initial_dir = self.root / "rubrics" / "initial"
        self.optimized_dir = self.root / "rubrics" / "optimized"
        self.manifest_dir = self.root / "rubrics" / "manifests"
        self.question_id = normalize_question_id(question_id) if question_id else None
        self.question: dict[str, Any] | None = None

        required = [self.database, self.teacher_db, self.answer_metadata]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "CSBench compatible data is incomplete. Missing: "
                + ", ".join(missing)
                + ". Run scripts/prepare_csbench.py first."
            )

        exam = json.loads(self.database.read_text(encoding="utf-8"))
        if self.question_id:
            self.question = next(
                (
                    item
                    for item in exam
                    if str(item.get("question_id", "")).upper()
                    == self.question_id
                ),
                None,
            )
            if not self.question:
                available = ", ".join(
                    sorted(str(item.get("question_id")) for item in exam)
                )
                raise ValueError(
                    f"Question {self.question_id} was not found. "
                    f"Available questions: {available}"
                )

    @property
    def group(self) -> str:
        assert self.question_id and self.question
        return str(
            self.question.get("rubric_group")
            or self.question_id.split("_", 1)[0]
        )

    @property
    def initial_rubric(self) -> Path:
        assert self.question_id
        return (
            self.initial_dir
            / self.group
            / f"{self.question_id}_rubric_standard.json"
        )

    @property
    def optimized_rubric(self) -> Path:
        assert self.question_id
        return (
            self.optimized_dir
            / self.group
            / f"{self.question_id}_rubric_standard.json"
        )

    @property
    def optimization_manifest(self) -> Path:
        assert self.question_id
        return (
            self.manifest_dir
            / self.group
            / f"{self.question_id}_optimization.json"
        )

    @property
    def split_file(self) -> Path:
        assert self.question_id
        return self.root / "splits" / "by_question" / f"{self.question_id}.json"

    def validate_initial(self, sample_size: int) -> None:
        assert self.question
        if not self.initial_rubric.is_file():
            raise FileNotFoundError(
                f"Initial rubric not found: {self.initial_rubric}"
            )
        if not self.split_file.is_file():
            raise FileNotFoundError(f"Question split not found: {self.split_file}")

        rubric = json.loads(self.initial_rubric.read_text(encoding="utf-8"))
        total = rubric_total(rubric)
        question_total = float(self.question.get("total_score", 0))
        if abs(total - question_total) > 1e-6:
            raise ValueError(
                f"Initial rubric total {total} does not match question total "
                f"{question_total}."
            )

        split = json.loads(self.split_file.read_text(encoding="utf-8"))
        calibration_count = len(split.get("calibration", []))
        if calibration_count < sample_size:
            raise ValueError(
                f"{self.question_id} has only {calibration_count} calibration "
                f"answers, fewer than --sample-size {sample_size}."
            )

    def validate_optimized(self) -> None:
        assert self.question
        if not self.optimized_rubric.is_file():
            raise FileNotFoundError(
                f"Optimized rubric not found: {self.optimized_rubric}. "
                f"Run: python scripts/run_csbench.py optimize {self.question_id}"
            )
        if not self.optimization_manifest.is_file():
            raise FileNotFoundError(
                f"Optimization manifest not found: {self.optimization_manifest}"
            )

        rubric = json.loads(self.optimized_rubric.read_text(encoding="utf-8"))
        total = rubric_total(rubric)
        question_total = float(self.question.get("total_score", 0))
        if abs(total - question_total) > 1e-6:
            raise ValueError(
                f"Optimized rubric total {total} does not match question total "
                f"{question_total}."
            )

        manifest = json.loads(
            self.optimization_manifest.read_text(encoding="utf-8")
        )
        if manifest.get("initial_sha256") != sha256_file(self.initial_rubric):
            raise ValueError(
                "The optimized rubric was generated from a different initial "
                "rubric. Re-run optimize with --force."
            )
        if manifest.get("optimized_sha256") != sha256_file(
            self.optimized_rubric
        ):
            raise ValueError(
                "The optimized rubric does not match its manifest. "
                "Re-run optimize with --force."
            )


def main_pipeline_base(contexts: list[CSBenchContext]) -> list[str]:
    ctx = contexts[0]
    return [
        "--questions",
        *(item.question_id for item in contexts),
        "--database-path",
        str(ctx.database),
        "--answer-metadata",
        str(ctx.answer_metadata),
        "--initial-rubric-dir",
        str(ctx.initial_dir),
        "--rubric-dir",
        str(ctx.optimized_dir),
        "--extraction-backend",
        "csbench_hybrid",
        "--ocr-cache-dir",
        str((PROJECT_ROOT / "ocr_cache" / "csbench").resolve()),
    ]


def execute(
    command: list[str],
    *,
    env_overrides: dict[str, str] | None = None,
    dry_run: bool = False,
) -> int:
    env_overrides = env_overrides or {}
    print(display_command(command, env_overrides))
    if dry_run:
        return 0
    env = os.environ.copy()
    env.update(env_overrides)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env).returncode


def git_output(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def portable_value(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {
            str(key): portable_value(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [portable_value(item, replacements) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        for source, replacement in replacements:
            source_normalized = source.replace("\\", "/").rstrip("/")
            if normalized == source_normalized:
                return replacement
            if normalized.startswith(source_normalized + "/"):
                return replacement + normalized[len(source_normalized) :]
        return value
    return value


def copy_portable_json(
    source: Path,
    destination: Path,
    replacements: list[tuple[str, str]],
) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            portable_value(payload, replacements),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def copy_if_exists(
    source: Path,
    destination: Path,
    replacements: list[tuple[str, str]],
) -> bool:
    if not source.is_file():
        return False
    if source.suffix.lower() in {".json", ".jsonl"}:
        if source.suffix.lower() == ".json":
            copy_portable_json(source, destination, replacements)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(encoding="utf-8") as input_handle, destination.open(
                "w", encoding="utf-8"
            ) as output_handle:
                for line in input_handle:
                    if line.strip():
                        payload = json.loads(line)
                        output_handle.write(
                            json.dumps(
                                portable_value(payload, replacements),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return True


def copy_json_directory(
    source: Path,
    destination: Path,
    replacements: list[tuple[str, str]],
) -> int:
    if not source.is_dir():
        return 0
    copied = 0
    for path in sorted(source.glob("*.json")):
        copy_portable_json(path, destination / path.name, replacements)
        copied += 1
    return copied


def ensure_background_slot_available() -> None:
    if os.name == "nt":
        return
    pid_file = PROJECT_ROOT / "logs" / "refgrader.pid"
    if not pid_file.is_file():
        return
    lines = pid_file.read_text(encoding="utf-8").splitlines()
    try:
        pid = int(lines[1])
    except (IndexError, ValueError):
        pid_file.unlink(missing_ok=True)
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return
    except PermissionError:
        pass
    raise RuntimeError(
        f"A background RefGrader experiment is already running with PID {pid}. "
        "Use `python scripts/run_csbench.py status` or stop it first."
    )


def optimize(args: argparse.Namespace) -> int:
    contexts = build_contexts(args.prepared_dir, args.questions)
    for ctx in contexts:
        ctx.validate_initial(args.sample_size)
    existing = [
        ctx.question_id
        for ctx in contexts
        if ctx.optimized_rubric.exists() or ctx.optimization_manifest.exists()
    ]
    if existing and not args.force:
        raise FileExistsError(
            "Optimized rubric already exists for "
            + ", ".join(existing)
            + ". "
            "Use --force to regenerate it."
        )

    slug = batch_slug(contexts)
    results_dir = (
        PROJECT_ROOT / "results_runs" / f"csbench_{slug}_rubric_opt"
    ).resolve()
    pipeline_args = [
        "--mode",
        "VARIANCE_OPT",
        *main_pipeline_base(contexts),
        "--sample-size",
        str(args.sample_size),
        "--results-dir",
        str(results_dir),
        "--progress-file",
        str(results_dir / "progress.json"),
    ]
    if args.force:
        pipeline_args.append("--force-rerun")

    env = {"REFGRADER_OCR_DEVICE": args.device}
    if args.background:
        if os.name == "nt":
            raise RuntimeError("--background is only supported on Linux.")
        ensure_background_slot_available()
        command = [
            str(PROJECT_ROOT / "run_experiment.sh"),
            "run",
            *pipeline_args,
        ]
    else:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "main_pipeline.py"),
            *pipeline_args,
        ]

    print("Questions: " + ", ".join(ctx.question_id for ctx in contexts))
    for ctx in contexts:
        print(f"{ctx.question_id} initial rubric: {ctx.initial_rubric}")
        print(f"{ctx.question_id} optimized rubric: {ctx.optimized_rubric}")
    return execute(command, env_overrides=env, dry_run=args.dry_run)


def grade(args: argparse.Namespace) -> int:
    contexts = build_contexts(args.prepared_dir, args.questions)
    for ctx in contexts:
        ctx.validate_optimized()
    slug = batch_slug(contexts)
    results_dir = (
        PROJECT_ROOT / "results_runs" / f"csbench_{slug}_full"
    ).resolve()
    pipeline_args = [
        "--mode",
        "FULL",
        *main_pipeline_base(contexts),
        "--teacher-db",
        str(contexts[0].teacher_db),
        "--answer-split",
        args.split,
        "--results-dir",
        str(results_dir),
        "--progress-file",
        str(results_dir / "progress.json"),
    ]
    if args.limit is not None:
        pipeline_args.extend(["--img-limit", str(args.limit)])
    if args.force:
        pipeline_args.append("--force-rerun")

    env = {"REFGRADER_OCR_DEVICE": args.device}
    if args.background:
        if os.name == "nt":
            raise RuntimeError("--background is only supported on Linux.")
        ensure_background_slot_available()
        command = [
            str(PROJECT_ROOT / "run_experiment.sh"),
            "run",
            *pipeline_args,
        ]
    else:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "main_pipeline.py"),
            *pipeline_args,
        ]

    print("Questions: " + ", ".join(ctx.question_id for ctx in contexts))
    print(f"Answer split: {args.split}")
    print(f"Results: {results_dir}")
    return execute(command, env_overrides=env, dry_run=args.dry_run)


def evaluate(args: argparse.Namespace) -> int:
    contexts = build_contexts(args.prepared_dir, args.questions)
    slug = batch_slug(contexts)
    results_dir = (
        PROJECT_ROOT / "results_runs" / f"csbench_{slug}_full"
    ).resolve()
    missing = [
        str(results_dir / f"{ctx.question_id}_grading_checkpoint.json")
        for ctx in contexts
        if not (
            results_dir / f"{ctx.question_id}_grading_checkpoint.json"
        ).is_file()
    ]
    if missing and not args.dry_run:
        raise FileNotFoundError(
            "Grading checkpoints not found: " + ", ".join(missing)
        )

    command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluate.py"),
        "--questions",
        *(ctx.question_id for ctx in contexts),
        "--results-dir",
        str(results_dir),
        "--result-source",
        "checkpoint",
        "--teacher-db",
        str(contexts[0].teacher_db),
        "--database-path",
        str(contexts[0].database),
        "--compare",
        "--compare-score-keys",
        "single",
        "avg",
        "selected",
        "3wd",
    ]
    if args.export:
        command.extend(
            [
                "--compare-output",
                str(
                    (
                        PROJECT_ROOT
                        / "outputs"
                        / f"csbench_{slug}_compare.csv"
                    ).resolve()
                ),
            ]
        )
    if args.detail:
        command.append("--detail")
    return execute(command, dry_run=args.dry_run)


def manage_background(args: argparse.Namespace) -> int:
    if os.name == "nt":
        raise RuntimeError(f"{args.action} is only supported on Linux.")
    command = [str(PROJECT_ROOT / "run_experiment.sh"), args.action]
    return execute(command, dry_run=getattr(args, "dry_run", False))


def monitor(args: argparse.Namespace) -> int:
    contexts = build_contexts(args.prepared_dir, args.questions)
    slug = batch_slug(contexts)
    progress = (
        PROJECT_ROOT
        / "results_runs"
        / f"csbench_{slug}_full"
        / "progress.json"
    ).resolve()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "monitor.py"),
        "--watch",
        "--progress-file",
        str(progress),
    ]
    return execute(command, dry_run=args.dry_run)


def show_outputs(args: argparse.Namespace) -> int:
    contexts = build_contexts(args.prepared_dir, args.questions)
    slug = batch_slug(contexts)
    optimize_dir = (
        PROJECT_ROOT / "results_runs" / f"csbench_{slug}_rubric_opt"
    ).resolve()
    grade_dir = (
        PROJECT_ROOT / "results_runs" / f"csbench_{slug}_full"
    ).resolve()
    metadata = {}
    with contexts[0].answer_metadata.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                metadata[str(record.get("answer_id"))] = record

    print(f"Combined optimize run: {optimize_dir}")
    print(f"Combined grading run: {grade_dir}")
    for ctx in contexts:
        split = json.loads(ctx.split_file.read_text(encoding="utf-8"))
        test_ids = [str(value) for value in split.get("test", [])]
        visual_flags = [
            answer_id
            for answer_id in test_ids
            if metadata.get(answer_id, {}).get("isimagine")
            or metadata.get(answer_id, {}).get(
                "visual_placeholder_detected"
            )
        ]
        visual_enabled = bool(
            ctx.question and ctx.question.get("requires_visual_evidence")
        )
        visual_ocr_count = len(visual_flags) if visual_enabled else 0

        print()
        print(f"[{ctx.question_id}]")
        print(f"Initial rubric: {ctx.initial_rubric}")
        print(f"Optimized rubric: {ctx.optimized_rubric}")
        print(f"Optimization manifest: {ctx.optimization_manifest}")
        print(
            "Variance checkpoint: "
            f"{optimize_dir / f'{ctx.question_id}_variance_checkpoint.json'}"
        )
        print(f"Optimization progress: {optimize_dir / 'progress.json'}")
        print(
            "Grading checkpoint: "
            f"{grade_dir / f'{ctx.question_id}_grading_checkpoint.json'}"
        )
        print(
            "Graded results: "
            f"{grade_dir / f'{ctx.question_id}_graded_results.json'}"
        )
        print(
            "Rejected results: "
            f"{grade_dir / f'{ctx.question_id}_rejected.json'}"
        )
        print(
            "Failed results: "
            f"{grade_dir / f'{ctx.question_id}_failed.json'}"
        )
        print(f"Grading progress: {grade_dir / 'progress.json'}")
        print(
            "Mapped fact cache: "
            f"{PROJECT_ROOT / 'ocr_cache' / 'csbench' / 'facts' / ctx.question_id / '<answer_id>.json'}"
        )
        print(
            "Variance fact cache: "
            f"{PROJECT_ROOT / 'ocr_cache' / 'csbench' / 'variance_facts' / ctx.question_id / '<answer_id>.json'}"
        )
        print(
            "Raw OCR cache (visual answers only): "
            f"{PROJECT_ROOT / 'ocr_cache' / 'csbench' / ctx.question_id / '<answer_id>.json'}"
        )
        print(
            f"Test answers: {len(test_ids)}; visual flags: "
            f"{len(visual_flags)}; expected OCR triggers: {visual_ocr_count}"
        )
        if visual_flags and not visual_enabled:
            print(
                "OCR note: visual flags exist, but the rubric does not require "
                "diagram evidence, so csbench_hybrid keeps the raw_text route."
            )

    print()
    print(
        "Evaluation CSV: "
        f"{(PROJECT_ROOT / 'outputs' / f'csbench_{slug}_compare.csv').resolve()}"
    )
    print("Runtime log: logs/experiment_<run_id>.log")
    return 0


def publish(args: argparse.Namespace) -> int:
    contexts = build_contexts(args.prepared_dir, args.questions)
    slug = batch_slug(contexts)
    artifacts_repo = Path(args.artifacts_repo).expanduser().resolve()
    if not (artifacts_repo / ".git").is_dir():
        raise ValueError(
            f"Artifacts repository is not a Git repository: {artifacts_repo}"
        )

    status = git_output(artifacts_repo, "status", "--porcelain")
    if status:
        raise RuntimeError(
            "Artifacts repository has uncommitted changes. Commit, discard, "
            "or pull them before publishing:\n" + status
        )
    if args.push:
        subprocess.run(
            ["git", "-C", str(artifacts_repo), "pull", "--rebase", "origin", "main"],
            check=True,
        )

    optimize_dir = (
        PROJECT_ROOT / "results_runs" / f"csbench_{slug}_rubric_opt"
    ).resolve()
    grade_dir = (
        PROJECT_ROOT / "results_runs" / f"csbench_{slug}_full"
    ).resolve()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    code_commit = git_output(PROJECT_ROOT, "rev-parse", "HEAD")

    prepared_manifest = {}
    prepared_manifest_path = contexts[0].root / "manifest.json"
    if prepared_manifest_path.is_file():
        prepared_manifest = json.loads(
            prepared_manifest_path.read_text(encoding="utf-8")
        )
    dataset_root = Path(
        prepared_manifest.get("dataset_root", "")
    ).expanduser()
    dataset_commit = None
    if dataset_root and (dataset_root / ".git").exists():
        try:
            dataset_commit = git_output(dataset_root, "rev-parse", "HEAD")
        except subprocess.CalledProcessError:
            dataset_commit = None

    replacements = [
        (str(PROJECT_ROOT), "${REFGRADER_ROOT}"),
        (str(contexts[0].root), "${PREPARED_CSBENCH_ROOT}"),
    ]
    if str(dataset_root):
        replacements.append((str(dataset_root), "${CSBENCH_ROOT}"))

    latest_log = None
    log_candidates = sorted(
        (PROJECT_ROOT / "logs").glob("experiment_*.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if log_candidates:
        latest_log = log_candidates[0]

    published_paths = []
    for ctx in contexts:
        ctx.validate_optimized()
        checkpoint = grade_dir / f"{ctx.question_id}_grading_checkpoint.json"
        has_full_results = checkpoint.is_file()
        if args.stage == "full" and not has_full_results:
            raise FileNotFoundError(
                f"Full grading checkpoint not found: {checkpoint}"
            )
        include_full = args.stage == "full" or (
            args.stage == "auto" and has_full_results
        )

        destination = (
            artifacts_repo
            / "csbench"
            / ctx.question_id
            / "runs"
            / run_id
        )
        if destination.exists():
            raise FileExistsError(
                f"Artifact run already exists: {destination}. "
                "Use a different --run-id."
            )

        copy_portable_json(
            ctx.initial_rubric,
            destination / "rubrics" / "initial_rubric.json",
            replacements,
        )
        copy_portable_json(
            ctx.optimized_rubric,
            destination / "rubrics" / "optimized_rubric.json",
            replacements,
        )
        copy_portable_json(
            ctx.optimization_manifest,
            destination / "rubrics" / "optimization_manifest.json",
            replacements,
        )
        copy_if_exists(
            optimize_dir / f"{ctx.question_id}_variance_checkpoint.json",
            destination
            / "rubric_optimization"
            / "variance_checkpoint.json",
            replacements,
        )
        copy_if_exists(
            optimize_dir / "progress.json",
            destination / "rubric_optimization" / "progress.json",
            replacements,
        )
        variance_facts_count = copy_json_directory(
            PROJECT_ROOT
            / "ocr_cache"
            / "csbench"
            / "variance_facts"
            / ctx.question_id,
            destination / "rubric_optimization" / "facts",
            replacements,
        )

        result_counts = {}
        facts_count = 0
        raw_ocr_count = 0
        if include_full:
            result_files = {
                "checkpoint": (
                    f"{ctx.question_id}_grading_checkpoint.json",
                    "grading_checkpoint.json",
                ),
                "graded": (
                    f"{ctx.question_id}_graded_results.json",
                    "graded_results.json",
                ),
                "rejected": (
                    f"{ctx.question_id}_rejected.json",
                    "rejected.json",
                ),
                "failed": (
                    f"{ctx.question_id}_failed.json",
                    "failed.json",
                ),
            }
            for key, (source_name, destination_name) in result_files.items():
                source = grade_dir / source_name
                if copy_if_exists(
                    source,
                    destination / "grading" / destination_name,
                    replacements,
                ):
                    payload = json.loads(source.read_text(encoding="utf-8"))
                    result_counts[key] = (
                        len(payload) if isinstance(payload, list) else 0
                    )
                else:
                    result_counts[key] = 0
            copy_if_exists(
                grade_dir / "progress.json",
                destination / "grading" / "progress.json",
                replacements,
            )
            facts_count = copy_json_directory(
                PROJECT_ROOT
                / "ocr_cache"
                / "csbench"
                / "facts"
                / ctx.question_id,
                destination / "facts",
                replacements,
            )
            if args.include_raw_ocr:
                raw_ocr_count = copy_json_directory(
                    PROJECT_ROOT
                    / "ocr_cache"
                    / "csbench"
                    / ctx.question_id,
                    destination / "raw_ocr",
                    replacements,
                )
            compare_csv = (
                PROJECT_ROOT / "outputs" / f"csbench_{slug}_compare.csv"
            )
            copy_if_exists(
                compare_csv,
                destination / "evaluation" / "compare.csv",
                replacements,
            )

        if latest_log:
            copy_if_exists(
                latest_log,
                destination / "logs" / "experiment.log",
                replacements,
            )

        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "question_id": ctx.question_id,
            "published_stage": "full" if include_full else "rubric",
            "question_batch": [item.question_id for item in contexts],
            "code_commit": code_commit,
            "dataset_commit": dataset_commit,
            "extraction_backend": "csbench_hybrid",
            "answer_split": "test" if include_full else None,
            "ocr_device": os.getenv(
                "REFGRADER_OCR_DEVICE", DEFAULT_OCR_DEVICE
            ),
            "source_server": socket.gethostname(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "portable_path_variables": {
                "${REFGRADER_ROOT}": "RefGrader project root",
                "${PREPARED_CSBENCH_ROOT}": "prepared data/csbench root",
                "${CSBENCH_ROOT}": "source CSBench repository root",
            },
            "result_counts": result_counts,
            "variance_fact_files": variance_facts_count,
            "fact_files": facts_count,
            "raw_ocr_files": raw_ocr_count,
        }
        (destination / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        published_paths.append(destination)

    index_path = artifacts_repo / "csbench" / "index.json"
    index = []
    if index_path.is_file():
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            index = loaded
    for destination, ctx in zip(published_paths, contexts):
        index.append(
            {
                "question_id": ctx.question_id,
                "run_id": run_id,
                "path": destination.relative_to(artifacts_repo).as_posix(),
                "published_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for path in published_paths:
        print(f"Published: {path}")

    if args.push:
        subprocess.run(
            ["git", "-C", str(artifacts_repo), "add", "csbench"],
            check=True,
        )
        commit_message = (
            "Publish CSBench "
            + ", ".join(ctx.question_id for ctx in contexts)
            + f" artifacts {run_id}"
        )
        subprocess.run(
            ["git", "-C", str(artifacts_repo), "commit", "-m", commit_message],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(artifacts_repo), "push", "origin", "main"],
            check=True,
        )
        print("Artifacts committed and pushed.")
    else:
        print(
            "Artifacts copied but not committed. Review them, then commit "
            "manually or rerun with --push using a new --run-id."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CSBench rubric optimization, grading, and evaluation with "
        "paths derived automatically from one or more question IDs."
        )
    )
    parser.add_argument(
        "--prepared-dir",
        default="data/csbench",
        help="Prepared CSBench directory. Default: data/csbench",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    optimize_parser = subparsers.add_parser(
        "optimize", help="Optimize one or more question rubrics."
    )
    optimize_parser.add_argument(
        "questions", nargs="+", type=normalize_question_id
    )
    optimize_parser.add_argument("--sample-size", type=int, default=5)
    optimize_parser.add_argument("--device", default=DEFAULT_OCR_DEVICE)
    optimize_parser.add_argument("--background", action="store_true")
    optimize_parser.add_argument("--force", action="store_true")
    optimize_parser.add_argument("--dry-run", action="store_true")
    optimize_parser.set_defaults(handler=optimize)

    grade_parser = subparsers.add_parser(
        "grade", help="Grade one or more questions."
    )
    grade_parser.add_argument(
        "questions", nargs="+", type=normalize_question_id
    )
    grade_parser.add_argument(
        "--split",
        choices=["all", "calibration", "validation", "test"],
        default="test",
    )
    grade_parser.add_argument("--limit", type=int)
    grade_parser.add_argument("--device", default=DEFAULT_OCR_DEVICE)
    grade_parser.add_argument("--background", action="store_true")
    grade_parser.add_argument("--force", action="store_true")
    grade_parser.add_argument("--dry-run", action="store_true")
    grade_parser.set_defaults(handler=grade)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate one or more question checkpoints."
    )
    evaluate_parser.add_argument(
        "questions", nargs="+", type=normalize_question_id
    )
    evaluate_parser.add_argument("--export", action="store_true")
    evaluate_parser.add_argument("--detail", action="store_true")
    evaluate_parser.add_argument("--dry-run", action="store_true")
    evaluate_parser.set_defaults(handler=evaluate)

    monitor_parser = subparsers.add_parser(
        "monitor", help="Monitor one or more questions in one grading run."
    )
    monitor_parser.add_argument(
        "questions", nargs="+", type=normalize_question_id
    )
    monitor_parser.add_argument("--dry-run", action="store_true")
    monitor_parser.set_defaults(handler=monitor)

    outputs_parser = subparsers.add_parser(
        "outputs", help="Show expected output files for one or more questions."
    )
    outputs_parser.add_argument(
        "questions", nargs="+", type=normalize_question_id
    )
    outputs_parser.set_defaults(handler=show_outputs)

    publish_parser = subparsers.add_parser(
        "publish",
        help="Publish portable experiment artifacts to a separate Git repository.",
    )
    publish_parser.add_argument(
        "questions", nargs="+", type=normalize_question_id
    )
    publish_parser.add_argument(
        "--artifacts-repo",
        default=str((PROJECT_ROOT.parent / "refgrader-artifacts").resolve()),
    )
    publish_parser.add_argument(
        "--stage", choices=["auto", "rubric", "full"], default="auto"
    )
    publish_parser.add_argument("--run-id")
    publish_parser.add_argument("--include-raw-ocr", action="store_true")
    publish_parser.add_argument("--push", action="store_true")
    publish_parser.set_defaults(handler=publish)

    for action in ("status", "tail", "stop"):
        action_parser = subparsers.add_parser(
            action, help=f"Run run_experiment.sh {action}."
        )
        action_parser.add_argument("--dry-run", action="store_true")
        action_parser.set_defaults(handler=manage_background)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except (
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
