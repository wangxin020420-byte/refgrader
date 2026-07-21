"""Unified command-line entry point for CSBench experiments.

Examples:
    python scripts/run_csbench.py run CO_1 CO_2 CO_3 --force --background
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rubric_semantics import (
    RUBRIC_SEMANTIC_CONTRACT_VERSION,
    validate_refined_rubric,
)
from model_runtime import (
    DEFAULT_TEXT_MODEL_PROVIDER,
    DEFAULT_TEXT_THINKING_MODE,
    DEFAULT_VLM_MODEL_PROVIDER,
    TEXT_MODEL_PROFILES,
    VLM_MODEL_PROFILES,
    model_environment,
    runtime_model_config,
)

DEFAULT_OCR_DEVICE = "cpu" if os.name == "nt" else "gpu:0"
ACTIVE_RUBRIC_SET_SCHEMA_VERSION = 1
ACTIVE_RUBRIC_SET_NAME = "active_rubric_set.json"
ACTIVE_A3WA_CONFIG_NAME = "active_a3wa_config.json"


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--text-provider",
        choices=sorted(TEXT_MODEL_PROFILES),
        default=DEFAULT_TEXT_MODEL_PROVIDER,
        help="Text grading model profile. Default: glm47 (glm-4.7).",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=["enabled", "disabled"],
        default=DEFAULT_TEXT_THINKING_MODE,
        help="Text-model thinking mode. Default: disabled.",
    )
    parser.add_argument(
        "--vlm-provider",
        choices=sorted(VLM_MODEL_PROFILES),
        default=DEFAULT_VLM_MODEL_PROVIDER,
        help="Visual extraction model profile. Default: glm4v.",
    )


def model_config_from_args(args: argparse.Namespace) -> dict[str, str]:
    return runtime_model_config(
        text_provider=getattr(args, "text_provider", None),
        thinking_mode=getattr(args, "thinking_mode", None),
        vlm_provider=getattr(args, "vlm_provider", None),
    )


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
        allow_unchanged_baseline = bool(
            manifest.get("semantic_validation_mode")
            == "noninferiority_baseline_fallback"
            and manifest.get("decomposition_deferred") is True
            and manifest.get("fallback_reason")
            and not (manifest.get("candidate_replay") or {}).get("accepted", False)
        )

        initial_rubric = json.loads(
            self.initial_rubric.read_text(encoding="utf-8")
        )
        semantic_valid, semantic_errors = validate_refined_rubric(
            initial_rubric,
            rubric,
            question_total,
            allow_unchanged_baseline=allow_unchanged_baseline,
        )
        if not semantic_valid:
            raise ValueError(
                "The optimized rubric violates the active semantic contract: "
                + "; ".join(semantic_errors)
                + ". Re-run optimize with --force."
            )

        if (
            manifest.get("rubric_semantic_contract_version")
            != RUBRIC_SEMANTIC_CONTRACT_VERSION
        ):
            raise ValueError(
                "The optimized rubric uses an obsolete semantic contract. "
                "Re-run optimize with --force."
            )
        if manifest.get("semantic_policy_validated") is not True:
            raise ValueError(
                "The optimized rubric has no successful semantic-policy "
                "validation. Re-run optimize with --force."
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


def build_publish_command(
    args: argparse.Namespace,
    questions: list[str],
    *,
    stage: str,
    include_facts: bool = False,
    include_raw_ocr: bool = False,
    push: bool = False,
    run_id: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_csbench.py"),
        "--prepared-dir",
        args.prepared_dir,
        "publish",
        *questions,
        "--stage",
        stage,
        "--artifacts-repo",
        args.artifacts_repo,
    ]
    selected_run_id = run_id or getattr(args, "run_id", None)
    if selected_run_id:
        command.extend(["--run-id", selected_run_id])
    if include_facts:
        command.append("--include-facts")
    if include_raw_ocr:
        command.append("--include-raw-ocr")
    if getattr(args, "a3wa_config", None):
        command.extend(["--a3wa-config", args.a3wa_config])
    for question_id in getattr(args, "a3wa_config_questions", None) or []:
        command.extend(["--a3wa-config-question", question_id])
    if push:
        command.append("--push")
    return command


def build_evaluate_command(
    args: argparse.Namespace,
    questions: list[str],
    *,
    a3wa_config_questions: list[str] | None = None,
    run_id: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_csbench.py"),
        "--prepared-dir",
        args.prepared_dir,
        "evaluate",
        *questions,
        "--export",
        "--artifacts-repo",
        args.artifacts_repo,
    ]
    selected_run_id = run_id or getattr(args, "run_id", None)
    if selected_run_id:
        command.extend(["--run-id", selected_run_id])
    if getattr(args, "a3wa_config", None):
        command.extend(["--a3wa-config", args.a3wa_config])
    for question_id in a3wa_config_questions or []:
        command.extend(["--a3wa-config-question", question_id])
    if args.include_raw_ocr:
        command.append("--include-raw-ocr")
    if args.include_facts:
        command.append("--include-facts")
    if args.push_artifacts:
        command.append("--push-artifacts")
    return command


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def completed_questions_for_split(
    contexts: list[CSBenchContext],
    results_dir: Path,
    *,
    split_name: str,
) -> set[str]:
    completed = set()
    for ctx in contexts:
        split = json.loads(ctx.split_file.read_text(encoding="utf-8"))
        expected_ids = {str(value) for value in split.get(split_name, [])}
        checkpoint = load_json_records(
            results_dir / f"{ctx.question_id}_grading_checkpoint.json"
        )
        checkpoint_ids = [
            str(item.get("student_id", "")) for item in checkpoint
        ]
        checkpoint_set = {value for value in checkpoint_ids if value}
        if (
            checkpoint_set == expected_ids
            and len(checkpoint_ids) == len(checkpoint_set)
        ):
            completed.add(ctx.question_id)
    return completed


def inspect_results(
    contexts: list[CSBenchContext],
    results_dir: Path,
    *,
    split_name: str,
) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    structural_errors = []
    for ctx in contexts:
        split = json.loads(ctx.split_file.read_text(encoding="utf-8"))
        expected_ids = {str(value) for value in split.get(split_name, [])}
        prefix = results_dir / ctx.question_id
        checkpoint_path = Path(f"{prefix}_grading_checkpoint.json")
        graded_path = Path(f"{prefix}_graded_results.json")
        rejected_path = Path(f"{prefix}_rejected.json")
        failed_path = Path(f"{prefix}_failed.json")

        checkpoint = load_json_records(checkpoint_path)
        graded = load_json_records(graded_path)
        rejected = load_json_records(rejected_path)
        failed = load_json_records(failed_path)

        checkpoint_ids = [str(item.get("student_id", "")) for item in checkpoint]
        checkpoint_set = {value for value in checkpoint_ids if value}
        graded_set = {
            str(item.get("student_id")) for item in graded if item.get("student_id")
        }
        rejected_set = {
            str(item.get("student_id")) for item in rejected if item.get("student_id")
        }
        failed_set = {
            str(item.get("student_id")) for item in failed if item.get("student_id")
        }

        issues = []
        duplicate_count = len(checkpoint_ids) - len(checkpoint_set)
        if duplicate_count:
            issues.append(f"checkpoint duplicates={duplicate_count}")
        missing = expected_ids - checkpoint_set
        outside = checkpoint_set - expected_ids
        if outside:
            issues.append(f"answers outside {split_name}={len(outside)}")
        overlap = graded_set & rejected_set
        if overlap:
            issues.append(f"graded/rejected overlap={len(overlap)}")
        result_union = graded_set | rejected_set
        if result_union != checkpoint_set:
            issues.append(
                "checkpoint/result mismatch="
                f"{len(result_union ^ checkpoint_set)}"
            )
        unresolved_failed = failed_set - checkpoint_set
        if any(not item.get("student_id") for item in failed):
            issues.append("failed records contain missing student IDs")

        if issues:
            structural_errors.append(
                f"{ctx.question_id}: " + "; ".join(issues)
            )

        complete = not missing and not unresolved_failed and not issues
        questions[ctx.question_id] = {
            "expected_count": len(expected_ids),
            "checkpoint_count": len(checkpoint_set),
            "graded_count": len(graded_set),
            "rejected_count": len(rejected_set),
            "failed_count": len(unresolved_failed),
            "missing_count": len(missing),
            "missing_ids": sorted(missing),
            "failed_ids": sorted(unresolved_failed),
            "outside_ids": sorted(outside),
            "duplicate_count": duplicate_count,
            "status": "complete" if complete else "partial",
        }

        print(
            f"Inspected {ctx.question_id} {split_name}: "
            f"checkpoint={len(checkpoint_set)}/{len(expected_ids)}, "
            f"graded={len(graded_set)}, rejected={len(rejected_set)}, "
            f"failed={len(unresolved_failed)}"
        )

    expected_total = sum(item["expected_count"] for item in questions.values())
    checkpoint_total = sum(
        item["checkpoint_count"] for item in questions.values()
    )
    report = {
        "schema_version": 1,
        "answer_split": split_name,
        "status": (
            "complete"
            if questions
            and all(item["status"] == "complete" for item in questions.values())
            and not structural_errors
            else "partial"
        ),
        "expected_total": expected_total,
        "checkpoint_total": checkpoint_total,
        "coverage": (
            checkpoint_total / expected_total if expected_total else 1.0
        ),
        "questions": questions,
        "structural_errors": structural_errors,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return report


def validate_result_structure(report: dict[str, Any]) -> None:
    errors = report.get("structural_errors") or []
    if errors:
        raise RuntimeError(
            "Refusing to evaluate or publish structurally invalid results:\n"
            + "\n".join(errors)
        )


def validate_complete_results(
    contexts: list[CSBenchContext],
    results_dir: Path,
    *,
    split_name: str,
) -> None:
    report = inspect_results(contexts, results_dir, split_name=split_name)
    errors = list(report.get("structural_errors") or [])
    if report["status"] != "complete":
        for question_id, item in report["questions"].items():
            issues = []
            if item["missing_count"]:
                issues.append(
                    f"missing {split_name} answers={item['missing_count']}"
                )
            if item["failed_count"]:
                issues.append(
                    f"unresolved failed answers={item['failed_count']}"
                )
            if issues:
                errors.append(f"{question_id}: " + "; ".join(issues))
    if errors:
        raise RuntimeError(
            "Refusing to use incomplete results for this operation:\n"
            + "\n".join(errors)
        )


def write_completion_report(
    results_dir: Path, report: dict[str, Any]
) -> Path:
    path = results_dir / "completion_report.json"
    _write_json_atomic(path, report)
    return path


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
            compared = normalized.casefold() if os.name == "nt" else normalized
            source_compared = (
                source_normalized.casefold()
                if os.name == "nt"
                else source_normalized
            )
            if compared == source_compared:
                return replacement
            if compared.startswith(source_compared + "/"):
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


def prepare_artifact_destination(
    destination: Path,
    *,
    question_id: str,
    run_id: str,
    stage_dir: str,
) -> Path:
    """Build an artifact update off to the side before replacing the run."""
    if destination.exists():
        manifest_path = destination / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(
                f"Existing artifact run has no manifest: {destination}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        identity = (
            manifest.get("question_id"),
            manifest.get("run_id"),
            manifest.get("artifact_stage_dir"),
        )
        if identity != (question_id, run_id, stage_dir):
            raise FileExistsError(
                f"Artifact run identity mismatch: {destination}"
            )
    temporary = destination.parent / f".{run_id}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    return temporary


def commit_artifact_destination(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        os.replace(temporary, destination)
        return
    backup = destination.parent / f".{destination.name}.bak-{os.getpid()}"
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except Exception:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup)


def optimization_evidence_paths(
    ctx: CSBenchContext, fallback_dir: Path
) -> tuple[Path | None, Path | None]:
    """Locate optimization evidence recorded by the rubric manifest.

    New manifests identify the exact variance checkpoint and its hash. This
    prevents a single-question publication from accidentally attaching an old
    checkpoint after the rubric was optimized as part of a larger batch.
    Legacy manifests fall back to the historical derived directory.
    """
    manifest = json.loads(
        ctx.optimization_manifest.read_text(encoding="utf-8")
    )
    recorded = manifest.get("variance_checkpoint")
    recorded_hash = manifest.get("variance_checkpoint_sha256")
    if recorded and recorded_hash:
        checkpoint = resolve_portable_path(
            str(recorded), prepared_root=getattr(ctx, "root", PROJECT_ROOT)
        )
        if checkpoint.is_file() and sha256_file(checkpoint) == recorded_hash:
            progress = checkpoint.parent / "progress.json"
            return checkpoint, progress if progress.is_file() else None
        return None, None

    checkpoint = fallback_dir / f"{ctx.question_id}_variance_checkpoint.json"
    progress = fallback_dir / "progress.json"
    return (
        checkpoint if checkpoint.is_file() else None,
        progress if progress.is_file() else None,
    )


def resolve_portable_path(value: str, *, prepared_root: Path) -> Path:
    """Resolve a project-portable path stored in tracked configuration."""
    normalized = str(value).replace("\\", "/")
    replacements = {
        "${REFGRADER_ROOT}": PROJECT_ROOT,
        "${PREPARED_CSBENCH_ROOT}": prepared_root,
    }
    for marker, root in replacements.items():
        if normalized == marker:
            return root.resolve()
        if normalized.startswith(marker + "/"):
            return (root / normalized[len(marker) + 1 :]).resolve()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def active_rubric_set_path(ctx: CSBenchContext) -> Path:
    return ctx.root / "rubrics" / ACTIVE_RUBRIC_SET_NAME


def active_a3wa_config_path(ctx: CSBenchContext) -> Path:
    return ctx.root / "calibration" / ACTIVE_A3WA_CONFIG_NAME


def normalize_optimization_manifest(ctx: CSBenchContext) -> None:
    """Remove device-specific absolute paths from a generated manifest."""
    if not ctx.optimization_manifest.is_file():
        raise FileNotFoundError(
            f"Optimization manifest not found: {ctx.optimization_manifest}"
        )
    payload = json.loads(
        ctx.optimization_manifest.read_text(encoding="utf-8-sig")
    )
    payload = portable_value(
        payload,
        [
            (str(ctx.root), "${PREPARED_CSBENCH_ROOT}"),
            (str(PROJECT_ROOT), "${REFGRADER_ROOT}"),
        ],
    )
    payload["path_format"] = "refgrader_placeholders_v1"
    _write_json_atomic(ctx.optimization_manifest, payload)


def _dataset_snapshot_hashes(ctx: CSBenchContext) -> dict[str, str]:
    files = [ctx.database, ctx.teacher_db, ctx.answer_metadata]
    for name in ("manifest.json", "embedded_manifest.json"):
        candidate = ctx.root / name
        if candidate.is_file():
            files.append(candidate)
    return {
        path.relative_to(ctx.root).as_posix(): sha256_file(path)
        for path in files
    }


def _active_rubric_entry(ctx: CSBenchContext) -> dict[str, Any]:
    manifest = json.loads(
        ctx.optimization_manifest.read_text(encoding="utf-8-sig")
    )
    return {
        "question_id": ctx.question_id,
        "rubric_group": ctx.group,
        "semantic_contract_version": RUBRIC_SEMANTIC_CONTRACT_VERSION,
        "initial_rubric": (
            ctx.initial_rubric.relative_to(ctx.root).as_posix()
        ),
        "optimized_rubric": (
            ctx.optimized_rubric.relative_to(ctx.root).as_posix()
        ),
        "optimization_manifest": (
            ctx.optimization_manifest.relative_to(ctx.root).as_posix()
        ),
        "initial_sha256": sha256_file(ctx.initial_rubric),
        "optimized_sha256": sha256_file(ctx.optimized_rubric),
        "optimization_manifest_sha256": sha256_file(
            ctx.optimization_manifest
        ),
        "split_sha256": sha256_file(ctx.split_file),
        "optimization_created_at": manifest.get("created_at"),
        "calibration_answer_ids": manifest.get("calibration_answer_ids", []),
    }


def _all_valid_rubric_contexts(seed: CSBenchContext) -> list[CSBenchContext]:
    exam = json.loads(seed.database.read_text(encoding="utf-8-sig"))
    valid = []
    for item in exam:
        question_id = normalize_question_id(item.get("question_id", ""))
        if not question_id:
            continue
        ctx = CSBenchContext(str(seed.root), question_id)
        if not (
            ctx.optimized_rubric.is_file()
            and ctx.optimization_manifest.is_file()
        ):
            continue
        try:
            normalize_optimization_manifest(ctx)
            ctx.validate_optimized()
        except (FileNotFoundError, ValueError):
            continue
        valid.append(ctx)
    return valid


def _active_a3wa_metadata(
    contexts: list[CSBenchContext],
    config_path: Path,
    *,
    source_validation_run_id: str | None,
) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    return {
        "status": "active",
        "path": (
            config_path.relative_to(contexts[0].root).as_posix()
        ),
        "sha256": sha256_file(config_path),
        "questions": [ctx.question_id for ctx in contexts],
        "optimized_rubric_sha256": {
            ctx.question_id: sha256_file(ctx.optimized_rubric)
            for ctx in contexts
        },
        "validation_split_sha256": {
            ctx.question_id: sha256_file(ctx.split_file) for ctx in contexts
        },
        "source_validation_run_id": source_validation_run_id,
        "score_calibration_enabled": bool(
            (payload.get("score_calibration") or {}).get("enabled", False)
        ),
        "model_config": payload.get("model_config"),
    }


def refresh_active_configuration(
    contexts: list[CSBenchContext],
    *,
    a3wa_config: Path | None = None,
    source_validation_run_id: str | None = None,
) -> Path:
    """Write the tracked current rubric/A3WA configuration atomically."""
    if not contexts:
        raise ValueError("At least one question is required to activate rubrics.")
    for ctx in contexts:
        normalize_optimization_manifest(ctx)
        ctx.validate_optimized()

    seed = contexts[0]
    bundle_path = active_rubric_set_path(seed)
    existing = {}
    if bundle_path.is_file():
        existing = json.loads(bundle_path.read_text(encoding="utf-8-sig"))

    valid_contexts = _all_valid_rubric_contexts(seed)
    entries = {
        ctx.question_id: _active_rubric_entry(ctx) for ctx in valid_contexts
    }
    active_config = active_a3wa_config_path(seed)
    a3wa_metadata = existing.get("active_a3wa")

    if a3wa_config is not None:
        source = Path(a3wa_config).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"A3WA calibration config not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        payload = portable_value(
            payload,
            [
                (str(seed.root), "${PREPARED_CSBENCH_ROOT}"),
                (str(PROJECT_ROOT), "${REFGRADER_ROOT}"),
            ],
        )
        _write_json_atomic(active_config, payload)
        a3wa_metadata = _active_a3wa_metadata(
            contexts,
            active_config,
            source_validation_run_id=source_validation_run_id,
        )
    elif isinstance(a3wa_metadata, dict):
        expected_rubrics = a3wa_metadata.get("optimized_rubric_sha256", {})
        reasons = []
        if not active_config.is_file():
            reasons.append("active A3WA file is missing")
        elif a3wa_metadata.get("sha256") != sha256_file(active_config):
            reasons.append("active A3WA file hash changed")
        for question_id, expected_hash in expected_rubrics.items():
            entry = entries.get(question_id)
            if not entry or entry.get("optimized_sha256") != expected_hash:
                reasons.append(f"{question_id} optimized rubric changed")
        if reasons:
            a3wa_metadata = dict(a3wa_metadata)
            a3wa_metadata["status"] = "stale"
            a3wa_metadata["stale_reasons"] = sorted(set(reasons))

    bundle = {
        "schema_version": ACTIVE_RUBRIC_SET_SCHEMA_VERSION,
        "semantic_contract_version": RUBRIC_SEMANTIC_CONTRACT_VERSION,
        "prepared_root": portable_value(
            str(seed.root), [(str(PROJECT_ROOT), "${REFGRADER_ROOT}")]
        ),
        "dataset_sha256": _dataset_snapshot_hashes(seed),
        "questions": entries,
        "active_a3wa": a3wa_metadata,
    }
    existing_comparable = dict(existing)
    existing_comparable.pop("updated_at", None)
    if existing_comparable == bundle:
        return bundle_path
    bundle["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json_atomic(bundle_path, bundle)
    return bundle_path


def validate_active_configuration(
    contexts: list[CSBenchContext],
) -> dict[str, Any]:
    if not contexts:
        raise ValueError("At least one question is required.")
    bundle_path = active_rubric_set_path(contexts[0])
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Active rubric set not found: {bundle_path}. Run optimize or "
            "restore/activate a verified rubric run first."
        )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8-sig"))
    if bundle.get("schema_version") != ACTIVE_RUBRIC_SET_SCHEMA_VERSION:
        raise ValueError("Unsupported active rubric set schema version.")
    if bundle.get("semantic_contract_version") != RUBRIC_SEMANTIC_CONTRACT_VERSION:
        raise ValueError("Active rubric set uses an obsolete semantic contract.")
    if bundle.get("dataset_sha256") != _dataset_snapshot_hashes(contexts[0]):
        raise ValueError(
            "Active rubric set was generated from a different CSBench snapshot. "
            "Re-run optimize or restore the matching configuration."
        )
    entries = bundle.get("questions") or {}
    for ctx in contexts:
        ctx.validate_optimized()
        entry = entries.get(ctx.question_id)
        if not entry:
            raise ValueError(
                f"{ctx.question_id} is not registered in the active rubric set."
            )
        checks = {
            "initial_sha256": sha256_file(ctx.initial_rubric),
            "optimized_sha256": sha256_file(ctx.optimized_rubric),
            "optimization_manifest_sha256": sha256_file(
                ctx.optimization_manifest
            ),
            "split_sha256": sha256_file(ctx.split_file),
        }
        for key, actual in checks.items():
            if entry.get(key) != actual:
                raise ValueError(
                    f"Active rubric set hash mismatch for {ctx.question_id}: {key}."
                )
    return bundle


def resolve_active_a3wa_config(
    contexts: list[CSBenchContext],
    *,
    model_config: dict[str, str] | None = None,
) -> Path | None:
    bundle = validate_active_configuration(contexts)
    metadata = bundle.get("active_a3wa")
    if not isinstance(metadata, dict) or metadata.get("status") != "active":
        return None
    requested = {ctx.question_id for ctx in contexts}
    covered = set(metadata.get("questions") or [])
    if not requested.issubset(covered):
        return None
    path = active_a3wa_config_path(contexts[0])
    if not path.is_file() or metadata.get("sha256") != sha256_file(path):
        raise ValueError("Tracked active A3WA config does not match its manifest.")
    expected = metadata.get("optimized_rubric_sha256") or {}
    for ctx in contexts:
        if expected.get(ctx.question_id) != sha256_file(ctx.optimized_rubric):
            raise ValueError(
                f"Active A3WA config is stale for {ctx.question_id}. Re-run "
                "validation and calibrate."
            )
    expected_model = metadata.get("model_config")
    requested_model = model_config or runtime_model_config()
    if expected_model != requested_model:
        raise ValueError(
            "Active A3WA config was calibrated with a different text/VLM "
            "model or thinking mode. Re-run validation and calibrate under "
            "the requested model configuration."
        )
    return path


def validate_a3wa_model_config(
    config_path: str | Path,
    model_config: dict[str, str],
) -> None:
    path = Path(config_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("model_config") != model_config:
        raise ValueError(
            f"A3WA config model contract does not match this run: {path}. "
            "Re-run validation and calibrate with the same model and thinking "
            "mode before test grading."
        )


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


def grading_results_dir(
    contexts: list["CSBenchContext"], answer_split: str
) -> Path:
    """Return the batch root used to hold versioned grading runs.

    Historical checkpoints may still live directly in this directory. New
    runs are stored below ``runs/<run_id>`` and selected through
    ``active_run.json`` so a fresh ``--force`` run cannot overwrite history.
    """
    slug = batch_slug(contexts)
    suffix = "full" if answer_split == "test" else answer_split
    return (
        PROJECT_ROOT / "results_runs" / f"csbench_{slug}_{suffix}"
    ).resolve()


RUN_STATE_SCHEMA_VERSION = 1


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _new_grading_run_id(batch_root: Path) -> str:
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base
    index = 1
    while (batch_root / "runs" / candidate).exists():
        candidate = f"{base}_{index:02d}"
        index += 1
    return candidate


def _run_signature(
    contexts: list["CSBenchContext"],
    answer_split: str,
    a3wa_config: str | None,
    model_config: dict[str, str] | None = None,
) -> dict[str, Any]:
    config_path = (
        Path(a3wa_config).expanduser().resolve() if a3wa_config else None
    )
    if config_path and not config_path.is_file():
        raise FileNotFoundError(f"A3WA calibration config not found: {config_path}")
    return {
        "questions": [ctx.question_id for ctx in contexts],
        "answer_split": answer_split,
        "split_sha256": {
            ctx.question_id: sha256_file(ctx.split_file) for ctx in contexts
        },
        "optimized_rubric_sha256": {
            ctx.question_id: sha256_file(ctx.optimized_rubric)
            for ctx in contexts
        },
        "a3wa_config_sha256": sha256_file(config_path) if config_path else None,
        "model_config": model_config or runtime_model_config(),
    }


def _run_state_path(results_dir: Path) -> Path:
    return results_dir / "run_state.json"


def _read_run_state(results_dir: Path) -> dict[str, Any] | None:
    path = _run_state_path(results_dir)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid run state: {path}")
    return payload


def _active_run_path(batch_root: Path) -> Path:
    return batch_root / "active_run.json"


def _activate_run(batch_root: Path, run_id: str, results_dir: Path) -> None:
    relative = results_dir.relative_to(batch_root).as_posix()
    _write_json_atomic(
        _active_run_path(batch_root),
        {
            "schema_version": RUN_STATE_SCHEMA_VERSION,
            "run_id": run_id,
            "relative_path": relative,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _legacy_has_results(batch_root: Path, contexts: list["CSBenchContext"]) -> bool:
    return any(
        (batch_root / f"{ctx.question_id}_grading_checkpoint.json").is_file()
        for ctx in contexts
    )


def select_grading_run(
    contexts: list["CSBenchContext"],
    answer_split: str,
    *,
    run_id: str | None = None,
    force_new: bool = False,
    create: bool = False,
    a3wa_config: str | None = None,
    model_config: dict[str, str] | None = None,
) -> tuple[Path, str | None]:
    """Select a versioned run while preserving legacy result compatibility."""
    batch_root = grading_results_dir(contexts, answer_split)
    signature = None

    if force_new:
        signature = _run_signature(
            contexts, answer_split, a3wa_config, model_config
        )
        selected_id = run_id or _new_grading_run_id(batch_root)
        results_dir = batch_root / "runs" / selected_id
        if results_dir.exists():
            raise FileExistsError(
                f"Grading run already exists: {results_dir}. Choose another "
                "--run-id or omit it to generate a timestamp automatically."
            )
        if create:
            results_dir.mkdir(parents=True, exist_ok=False)
            now = datetime.now().isoformat(timespec="seconds")
            _write_json_atomic(
                _run_state_path(results_dir),
                {
                    "schema_version": RUN_STATE_SCHEMA_VERSION,
                    "run_id": selected_id,
                    "status": "created",
                    "created_at": now,
                    "updated_at": now,
                    "signature": signature,
                },
            )
            _activate_run(batch_root, selected_id, results_dir)
        return results_dir, selected_id

    selected_id = run_id
    results_dir = None
    if selected_id:
        candidate = batch_root / "runs" / selected_id
        if candidate.is_dir():
            results_dir = candidate
        elif not create:
            raise FileNotFoundError(f"Grading run not found: {candidate}")
        else:
            results_dir = candidate
    else:
        active_path = _active_run_path(batch_root)
        if active_path.is_file():
            active = json.loads(active_path.read_text(encoding="utf-8-sig"))
            selected_id = str(active.get("run_id") or "") or None
            relative = str(active.get("relative_path") or "")
            candidate = (batch_root / relative).resolve() if relative else None
            if candidate and candidate.is_dir():
                results_dir = candidate
        if results_dir is None and _legacy_has_results(batch_root, contexts):
            results_dir = batch_root
            state = _read_run_state(results_dir)
            selected_id = str(state.get("run_id")) if state else "legacy"

    if results_dir is None:
        if not create:
            return batch_root, None
        selected_id = _new_grading_run_id(batch_root)
        results_dir = batch_root / "runs" / selected_id

    if create:
        signature = _run_signature(
            contexts, answer_split, a3wa_config, model_config
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        state = _read_run_state(results_dir)
        if state:
            if state.get("signature") != signature:
                raise RuntimeError(
                    "The selected grading run was created with a different "
                    "split, rubric, A3WA config, or model contract. Start a "
                    "new run with --force."
                )
        else:
            now = datetime.now().isoformat(timespec="seconds")
            _write_json_atomic(
                _run_state_path(results_dir),
                {
                    "schema_version": RUN_STATE_SCHEMA_VERSION,
                    "run_id": selected_id,
                    "status": "created",
                    "created_at": now,
                    "updated_at": now,
                    "signature": signature,
                    "legacy_layout": results_dir == batch_root,
                },
            )
        _activate_run(batch_root, str(selected_id), results_dir)
    return results_dir, selected_id


def update_run_state(
    results_dir: Path,
    *,
    status: str,
    completion: dict[str, Any] | None = None,
) -> None:
    state = _read_run_state(results_dir)
    if not state:
        return
    state["status"] = status
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if completion is not None:
        state["completion"] = completion
    _write_json_atomic(_run_state_path(results_dir), state)


def register_restored_run(
    contexts: list["CSBenchContext"],
    answer_split: str,
    run_id: str,
    results_dir: Path,
    *,
    a3wa_config: str | None = None,
    completion: dict[str, Any] | None = None,
    model_config: dict[str, str] | None = None,
) -> None:
    batch_root = grading_results_dir(contexts, answer_split)
    now = datetime.now().isoformat(timespec="seconds")
    existing = _read_run_state(results_dir) or {}
    _write_json_atomic(
        _run_state_path(results_dir),
        {
            **existing,
            "schema_version": RUN_STATE_SCHEMA_VERSION,
            "run_id": run_id,
            "status": completion.get("status", "restored")
            if completion
            else "restored",
            "created_at": existing.get("created_at", now),
            "updated_at": now,
            "signature": _run_signature(
                contexts, answer_split, a3wa_config, model_config
            ),
            "completion": completion,
            "restored_from_artifacts": True,
        },
    )
    _activate_run(batch_root, run_id, results_dir)


RESULT_ARTIFACT_STAGES = {
    "validation": ("validation", "validation_runs"),
    "calibration": ("calibration", "calibration_runs"),
    "full": ("test", "grading_runs"),
}


def build_run_command(args: argparse.Namespace, *, background: bool) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_csbench.py"),
        "--prepared-dir",
        args.prepared_dir,
        "run",
        *args.questions,
    ]
    if args.dataset_root:
        command.extend(["--dataset-root", args.dataset_root])
        command.extend(["--link-mode", args.link_mode])
        if args.exclude_questions:
            command.extend(["--exclude-questions", *args.exclude_questions])
    if args.sample_size != 5:
        command.extend(["--sample-size", str(args.sample_size)])
    if args.split != "test":
        command.extend(["--split", args.split])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.device != DEFAULT_OCR_DEVICE:
        command.extend(["--device", args.device])
    command.extend(["--text-provider", args.text_provider])
    command.extend(["--thinking-mode", args.thinking_mode])
    command.extend(["--vlm-provider", args.vlm_provider])
    if getattr(args, "a3wa_config", None):
        command.extend(["--a3wa-config", args.a3wa_config])
    if getattr(args, "no_active_a3wa", False):
        command.append("--no-active-a3wa")
    if background:
        command.append("--background")
    if args.force:
        command.append("--force")
    if getattr(args, "run_id", None):
        command.extend(["--run-id", args.run_id])
    if args.dry_run:
        command.append("--dry-run")
    command.extend(["--artifacts-repo", args.artifacts_repo])
    if args.no_artifacts:
        command.append("--no-artifacts")
    if args.push_artifacts:
        command.append("--push-artifacts")
    if args.include_raw_ocr:
        command.append("--include-raw-ocr")
    if args.include_facts:
        command.append("--include-facts")
    return command


def start_run_in_background(args: argparse.Namespace) -> int:
    if os.name == "nt":
        raise RuntimeError("--background is only supported on Linux.")
    ensure_background_slot_available()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"experiment_{run_id}.log"
    pid_file = log_dir / "refgrader.pid"
    command = build_run_command(args, background=False)
    env = os.environ.copy()
    env["REFGRADER_ARTIFACT_RUN_ID"] = run_id
    env["REFGRADER_RUN_BACKGROUND_CHILD"] = "1"
    wrapper = [
        "bash",
        "-c",
        """
child_pid=""
terminate_child() {
    if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    exit 143
}
trap terminate_child TERM INT
"$@" &
child_pid=$!
wait "$child_pid"
status=$?
child_pid=""
trap - TERM INT
exit "$status"
""",
        "_",
        *command,
    ]
    with log_file.open("ab") as handle:
        process = subprocess.Popen(
            wrapper,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(f"{log_file}\n{process.pid}\n", encoding="utf-8")
    print(f"[{run_id}] CSBench workflow started in background.")
    print(f"  PID: {process.pid}")
    print(f"  Log: {log_file}")
    print("")
    print("  Manage:")
    print("    python scripts/run_csbench.py status")
    print("    python scripts/run_csbench.py tail")
    print("    python scripts/run_csbench.py stop")
    return 0


def optimize(args: argparse.Namespace) -> int:
    model_config = model_config_from_args(args)
    requested_contexts = build_contexts(args.prepared_dir, args.questions)
    for ctx in requested_contexts:
        ctx.validate_initial(args.sample_size)
    slug = batch_slug(requested_contexts)
    resume = bool(getattr(args, "resume", False))

    contexts = requested_contexts
    if resume:
        pending_contexts = []
        for ctx in requested_contexts:
            try:
                ctx.validate_optimized()
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                pending_contexts.append(ctx)
                print(f"Resume {ctx.question_id}: pending ({exc})")
            else:
                print(
                    f"Resume {ctx.question_id}: already complete under semantic "
                    f"contract v{RUBRIC_SEMANTIC_CONTRACT_VERSION}; skipped."
                )
        contexts = pending_contexts
        if not contexts:
            bundle = refresh_active_configuration(requested_contexts)
            print("All requested rubric optimizations are already complete.")
            print(f"Active rubric set updated: {bundle}")
            return 0

    existing = [
        ctx.question_id
        for ctx in contexts
        if ctx.optimized_rubric.exists() or ctx.optimization_manifest.exists()
    ]
    if existing and not args.force and not resume:
        raise FileExistsError(
            "Optimized rubric already exists for "
            + ", ".join(existing)
            + ". "
            "Use --force to regenerate it."
        )

    results_dir = (
        PROJECT_ROOT / "results_runs" / f"csbench_{slug}_rubric_opt"
    ).resolve()
    pipeline_args = [
        "--mode",
        "VARIANCE_OPT",
        *main_pipeline_base(contexts),
        "--teacher-db",
        str(contexts[0].teacher_db),
        "--sample-size",
        str(args.sample_size),
        "--results-dir",
        str(results_dir),
        "--progress-file",
        str(results_dir / "progress.json"),
    ]
    if args.force:
        pipeline_args.append("--force-rerun")
    elif resume:
        pipeline_args.append("--resume-optimization")

    env = {
        "REFGRADER_OCR_DEVICE": args.device,
        # Never let an old shell-level config leak into a new pipeline stage.
        # A non-empty path is set only by an explicit command argument.
        "A3WA_CALIBRATION_CONFIG": "",
        **model_environment(model_config),
    }
    if getattr(args, "a3wa_config", None):
        env["A3WA_CALIBRATION_CONFIG"] = str(
            Path(args.a3wa_config).expanduser().resolve()
        )
    if args.background:
        if os.name == "nt":
            raise RuntimeError("--background is only supported on Linux.")
        ensure_background_slot_available()
        if not args.no_artifacts:
            env["REFGRADER_POST_SUCCESS_CMD"] = shlex.join(
                build_publish_command(
                    args,
                    args.questions,
                    stage="rubric",
                    include_facts=args.include_facts,
                    push=args.push_artifacts,
                )
            )
        else:
            env["REFGRADER_POST_SUCCESS_CMD"] = shlex.join(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_csbench.py"),
                    "--prepared-dir",
                    args.prepared_dir,
                    "activate",
                    *args.questions,
                ]
            )
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
    return_code = execute(
        command, env_overrides=env, dry_run=args.dry_run
    )
    if return_code != 0 or args.dry_run:
        return return_code
    if args.background:
        if args.no_artifacts:
            print(
                "Background optimization started. The tracked active rubric "
                "set will be refreshed after the job finishes successfully."
            )
        else:
            print(
                "Background optimization started. Rubric artifacts will be "
                "published automatically after the job finishes successfully."
            )
        return return_code
    bundle = refresh_active_configuration(requested_contexts)
    print(f"Active rubric set updated: {bundle}")
    if args.no_artifacts:
        return return_code
    print("Optimization finished; publishing rubric artifacts automatically.")
    return publish(
        argparse.Namespace(
            prepared_dir=args.prepared_dir,
            questions=[ctx.question_id for ctx in contexts],
            artifacts_repo=args.artifacts_repo,
            stage="rubric",
            run_id=None,
            include_raw_ocr=False,
            include_facts=args.include_facts,
            push=args.push_artifacts,
        )
    )


def grade(args: argparse.Namespace) -> int:
    model_config = model_config_from_args(args)
    contexts = build_contexts(args.prepared_dir, args.questions)
    for ctx in contexts:
        ctx.validate_optimized()
    validate_active_configuration(contexts)
    explicit_a3wa = getattr(args, "a3wa_config", None)
    disable_active_a3wa = bool(getattr(args, "no_active_a3wa", False))
    if explicit_a3wa and disable_active_a3wa:
        raise ValueError("Use either --a3wa-config or --no-active-a3wa, not both.")
    if args.split != "test" and explicit_a3wa:
        raise ValueError(
            "Validation/calibration grading must be uncalibrated; "
            "--a3wa-config is allowed only for test."
        )
    if args.split == "test" and explicit_a3wa:
        validate_a3wa_model_config(explicit_a3wa, model_config)
    if args.split == "test" and not explicit_a3wa and not disable_active_a3wa:
        active_config = resolve_active_a3wa_config(
            contexts, model_config=model_config
        )
        if active_config:
            args.a3wa_config = str(active_config)
            print(f"Using tracked active A3WA config: {active_config}")
        else:
            raise ValueError(
                "No valid active A3WA config covers this test batch. Run "
                "validation and calibrate first, restore a matching calibrated "
                "run, or pass --no-active-a3wa explicitly for an uncalibrated "
                "ablation."
            )
    results_dir, local_run_id = select_grading_run(
        contexts,
        args.split,
        run_id=getattr(args, "run_id", None),
        force_new=bool(args.force),
        create=not args.dry_run,
        a3wa_config=getattr(args, "a3wa_config", None),
        model_config=model_config,
    )
    preexisting_complete = (
        completed_questions_for_split(
            contexts,
            results_dir,
            split_name="test",
        )
        if args.split == "test"
        else set()
    )
    run_state = _read_run_state(results_dir)
    a3wa_config_questions = []
    if getattr(args, "a3wa_config", None):
        if run_state and not run_state.get("legacy_layout"):
            a3wa_config_questions = [ctx.question_id for ctx in contexts]
        else:
            a3wa_config_questions = [
                ctx.question_id
                for ctx in contexts
                if ctx.question_id not in preexisting_complete
            ]
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
    if local_run_id:
        pipeline_args.extend(["--run-id", str(local_run_id)])
    if args.limit is not None:
        pipeline_args.extend(["--img-limit", str(args.limit)])
    if args.force:
        pipeline_args.append("--force-rerun")

    env = {
        "REFGRADER_OCR_DEVICE": args.device,
        # Validation must be uncalibrated, and test must use only the config
        # explicitly supplied to this command.
        "A3WA_CALIBRATION_CONFIG": "",
        **model_environment(model_config),
    }
    if getattr(args, "a3wa_config", None):
        env["A3WA_CALIBRATION_CONFIG"] = str(
            Path(args.a3wa_config).expanduser().resolve()
        )
    if args.background:
        if os.name == "nt":
            raise RuntimeError("--background is only supported on Linux.")
        ensure_background_slot_available()
        if not args.no_artifacts and args.limit is None:
            if args.split == "test":
                env["REFGRADER_POST_SUCCESS_CMD"] = shlex.join(
                    build_evaluate_command(
                        args,
                        args.questions,
                        a3wa_config_questions=a3wa_config_questions,
                        run_id=local_run_id,
                    )
                )
            elif args.split in ("validation", "calibration"):
                env["REFGRADER_POST_SUCCESS_CMD"] = shlex.join(
                    build_publish_command(
                        args,
                        args.questions,
                        stage=args.split,
                        include_facts=args.include_facts,
                        include_raw_ocr=args.include_raw_ocr,
                        push=args.push_artifacts,
                        run_id=local_run_id,
                    )
                )
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
    print(
        "Model contract: "
        f"{model_config['text_model']} "
        f"(thinking={model_config['text_thinking']}), "
        f"VLM={model_config['vlm_model']}"
    )
    print(f"Run ID: {local_run_id or 'legacy/unversioned'}")
    print(f"Results: {results_dir}")
    if not args.dry_run:
        update_run_state(results_dir, status="running")
    return_code = execute(command, env_overrides=env, dry_run=args.dry_run)
    if return_code != 0:
        update_run_state(results_dir, status="interrupted")
        return return_code
    if args.dry_run:
        return return_code
    if args.background:
        print(
            f"Background {args.split} grading started for run "
            f"{local_run_id}. Finalization will inspect coverage, evaluate "
            "test results when applicable, and update the same artifact run."
        )
        return return_code
    report = inspect_results(contexts, results_dir, split_name=args.split)
    validate_result_structure(report)
    write_completion_report(results_dir, report)
    update_run_state(results_dir, status=report["status"], completion=report)
    if args.no_artifacts:
        return return_code
    if args.split != "test":
        state = "started" if args.background else "finished"
        if args.split not in ("validation", "calibration"):
            print(
                f"{args.split} grading {state}. This split remains local "
                "because it is not a publishable calibration stage."
            )
            return return_code
        if args.limit is not None:
            print(
                f"Limited {args.split} grading {state}. Partial runs remain "
                "local and are not published."
            )
            return return_code
        if args.background:
            print(
                f"Background {args.split} grading started. Complete split "
                "results will be validated and copied to artifacts after "
                "the job finishes successfully."
            )
            return return_code
        print(
            f"{args.split} grading finished with status={report['status']}; "
            "publishing a split-specific artifact run."
        )
        return publish(
            argparse.Namespace(
                prepared_dir=args.prepared_dir,
                questions=args.questions,
                artifacts_repo=args.artifacts_repo,
                stage=args.split,
                run_id=local_run_id,
                include_raw_ocr=args.include_raw_ocr,
                include_facts=args.include_facts,
                push=args.push_artifacts,
                a3wa_config=args.a3wa_config,
                a3wa_config_questions=(
                    args.questions if args.a3wa_config else []
                ),
                results_dir=str(results_dir),
            )
        )
    if args.limit is not None:
        print(
            "Limited test grading finished. Partial runs remain local and are "
            "not evaluated or published automatically."
        )
        return return_code
    if args.background:
        print(
            "Background test grading started. Complete results will be "
            "validated, evaluated, exported, and copied to artifacts after "
            "the job finishes successfully."
        )
        return return_code
    print(
        f"Test grading finished with status={report['status']}; evaluating all "
        "available score records and copying the run to artifacts."
    )
    return evaluate(
        argparse.Namespace(
            prepared_dir=args.prepared_dir,
            questions=args.questions,
            artifacts_repo=args.artifacts_repo,
            export=True,
            detail=False,
            dry_run=False,
            no_artifacts=False,
            push_artifacts=args.push_artifacts,
            include_facts=args.include_facts,
            include_raw_ocr=args.include_raw_ocr,
            a3wa_config=args.a3wa_config,
            a3wa_config_questions=a3wa_config_questions,
            run_id=local_run_id,
            results_dir=str(results_dir),
            require_complete=False,
        )
    )


def run_experiment(args: argparse.Namespace) -> int:
    if args.background and not os.getenv("REFGRADER_RUN_BACKGROUND_CHILD"):
        if args.dry_run:
            print(display_command(build_run_command(args, background=False), {}))
            return 0
        return start_run_in_background(args)

    if args.dataset_root:
        prepare_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_csbench.py"),
            "--dataset-root",
            args.dataset_root,
            "--output-dir",
            args.prepared_dir,
            "--link-mode",
            args.link_mode,
        ]
        if args.exclude_questions:
            prepare_command.extend(["--exclude-questions", *args.exclude_questions])
        if args.force:
            prepare_command.append("--force")
        print("Preparing CSBench compatible view.")
        prepare_code = execute(prepare_command, dry_run=args.dry_run)
        if prepare_code != 0:
            return prepare_code
        embed_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "embed_csbench_snapshot.py"),
            "--prepared-dir",
            args.prepared_dir,
            "--source-root",
            args.dataset_root,
        ]
        print("Finalizing the embedded, portable CSBench snapshot.")
        embed_code = execute(embed_command, dry_run=args.dry_run)
        if embed_code != 0:
            return embed_code
        audit_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "audit_csbench_snapshot.py"),
            "--prepared-dir",
            args.prepared_dir,
            "--source-root",
            args.dataset_root,
        ]
        print("Auditing the imported source and embedded snapshot.")
        audit_code = execute(audit_command, dry_run=args.dry_run)
        if audit_code != 0:
            return audit_code

    print("Step 1/2: optimizing rubrics.")
    optimize_code = optimize(
        argparse.Namespace(
            prepared_dir=args.prepared_dir,
            questions=args.questions,
            sample_size=args.sample_size,
            device=args.device,
            background=False,
            force=args.force,
            dry_run=args.dry_run,
            artifacts_repo=args.artifacts_repo,
            no_artifacts=args.no_artifacts,
            push_artifacts=args.push_artifacts,
            include_facts=args.include_facts,
            text_provider=args.text_provider,
            thinking_mode=args.thinking_mode,
            vlm_provider=args.vlm_provider,
        )
    )
    if optimize_code != 0:
        return optimize_code
    if args.dry_run:
        print("Step 2/2: grading skipped after dry-run optimization preview.")
        return 0

    print("Step 2/2: grading answers.")
    return grade(
        argparse.Namespace(
            prepared_dir=args.prepared_dir,
            questions=args.questions,
            split=args.split,
            limit=args.limit,
            device=args.device,
            background=False,
            force=args.force,
            dry_run=False,
            artifacts_repo=args.artifacts_repo,
            no_artifacts=args.no_artifacts,
            push_artifacts=args.push_artifacts,
            include_raw_ocr=args.include_raw_ocr,
            include_facts=args.include_facts,
            a3wa_config=args.a3wa_config,
            no_active_a3wa=getattr(args, "no_active_a3wa", False),
            run_id=getattr(args, "run_id", None),
            text_provider=args.text_provider,
            thinking_mode=args.thinking_mode,
            vlm_provider=args.vlm_provider,
        )
    )


def calibrate(args: argparse.Namespace) -> int:
    """Calibrate A3WA from a complete validation split and publish it."""
    model_config = model_config_from_args(args)
    contexts = build_contexts(args.prepared_dir, args.questions)
    for ctx in contexts:
        ctx.validate_optimized()
    validate_active_configuration(contexts)
    validation_dir, validation_run_id = select_grading_run(
        contexts,
        "validation",
        run_id=getattr(args, "source_run_id", None),
    )
    validation_state = _read_run_state(validation_dir)
    source_model_config = (
        (validation_state.get("signature") or {}).get("model_config")
        if validation_state
        else None
    )
    if source_model_config != model_config:
        raise ValueError(
            "The selected validation run was produced by a different or "
            "legacy model contract. Re-run validation with the requested "
            "text model and thinking mode before calibration."
        )
    validate_complete_results(
        contexts,
        validation_dir,
        split_name="validation",
    )
    print(f"Calibration source run: {validation_run_id or 'legacy/unversioned'}")

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (
            PROJECT_ROOT
            / "results_runs"
            / f"csbench_{batch_slug(contexts)}_a3wa_calibration.json"
        ).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [
        validation_dir / f"{ctx.question_id}_grading_checkpoint.json"
        for ctx in contexts
    ]
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "calibrate_a3wa.py"),
        "--files",
        *(str(path) for path in files),
        "--teacher-db",
        str(contexts[0].teacher_db),
        "--database-path",
        str(contexts[0].database),
        "--output",
        str(output),
        "--bnd-max",
        str(args.bnd_max),
        "--neg-max",
        str(getattr(args, "neg_max", 0.35)),
        "--top-k",
        str(args.top_k),
        "--min-cell-count",
        str(args.min_cell_count),
        "--direction-guard-min-count",
        str(getattr(args, "direction_guard_min_count", 3)),
        "--shrinkage-k",
        str(args.shrinkage_k),
        "--max-correction-ratio",
        str(args.max_correction_ratio),
        "--max-correction-points",
        str(args.max_correction_points),
        "--conformal-coverage",
        str(getattr(args, "conformal_coverage", 0.90)),
        "--conformal-scale-floor",
        str(getattr(args, "conformal_scale_floor", 0.05)),
        "--safe-error-ratio",
        str(getattr(args, "safe_error_ratio", 0.10)),
        "--safe-error-points",
        str(getattr(args, "safe_error_points", 0.50)),
        "--bnd-review-cost",
        str(getattr(args, "bnd_review_cost", 0.02)),
        "--neg-human-cost",
        str(getattr(args, "neg_human_cost", 0.10)),
        "--unsafe-pos-cost",
        str(getattr(args, "unsafe_pos_cost", 1.00)),
        "--max-unsafe-pos-rate",
        str(getattr(args, "max_unsafe_pos_rate", 0.10)),
        "--text-provider",
        model_config["text_provider"],
        "--text-model",
        model_config["text_model"],
        "--thinking-mode",
        model_config["text_thinking"],
        "--vlm-provider",
        model_config["vlm_provider"],
        "--vlm-model",
        model_config["vlm_model"],
    ]
    if getattr(args, "score_calibration", False):
        command.append("--score-calibration")
    if args.no_score_calibration:
        command.append("--no-score-calibration")

    return_code = execute(command, dry_run=args.dry_run)
    if return_code != 0 or args.dry_run:
        return return_code
    if not output.is_file():
        raise FileNotFoundError(f"A3WA calibration config not generated: {output}")

    bundle = refresh_active_configuration(
        contexts,
        a3wa_config=output,
        source_validation_run_id=validation_run_id,
    )
    active_config = active_a3wa_config_path(contexts[0])
    print(f"A3WA calibration config: {output}")
    print(f"Tracked active A3WA config: {active_config}")
    print(f"Active rubric set updated: {bundle}")
    if args.no_artifacts:
        return 0
    print(
        "Publishing complete validation checkpoints and the derived A3WA "
        "configuration."
    )
    return publish(
        argparse.Namespace(
            prepared_dir=args.prepared_dir,
            questions=args.questions,
            artifacts_repo=args.artifacts_repo,
            stage="validation",
            run_id=(
                args.run_id
                or datetime.now().strftime("%Y%m%d_%H%M%S_calibrated")
            ),
            include_raw_ocr=args.include_raw_ocr,
            include_facts=args.include_facts,
            push=args.push_artifacts,
            a3wa_config=str(active_config),
            a3wa_config_questions=args.questions,
        )
    )


def activate(args: argparse.Namespace) -> int:
    """Register existing verified rubrics/config as the tracked active set."""
    contexts = build_contexts(args.prepared_dir, args.questions)
    config = (
        Path(args.a3wa_config).expanduser().resolve()
        if getattr(args, "a3wa_config", None)
        else None
    )
    bundle = refresh_active_configuration(
        contexts,
        a3wa_config=config,
        source_validation_run_id=getattr(args, "source_validation_run_id", None),
    )
    print(f"Active rubric set: {bundle}")
    if config:
        print(f"Active A3WA config: {active_a3wa_config_path(contexts[0])}")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    contexts = build_contexts(args.prepared_dir, args.questions)
    explicit_results_dir = getattr(args, "results_dir", None)
    if explicit_results_dir:
        results_dir = Path(explicit_results_dir).expanduser().resolve()
        local_run_id = getattr(args, "run_id", None)
    else:
        results_dir, local_run_id = select_grading_run(
            contexts,
            "test",
            run_id=getattr(args, "run_id", None),
            a3wa_config=getattr(args, "a3wa_config", None),
        )
    missing = [
        str(results_dir / f"{ctx.question_id}_grading_checkpoint.json")
        for ctx in contexts
        if not (
            results_dir / f"{ctx.question_id}_grading_checkpoint.json"
        ).is_file()
    ]
    report = None
    if not args.dry_run:
        report = inspect_results(contexts, results_dir, split_name="test")
        validate_result_structure(report)
        if getattr(args, "require_complete", False):
            validate_complete_results(
                contexts,
                results_dir,
                split_name="test",
            )
        write_completion_report(results_dir, report)
        update_run_state(
            results_dir, status=report["status"], completion=report
        )
        if missing:
            if getattr(args, "require_complete", False):
                raise FileNotFoundError(
                    "Grading checkpoints not found: " + ", ".join(missing)
                )
            print(
                "Evaluation skipped because one or more questions have zero "
                "successful checkpoints. Publishing failed/partial records "
                "with their completion report instead."
            )
            if args.no_artifacts:
                return 0
            return publish(
                argparse.Namespace(
                    prepared_dir=args.prepared_dir,
                    questions=args.questions,
                    artifacts_repo=args.artifacts_repo,
                    stage="full",
                    run_id=local_run_id,
                    include_facts=args.include_facts,
                    include_raw_ocr=args.include_raw_ocr,
                    a3wa_config=getattr(args, "a3wa_config", None),
                    a3wa_config_questions=getattr(
                        args, "a3wa_config_questions", None
                    ),
                    push=args.push_artifacts,
                    results_dir=str(results_dir),
                )
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
        "3wd-core",
        "3wd",
    ]
    if args.export:
        evaluation_dir = results_dir / "evaluation"
        summary_output = evaluation_dir / "summary.json"
        command.extend(
            [
                "--compare-output",
                str(evaluation_dir / "compare.csv"),
                "--summary-output",
                str(summary_output),
            ]
        )
    if args.detail:
        command.append("--detail")
    return_code = execute(command, dry_run=args.dry_run)
    if return_code == 0 and not args.dry_run and report is not None:
        summary_path = results_dir / "evaluation" / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
            if isinstance(summary, dict):
                summary["completion"] = report
                _write_json_atomic(summary_path, summary)
    if return_code != 0 or args.dry_run or args.no_artifacts:
        return return_code
    print(
        "Evaluation finished; publishing experiment artifacts automatically "
        f"with status={report['status'] if report else 'unknown'}."
    )
    return publish(
        argparse.Namespace(
            prepared_dir=args.prepared_dir,
            questions=args.questions,
            artifacts_repo=args.artifacts_repo,
            stage="full",
            run_id=local_run_id,
            include_facts=args.include_facts,
            include_raw_ocr=args.include_raw_ocr,
            a3wa_config=getattr(args, "a3wa_config", None),
            a3wa_config_questions=getattr(
                args, "a3wa_config_questions", None
            ),
            push=args.push_artifacts,
            results_dir=str(results_dir),
        )
    )


def manage_background(args: argparse.Namespace) -> int:
    if os.name == "nt":
        raise RuntimeError(f"{args.action} is only supported on Linux.")
    command = [str(PROJECT_ROOT / "run_experiment.sh"), args.action]
    return execute(command, dry_run=getattr(args, "dry_run", False))


def monitor(args: argparse.Namespace) -> int:
    contexts = build_contexts(args.prepared_dir, args.questions)
    grade_dir, run_id = select_grading_run(
        contexts, args.split, run_id=getattr(args, "run_id", None)
    )
    progress = grade_dir / "progress.json"
    print(f"Monitoring run: {run_id or 'legacy/unversioned'}")
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
    grade_dir, run_id = select_grading_run(
        contexts, args.split, run_id=getattr(args, "run_id", None)
    )
    metadata = {}
    with contexts[0].answer_metadata.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                metadata[str(record.get("answer_id"))] = record

    print(f"Combined optimize run: {optimize_dir}")
    print(f"Selected run ID: {run_id or 'legacy/unversioned'}")
    print(f"Combined grading run: {grade_dir}")
    for ctx in contexts:
        split = json.loads(ctx.split_file.read_text(encoding="utf-8"))
        split_ids = [str(value) for value in split.get(args.split, [])]
        visual_flags = [
            answer_id
            for answer_id in split_ids
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
            f"{args.split} answers: {len(split_ids)}; visual flags: "
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
        f"{grade_dir / 'evaluation' / 'compare.csv'}"
    )
    print(f"Evaluation summary: {grade_dir / 'evaluation' / 'summary.json'}")
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
    if args.push and status:
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
    requested_stage = args.stage
    if requested_stage == "auto":
        test_dir, _ = select_grading_run(
            contexts,
            "test",
            run_id=getattr(args, "run_id", None),
        )
        has_test = all(
            (test_dir / f"{ctx.question_id}_grading_checkpoint.json").is_file()
            for ctx in contexts
        )
        requested_stage = "full" if has_test else "rubric"

    for ctx in contexts:
        normalize_optimization_manifest(ctx)
        ctx.validate_optimized()
    if requested_stage == "rubric":
        refresh_active_configuration(contexts)

    result_stage = RESULT_ARTIFACT_STAGES.get(requested_stage)
    answer_split = result_stage[0] if result_stage else None
    explicit_results_dir = getattr(args, "results_dir", None)
    local_run_id = None
    if explicit_results_dir:
        grade_dir = Path(explicit_results_dir).expanduser().resolve()
        local_run_id = getattr(args, "run_id", None)
    elif answer_split:
        requested_local_run = getattr(args, "run_id", None)
        if requested_local_run:
            candidate = (
                grading_results_dir(contexts, answer_split)
                / "runs"
                / requested_local_run
            )
            if not candidate.is_dir():
                # Historically --run-id named only the artifact destination.
                # Preserve that behavior when no matching local run exists.
                requested_local_run = None
        grade_dir, local_run_id = select_grading_run(
            contexts,
            answer_split,
            run_id=requested_local_run,
        )
    else:
        grade_dir = grading_results_dir(contexts, "test")
    completion_report = None
    if answer_split:
        completion_report = inspect_results(
            contexts, grade_dir, split_name=answer_split
        )
        validate_result_structure(completion_report)
        write_completion_report(grade_dir, completion_report)
    run_state = _read_run_state(grade_dir) if grade_dir.is_dir() else None
    published_model_config = (
        (run_state.get("signature") or {}).get("model_config")
        if run_state
        else None
    )
    if published_model_config is None and requested_stage == "rubric":
        published_model_config = runtime_model_config()
    run_id = (
        args.run_id
        or local_run_id
        or os.getenv("REFGRADER_ARTIFACT_RUN_ID")
        or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    code_commit = git_output(PROJECT_ROOT, "rev-parse", "HEAD")
    a3wa_config = None
    a3wa_config_arg = getattr(args, "a3wa_config", None)
    a3wa_config_questions = set(
        getattr(args, "a3wa_config_questions", None) or []
    )
    if a3wa_config_arg:
        a3wa_config = Path(a3wa_config_arg).expanduser().resolve()
        if not a3wa_config.is_file():
            raise FileNotFoundError(
                f"A3WA calibration config not found: {a3wa_config}"
            )

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
        if (
            requested_stage in RESULT_ARTIFACT_STAGES
            and not has_full_results
            and not completion_report
        ):
            raise FileNotFoundError(
                f"{answer_split} grading checkpoint not found: {checkpoint}"
            )
        include_results = requested_stage in RESULT_ARTIFACT_STAGES
        published_stage = requested_stage if include_results else "rubric"
        stage_dir = (
            RESULT_ARTIFACT_STAGES[published_stage][1]
            if include_results
            else "rubric_optimizations"
        )
        final_destination = (
            artifacts_repo
            / "csbench"
            / ctx.question_id
            / stage_dir
            / run_id
        )
        previous_created_at = None
        previous_manifest = final_destination / "run_manifest.json"
        if previous_manifest.is_file():
            previous_payload = json.loads(
                previous_manifest.read_text(encoding="utf-8-sig")
            )
            previous_created_at = previous_payload.get("created_at")
        destination = prepare_artifact_destination(
            final_destination,
            question_id=ctx.question_id,
            run_id=run_id,
            stage_dir=stage_dir,
        )

        (destination / "rubrics").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            ctx.initial_rubric,
            destination / "rubrics" / "initial_rubric.json",
        )
        shutil.copy2(
            ctx.optimized_rubric,
            destination / "rubrics" / "optimized_rubric.json",
        )
        copy_portable_json(
            ctx.optimization_manifest,
            destination / "rubrics" / "optimization_manifest.json",
            replacements,
        )
        config_applies = bool(a3wa_config) and (
            not a3wa_config_questions
            or ctx.question_id in a3wa_config_questions
        )
        if config_applies:
            copy_portable_json(
                a3wa_config,
                destination / "calibration" / "a3wa_config.json",
                replacements,
            )
        variance_checkpoint, optimization_progress = optimization_evidence_paths(
            ctx, optimize_dir
        )
        if variance_checkpoint:
            copy_if_exists(
                variance_checkpoint,
                destination
                / "rubric_optimization"
                / "variance_checkpoint.json",
                replacements,
            )
        if optimization_progress:
            copy_if_exists(
                optimization_progress,
                destination / "rubric_optimization" / "progress.json",
                replacements,
            )
        variance_facts_count = 0
        if args.include_facts:
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
        if include_results:
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
            copy_if_exists(
                grade_dir / "completion_report.json",
                destination / "grading" / "completion_report.json",
                replacements,
            )
            if args.include_facts:
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
            if answer_split == "test":
                compare_csv = grade_dir / "evaluation" / "compare.csv"
                copy_if_exists(
                    compare_csv,
                    destination / "evaluation" / "compare.csv",
                    replacements,
                )
                summary_json = grade_dir / "evaluation" / "summary.json"
                copy_if_exists(
                    summary_json,
                    destination / "evaluation" / "summary.json",
                    replacements,
                )

        copy_portable_json(
            ctx.split_file,
            destination / "dataset" / "question_split.json",
            replacements,
        )

        if latest_log:
            copy_if_exists(
                latest_log,
                destination / "logs" / "experiment.log",
                replacements,
            )

        artifact_hashes = {
            path.relative_to(destination).as_posix(): sha256_file(path)
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": 2,
            "run_id": run_id,
            "question_id": ctx.question_id,
            "published_stage": published_stage,
            "artifact_stage_dir": stage_dir,
            "artifact_path": final_destination.relative_to(
                artifacts_repo
            ).as_posix(),
            "question_batch": [item.question_id for item in contexts],
            "code_commit": code_commit,
            "dataset_commit": dataset_commit,
            "a3wa_calibration": (
                {
                    "path": "calibration/a3wa_config.json",
                    "sha256": sha256_file(a3wa_config),
                    "source_name": a3wa_config.name,
                    "status": (
                        "derived_from_validation"
                        if answer_split == "validation"
                        else "applied_to_new_grading"
                    ),
                }
                if config_applies
                else (
                    {
                        "path": None,
                        "sha256": None,
                        "source_name": None,
                        "status": "preexisting_completed_checkpoint",
                    }
                    if a3wa_config
                    else None
                )
            ),
            "extraction_backend": "csbench_hybrid",
            "model_config": published_model_config,
            "answer_split": answer_split if include_results else None,
            "ocr_device": os.getenv(
                "REFGRADER_OCR_DEVICE", DEFAULT_OCR_DEVICE
            ),
            "source_server": socket.gethostname(),
            "created_at": previous_created_at
            or datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "completion": (
                completion_report["questions"].get(ctx.question_id)
                if completion_report
                else None
            ),
            "run_status": (
                completion_report.get("status")
                if completion_report
                else "complete"
            ),
            "portable_path_variables": {
                "${REFGRADER_ROOT}": "RefGrader project root",
                "${PREPARED_CSBENCH_ROOT}": "prepared data/csbench root",
                "${CSBENCH_ROOT}": "source CSBench repository root",
            },
            "result_counts": result_counts,
            "variance_fact_files": variance_facts_count,
            "fact_files": facts_count,
            "raw_ocr_files": raw_ocr_count,
            "file_sha256": artifact_hashes,
        }
        (destination / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        commit_artifact_destination(destination, final_destination)
        published_paths.append(final_destination)

    index_path = artifacts_repo / "csbench" / "index.json"
    index = []
    if index_path.is_file():
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            index = loaded
    for destination, ctx in zip(published_paths, contexts):
        entry_stage_dir = destination.parent.name
        entry_stage = {
            "grading_runs": "full",
            "validation_runs": "validation",
            "calibration_runs": "calibration",
            "rubric_optimizations": "rubric",
        }.get(entry_stage_dir, published_stage)
        entry = {
            "question_id": ctx.question_id,
            "run_id": run_id,
            "stage": entry_stage,
            "stage_dir": entry_stage_dir,
            "path": destination.relative_to(artifacts_repo).as_posix(),
            "published_at": datetime.now().isoformat(timespec="seconds"),
            "run_status": (
                completion_report.get("status")
                if completion_report
                else "complete"
            ),
        }
        identity = (ctx.question_id, run_id, entry_stage_dir)
        replaced = False
        for position, existing in enumerate(index):
            if (
                existing.get("question_id"),
                existing.get("run_id"),
                existing.get("stage_dir"),
            ) == identity:
                index[position] = entry
                replaced = True
                break
        if not replaced:
            index.append(entry)
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
            "manually. A later resume of the same run will update this same "
            "run_id directory."
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

    run_parser = subparsers.add_parser(
        "run",
        help=(
            "Prepare optionally, optimize rubrics, then grade one or more "
            "questions with one command."
        ),
    )
    run_parser.add_argument(
        "questions", nargs="+", type=normalize_question_id
    )
    run_parser.add_argument(
        "--dataset-root",
        help=(
            "Optional source CSBench repository. When provided, prepare_csbench "
            "runs before optimization."
        ),
    )
    run_parser.add_argument(
        "--link-mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="Link mode passed to prepare_csbench when --dataset-root is used.",
    )
    run_parser.add_argument(
        "--exclude-questions",
        nargs="*",
        default=["OS_1", "OS_2"],
        help="Questions excluded when --dataset-root triggers prepare_csbench.",
    )
    run_parser.add_argument("--sample-size", type=int, default=5)
    run_parser.add_argument(
        "--split",
        choices=["all", "calibration", "validation", "test"],
        default="test",
    )
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--device", default=DEFAULT_OCR_DEVICE)
    add_model_arguments(run_parser)
    run_parser.add_argument(
        "--a3wa-config",
        help="A3WA calibration config used during the grading stage.",
    )
    run_parser.add_argument(
        "--no-active-a3wa",
        action="store_true",
        help="Do not automatically use the tracked active A3WA config for test.",
    )
    run_parser.add_argument(
        "--background",
        action="store_true",
        help=(
            "Run the whole prepare/optimize/grade workflow in background."
        ),
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Regenerate prepared data, optimized rubrics, and grading "
            "checkpoints where applicable."
        ),
    )
    run_parser.add_argument(
        "--run-id",
        help="Explicit grading run ID; normally generated automatically.",
    )
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument(
        "--artifacts-repo",
        default=str((PROJECT_ROOT.parent / "refgrader-artifacts").resolve()),
    )
    run_parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Do not copy completed artifacts to refgrader-artifacts.",
    )
    run_parser.add_argument(
        "--push-artifacts",
        action="store_true",
        help="Commit and push artifacts after automatic publishing.",
    )
    run_parser.add_argument(
        "--include-raw-ocr",
        action="store_true",
        help="Include raw PaddleOCR JSON files in automatic publishing.",
    )
    run_parser.add_argument(
        "--include-facts",
        action="store_true",
        help="Include per-answer Stage1 fact cache JSON files in publishing.",
    )
    run_parser.set_defaults(handler=run_experiment)

    optimize_parser = subparsers.add_parser(
        "optimize", help="Optimize one or more question rubrics."
    )
    optimize_parser.add_argument(
        "questions", nargs="+", type=normalize_question_id
    )
    optimize_parser.add_argument("--sample-size", type=int, default=5)
    optimize_parser.add_argument("--device", default=DEFAULT_OCR_DEVICE)
    add_model_arguments(optimize_parser)
    optimize_parser.add_argument("--background", action="store_true")
    optimize_mode = optimize_parser.add_mutually_exclusive_group()
    optimize_mode.add_argument("--force", action="store_true")
    optimize_mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip rubrics already valid under the current semantic contract "
            "and resume pending questions from the shared variance checkpoint."
        ),
    )
    optimize_parser.add_argument("--dry-run", action="store_true")
    optimize_parser.add_argument(
        "--artifacts-repo",
        default=str((PROJECT_ROOT.parent / "refgrader-artifacts").resolve()),
    )
    optimize_parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Do not copy completed rubric artifacts to refgrader-artifacts.",
    )
    optimize_parser.add_argument(
        "--push-artifacts",
        action="store_true",
        help="Commit and push artifacts after automatic publishing.",
    )
    optimize_parser.add_argument(
        "--include-facts",
        action="store_true",
        help="Include per-answer Stage1 fact cache JSON files in publishing.",
    )
    optimize_parser.set_defaults(handler=optimize)

    activate_parser = subparsers.add_parser(
        "activate",
        help="Register verified rubrics and an optional A3WA config as active.",
    )
    activate_parser.add_argument(
        "questions", nargs="+", type=normalize_question_id
    )
    activate_parser.add_argument(
        "--a3wa-config",
        help="Verified A3WA config to copy into the tracked active location.",
    )
    activate_parser.add_argument(
        "--source-validation-run-id",
        help="Validation run ID recorded as the calibration source.",
    )
    activate_parser.set_defaults(handler=activate)

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
    add_model_arguments(grade_parser)
    grade_parser.add_argument(
        "--a3wa-config",
        help=(
            "Explicit A3WA config used for test. When omitted, a valid tracked "
            "active config is selected automatically."
        ),
    )
    grade_parser.add_argument(
        "--no-active-a3wa",
        action="store_true",
        help="Do not automatically use the tracked active A3WA config for test.",
    )
    grade_parser.add_argument("--background", action="store_true")
    grade_parser.add_argument("--force", action="store_true")
    grade_parser.add_argument(
        "--run-id",
        help=(
            "Resume a specific versioned run. With --force the ID must not "
            "already exist."
        ),
    )
    grade_parser.add_argument("--dry-run", action="store_true")
    grade_parser.add_argument(
        "--artifacts-repo",
        default=str((PROJECT_ROOT.parent / "refgrader-artifacts").resolve()),
    )
    grade_parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Do not copy completed grading artifacts to refgrader-artifacts.",
    )
    grade_parser.add_argument(
        "--push-artifacts",
        action="store_true",
        help="Commit and push artifacts after automatic publishing.",
    )
    grade_parser.add_argument(
        "--include-raw-ocr",
        action="store_true",
        help="Include raw PaddleOCR JSON files in automatic publishing.",
    )
    grade_parser.add_argument(
        "--include-facts",
        action="store_true",
        help="Include per-answer Stage1 fact cache JSON files in publishing.",
    )
    grade_parser.set_defaults(handler=grade)

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help=(
            "Calibrate A3WA from complete validation checkpoints and publish "
            "a reproducible validation artifact run."
        ),
    )
    calibrate_parser.add_argument(
        "questions", nargs="+", type=normalize_question_id
    )
    calibrate_parser.add_argument("--output")
    calibrate_parser.add_argument("--bnd-max", type=float, default=0.60)
    calibrate_parser.add_argument("--neg-max", type=float, default=0.35)
    calibrate_parser.add_argument("--top-k", type=int, default=8)
    calibrate_parser.add_argument("--min-cell-count", type=int, default=20)
    calibrate_parser.add_argument(
        "--direction-guard-min-count", type=int, default=3
    )
    calibrate_parser.add_argument("--shrinkage-k", type=float, default=8.0)
    calibrate_parser.add_argument(
        "--max-correction-ratio", type=float, default=0.12
    )
    calibrate_parser.add_argument(
        "--max-correction-points", type=float, default=2.0
    )
    calibrate_parser.add_argument(
        "--score-calibration", action="store_true"
    )
    calibrate_parser.add_argument(
        "--no-score-calibration", action="store_true"
    )
    calibrate_parser.add_argument("--conformal-coverage", type=float, default=0.90)
    calibrate_parser.add_argument("--conformal-scale-floor", type=float, default=0.05)
    calibrate_parser.add_argument("--safe-error-ratio", type=float, default=0.10)
    calibrate_parser.add_argument("--safe-error-points", type=float, default=0.50)
    calibrate_parser.add_argument("--bnd-review-cost", type=float, default=0.02)
    calibrate_parser.add_argument("--neg-human-cost", type=float, default=0.10)
    calibrate_parser.add_argument("--unsafe-pos-cost", type=float, default=1.00)
    calibrate_parser.add_argument("--max-unsafe-pos-rate", type=float, default=0.10)
    add_model_arguments(calibrate_parser)
    calibrate_parser.add_argument("--dry-run", action="store_true")
    calibrate_parser.add_argument(
        "--artifacts-repo",
        default=str((PROJECT_ROOT.parent / "refgrader-artifacts").resolve()),
    )
    calibrate_parser.add_argument("--run-id")
    calibrate_parser.add_argument(
        "--source-run-id",
        help="Specific validation run to calibrate; defaults to active.",
    )
    calibrate_parser.add_argument("--no-artifacts", action="store_true")
    calibrate_parser.add_argument("--push-artifacts", action="store_true")
    calibrate_parser.add_argument("--include-raw-ocr", action="store_true")
    calibrate_parser.add_argument("--include-facts", action="store_true")
    calibrate_parser.set_defaults(handler=calibrate)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate one or more question checkpoints."
    )
    evaluate_parser.add_argument(
        "questions", nargs="+", type=normalize_question_id
    )
    evaluate_parser.add_argument("--export", action="store_true")
    evaluate_parser.add_argument("--detail", action="store_true")
    evaluate_parser.add_argument(
        "--run-id", help="Specific test run to evaluate; defaults to active."
    )
    evaluate_parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Reject partial coverage instead of evaluating available IDs.",
    )
    evaluate_parser.add_argument("--dry-run", action="store_true")
    evaluate_parser.add_argument(
        "--a3wa-config",
        help=(
            "A3WA calibration config to archive with the evaluated run. "
            "Automatic post-grade evaluation sets this from grade."
        ),
    )
    evaluate_parser.add_argument(
        "--a3wa-config-question",
        dest="a3wa_config_questions",
        action="append",
        type=normalize_question_id,
        help=argparse.SUPPRESS,
    )
    evaluate_parser.add_argument(
        "--artifacts-repo",
        default=str((PROJECT_ROOT.parent / "refgrader-artifacts").resolve()),
    )
    evaluate_parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Do not copy completed experiment artifacts to refgrader-artifacts.",
    )
    evaluate_parser.add_argument(
        "--push-artifacts",
        action="store_true",
        help="Commit and push artifacts after automatic publishing.",
    )
    evaluate_parser.add_argument(
        "--include-raw-ocr",
        action="store_true",
        help="Include raw PaddleOCR JSON files in automatic publishing.",
    )
    evaluate_parser.add_argument(
        "--include-facts",
        action="store_true",
        help="Include per-answer Stage1 fact cache JSON files in publishing.",
    )
    evaluate_parser.set_defaults(handler=evaluate)

    monitor_parser = subparsers.add_parser(
        "monitor", help="Monitor one or more questions in one grading run."
    )
    monitor_parser.add_argument(
        "questions", nargs="+", type=normalize_question_id
    )
    monitor_parser.add_argument(
        "--split",
        choices=["all", "calibration", "validation", "test"],
        default="test",
    )
    monitor_parser.add_argument("--dry-run", action="store_true")
    monitor_parser.add_argument("--run-id")
    monitor_parser.set_defaults(handler=monitor)

    outputs_parser = subparsers.add_parser(
        "outputs", help="Show expected output files for one or more questions."
    )
    outputs_parser.add_argument(
        "questions", nargs="+", type=normalize_question_id
    )
    outputs_parser.add_argument(
        "--split",
        choices=["all", "calibration", "validation", "test"],
        default="test",
    )
    outputs_parser.add_argument("--run-id")
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
        "--stage",
        choices=["auto", "rubric", "validation", "calibration", "full"],
        default="auto",
    )
    publish_parser.add_argument("--run-id")
    publish_parser.add_argument("--results-dir", help=argparse.SUPPRESS)
    publish_parser.add_argument(
        "--a3wa-config",
        help="A3WA calibration config to copy into the artifact run.",
    )
    publish_parser.add_argument(
        "--a3wa-config-question",
        dest="a3wa_config_questions",
        action="append",
        type=normalize_question_id,
        help=argparse.SUPPRESS,
    )
    publish_parser.add_argument("--include-facts", action="store_true")
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
