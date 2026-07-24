"""Restore a reproducible CSBench run from refgrader-artifacts.

The artifact repository is the cross-device transport. This command restores
the files needed to continue a validation -> A3WA calibration -> test workflow
without copying ignored ``results_runs`` directories through the code repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

if __package__:
    from .run_csbench import (
        PROJECT_ROOT,
        build_contexts,
        grading_results_dir,
        inspect_results,
        normalize_question_id,
        normalize_optimization_manifest,
        refresh_active_configuration,
        register_restored_run,
        sha256_file,
        validate_result_structure,
        write_completion_report,
    )
else:
    from run_csbench import (
        PROJECT_ROOT,
        build_contexts,
        grading_results_dir,
        inspect_results,
        normalize_question_id,
        normalize_optimization_manifest,
        refresh_active_configuration,
        register_restored_run,
        sha256_file,
        validate_result_structure,
        write_completion_report,
    )

from sample_quality import default_policy_path


DEFAULT_ARTIFACTS_REPO = PROJECT_ROOT.parent / "refgrader-artifacts"
STAGE_LAYOUT = {
    "rubric": ("rubric_optimizations", None),
    "audit": ("audit_runs", "all"),
    "validation": ("validation_runs", "validation"),
    "calibration": ("calibration_runs", "calibration"),
    "test": ("grading_runs", "test"),
}
GRADING_FILES = {
    "grading_checkpoint.json": "grading_checkpoint",
    "graded_results.json": "graded_results",
    "rejected.json": "rejected",
    "failed.json": "failed",
}


def common_run_id(
    artifacts_repo: Path,
    questions: list[str],
    stage_dir: str,
    requested: str | None,
) -> str:
    run_sets = []
    for question in questions:
        root = artifacts_repo / "csbench" / question / stage_dir
        if not root.is_dir():
            raise FileNotFoundError(f"Artifact stage directory not found: {root}")
        run_sets.append({path.name for path in root.iterdir() if path.is_dir()})
    common = set.intersection(*run_sets) if run_sets else set()
    if requested:
        if requested not in common:
            raise FileNotFoundError(
                f"Run {requested} is not present for every requested question "
                f"under {stage_dir}."
            )
        return requested
    if not common:
        raise FileNotFoundError(
            f"No common {stage_dir} run exists for: {', '.join(questions)}"
        )
    return sorted(common, reverse=True)[0]


def copy_checked(source: Path, destination: Path, *, force: bool) -> None:
    if not source.is_file():
        return
    if destination.is_file():
        if sha256_file(source) == sha256_file(destination):
            return
        if not force:
            raise FileExistsError(
                f"Refusing to overwrite a different file: {destination}. "
                "Use --force after verifying the artifact run."
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def config_digest(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def canonical_json_digest(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def restore_manifest(
    source: Path,
    ctx,
    *,
    force: bool,
    variance_checkpoint: Path | None = None,
) -> None:
    destination = ctx.optimization_manifest
    if destination.is_file() and not force:
        existing = json.loads(destination.read_text(encoding="utf-8-sig"))
        incoming = json.loads(source.read_text(encoding="utf-8-sig"))
        if existing.get("optimized_sha256") != incoming.get("optimized_sha256"):
            raise FileExistsError(
                f"Refusing to overwrite a different manifest: {destination}. "
                "Use --force after verifying the artifact run."
            )
    manifest = json.loads(source.read_text(encoding="utf-8-sig"))
    manifest["initial_rubric"] = str(ctx.initial_rubric)
    manifest["optimized_rubric"] = str(ctx.optimized_rubric)
    manifest["initial_sha256"] = sha256_file(ctx.initial_rubric)
    manifest["optimized_sha256"] = sha256_file(ctx.optimized_rubric)
    if variance_checkpoint and variance_checkpoint.is_file():
        manifest["optimization_results_dir"] = str(
            variance_checkpoint.parent
        )
        manifest["variance_checkpoint"] = str(variance_checkpoint)
        manifest["variance_checkpoint_sha256"] = sha256_file(
            variance_checkpoint
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    normalize_optimization_manifest(ctx)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore rubrics, split results, calibration and evaluation artifacts."
    )
    parser.add_argument("questions", nargs="+", type=normalize_question_id)
    parser.add_argument(
        "--stage", choices=sorted(STAGE_LAYOUT), default="validation"
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--artifacts-repo", type=Path, default=DEFAULT_ARTIFACTS_REPO
    )
    parser.add_argument("--prepared-dir", default="data/csbench")
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    artifacts_repo = args.artifacts_repo.expanduser().resolve()
    if not (artifacts_repo / ".git").is_dir():
        raise ValueError(f"Not an artifacts Git repository: {artifacts_repo}")

    contexts = build_contexts(args.prepared_dir, args.questions)
    questions = [ctx.question_id for ctx in contexts]
    stage_dir, answer_split = STAGE_LAYOUT[args.stage]
    run_id = common_run_id(
        artifacts_repo, questions, stage_dir, args.run_id
    )
    results_dir = None
    if answer_split:
        results_dir = (
            grading_results_dir(contexts, answer_split) / "runs" / run_id
        )
    restored_optimization_dir = (
        PROJECT_ROOT
        / "results_runs"
        / f"restored_rubric_optimization_{run_id}"
    ).resolve()

    configs: list[Path] = []
    artifact_model_configs: list[dict] = []
    artifact_policy_descriptors: list[dict] = []
    artifact_policy_files: list[Path] = []
    for ctx in contexts:
        source = artifacts_repo / "csbench" / ctx.question_id / stage_dir / run_id
        manifest_path = source / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("question_id") != ctx.question_id:
            raise ValueError(f"Question mismatch in {manifest_path}")
        if answer_split and manifest.get("answer_split") != answer_split:
            raise ValueError(
                f"Split mismatch in {manifest_path}: "
                f"expected {answer_split}, got {manifest.get('answer_split')}"
            )
        artifact_model_configs.append(
            manifest.get("model_config")
            or {"status": "legacy_unspecified"}
        )
        policy_descriptor = manifest.get("sample_quality_policy") or {
            "mode": "raw",
            "policy_id": "raw",
            "sha256": None,
        }
        artifact_policy_descriptors.append(policy_descriptor)
        artifact_policy = source / "dataset" / "sample_quality_policy.json"
        if policy_descriptor.get("mode") == "active":
            if not artifact_policy.is_file():
                raise FileNotFoundError(
                    "Active sample-quality policy declared but missing: "
                    f"{artifact_policy}"
                )
            artifact_policy_files.append(artifact_policy)
        for relative, expected_hash in manifest.get("file_sha256", {}).items():
            artifact_file = source / relative
            if not artifact_file.is_file():
                raise FileNotFoundError(
                    f"Artifact declared by manifest is missing: {artifact_file}"
                )
            if sha256_file(artifact_file) != expected_hash:
                raise ValueError(f"Artifact hash mismatch: {artifact_file}")

        artifact_split = source / "dataset" / "question_split.json"
        if artifact_split.is_file() and canonical_json_digest(
            artifact_split
        ) != canonical_json_digest(ctx.split_file):
            raise ValueError(
                f"Question split differs for {ctx.question_id}; pull the "
                "matching refgrader data snapshot before restoring this run."
            )

        artifact_initial = source / "rubrics" / "initial_rubric.json"
        if canonical_json_digest(artifact_initial) != canonical_json_digest(
            ctx.initial_rubric
        ):
            raise ValueError(
                f"Initial rubric differs for {ctx.question_id}; pull the matching "
                "refgrader code/data commit before restoring this run."
            )
        copy_checked(
            source / "rubrics" / "optimized_rubric.json",
            ctx.optimized_rubric,
            force=args.force,
        )
        restored_variance = (
            restored_optimization_dir
            / f"{ctx.question_id}_variance_checkpoint.json"
        )
        copy_checked(
            source / "rubric_optimization" / "variance_checkpoint.json",
            restored_variance,
            force=args.force,
        )
        copy_checked(
            source / "rubric_optimization" / "progress.json",
            restored_optimization_dir / "progress.json",
            force=args.force,
        )
        restore_manifest(
            source / "rubrics" / "optimization_manifest.json",
            ctx,
            force=args.force,
            variance_checkpoint=(
                restored_variance if restored_variance.is_file() else None
            ),
        )

        if results_dir:
            grading_dir = source / "grading"
            for source_name, suffix in GRADING_FILES.items():
                copy_checked(
                    grading_dir / source_name,
                    results_dir / f"{ctx.question_id}_{suffix}.json",
                    force=args.force,
                )
            copy_checked(
                grading_dir / "progress.json",
                results_dir / "progress.json",
                force=args.force,
            )

        config = source / "calibration" / "a3wa_config.json"
        if config.is_file():
            configs.append(config)

    restored_config = None
    if configs:
        if len(configs) != len(contexts):
            raise ValueError(
                "A3WA config is missing from one or more question artifacts."
            )
        hashes = {config_digest(path) for path in configs}
        if len(hashes) != 1:
            raise ValueError("Question artifacts contain different A3WA configs.")
        restored_config = (
            args.config_output.expanduser().resolve()
            if args.config_output
            else (
                PROJECT_ROOT
                / "results_runs"
                / f"restored_a3wa_{run_id}.json"
            ).resolve()
        )
        copy_checked(configs[0], restored_config, force=args.force)

    serialized_policy_descriptors = {
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        for value in artifact_policy_descriptors
    }
    if len(serialized_policy_descriptors) != 1:
        raise ValueError(
            "Question artifacts contain different sample-quality policies."
        )
    restored_policy_descriptor = artifact_policy_descriptors[0]
    active_policy_path = default_policy_path(contexts[0].root)
    if restored_policy_descriptor.get("mode") == "active":
        if len(artifact_policy_files) != len(contexts):
            raise ValueError(
                "Sample-quality policy is missing from one or more artifacts."
            )
        policy_hashes = {
            canonical_json_digest(path) for path in artifact_policy_files
        }
        if len(policy_hashes) != 1:
            raise ValueError(
                "Question artifacts contain different policy payloads."
            )
        copy_checked(
            artifact_policy_files[0],
            active_policy_path,
            force=args.force,
        )
    elif active_policy_path.is_file():
        if not args.force:
            raise FileExistsError(
                "The artifact run used raw labels, but an active local "
                "sample-quality policy exists. Use --force after verifying "
                "that the raw policy should be restored."
            )
        active_policy_path.unlink()

    serialized_model_configs = {
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        for value in artifact_model_configs
    }
    if len(serialized_model_configs) != 1:
        raise ValueError("Question artifacts contain different model contracts.")
    restored_model_config = artifact_model_configs[0]

    if args.stage == "test":
        evaluation_dir = (
            artifacts_repo
            / "csbench"
            / questions[0]
            / stage_dir
            / run_id
            / "evaluation"
        )
        copy_checked(
            evaluation_dir / "compare.csv",
            PROJECT_ROOT / "outputs" / f"restored_{run_id}_compare.csv",
            force=args.force,
        )
        copy_checked(
            evaluation_dir / "summary.json",
            PROJECT_ROOT / "outputs" / f"restored_{run_id}_summary.json",
            force=args.force,
        )

    for ctx in contexts:
        ctx.validate_optimized()
    active_bundle = refresh_active_configuration(
        contexts,
        a3wa_config=restored_config,
        source_validation_run_id=(
            run_id
            if restored_config and args.stage in {"validation", "calibration"}
            else None
        ),
    )
    if results_dir and answer_split:
        report = inspect_results(
            contexts, results_dir, split_name=answer_split
        )
        validate_result_structure(report)
        write_completion_report(results_dir, report)
        register_restored_run(
            contexts,
            answer_split,
            run_id,
            results_dir,
            a3wa_config=(
                str(restored_config)
                if restored_config and answer_split == "test"
                else None
            ),
            completion=report,
            model_config=restored_model_config,
        )
    print(f"Restored {args.stage} run {run_id} for: {', '.join(questions)}")
    if results_dir:
        print(f"Results directory: {results_dir}")
    if restored_config:
        print(f"A3WA config: {restored_config}")
    print(f"Active configuration: {active_bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
