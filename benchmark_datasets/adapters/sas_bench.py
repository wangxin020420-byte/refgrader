from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_datasets.contract import (
    BENCHMARK_SCHEMA_VERSION,
    audit_prepared_benchmark,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from rubric_semantics import (
    RUBRIC_SEMANTIC_CONTRACT_VERSION,
    prepare_rubric_semantic_contract,
    validate_refined_rubric,
)


SPLIT_NAMES = ("train", "calibration", "validation", "test")


def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _number(value: Any, *, field: str, record_id: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} for {record_id}: {value!r}") from exc
    if not result >= 0.0:
        raise ValueError(f"Negative {field} for {record_id}: {result}")
    return result


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.glob("*.jsonl") if item.is_file()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _task_slug(task: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", task).strip("_").upper()
    return slug or "TASK"


def _context_key(task: str, record: dict[str, Any]) -> tuple[str, ...]:
    return (
        task,
        _normalized_text(record.get("question")),
        _normalized_text(record.get("reference")),
        _normalized_text(record.get("analysis")),
        format(float(record.get("total")), ".12g"),
    )


def _question_id(context: tuple[str, ...]) -> str:
    digest = hashlib.sha256(
        json.dumps(context, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    return f"SASB_{_task_slug(context[0])}_{digest}"


def _student_text(steps: Any, *, record_id: str) -> str:
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Missing student response steps for {record_id}")
    responses = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Invalid step {index} for {record_id}")
        response = _normalized_text(step.get("response"))
        if response:
            responses.append(response)
    if not responses:
        raise ValueError(f"Empty student response for {record_id}")
    return "\n".join(responses)


def _step_label_sum(steps: Any) -> float | None:
    if not isinstance(steps, list):
        return None
    values = []
    for step in steps:
        if not isinstance(step, dict):
            return None
        try:
            values.append(float(step.get("label")))
        except (TypeError, ValueError):
            return None
    return sum(values)


def _rubric(reference: str, analysis: str, total: float) -> list[dict[str, Any]]:
    guidance = analysis or (
        "Assess the response for correctness and completeness against the "
        "official reference answer, awarding proportional partial credit."
    )
    rubric = prepare_rubric_semantic_contract(
        [
            {
                "id": "overall_response",
                "item": guidance,
                "points": total,
                "answer_type": "free_text",
                "role": "final",
                "score_layer": "core",
                "canonicalization": {"type": "text"},
                "evidence_source": "text",
                "standard_answer_text": reference,
                "standard_answer_image": None,
                "source_text": guidance,
                "parent_official_item": "official SAS-Bench response score",
                "metadata_source": "sas_bench_question_reference_analysis",
                "metadata_hard_enabled": False,
                "metadata_confidence": 1.0,
                "parent_id": "overall_response",
                "parent_points": total,
                "scoring_policy": "strict_atomic",
                "split_policy": "preserve_atomic",
                "weighting_policy": "preserve_parent",
                "full_credit_policy": "rubric_evidence_required",
                "full_credit_trigger": False,
            }
        ]
    )
    valid, errors = validate_refined_rubric(
        rubric,
        rubric,
        total,
        allow_unchanged_baseline=True,
    )
    if not valid:
        raise ValueError("Generated SAS-Bench rubric is invalid: " + "; ".join(errors))
    return rubric


def _source_files(source: Path) -> list[Path]:
    files = sorted(path for path in source.glob("*.jsonl") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No SAS-Bench task JSONL files found in: {source}")
    return files


def prepare_sas_bench(
    source_path: str | Path,
    spec_path: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve()
    spec_file = Path(spec_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"SAS-Bench source directory not found: {source}")
    if not spec_file.is_file():
        raise FileNotFoundError(f"SAS-Bench spec file not found: {spec_file}")
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(
            f"Prepared directory is not empty: {output}. Use --force to replace it."
        )
    if output.exists() and force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    dataset_id = str(spec.get("dataset_id", "sas_bench_v1")).strip()
    task_files = _source_files(source)
    raw_records: list[dict[str, Any]] = []
    source_contexts: set[tuple[str, ...]] = set()
    seen_answer_ids: set[str] = set()
    excluded_records: list[dict[str, Any]] = []
    records_by_context: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)

    for task_file in task_files:
        task = task_file.stem
        for source_index, record in enumerate(read_jsonl(task_file), start=1):
            raw_id = str(record.get("id", "")).strip()
            if not raw_id:
                raise ValueError(f"Missing id in {task_file}:{source_index}")
            answer_id = raw_id
            if answer_id in seen_answer_ids:
                raise ValueError(f"Duplicate SAS-Bench answer id: {answer_id}")
            seen_answer_ids.add(answer_id)
            total = _number(record.get("total"), field="total", record_id=raw_id)
            score = _number(
                record.get("manual_label"),
                field="manual_label",
                record_id=raw_id,
            )
            context = _context_key(task, record)
            source_contexts.add(context)
            step_sum = _step_label_sum(record.get("steps"))
            raw_records.append(record)
            normalized = {
                "answer_id": answer_id,
                "source_answer_id": raw_id,
                "source_task": task,
                "source_file": task_file.name,
                "source_line": source_index,
                "question": _normalized_text(record.get("question")),
                "reference": _normalized_text(record.get("reference")),
                "analysis": _normalized_text(record.get("analysis")),
                "total": total,
                "score": score,
                "raw_text": _student_text(record.get("steps"), record_id=raw_id),
                "steps": record.get("steps"),
                "step_label_sum": step_sum,
                "step_score_consistent": (
                    step_sum is not None and abs(step_sum - score) <= 1e-9
                ),
            }
            if score > total + 1e-9:
                excluded_records.append(
                    {
                        "answer_id": answer_id,
                        "source_task": task,
                        "source_file": task_file.name,
                        "source_line": source_index,
                        "manual_label": score,
                        "total": total,
                        "step_label_sum": step_sum,
                        "reason": "manual_label_exceeds_total",
                    }
                )
                continue
            records_by_context[context].append(normalized)

    expected = spec.get("expected_counts") or {}
    checks = {
        "source_records": len(raw_records),
        "source_question_contexts": len(source_contexts),
        "formal_test_records": sum(len(items) for items in records_by_context.values()),
        "excluded_records": len(excluded_records),
        "task_files": len(task_files),
    }
    for key, actual in checks.items():
        if key in expected and int(expected[key]) != actual:
            raise ValueError(
                f"SAS-Bench {key} count mismatch: expected={expected[key]}, actual={actual}"
            )

    exam_database = []
    teacher_scores: dict[str, dict[str, float]] = {}
    answer_metadata = []
    normalized_questions = []
    normalized_labels = []
    gold_step_labels = []
    question_ids_seen: set[str] = set()
    label_leakage_fields = {"label", "errors", "manual_label"}

    for context in sorted(records_by_context):
        question_records = records_by_context[context]
        question_id = _question_id(context)
        if question_id in question_ids_seen:
            raise ValueError(f"SAS-Bench question hash collision: {question_id}")
        question_ids_seen.add(question_id)
        task, question_text, reference, analysis, total_text = context
        total = float(total_text)
        rubric = _rubric(reference, analysis, total)
        rubric_group = "SASBENCH"
        rubric_name = f"{question_id}_rubric_standard.json"
        initial_relative = Path("rubrics") / "initial" / rubric_group / rubric_name
        optimized_relative = Path("rubrics") / "optimized" / rubric_group / rubric_name
        manifest_relative = (
            Path("rubrics")
            / "manifests"
            / rubric_group
            / f"{question_id}_optimization.json"
        )
        split_relative = Path("splits") / f"{question_id}.json"
        write_json(output / initial_relative, rubric)
        write_json(output / optimized_relative, rubric)
        answer_ids = [record["answer_id"] for record in question_records]
        split = {"train": [], "calibration": [], "validation": [], "test": answer_ids}
        write_json(
            output / split_relative,
            {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "question_id": question_id,
                "policy": "external_test_only_no_public_label_tuning",
                **split,
            },
        )
        write_json(
            output / manifest_relative,
            {
                "question_id": question_id,
                "rubric_group": rubric_group,
                "method": "label_blind_reference_baseline",
                "rubric_semantic_contract_version": RUBRIC_SEMANTIC_CONTRACT_VERSION,
                "semantic_policy_validated": True,
                "semantic_validation_mode": "prepared_label_blind_baseline",
                "selected_variant": "baseline",
                "initial_rubric": initial_relative.as_posix(),
                "optimized_rubric": optimized_relative.as_posix(),
                "initial_sha256": sha256_file(output / initial_relative),
                "optimized_sha256": sha256_file(output / optimized_relative),
                "calibration_answer_ids": [],
                "path_format": "prepared_relative_v1",
                "note": (
                    "The optimized slot mirrors a label-blind baseline generated only "
                    "from question, reference, analysis, and total. Student gold labels "
                    "and error annotations were not used."
                ),
            },
        )
        grading_guidance = analysis or "Grade against the official reference answer."
        exam_database.append(
            {
                "question_id": question_id,
                "subject": "public_benchmark",
                "exam_source": dataset_id,
                "question_text": question_text,
                "question_image": None,
                "ref_text": reference,
                "ref_image": None,
                "total_score": total,
                "official_rubric": grading_guidance,
                "rubric_group": rubric_group,
                "source_rubric_path": initial_relative.as_posix(),
                "initial_rubric_path": initial_relative.as_posix(),
                "optimized_rubric_path": optimized_relative.as_posix(),
                "rubric_split_path": split_relative.as_posix(),
                "rubric_calibration_ids": [],
                "student_images_dir": (Path("text_samples") / question_id).as_posix(),
                "requires_visual_evidence": False,
                "sample_count": len(question_records),
                "source_task": task,
            }
        )
        normalized_questions.append(
            {
                "question_id": question_id,
                "source_task": task,
                "question_text": question_text,
                "reference_answer": reference,
                "grading_guidance": grading_guidance,
                "max_score": total,
            }
        )
        for record in question_records:
            answer_id = record["answer_id"]
            teacher_scores[answer_id] = {question_id: record["score"]}
            metadata = {
                "answer_id": answer_id,
                "question_id": question_id,
                "subject": "public_benchmark",
                "subject_name": "SAS-Bench",
                "raw_text": record["raw_text"],
                "isimagine": False,
                "visual_placeholder_detected": False,
                "student_image": None,
                "source_image": None,
                "source_file": record["source_file"],
                "source_line": record["source_line"],
                "source_answer_id": record["source_answer_id"],
                "source_task": task,
                "score_source": "official_manual_label",
            }
            if label_leakage_fields & metadata.keys():
                raise AssertionError("Gold-only fields leaked into answer metadata")
            answer_metadata.append(metadata)
            normalized_labels.append(
                {
                    "answer_id": answer_id,
                    "question_id": question_id,
                    "score": record["score"],
                    "score_source": "official_manual_label",
                    "step_label_sum": record["step_label_sum"],
                    "step_score_consistent": record["step_score_consistent"],
                }
            )
            gold_step_labels.append(
                {
                    "answer_id": answer_id,
                    "question_id": question_id,
                    "source_task": task,
                    "steps": [
                        {
                            "step_index": index,
                            "label": step.get("label"),
                            "errors": step.get("errors"),
                        }
                        for index, step in enumerate(record["steps"], start=1)
                    ],
                }
            )

    write_json(output / "exam_database.json", exam_database)
    write_json(output / "teacher_scores.json", teacher_scores)
    write_jsonl(output / "answer_metadata.jsonl", answer_metadata)
    write_jsonl(output / "normalized" / "questions.jsonl", normalized_questions)
    write_jsonl(output / "normalized" / "answers.jsonl", answer_metadata)
    write_jsonl(output / "normalized" / "gold_labels.jsonl", normalized_labels)
    write_jsonl(output / "gold_only" / "step_labels_and_errors.jsonl", gold_step_labels)
    write_jsonl(output / "quality_control" / "excluded_records.jsonl", excluded_records)
    (output / "source").mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec_file, output / "source" / "adapter_spec.json")

    manifest = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "adapter": "sas_bench",
        "task_type": "numeric_answer_scoring",
        "extraction_backend": "text_only",
        "source": {
            "directory": source.name,
            "sha256": _tree_sha256(source),
            "adapter_spec_sha256": sha256_file(spec_file),
        },
        "score_label_policy": {
            "priority": "official_manual_label",
            "excluded_when_manual_label_exceeds_total": True,
            "gold_step_annotations_isolated": True,
        },
        "rubric_policy": {
            "type": "label_blind_question_reference_analysis_baseline",
            "uses_student_step_labels": False,
            "uses_student_error_annotations": False,
            "optimized_slot_mirrors_validated_baseline": True,
        },
        "split": {
            "policy": "external_test_only_no_public_label_tuning",
            "ratios": {"train": 0.0, "calibration": 0.0, "validation": 0.0, "test": 1.0},
        },
        "counts": {
            "questions": len(exam_database),
            "answers": len(answer_metadata),
            "source_records": len(raw_records),
            "source_question_contexts": len(source_contexts),
            "excluded_records": len(excluded_records),
            "task_files": len(task_files),
        },
        "question_ids": [item["question_id"] for item in exam_database],
        "gold_only_files": ["gold_only/step_labels_and_errors.jsonl"],
        "quality_control_files": ["quality_control/excluded_records.jsonl"],
    }
    write_json(output / "manifest.json", manifest)
    audit = audit_prepared_benchmark(output)
    audit.update(
        {
            "source_record_count": len(raw_records),
            "source_question_context_count": len(source_contexts),
            "formal_test_record_count": len(answer_metadata),
            "excluded_record_count": len(excluded_records),
            "task_file_count": len(task_files),
            "label_leakage_count": 0,
        }
    )
    write_json(output / "audit.json", audit)
    return audit
