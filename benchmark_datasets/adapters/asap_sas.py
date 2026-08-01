from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_datasets.contract import (
    BENCHMARK_SCHEMA_VERSION,
    audit_prepared_benchmark,
    sha256_file,
    write_json,
    write_jsonl,
)


DEFAULT_COLUMNS = {
    "answer_id": ("Id", "id", "answer_id", "essay_id"),
    "question_id": ("EssaySet", "essay_set", "question_id", "prompt_id"),
    "answer_text": ("EssayText", "essay_text", "answer", "answer_text"),
    "resolved_score": (
        "ResolvedScore",
        "resolved_score",
        "domain1_score",
        "score",
    ),
    "score_1": ("Score1", "score_1", "rater1_score"),
    "score_2": ("Score2", "score_2", "rater2_score"),
}


def _column_name(
    fieldnames: list[str],
    explicit: dict[str, str],
    logical_name: str,
    *,
    required: bool,
) -> str | None:
    configured = explicit.get(logical_name)
    if configured:
        if configured not in fieldnames:
            raise ValueError(
                f"Configured column {configured!r} for {logical_name} "
                "does not exist"
            )
        return configured
    for candidate in DEFAULT_COLUMNS[logical_name]:
        if candidate in fieldnames:
            return candidate
    if required:
        raise ValueError(
            f"Cannot detect required {logical_name} column. "
            f"Available columns: {fieldnames}"
        )
    return None


def _safe_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_.-") or "unknown"


def _numeric_score(row: dict[str, str], columns: dict[str, str | None]) -> tuple[float, str]:
    resolved = columns.get("resolved_score")
    if resolved and str(row.get(resolved, "")).strip():
        return float(row[resolved]), f"resolved:{resolved}"
    values = []
    sources = []
    for key in ("score_1", "score_2"):
        column = columns.get(key)
        if column and str(row.get(column, "")).strip():
            values.append(float(row[column]))
            sources.append(column)
    if not values:
        raise ValueError("No usable score found in row")
    return sum(values) / len(values), "mean:" + ",".join(sources)


def _score_stratum(score: float) -> str:
    return f"{score:.8f}".rstrip("0").rstrip(".")


def _partition_counts(size: int, ratios: dict[str, float]) -> dict[str, int]:
    names = ("train", "calibration", "validation", "test")
    raw = {name: size * ratios[name] for name in names}
    counts = {name: int(raw[name]) for name in names}
    remaining = size - sum(counts.values())
    order = sorted(
        names,
        key=lambda name: (raw[name] - counts[name], -names.index(name)),
        reverse=True,
    )
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def _deterministic_splits(
    records: list[dict[str, Any]],
    *,
    seed: str,
    ratios: dict[str, float],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[_score_stratum(float(record["score"]))].append(record["answer_id"])

    # Interleave score strata first, then assign an exact question-level quota.
    # This avoids starving validation/test when individual score strata are small.
    ordered_groups = {}
    for stratum in sorted(groups):
        ordered_groups[stratum] = sorted(
            groups[stratum],
            key=lambda answer_id: hashlib.sha256(
                f"{seed}|{stratum}|{answer_id}".encode("utf-8")
            ).hexdigest(),
        )
    stratum_order = sorted(
        ordered_groups,
        key=lambda stratum: hashlib.sha256(
            f"{seed}|stratum|{stratum}".encode("utf-8")
        ).hexdigest(),
    )
    interleaved_ids = []
    while any(ordered_groups.values()):
        for stratum in stratum_order:
            if ordered_groups[stratum]:
                interleaved_ids.append(ordered_groups[stratum].pop(0))

    counts = _partition_counts(len(interleaved_ids), ratios)
    slots = [
        (name, index)
        for name in ("train", "calibration", "validation", "test")
        for index in range(counts[name])
    ]
    slots.sort(
        key=lambda slot: hashlib.sha256(
            f"{seed}|slot|{slot[0]}|{slot[1]}".encode("utf-8")
        ).hexdigest()
    )
    split = {name: [] for name in ("train", "calibration", "validation", "test")}
    for answer_id, (name, _) in zip(interleaved_ids, slots):
        split[name].append(answer_id)
    return {name: sorted(values) for name, values in split.items()}


def _default_rubric(question: dict[str, Any], max_score: float) -> list[dict[str, Any]]:
    reference_answer = str(question.get("reference_answer", "")).strip()
    grading_guidance = str(question.get("grading_guidance", "")).strip()
    if not reference_answer or not grading_guidance:
        raise ValueError(
            "Every question without an explicit rubric requires both "
            "reference_answer and grading_guidance"
        )
    return [
        {
            "id": "overall_response",
            "item": grading_guidance,
            "points": max_score,
            "answer_type": "free_text",
            "role": "final",
            "score_layer": "core",
            "canonicalization": {"type": "text"},
            "evidence_source": "text",
            "standard_answer_text": reference_answer,
            "standard_answer_image": None,
            "source_text": grading_guidance,
            "parent_official_item": "holistic short-answer score",
            "metadata_source": "asap_sas",
            "metadata_hard_enabled": False,
            "metadata_confidence": 1.0,
            "parent_id": "overall_response",
            "semantic_contract_version": 5,
            "parent_points": max_score,
            "scoring_policy": "preserve_atomic",
            "split_policy": "preserve_atomic",
            "weighting_policy": "preserve_parent",
            "full_credit_policy": "rubric_evidence_required",
            "full_credit_trigger": False,
        }
    ]


def _question_spec(spec: dict[str, Any], source_question_id: str) -> dict[str, Any]:
    questions = spec.get("questions")
    if not isinstance(questions, dict):
        raise ValueError("Spec must contain a questions object")
    question = questions.get(str(source_question_id))
    if not isinstance(question, dict):
        raise ValueError(
            f"Missing question metadata for source question {source_question_id}"
        )
    if not str(question.get("question_text", "")).strip():
        raise ValueError(
            f"Missing question_text for source question {source_question_id}"
        )
    return question


def prepare_asap_sas(
    source_path: str | Path,
    spec_path: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve()
    spec_file = Path(spec_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ASAP-SAS source file not found: {source}")
    if not spec_file.is_file():
        raise FileNotFoundError(f"ASAP-SAS spec file not found: {spec_file}")
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(
            f"Prepared directory is not empty: {output}. Use --force to replace it."
        )
    if output.exists() and force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    dataset_id = str(spec.get("dataset_id", "asap_sas")).strip()
    delimiter = str(spec.get("delimiter", "\t"))
    encoding = str(spec.get("encoding", "utf-8-sig"))
    explicit_columns = spec.get("columns") or {}
    if not isinstance(explicit_columns, dict):
        raise ValueError("columns must be an object")

    with source.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("ASAP-SAS source has no header")
        fieldnames = list(reader.fieldnames)
        columns = {
            "answer_id": _column_name(
                fieldnames, explicit_columns, "answer_id", required=True
            ),
            "question_id": _column_name(
                fieldnames, explicit_columns, "question_id", required=True
            ),
            "answer_text": _column_name(
                fieldnames, explicit_columns, "answer_text", required=True
            ),
            "resolved_score": _column_name(
                fieldnames, explicit_columns, "resolved_score", required=False
            ),
            "score_1": _column_name(
                fieldnames, explicit_columns, "score_1", required=False
            ),
            "score_2": _column_name(
                fieldnames, explicit_columns, "score_2", required=False
            ),
        }
        rows = list(reader)

    prepared_records: list[dict[str, Any]] = []
    score_sources = set()
    seen_answer_ids = set()
    for row_number, row in enumerate(rows, start=2):
        source_question_id = str(row[columns["question_id"]]).strip()
        source_answer_id = str(row[columns["answer_id"]]).strip()
        answer_text = str(row[columns["answer_text"]]).strip()
        if not source_question_id or not source_answer_id or not answer_text:
            raise ValueError(
                f"Missing question, answer ID, or answer text at row {row_number}"
            )
        question_id = f"ASAP_SAS_{_safe_id(source_question_id)}"
        answer_id = (
            f"{question_id}_{_safe_id(source_answer_id)}"
        )
        if answer_id in seen_answer_ids:
            raise ValueError(f"Duplicate normalized answer ID: {answer_id}")
        seen_answer_ids.add(answer_id)
        score, score_source = _numeric_score(row, columns)
        score_sources.add(score_source)
        prepared_records.append(
            {
                "answer_id": answer_id,
                "question_id": question_id,
                "source_answer_id": source_answer_id,
                "source_question_id": source_question_id,
                "raw_text": answer_text,
                "score": float(score),
                "score_source": score_source,
                "source_row": row_number,
            }
        )

    records_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in prepared_records:
        records_by_question[record["question_id"]].append(record)

    split_spec = spec.get("split") or {}
    seed = str(split_spec.get("seed", "asap-sas-v1"))
    ratios = {
        "train": float(split_spec.get("train", 0.60)),
        "calibration": float(split_spec.get("calibration", 0.15)),
        "validation": float(split_spec.get("validation", 0.10)),
        "test": float(split_spec.get("test", 0.15)),
    }
    if any(value < 0.0 for value in ratios.values()):
        raise ValueError("Split ratios must be non-negative")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    exam_database = []
    teacher_scores = {}
    answer_metadata = []
    normalized_questions = []
    normalized_labels = []
    for question_id in sorted(records_by_question):
        question_records = records_by_question[question_id]
        source_question_id = question_records[0]["source_question_id"]
        question = _question_spec(spec, source_question_id)
        max_score = float(question["max_score"])
        if max_score <= 0:
            raise ValueError(f"max_score must be positive for {question_id}")
        for record in question_records:
            if record["score"] < 0.0 or record["score"] > max_score:
                raise ValueError(
                    f"Score outside [0, {max_score}] for {record['answer_id']}: "
                    f"{record['score']}"
                )

        rubric = question.get("rubric")
        if rubric is None:
            rubric = _default_rubric(question, max_score)
        if not isinstance(rubric, list) or not rubric:
            raise ValueError(f"rubric must be a non-empty list for {question_id}")
        rubric_total = sum(float(item.get("points", 0.0)) for item in rubric)
        if abs(rubric_total - max_score) > 1e-6:
            raise ValueError(
                f"Rubric total {rubric_total} != max_score {max_score} "
                f"for {question_id}"
            )

        rubric_group = "ASAP_SAS"
        rubric_name = f"{question_id}_rubric_standard.json"
        initial_relative = Path("rubrics") / "initial" / rubric_group / rubric_name
        optimized_relative = (
            Path("rubrics") / "optimized" / rubric_group / rubric_name
        )
        split_relative = Path("splits") / f"{question_id}.json"
        write_json(output / initial_relative, rubric)
        write_json(output / optimized_relative, rubric)
        split = _deterministic_splits(
            question_records,
            seed=f"{seed}|{question_id}",
            ratios=ratios,
        )
        empty_required_splits = [
            name for name, ratio in ratios.items() if ratio > 0.0 and not split[name]
        ]
        if empty_required_splits:
            raise ValueError(
                f"{question_id} has too few answers for non-empty formal splits: "
                f"{empty_required_splits}"
            )
        split_payload = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "question_id": question_id,
            "seed": seed,
            "stratification": "question_score_interleaved_exact_quota",
            **split,
        }
        write_json(output / split_relative, split_payload)

        reference_answer = str(question.get("reference_answer", "")).strip()
        grading_guidance = str(question.get("grading_guidance", "")).strip()
        exam_database.append(
            {
                "question_id": question_id,
                "subject": "public_benchmark",
                "exam_source": dataset_id,
                "question_text": str(question["question_text"]).strip(),
                "question_image": None,
                "ref_text": reference_answer,
                "ref_image": None,
                "total_score": max_score,
                "official_rubric": grading_guidance,
                "rubric_group": rubric_group,
                "source_rubric_path": str(initial_relative.as_posix()),
                "initial_rubric_path": str(initial_relative.as_posix()),
                "optimized_rubric_path": str(optimized_relative.as_posix()),
                "rubric_split_path": str(split_relative.as_posix()),
                "rubric_calibration_ids": split["calibration"],
                "student_images_dir": str(
                    (Path("text_samples") / question_id).as_posix()
                ),
                "requires_visual_evidence": False,
                "sample_count": len(question_records),
                "source_question_id": source_question_id,
            }
        )
        normalized_questions.append(
            {
                "question_id": question_id,
                "source_question_id": source_question_id,
                "question_text": str(question["question_text"]).strip(),
                "reference_answer": reference_answer,
                "grading_guidance": grading_guidance,
                "max_score": max_score,
            }
        )
        for record in question_records:
            teacher_scores[record["answer_id"]] = {
                question_id: record["score"]
            }
            answer_metadata.append(
                {
                    "answer_id": record["answer_id"],
                    "question_id": question_id,
                    "subject": "public_benchmark",
                    "subject_name": "ASAP-SAS",
                    "raw_text": record["raw_text"],
                    "isimagine": False,
                    "visual_placeholder_detected": False,
                    "student_image": None,
                    "source_image": None,
                    "actual_score": record["score"],
                    "source_file": source.name,
                    "source_line": record["source_row"],
                    "source_answer_id": record["source_answer_id"],
                    "source_question_id": source_question_id,
                    "score_source": record["score_source"],
                }
            )
            normalized_labels.append(
                {
                    "answer_id": record["answer_id"],
                    "question_id": question_id,
                    "score": record["score"],
                    "score_source": record["score_source"],
                }
            )

    write_json(output / "exam_database.json", exam_database)
    write_json(output / "teacher_scores.json", teacher_scores)
    write_jsonl(output / "answer_metadata.jsonl", answer_metadata)
    write_jsonl(output / "normalized" / "questions.jsonl", normalized_questions)
    write_jsonl(output / "normalized" / "answers.jsonl", answer_metadata)
    write_jsonl(output / "normalized" / "gold_labels.jsonl", normalized_labels)
    (output / "source").mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec_file, output / "source" / "adapter_spec.json")

    manifest = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "adapter": "asap_sas",
        "task_type": "numeric_short_answer_scoring",
        "extraction_backend": "text_only",
        "source": {
            "filename": source.name,
            "sha256": sha256_file(source),
            "adapter_spec_sha256": sha256_file(spec_file),
        },
        "score_label_policy": {
            "priority": "resolved_score_then_mean_available_raters",
            "observed_sources": sorted(score_sources),
        },
        "split": {
            "seed": seed,
            "ratios": ratios,
            "stratification": "question_score_interleaved_exact_quota",
        },
        "counts": {
            "questions": len(exam_database),
            "answers": len(answer_metadata),
        },
        "question_ids": [item["question_id"] for item in exam_database],
    }
    write_json(output / "manifest.json", manifest)
    audit = audit_prepared_benchmark(output)
    write_json(output / "audit.json", audit)
    return audit
