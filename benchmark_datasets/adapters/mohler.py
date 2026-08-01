from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_datasets.adapters.asap_sas import _deterministic_splits
from benchmark_datasets.contract import (
    BENCHMARK_SCHEMA_VERSION,
    audit_prepared_benchmark,
    sha256_file,
    write_json,
    write_jsonl,
)
from rubric_semantics import (
    RUBRIC_SEMANTIC_CONTRACT_VERSION,
    prepare_rubric_semantic_contract,
    validate_refined_rubric,
)


QUESTION_ID_PATTERN = re.compile(r"^(\d+\.\d+)\s+(.*)$")
BREAK_PATTERN = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
SPLIT_NAMES = ("train", "calibration", "validation", "test")


def _clean_text(value: str) -> str:
    text = BREAK_PATTERN.sub("\n", html.unescape(str(value)))
    return " ".join(text.split())


def _load_keyed_text(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current_id: str | None = None
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        match = QUESTION_ID_PATTERN.match(line)
        if match:
            current_id = match.group(1)
            if current_id in records:
                raise ValueError(f"Duplicate question ID in {path}:{line_number}")
            records[current_id] = [match.group(2)]
        elif current_id is None:
            raise ValueError(f"Text before first question ID in {path}:{line_number}")
        else:
            records[current_id].append(line)
    return {
        question_id: _clean_text(" ".join(parts))
        for question_id, parts in records.items()
    }


def _included_question_ids(path: Path) -> list[str]:
    question_ids = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if not re.fullmatch(r"\d+\.\d+", value):
            raise ValueError(f"Invalid Mohler question ID in {path}: {value!r}")
        question_ids.append(value)
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"Duplicate question IDs in {path}")
    return question_ids


def _excluded_question_ids(path: Path) -> list[str]:
    question_ids = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        value = line.strip()
        if not value.startswith("#"):
            continue
        question_id = value[1:].strip()
        if not re.fullmatch(r"\d+\.\d+", question_id):
            raise ValueError(
                f"Invalid excluded Mohler question ID in {path}: {value!r}"
            )
        question_ids.append(question_id)
    return question_ids


def _student_answers(path: Path, question_id: str) -> list[str]:
    answers = []
    prefix = f"{question_id} "
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith(prefix):
            raise ValueError(
                f"Unexpected answer prefix in {path}:{line_number}; "
                f"expected {question_id!r}"
            )
        answer = _clean_text(line[len(prefix) :])
        if not answer:
            raise ValueError(f"Empty student answer in {path}:{line_number}")
        answers.append(answer)
    return answers


def _scores(path: Path) -> list[float]:
    values = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="strict").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            values.append(float(line.strip()))
        except ValueError as exc:
            raise ValueError(f"Invalid score in {path}:{line_number}") from exc
    return values


def _normalized_rater_scores(question_id: str, values: list[float]) -> list[float]:
    assignment = int(question_id.split(".", 1)[0])
    divisor = 2.0 if assignment in {11, 12} else 1.0
    return [value / divisor for value in values]


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _rubric(reference_answer: str) -> list[dict[str, Any]]:
    guidance = (
        "Assign a score from 0 to 5 for semantic correctness and completeness "
        "relative to the instructor answer. Give proportional partial credit; "
        "do not require exact wording when the meaning is equivalent."
    )
    return prepare_rubric_semantic_contract([
        {
            "id": "overall_response",
            "item": guidance,
            "points": 5.0,
            "answer_type": "free_text",
            "role": "final",
            "score_layer": "core",
            "canonicalization": {"type": "text"},
            "evidence_source": "text",
            "standard_answer_text": reference_answer,
            "standard_answer_image": None,
            "source_text": "Reconstructed from the official Mohler 0-5 annotation scale.",
            "parent_official_item": "holistic short-answer correctness",
            "metadata_source": "mohler_reference_based",
            "metadata_hard_enabled": False,
            "metadata_confidence": 1.0,
            "parent_id": "overall_response",
            "semantic_contract_version": RUBRIC_SEMANTIC_CONTRACT_VERSION,
            "parent_points": 5.0,
            "scoring_policy": "preserve_atomic",
            "split_policy": "preserve_atomic",
            "weighting_policy": "preserve_parent",
            "full_credit_policy": "rubric_evidence_required",
            "full_credit_trigger": False,
        }
    ])


def _resolve_data_root(source: Path) -> Path:
    candidates = (source / "data", source)
    for candidate in candidates:
        if (candidate / "docs" / "files").is_file():
            return candidate
    raise FileNotFoundError(
        "Mohler source must be the extracted archive root or its data directory"
    )


def prepare_mohler(
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
        raise FileNotFoundError(f"Extracted Mohler source not found: {source}")
    if not spec_file.is_file():
        raise FileNotFoundError(f"Mohler spec file not found: {spec_file}")
    data_root = _resolve_data_root(source)
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(
            f"Prepared directory is not empty: {output}. Use --force to replace it."
        )
    if output.exists() and force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    dataset_id = str(spec.get("dataset_id", "mohler_v1")).strip()
    split_spec = spec.get("split") or {}
    seed = str(split_spec.get("seed", "mohler-v1"))
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

    question_list_path = data_root / "docs" / "files"
    question_ids = _included_question_ids(question_list_path)
    excluded_question_ids = _excluded_question_ids(question_list_path)
    questions = _load_keyed_text(data_root / "raw" / "questions")
    references = _load_keyed_text(data_root / "raw" / "answers")
    missing_questions = set(question_ids) - set(questions)
    missing_references = set(question_ids) - set(references)
    if missing_questions or missing_references:
        raise ValueError(
            "Mohler metadata is incomplete: "
            f"questions={sorted(missing_questions)}, "
            f"references={sorted(missing_references)}"
        )

    records_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_question_id in question_ids:
        answers = _student_answers(
            data_root / "raw" / source_question_id,
            source_question_id,
        )
        averages = _scores(data_root / "scores" / source_question_id / "ave")
        grader_mohler_raw = _scores(
            data_root / "scores" / source_question_id / "me"
        )
        grader_ta_raw = _scores(
            data_root / "scores" / source_question_id / "other"
        )
        grader_mohler = _normalized_rater_scores(
            source_question_id,
            grader_mohler_raw,
        )
        grader_ta = _normalized_rater_scores(
            source_question_id,
            grader_ta_raw,
        )
        lengths = {len(answers), len(averages), len(grader_mohler), len(grader_ta)}
        if len(lengths) != 1:
            raise ValueError(
                f"Answer/score count mismatch for {source_question_id}: "
                f"answers={len(answers)}, average={len(averages)}, "
                f"grader_mohler={len(grader_mohler)}, grader_ta={len(grader_ta)}"
            )
        question_id = f"MOHLER_{source_question_id.replace('.', '_')}"
        for index, (
            answer,
            average,
            rater_1,
            rater_2,
            rater_1_raw,
            rater_2_raw,
        ) in enumerate(
            zip(
                answers,
                averages,
                grader_ta,
                grader_mohler,
                grader_ta_raw,
                grader_mohler_raw,
            ),
            start=1,
        ):
            if not 0.0 <= average <= 5.0:
                raise ValueError(
                    f"Gold score outside [0, 5] for {source_question_id}/{index}: "
                    f"{average}"
                )
            records_by_question[question_id].append(
                {
                    "answer_id": f"{question_id}_{index:03d}",
                    "question_id": question_id,
                    "source_question_id": source_question_id,
                    "source_answer_index": index - 1,
                    "raw_text": answer,
                    "score": average,
                    "rater_1_score": rater_1,
                    "rater_2_score": rater_2,
                    "rater_1_raw_score": rater_1_raw,
                    "rater_2_raw_score": rater_2_raw,
                }
            )

    expected_counts = spec.get("expected_counts") or {}
    answer_count = sum(len(records) for records in records_by_question.values())
    expected_questions = expected_counts.get("questions")
    expected_answers = expected_counts.get("answers")
    if expected_questions is not None and int(expected_questions) != len(
        records_by_question
    ):
        raise ValueError(
            "Mohler question count does not match the adapter specification: "
            f"expected={expected_questions}, actual={len(records_by_question)}"
        )
    if expected_answers is not None and int(expected_answers) != answer_count:
        raise ValueError(
            "Mohler answer count does not match the adapter specification: "
            f"expected={expected_answers}, actual={answer_count}"
        )

    exam_database = []
    teacher_scores = {}
    answer_metadata = []
    normalized_questions = []
    normalized_labels = []
    for question_id in sorted(records_by_question):
        question_records = records_by_question[question_id]
        source_question_id = question_records[0]["source_question_id"]
        question_text = questions[source_question_id]
        reference_answer = references[source_question_id]
        rubric = _rubric(reference_answer)
        rubric_group = "MOHLER"
        rubric_name = f"{question_id}_rubric_standard.json"
        initial_relative = Path("rubrics") / "initial" / rubric_group / rubric_name
        optimized_relative = Path("rubrics") / "optimized" / rubric_group / rubric_name
        optimization_manifest_relative = (
            Path("rubrics")
            / "manifests"
            / rubric_group
            / f"{question_id}_optimization.json"
        )
        split_relative = Path("splits") / f"{question_id}.json"
        write_json(output / initial_relative, rubric)
        write_json(output / optimized_relative, rubric)
        split = _deterministic_splits(
            question_records,
            seed=f"{seed}|{question_id}",
            ratios=ratios,
        )
        empty_required = [
            name for name in SPLIT_NAMES if ratios[name] > 0.0 and not split[name]
        ]
        if empty_required:
            raise ValueError(
                f"{question_id} has too few answers for formal splits: "
                f"{empty_required}"
            )
        write_json(
            output / split_relative,
            {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "question_id": question_id,
                "seed": seed,
                "stratification": "question_score_interleaved_exact_quota",
                **split,
            },
        )
        semantic_valid, semantic_errors = validate_refined_rubric(
            rubric,
            rubric,
            5.0,
            allow_unchanged_baseline=True,
        )
        if not semantic_valid:
            raise ValueError(
                f"Generated baseline rubric is invalid for {question_id}: "
                + "; ".join(semantic_errors)
            )
        write_json(
            output / optimization_manifest_relative,
            {
                "question_id": question_id,
                "rubric_group": rubric_group,
                "method": "prepared_reference_baseline",
                "rubric_semantic_contract_version": (
                    RUBRIC_SEMANTIC_CONTRACT_VERSION
                ),
                "semantic_policy_validated": True,
                "initial_rubric": initial_relative.as_posix(),
                "optimized_rubric": optimized_relative.as_posix(),
                "initial_sha256": sha256_file(output / initial_relative),
                "optimized_sha256": sha256_file(output / optimized_relative),
                "calibration_answer_ids": split["calibration"],
                "path_format": "prepared_relative_v1",
                "note": (
                    "The optimized slot intentionally mirrors the validated "
                    "reconstructed baseline; no test labels were used."
                ),
            },
        )
        grading_guidance = (
            "Reference-based holistic correctness on the official 0-5 Mohler scale; "
            "this is a reconstructed operational rubric, not an official fine-grained rubric."
        )
        exam_database.append(
            {
                "question_id": question_id,
                "subject": "public_benchmark",
                "exam_source": dataset_id,
                "question_text": question_text,
                "question_image": None,
                "ref_text": reference_answer,
                "ref_image": None,
                "total_score": 5.0,
                "official_rubric": grading_guidance,
                "rubric_group": rubric_group,
                "source_rubric_path": initial_relative.as_posix(),
                "initial_rubric_path": initial_relative.as_posix(),
                "optimized_rubric_path": optimized_relative.as_posix(),
                "rubric_split_path": split_relative.as_posix(),
                "rubric_calibration_ids": split["calibration"],
                "student_images_dir": (Path("text_samples") / question_id).as_posix(),
                "requires_visual_evidence": False,
                "sample_count": len(question_records),
                "source_question_id": source_question_id,
            }
        )
        normalized_questions.append(
            {
                "question_id": question_id,
                "source_question_id": source_question_id,
                "question_text": question_text,
                "reference_answer": reference_answer,
                "grading_guidance": grading_guidance,
                "max_score": 5.0,
            }
        )
        for record in question_records:
            answer_id = record["answer_id"]
            teacher_scores[answer_id] = {question_id: record["score"]}
            metadata = {
                "answer_id": answer_id,
                "question_id": question_id,
                "subject": "public_benchmark",
                "subject_name": "Mohler Short Answer Grading",
                "raw_text": record["raw_text"],
                "isimagine": False,
                "visual_placeholder_detected": False,
                "student_image": None,
                "source_image": None,
                "actual_score": record["score"],
                "source_file": f"data/raw/{source_question_id}",
                "source_line": record["source_answer_index"] + 1,
                "source_answer_id": str(record["source_answer_index"]),
                "source_question_id": source_question_id,
                "score_source": "official_average_normalized_0_5",
                "rater_1_score": record["rater_1_score"],
                "rater_2_score": record["rater_2_score"],
                "rater_1_raw_score": record["rater_1_raw_score"],
                "rater_2_raw_score": record["rater_2_raw_score"],
            }
            answer_metadata.append(metadata)
            normalized_labels.append(
                {
                    "answer_id": answer_id,
                    "question_id": question_id,
                    "score": record["score"],
                    "score_source": "official_average_normalized_0_5",
                    "rater_1_score": record["rater_1_score"],
                    "rater_2_score": record["rater_2_score"],
                    "rater_1_raw_score": record["rater_1_raw_score"],
                    "rater_2_raw_score": record["rater_2_raw_score"],
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
        "adapter": "mohler",
        "task_type": "numeric_short_answer_scoring",
        "extraction_backend": "text_only",
        "source": {
            "filename": source.name,
            "sha256": _tree_sha256(source),
            "adapter_spec_sha256": sha256_file(spec_file),
        },
        "score_label_policy": {
            "priority": "official_average_normalized_0_5",
            "raw_raters_retained": True,
        },
        "rubric_policy": {
            "type": "reference_based_reconstructed_holistic",
            "official_fine_grained_rubric": False,
        },
        "question_selection": {
            "authority": "data/docs/files",
            "included_count": len(question_ids),
            "excluded_question_ids": excluded_question_ids,
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
