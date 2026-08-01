from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


BENCHMARK_SCHEMA_VERSION = 1
SPLIT_NAMES = ("train", "calibration", "validation", "test")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(target)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            records.append(payload)
    return records


def _rubric_total(path: Path) -> float:
    rubric = load_json(path)
    if not isinstance(rubric, list) or not rubric:
        raise ValueError(f"Rubric must be a non-empty list: {path}")
    return sum(float(item.get("points", 0.0)) for item in rubric)


def audit_prepared_benchmark(prepared_dir: str | Path) -> dict[str, Any]:
    root = Path(prepared_dir).expanduser().resolve()
    required = {
        "manifest": root / "manifest.json",
        "database": root / "exam_database.json",
        "teacher_scores": root / "teacher_scores.json",
        "answer_metadata": root / "answer_metadata.jsonl",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Prepared benchmark is missing: {', '.join(missing)}"
        )

    manifest = load_json(required["manifest"])
    if manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported benchmark schema version: "
            f"{manifest.get('schema_version')}"
        )
    questions = load_json(required["database"])
    teacher_scores = load_json(required["teacher_scores"])
    metadata = read_jsonl(required["answer_metadata"])
    if not isinstance(questions, list) or not questions:
        raise ValueError("exam_database.json must contain at least one question")
    if not isinstance(teacher_scores, dict):
        raise ValueError("teacher_scores.json must be an object")
    adapter_spec_path = root / "source" / "adapter_spec.json"
    expected_spec_hash = manifest.get("source", {}).get("adapter_spec_sha256")
    if expected_spec_hash:
        if not adapter_spec_path.is_file():
            raise FileNotFoundError(
                f"Prepared benchmark is missing adapter spec: {adapter_spec_path}"
            )
        if sha256_file(adapter_spec_path) != expected_spec_hash:
            raise ValueError("Adapter spec hash does not match manifest")

    metadata_by_id: dict[str, dict[str, Any]] = {}
    question_to_answers: dict[str, set[str]] = {}
    for record in metadata:
        answer_id = str(record.get("answer_id", ""))
        question_id = str(record.get("question_id", ""))
        if not answer_id or not question_id:
            raise ValueError("Every answer requires answer_id and question_id")
        if answer_id in metadata_by_id:
            raise ValueError(f"Duplicate answer_id: {answer_id}")
        if not str(record.get("raw_text", "")).strip():
            raise ValueError(f"Empty raw_text: {answer_id}")
        metadata_by_id[answer_id] = record
        question_to_answers.setdefault(question_id, set()).add(answer_id)

    database_ids = {str(item.get("question_id", "")) for item in questions}
    if "" in database_ids or len(database_ids) != len(questions):
        raise ValueError("Question IDs must be unique and non-empty")
    unknown_questions = set(question_to_answers) - database_ids
    if unknown_questions:
        raise ValueError(
            f"Answers reference unknown questions: {sorted(unknown_questions)}"
        )

    split_counts = {name: 0 for name in SPLIT_NAMES}
    split_ratios = (manifest.get("split") or {}).get("ratios") or {}
    question_assets = {}
    score_count = 0
    for question in questions:
        question_id = str(question["question_id"])
        total_score = float(question["total_score"])
        split_path = Path(str(question.get("rubric_split_path", "")))
        if not split_path.is_absolute():
            split_path = root / split_path
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Missing split file for {question_id}: {split_path}"
            )
        split = load_json(split_path)
        split_sets = {
            name: {str(value) for value in split.get(name, [])}
            for name in SPLIT_NAMES
        }
        for index, left in enumerate(SPLIT_NAMES):
            for right in SPLIT_NAMES[index + 1 :]:
                overlap = split_sets[left] & split_sets[right]
                if overlap:
                    raise ValueError(
                        f"{question_id} split overlap {left}/{right}: "
                        f"{sorted(overlap)[:5]}"
                    )
        split_union = set().union(*split_sets.values())
        expected = question_to_answers.get(question_id, set())
        if not expected:
            raise ValueError(f"{question_id} has no answer records")
        if split_union != expected:
            raise ValueError(
                f"{question_id} split coverage mismatch: "
                f"missing={len(expected - split_union)}, "
                f"extra={len(split_union - expected)}"
            )
        empty_required = [
            name
            for name in SPLIT_NAMES
            if float(split_ratios.get(name, 0.0)) > 0.0
            and not split_sets[name]
        ]
        if empty_required:
            raise ValueError(
                f"{question_id} has empty required splits: {empty_required}"
            )
        for name, values in split_sets.items():
            split_counts[name] += len(values)

        initial_path = Path(str(question.get("initial_rubric_path", "")))
        optimized_path = Path(str(question.get("optimized_rubric_path", "")))
        if not initial_path.is_absolute():
            initial_path = root / initial_path
        if not optimized_path.is_absolute():
            optimized_path = root / optimized_path
        for rubric_path in (initial_path, optimized_path):
            if not rubric_path.is_file():
                raise FileNotFoundError(
                    f"Missing rubric for {question_id}: {rubric_path}"
                )
            if abs(_rubric_total(rubric_path) - total_score) > 1e-6:
                raise ValueError(
                    f"{question_id} rubric total does not equal {total_score}: "
                    f"{rubric_path}"
                )
        question_assets[question_id] = {
            "split_sha256": sha256_file(split_path),
            "initial_rubric_sha256": sha256_file(initial_path),
            "optimized_rubric_sha256": sha256_file(optimized_path),
        }

        for answer_id in expected:
            score_record = teacher_scores.get(answer_id)
            if not isinstance(score_record, dict) or question_id not in score_record:
                raise ValueError(
                    f"Missing gold score for {question_id}/{answer_id}"
                )
            score = float(score_record[question_id])
            if score < 0.0 or score > total_score:
                raise ValueError(
                    f"Gold score outside [0, {total_score}] for {answer_id}: "
                    f"{score}"
                )
            score_count += 1

    if set(metadata_by_id) != set(teacher_scores):
        raise ValueError(
            "teacher_scores and answer_metadata must contain the same answer IDs"
        )

    snapshot = {
        "dataset_id": manifest["dataset_id"],
        "schema_version": manifest["schema_version"],
        "question_count": len(questions),
        "answer_count": len(metadata),
        "score_count": score_count,
        "split_counts": split_counts,
        "source_sha256": manifest.get("source", {}).get("sha256"),
        "question_assets_sha256": sha256_json(question_assets),
        "prepared_content_sha256": sha256_json(
            {
                "questions": questions,
                "teacher_scores": teacher_scores,
                "answer_metadata": metadata,
                "question_assets": question_assets,
                "adapter_spec_sha256": (
                    sha256_file(adapter_spec_path)
                    if adapter_spec_path.is_file()
                    else None
                ),
            }
        ),
    }
    expected_counts = manifest.get("counts") or {}
    if expected_counts and (
        int(expected_counts.get("questions", -1)) != len(questions)
        or int(expected_counts.get("answers", -1)) != len(metadata)
    ):
        raise ValueError("Manifest counts do not match prepared content")
    return snapshot
