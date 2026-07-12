"""Strictly audit an external CSBench source and the embedded grading snapshot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANSWER_FILES = (
    "answer.jsonl",
    "answer_CPL.jsonl",
    "answer_DM.jsonl",
    "answer_ISC.jsonl",
    "answer_ML.jsonl",
    "answer_POC.jsonl",
    "answer_POC_25.jsonl",
    "answer_POC_26.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root")
    parser.add_argument("--prepared-dir", default="data/csbench")
    parser.add_argument("--exclude-questions", nargs="*", default=["OS_1", "OS_2"])
    return parser.parse_args()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_loads(text: str) -> Any:
    return json.loads(text, object_pairs_hook=reject_duplicate_keys)


def flatten(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    elif isinstance(value, dict):
        yield value


def source_image_path(source_root: Path, raw_path: str) -> Path | None:
    relative = Path(raw_path)
    candidates = [source_root / relative]
    if relative.parts and relative.parts[0].lower() == "images":
        candidates.insert(
            0,
            source_root / "images" / "cleaned" / Path(*relative.parts[1:]),
        )
    return next((path for path in candidates if path.is_file()), None)


def audit_source(
    source_root: Path,
    excluded_questions: set[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    questions: dict[str, Any] = {}
    question_files = 0
    nested_question_files = []
    for path in sorted((source_root / "question").glob("*.json")):
        payload = strict_loads(path.read_text(encoding="utf-8"))
        question_files += 1
        if isinstance(payload, list) and any(isinstance(item, list) for item in payload):
            nested_question_files.append(path.name)
        for question in flatten(payload):
            question_id = str(question.get("question_id", ""))
            if not question_id:
                continue
            if question_id in questions:
                raise ValueError(f"duplicate question_id: {question_id}")
            rubric = question.get("grading_rubric") or []
            total = sum(float(item.get("score", 0)) for item in rubric)
            if abs(total - float(question.get("max_score", 0))) > 1e-6:
                raise ValueError(
                    f"{question_id} rubric total {total} != {question.get('max_score')}"
                )
            step_ids = [str(item.get("step_id")) for item in rubric]
            if len(step_ids) != len(set(step_ids)):
                raise ValueError(f"{question_id} has duplicate rubric step_id")
            for raw in question.get("content", {}).get("image_paths") or []:
                if not (source_root / raw).is_file():
                    raise FileNotFoundError(f"missing question image: {raw}")
            for item in rubric:
                raw = item.get("standard_answer_image")
                if raw and not (source_root / raw).is_file():
                    raise FileNotFoundError(f"missing standard image: {raw}")
            questions[question_id] = question

    answers: dict[str, Any] = {}
    answer_counts: Counter[str] = Counter()
    file_counts: dict[str, int] = {}
    excluded_answer_count = 0
    score_out_of_range = []
    for filename in ANSWER_FILES:
        path = source_root / "answer" / filename
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    answer = strict_loads(line)
                except Exception as exc:
                    raise ValueError(f"{filename}:{line_number}: {exc}") from exc
                answer_id = str(answer.get("answer_id", ""))
                question_id = str(answer.get("question_id", ""))
                if not answer_id or answer_id in answers:
                    raise ValueError(f"missing or duplicate answer_id: {answer_id}")
                if question_id not in questions:
                    raise ValueError(f"{answer_id} references unknown {question_id}")
                if question_id in excluded_questions:
                    excluded_answer_count += 1
                    count += 1
                    continue
                max_score = float(questions[question_id].get("max_score", 0))
                actual_score = float(answer.get("actual_score", 0))
                if actual_score < 0 or actual_score > max_score:
                    score_out_of_range.append(
                        {
                            "answer_id": answer_id,
                            "question_id": question_id,
                            "score": actual_score,
                            "max_score": max_score,
                        }
                    )
                image_paths = answer.get("student_input", {}).get("image_paths") or []
                if len(image_paths) != 1 or not source_image_path(source_root, image_paths[0]):
                    raise FileNotFoundError(f"{answer_id} has invalid image reference")
                answers[answer_id] = answer
                answer_counts[question_id] += 1
                count += 1
        file_counts[filename] = count

    return questions, answers, {
        "question_file_count": question_files,
        "question_count": len(questions),
        "nested_question_files": nested_question_files,
        "answer_count": len(answers),
        "excluded_answer_count": excluded_answer_count,
        "score_out_of_range": score_out_of_range,
        "answer_file_counts": file_counts,
        "answer_question_counts": dict(sorted(answer_counts.items())),
    }


def audit_prepared(prepared: Path, source_answers: dict[str, Any] | None) -> dict[str, Any]:
    database = strict_loads((prepared / "exam_database.json").read_text(encoding="utf-8"))
    questions = {item["question_id"]: item for item in database}
    if len(questions) != len(database):
        raise ValueError("embedded exam_database has duplicate question_id")

    metadata = {}
    with (prepared / "answer_metadata.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = strict_loads(line)
            answer_id = str(record["answer_id"])
            if answer_id in metadata:
                raise ValueError(f"duplicate embedded answer_id: {answer_id}")
            image = PROJECT_ROOT / record["student_image"]
            if not image.is_file():
                raise FileNotFoundError(f"missing embedded image: {image}")
            metadata[answer_id] = record

    teacher_scores = strict_loads(
        (prepared / "teacher_scores.json").read_text(encoding="utf-8")
    )
    if set(metadata) != set(teacher_scores):
        raise ValueError("embedded metadata and teacher score IDs differ")

    split_ids = set()
    for question_id in questions:
        split = strict_loads(
            (prepared / "splits" / "by_question" / f"{question_id}.json").read_text(
                encoding="utf-8"
            )
        )
        local_ids = []
        for name in ("calibration", "validation", "test"):
            local_ids.extend(split.get(name, []))
        if len(local_ids) != len(set(local_ids)):
            raise ValueError(f"{question_id} split overlap or duplicate")
        split_ids.update(local_ids)
    if split_ids != set(metadata):
        raise ValueError("embedded split IDs do not cover metadata exactly")

    for question in database:
        for key in ("source_rubric_path", "initial_rubric_path", "rubric_split_path"):
            if not (PROJECT_ROOT / question[key]).is_file():
                raise FileNotFoundError(f"missing embedded path: {question[key]}")
        for key in ("question_image", "ref_image"):
            if question.get(key) and not (PROJECT_ROOT / question[key]).is_file():
                raise FileNotFoundError(f"missing embedded path: {question[key]}")

    compared = 0
    if source_answers is not None:
        expected = {
            answer_id: answer
            for answer_id, answer in source_answers.items()
            if answer.get("question_id") in questions
        }
        if set(expected) != set(metadata):
            raise ValueError("embedded answer IDs differ from source after exclusions")
        for answer_id, record in metadata.items():
            source = expected[answer_id]
            if float(record["actual_score"]) != float(source["actual_score"]):
                raise ValueError(f"teacher score mismatch for {answer_id}")
            if record["raw_text"] != str(source.get("student_input", {}).get("raw_text", "")):
                raise ValueError(f"raw_text mismatch for {answer_id}")
            compared += 1

    return {
        "question_count": len(questions),
        "answer_count": len(metadata),
        "source_compared_answers": compared,
        "student_image_count": len(list((prepared / "student_images").rglob("*.*"))),
        "reference_image_count": len(list((prepared / "reference_images").rglob("*.*"))),
    }


def main() -> int:
    args = parse_args()
    source_report = None
    source_answers = None
    if args.source_root:
        _, source_answers, source_report = audit_source(
            Path(args.source_root).resolve(),
            set(args.exclude_questions or []),
        )
    prepared_report = audit_prepared(
        (PROJECT_ROOT / args.prepared_dir).resolve(), source_answers
    )
    print(
        json.dumps(
            {"source": source_report, "embedded": prepared_report},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
