"""Prepare the private CSBench test split for the SAS-Bench scoring method."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


GENERIC_ERRORS = [
    {"name": "概念错误", "description": "核心概念或知识点理解错误"},
    {"name": "计算错误", "description": "数值计算、符号运算或单位换算错误"},
    {"name": "推理错误", "description": "推导关系、逻辑顺序或因果判断错误"},
    {"name": "步骤缺失", "description": "必要步骤、条件或论证缺失"},
    {"name": "结论错误", "description": "最终结论与正确答案不一致"},
    {"name": "答非所问", "description": "回答内容未对应题目要求"},
    {"name": "表达不清", "description": "表达含混，无法确定其准确含义"},
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(paths: list[Path], *, relative_to: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(relative_to).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def task_filename(index: int, question_id: str) -> str:
    safe_id = "".join(char if char.isalnum() else "_" for char in question_id)
    return f"{index}_{safe_id}_ShortAns.jsonl"


def response_type(item: dict[str, Any]) -> str:
    raw_text = str(item.get("raw_text") or "").strip()
    if not raw_text:
        return "blank"
    if bool(item.get("isimagine")) or bool(item.get("visual_placeholder_detected")):
        return "visual_dependent"
    return "pure_text"


def prepare(
    prepared_dir: Path,
    output_dir: Path,
    *,
    split: str = "test",
    selection_protocol: str = "outer-split",
    include_empty: bool = False,
    limit_per_task: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    prepared_dir = prepared_dir.resolve()
    output_dir = output_dir.resolve()
    required = {
        "exam_database": prepared_dir / "exam_database.json",
        "teacher_scores": prepared_dir / "teacher_scores.json",
        "answer_metadata": prepared_dir / "answer_metadata.jsonl",
    }
    if selection_protocol == "outer-split":
        required["split"] = prepared_dir / "splits" / f"{split}.json"
    elif selection_protocol != "per-question-test":
        raise ValueError(f"Unsupported selection_protocol: {selection_protocol}")
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing CSBench inputs: " + ", ".join(missing))
    if limit_per_task is not None and limit_per_task < 1:
        raise ValueError("limit_per_task must be at least 1")
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists; use --force: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    questions = {item["question_id"]: item for item in read_json(required["exam_database"])}
    scores = read_json(required["teacher_scores"])
    metadata_rows = read_jsonl(required["answer_metadata"])
    metadata = {item["answer_id"]: item for item in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise ValueError("Duplicate answer_id in answer_metadata.jsonl")
    split_paths: list[Path]
    if selection_protocol == "outer-split":
        split_paths = [required["split"]]
        split_data = read_json(required["split"])
        question_ids = list(split_data.get("question_ids") or [])
        answer_ids = list(split_data.get("answer_ids") or [])
    else:
        question_ids = list(questions)
        split_paths = [
            prepared_dir / "splits" / "by_question" / f"{question_id}.json"
            for question_id in question_ids
        ]
        missing_splits = [str(path) for path in split_paths if not path.is_file()]
        if missing_splits:
            raise FileNotFoundError(
                "Missing per-question CSBench splits: " + ", ".join(missing_splits)
            )
        answer_ids = []
        for question_id, path in zip(question_ids, split_paths):
            split_data = read_json(path)
            if split_data.get("question_id") not in (None, question_id):
                raise ValueError(f"Question split mismatch in {path}")
            answer_ids.extend(split_data.get(split) or [])
    if not question_ids or not answer_ids:
        raise ValueError(f"Split {split!r} has no questions or answers")

    grouped: dict[str, list[dict[str, Any]]] = {qid: [] for qid in question_ids}
    excluded: list[dict[str, Any]] = []
    selected_type_counts = {"pure_text": 0, "visual_dependent": 0, "blank": 0}
    for answer_id in answer_ids:
        item = metadata.get(answer_id)
        if item is None:
            raise ValueError(f"Missing metadata for {answer_id}")
        question_id = str(item.get("question_id", ""))
        if question_id not in grouped:
            raise ValueError(f"Answer {answer_id} is outside split questions: {question_id}")
        question = questions.get(question_id)
        if question is None:
            raise ValueError(f"Missing question metadata for {question_id}")
        raw_text = str(item.get("raw_text") or "").strip()
        item_response_type = response_type(item)
        selected_type_counts[item_response_type] += 1
        if item_response_type == "blank" and not include_empty:
            excluded.append(
                {
                    "answer_id": answer_id,
                    "question_id": question_id,
                    "reason": "empty_raw_text",
                    "student_image": item.get("student_image"),
                }
            )
            continue
        teacher = (scores.get(answer_id) or {}).get(question_id)
        try:
            teacher_score = float(teacher)
            total = float(question["total_score"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid score contract for {answer_id}") from exc
        if not math.isfinite(teacher_score) or not 0 <= teacher_score <= total:
            raise ValueError(
                f"Teacher score outside [0, total] for {answer_id}: "
                f"{teacher_score}/{total}"
            )
        grouped[question_id].append(
            {
                "id": answer_id,
                "question": str(question.get("question_text") or ""),
                "reference": str(question.get("ref_text") or ""),
                "analysis": "",
                "total": total,
                "manual_label": teacher_score,
                "steps": [
                    {"response": raw_text, "label": None, "errors": []}
                ],
                "source_question_id": question_id,
                "response_type": item_response_type,
                "source_isimagine": bool(item.get("isimagine")),
                "source_visual_placeholder_detected": bool(
                    item.get("visual_placeholder_detected")
                ),
            }
        )

    eligible = sum(len(rows) for rows in grouped.values())
    task_manifest = []
    error_contracts = []
    written = 0
    formal_type_counts = {"pure_text": 0, "visual_dependent": 0, "blank": 0}
    for index, question_id in enumerate(question_ids):
        question = questions[question_id]
        rows = grouped[question_id]
        if limit_per_task is not None:
            rows = rows[:limit_per_task]
        filename = task_filename(index, question_id)
        write_jsonl(output_dir / filename, rows)
        task_manifest.append(
            {
                "task_id": index,
                "question_id": question_id,
                "file": filename,
                "records": len(rows),
            }
        )
        error_contracts.append(
            {
                "q_id": index,
                "course": str(question.get("subject") or "CS"),
                "question_type": "ShortAns",
                "guideline": str(question.get("official_rubric") or ""),
                "errors": GENERIC_ERRORS,
            }
        )
        written += len(rows)
        for row in rows:
            formal_type_counts[row["response_type"]] += 1

    write_jsonl(output_dir / "error_type.jsonl", error_contracts)
    write_jsonl(output_dir / "excluded_records.jsonl", excluded)
    manifest = {
        "schema_version": 1,
        "source": "private_csbench",
        "target_method": "sas_bench_official_prompt_protocol",
        "selection_protocol": selection_protocol,
        "split": split,
        "source_question_count": len(question_ids),
        "source_record_count": len(answer_ids),
        "eligible_record_count": eligible,
        "formal_record_count": written,
        "excluded_record_count": len(excluded),
        "exclusion_reasons": {"empty_raw_text": len(excluded)},
        "student_step_policy": "single_step_full_raw_text",
        "analysis_policy": "empty_no_fabrication",
        "guideline_source": "exam_database.official_rubric",
        "reference_source": "exam_database.ref_text",
        "optimized_rubric_used": False,
        "teacher_label_in_prompt": False,
        "include_empty": include_empty,
        "selected_response_type_counts": selected_type_counts,
        "formal_response_type_counts": formal_type_counts,
        "limit_per_task": limit_per_task,
        "task_files": task_manifest,
        "input_sha256": {
            **{name: sha256_file(path) for name, path in required.items()},
            "selection_splits": sha256_files(
                split_paths,
                relative_to=prepared_dir,
            ),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--selection-protocol",
        choices=("outer-split", "per-question-test"),
        default="outer-split",
    )
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = prepare(
        args.prepared_dir,
        args.output_dir,
        split=args.split,
        selection_protocol=args.selection_protocol,
        include_empty=args.include_empty,
        limit_per_task=args.limit_per_task,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
