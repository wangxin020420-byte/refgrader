"""Prepare a lightweight RefGrader view of the external CSBench dataset.

The source dataset stays untouched. Student images are exposed per question by
hard links (default), symbolic links, or copies, while JSON metadata is
converted to the format expected by RefGrader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ANSWER_FILES = (
    "answer.jsonl",
    "answer_CPL.jsonl",
    "answer_DM.jsonl",
    "answer_ISC.jsonl",
    "answer_ML.jsonl",
    "answer_POC.jsonl",
    "answer_POC_25.jsonl",
    "answer_POC_26.jsonl",
)
SUBJECT_NAMES = {
    "CO": "计算机组成原理",
    "CPL": "C语言/程序设计",
    "DM": "离散数学",
    "ISC": "计算机导论",
    "ML": "数字逻辑",
    "POC": "编译原理",
    "POC_24": "编译原理",
    "POC_25": "编译原理",
    "POC_26": "编译原理",
}
VISUAL_PLACEHOLDERS = (
    "如图所示",
    "见图",
    "图如下",
    "如下图",
    "答案见图",
    "状态图略",
    "如下表",
)
VISUAL_KEYWORDS = (
    "画出",
    "绘制",
    "图",
    "表格",
    "连线",
    "状态转移",
    "状态表",
    "哈斯",
    "语法树",
    "项目集",
    "拓扑",
    "波形",
    "时序",
)
TABLE_OR_GRID_KEYWORDS = (
    "table",
    "grid",
    "matrix",
    "truth table",
    "cache",
    "tag",
    "index",
    "offset",
    "field",
    "row",
    "column",
    "cell",
    "表",
    "表格",
    "矩阵",
    "真值表",
    "字段",
    "地址划分",
    "位划分",
    "组号",
    "组索引",
    "块内地址",
    "行",
    "列",
)

TOPOLOGY_OR_RELATION_KEYWORDS = (
    "diagram_relation",
    "topology",
    "graph",
    "tree",
    "syntax tree",
    "parse tree",
    "state diagram",
    "flowchart",
    "timing",
    "sequence",
    "staircase",
    "hasse",
    "circuit",
    "edge",
    "edges",
    "arrow",
    "arrows",
    "node",
    "nodes",
    "path",
    "拓扑",
    "关系图",
    "结构图",
    "语法树",
    "分析树",
    "状态图",
    "流程图",
    "时序图",
    "顺序图",
    "阶梯图",
    "哈斯图",
    "电路图",
    "连线",
    "箭头",
    "路径",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CSBench for RefGrader")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default="data/csbench")
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
    )
    parser.add_argument("--exclude-questions", nargs="*", default=["OS_1", "OS_2"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--split-seed", default="csbench-v1")
    parser.add_argument("--calibration-ratio", type=float, default=0.10)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--minimum-calibration-size", type=int, default=5)
    return parser.parse_args()


def load_question_definitions(dataset_root: Path) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    for path in sorted((dataset_root / "question").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            flattened = []
            for value in data.values():
                flattened.extend(value if isinstance(value, list) else [value])
            data = flattened
        if not isinstance(data, list):
            continue
        for item in data:
            if isinstance(item, dict) and item.get("question_id"):
                questions[str(item["question_id"])] = item
    return questions


def iter_answers(dataset_root: Path):
    for filename in DEFAULT_ANSWER_FILES:
        path = dataset_root / "answer" / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing CSBench answer file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_source_file"] = filename
                record["_source_line"] = line_number
                yield record


def resolve_cleaned_image(dataset_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.parts and relative.parts[0].lower() == "images":
        cleaned = dataset_root / "images" / "cleaned" / Path(*relative.parts[1:])
        if cleaned.exists():
            return cleaned.resolve()
    original = (dataset_root / relative).resolve()
    if original.exists():
        return original
    raise FileNotFoundError(f"Student image not found: {relative_path}")


def absolute_dataset_path(dataset_root: Path, relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    path = (dataset_root / relative_path).resolve()
    return str(path) if path.exists() else None


def is_placeholder_text(text: str) -> bool:
    compact = "".join(str(text or "").split())
    return any(marker in compact for marker in VISUAL_PLACEHOLDERS)


def contains_marker(text: str, marker: str) -> bool:
    marker = str(marker or "").strip()
    if not marker:
        return False
    if marker.isascii():
        return bool(
            re.search(
                rf"\b{re.escape(marker.lower())}\b",
                str(text or "").lower(),
            )
        )
    return marker in str(text or "")


def has_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(contains_marker(text, marker) for marker in markers)


def classify_rubric_item(item: dict[str, Any]) -> tuple[str, str]:
    description = str(item.get("description", ""))
    standard_text = str(item.get("standard_answer_text", ""))
    has_standard_image = bool(item.get("standard_answer_image"))
    item_text = " ".join(
        str(item.get(key, ""))
        for key in (
            "description",
            "standard_answer_text",
            "answer_type",
            "evidence_source",
        )
    )
    item_text_lower = item_text.lower()
    looks_table_or_grid = has_any_marker(item_text_lower, TABLE_OR_GRID_KEYWORDS)
    looks_topology_or_relation = has_any_marker(
        item_text_lower, TOPOLOGY_OR_RELATION_KEYWORDS
    )
    looks_visual = (
        has_standard_image
        or looks_table_or_grid
        or looks_topology_or_relation
        or any(keyword in description for keyword in VISUAL_KEYWORDS)
    )
    substantive_text = bool(
        standard_text.strip()
        and not is_placeholder_text(standard_text)
        and standard_text.strip() not in {"详见标准答案图片", "见标准答案图片"}
    )
    if looks_visual and looks_table_or_grid and not looks_topology_or_relation:
        return "table", "text_and_ocr" if substantive_text else "ocr_table"
    if looks_visual and looks_topology_or_relation:
        return "diagram_relation", "text_and_diagram" if substantive_text else "diagram"
    if looks_visual and substantive_text:
        return "diagram_relation", "text_and_diagram"
    if looks_visual:
        return "diagram_relation", "diagram"
    return "text", "transcription"


def convert_rubric(
    question: dict[str, Any],
    dataset_root: Path,
) -> list[dict[str, Any]]:
    converted = []
    for index, item in enumerate(question.get("grading_rubric") or [], start=1):
        answer_type, evidence_source = classify_rubric_item(item)
        standard_image = absolute_dataset_path(
            dataset_root, item.get("standard_answer_image")
        )
        converted.append(
            {
                "id": f"step_{item.get('step_id', index)}",
                "item": str(item.get("description", "")),
                "points": float(item.get("score", 0)),
                "answer_type": answer_type,
                "role": "final",
                "canonicalization": answer_type,
                "evidence_source": evidence_source,
                "standard_answer_text": str(
                    item.get("standard_answer_text", "")
                ),
                "standard_answer_image": standard_image,
                "source_text": str(item.get("description", "")),
                "parent_official_item": str(item.get("description", "")),
                "metadata_source": "csbench",
                "metadata_hard_enabled": True,
                "metadata_confidence": 1.0,
            }
        )
    return converted


def make_reference_text(rubric: list[dict[str, Any]]) -> str:
    lines = []
    for item in rubric:
        expected = item.get("standard_answer_text") or "见标准答案图片"
        lines.append(f"{item['id']}: {expected}")
    return "\n".join(lines)


def make_official_rubric(rubric: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{item['item']}；参考：{item.get('standard_answer_text') or '见图'}"
        f" ---------------{item['points']:g}分"
        for item in rubric
    )


def link_file(source: Path, destination: Path, mode: str, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not force:
            try:
                if destination.samefile(source):
                    return
            except OSError:
                pass
            raise FileExistsError(f"Destination already exists: {destination}")
        destination.unlink()
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source)
    else:
        shutil.copy2(source, destination)


def rubric_group_for(question_id: str, question: dict[str, Any]) -> str:
    """Return a stable, filesystem-safe rubric group."""
    prefix = str(question_id).split("_", 1)[0].strip()
    return prefix or str(question.get("subject") or "unknown").strip() or "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def split_answer_ids(
    answer_ids: list[str],
    seed: str,
    calibration_ratio: float,
    validation_ratio: float,
    minimum_calibration_size: int,
) -> dict[str, list[str]]:
    """Create deterministic, non-overlapping per-question data partitions."""
    if calibration_ratio < 0 or validation_ratio < 0:
        raise ValueError("Split ratios must be non-negative.")
    if calibration_ratio + validation_ratio >= 1:
        raise ValueError("calibration-ratio + validation-ratio must be < 1.")

    ordered = sorted(
        answer_ids,
        key=lambda answer_id: hashlib.sha256(
            f"{seed}:{answer_id}".encode("utf-8")
        ).hexdigest(),
    )
    total = len(ordered)
    if not total:
        return {"calibration": [], "validation": [], "test": []}

    calibration_count = max(
        min(minimum_calibration_size, total),
        int(round(total * calibration_ratio)),
    )
    remaining = total - calibration_count
    validation_count = min(
        remaining,
        max(1 if remaining > 1 else 0, int(round(total * validation_ratio))),
    )
    return {
        "calibration": ordered[:calibration_count],
        "validation": ordered[
            calibration_count : calibration_count + validation_count
        ],
        "test": ordered[calibration_count + validation_count :],
    }


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    excluded = set(args.exclude_questions or [])

    if not (dataset_root / "answer").is_dir():
        raise FileNotFoundError(f"Invalid CSBench root: {dataset_root}")

    questions = load_question_definitions(dataset_root)
    rubrics_dir = output_dir / "rubrics"
    source_rubrics_dir = rubrics_dir / "source"
    initial_rubrics_dir = rubrics_dir / "initial"
    optimized_rubrics_dir = rubrics_dir / "optimized"
    rubric_manifests_dir = rubrics_dir / "manifests"
    student_root = output_dir / "student_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rubrics_dir.mkdir(parents=True, exist_ok=True)
    initial_rubrics_dir.mkdir(parents=True, exist_ok=True)
    optimized_rubrics_dir.mkdir(parents=True, exist_ok=True)
    rubric_manifests_dir.mkdir(parents=True, exist_ok=True)
    student_root.mkdir(parents=True, exist_ok=True)

    teacher_scores: dict[str, dict[str, float]] = {}
    metadata_records = []
    question_counts: Counter[str] = Counter()
    missing_question_counts: Counter[str] = Counter()
    seen_answer_ids = set()

    for answer in iter_answers(dataset_root):
        answer_id = str(answer["answer_id"])
        question_id = str(answer["question_id"])
        if answer_id in seen_answer_ids:
            raise ValueError(f"Duplicate answer_id in selected files: {answer_id}")
        seen_answer_ids.add(answer_id)
        if question_id in excluded:
            continue
        if question_id not in questions:
            missing_question_counts[question_id] += 1
            continue

        image_paths = answer.get("student_input", {}).get("image_paths") or []
        if len(image_paths) != 1:
            raise ValueError(
                f"{answer_id} must have exactly one image, got {len(image_paths)}"
            )
        source_image = resolve_cleaned_image(dataset_root, image_paths[0])
        destination = student_root / question_id / f"{answer_id}{source_image.suffix.lower()}"
        link_file(source_image, destination, args.link_mode, args.force)

        raw_text = str(answer.get("student_input", {}).get("raw_text", ""))
        metadata_records.append(
            {
                "answer_id": answer_id,
                "question_id": question_id,
                "subject": questions[question_id].get("subject"),
                "subject_name": SUBJECT_NAMES.get(
                    str(questions[question_id].get("subject", "")),
                    str(questions[question_id].get("subject", "")),
                ),
                "raw_text": raw_text,
                "isimagine": bool(answer.get("isimagine")),
                "visual_placeholder_detected": is_placeholder_text(raw_text),
                "student_image": str(destination.resolve()),
                "source_image": str(source_image),
                "actual_score": float(answer.get("actual_score", 0)),
                "source_file": answer["_source_file"],
                "source_line": answer["_source_line"],
            }
        )
        teacher_scores[answer_id] = {
            question_id: float(answer.get("actual_score", 0))
        }
        question_counts[question_id] += 1

    metadata_by_question: dict[str, list[str]] = {}
    for record in metadata_records:
        metadata_by_question.setdefault(record["question_id"], []).append(
            record["answer_id"]
        )

    per_question_splits: dict[str, dict[str, list[str]]] = {}
    splits_by_question_dir = output_dir / "splits" / "by_question"
    for question_id, answer_ids in sorted(metadata_by_question.items()):
        split_payload = split_answer_ids(
            answer_ids,
            seed=f"{args.split_seed}:{question_id}",
            calibration_ratio=args.calibration_ratio,
            validation_ratio=args.validation_ratio,
            minimum_calibration_size=args.minimum_calibration_size,
        )
        per_question_splits[question_id] = split_payload
        write_json(
            splits_by_question_dir / f"{question_id}.json",
            {
                "question_id": question_id,
                "split_seed": args.split_seed,
                **split_payload,
            },
        )

    exam_database = []
    for question_id in sorted(question_counts):
        question = questions[question_id]
        rubric = convert_rubric(question, dataset_root)
        rubric_group = rubric_group_for(question_id, question)
        source_rubric_path = (
            source_rubrics_dir / rubric_group / f"{question_id}.json"
        )
        initial_rubric_path = (
            initial_rubrics_dir
            / rubric_group
            / f"{question_id}_rubric_standard.json"
        )
        optimized_rubric_path = (
            optimized_rubrics_dir
            / rubric_group
            / f"{question_id}_rubric_standard.json"
        )
        write_json(
            source_rubric_path,
            {
                "question_id": question_id,
                "subject": question.get("subject"),
                "subject_name": SUBJECT_NAMES.get(
                    str(question.get("subject", "")),
                    str(question.get("subject", "")),
                ),
                "max_score": question.get("max_score"),
                "grading_rubric": question.get("grading_rubric") or [],
                "metadata_source": "csbench",
            },
        )
        write_json(initial_rubric_path, rubric)
        question_images = question.get("content", {}).get("image_paths") or []
        standard_images = [
            item.get("standard_answer_image")
            for item in rubric
            if item.get("standard_answer_image")
        ]
        exam_database.append(
            {
                "question_id": question_id,
                "subject": question.get("subject"),
                "rubric_group": rubric_group,
                "source_rubric_path": str(source_rubric_path.resolve()),
                "initial_rubric_path": str(initial_rubric_path.resolve()),
                "optimized_rubric_path": str(optimized_rubric_path.resolve()),
                "rubric_split_path": str(
                    (splits_by_question_dir / f"{question_id}.json").resolve()
                ),
                "rubric_calibration_ids": per_question_splits[question_id][
                    "calibration"
                ],
                "exam_source": question.get("exam_source"),
                "total_score": float(question.get("max_score", 0)),
                "question_text": str(question.get("content", {}).get("text", "")),
                "question_image": absolute_dataset_path(
                    dataset_root, question_images[0] if question_images else None
                ),
                "ref_text": make_reference_text(rubric),
                "official_rubric": make_official_rubric(rubric),
                "ref_image": standard_images[0] if standard_images else None,
                "student_images_dir": str(
                    (student_root / question_id).resolve()
                ),
                "requires_visual_evidence": any(
                    str(item.get("evidence_source", "")) in {
                        "diagram",
                        "text_and_diagram",
                        "ocr_table",
                        "text_and_ocr",
                    }
                    or str(item.get("answer_type", "")) in {
                        "diagram_relation",
                        "table",
                    }
                    for item in rubric
                ),
                "sample_count": question_counts[question_id],
            }
        )

    split_questions = {"train": [], "validation": [], "test": []}
    for question in exam_database:
        question_id = question["question_id"]
        digest = hashlib.sha256(
            f"{args.split_seed}:{question_id}".encode("utf-8")
        ).digest()
        bucket = int.from_bytes(digest[:4], "big") / 2**32
        split_name = (
            "train" if bucket < 0.70
            else "validation" if bucket < 0.85
            else "test"
        )
        split_questions[split_name].append(question_id)
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, question_ids in split_questions.items():
        split_payload = {
            "split": split_name,
            "split_seed": args.split_seed,
            "question_ids": sorted(question_ids),
            "answer_ids": sorted(
                answer_id
                for question_id in question_ids
                for answer_id in metadata_by_question.get(question_id, [])
            ),
        }
        (splits_dir / f"{split_name}.json").write_text(
            json.dumps(split_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (output_dir / "exam_database.json").write_text(
        json.dumps(exam_database, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "teacher_scores.json").write_text(
        json.dumps(teacher_scores, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "answer_metadata.jsonl").open("w", encoding="utf-8") as handle:
        for record in metadata_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "link_mode": args.link_mode,
        "answer_files": list(DEFAULT_ANSWER_FILES),
        "excluded_questions": sorted(excluded),
        "prepared_questions": len(exam_database),
        "prepared_answers": len(metadata_records),
        "question_counts": dict(sorted(question_counts.items())),
        "missing_question_counts": dict(sorted(missing_question_counts.items())),
        "split_seed": args.split_seed,
        "calibration_ratio": args.calibration_ratio,
        "validation_ratio": args.validation_ratio,
        "minimum_calibration_size": args.minimum_calibration_size,
        "split_question_counts": {
            key: len(value) for key, value in split_questions.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
