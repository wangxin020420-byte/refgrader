"""Make a prepared CSBench view self-contained inside refgrader-main."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default="data/csbench")
    parser.add_argument("--source-root")
    return parser.parse_args()


def project_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_existing_path(raw: str | None, source_root: Path | None) -> Path | None:
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()
    if source_root:
        candidate = source_root / raw
        if candidate.is_file():
            return candidate.resolve()
    return None


def copy_reference(
    source: Path | None,
    destination_dir: Path,
    question_id: str,
) -> str | None:
    if not source:
        return None
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{question_id}{source.suffix.lower()}"
    if source.resolve() == destination.resolve():
        return project_path(destination)
    shutil.copy2(source, destination)
    return project_path(destination)


def hash_tree(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    prepared = (PROJECT_ROOT / Path(args.prepared_dir)).resolve()
    source_root = Path(args.source_root).resolve() if args.source_root else None
    database_path = prepared / "exam_database.json"
    metadata_path = prepared / "answer_metadata.jsonl"
    manifest_path = prepared / "manifest.json"

    questions = read_json(database_path)
    question_refs = prepared / "reference_images" / "questions"
    standard_refs = prepared / "reference_images" / "standard_answers"

    for question in questions:
        question_id = str(question["question_id"])
        question["source_rubric_path"] = project_path(
            prepared / "rubrics" / "source" / question["rubric_group"] / f"{question_id}.json"
        )
        question["initial_rubric_path"] = project_path(
            prepared
            / "rubrics"
            / "initial"
            / question["rubric_group"]
            / f"{question_id}_rubric_standard.json"
        )
        question["optimized_rubric_path"] = project_path(
            prepared
            / "rubrics"
            / "optimized"
            / question["rubric_group"]
            / f"{question_id}_rubric_standard.json"
        )
        question["rubric_split_path"] = project_path(
            prepared / "splits" / "by_question" / f"{question_id}.json"
        )
        question["student_images_dir"] = project_path(
            prepared / "student_images" / question_id
        )

        question_source = resolve_existing_path(question.get("question_image"), source_root)
        question["question_image"] = copy_reference(
            question_source, question_refs, question_id
        )
        standard_source = resolve_existing_path(question.get("ref_image"), source_root)
        question["ref_image"] = copy_reference(
            standard_source, standard_refs, question_id
        )

        initial_path = Path(question["initial_rubric_path"])
        initial = read_json(PROJECT_ROOT / initial_path)
        changed = False
        for item in initial:
            source = resolve_existing_path(item.get("standard_answer_image"), source_root)
            if source:
                item["standard_answer_image"] = copy_reference(
                    source, standard_refs, question_id
                )
                changed = True
        if changed:
            write_json(PROJECT_ROOT / initial_path, initial)

    write_json(database_path, questions)

    portable_records = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            source = Path(str(record["student_image"]))
            suffix = source.suffix.lower()
            embedded_image = (
                prepared
                / "student_images"
                / str(record["question_id"])
                / f"{record['answer_id']}{suffix}"
            )
            if not embedded_image.is_file():
                raise FileNotFoundError(f"Missing embedded student image: {embedded_image}")
            portable = project_path(embedded_image)
            record["student_image"] = portable
            record["source_image"] = portable
            portable_records.append(record)
    metadata_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in portable_records
        ),
        encoding="utf-8",
    )

    manifest = read_json(manifest_path)
    manifest["dataset_root"] = project_path(prepared)
    manifest["output_dir"] = project_path(prepared)
    manifest["link_mode"] = "embedded_copy"
    manifest["embedded"] = True
    write_json(manifest_path, manifest)

    # Remove the obsolete flat rubric copies. The active layout is
    # rubrics/{source,initial,optimized,manifests}/<group>/.
    for legacy_rubric in (prepared / "rubrics").glob("*_rubric_standard.json"):
        legacy_rubric.unlink()

    tracked_files = [
        path
        for path in prepared.rglob("*")
        if path.is_file()
        and path.name != "embedded_manifest.json"
        and "rubrics/optimized" not in path.as_posix()
        and "rubrics/manifests" not in path.as_posix()
    ]
    embedded_manifest = {
        "schema_version": 1,
        "prepared_dir": project_path(prepared),
        "question_count": len(questions),
        "answer_count": len(portable_records),
        "student_image_count": len(list((prepared / "student_images").rglob("*.*"))),
        "reference_image_count": len(list((prepared / "reference_images").rglob("*.*"))),
        "snapshot_sha256": hash_tree(tracked_files),
    }
    write_json(prepared / "embedded_manifest.json", embedded_manifest)
    print(json.dumps(embedded_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
