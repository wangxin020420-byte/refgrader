"""Read-only cross-machine inventory of specific scoring inputs.

Teacher-score files, audit decisions, caches and historical outputs are not
opened. This proves byte/text identity for enumerated files, not that student
images or text lack teacher markings or that splits are label independent.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


FILE_GROUPS = {
    "question_database": ("exam_database.json", "text"),
    "answer_metadata": ("answer_metadata.jsonl", "text"),
}
TREE_GROUPS = {
    "initial_rubrics": ("rubrics/initial", {".json"}, "text"),
    "question_splits": ("splits/by_question", {".json"}, "text"),
    "student_images": ("student_images", {".png", ".jpg", ".jpeg"}, "binary"),
    "reference_images": ("reference_images", {".png", ".jpg", ".jpeg"}, "binary"),
}
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _hash_file(path, mode):
    if path.is_symlink():
        raise ValueError(f"Symlink input is not supported: {path}")
    if mode == "text":
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        first = stream.read(1024 * 1024)
        if first.startswith(LFS_POINTER_PREFIX):
            raise ValueError(f"Unmaterialized Git LFS pointer: {path}")
        digest.update(first)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_inventory(root):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Prepared data root must be a directory")
    groups = {}
    for name, (relative, mode) in FILE_GROUPS.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
        groups[name] = {"count": 1, "sha256": _hash_file(path, mode)}
    for name, (relative, suffixes, mode) in TREE_GROUPS.items():
        directory = root / relative
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Missing or symlinked input directory: {directory}")
        paths = []
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Symlink input is not supported: {path}")
            if path.is_file() and path.suffix.lower() in suffixes:
                paths.append(path)
        if not paths:
            raise ValueError(f"No input files found under {directory}")
        digest = hashlib.sha256()
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
            relative_path = path.relative_to(root).as_posix()
            file_hash = _hash_file(path, mode)
            digest.update(json.dumps([relative_path, file_hash], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
        groups[name] = {"count": len(paths), "sha256": digest.hexdigest()}
    return {
        "schema": "label_free_input_inventory_v1",
        "text_normalization": "UTF-8, strip BOM and normalize CRLF to LF",
        "groups": groups,
        "teacher_scores_included": False,
        "image_annotation_audited": False,
        "metadata_content_audited": False,
        "split_provenance_audited": False,
        "caches_or_run_results_included": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default="data/csbench")
    args = parser.parse_args(argv)
    try:
        report = build_inventory(args.prepared_dir)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Inventory failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
