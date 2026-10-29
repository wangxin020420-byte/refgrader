"""Verify candidate question snapshot contents and hashes, without labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def verify(directory):
    root = Path(directory).resolve(strict=True)
    raw = (root / "snapshot.json").read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    snapshot = json.loads(raw)
    if snapshot.get("schema") != "label_free_question_candidate_v1" or snapshot.get("approved_for_formal_experiment") is not False:
        raise ValueError("Invalid or unexpectedly approved snapshot")
    questions = snapshot.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Missing questions")
    ids = [q["question_id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ID")
    images = {}
    for q in questions:
        references = list(q["question_images"]) + [item["reference_image"] for item in q["rubric"] if item["reference_image"]]
        for reference in references:
            path = (root / reference["path"]).resolve(strict=True)
            if not path.is_relative_to(root / "images") or not path.is_file():
                raise ValueError("Unsafe image path")
            data = path.read_bytes()
            if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError("Snapshot image is an unmaterialized Git LFS pointer")
            digest = hashlib.sha256(data).hexdigest()
            if digest != reference["sha256"]:
                raise ValueError("Snapshot image hash mismatch")
            images[reference["path"]] = digest
    actual = {p.relative_to(root).as_posix() for p in (root / "images").iterdir() if p.is_file()}
    if set(images) != actual:
        raise ValueError("Snapshot contains missing or unreferenced image files")
    return {"schema": "label_free_question_candidate_verification_v1",
            "source_revision": snapshot["source_revision"], "questions": len(questions), "images": len(images),
            "snapshot_sha256_normalized": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "flagged_questions": snapshot["reference_with_teacher_word_questions"],
            "approved_for_formal_experiment": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.snapshot_dir), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Snapshot verification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
