"""Build a portable, non-overwriting candidate question snapshot.

Reads only upstream question JSON and the question/reference images explicitly
referenced there. Never reads student answers or teacher-score files. A snapshot
preserves source content, including potentially teacher-informed reference text;
it is explicitly NOT certified label-free or approved for formal experiments.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _read_image(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Invalid image path")
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Unsafe or unsupported source image")
    data = path.read_bytes()
    if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError("Source image is an unmaterialized Git LFS pointer")
    digest = hashlib.sha256(data).hexdigest()
    return path, data, digest, "images/" + digest + path.suffix.lower()


def build(dataset_root, destination, source_revision):
    root = Path(dataset_root).resolve(strict=True)
    dest = Path(destination).absolute()
    if dest.exists():
        raise FileExistsError("Snapshot destination already exists")
    if not dest.parent.is_dir():
        raise ValueError("Snapshot parent must exist")
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("Full source Git revision required")
    files = sorted((root / "question").glob("*.json"))
    if not files:
        raise ValueError("No upstream question files")
    questions, image_bytes, source_hashes, source_paths = [], {}, {}, []
    seen = set()
    for path in files:
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        source_hashes[path.relative_to(root).as_posix()] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_paths.append(path.relative_to(root).as_posix())
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError("Question file must be a list")
        for q in rows:
            qid = q.get("question_id")
            if not isinstance(qid, str) or not re.fullmatch(r"[A-Za-z0-9_]+", qid) or qid in seen:
                raise ValueError("Invalid or duplicate question ID")
            seen.add(qid)
            content = q.get("content")
            if not isinstance(content, dict) or not isinstance(content.get("text"), str):
                raise ValueError("Invalid question content")
            def image_entry(value):
                image_path, data, digest, relative = _read_image(root, value)
                image_bytes[relative] = data
                source_paths.append(image_path.relative_to(root).as_posix())
                return {"path": relative, "sha256": digest, "source_path": str(value).replace("\\", "/")}
            question_images = [image_entry(v) for v in (content.get("image_paths") or [])]
            raw_rubric = q.get("grading_rubric")
            if not isinstance(raw_rubric, list) or not raw_rubric:
                raise ValueError("Missing upstream rubric")
            items, item_ids = [], set()
            for index, item in enumerate(raw_rubric, 1):
                ident = str(item.get("step_id", index))
                if not ident or ident in item_ids or isinstance(item.get("score"), bool):
                    raise ValueError("Invalid upstream rubric item")
                item_ids.add(ident)
                points = float(item["score"])
                if not math.isfinite(points) or points < 0:
                    raise ValueError("Invalid upstream rubric points")
                ref = item.get("standard_answer_text") or ""
                if not isinstance(ref, str) or not isinstance(item.get("description"), str):
                    raise ValueError("Invalid upstream rubric text")
                image = item.get("standard_answer_image")
                items.append({"id": "step_" + ident, "description": item["description"], "points": points,
                              "reference_text": ref, "reference_image": image_entry(image) if image else None})
            maximum = float(q["max_score"])
            if not math.isfinite(maximum) or maximum <= 0 or abs(sum(i["points"] for i in items) - maximum) > 1e-6:
                raise ValueError("Rubric total differs from max score")
            questions.append({"question_id": qid, "question_text": content["text"], "max_score": maximum,
                              "question_images": question_images, "rubric": items})
    teacher_words = re.compile(r"教师评分|老师评分|teacher\s*score", re.IGNORECASE)
    flagged = sorted(q["question_id"] for q in questions if any(teacher_words.search(i["reference_text"]) for i in q["rubric"]))
    snapshot = {"schema": "label_free_question_candidate_v1", "source_revision": source_revision,
                "source_question_hashes": source_hashes, "questions": sorted(questions, key=lambda q: q["question_id"]),
                "reference_with_teacher_word_questions": flagged,
                "teacher_scores_opened": False, "source_provenance_audited": False,
                "approved_for_formal_experiment": False}
    actual_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual_revision != source_revision:
        raise ValueError("Source revision changed during snapshot construction")
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *sorted(set(source_paths))],
                   cwd=root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *sorted(set(source_paths))], cwd=root, text=True)
    if dirty.strip():
        raise ValueError("Snapshot source files have uncommitted changes")
    with tempfile.TemporaryDirectory(prefix=".candidate-", dir=dest.parent) as tmp:
        temp = Path(tmp)
        (temp / "images").mkdir()
        for relative, data in image_bytes.items():
            (temp / relative).write_bytes(data)
        (temp / "snapshot.json").write_bytes(_json_bytes(snapshot))
        # A concurrent destination creation fails instead of deliberately replacing it.
        if dest.exists():
            raise FileExistsError("Snapshot destination appeared during construction")
        os.rename(temp, dest)
    return {"questions": len(questions), "images": len(image_bytes), "flagged_questions": flagged,
            "source_revision": source_revision, "approved_for_formal_experiment": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        root = Path(args.dataset_root).resolve(strict=True)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        # Inspect only question and reference images, never the answers tree.
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "question", "images"], cwd=root, text=True)
        if dirty.strip():
            raise ValueError("Upstream question/images have uncommitted changes")
        print(json.dumps(build(root, args.output, revision), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"Candidate snapshot failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
