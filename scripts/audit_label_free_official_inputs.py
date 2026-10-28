"""Compare upstream question/rubric structure without reading teacher scores.

This is a provenance diagnostic, not proof that upstream inputs are unbiased
or that the prepared data can be used in a formal label-free experiment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _text(value):
    return str(value or "").replace("\r\n", "\n").strip()


def _points(value):
    if isinstance(value, bool):
        raise ValueError("Invalid rubric score")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("Invalid rubric score")
    return number


def _image_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(dataset_root, prepared_root):
    dataset_root = Path(dataset_root)
    prepared_root = Path(prepared_root)
    files = sorted((dataset_root / "question").glob("*.json"))
    if not files:
        raise ValueError("No upstream question files found")
    official = {}
    for file in files:
        rows = _json(file)
        if not isinstance(rows, list):
            raise ValueError("Upstream question file must contain a list")
        for row in rows:
            qid = row.get("question_id")
            if not isinstance(qid, str) or not re.fullmatch(r"[A-Za-z0-9_]+", qid) or qid in official:
                raise ValueError("Invalid or duplicate upstream question ID")
            official[qid] = row
    prepared = _json(prepared_root / "exam_database.json")
    if not isinstance(prepared, list):
        raise ValueError("Prepared question database must be a list")
    ids = [row.get("question_id") for row in prepared]
    if len(ids) != len(set(ids)) or any(q not in official for q in ids):
        raise ValueError("Prepared IDs are duplicated or absent from upstream")
    differences = []
    multiple_reference_images = []
    teacher_word_reference_questions = []
    reference_image_mismatches = []
    for row in prepared:
        qid = row["question_id"]
        group = row.get("rubric_group") or qid.split("_", 1)[0]
        if not isinstance(group, str) or not re.fullmatch(r"[A-Za-z0-9_]+", group):
            raise ValueError("Unsafe rubric group")
        initial = _json(prepared_root / "rubrics" / "initial" / group / f"{qid}_rubric_standard.json")
        original = official[qid]
        upstream = original.get("grading_rubric")
        if not isinstance(upstream, list) or not isinstance(initial, list):
            raise ValueError("Invalid rubric")
        official_core = [{"id": f"step_{item.get('step_id', index)}", "item": _text(item.get("description")),
                          "points": _points(item.get("score")), "reference": _text(item.get("standard_answer_text"))}
                         for index, item in enumerate(upstream, 1)]
        prepared_core = [{"id": str(item.get("id") or ""), "item": _text(item.get("item")),
                          "points": _points(item.get("points")), "reference": _text(item.get("standard_answer_text"))}
                         for item in initial]
        if len({item["id"] for item in official_core}) != len(official_core):
            raise ValueError("Duplicate upstream rubric item ID")
        if abs(sum(x["points"] for x in official_core) - _points(original.get("max_score"))) > 1e-6:
            raise ValueError("Upstream rubric does not sum to maximum score")
        if abs(sum(x["points"] for x in prepared_core) - _points(row.get("total_score"))) > 1e-6:
            raise ValueError("Prepared rubric does not sum to maximum score")
        images = sorted({str(item["standard_answer_image"]) for item in upstream if item.get("standard_answer_image")})
        if len(images) > 1:
            multiple_reference_images.append({"question_id": qid, "upstream_reference_image_count": len(images)})
        if images:
            prepared_images = [p for p in (prepared_root / "reference_images" / "standard_answers").glob(qid + ".*")
                               if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            prepared_hashes = {_image_hash(p) for p in prepared_images}
            unmatched = 0
            for image in images:
                source_path = (dataset_root / image).resolve()
                if not source_path.is_relative_to(dataset_root.resolve()) or not source_path.is_file():
                    raise ValueError("Missing or unsafe upstream reference image")
                if _image_hash(source_path) not in prepared_hashes:
                    unmatched += 1
            if unmatched:
                reference_image_mismatches.append({"question_id": qid, "unmatched_source_image_count": unmatched,
                                                   "prepared_reference_image_count": len(prepared_images)})
        teacher_word = re.compile(r"教师评分|老师评分|teacher\s*score", re.IGNORECASE)
        if any(teacher_word.search(item["reference"]) for item in official_core) or teacher_word.search(_text(row.get("ref_text"))):
            teacher_word_reference_questions.append(qid)
        comparison = {
            "question_id": qid,
            "question_text_changed": _text((original.get("content") or {}).get("text")) != _text(row.get("question_text")),
            "rubric_core_changed": official_core != prepared_core,
            "reference_text_changed": _text("\n".join(
                f"{item['id']}: {item['reference'] or '见标准答案图片'}" for item in official_core
            )) != _text(row.get("ref_text")),
            "official_item_count": len(official_core),
            "prepared_item_count": len(prepared_core),
            "official_reference_image_count": len(images),
        }
        if comparison["question_text_changed"] or comparison["rubric_core_changed"] or comparison["reference_text_changed"]:
            differences.append(comparison)
    return {
        "schema": "label_free_official_input_audit_v1",
        "upstream_question_files": len(files), "selected_questions": len(ids),
        "question_text_changed_count": sum(x["question_text_changed"] for x in differences),
        "rubric_core_changed_count": sum(x["rubric_core_changed"] for x in differences),
        "reference_text_changed_count": sum(x["reference_text_changed"] for x in differences),
        "changed_questions": differences,
        "multiple_reference_images": multiple_reference_images,
        "reference_image_mismatches": reference_image_mismatches,
        "reference_with_teacher_word_questions": sorted(teacher_word_reference_questions),
        "teacher_score_file_opened": False,
        "notes": ["Keyword flag is not proof of score leakage; absence is not proof of independence.",
                  "Comparison only; text provenance is not certified. Image hashes establish byte identity only.",
                  "No model calls; do not use these diagnostics to select samples by score."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--prepared-dir", default="data/csbench")
    args = parser.parse_args(argv)
    try:
        result = audit(args.dataset_root, args.prepared_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Official input audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
