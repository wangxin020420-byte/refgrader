"""Freeze the existing within-question CSBench splits for SCALE, offline."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

EXPECTED = {
    "exam_database.json": "74d9c7fd16e06b4cec5b3d2bcdcbf6cc6217b94ae676c7d42f5c7f81b2bb2ea4",
    "answer_metadata.jsonl": "af3a20bc8558e3762bce5d5b64e1c9c9115e0b534f5c4b63a9e2b5dd21e60acd",
    "teacher_scores.json": "bc3090de26a99f51d74b1fea7335418ced13de3fe95bc3bb9ba63ee610ccc504",
}


def digest(b):
    return hashlib.sha256(b).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonlines(rows):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in rows).encode("utf-8")


def partition(split, expected):
    result = {}
    for old, new in (("calibration", "train"), ("validation", "validation"), ("test", "test")):
        values = split[old]
        if not isinstance(values, list):
            raise ValueError("Split must contain ID lists")
        for aid in values:
            if aid in result:
                raise ValueError("Repeated or overlapping split ID: " + aid)
            result[aid] = new
    if set(result) != set(expected):
        raise ValueError("Split does not exactly cover question answers")
    return result


def build(project, out):
    project, out = project.resolve(), out.resolve()
    source = project / "data/csbench"
    if out == project or out.is_relative_to(project):
        raise ValueError("Output must be outside the source project")
    hashes = {}
    for name, expected in EXPECTED.items():
        hashes[name] = digest((source / name).read_bytes())
        if hashes[name] != expected:
            raise ValueError("Source snapshot differs: " + name)
    questions = read(source / "exam_database.json")
    metadata = [json.loads(l) for l in (source / "answer_metadata.jsonl").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    teachers = read(source / "teacher_scores.json")
    qm = {q["question_id"]: q for q in questions}
    am = {a["answer_id"]: a for a in metadata}
    if len(qm) != len(questions) or len(am) != len(metadata) or (len(qm), len(am)) != (45, 3411):
        raise ValueError("Unexpected counts or duplicate IDs")
    if set(teachers) != set(am) or any(a["question_id"] not in qm for a in metadata):
        raise ValueError("Label/answer/question identity mismatch")

    def local_path(value):
        if not value:
            return None
        p = Path(value)
        return str((p if p.is_absolute() else project / p).resolve())

    assignments = {}
    question_rows = []
    for qid, q in qm.items():
        p = Path(q["rubric_split_path"])
        p = p if p.is_absolute() else project / p
        s = read(p)
        if s.get("question_id") != qid:
            raise ValueError("Split question mismatch")
        assignments.update(partition(s, [aid for aid, a in am.items() if a["question_id"] == qid]))
        hashes["splits/by_question/" + qid + ".json"] = digest(p.read_bytes())
        question_rows.append({"question_id": qid, "question": q["question_text"],
            "references": [q["ref_text"]], "score_max_original": q["total_score"],
            "question_image": local_path(q.get("question_image")),
            "reference_image": local_path(q.get("ref_image")),
            "visual_context_review_required": bool(q.get("requires_visual_evidence"))})

    inputs = {s: [] for s in ("train", "validation", "test")}
    labels = {s: [] for s in inputs}
    review, blanks = [], Counter()
    for aid, a in am.items():
        qid = a["question_id"]
        q, split = qm[qid], assignments[aid]
        score, maximum = teachers[aid][qid], q["total_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= maximum or maximum <= 0:
            raise ValueError("Invalid original score: " + aid)
        text = a["raw_text"]
        if not isinstance(text, str):
            raise ValueError("Non-string answer: " + aid)
        blank = not text.strip()
        if blank:
            blanks[split] += 1
        reasons = []
        if not blank:
            if a.get("visual_placeholder_detected"):
                reasons.append("visual_placeholder")
            if q.get("requires_visual_evidence"):
                reasons.append("question_requires_visual_evidence")
        row = {"sample_id": aid, "question_id": qid, "question": q["question_text"],
               "references": [q["ref_text"]], "answer": text,
               "blank_passthrough": blank, "text_review_required": bool(reasons),
               "student_image": local_path(a.get("student_image"))}
        inputs[split].append(row)
        labels[split].append({"sample_id": aid, "question_id": qid,
            "score_original": score, "score_max_original": maximum,
            "score_0_10": 10.0 * score / maximum})
        if reasons:
            review.append({"sample_id": aid, "question_id": qid, "split": split,
                           "answer": text, "student_image": row["student_image"], "reasons": reasons})
    counts = {s: len(rows) for s, rows in inputs.items()}
    if counts != {"train": 345, "validation": 343, "test": 2723} or sum(blanks.values()) != 66:
        raise ValueError("Unexpected split or blank counts")
    payload = {s + ".inputs.jsonl": jsonlines(rows) for s, rows in inputs.items()}
    payload.update({s + ".labels.jsonl": jsonlines(rows) for s, rows in labels.items()})
    payload["questions.jsonl"] = jsonlines(question_rows)
    payload["nonblank_text_review_queue.jsonl"] = jsonlines(review)
    manifest = {"status": "CSBENCH_WITHIN_QUESTION_SPLIT_FROZEN_TEXT_PENDING",
        "dataset": "CSBench", "questions": 45, "counts": counts,
        "split_policy": "Existing per-question calibration->train, validation->validation, test->test; no reshuffling",
        "blank_policy": "66 empty-text answers preserved as blank per user instruction; original labels unchanged",
        "blank_counts": dict(blanks), "nonblank_review_records": len(review),
        "question_visual_review_records": sum(q["visual_context_review_required"] for q in question_rows),
        "review_records_without_image": sum(not r["student_image"] or not Path(r["student_image"]).is_file() for r in review),
        "score_policy": "Original gold retained; explicit 0-10 conversion = 10*score/original maximum; no clipping or rounding",
        "beta_scale_verified": False, "source_sha256": hashes,
        "builder_sha256": digest(Path(__file__).read_bytes()),
        "files_sha256": {n: digest(b) for n, b in payload.items()},
        "labels_read_for_export": True, "test_labels_used_for_training_or_selection": False,
        "api_calls": 0, "training_started": False, "ready_for_training": False}
    payload["manifest.json"] = encoded(manifest)
    # Interrupted writes are resumable only when every existing file matches.
    out.mkdir(parents=True, exist_ok=True)
    for name, raw in payload.items():
        if (out / name).exists() and (out / name).read_bytes() != raw:
            raise ValueError("Output exists with different inputs: " + str(out / name))
    for name, raw in payload.items():
        if not (out / name).exists():
            with (out / name).open("xb") as f:
                f.write(raw)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("source_sha256", "files_sha256")}, ensure_ascii=False, indent=2))
    print("OUTPUT:", out)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    build(a.project, a.out)
