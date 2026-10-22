"""Offline, append-only snapshots of the reviewed85 PC1/PC2 baselines.

Standard library only; no model imports, Git mutations, active-policy changes,
or grading calls. Original absolute-path provenance is copied without rebasing.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "pc1-optimized": ("reviewed_csbench45_glm52", "reviewed45_glm52_20261019_221414_317274"),
    "pc2-initial": ("initial_rubric_ablation", "initial85_glm52_20261020_pc2"),
}
SCORES = {
    "single": "single_first_score", "avg": "model_avg_score",
    "selected": "selected_baseline_score", "3WD-Core": "three_way_core_score",
    "3WD": "final_calibrated_score",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def contained(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path outside intended root: {path}") from exc
    return path


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite score")
    return result


def checkpoint_score(record, field):
    if field == "single_first_score":
        history = record.get("model_scores_history") or []
        if not isinstance(history, list) or not history:
            raise ValueError("Missing first scoring probe")
        return finite(history[0])
    return finite(record[field])


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key = (row["question"], row["student_id"])
        if key in result:
            raise ValueError(f"Duplicate evaluation answer: {key}")
        for field in ("teacher", *SCORES.values()):
            finite(row[field])
        result[key] = row
    return result


def qwk(teacher, prediction):
    # sklearn's default sorted union of integer labels; squared index distances.
    labels = sorted(set(teacher) | set(prediction))
    if len(labels) < 2:
        return None
    positions = {label: i for i, label in enumerate(labels)}
    cy, cp = Counter(teacher), Counter(prediction)
    observed = sum((positions[y] - positions[p]) ** 2
                   for y, p in zip(teacher, prediction))
    expected = sum(ny * np * (positions[y] - positions[p]) ** 2
                   for y, ny in cy.items() for p, np in cp.items()) / len(teacher)
    return 1.0 - observed / expected if expected else None


def score_metrics(rows, summary):
    result = {}
    for stage, field in SCORES.items():
        refs = {r["q_id"]: r for r in summary["per_question"] if r["score_type"] == stage}
        groups = {}
        for (qid, _), row in rows.items():
            groups.setdefault(qid, []).append(row)
        if set(groups) != set(refs):
            raise ValueError(f"Summary question set differs: {stage}")
        all_y, all_p, values, weights, errors = [], [], [], [], []
        for qid, items in sorted(groups.items()):
            ref = refs[qid]
            if len(items) != int(ref["n"]):
                raise ValueError(f"Summary count differs: {stage}/{qid}")
            maximum = finite(ref["total"])
            if maximum <= 0:
                raise ValueError(f"Invalid maximum: {qid}")
            y = [finite(r["teacher"]) for r in items]
            p = [finite(r[field]) for r in items]
            yi = [int(min(maximum, max(0, round(v)))) for v in y]
            pi = [int(min(maximum, max(0, round(v)))) for v in p]
            value = qwk(yi, pi)
            archived = ref.get("QWK")
            if value is not None:
                if archived is None or not math.isclose(value, finite(archived), abs_tol=1e-9):
                    raise ValueError(f"Per-question QWK differs: {stage}/{qid}")
                values.append(value)
                weights.append(len(items))
            all_y.extend(yi)
            all_p.extend(pi)
            errors.extend(pv - yv for yv, pv in zip(y, p))
        archived_global = next(r for r in summary["global"] if r["score_type"] == stage)
        n = len(errors)
        mae = sum(abs(e) for e in errors) / n
        rmse = math.sqrt(sum(e * e for e in errors) / n)
        if n != int(archived_global["n"]) or not math.isclose(mae, finite(archived_global["MAE"]), abs_tol=1e-9):
            raise ValueError(f"CSV/global summary differs: {stage}")
        if not math.isclose(rmse, finite(archived_global["RMSE"]), abs_tol=1e-9):
            raise ValueError(f"CSV/global RMSE differs: {stage}")
        result[stage] = {
            "n": n, "macro_qwk": sum(values) / len(values) if values else None,
            "weighted_qwk": sum(v * w for v, w in zip(values, weights)) / sum(weights) if weights else None,
            "pooled_raw_score_qwk": qwk(all_y, all_p), "mae": mae, "rmse": rmse,
            "bias": sum(errors) / n,
        }
    return result


def make_plan(repo_root, arm, run_id):
    repo = Path(repo_root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", run_id):
        raise ValueError("Invalid run ID")
    job = repo / "results_runs" / ARMS[arm][0] / run_id
    state = read(job / "workflow.json")
    if state.get("run_id") != run_id or state.get("status") != "completed":
        raise ValueError("Only this completed run can be frozen; do not rerun grading to archive it")
    selection, policy = read(job / "selection.json"), read(job / "sample_policy.json")
    questions = selection["questions"]
    if not questions or len(set(questions)) != len(questions) or set(questions) != set(state["questions"]):
        raise ValueError("Question selection mismatch")
    config = read(job / "a3wa_config.json")
    calibration_proof = {Path(k): v for k, v in
                         (state.get("completed", {}).get("calibrate") or {}).get("proof", {}).items()}
    if calibration_proof.get((job / "a3wa_config.json").resolve()) != digest(job / "a3wa_config.json"):
        raise ValueError("Completed calibration proof is missing or differs")
    for filename, key in (("selection.json", "selection_sha256"),
                          ("sample_policy.json", "policy_sha256"),
                          ("review_decisions.jsonl", "review_sha256"),
                          ("input_sha256.json", "input_manifest_sha256")):
        if digest(job / filename) != state.get(key):
            raise ValueError(f"Frozen workflow fingerprint differs: {filename}")
    if (config.get("sample_quality_policy") or {}).get("sha256") != digest(job / "sample_policy.json"):
        raise ValueError("Calibration uses a different sample policy")
    model = config.get("model_config") or {}
    contract = state.get("model") or {}
    for ck, sk in (("text_model", "text_model"), ("text_thinking", "thinking"), ("vlm_model", "vlm_model")):
        if not contract.get(sk) or model.get(ck) != contract[sk]:
            raise ValueError(f"Model contract differs: {ck}")
    sources = {}

    def add(source, target, expected=None):
        source = contained(source, repo)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Missing regular file: {source}")
        if target in sources:
            raise ValueError(f"Duplicate archive target: {target}")
        actual = digest(source)
        if expected is not None and actual != expected:
            raise ValueError(f"Frozen source fingerprint differs: {source}")
        sources[target] = {"path": str(source), "sha256": actual, "size": source.stat().st_size}

    for name in ("workflow.json", "selection.json", "sample_policy.json", "review_decisions.jsonl",
                 "input_sha256.json", "a3wa_config.json"):
        add(job / name, "workflow/" + name)
    if (job / "resume_code_repair.json").is_file():
        add(job / "resume_code_repair.json", "workflow/resume_code_repair.json",
            state.get("resume_code_repair_sha256"))

    prepared = contained(state["prepared"], repo)
    original_inputs = {Path(k): v for k, v in read(job / "input_sha256.json").items()}
    dataset_files = [prepared / n for n in ("teacher_scores.json", "answer_metadata.jsonl", "exam_database.json")]
    for qid in questions:
        group = qid.split("_")[0]
        dataset_files.extend((prepared / "splits/by_question" / (qid + ".json"),
                              prepared / "rubrics/initial" / group / (qid + "_rubric_standard.json")))
    for source in dataset_files:
        expected = original_inputs.get(source.resolve())
        if expected is None:
            raise ValueError(f"Dataset file absent from original input manifest: {source}")
        add(source, "inputs/" + source.relative_to(prepared).as_posix(), expected)
    if arm == "pc2-initial":
        for source, target in ((prepared / "snapshot_contract.json", "inputs/snapshot_contract.json"),
                               (job / "exam_database.json", "workflow/exam_database.json")):
            expected = original_inputs.get(source.resolve())
            if expected is None:
                raise ValueError(f"Missing original fingerprint: {source}")
            add(source, target, expected)
    else:
        for qid in questions:
            rubric = prepared / "rubrics/optimized" / qid.split("_")[0] / (qid + "_rubric_standard.json")
            proof = (state.get("completed", {}).get("rubric_" + qid) or {}).get("proof", {})
            expected = {Path(k): v for k, v in proof.items()}.get(rubric.resolve())
            if not expected:
                raise ValueError(f"Missing completed optimized-rubric proof: {qid}")
            add(rubric, "inputs/" + rubric.relative_to(prepared).as_posix(), expected)

    teacher_db = read(prepared / "teacher_scores.json")
    test_identity = set()
    checkpoint_records = {}
    for split in ("validation", "test"):
        directory = (job / split if arm == "pc2-initial" else Path(state["paths"][split]))
        directory = contained(directory, repo)
        total = 0
        proof = (state.get("completed", {}).get("grade_" + split) or {}).get("proof", {})
        for qid in questions:
            filename = qid + "_grading_checkpoint.json"
            path = directory / filename
            records = read(path)
            ids = [r.get("student_id") for r in records]
            split_ids = read(prepared / "splits/by_question" / (qid + ".json"))[split]
            if len(split_ids) != len(set(split_ids)):
                raise ValueError(f"Duplicate IDs in split: {qid}/{split}")
            expected = set(split_ids) - set(policy.get("excluded", {}).get(qid, {}))
            if len(ids) != len(set(ids)) or set(ids) != expected:
                raise ValueError(f"Incomplete or incompatible checkpoint IDs: {qid}/{split}")
            for record in records:
                for field in SCORES.values():
                    # selected_baseline was added before these paired runs.
                    checkpoint_score(record, field)
                if split == "test":
                    key = (qid, record["student_id"])
                    test_identity.add(key)
                    checkpoint_records[key] = record
            total += len(records)
            expected_proof = {Path(k): v for k, v in proof.items()}.get(path.resolve())
            if not expected_proof:
                raise ValueError(f"Missing completed checkpoint proof: {split}/{qid}")
            add(path, split + "/" + filename, expected_proof)
        if total != int(selection["splits"][split]["effective"]):
            raise ValueError(f"Effective split count differs: {split}")
    evaluation = job / "reports" if arm == "pc2-initial" else directory / "evaluation"
    csv_name = "score_comparison.csv" if arm == "pc2-initial" else "compare.csv"
    rows, summary = csv_rows(evaluation / csv_name), read(evaluation / "summary.json")
    if set(rows) != test_identity:
        raise ValueError("Evaluation/checkpoint answer identities differ")
    for (qid, sid), row in rows.items():
        truth = policy.get("corrected_scores", {}).get(qid, {}).get(sid, teacher_db.get(sid, {}).get(qid))
        if truth is None or finite(truth) != finite(row["teacher"]):
            raise ValueError(f"Evaluation teacher label differs from frozen dataset: {qid}/{sid}")
        for field in SCORES.values():
            if not math.isclose(finite(row[field]), round(checkpoint_score(checkpoint_records[qid, sid], field), 2), abs_tol=1e-9):
                raise ValueError(f"Evaluation/checkpoint score differs: {qid}/{sid}/{field}")
    metrics = score_metrics(rows, summary)
    add(evaluation / csv_name, "evaluation/compare.csv")
    add(evaluation / "summary.json", "evaluation/summary.json")
    for folder in (job / "calibration_diagnostics", job / "a3wa_config_diagnostics"):
        if folder.is_dir():
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in (".json", ".csv", ".md"):
                    add(path, "diagnostics/" + folder.name + "/" + path.relative_to(folder).as_posix())
    gate = bool((config.get("deployment_gate") or {}).get("passed", False))
    metadata = {
        "schema_version": 1, "arm": arm, "run_id": run_id,
        "status": "completed", "result_class": "formal" if gate else "experimental",
        "deployment_gate_passed": gate, "model_config": model,
        "review_snapshot_records": sum(bool(line.strip()) for line in (job / "review_decisions.jsonl").read_text(encoding="utf-8-sig").splitlines()),
        "selection_sha256": digest(job / "selection.json"), "policy_sha256": digest(job / "sample_policy.json"),
        "splits": selection["splits"], "question_ids": questions,
        "source_files": sources,
        "qwk_method": "Per-question round-to-even, clip to question maximum, int conversion; pooled raw-score categories, no score normalization",
        "scope": "Frozen paired-run evidence, not a new grading run or a portable resume state. Images remain in the versioned dataset; original image fingerprints remain in input_sha256.json.",
        "network_actions": "none",
    }
    return metadata, metrics


def verify(snapshot):
    root = Path(snapshot).resolve()
    metadata = read(root / "baseline_manifest.json")
    if metadata.get("freeze_complete") is not True:
        raise ValueError("Snapshot is incomplete")
    files = metadata["files"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(files) | {"baseline_manifest.json"}:
        raise ValueError("Snapshot contains missing or unrecorded files")
    for name, expected in files.items():
        path = contained(root / name, root)
        if path.is_symlink() or digest(path) != expected["sha256"] or path.stat().st_size != expected["size"]:
            raise ValueError(f"Snapshot hash differs: {name}")
    return metadata


def freeze(repo, arm, run_id, artifacts, dry_run=False):
    artifact_root, code_root = Path(artifacts).resolve(), Path(repo).resolve()
    try:
        artifact_root.relative_to(code_root)
    except ValueError:
        pass
    else:
        raise ValueError("Artifacts must be a separate repository, outside the code checkout")
    if not (artifact_root / ".git").exists():
        raise ValueError("Artifacts repository not found; clone or specify the existing repository first")
    metadata, metrics = make_plan(repo, arm, run_id)
    destination = artifact_root / "csbench/frozen_baselines" / run_id
    if destination.exists():
        old = verify(destination)
        if old["arm"] != arm or old["source_files"] != metadata["source_files"]:
            raise ValueError("An existing snapshot differs; it will not be overwritten")
        return {"status": "already_frozen_verified", "path": str(destination), "files": len(old["files"])}
    if dry_run:
        return {"status": "dry_run_passed", "path": str(destination), "source_files": len(metadata["source_files"]),
                "splits": metadata["splits"], "result_class": metadata["result_class"], "metrics": metrics}
    destination.mkdir(parents=True, exist_ok=False)
    for target, source in metadata["source_files"].items():
        output = contained(destination / target, destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source["path"], output)
        if digest(output) != source["sha256"]:
            raise ValueError(f"Source changed while copying; incomplete snapshot retained: {target}")
    # Detect changes to earlier inputs while later, larger checkpoint files copied.
    for target, source in metadata["source_files"].items():
        if digest(source["path"]) != source["sha256"]:
            raise ValueError(f"Source changed during freeze; incomplete snapshot retained: {target}")
    (destination / ".gitattributes").write_text("* -text\n**/* -text\n", encoding="utf-8")
    write(destination / "metrics.json", metrics)
    (destination / "README.md").write_text(
        f"# {arm}: {run_id}\n\nFrozen completed baseline; SHA-256 verification is required after Git checkout.\n\n"
        "`workflow/` is original provenance, not cross-device resume state. `inputs/` holds the frozen labels, splits and rubrics. "
        "`validation/` and `test/` hold completed checkpoints. `evaluation/` holds original reports; `metrics.json` adds pooled QWK.\n\n"
        "Do not put subsequent review decisions, new evaluations or modified configuration in this directory. "
        "The manifest preserves the actual deployment gate; completed does not mean safe for deployment.\n",
        encoding="utf-8")
    metadata["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["files"] = {p.relative_to(destination).as_posix(): {"sha256": digest(p), "size": p.stat().st_size}
                         for p in sorted(destination.rglob("*")) if p.is_file()}
    metadata["freeze_complete"] = True
    write(destination / "baseline_manifest.json", metadata)
    verify(destination)
    return {"status": "frozen_verified", "path": str(destination), "files": len(metadata["files"]),
            "result_class": metadata["result_class"], "metrics": metrics}


def compare(optimized, initial):
    a, b = verify(optimized), verify(initial)
    if a["arm"] != "pc1-optimized" or b["arm"] != "pc2-initial":
        raise ValueError("Wrong comparison arms")
    same = {}
    for key in ("selection_sha256", "policy_sha256", "question_ids", "splits"):
        same[key] = a[key] == b[key]
    for name in ("workflow/review_decisions.jsonl", "inputs/teacher_scores.json", "inputs/answer_metadata.jsonl"):
        same[name] = a["files"][name]["sha256"] == b["files"][name]["sha256"]
    for prefix in ("inputs/splits/", "inputs/rubrics/initial/"):
        same[prefix] = ({k: v["sha256"] for k, v in a["files"].items() if k.startswith(prefix)} ==
                        {k: v["sha256"] for k, v in b["files"].items() if k.startswith(prefix)})
    same["model_config"] = a["model_config"] == b["model_config"]
    ar, br = csv_rows(Path(optimized) / "evaluation/compare.csv"), csv_rows(Path(initial) / "evaluation/compare.csv")
    same["test_answer_ids"] = set(ar) == set(br)
    same["effective_test_labels"] = same["test_answer_ids"] and all(finite(ar[k]["teacher"]) == finite(br[k]["teacher"]) for k in ar)
    return {"paired_data_and_model_contract_passed": all(same.values()), "checks": same,
            "optimized_run": a["run_id"], "initial_run": b["run_id"],
            "limits": "Calibration parameters are independently fitted. This check does not establish equal OCR cache contents, installed dependencies, random samples, model service revisions or every runtime environment variable."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--run-id")
    p.add_argument("--artifacts-repo", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("verify")
    p.add_argument("--snapshot", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--optimized-snapshot", type=Path, required=True)
    p.add_argument("--initial-snapshot", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            result = freeze(args.repo_root, args.arm, args.run_id or ARMS[args.arm][1], args.artifacts_repo, args.dry_run)
        elif args.command == "verify":
            data = verify(args.snapshot)
            result = {"status": "verified", "run_id": data["run_id"], "files": len(data["files"])}
        else:
            result = compare(args.optimized_snapshot, args.initial_snapshot)
            if not result["paired_data_and_model_contract_passed"]:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 1
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError, StopIteration) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
