"""Pinned SCALE public-data adapter for RefGrader (not the SCALE method)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_datasets.contract import write_json, write_jsonl, sha256_file, audit_prepared_benchmark
from benchmark_datasets.adapters.sas_bench import _rubric
from rubric_semantics import RUBRIC_SEMANTIC_CONTRACT_VERSION

COMMIT = "9e1a866d58edaa0d78af374d2f3529a10d72239a"
DATASETS = ("ADS", "LE", "ASAG", "SR")
EXPECTED = {"ADS": (1107, 475), "LE": (409, 176), "ASAG": (440, 189), "SR": (231, 100)}


def tab(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f, delimiter="\t"))


def source_records(source, dataset):
    source = Path(source)
    if dataset not in DATASETS:
        raise ValueError(dataset)
    tree = json.loads((source / "tree.json").read_text(encoding="utf8"))
    if tree["sha"] != COMMIT:
        raise ValueError("Source commit mismatch")
    entries = {x["path"]: x["sha"] for x in tree["tree"] if x["type"] == "blob"}
    hashes = {}

    def checked(name):
        path = source / dataset / name
        raw = path.read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if digest != entries.get(dataset + "/" + name):
            raise ValueError("Pinned source hash mismatch: " + str(path))
        hashes[dataset + "/" + name] = sha256_file(path)
        return path

    qa = {}
    if dataset in ("ASAG", "LE"):
        for row in tab(checked("qa_zh.csv" if dataset == "ASAG" else "qa.csv")):
            if len(row) != 3 or row[0] in qa:
                raise ValueError("Question table schema mismatch")
            qa[row[0]] = (row[1], row[2])
    output = {}
    for split in ("train", "test"):
        filename = split + ".jsonl" if dataset == "ADS" else "s_" + ("train" if split == "train" else "eval") + "_zh.csv"
        path = checked(filename)
        rows = ([json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
                if dataset == "ADS" else tab(path))
        result = []
        for line, row in enumerate(rows, 1):
            if dataset == "ADS":
                qid, question, answer = row["q_id"], row["question"], row["answer"]
                refs = [row.get("reference" + str(i), "") for i in (1, 2, 3)]
                refs = [r for r in refs if r.strip()]
                gold, source_max = float(row["score"]), 10.0
            elif dataset == "LE":
                if len(row) != 4:
                    raise ValueError("LE row schema mismatch")
                qid, ref, answer, score = row
                question = qa[qid][0]
                refs, gold, source_max = [ref], float(score), 1.0
            elif dataset == "ASAG":
                if len(row) != 4:
                    raise ValueError("ASAG row schema mismatch")
                qid, score, _, answer = row
                question, ref = qa[qid]
                refs, gold, source_max = [ref], float(score), 5.0
            else:
                if len(row) != 8:
                    raise ValueError("SR row schema mismatch")
                _, answer, topic, score, _, _, qid, ref = row
                # No fabricated original question; published topic is the available context.
                question = "[原始题干未提供] 主题：" + topic
                refs, gold, source_max = [ref], float(score), 1.0
            if not math.isfinite(gold) or not 0 <= gold <= source_max or not refs:
                raise ValueError("Invalid gold/reference at " + str(path) + ":" + str(line))
            context = json.dumps([str(qid), question, refs], ensure_ascii=False, separators=(",", ":"))
            key = hashlib.sha256(context.encode()).hexdigest()[:16]
            result.append(dict(answer_id=f"{dataset}_{split}_{line:05d}", question_id=f"{dataset}_{key}",
                               source_question_id=str(qid), question=question, references=refs, raw_text=answer,
                               source_score=gold, source_max=source_max, score=gold / source_max * 10,
                               source_split=split, source_line=line, source_file=dataset + "/" + filename))
        output[split] = result
    counts = tuple(len(output[s]) for s in ("train", "test"))
    if counts != EXPECTED[dataset]:
        raise ValueError(f"Unexpected {dataset} counts: {counts}, expected {EXPECTED[dataset]}")
    return output, hashes


def prepare(source, dataset, output):
    output = Path(output)
    records, hashes = source_records(source, dataset)
    spec = dict(dataset=dataset, commit=COMMIT, split_seed=42, validation_count=50,
                rubric_calibration_per_context=5, score_scale="linear 0-10",
                context_policy="question plus exact ordered references", method="RefGrader",
                source_hashes=hashes, adapter_sha256=sha256_file(Path(__file__)))
    if (output / "manifest.json").is_file():
        if json.loads((output / "source/adapter_spec.json").read_text(encoding="utf8")) != spec:
            raise ValueError("Existing dataset belongs to another protocol")
        result = audit_prepared_benchmark(output)
        if result["prepared_content_sha256"] != json.loads((output / "audit.json").read_text())["prepared_content_sha256"]:
            raise ValueError("Prepared dataset changed")
        return result
    if output.exists() and any(output.iterdir()):
        raise ValueError("Partial prepared directory; inspect before recovery: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    validation = set(random.Random(42).sample(range(len(records["test"])), 50))
    for i, r in enumerate(records["test"]):
        r["split"] = "validation" if i in validation else "test"
    train_groups = defaultdict(list)
    for r in records["train"]:
        train_groups[r["question_id"]].append(r)
    for qid, rs in train_groups.items():
        selected = sorted(rs, key=lambda r: hashlib.sha256(("42|" + r["answer_id"]).encode()).hexdigest())[:5]
        ids = {r["answer_id"] for r in selected}
        for r in rs:
            r["split"] = "calibration" if r["answer_id"] in ids else "train"
    groups = defaultdict(list)
    for r in records["train"] + records["test"]:
        groups[r["question_id"]].append(r)
    questions, metadata, labels, exclusions = [], [], {}, []
    for qid, rs in sorted(groups.items()):
        first = rs[0]
        ref = "\n\n".join(first["references"])
        guidance = "根据给定参考答案评价正确性与完整性，允许等价表达及部分得分。此为参考答案重建规则，非官方细则；不得推测未提供的题干或事实。"
        rubric = _rubric(ref, guidance, 10.0)
        rubric[0]["parent_official_item"] = "reference_based_reconstruction_not_official"
        rubric[0]["metadata_source"] = "scale_public_reference_reconstruction"
        initial = f"rubrics/initial/{dataset}/{qid}_rubric_standard.json"
        optimized = f"rubrics/optimized/{dataset}/{qid}_rubric_standard.json"
        manifest_path = f"rubrics/manifests/{dataset}/{qid}_optimization.json"
        split_path = f"splits/{qid}.json"
        split = {name: [r["answer_id"] for r in rs if r["split"] == name]
                 for name in ("train", "calibration", "validation", "test")}
        write_json(output / initial, rubric)
        write_json(output / optimized, rubric)
        write_json(output / split_path, split)
        write_json(output / manifest_path, dict(question_id=qid, rubric_group=dataset,
            method="label_blind_reference_baseline", selected_variant="baseline",
            rubric_semantic_contract_version=RUBRIC_SEMANTIC_CONTRACT_VERSION,
            semantic_policy_validated=True, semantic_validation_mode="prepared_label_blind_baseline",
            initial_rubric=initial, optimized_rubric=optimized,
            initial_sha256=sha256_file(output / initial), optimized_sha256=sha256_file(output / optimized),
            calibration_answer_ids=split["calibration"], path_format="prepared_relative_v1"))
        questions.append(dict(question_id=qid, subject="public_benchmark", exam_source=f"scale4_{dataset.lower()}_v1",
            question_text=first["question"], question_image=None, ref_text=ref, ref_image=None, total_score=10.0,
            official_rubric=guidance, rubric_group=dataset, source_rubric_path=initial, initial_rubric_path=initial,
            optimized_rubric_path=optimized, rubric_split_path=split_path, rubric_calibration_ids=split["calibration"],
            student_images_dir=f"text_samples/{qid}", requires_visual_evidence=False, sample_count=len(rs),
            source_question_id=first["source_question_id"], source_task=dataset))
        if not split["calibration"]:
            exclusions.append(dict(question_id=qid, reason="No upstream-training answer with this exact reference context; label-blind rubric only"))
        for r in rs:
            aid = r["answer_id"]
            labels[aid] = {qid: r["score"]}
            metadata.append({k: r[k] for k in ("answer_id", "question_id", "source_question_id", "source_file", "source_line", "source_split")} | dict(
                raw_text=r["raw_text"] if r["raw_text"].strip() else "[空白作答]",
                original_raw_text=r["raw_text"], blank_passthrough=not bool(r["raw_text"].strip()),
                subject="public_benchmark", subject_name=dataset, isimagine=False,
                visual_placeholder_detected=False, student_image=None, source_image=None))
    write_json(output / "exam_database.json", questions)
    write_json(output / "teacher_scores.json", labels)
    write_jsonl(output / "answer_metadata.jsonl", metadata)
    write_jsonl(output / "gold_only/source_scores.jsonl", [dict(answer_id=r["answer_id"], source_score=r["source_score"], source_max=r["source_max"]) for rs in records.values() for r in rs])
    write_json(output / "source/adapter_spec.json", spec)
    overlap = defaultdict(lambda: defaultdict(list))
    for rs in records.values():
        for r in rs:
            key = hashlib.sha256(json.dumps([r["question_id"], r["raw_text"]], ensure_ascii=False).encode()).hexdigest()
            overlap[key][r["split"]].append(r["answer_id"])
    write_json(output / "overlap_audit.json", [dict(text_sha256=k, splits=dict(v)) for k, v in overlap.items() if len(v) > 1])
    write_json(output / "no_training_contexts.json", exclusions)
    write_json(output / "manifest.json", dict(schema_version=1, dataset_id=f"scale4_{dataset.lower()}_v1", adapter="scale4",
        source=dict(sha256=COMMIT, adapter_spec_sha256=sha256_file(output / "source/adapter_spec.json")),
        counts=dict(questions=len(questions), answers=len(metadata)), split=dict(policy="upstream train; 50 fixed upstream-test validation; remainder test", ratios={}),
        score_policy="source score / source maximum * 10; retain source scores separately",
        limits=["ASAG/SR public Chinese translations", "ASAG pinned public files contain 629 not 630 rows", "SR question text unavailable", "reference-derived rubric, not official scoring criteria", "author split seed and QWK discretization unverified", "audit exact text overlaps without dropping rows"],
        blank_policy="retain rows; explicit blank marker only in model input; exact original text separately retained"))
    audit = audit_prepared_benchmark(output)
    write_json(output / "audit.json", audit)
    return audit
