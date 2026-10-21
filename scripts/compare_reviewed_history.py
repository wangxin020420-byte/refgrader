"""Offline decomposition of historical, sample-removal, relabeling and rerun effects.

Historical comparison is descriptive: code/rubric/calibration and stochastic
inference may differ. No model API calls or changes to source results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]
OLD_RUN = "csbench45_full_experimental_glm52_20260825_163432"
NEW_RUN = "reviewed45_glm52_20261019_221414_317274"
KEYS = ("single_first_score", "model_avg_score", "selected_baseline_score",
        "three_way_core_score", "final_calibrated_score")
SCENARIOS = ("old_raw_all", "old_retained_raw_labels",
             "old_retained_reviewed_labels", "new_retained_reviewed_labels")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path):
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["question"], row["student_id"])
            if not all(key) or key in result:
                raise ValueError("Missing or duplicate full (question, answer) key")
            result[key] = {field: float(row[field]) for field in ("teacher", *KEYS)}
            if not all(math.isfinite(value) for value in result[key].values()):
                raise ValueError("Non-finite score")
    return result


def max_scores(summary):
    result = {}
    for row in summary["per_question"]:
        q, value = row["q_id"], float(row["total"])
        if value <= 0 or (q in result and result[q] != value):
            raise ValueError("Inconsistent question maximum")
        result[q] = value
    return result


def metrics(rows, score_key, maxima):
    grouped = defaultdict(list)
    errors, normalized = [], []
    for (q, _), row in rows.items():
        grouped[q].append((row["teacher"], row[score_key]))
        errors.append(row[score_key] - row["teacher"])
        normalized.append(abs(errors[-1]) / maxima[q])
    errors = np.asarray(errors)
    question_metrics = []
    for q, values in sorted(grouped.items()):
        array = np.asarray(values)
        y, prediction = array[:, 0], array[:, 1]
        union = np.unique(np.r_[np.clip(np.round(y), 0, maxima[q]),
                                np.clip(np.round(prediction), 0, maxima[q])])
        qwk = None
        if len(union) > 1:
            value = cohen_kappa_score(np.clip(np.round(y), 0, maxima[q]).astype(int),
                                     np.clip(np.round(prediction), 0, maxima[q]).astype(int), weights="quadratic")
            if math.isfinite(value):
                qwk = float(value)
        question_metrics.append({"question_id": q, "n": len(y), "qwk": qwk,
            "mae": float(np.mean(abs(prediction-y))),
            "nmae": float(np.mean(abs(prediction-y)) / maxima[q])})
    defined = [row for row in question_metrics if row["qwk"] is not None]
    result = {"n": len(rows), "questions": len(grouped),
        "macro_qwk": float(np.mean([row["qwk"] for row in defined])) if defined else None,
        "weighted_qwk": float(np.average([row["qwk"] for row in defined], weights=[row["n"] for row in defined])) if defined else None,
        "qwk_valid_questions": len(defined),
        "mae": float(np.mean(abs(errors))), "rmse": float(np.sqrt(np.mean(errors**2))),
        "nmae": float(np.mean(normalized)), "bias": float(np.mean(errors)),
        "tar2": float(np.mean(abs(errors) <= 2))}
    return result, question_metrics


def decompose(old, new, maxima, policy):
    old_ids, new_ids = set(old), set(new)
    if not new_ids or not new_ids <= old_ids:
        raise ValueError("Reviewed results must be a nonempty subset of historical answer IDs")
    removed = old_ids - new_ids
    excluded = {(q, sid) for q, answers in policy["excluded"].items() for sid in answers}
    if removed != old_ids & excluded or new_ids & excluded:
        raise ValueError("Removed IDs do not match the frozen policy")
    retained = {key: dict(old[key]) for key in sorted(new_ids)}
    relabeled = {key: dict(old[key]) for key in sorted(new_ids)}
    corrections = policy.get("corrected_scores", {})
    changed = []
    for key in sorted(new_ids):
        q, sid = key
        expected = float(corrections.get(q, {}).get(sid, old[key]["teacher"]))
        if not math.isclose(new[key]["teacher"], expected, abs_tol=1e-9):
            raise ValueError("Reviewed label does not match frozen policy: " + str(key))
        relabeled[key]["teacher"] = expected
        if old[key]["teacher"] != expected:
            changed.append({"question_id": q, "student_id": sid,
                "old_teacher": old[key]["teacher"], "reviewed_teacher": expected,
                **{name+"_old_error_reduction": abs(old[key][name]-old[key]["teacher"]) - abs(old[key][name]-expected) for name in KEYS}})
    datasets = dict(zip(SCENARIOS, (old, retained, relabeled, new)))
    summaries, per_question = {}, []
    for scenario, rows in datasets.items():
        summaries[scenario] = {}
        for key in KEYS:
            overall, questions = metrics(rows, key, maxima)
            summaries[scenario][key] = overall
            per_question.extend({"scenario": scenario, "score_key": key, **row} for row in questions)
    effects = {}
    for key in KEYS:
        values = [summaries[name][key] for name in SCENARIOS]
        effects[key] = [{"transition": name,
            "mae_reduction": a["mae"]-b["mae"],
            "macro_qwk_change": b["macro_qwk"]-a["macro_qwk"] if a["macro_qwk"] is not None and b["macro_qwk"] is not None else None}
            for name, a, b in zip(("sample_removal", "label_correction", "rerun_and_other_changes"), values, values[1:])]
    return {"counts": {"old_test": len(old), "retained_test": len(new),
                       "excluded_test": len(removed), "relabelled_test": len(changed)},
            "scenarios": summaries, "effects": effects}, per_question, changed, datasets


def bootstrap_paired(old, new, maxima, seed=20261021, iterations=2000):
    """Paired question-cluster bootstrap; positive values mean improvement."""
    questions = sorted({q for q, _ in new})
    draws = np.random.default_rng(seed).integers(0, len(questions), (iterations, len(questions)))
    output = {}
    for key in KEYS:
        _, old_q = metrics(old, key, maxima)
        _, new_q = metrics(new, key, maxima)
        old_lookup, new_lookup = ({row["question_id"]: row for row in rows} for rows in (old_q, new_q))
        n = np.asarray([new_lookup[q]["n"] for q in questions])
        reduction = np.asarray([old_lookup[q]["mae"]-new_lookup[q]["mae"] for q in questions])
        samples = np.sum((reduction*n)[draws], axis=1) / np.sum(n[draws], axis=1)
        item = {"mae_reduction": float(np.sum(reduction*n)/sum(n)),
                "mae_reduction_ci95": np.quantile(samples, [.025, .975]).tolist()}
        if all(old_lookup[q]["qwk"] is not None and new_lookup[q]["qwk"] is not None for q in questions):
            changes = np.asarray([new_lookup[q]["qwk"]-old_lookup[q]["qwk"] for q in questions])
            item.update(macro_qwk_change=float(changes.mean()),
                        macro_qwk_change_ci95=np.quantile(changes[draws].mean(axis=1), [.025, .975]).tolist())
        output[key] = item
    return {"method": "paired question-cluster percentile bootstrap", "iterations": iterations,
            "seed": seed, "interpretation": "Descriptive uncertainty across questions; not a causal estimate of review benefit.",
            "scores": output}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "results_runs/csbench_q45_12039bfe25ca4168_full/runs"
    parser.add_argument("--old-evaluation", type=Path, default=base/OLD_RUN/"evaluation")
    parser.add_argument("--new-evaluation", type=Path, default=base/(NEW_RUN+"_test")/"evaluation")
    parser.add_argument("--job", type=Path, default=ROOT/"results_runs/reviewed_csbench45_glm52"/NEW_RUN)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    old, new = (load_rows(p/"compare.csv") for p in (args.old_evaluation, args.new_evaluation))
    old_summary, new_summary = (read(p/"summary.json") for p in (args.old_evaluation, args.new_evaluation))
    maxima = max_scores(old_summary)
    if maxima != max_scores(new_summary):
        raise ValueError("Question maxima or question sets differ")
    policy = read(args.job/"sample_policy.json")
    report, questions, changes, datasets = decompose(old, new, maxima, policy)
    # Detect a mismatched historical export instead of silently comparing it.
    for summary, scenario in ((old_summary, SCENARIOS[0]), (new_summary, SCENARIOS[-1])):
        for key in KEYS:
            archived = next(row for row in summary["global"] if row["score_key"] == key)
            if not math.isclose(archived["MAE"], report["scenarios"][scenario][key]["mae"], abs_tol=1e-10):
                raise ValueError("CSV and archived summary disagree: " + scenario + " " + key)
            qwk = [row["QWK"] for row in summary["per_question"]
                   if row["score_key"] == key and row.get("QWK") is not None and math.isfinite(row["QWK"])]
            actual_qwk = report["scenarios"][scenario][key]["macro_qwk"]
            if qwk and not math.isclose(sum(qwk)/len(qwk), actual_qwk, abs_tol=1e-10):
                raise ValueError("CSV and archived QWK disagree: " + scenario + " " + key)
    report["bootstrap"] = bootstrap_paired(datasets[SCENARIOS[2]], new, maxima)
    report["runs"] = {"historical": OLD_RUN, "reviewed": NEW_RUN,
                      "reviewed_policy_id": policy["policy_id"]}
    report["limitations"] = [
        "Historical and reviewed runs are not a controlled causal ablation of label review.",
        "Sample removal changes the evaluation population; a better metric does not prove labels were wrong.",
        "Corrected labels need independent evidence; no assumption that they must favor model predictions.",
        "The rerun term combines changed inference, rubric optimization, calibration and code/version effects.",
        "TAR2 uses the historical fixed two-point threshold for comparability, not the routing safety tolerance."]
    inputs = [p/name for p in (args.old_evaluation, args.new_evaluation) for name in ("compare.csv", "summary.json")]
    inputs += [args.job/"sample_policy.json", args.job/"selection.json"]
    report["input_sha256"] = {str(p.resolve()): sha(p) for p in inputs}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    write_csv(out/"per_question.csv", questions)
    write_csv(out/"changed_labels.csv", changes)
    table = []
    for scenario in SCENARIOS:
        table.append({"scenario": scenario, **report["scenarios"][scenario]["final_calibrated_score"]})
    write_csv(out/"decomposition_final.csv", table)
    print(json.dumps({"counts":report["counts"], "final_score_decomposition":table,
                      "final_score_bootstrap":report["bootstrap"]["scores"]["final_calibrated_score"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
