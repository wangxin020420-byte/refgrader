"""Pooled (global) QWK for the two locally archived runs.

Same convention as scripts/compare_reviewed_history.py metrics():
clip(round(score), 0, question_max) then quadratic-weighted kappa.
Per-question QWK is recomputed to cross-check against metrics.json;
global QWK pools all answers of all questions before the kappa call.
Offline only: reads saved evaluation CSVs, no API calls.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

def cohen_kappa_score(y_true, y_pred, weights="quadratic"):
    """Quadratic-weighted kappa over the labels present in y_true/y_pred."""
    labels = np.unique(np.r_[y_true, y_pred])
    lut = {v: i for i, v in enumerate(labels)}
    k = len(labels)
    t = np.array([lut[v] for v in y_true])
    p = np.array([lut[v] for v in y_pred])
    O = np.zeros((k, k))
    np.add.at(O, (t, p), 1.0)
    O /= O.sum()
    hist_t, hist_p = O.sum(1), O.sum(0)
    E = np.outer(hist_t, hist_p)
    w = np.zeros((k, k)) if k == 1 else np.fromfunction(
        lambda i, j: (i - j) ** 2 / (k - 1) ** 2, (k, k))
    return 1.0 - (w * O).sum() / (w * E).sum()

ART = Path("C:/Users/wx/Desktop/refgrader-artifacts/csbench")
RUNS = {
    "optimized": "reviewed45_glm52_20261019_221414_317274_test",
    "historical": "csbench45_full_experimental_glm52_20260825_163432",
}
KEYS = ("single_first_score", "model_avg_score",
        "three_way_core_score", "final_calibrated_score")
STAGES = {"single_first_score": "single", "model_avg_score": "avg",
          "three_way_core_score": "3WD-Core", "final_calibrated_score": "3WD"}
QUESTIONS = sorted(p.name for p in ART.iterdir()
                   if p.is_dir() and "_" in p.name and p.name.split("_")[0].isupper())


def pooled(run_dir: str) -> dict:
    rows = {}
    maxima = {}
    for q in QUESTIONS:
        base = ART / q / "grading_runs" / run_dir / "evaluation"
        # every question dir carries the full-test compare.csv; dedupe by key
        with (base / "compare.csv").open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                key = (row["question"], row["student_id"])
                if not all(key) or key in rows:
                    continue
                rows[key] = (row["question"], float(row["teacher"]),
                             *[float(row[k]) for k in KEYS])
        summary = json.loads((base / "summary.json").read_text(encoding="utf-8-sig"))
        pq = {r["q_id"]: float(r["total"]) for r in summary["per_question"]}
        maxima.update({k: v for k, v in pq.items() if k == q})
    rows = list(rows.values())
    out = {"n": len(rows)}
    per_q = {k: {} for k in KEYS}
    for key in KEYS:
        idx = 2 + KEYS.index(key)
        y = np.array([np.clip(round(r[1]), 0, maxima[r[0]]) for r in rows])
        p = np.array([np.clip(round(r[idx]), 0, maxima[r[0]]) for r in rows])
        # per-question values for cross-check
        qs = sorted({r[0] for r in rows})
        vals = []
        for q in qs:
            sel = [r for r in rows if r[0] == q]
            yq = np.array([np.clip(round(r[1]), 0, maxima[q]) for r in sel])
            pq_ = np.array([np.clip(round(r[idx]), 0, maxima[q]) for r in sel])
            if len(np.unique(np.r_[yq, pq_])) > 1:
                v = cohen_kappa_score(yq, pq_, weights="quadratic")
                if math.isfinite(v):
                    per_q[key][q] = v
                    vals.append(v)
        out[STAGES[key]] = {
            "global_qwk": float(cohen_kappa_score(y, p, weights="quadratic")),
            "macro_qwk": float(np.mean(vals)),
            "weighted_qwk": float(np.average(vals, weights=[1] * len(vals))) if False else None,
        }
        # weighted by n per question
        ns = {q: sum(1 for r in rows if r[0] == q) for q in per_q[key]}
        out[STAGES[key]]["weighted_qwk"] = float(np.average(
            [per_q[key][q] for q in per_q[key]], weights=[ns[q] for q in per_q[key]]))
        out[STAGES[key]]["n_valid_questions"] = len(per_q[key])
    return out


if __name__ == "__main__":
    result = {name: pooled(run) for name, run in RUNS.items()}
    out_path = Path(__file__).parent / "global_qwk.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
