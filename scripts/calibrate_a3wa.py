import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from calibration_utils import (  # noqa: E402
    apply_boundary_action_policy,
    build_a3wa_decision,
    compute_a3wa_thresholds,
    normalized_risk_weights,
    route_score_band,
    safe_float,
)


LEGACY_SCORES_MAP = {
    "Q1": 5,
    "Q2": 20,
    "Q3": 10,
    "Q4": 20,
    "Q5": 15,
    "Q6": 20,
    "Q7": 10,
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def expand_input_files(patterns):
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(matches)
        else:
            files.append(pattern)
    return files


def load_question_scores(database_path):
    if not database_path:
        return {}
    if not os.path.exists(database_path):
        return {}
    questions = load_json(database_path)
    if not isinstance(questions, list):
        return {}
    return {
        str(question.get("question_id")): safe_float(
            question.get("total_score", question.get("max_score", 0.0)), 0.0
        )
        for question in questions
        if question.get("question_id")
    }


def teacher_score(db, student_id, qid):
    student_id = str(student_id)
    exact_record = db.get(student_id, {})
    if isinstance(exact_record, dict) and qid in exact_record:
        return exact_record.get(qid)
    legacy_record = db.get(student_id.split("_")[0], {})
    if isinstance(legacy_record, dict):
        return legacy_record.get(qid)
    return None


def question_id_from_path(path):
    name = os.path.basename(path)
    for suffix in (
        "_grading_checkpoint.json",
        "_graded_results.json",
        "_rejected.json",
        "_failed.json",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if name.endswith(".json"):
        name = name[:-5]
    return name


def load_records(files, teacher_db, question_scores):
    records = []
    for path in files:
        qid = question_id_from_path(path)
        max_score = question_scores.get(qid, LEGACY_SCORES_MAP.get(qid, 20))
        for row in load_json(path):
            teacher = teacher_score(teacher_db, row.get("student_id", ""), qid)
            avg = row.get("selected_baseline_score", row.get("model_avg_score"))
            final = row.get("final_calibrated_score")
            if teacher is None or teacher < 0 or avg is None or final is None:
                continue
            risk = row.get("risk_features", {}) or {}
            a3 = row.get("a3wa_decision", {}) or {}
            gate = row.get("boundary_gate", {}) or {}
            raw_candidate = gate.get("raw_candidate_score")
            if raw_candidate is None:
                raw_candidate = final
            records.append({
                "qid": qid,
                "student_id": row.get("student_id", ""),
                "teacher": safe_float(teacher, 0.0),
                "avg": safe_float(avg, 0.0),
                "model_avg": safe_float(row.get("model_avg_score", avg), safe_float(avg, 0.0)),
                "final": safe_float(final, safe_float(avg, 0.0)),
                "raw_candidate": safe_float(raw_candidate, safe_float(final, safe_float(avg, 0.0))),
                "max_score": max_score,
                "model_scores": row.get("model_scores_history") or [],
                "std_dev": safe_float(row.get("std_dev", 0.0), 0.0),
                "blank_rate": safe_float(row.get("blank_rate", risk.get("blank_rate", 0.0)), 0.0),
                "low_quality_rate": safe_float(
                    row.get("low_quality_extraction_rate", risk.get("low_quality_rate", 0.0)), 0.0
                ),
                "perception_failure_rate": safe_float(
                    row.get("perception_failure_rate", risk.get("perception_failure_rate", 0.0)), 0.0
                ),
                "structure_missing_rate": safe_float(
                    row.get("structure_missing_rate", risk.get("structure_missing_rate", 0.0)), 0.0
                ),
                "extraction_risk": safe_float(
                    row.get("extraction_risk", risk.get("extraction_risk", 0.0)), 0.0
                ),
                "extraction_quality": row.get("extraction_quality", ""),
                "fatal_points_ratio": safe_float(
                    risk.get("fatal_points_ratio", row.get("fatal_points_ratio", 0.0)), 0.0
                ),
                "high_blank_high_score": bool(risk.get("high_blank_high_score", row.get("high_blank_high_score", False))),
                "post_calibration": row.get("post_calibration", {}) or {},
                "old_route": row.get("3wd_route", ""),
                "old_mu": safe_float(a3.get("mu", risk.get("a3wa_mu", 0.0)), 0.0),
            })
    return records


def mean(values):
    return sum(values) / len(values) if values else 0.0


def metrics(records, score_key):
    diffs = [r[score_key] - r["teacher"] for r in records]
    return {
        "n": len(records),
        "mae": mean([abs(d) for d in diffs]),
        "rmse": math.sqrt(mean([d * d for d in diffs])),
        "tar2": mean([1.0 if abs(d) <= 2 else 0.0 for d in diffs]),
        "bias": mean(diffs),
        "over2": sum(1 for d in diffs if d > 2),
        "under2": sum(1 for d in diffs if d < -2),
    }


def candidate_weights():
    return [
        {"extract": 0.35, "score": 0.30, "semantic": 0.20, "blank": 0.15, "overcredit": 0.00},
        {"extract": 0.30, "score": 0.25, "semantic": 0.20, "blank": 0.10, "overcredit": 0.15},
        {"extract": 0.25, "score": 0.25, "semantic": 0.25, "blank": 0.10, "overcredit": 0.15},
        {"extract": 0.25, "score": 0.20, "semantic": 0.25, "blank": 0.10, "overcredit": 0.20},
        {"extract": 0.20, "score": 0.25, "semantic": 0.25, "blank": 0.10, "overcredit": 0.20},
        {"extract": 0.40, "score": 0.20, "semantic": 0.15, "blank": 0.15, "overcredit": 0.10},
    ]


def candidate_loss_params():
    for m in [0.20, 0.30, 0.40, 0.50]:
        for lambda1 in [3.0, 4.0, 5.0]:
            for lambda2 in [1.0, 2.0]:
                for mu1 in [2.0, 3.0]:
                    for mu2 in [5.0, 7.0]:
                        yield {
                            "lambda1": lambda1,
                            "lambda2": lambda2,
                            "mu1": mu1,
                            "mu2": mu2,
                            "m": m,
                        }


def apply_action_policy(record, route, decision):
    avg = record["avg"]
    max_score = max(record["max_score"], 1.0)
    if route != "BND":
        return avg
    risk_profile = {
        "fatal_points_ratio": record["fatal_points_ratio"],
        "high_blank_high_score": record["high_blank_high_score"],
        "risk_features": {
            "blank_rate": record["blank_rate"],
            "low_quality_rate": record["low_quality_rate"],
            "perception_failure_rate": record["perception_failure_rate"],
        },
    }
    gate = apply_boundary_action_policy(
        avg_model_score=avg,
        candidate_score=record["raw_candidate"],
        max_score=max_score,
        a3wa_decision=decision,
        risk_profile=risk_profile,
        post_calibration=record["post_calibration"],
    )
    return round(gate["final_score"], 2)


def evaluate_params(records, loss_params, weights, bnd_max, return_trial=False):
    trial = []
    route_counts = Counter()
    route_errors = {"POS": [], "BND": [], "NEG": []}
    bnd_gains = []
    weights = normalized_risk_weights(weights)
    for record in records:
        decision = build_a3wa_decision(
            model_scores=record["model_scores"],
            avg_model_score=record["avg"],
            std_dev=record["std_dev"],
            max_score=record["max_score"],
            blank_rate=record["blank_rate"],
            low_quality_rate=record["low_quality_rate"],
            perception_failure_rate=record["perception_failure_rate"],
            extraction_quality=record["extraction_quality"],
            structure_missing_rate=record["structure_missing_rate"],
            extraction_risk=record["extraction_risk"],
            fatal_points_ratio=record["fatal_points_ratio"],
            high_blank_high_score=record["high_blank_high_score"],
            post_calibration=record["post_calibration"],
            weights=weights,
            loss_params=loss_params,
        )
        route = decision["route"]
        score = apply_action_policy(record, route, decision)
        row = dict(record)
        row["trial_score"] = score
        row["trial_route"] = route
        trial.append(row)
        route_counts[route] += 1
        route_errors[route].append(abs(score - record["teacher"]))
        if route == "BND":
            bnd_gains.append(abs(record["avg"] - record["teacher"]) - abs(score - record["teacher"]))

    avg_metric = metrics(trial, "avg")
    final_metric = metrics(trial, "trial_score")
    bnd_ratio = route_counts["BND"] / max(len(trial), 1)
    pos_mae = mean(route_errors["POS"])
    bnd_mae = mean(route_errors["BND"])
    bnd_gain = mean(bnd_gains)
    objective = final_metric["mae"]
    objective += 2.0 * max(0.0, final_metric["mae"] - avg_metric["mae"])
    objective += 1.0 * max(0.0, avg_metric["tar2"] - final_metric["tar2"])
    objective += 0.5 * max(0.0, bnd_ratio - bnd_max)
    if route_counts["POS"] > 0 and route_counts["BND"] > 0:
        objective += 0.5 * max(0.0, pos_mae - bnd_mae)
    objective += 1.0 * max(0.0, -bnd_gain)
    result = {
        "objective": objective,
        "metrics": final_metric,
        "baseline_metrics": avg_metric,
        "route_counts": dict(route_counts),
        "bnd_ratio": bnd_ratio,
        "pos_mae": pos_mae,
        "bnd_mae": bnd_mae,
        "bnd_gain": bnd_gain,
        "loss_params": loss_params,
        "risk_weights": weights,
    }
    if return_trial:
        result["trial_records"] = trial
    return result


def _residual_entry(items, shrinkage_k, max_correction):
    residuals = [item["teacher"] - item["trial_score"] for item in items]
    n = len(residuals)
    raw_mean = mean(residuals)
    shrink = n / (n + max(shrinkage_k, 0.0)) if n > 0 else 0.0
    correction = max(-max_correction, min(max_correction, raw_mean * shrink))
    return {
        "n": n,
        "mean_residual": round(raw_mean, 6),
        "correction": round(correction, 6),
    }


def build_score_calibration(
    trial_records,
    *,
    min_cell_count=5,
    shrinkage_k=8.0,
    max_correction_ratio=0.12,
    max_correction_points=2.0,
):
    """Build an interpretable validation residual correction table.

    Corrections are additive and grouped from specific to general:
    question+route+score-band -> question+route -> question -> route -> global.
    Runtime guards in calibration_utils decide whether a correction is safe to
    apply for the current sample.
    """
    grouped = {
        "question_route_band": {},
        "question_route": {},
        "question": {},
        "route": {},
        "global": {"*": []},
    }
    for item in trial_records:
        max_score = max(safe_float(item.get("max_score"), 0.0), 1.0)
        cap = min(max_correction_points, max_correction_ratio * max_score)
        if cap <= 0:
            continue
        qid = str(item.get("qid", ""))
        route = str(item.get("trial_route", "UNKNOWN"))
        band = route_score_band(item.get("trial_score", item.get("avg", 0.0)), max_score)
        grouped["question_route_band"].setdefault(f"{qid}|{route}|{band}", []).append(item)
        grouped["question_route"].setdefault(f"{qid}|{route}", []).append(item)
        grouped["question"].setdefault(qid, []).append(item)
        grouped["route"].setdefault(route, []).append(item)
        grouped["global"]["*"].append(item)

    table = {}
    for group_name, cells in grouped.items():
        table[group_name] = {}
        for key, items in cells.items():
            if group_name != "global" and len(items) < min_cell_count:
                continue
            max_score = max(mean([safe_float(item.get("max_score"), 1.0) for item in items]), 1.0)
            max_correction = min(max_correction_points, max_correction_ratio * max_score)
            table[group_name][key] = _residual_entry(items, shrinkage_k, max_correction)

    return {
        "version": 1,
        "enabled": True,
        "method": "validation_residual_additive",
        "score_band": "low:<35%, mid:<70%, high:>=70%",
        "min_cell_count": int(min_cell_count),
        "shrinkage_k": float(shrinkage_k),
        "max_correction_ratio": float(max_correction_ratio),
        "max_correction_points": float(max_correction_points),
        "table": table,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate A3WA loss parameters and risk weights.")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--teacher-db", default="database/teacher_scores.json")
    parser.add_argument("--database-path", default="database/exam_database.json")
    parser.add_argument("--output", default="results_rrd_vlm/a3wa_calibration_config.json")
    parser.add_argument("--bnd-max", type=float, default=0.60)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--score-calibration", action="store_true", default=True)
    parser.add_argument("--no-score-calibration", dest="score_calibration", action="store_false")
    parser.add_argument("--min-cell-count", type=int, default=5)
    parser.add_argument("--shrinkage-k", type=float, default=8.0)
    parser.add_argument("--max-correction-ratio", type=float, default=0.12)
    parser.add_argument("--max-correction-points", type=float, default=2.0)
    args = parser.parse_args()

    teacher_db = load_json(args.teacher_db)
    question_scores = load_question_scores(args.database_path)
    input_files = expand_input_files(args.files)
    records = load_records(input_files, teacher_db, question_scores)
    if not records:
        raise SystemExit("No valid records found for calibration.")

    results = []
    for loss_params in candidate_loss_params():
        for weights in candidate_weights():
            results.append(evaluate_params(records, loss_params, weights, args.bnd_max))
    results.sort(key=lambda item: item["objective"])
    best = results[0]
    best_with_trial = evaluate_params(
        records,
        best["loss_params"],
        best["risk_weights"],
        args.bnd_max,
        return_trial=True,
    )
    alpha, beta = compute_a3wa_thresholds(**best["loss_params"])

    config = {
        "version": 1,
        "source": "scripts/calibrate_a3wa.py",
        "database_path": args.database_path,
        "teacher_db": args.teacher_db,
        "files": input_files,
        "selection_objective": "cost_sensitive_validation_calibration",
        "loss_params": best["loss_params"],
        "risk_weights": best["risk_weights"],
        "thresholds": {
            "alpha": round(alpha, 6),
            "beta": round(beta, 6),
        },
        "validation_summary": {
            "n": len(records),
            "objective": round(best["objective"], 6),
            "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in best["metrics"].items()},
            "baseline_metrics": {
                k: round(v, 6) if isinstance(v, float) else v for k, v in best["baseline_metrics"].items()
            },
            "route_counts": best["route_counts"],
            "bnd_ratio": round(best["bnd_ratio"], 6),
            "pos_mae": round(best["pos_mae"], 6),
            "bnd_mae": round(best["bnd_mae"], 6),
            "bnd_gain": round(best["bnd_gain"], 6),
        },
    }
    if args.score_calibration:
        config["score_calibration"] = build_score_calibration(
            best_with_trial["trial_records"],
            min_cell_count=args.min_cell_count,
            shrinkage_k=args.shrinkage_k,
            max_correction_ratio=args.max_correction_ratio,
            max_correction_points=args.max_correction_points,
        )
    write_json(args.output, config)

    print(f"Wrote {args.output}")
    for idx, item in enumerate(results[: args.top_k], start=1):
        print(
            f"#{idx} obj={item['objective']:.4f} "
            f"MAE={item['metrics']['mae']:.4f} base={item['baseline_metrics']['mae']:.4f} "
            f"TAR2={item['metrics']['tar2']:.1%} routes={item['route_counts']} "
            f"bnd_gain={item['bnd_gain']:+.4f} loss={item['loss_params']} weights={item['risk_weights']}"
        )


if __name__ == "__main__":
    main()
