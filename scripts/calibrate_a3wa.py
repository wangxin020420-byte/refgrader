import argparse
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
    safe_float,
)


SCORES_MAP = {"Q1": 5, "Q2": 20, "Q3": 10, "Q4": 20, "Q5": 15, "Q6": 20, "Q7": 10}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def teacher_score(db, student_id, qid):
    return db.get(str(student_id).split("_")[0], {}).get(qid)


def question_id_from_path(path):
    return os.path.basename(path).split("_")[0]


def load_records(files, teacher_db):
    records = []
    for path in files:
        qid = question_id_from_path(path)
        max_score = SCORES_MAP.get(qid, 20)
        for row in load_json(path):
            teacher = teacher_score(teacher_db, row.get("student_id", ""), qid)
            avg = row.get("model_avg_score")
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


def evaluate_params(records, loss_params, weights, bnd_max):
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
    return {
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


def main():
    parser = argparse.ArgumentParser(description="Calibrate A3WA loss parameters and risk weights.")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--teacher-db", default="database/teacher_scores.json")
    parser.add_argument("--output", default="results_rrd_vlm/a3wa_calibration_config.json")
    parser.add_argument("--bnd-max", type=float, default=0.60)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    teacher_db = load_json(args.teacher_db)
    records = load_records(args.files, teacher_db)
    if not records:
        raise SystemExit("No valid records found for calibration.")

    results = []
    for loss_params in candidate_loss_params():
        for weights in candidate_weights():
            results.append(evaluate_params(records, loss_params, weights, args.bnd_max))
    results.sort(key=lambda item: item["objective"])
    best = results[0]
    alpha, beta = compute_a3wa_thresholds(**best["loss_params"])

    config = {
        "version": 1,
        "source": "scripts/calibrate_a3wa.py",
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
