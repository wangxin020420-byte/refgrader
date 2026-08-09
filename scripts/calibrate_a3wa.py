import argparse
import glob
import json
import math
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from calibration_utils import (  # noqa: E402
    apply_structured_boundary_action_policy,
    build_a3wa_decision,
    conformal_score_interval,
    compute_a3wa_thresholds,
    normalized_risk_weights,
    route_score_band,
    safe_float,
)
from model_runtime import runtime_model_config  # noqa: E402
from sample_quality import load_policy_for_data_path  # noqa: E402


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


def load_records(files, teacher_db, question_scores, sample_policy=None):
    records = []
    for path in files:
        qid = question_id_from_path(path)
        max_score = question_scores.get(qid, LEGACY_SCORES_MAP.get(qid, 20))
        for row in load_json(path):
            student_id = row.get("student_id", "")
            teacher = teacher_score(teacher_db, student_id, qid)
            if sample_policy is not None:
                teacher = sample_policy.effective_teacher_score(
                    qid, student_id, teacher
                )
            avg = row.get("selected_baseline_score", row.get("model_avg_score"))
            final = row.get("final_calibrated_score")
            if teacher is None or teacher < 0 or avg is None or final is None:
                continue
            risk = row.get("risk_features", {}) or {}
            post_calibration = row.get("post_calibration", {}) or {}
            primary_risks = (
                post_calibration.get("primary_risks")
                or post_calibration.get("three_way_primary_risks")
                or {}
            )
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
                "post_calibration": post_calibration,
                "U_E": safe_float(primary_risks.get("U_E", 0.0), 0.0),
                "U_S": safe_float(primary_risks.get("U_S", 0.0), 0.0),
                "U_R": safe_float(primary_risks.get("U_R", 0.0), 0.0),
                "agent_evidence": gate.get("agent_evidence_summary") or {},
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


def finite_sample_quantile(values, coverage):
    values = sorted(safe_float(value, 0.0) for value in values)
    if not values:
        return 0.0
    rank = int(math.ceil((len(values) + 1) * coverage)) - 1
    return values[max(0, min(rank, len(values) - 1))]


def fit_score_uncertainty(
    records,
    coverage=0.90,
    scale_floor=0.05,
    safe_tolerance_ratio=0.10,
):
    nonconformity = []
    for record in records:
        max_score = max(record["max_score"], 1.0)
        scores = record.get("model_scores") or []
        spread_ratio = (
            (max(scores) - min(scores)) / max_score if len(scores) >= 2 else 0.0
        )
        local_scale = max(spread_ratio, scale_floor)
        residual_ratio = abs(record["avg"] - record["teacher"]) / max_score
        nonconformity.append(residual_ratio / local_scale)
    quantile = finite_sample_quantile(nonconformity, coverage)
    covered = 0
    widths = []
    for record in records:
        max_score = max(record["max_score"], 1.0)
        scores = record.get("model_scores") or []
        spread_ratio = (
            (max(scores) - min(scores)) / max_score if len(scores) >= 2 else 0.0
        )
        half_width_ratio = quantile * max(spread_ratio, scale_floor)
        widths.append(2.0 * half_width_ratio)
        if abs(record["avg"] - record["teacher"]) / max_score <= half_width_ratio:
            covered += 1
    return {
        "version": 1,
        "enabled": True,
        "method": "locally_scaled_split_conformal",
        "coverage": round(coverage, 6),
        "calibration_n": len(records),
        "scale_floor": round(scale_floor, 6),
        "safe_tolerance_ratio": round(safe_tolerance_ratio, 6),
        "nonconformity_quantile": round(quantile, 6),
        "empirical_coverage": round(covered / max(len(records), 1), 6),
        "mean_interval_width_ratio": round(mean(widths), 6),
        "assumption": "exchangeable_validation_and_test_records",
    }


def membership_features(record):
    return [record["U_E"], record["U_S"], record["U_R"]]


def safe_auto_grade_label(record, safe_error_ratio, safe_error_points):
    tolerance = max(safe_error_points, safe_error_ratio * max(record["max_score"], 1.0))
    return 1.0 if abs(record["avg"] - record["teacher"]) <= tolerance else 0.0


def fit_monotonic_membership(
    records,
    safe_error_ratio=0.10,
    safe_error_points=0.50,
    l2=0.05,
    iterations=3000,
    learning_rate=0.05,
):
    features = [membership_features(record) for record in records]
    labels = [
        safe_auto_grade_label(record, safe_error_ratio, safe_error_points)
        for record in records
    ]
    positive_rate = min(0.999, max(0.001, mean(labels)))
    intercept = math.log(positive_rate / (1.0 - positive_rate))
    coefficients = [1.0, 1.0, 1.0]
    n = max(len(records), 1)
    for _ in range(iterations):
        grad_intercept = 0.0
        grad_coefficients = [0.0, 0.0, 0.0]
        for x, label in zip(features, labels):
            linear = intercept - sum(c * value for c, value in zip(coefficients, x))
            probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, linear))))
            error = probability - label
            grad_intercept += error
            for idx, value in enumerate(x):
                grad_coefficients[idx] += -error * value
        intercept -= learning_rate * grad_intercept / n
        for idx in range(3):
            gradient = grad_coefficients[idx] / n + l2 * coefficients[idx]
            coefficients[idx] = max(0.0, min(20.0, coefficients[idx] - learning_rate * gradient))

    probabilities = []
    for x in features:
        linear = intercept - sum(c * value for c, value in zip(coefficients, x))
        probabilities.append(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, linear)))))
    brier = mean([(probability - label) ** 2 for probability, label in zip(probabilities, labels)])
    return {
        "version": 1,
        "type": "monotonic_logistic",
        "target": "safe_auto_grading",
        "safe_error_ratio": round(safe_error_ratio, 6),
        "safe_error_points": round(safe_error_points, 6),
        "intercept": round(intercept, 8),
        "coefficients": {
            "U_E": round(coefficients[0], 8),
            "U_S": round(coefficients[1], 8),
            "U_R": round(coefficients[2], 8),
        },
        "constraints": "non_negative_risk_coefficients",
        "regularization_l2": round(l2, 6),
        "training_n": len(records),
        "positive_rate": round(positive_rate, 6),
        "brier_score": round(brier, 6),
    }


def candidate_loss_params():
    for m in [0.20, 0.30, 0.40, 0.50]:
        for positive_loss_ratio in [0.25, 0.50, 1.0, 2.0, 4.0]:
            for negative_loss_ratio in [0.25, 0.50, 1.0, 2.0, 4.0]:
                yield {
                    "lambda1": positive_loss_ratio,
                    "lambda2": 1.0,
                    "mu1": 1.0,
                    "mu2": negative_loss_ratio,
                    "m": m,
                }


def _percentile(values, probability):
    values = sorted(safe_float(value, 0.0) for value in values)
    if not values:
        return 0.0
    position = (len(values) - 1) * min(1.0, max(0.0, probability))
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _numeric_summary(values):
    values = [safe_float(value, 0.0) for value in values]
    return {
        "n": len(values),
        "mean": round(mean(values), 6),
        "p50": round(_percentile(values, 0.50), 6),
        "p90": round(_percentile(values, 0.90), 6),
        "min": round(min(values), 6) if values else 0.0,
        "max": round(max(values), 6) if values else 0.0,
    }


def summarize_sequential_outcomes(trial_records):
    """Separate A3WA routing cost from actual human-review demand."""
    route_counts = Counter()
    outcome_counts = Counter()
    human_review_count = 0
    bnd_auto_resolved = 0
    for row in trial_records:
        route = str(row.get("trial_route") or "")
        outcome = str(row.get("sequential_outcome") or "unknown")
        route_counts[route] += 1
        outcome_counts[outcome] += 1
        if bool(row.get("requires_human_review", False)):
            human_review_count += 1
        if route == "BND" and outcome in {
            "auto_adjusted",
            "auto_kept_after_review",
        }:
            bnd_auto_resolved += 1

    total = max(len(trial_records), 1)
    bnd_count = route_counts.get("BND", 0)
    return {
        "route_counts": dict(route_counts),
        "outcome_counts": dict(outcome_counts),
        "boundary_agent_count": bnd_count,
        "boundary_agent_ratio": round(bnd_count / total, 6),
        "bnd_auto_resolved_count": bnd_auto_resolved,
        "bnd_auto_resolved_rate": round(
            bnd_auto_resolved / max(bnd_count, 1), 6
        ),
        "human_review_count": human_review_count,
        "human_review_ratio": round(human_review_count / total, 6),
    }


def summarize_risk_distributions(trial_records):
    """Describe U_E/U_S/U_R separation without fitting on test data."""
    groups = {
        "safe_baseline": [],
        "unsafe_baseline": [],
        "route_POS": [],
        "route_BND": [],
        "route_NEG": [],
    }
    for row in trial_records:
        max_score = max(safe_float(row.get("max_score"), 0.0), 1.0)
        tolerance = max(
            safe_float(row.get("safe_error_points"), 0.5),
            safe_float(row.get("safe_error_ratio"), 0.1) * max_score,
        )
        baseline_error = abs(
            safe_float(row.get("avg"), 0.0)
            - safe_float(row.get("teacher"), 0.0)
        )
        safety_group = (
            "safe_baseline" if baseline_error <= tolerance else "unsafe_baseline"
        )
        groups[safety_group].append(row)
        route_group = f"route_{row.get('trial_route', '')}"
        if route_group in groups:
            groups[route_group].append(row)

    summary = {}
    for group_name, rows in groups.items():
        summary[group_name] = {
            "n": len(rows),
            "membership": _numeric_summary(
                row.get("trial_membership", 0.0) for row in rows
            ),
            "U_E": _numeric_summary(
                (row.get("trial_risk_components") or {}).get("U_E", 0.0)
                for row in rows
            ),
            "U_S": _numeric_summary(
                (row.get("trial_risk_components") or {}).get("U_S", 0.0)
                for row in rows
            ),
            "U_R": _numeric_summary(
                (row.get("trial_risk_components") or {}).get("U_R", 0.0)
                for row in rows
            ),
        }
    return summary


def _candidate_snapshot(item):
    alpha, beta = compute_a3wa_thresholds(**item["loss_params"])
    return {
        "alpha": round(alpha, 6),
        "beta": round(beta, 6),
        "loss_params": item["loss_params"],
        "objective": round(item["expected_system_cost"], 6),
        "mae": round(item["metrics"]["mae"], 6),
        "bnd_ratio": round(item["bnd_ratio"], 6),
        "neg_ratio": round(item["neg_ratio"], 6),
        "unsafe_pos_rate": round(item["unsafe_pos_rate"], 6),
        "bnd_gain": round(item["bnd_gain"], 6),
        "constraint_violations": item["constraint_violations"],
        "constraint_excess": round(item["constraint_excess"], 6),
        "constraint_status": item["constraint_status"],
        "route_counts": item["route_counts"],
    }


def build_candidate_diagnostics(results, top_k=20):
    """Persist the constrained search frontier for reproducible audits."""
    ordered = sorted(
        results,
        key=lambda item: (
            item["constraint_violations"],
            item["constraint_excess"],
            item["expected_system_cost"],
            item["metrics"]["mae"],
        ),
    )
    feasible = [item for item in ordered if item["constraint_violations"] == 0]
    bnd_budget_candidates = [
        item
        for item in ordered
        if item["constraint_status"].get("bnd_ratio_within_budget", False)
        and item["constraint_status"].get("neg_ratio_within_budget", False)
    ]
    safe_pos_candidates = [
        item
        for item in ordered
        if item["constraint_status"].get(
            "unsafe_pos_rate_within_budget", False
        )
        and item["constraint_status"].get("neg_ratio_within_budget", False)
    ]
    failures = Counter()
    for item in ordered:
        for name, passed in item["constraint_status"].items():
            if not passed:
                failures[name] += 1

    pareto = []
    for candidate in ordered:
        candidate_values = (
            candidate["metrics"]["mae"],
            candidate["bnd_ratio"],
            candidate["unsafe_pos_rate"],
        )
        dominated = False
        for other in ordered:
            if other is candidate:
                continue
            other_values = (
                other["metrics"]["mae"],
                other["bnd_ratio"],
                other["unsafe_pos_rate"],
            )
            if all(a <= b for a, b in zip(other_values, candidate_values)) and any(
                a < b for a, b in zip(other_values, candidate_values)
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(candidate)

    return {
        "candidate_count": len(ordered),
        "feasible_candidate_count": len(feasible),
        "has_feasible_candidate": bool(feasible),
        "constraint_failure_counts": dict(sorted(failures.items())),
        "best_candidate": _candidate_snapshot(ordered[0]) if ordered else None,
        "best_feasible_candidate": (
            _candidate_snapshot(feasible[0]) if feasible else None
        ),
        "feasibility_tradeoff": {
            "minimum_unsafe_pos_rate_within_bnd_budget": round(
                min(
                    (
                        item["unsafe_pos_rate"]
                        for item in bnd_budget_candidates
                    ),
                    default=0.0,
                ),
                6,
            ),
            "minimum_bnd_ratio_within_unsafe_pos_budget": round(
                min(
                    (item["bnd_ratio"] for item in safe_pos_candidates),
                    default=0.0,
                ),
                6,
            ),
            "bnd_budget_candidate_count": len(bnd_budget_candidates),
            "unsafe_pos_budget_candidate_count": len(safe_pos_candidates),
        },
        "top_candidates": [
            _candidate_snapshot(item) for item in ordered[: max(1, int(top_k))]
        ],
        "pareto_frontier": [_candidate_snapshot(item) for item in pareto],
    }


def apply_action_policy(record, route, boundary_policy=None, return_gate=False):
    avg = record["avg"]
    max_score = max(record["max_score"], 1.0)
    if route != "BND":
        sequential_outcome = "auto_accepted" if route == "POS" else "defer_human"
        gate = {
            "final_score": avg,
            "action": "not_bnd",
            "accepted": False,
            "gate_reason": "route_not_boundary",
            "sequential_outcome": sequential_outcome,
            "requires_human_review": route == "NEG",
        }
        return gate if return_gate else avg
    gate = apply_structured_boundary_action_policy(
        avg_model_score=avg,
        candidate_score=record["raw_candidate"],
        max_score=max_score,
        post_calibration=record["post_calibration"],
        agent_evidence=record.get("agent_evidence"),
        config=boundary_policy,
    )
    gate = dict(gate)
    gate["final_score"] = round(gate["final_score"], 2)
    return gate if return_gate else gate["final_score"]


def summarize_boundary_actions(trial_records):
    """Measure validation gain separately for every BND action."""
    grouped = {}
    for row in trial_records:
        if row.get("trial_route") != "BND":
            continue
        action = str(row.get("boundary_action") or "keep_baseline")
        baseline_error = abs(float(row["avg"]) - float(row["teacher"]))
        action_error = abs(float(row["trial_score"]) - float(row["teacher"]))
        gain = baseline_error - action_error
        grouped.setdefault(action, []).append(gain)
    summary = {}
    for action, gains in sorted(grouped.items()):
        summary[action] = {
            "n": len(gains),
            "mean_gain": mean(gains),
            "improved": sum(value > 1e-9 for value in gains),
            "unchanged": sum(abs(value) <= 1e-9 for value in gains),
            "worsened": sum(value < -1e-9 for value in gains),
        }
    return summary


def calibrate_boundary_action_gate(trial_records, min_count=3):
    """Enable each directional BND action only after positive validation gain."""
    stats = summarize_boundary_actions(trial_records)
    directions = {
        "raise": "accept_structured_raise",
        "lower": "accept_structured_lower",
    }
    result = {"minimum_count": int(min_count), "actions": stats}
    for direction, action in directions.items():
        item = stats.get(action, {})
        enabled = (
            int(item.get("n", 0)) >= int(min_count)
            and float(item.get("mean_gain", 0.0)) > 0.0
        )
        result[f"allow_{direction}"] = enabled
        result[f"{direction}_reason"] = (
            "positive_validation_gain"
            if enabled
            else "insufficient_or_nonpositive_validation_gain"
        )
    return result


def evaluate_params(
    records,
    loss_params,
    membership_model,
    score_uncertainty,
    bnd_max,
    neg_max,
    bnd_cost,
    neg_cost,
    unsafe_pos_cost,
    safe_error_ratio,
    safe_error_points,
    max_unsafe_pos_rate,
    boundary_policy=None,
    return_trial=False,
):
    trial = []
    route_counts = Counter()
    route_errors = {"POS": [], "BND": [], "NEG": []}
    bnd_gains = []
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
            weights=None,
            loss_params=loss_params,
            membership_model=membership_model,
            score_uncertainty=score_uncertainty,
        )
        route = decision["route"]
        gate = apply_action_policy(
            record, route, boundary_policy, return_gate=True
        )
        score = gate["final_score"]
        row = dict(record)
        row["trial_score"] = score
        row["trial_route"] = route
        row["trial_membership"] = decision["mu"]
        row["trial_risk_components"] = decision["risk_components"]
        row["safe_error_ratio"] = safe_error_ratio
        row["safe_error_points"] = safe_error_points
        row["boundary_action"] = gate.get("action")
        row["boundary_gate_reason"] = gate.get("gate_reason")
        row["sequential_outcome"] = gate.get("sequential_outcome")
        row["requires_human_review"] = bool(
            gate.get("requires_human_review", False)
        )
        trial.append(row)
        route_counts[route] += 1
        route_errors[route].append(abs(score - record["teacher"]))
        if route == "BND":
            bnd_gains.append(abs(record["avg"] - record["teacher"]) - abs(score - record["teacher"]))

    avg_metric = metrics(trial, "avg")
    final_metric = metrics(trial, "trial_score")
    bnd_ratio = route_counts["BND"] / max(len(trial), 1)
    neg_ratio = route_counts["NEG"] / max(len(trial), 1)
    pos_mae = mean(route_errors["POS"])
    bnd_mae = mean(route_errors["BND"])
    bnd_gain = mean(bnd_gains)
    unsafe_pos = 0
    normalized_score_loss = 0.0
    for row in trial:
        max_score = max(row["max_score"], 1.0)
        tolerance = max(safe_error_points, safe_error_ratio * max_score)
        if row["trial_route"] == "POS" and abs(row["avg"] - row["teacher"]) > tolerance:
            unsafe_pos += 1
        if row["trial_route"] != "NEG":
            normalized_score_loss += abs(row["trial_score"] - row["teacher"]) / max_score
    n = max(len(trial), 1)
    unsafe_pos_rate = unsafe_pos / max(route_counts["POS"], 1)
    expected_system_cost = (
        normalized_score_loss / n
        + bnd_cost * bnd_ratio
        + neg_cost * neg_ratio
        + unsafe_pos_cost * unsafe_pos / n
    )
    constraint_violations = (
        int(bnd_ratio > bnd_max)
        + int(neg_ratio > neg_max)
        + int(unsafe_pos_rate > max_unsafe_pos_rate)
    )
    constraint_excess = (
        max(0.0, bnd_ratio - bnd_max)
        + max(0.0, neg_ratio - neg_max)
        + max(0.0, unsafe_pos_rate - max_unsafe_pos_rate)
    )
    sequential_summary = summarize_sequential_outcomes(trial)
    result = {
        "objective": expected_system_cost,
        "expected_system_cost": expected_system_cost,
        "constraint_violations": constraint_violations,
        "constraint_excess": constraint_excess,
        "metrics": final_metric,
        "baseline_metrics": avg_metric,
        "route_counts": dict(route_counts),
        "bnd_ratio": bnd_ratio,
        "neg_ratio": neg_ratio,
        "unsafe_pos_count": unsafe_pos,
        "unsafe_pos_rate": unsafe_pos_rate,
        "pos_mae": pos_mae,
        "bnd_mae": bnd_mae,
        "bnd_gain": bnd_gain,
        "boundary_action_summary": summarize_boundary_actions(trial),
        "sequential_outcome_summary": sequential_summary,
        "constraint_status": {
            "bnd_ratio_within_budget": bnd_ratio <= bnd_max,
            "neg_ratio_within_budget": neg_ratio <= neg_max,
            "unsafe_pos_rate_within_budget": (
                unsafe_pos_rate <= max_unsafe_pos_rate
            ),
        },
        "loss_params": loss_params,
        "membership_model": membership_model,
        "score_uncertainty": score_uncertainty,
    }
    if return_trial:
        result["trial_records"] = trial
    return result


def _residual_entry(items, shrinkage_k, max_correction, min_stable_count=20):
    residuals = [item["teacher"] - item["trial_score"] for item in items]
    n = len(residuals)
    raw_mean = mean(residuals)
    variance = (
        sum((value - raw_mean) ** 2 for value in residuals) / (n - 1)
        if n >= 2 else 0.0
    )
    standard_error = math.sqrt(variance / n) if n > 0 else 0.0
    ci_low = raw_mean - 1.96 * standard_error
    ci_high = raw_mean + 1.96 * standard_error
    sign_stable = n >= min_stable_count and (ci_low > 0.0 or ci_high < 0.0)
    shrink = n / (n + max(shrinkage_k, 0.0)) if n > 0 else 0.0
    correction = (
        max(-max_correction, min(max_correction, raw_mean * shrink))
        if sign_stable else 0.0
    )
    return {
        "n": n,
        "mean_residual": round(raw_mean, 6),
        "standard_error": round(standard_error, 6),
        "ci95": [round(ci_low, 6), round(ci_high, 6)],
        "sign_stable": sign_stable,
        "correction": round(correction, 6),
    }


def build_score_calibration(
    trial_records,
    *,
    min_cell_count=20,
    shrinkage_k=8.0,
    max_correction_ratio=0.12,
    max_correction_points=2.0,
    direction_guard_min_count=3,
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
    diagnostics = {}
    for group_name, cells in grouped.items():
        table[group_name] = {}
        diagnostics[group_name] = {}
        for key, items in cells.items():
            max_score = max(mean([safe_float(item.get("max_score"), 1.0) for item in items]), 1.0)
            max_correction = min(max_correction_points, max_correction_ratio * max_score)
            diagnostic = _residual_entry(
                items,
                shrinkage_k,
                max_correction,
                min_stable_count=min_cell_count,
            )
            diagnostics[group_name][key] = diagnostic
            if group_name != "global" and len(items) < min_cell_count:
                continue
            table[group_name][key] = diagnostic

    return {
        "version": 3,
        "enabled": True,
        "method": "validation_residual_additive",
        "score_band": "low:<35%, mid:<70%, high:>=70%",
        "min_cell_count": int(min_cell_count),
        "shrinkage_k": float(shrinkage_k),
        "max_correction_ratio": float(max_correction_ratio),
        "max_correction_points": float(max_correction_points),
        "direction_guard_min_count": int(max(1, direction_guard_min_count)),
        "direction_guard_floor_ratio": 0.02,
        "table": table,
        "diagnostics": diagnostics,
    }


def select_best_candidate(results):
    return min(
        results,
        key=lambda item: (
            item["constraint_violations"],
            item["constraint_excess"],
            item["expected_system_cost"],
            item["metrics"]["mae"],
        ),
    )


def paired_bootstrap_normalized_mae_delta(records, iterations=2000, seed=20260714):
    deltas = [
        (
            abs(record["trial_score"] - record["teacher"])
            - abs(record["avg"] - record["teacher"])
        ) / max(record["max_score"], 1.0)
        for record in records
    ]
    if not deltas:
        return {"mean": 0.0, "ci95": [0.0, 0.0], "iterations": 0}
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        samples.append(mean([deltas[rng.randrange(len(deltas))] for _ in deltas]))
    samples.sort()
    low = samples[int(0.025 * (len(samples) - 1))]
    high = samples[int(0.975 * (len(samples) - 1))]
    return {
        "mean": round(mean(deltas), 6),
        "ci95": [round(low, 6), round(high, 6)],
        "iterations": iterations,
        "seed": seed,
    }


def leave_one_question_out_validation(records, calibration_args):
    qids = sorted(set(record["qid"] for record in records))
    if len(qids) < 2:
        return {
            "available": False,
            "reason": "at_least_two_questions_required",
            "folds": [],
        }
    all_trial = []
    folds = []
    for held_out in qids:
        train = [record for record in records if record["qid"] != held_out]
        test = [record for record in records if record["qid"] == held_out]
        uncertainty = fit_score_uncertainty(
            train,
            coverage=calibration_args["conformal_coverage"],
            scale_floor=calibration_args["conformal_scale_floor"],
            safe_tolerance_ratio=calibration_args["safe_error_ratio"],
        )
        membership = fit_monotonic_membership(
            train,
            safe_error_ratio=calibration_args["safe_error_ratio"],
            safe_error_points=calibration_args["safe_error_points"],
        )
        fold_boundary_policy = dict(calibration_args["boundary_policy"])
        fold_boundary_policy.update({
            "allow_raise": True,
            "allow_lower": True,
        })
        fold_boundary_policy.pop("action_validation", None)
        shared = {
            "membership_model": membership,
            "score_uncertainty": uncertainty,
            "bnd_max": calibration_args["bnd_max"],
            "neg_max": calibration_args["neg_max"],
            "bnd_cost": calibration_args["bnd_cost"],
            "neg_cost": calibration_args["neg_cost"],
            "unsafe_pos_cost": calibration_args["unsafe_pos_cost"],
            "safe_error_ratio": calibration_args["safe_error_ratio"],
            "safe_error_points": calibration_args["safe_error_points"],
            "max_unsafe_pos_rate": calibration_args["max_unsafe_pos_rate"],
            "boundary_policy": fold_boundary_policy,
        }
        train_results = [
            evaluate_params(train, loss_params=params, **shared)
            for params in candidate_loss_params()
        ]
        fold_best = select_best_candidate(train_results)
        fold_train_trial = evaluate_params(
            train,
            loss_params=fold_best["loss_params"],
            return_trial=True,
            **shared,
        )
        fold_action_gate = calibrate_boundary_action_gate(
            fold_train_trial["trial_records"],
            min_count=calibration_args.get("boundary_action_min_count", 3),
        )
        fold_boundary_policy.update({
            "version": 2,
            "allow_raise": fold_action_gate["allow_raise"],
            "allow_lower": fold_action_gate["allow_lower"],
            "action_validation": fold_action_gate,
        })
        # Re-select the fold thresholds after fitting the action gate only on
        # the fold's training questions. The held-out question remains unseen.
        train_results = [
            evaluate_params(train, loss_params=params, **shared)
            for params in candidate_loss_params()
        ]
        fold_best = select_best_candidate(train_results)
        held_out_result = evaluate_params(
            test,
            loss_params=fold_best["loss_params"],
            return_trial=True,
            **shared,
        )
        all_trial.extend(held_out_result["trial_records"])
        folds.append({
            "held_out_question": held_out,
            "train_n": len(train),
            "test_n": len(test),
            "loss_params": fold_best["loss_params"],
            "baseline_mae": round(held_out_result["baseline_metrics"]["mae"], 6),
            "candidate_mae": round(held_out_result["metrics"]["mae"], 6),
            "route_counts": held_out_result["route_counts"],
            "bnd_gain": round(held_out_result["bnd_gain"], 6),
            "boundary_action_gate": fold_action_gate,
            "sequential_outcome_summary": held_out_result[
                "sequential_outcome_summary"
            ],
        })
    bootstrap = paired_bootstrap_normalized_mae_delta(all_trial)
    improved_questions = sum(
        1 for fold in folds if fold["candidate_mae"] < fold["baseline_mae"]
    )
    noninferiority_margin = 0.01
    return {
        "available": True,
        "method": "leave_one_question_out",
        "n": len(all_trial),
        "folds": folds,
        "improved_questions": improved_questions,
        "normalized_mae_delta": bootstrap,
        "noninferiority_margin": noninferiority_margin,
        "noninferior": bootstrap["ci95"][1] <= noninferiority_margin,
        "superior": bootstrap["ci95"][1] < 0.0,
    }


def main():
    default_model_config = runtime_model_config()
    parser = argparse.ArgumentParser(description="Calibrate A3WA loss parameters and risk weights.")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--teacher-db", default="database/teacher_scores.json")
    parser.add_argument("--database-path", default="database/exam_database.json")
    parser.add_argument("--output", default="results_rrd_vlm/a3wa_calibration_config.json")
    parser.add_argument("--bnd-max", type=float, default=0.60)
    parser.add_argument("--neg-max", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--score-calibration", action="store_true", default=False)
    parser.add_argument("--no-score-calibration", dest="score_calibration", action="store_false")
    parser.add_argument("--min-cell-count", type=int, default=20)
    parser.add_argument("--direction-guard-min-count", type=int, default=3)
    parser.add_argument("--shrinkage-k", type=float, default=8.0)
    parser.add_argument("--max-correction-ratio", type=float, default=0.12)
    parser.add_argument("--max-correction-points", type=float, default=2.0)
    parser.add_argument("--conformal-coverage", type=float, default=0.90)
    parser.add_argument("--conformal-scale-floor", type=float, default=0.05)
    parser.add_argument("--safe-error-ratio", type=float, default=0.10)
    parser.add_argument("--safe-error-points", type=float, default=0.50)
    parser.add_argument("--bnd-review-cost", type=float, default=0.02)
    parser.add_argument("--neg-human-cost", type=float, default=0.10)
    parser.add_argument("--unsafe-pos-cost", type=float, default=1.00)
    parser.add_argument("--max-unsafe-pos-rate", type=float, default=0.10)
    parser.add_argument("--boundary-action-min-count", type=int, default=3)
    parser.add_argument(
        "--text-provider", default=default_model_config["text_provider"]
    )
    parser.add_argument(
        "--text-model", default=default_model_config["text_model"]
    )
    parser.add_argument(
        "--thinking-mode",
        choices=["enabled", "disabled"],
        default=default_model_config["text_thinking"],
    )
    parser.add_argument(
        "--vlm-provider", default=default_model_config["vlm_provider"]
    )
    parser.add_argument(
        "--vlm-model", default=default_model_config["vlm_model"]
    )
    args = parser.parse_args()

    probability_args = {
        "bnd_max": args.bnd_max,
        "neg_max": args.neg_max,
        "conformal_coverage": args.conformal_coverage,
        "conformal_scale_floor": args.conformal_scale_floor,
        "safe_error_ratio": args.safe_error_ratio,
        "max_correction_ratio": args.max_correction_ratio,
        "max_unsafe_pos_rate": args.max_unsafe_pos_rate,
    }
    for name, value in probability_args.items():
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    for name, value in {
        "safe_error_points": args.safe_error_points,
        "bnd_review_cost": args.bnd_review_cost,
        "neg_human_cost": args.neg_human_cost,
        "unsafe_pos_cost": args.unsafe_pos_cost,
        "max_correction_points": args.max_correction_points,
        "shrinkage_k": args.shrinkage_k,
    }.items():
        if value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.min_cell_count < 1:
        parser.error("--min-cell-count must be at least 1")
    if args.direction_guard_min_count < 1:
        parser.error("--direction-guard-min-count must be at least 1")
    if args.boundary_action_min_count < 1:
        parser.error("--boundary-action-min-count must be at least 1")

    teacher_db = load_json(args.teacher_db)
    sample_policy = load_policy_for_data_path(args.teacher_db)
    question_scores = load_question_scores(args.database_path)
    input_files = expand_input_files(args.files)
    records = load_records(
        input_files, teacher_db, question_scores, sample_policy
    )
    if not records:
        raise SystemExit("No valid records found for calibration.")

    score_uncertainty = fit_score_uncertainty(
        records,
        coverage=args.conformal_coverage,
        scale_floor=args.conformal_scale_floor,
        safe_tolerance_ratio=args.safe_error_ratio,
    )
    membership_model = fit_monotonic_membership(
        records,
        safe_error_ratio=args.safe_error_ratio,
        safe_error_points=args.safe_error_points,
    )
    boundary_policy = {
        "version": 1,
        "method": "structured_item_evidence",
        "min_evidence_confidence": 0.60,
        "auto_keep_confidence": 0.80,
        "max_adjustment_ratio": 0.20,
    }
    evaluation_args = {
        "membership_model": membership_model,
        "score_uncertainty": score_uncertainty,
        "bnd_max": args.bnd_max,
        "neg_max": args.neg_max,
        "bnd_cost": args.bnd_review_cost,
        "neg_cost": args.neg_human_cost,
        "unsafe_pos_cost": args.unsafe_pos_cost,
        "safe_error_ratio": args.safe_error_ratio,
        "safe_error_points": args.safe_error_points,
        "max_unsafe_pos_rate": args.max_unsafe_pos_rate,
        "boundary_policy": boundary_policy,
    }
    results = [
        evaluate_params(records, loss_params=loss_params, **evaluation_args)
        for loss_params in candidate_loss_params()
    ]
    results.sort(
        key=lambda item: (
            item["constraint_violations"],
            item["constraint_excess"],
            item["expected_system_cost"],
            item["metrics"]["mae"],
        )
    )
    best = results[0]
    best_with_trial = evaluate_params(
        records,
        loss_params=best["loss_params"],
        return_trial=True,
        **evaluation_args,
    )
    action_gate = calibrate_boundary_action_gate(
        best_with_trial["trial_records"],
        min_count=args.boundary_action_min_count,
    )
    boundary_policy.update({
        "version": 2,
        "allow_raise": action_gate["allow_raise"],
        "allow_lower": action_gate["allow_lower"],
        "action_validation": action_gate,
    })
    # Re-select thresholds after disabling any BND direction that did not
    # demonstrate positive validation gain.
    results = [
        evaluate_params(records, loss_params=loss_params, **evaluation_args)
        for loss_params in candidate_loss_params()
    ]
    results.sort(
        key=lambda item: (
            item["constraint_violations"],
            item["constraint_excess"],
            item["expected_system_cost"],
            item["metrics"]["mae"],
        )
    )
    best = results[0]
    best_with_trial = evaluate_params(
        records,
        loss_params=best["loss_params"],
        return_trial=True,
        **evaluation_args,
    )
    candidate_diagnostics = build_candidate_diagnostics(
        results,
        top_k=max(args.top_k, 20),
    )
    sequential_diagnostics = best_with_trial["sequential_outcome_summary"]
    risk_diagnostics = summarize_risk_distributions(
        best_with_trial["trial_records"]
    )
    cross_validation = leave_one_question_out_validation(
        records,
        {
            "conformal_coverage": args.conformal_coverage,
            "conformal_scale_floor": args.conformal_scale_floor,
            "safe_error_ratio": args.safe_error_ratio,
            "safe_error_points": args.safe_error_points,
            "max_unsafe_pos_rate": args.max_unsafe_pos_rate,
            "bnd_max": args.bnd_max,
            "neg_max": args.neg_max,
            "bnd_cost": args.bnd_review_cost,
            "neg_cost": args.neg_human_cost,
            "unsafe_pos_cost": args.unsafe_pos_cost,
            "boundary_policy": boundary_policy,
            "boundary_action_min_count": args.boundary_action_min_count,
        },
    )
    deployment_gate = {
        "passed": bool(
            cross_validation.get("available", False)
            and cross_validation.get("noninferior", False)
            and best["constraint_violations"] == 0
            and best["bnd_gain"] > 0.0
            and (
                not boundary_policy["allow_raise"]
                or best["boundary_action_summary"].get(
                    "accept_structured_raise", {}
                ).get("mean_gain", 0.0) > 0.0
            )
            and (
                not boundary_policy["allow_lower"]
                or best["boundary_action_summary"].get(
                    "accept_structured_lower", {}
                ).get("mean_gain", 0.0) > 0.0
            )
        ),
        "requirements": {
            "loqo_normalized_mae_noninferior": bool(cross_validation.get("noninferior", False)),
            "route_budget_constraints_met": best["constraint_violations"] == 0,
            "validation_bnd_gain_positive": best["bnd_gain"] > 0.0,
            "enabled_boundary_actions_positive": all(
                not boundary_policy[f"allow_{direction}"]
                or best["boundary_action_summary"].get(action, {}).get(
                    "mean_gain", 0.0
                ) > 0.0
                for direction, action in (
                    ("raise", "accept_structured_raise"),
                    ("lower", "accept_structured_lower"),
                )
            ),
        },
        "note": "A failed gate marks the config experimental; it does not tune on test data.",
    }
    alpha, beta = compute_a3wa_thresholds(**best["loss_params"])

    config = {
        "version": 2,
        "source": "scripts/calibrate_a3wa.py",
        "model_config": {
            "text_provider": args.text_provider,
            "text_model": args.text_model,
            "text_family": (
                "deepseek" if args.text_provider.startswith("deepseek") else "glm"
            ),
            "text_thinking": args.thinking_mode,
            "vlm_provider": args.vlm_provider,
            "vlm_model": args.vlm_model,
        },
        "database_path": args.database_path,
        "teacher_db": args.teacher_db,
        "sample_quality_policy": sample_policy.descriptor(),
        "files": input_files,
        "selection_objective": "constrained_expected_system_cost",
        "loss_params": best["loss_params"],
        "risk_weights": normalized_risk_weights(),
        "membership_model": membership_model,
        "score_uncertainty": score_uncertainty,
        "boundary_policy": boundary_policy,
        "operational_costs": {
            "normalized_bnd_review": args.bnd_review_cost,
            "normalized_neg_human_review": args.neg_human_cost,
            "unsafe_pos": args.unsafe_pos_cost,
            "bnd_max": args.bnd_max,
            "neg_max": args.neg_max,
            "max_unsafe_pos_rate": args.max_unsafe_pos_rate,
        },
        "cross_validation": cross_validation,
        "candidate_diagnostics": candidate_diagnostics,
        "risk_diagnostics": risk_diagnostics,
        "deployment_gate": deployment_gate,
        "thresholds": {
            "alpha": round(alpha, 6),
            "beta": round(beta, 6),
        },
        "validation_summary": {
            "n": len(records),
            "objective": round(best["objective"], 6),
            "expected_system_cost": round(best["expected_system_cost"], 6),
            "constraint_violations": best["constraint_violations"],
            "constraint_excess": round(best["constraint_excess"], 6),
            "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in best["metrics"].items()},
            "baseline_metrics": {
                k: round(v, 6) if isinstance(v, float) else v for k, v in best["baseline_metrics"].items()
            },
            "route_counts": best["route_counts"],
            "bnd_ratio": round(best["bnd_ratio"], 6),
            "neg_ratio": round(best["neg_ratio"], 6),
            "unsafe_pos_count": best["unsafe_pos_count"],
            "unsafe_pos_rate": round(best["unsafe_pos_rate"], 6),
            "pos_mae": round(best["pos_mae"], 6),
            "bnd_mae": round(best["bnd_mae"], 6),
            "bnd_gain": round(best["bnd_gain"], 6),
            "boundary_action_summary": best["boundary_action_summary"],
            "sequential_outcome_summary": sequential_diagnostics,
            "actual_human_review_count": sequential_diagnostics[
                "human_review_count"
            ],
            "actual_human_review_ratio": sequential_diagnostics[
                "human_review_ratio"
            ],
        },
    }
    config["score_calibration"] = {
        "version": 3,
        "enabled": False,
        "method": "disabled_core_ablation",
        "table": {},
        "diagnostics": {},
    }
    if args.score_calibration:
        config["score_calibration"] = build_score_calibration(
            best_with_trial["trial_records"],
            min_cell_count=args.min_cell_count,
            shrinkage_k=args.shrinkage_k,
            max_correction_ratio=args.max_correction_ratio,
            max_correction_points=args.max_correction_points,
            direction_guard_min_count=args.direction_guard_min_count,
        )
    write_json(args.output, config)

    print(f"Wrote {args.output}")
    for idx, item in enumerate(results[: args.top_k], start=1):
        print(
            f"#{idx} obj={item['objective']:.4f} "
            f"MAE={item['metrics']['mae']:.4f} base={item['baseline_metrics']['mae']:.4f} "
            f"TAR2={item['metrics']['tar2']:.1%} routes={item['route_counts']} "
            f"bnd_gain={item['bnd_gain']:+.4f} unsafe_pos={item['unsafe_pos_count']} "
            f"loss={item['loss_params']}"
        )


if __name__ == "__main__":
    main()
