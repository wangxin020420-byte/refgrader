import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from calibration_utils import (  # noqa: E402
    a3wa_dynamic_bounds,
    apply_boundary_no_harm_gate,
    build_a3wa_decision,
    build_post_grading_calibration,
    parse_json_maybe,
    prepare_rubrics_for_calibration,
    safe_float,
)


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def pearson(xs, ys):
    if len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return 0.0
    return num / (den_x * den_y)


def quadratic_weighted_kappa(xs, ys, max_score):
    if not xs:
        return 0.0
    max_rating = int(round(max_score))
    xs_i = [max(0, min(max_rating, int(round(x)))) for x in xs]
    ys_i = [max(0, min(max_rating, int(round(y)))) for y in ys]
    n = len(xs_i)

    obs = [[0 for _ in range(max_rating + 1)] for _ in range(max_rating + 1)]
    hist_x = [0 for _ in range(max_rating + 1)]
    hist_y = [0 for _ in range(max_rating + 1)]
    for x, y in zip(xs_i, ys_i):
        obs[x][y] += 1
        hist_x[x] += 1
        hist_y[y] += 1

    observed = 0.0
    expected = 0.0
    denom = max(max_rating * max_rating, 1)
    for i in range(max_rating + 1):
        for j in range(max_rating + 1):
            weight = ((i - j) ** 2) / denom
            observed += weight * obs[i][j] / n
            expected += weight * (hist_x[i] * hist_y[j]) / (n * n)
    if expected <= 1e-12:
        return 1.0 if observed <= 1e-12 else 0.0
    return 1.0 - observed / expected


def metrics(rows, score_key, max_score):
    teachers = [r["teacher_score"] for r in rows]
    scores = [r[score_key] for r in rows]
    diffs = [s - t for s, t in zip(scores, teachers)]
    return {
        "n": len(rows),
        "mae": mean([abs(d) for d in diffs]),
        "rmse": math.sqrt(mean([d * d for d in diffs])),
        "pearson": pearson(teachers, scores),
        "qwk": quadratic_weighted_kappa(teachers, scores, max_score),
        "tar2": mean([1.0 if abs(d) <= 2 else 0.0 for d in diffs]),
        "bias": mean(diffs),
        "high_over": sum(1 for d in diffs if d > 2),
        "high_under": sum(1 for d in diffs if d < -2),
    }


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_question_id(path):
    name = os.path.basename(path)
    return name.split("_")[0]


def load_rubric(results_dir, qid):
    candidates = [
        os.path.join(results_dir, f"{qid}_rubric_standard.json"),
        os.path.join(results_dir, f"{qid}_rubric.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return prepare_rubrics_for_calibration(load_json(path))
    raise FileNotFoundError(f"Cannot find rubric for {qid} in {results_dir}")


def build_risk_profile_from_record(record):
    risk = dict(record.get("risk_features", {}) or {})
    for key in (
        "perception_risk",
        "uncertainty_index",
        "fatal_points_ratio",
        "high_blank_high_score",
        "lenient_review_signal",
        "reject_domain",
        "boundary_domain",
    ):
        if key in record:
            risk[key] = record.get(key)
    return {
        "perception_risk": safe_float(risk.get("perception_risk", 0.0)),
        "uncertainty_index": safe_float(risk.get("uncertainty_index", 0.0)),
        "fatal_points_ratio": safe_float(risk.get("fatal_points_ratio", 0.0)),
        "high_blank_high_score": bool(risk.get("high_blank_high_score", False)),
        "lenient_review_signal": bool(risk.get("lenient_review_signal", False)),
        "reject_domain": bool(risk.get("reject_domain", False)),
        "boundary_domain": bool(risk.get("boundary_domain", False)),
        "risk_features": risk,
    }


def replay_record(record, rubrics_data, max_score):
    facts = parse_json_maybe(record.get("facts", {}), {})
    strict_cots = record.get("strict_cots_all")
    if not strict_cots:
        strict_cot = record.get("strict_cot", {})
        strict_cots = [strict_cot] if strict_cot else []

    avg_model_score = safe_float(record.get("model_avg_score", record.get("final_calibrated_score", 0.0)))
    old_score = safe_float(record.get("final_calibrated_score", avg_model_score))
    blank_rate = safe_float(record.get("blank_rate", 0.0))
    risk_profile = build_risk_profile_from_record(record)
    risk_features = risk_profile.get("risk_features", {})
    model_scores = record.get("model_scores_history") or []
    low_quality_rate = safe_float(
        record.get("low_quality_extraction_rate", risk_features.get("low_quality_rate", 0.0)),
        0.0,
    )

    post = build_post_grading_calibration(
        facts_dict=facts,
        rubrics_data=rubrics_data,
        strict_cots=strict_cots,
        avg_model_score=avg_model_score,
        max_score=max_score,
        blank_rate=blank_rate,
        risk_profile=risk_profile,
    )
    risk_profile["risk_features"].update({
        "unsupported_match_points_ratio": post["unsupported_match_points_ratio"],
        "method_final_verified_ratio": post["method_final_verified_ratio"],
        "direct_awarded_ratio": post["direct_awarded_ratio"],
        "metadata_coverage": post["metadata_coverage"],
        "explicit_chain_coverage": post["explicit_chain_coverage"],
        "core_anchor_failed": post["core_anchor_failed"],
        "visual_blank_review": post["visual_blank_review"],
        "calibration_rule_hits": post["rule_hits"],
    })
    a3wa = build_a3wa_decision(
        model_scores=model_scores,
        avg_model_score=avg_model_score,
        std_dev=safe_float(record.get("std_dev", 0.0), 0.0),
        max_score=max_score,
        blank_rate=blank_rate,
        low_quality_rate=low_quality_rate,
        perception_failure_rate=safe_float(record.get("perception_failure_rate", 0.0), 0.0),
        extraction_quality=record.get("extraction_quality", ""),
        fatal_points_ratio=risk_profile.get("fatal_points_ratio", 0.0),
        high_blank_high_score=risk_profile.get("high_blank_high_score", False),
        post_calibration=post,
    )

    if a3wa["route"] == "NEG":
        replay_route = "NEG"
        replay_score = avg_model_score
    else:
        replay_route = a3wa["route"]
        if replay_route == "BND":
            lower, upper, _ = a3wa_dynamic_bounds(
                avg_model_score=avg_model_score,
                max_score=max_score,
                a3wa_decision=a3wa,
                risk_profile=risk_profile,
                post_calibration=post,
            )
            gate = apply_boundary_no_harm_gate(
                avg_model_score=avg_model_score,
                candidate_score=old_score,
                max_score=max_score,
                a3wa_decision=a3wa,
                risk_profile=risk_profile,
                post_calibration=post,
                lower_bound=lower,
                upper_bound=upper,
            )
            replay_score = round(clamp(gate["final_score"], 0.0, max_score), 2)
        else:
            replay_score = avg_model_score

    return {
        "student_id": record.get("student_id", ""),
        "teacher_score": safe_float(record.get("teacher_score", 0.0)),
        "old_score": old_score,
        "replay_score": replay_score,
        "old_route": record.get("3wd_route", ""),
        "replay_route": replay_route,
        "post_calibration": post,
        "a3wa_decision": a3wa,
    }


def print_metrics_line(label, metric):
    print(
        f"{label:<10} "
        f"N={metric['n']:>3} "
        f"MAE={metric['mae']:.3f} "
        f"RMSE={metric['rmse']:.3f} "
        f"QWK={metric['qwk']:.3f} "
        f"Pearson={metric['pearson']:.3f} "
        f"TAR2={metric['tar2']:.1%} "
        f"Bias={metric['bias']:+.3f} "
        f"Over>2={metric['high_over']} "
        f"Under>2={metric['high_under']}"
    )


def replay_file(path, results_dir):
    qid = infer_question_id(path)
    data = load_json(path)
    rubrics_data = load_rubric(results_dir, qid)
    max_score = sum(safe_float(item.get("points", 0.0)) for item in rubrics_data)

    rows = []
    rule_counts = Counter()
    route_counts = Counter()
    for record in data:
        if record.get("teacher_score") is None:
            continue
        row = replay_record(record, rubrics_data, max_score)
        rows.append(row)
        route_counts[(row["old_route"], row["replay_route"])] += 1
        for rule in row["post_calibration"].get("rule_hits", []):
            rule_counts[rule] += 1

    old_metric = metrics(rows, "old_score", max_score)
    replay_metric = metrics(rows, "replay_score", max_score)

    print(f"\n{qid} | max_score={max_score:g} | source={os.path.basename(path)}")
    print_metrics_line("current", old_metric)
    print_metrics_line("replay", replay_metric)
    print(f"rules: {dict(rule_counts)}")
    print(f"routes: {dict(route_counts)}")

    serious = sorted(
        rows,
        key=lambda r: abs((r["replay_score"] - r["teacher_score"]) - (r["old_score"] - r["teacher_score"])),
        reverse=True,
    )[:8]
    print("largest score changes:")
    for row in serious:
        delta = row["replay_score"] - row["old_score"]
        if abs(delta) <= 1e-9:
            continue
        print(
            f"  {row['student_id']}: teacher={row['teacher_score']:.1f} "
            f"old={row['old_score']:.1f} replay={row['replay_score']:.1f} "
            f"delta={delta:+.1f} route={row['replay_route']} "
            f"mu={row['a3wa_decision'].get('mu', 0):.3f} rules={row['post_calibration'].get('rule_hits', [])}"
        )
    return rows, max_score


def main():
    parser = argparse.ArgumentParser(description="Replay generic post-grading calibration without API calls.")
    parser.add_argument("--results-dir", default="results_rrd_vlm")
    parser.add_argument(
        "--files",
        nargs="*",
        help="Checkpoint/result JSON files. Defaults to *_grading_checkpoint.json in results-dir.",
    )
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        files = [
            os.path.join(args.results_dir, name)
            for name in os.listdir(args.results_dir)
            if name.endswith("_grading_checkpoint.json")
        ]
        files.sort()

    all_rows = []
    weighted_max = []
    for path in files:
        rows, max_score = replay_file(path, args.results_dir)
        all_rows.extend(rows)
        weighted_max.extend([max_score] * len(rows))

    if all_rows:
        global_max = max(weighted_max) if weighted_max else 100.0
        print("\nGLOBAL")
        print_metrics_line("current", metrics(all_rows, "old_score", global_max))
        print_metrics_line("replay", metrics(all_rows, "replay_score", global_max))


if __name__ == "__main__":
    main()
