"""
RefGrader evaluation script.

用法:
  # 使用默认题目集合进行最终三支决策分数评估。
  python evaluate.py

  # 指定某一种分数来源进行评估。支持以下简写:
  # single -> 第一次模型评分 single_first_score
  # avg -> 三次模型评分均分 model_avg_score
  # selected -> 三支决策选择后的基础分 selected_baseline_score
  # 3wd -> 最终三支决策分数 final_calibrated_score
  python evaluate.py --score-key 3wd
  python evaluate.py --score-key avg
  python evaluate.py --score-key selected
  python evaluate.py --score-key single

  # 只评估指定题目，例如只评估 Q6 和 Q7。
  python evaluate.py --questions Q6 Q7

  # 输出逐学生明细，并按绝对误差从大到小排序，便于定位异常样本。
  python evaluate.py --questions Q6 Q7 --detail

  # 在同一张表中对比 single / avg / selected / 3wd 四种分数形式。
  python evaluate.py --compare --questions Q4 Q5 Q6 Q7 --compare-score-keys single avg selected 3wd

  # 使用完整 checkpoint 口径评估，会纳入 NEG/rejected 样本，适合正式论文分析。
  python evaluate.py --result-source checkpoint --compare --questions Q2 Q3 Q4 Q5 --compare-score-keys single avg selected 3wd

  # 指定结果目录评估独立 run。rubric-dir 参数保留用于 run 目录扩展；当前评估指标不依赖 rubric 文件。
  python evaluate.py --results-dir results_runs/20260617_q4_verify --result-source checkpoint --questions Q4

  # 将逐学生的 single / avg / selected / 3wd 对比结果导出为 CSV，便于人工分析。
  python evaluate.py --compare --questions Q6 Q7 --compare-score-keys single avg selected 3wd --compare-output outputs/q6_q7_compare.csv
"""

import json
import argparse
import csv
import os
import numpy as np
from collections import Counter
from scipy import stats
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
except ModuleNotFoundError:
    def mean_absolute_error(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs(y_true - y_pred)))

    def mean_squared_error(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean((y_true - y_pred) ** 2))

    def cohen_kappa_score(y_true, y_pred, weights=None):
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        if not labels:
            return 0.0
        index = {label: i for i, label in enumerate(labels)}
        n_labels = len(labels)
        n = len(y_true)
        if n == 0:
            return 0.0

        observed = np.zeros((n_labels, n_labels), dtype=float)
        for t, p in zip(y_true, y_pred):
            observed[index[t], index[p]] += 1.0
        observed /= n

        hist_true = observed.sum(axis=1)
        hist_pred = observed.sum(axis=0)
        expected = np.outer(hist_true, hist_pred)

        if weights == "quadratic":
            denom = max((n_labels - 1) ** 2, 1)
            weight_matrix = np.fromfunction(
                lambda i, j: ((i - j) ** 2) / denom,
                (n_labels, n_labels),
                dtype=float,
            )
        elif weights == "linear":
            denom = max(n_labels - 1, 1)
            weight_matrix = np.fromfunction(
                lambda i, j: np.abs(i - j) / denom,
                (n_labels, n_labels),
                dtype=float,
            )
        else:
            weight_matrix = np.ones((n_labels, n_labels), dtype=float)
            np.fill_diagonal(weight_matrix, 0.0)

        observed_loss = float(np.sum(weight_matrix * observed))
        expected_loss = float(np.sum(weight_matrix * expected))
        if expected_loss <= 1e-12:
            return 1.0 if observed_loss <= 1e-12 else 0.0
        return 1.0 - observed_loss / expected_loss

SCORES_MAP = {"Q1": 5, "Q2": 20, "Q3": 10, "Q4": 20, "Q5": 15, "Q6": 20, "Q7": 10}
RESULTS_DIR = "./results_rrd_vlm"
RUBRIC_DIR = RESULTS_DIR
TEACHER_DB = "./database/teacher_scores.json"
DATABASE_PATH = "./database/exam_database.json"
RESULT_SOURCE = "graded"

SCORE_KEY_LABEL = {
    "final_calibrated_score": "3WD final score",
    "model_avg_score": "model average score",
    "selected_baseline_score": "3WD selected baseline score",
    "single_first_score": "single first score",
}

CMP_LABEL = {
    "single_first_score": "single",
    "model_avg_score": "avg",
    "selected_baseline_score": "selected",
    "final_calibrated_score": "3WD",
}

COMPARE_SCORE_KEYS = (
    "single_first_score",
    "model_avg_score",
    "selected_baseline_score",
    "final_calibrated_score",
)

SERIOUS_ERROR_THRESHOLD = 2.0
REVIEW_ROUTES = {"BND", "NEG"}

SCORE_KEY_ALIASES = {
    "single": "single_first_score",
    "single_first": "single_first_score",
    "single_first_score": "single_first_score",
    "avg": "model_avg_score",
    "average": "model_avg_score",
    "model_avg": "model_avg_score",
    "model_avg_score": "model_avg_score",
    "selected": "selected_baseline_score",
    "baseline": "selected_baseline_score",
    "selected_baseline": "selected_baseline_score",
    "selected_baseline_score": "selected_baseline_score",
    "3wd": "final_calibrated_score",
    "final": "final_calibrated_score",
    "final_calibrated": "final_calibrated_score",
    "final_calibrated_score": "final_calibrated_score",
}


def normalize_score_keys(raw_keys, default_keys=COMPARE_SCORE_KEYS):
    """Normalize score-key aliases and preserve user order without duplicates."""
    if not raw_keys:
        return tuple(default_keys)
    normalized = []
    invalid = []
    for raw_key in raw_keys:
        key = SCORE_KEY_ALIASES.get(str(raw_key).strip().lower())
        if key is None:
            invalid.append(str(raw_key))
            continue
        if key not in normalized:
            normalized.append(key)
    if invalid:
        valid = "single, avg, selected, 3wd"
        raise ValueError(f"Unsupported score key(s): {', '.join(invalid)}. Valid choices: {valid}")
    return tuple(normalized or default_keys)


def result_path_for(q_id):
    suffix = "grading_checkpoint" if RESULT_SOURCE == "checkpoint" else "graded_results"
    return os.path.join(RESULTS_DIR, f"{q_id}_{suffix}.json")


def failed_path_for(q_id):
    return os.path.join(RESULTS_DIR, f"{q_id}_failed.json")


def load_failed_records(q_id):
    path = failed_path_for(q_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def load_teacher_scores():
    with open(TEACHER_DB, "r", encoding="utf-8") as f:
        return json.load(f)


def get_teacher(ts_db, student_id, q_id):
    record = ts_db.get(student_id)
    if isinstance(record, dict) and q_id in record:
        return record.get(q_id)
    # Backward compatibility for legacy E12314093_Q2 result IDs.
    pure_id = student_id.split("_")[0]
    return ts_db.get(pure_id, {}).get(q_id, None)


def load_score_map(database_path):
    score_map = dict(SCORES_MAP)
    if not database_path or not os.path.exists(database_path):
        return score_map
    with open(database_path, "r", encoding="utf-8") as handle:
        questions = json.load(handle)
    if isinstance(questions, list):
        for question in questions:
            if not isinstance(question, dict) or not question.get("question_id"):
                continue
            score_map[str(question["question_id"])] = float(
                question.get("total_score", question.get("max_score", 20))
            )
    return score_map


def get_score_value(record, score_key):
    """Return a score value, including derived fields not stored directly in JSON."""
    if score_key == "single_first_score":
        history = record.get("model_scores_history") or []
        if isinstance(history, list) and history:
            return history[0]
        return None
    if score_key == "selected_baseline_score":
        return record.get("selected_baseline_score", None)
    return record.get(score_key, None)


def dw(s):
    """Display width: non-ASCII=2, ASCII=1."""
    return sum(2 if ord(c) > 127 else 1 for c in str(s))


def cpad(s, width):
    """Center-align text by display width."""
    s = str(s)
    total = width - dw(s)
    if total <= 0:
        return s
    left = total // 2
    return " " * left + s + " " * (total - left)


def make_line(col_widths, left="+", mid="+", right="+", fill="-"):
    return left + mid.join(fill * (w + 2) for w in col_widths) + right


def make_row(cells, col_widths, sep="|"):
    return sep + sep.join(" " + cpad(c, w) + " " for c, w in zip(cells, col_widths)) + sep


def print_closed_table(headers, rows, col_widths):
    """Print a closed table with centered cells."""
    print("  " + make_line(col_widths))
    print("  " + make_row(headers, col_widths))
    print("  " + make_line(col_widths, left="+", mid="+", right="+", fill="="))
    for row in rows:
        print("  " + make_row(row, col_widths))
    print("  " + make_line(col_widths))


def append_row_to_table(col_widths, cells):
    """Append a standalone row block."""
    print("  " + make_line(col_widths, left="+", mid="+", right="+", fill="-"))
    print("  " + make_row(cells, col_widths))
    print("  " + make_line(col_widths))


def compute_question_metrics(q_id, total_score, ts_db, score_key="final_calibrated_score"):
    """Compute metrics for one question and return (metrics, details) or None."""
    path = result_path_for(q_id)
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

    details = []
    for r in results:
        sid = r.get("student_id", "")
        t = get_teacher(ts_db, sid, q_id)
        if t is None or t < 0:
            continue
        m = get_score_value(r, score_key)
        if m is None:
            continue
        m = round(float(m), 2)
        diff = round(m - t, 2)

        facts_raw = r.get("facts", {})
        if isinstance(facts_raw, str):
            try:
                facts = json.loads(facts_raw)
            except Exception:
                facts = {}
        else:
            facts = facts_raw if isinstance(facts_raw, dict) else {}
        blank_markers = ("未", "无", "空", "blank", "not written", "unwritten")
        unwritten = sum(
            1 for v in facts.values()
            if str(v).strip() == "" or any(marker in str(v).lower() for marker in blank_markers)
        )

        details.append({
            "sid": sid.split("_")[0],
            "teacher": t, "model": m, "diff": diff,
            "route": r.get("3wd_route", ""),
            "blank_rate": r.get("blank_rate", 0),
            "single": get_score_value(r, "single_first_score"),
            "avg": r.get("model_avg_score", None),
            "selected": get_score_value(r, "selected_baseline_score"),
            "std": r.get("std_dev", 0),
            "total_items": len(facts),
            "unwritten": unwritten,
            "facts": facts,
        })

    if not details:
        return None

    t_arr = np.array([d["teacher"] for d in details])
    m_arr = np.array([d["model"] for d in details])
    n = len(t_arr)

    mae = mean_absolute_error(t_arr, m_arr)
    rmse = np.sqrt(mean_squared_error(t_arr, m_arr))
    pearson_r, pearson_p = stats.pearsonr(t_arr, m_arr) if n >= 3 else (0.0, 1.0)
    t_int = np.clip(np.round(t_arr), 0, total_score).astype(int)
    m_int = np.clip(np.round(m_arr), 0, total_score).astype(int)
    qwk = cohen_kappa_score(t_int, m_int, weights="quadratic")
    tar2 = float(np.mean(np.abs(t_arr - m_arr) <= 2))

    serious = [d for d in details if abs(d["diff"]) > 2]
    high_over = sum(1 for d in details if d["diff"] > 2)
    high_under = sum(1 for d in details if d["diff"] < -2)
    route_counts = Counter(d["route"] for d in details)
    risk_lookup = {
        r.get("student_id", "").split("_")[0]: r.get("risk_features", {})
        for r in results
    }
    risk_counts = Counter(
        (
            "NEG" if risk_lookup.get(d["sid"], {}).get("reject_domain") else
            ("BND" if risk_lookup.get(d["sid"], {}).get("boundary_domain") else "POS")
        )
        for d in details
    )

    metrics = {
        "q_id": q_id, "n": n, "total": total_score,
        "MAE": mae, "RMSE": rmse, "QWK": qwk,
        "Pearson": pearson_r, "Pearson_p": pearson_p, "TAR2": tar2,
        "SER2": float(len(serious) / n) if n else 0.0,
        "teacher_mean": float(np.mean(t_arr)),
        "model_mean": float(np.mean(m_arr)),
        "bias": float(np.mean(m_arr - t_arr)),
        "serious": len(serious), "high_over": high_over, "high_under": high_under,
        "route_counts": dict(route_counts),
        "risk_counts": dict(risk_counts),
    }
    return metrics, details


def export_single_avg_3wd_csv(questions, ts_db, output_path):
    """Export per-student comparison among single, avg, selected baseline, and 3WD."""
    rows = []
    for q_id in questions:
        path = result_path_for(q_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except FileNotFoundError:
            continue

        for r in results:
            sid = r.get("student_id", "")
            teacher = get_teacher(ts_db, sid, q_id)
            if teacher is None or teacher < 0:
                continue

            single = get_score_value(r, "single_first_score")
            avg = get_score_value(r, "model_avg_score")
            selected = get_score_value(r, "selected_baseline_score")
            final = get_score_value(r, "final_calibrated_score")
            if single is None or avg is None or selected is None or final is None:
                continue

            teacher = float(teacher)
            single = round(float(single), 2)
            avg = round(float(avg), 2)
            selected = round(float(selected), 2)
            final = round(float(final), 2)
            single_abs = abs(single - teacher)
            avg_abs = abs(avg - teacher)
            selected_abs = abs(selected - teacher)
            final_abs = abs(final - teacher)
            gate = r.get("boundary_gate") or {}
            risk_features = r.get("risk_features") or {}
            post_calibration = r.get("post_calibration") or {}
            primary_risks = (
                post_calibration.get("primary_risks")
                or post_calibration.get("three_way_primary_risks")
                or {}
            )
            a3wa_decision = r.get("a3wa_decision") or {}
            risk_components = a3wa_decision.get("risk_components") or {}
            score_calibration = r.get("score_calibration") or {}
            rows.append({
                "question": q_id,
                "student_id": sid,
                "teacher": teacher,
                "single_first_score": single,
                "model_avg_score": avg,
                "selected_baseline_score": selected,
                "final_calibrated_score": final,
                "single_diff": round(single - teacher, 2),
                "avg_diff": round(avg - teacher, 2),
                "selected_diff": round(selected - teacher, 2),
                "final_diff": round(final - teacher, 2),
                "single_abs_error": round(single_abs, 2),
                "avg_abs_error": round(avg_abs, 2),
                "selected_abs_error": round(selected_abs, 2),
                "final_abs_error": round(final_abs, 2),
                "avg_gain_vs_single": round(single_abs - avg_abs, 2),
                "selected_gain_vs_avg": round(avg_abs - selected_abs, 2),
                "final_gain_vs_selected": round(selected_abs - final_abs, 2),
                "final_gain_vs_avg": round(avg_abs - final_abs, 2),
                "final_gain_vs_single": round(single_abs - final_abs, 2),
                "route": r.get("3wd_route", ""),
                "baseline_serious_error": selected_abs > SERIOUS_ERROR_THRESHOLD,
                "risk_captured_by_route": (
                    selected_abs > SERIOUS_ERROR_THRESHOLD
                    and r.get("3wd_route", "") in REVIEW_ROUTES
                ),
                "safe_pos": (
                    r.get("3wd_route", "") == "POS"
                    and final_abs <= SERIOUS_ERROR_THRESHOLD
                ),
                "task_type": post_calibration.get("task_type", risk_features.get("task_type", "")),
                "complex_derivation_task": post_calibration.get(
                    "complex_derivation_task",
                    risk_features.get("complex_derivation_task", ""),
                ),
                "upper_consensus_eligible": post_calibration.get(
                    "upper_consensus_eligible",
                    risk_features.get("upper_consensus_eligible", ""),
                ),
                "baseline_policy": r.get("baseline_policy", ""),
                "baseline_score_source": r.get("baseline_score_source", ""),
                "std_dev": r.get("std_dev", ""),
                "blank_rate": r.get("blank_rate", ""),
                "U_E": primary_risks.get("U_E", risk_features.get("U_E", risk_components.get("U_E", ""))),
                "U_S": primary_risks.get("U_S", risk_features.get("U_S", risk_components.get("U_S", ""))),
                "U_R": primary_risks.get("U_R", risk_features.get("U_R", risk_components.get("U_R", ""))),
                "primary_risk": primary_risks.get("risk", risk_features.get("primary_risk", "")),
                "primary_mu": primary_risks.get("mu", risk_features.get("primary_mu", "")),
                "boundary_action": gate.get("action", ""),
                "score_calibration_applied": score_calibration.get("applied", ""),
                "score_calibration_correction": score_calibration.get("correction", ""),
                "score_calibration_reason": score_calibration.get("reason", ""),
                "score_calibration_lookup": score_calibration.get("lookup_key", ""),
            })

    if not rows:
        return 0

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fieldnames = [
        "question", "student_id", "teacher",
        "single_first_score", "model_avg_score", "selected_baseline_score", "final_calibrated_score",
        "single_diff", "avg_diff", "selected_diff", "final_diff",
        "single_abs_error", "avg_abs_error", "selected_abs_error", "final_abs_error",
        "avg_gain_vs_single", "selected_gain_vs_avg", "final_gain_vs_selected",
        "final_gain_vs_avg", "final_gain_vs_single",
        "route", "baseline_serious_error", "risk_captured_by_route", "safe_pos",
        "task_type", "complex_derivation_task", "upper_consensus_eligible",
        "baseline_policy", "baseline_score_source", "std_dev", "blank_rate",
        "U_E", "U_S", "U_R", "primary_risk", "primary_mu", "boundary_action",
        "score_calibration_applied", "score_calibration_correction",
        "score_calibration_reason", "score_calibration_lookup",
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_comparison_summary(questions, score_keys, ts_db, score_map):
    """Return machine-readable per-question and global comparison metrics."""
    per_question = []
    three_way_audit = []
    global_values = {key: ([], []) for key in score_keys}
    for q_id in questions:
        total = score_map.get(q_id, 20)
        for key in score_keys:
            computed = compute_question_metrics(
                q_id, total, ts_db, score_key=key
            )
            if computed is None:
                continue
            metrics, _ = computed
            per_question.append({
                "score_key": key,
                "score_type": CMP_LABEL.get(key, SCORE_KEY_LABEL.get(key, key)),
                **{
                    name: (
                        value.item()
                        if isinstance(value, np.generic)
                        else value
                    )
                    for name, value in metrics.items()
                },
            })

        try:
            with open(result_path_for(q_id), "r", encoding="utf-8") as handle:
                records = json.load(handle)
        except FileNotFoundError:
            continue
        audit_rows = []
        for record in records:
            teacher = get_teacher(ts_db, record.get("student_id", ""), q_id)
            if teacher is None or teacher < 0:
                continue
            for key in score_keys:
                score = get_score_value(record, key)
                if score is not None:
                    global_values[key][0].append(float(teacher))
                    global_values[key][1].append(round(float(score), 2))
            selected = get_score_value(record, "selected_baseline_score")
            final = get_score_value(record, "final_calibrated_score")
            if selected is not None and final is not None:
                selected_error = abs(float(selected) - float(teacher))
                final_error = abs(float(final) - float(teacher))
                audit_rows.append({
                    "route": record.get("3wd_route", ""),
                    "selected_error": selected_error,
                    "final_error": final_error,
                    "gain": selected_error - final_error,
                })

        if audit_rows:
            serious = [
                row for row in audit_rows
                if row["selected_error"] > SERIOUS_ERROR_THRESHOLD
            ]
            risk_captured = [
                row for row in serious if row["route"] in REVIEW_ROUTES
            ]
            pos = [row for row in audit_rows if row["route"] == "POS"]
            bnd = [row for row in audit_rows if row["route"] == "BND"]
            three_way_audit.append({
                "q_id": q_id,
                "n": len(audit_rows),
                "route_counts": dict(Counter(row["route"] for row in audit_rows)),
                "pos_coverage": len(pos) / len(audit_rows),
                "safe_pos_rate": (
                    sum(
                        row["final_error"] <= SERIOUS_ERROR_THRESHOLD
                        for row in pos
                    ) / len(pos)
                    if pos else None
                ),
                "baseline_serious_errors": len(serious),
                "risk_recall": (
                    len(risk_captured) / len(serious) if serious else None
                ),
                "bnd_n": len(bnd),
                "bnd_mean_gain": (
                    float(np.mean([row["gain"] for row in bnd]))
                    if bnd else None
                ),
                "bnd_improved": sum(row["gain"] > 1e-9 for row in bnd),
                "bnd_unchanged": sum(abs(row["gain"]) <= 1e-9 for row in bnd),
                "bnd_worsened": sum(row["gain"] < -1e-9 for row in bnd),
            })

    global_metrics = []
    for key in score_keys:
        teacher, predicted = global_values[key]
        if len(teacher) < 3:
            continue
        teacher_array = np.array(teacher)
        predicted_array = np.array(predicted)
        pearson, _ = stats.pearsonr(teacher_array, predicted_array)
        errors = predicted_array - teacher_array
        global_metrics.append({
            "score_key": key,
            "score_type": CMP_LABEL.get(key, SCORE_KEY_LABEL.get(key, key)),
            "q_id": "GLOBAL",
            "n": len(teacher),
            "MAE": float(mean_absolute_error(teacher_array, predicted_array)),
            "RMSE": float(np.sqrt(mean_squared_error(teacher_array, predicted_array))),
            "QWK": None,
            "Pearson": float(pearson),
            "TAR2": float(np.mean(np.abs(errors) <= 2)),
            "SER2": float(np.mean(np.abs(errors) > 2)),
            "bias": float(np.mean(errors)),
        })
    return {
        "schema_version": 1,
        "questions": list(questions),
        "result_source": RESULT_SOURCE,
        "score_types": [
            CMP_LABEL.get(key, SCORE_KEY_LABEL.get(key, key))
            for key in score_keys
        ],
        "per_question": per_question,
        "global": global_metrics,
        "three_way_audit": three_way_audit,
    }


def _format_rate(numerator, denominator):
    if denominator <= 0:
        return "--"
    return f"{numerator / denominator:.1%}"


def _format_mean(values, precision=3, signed=True):
    if not values:
        return "--"
    fmt = f"{{:{'+' if signed else ''}.{precision}f}}"
    return fmt.format(float(np.mean(values)))


def collect_3wd_mechanism_records(questions, ts_db):
    """Collect records needed to evaluate 3WD routing and BND correction."""
    records = []
    for q_id in questions:
        path = result_path_for(q_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except FileNotFoundError:
            continue

        for r in results:
            sid = r.get("student_id", "")
            teacher = get_teacher(ts_db, sid, q_id)
            baseline = get_score_value(r, "selected_baseline_score")
            final = get_score_value(r, "final_calibrated_score")
            if teacher is None or teacher < 0 or baseline is None or final is None:
                continue
            teacher = float(teacher)
            baseline = float(baseline)
            final = float(final)
            records.append({
                "q_id": q_id,
                "sid": sid,
                "teacher": teacher,
                "baseline": baseline,
                "final": final,
                "route": r.get("3wd_route", ""),
            })
    return records


def summarize_3wd_mechanism(records):
    """Summarize route safety, serious-error capture, and BND correction gain."""
    n = len(records)
    pos_items = [r for r in records if r["route"] == "POS"]
    review_items = [r for r in records if r["route"] in REVIEW_ROUTES]
    serious_items = [
        r for r in records
        if abs(r["baseline"] - r["teacher"]) > SERIOUS_ERROR_THRESHOLD
    ]
    captured_items = [r for r in serious_items if r["route"] in REVIEW_ROUTES]
    safe_pos_items = [
        r for r in pos_items
        if abs(r["final"] - r["teacher"]) <= SERIOUS_ERROR_THRESHOLD
    ]
    bnd_items = [r for r in records if r["route"] == "BND"]
    bnd_gains = [
        abs(r["baseline"] - r["teacher"]) - abs(r["final"] - r["teacher"])
        for r in bnd_items
    ]

    return {
        "n": n,
        "pos": len(pos_items),
        "review": len(review_items),
        "serious": len(serious_items),
        "captured": len(captured_items),
        "safe_pos": len(safe_pos_items),
        "bnd": len(bnd_items),
        "bnd_gains": bnd_gains,
        "bnd_improved": sum(1 for g in bnd_gains if g > 1e-9),
        "bnd_worsened": sum(1 for g in bnd_gains if g < -1e-9),
        "bnd_same": sum(1 for g in bnd_gains if abs(g) <= 1e-9),
    }


def print_3wd_mechanism_summary(questions, ts_db):
    """Print compact mechanism metrics for the 3WD route itself."""
    all_records = collect_3wd_mechanism_records(questions, ts_db)
    if not all_records:
        return

    print()
    print("  3WD mechanism summary | baseline=selected_baseline_score | serious error: abs(error) > 2")
    print()

    headers = [
        "Q", "N", "POS cov", "Safe POS", "Risk recall",
        "serious", "BND N", "BND gain", "BND +/0/-",
    ]
    rows = []
    for q_id in questions:
        q_records = [r for r in all_records if r["q_id"] == q_id]
        if not q_records:
            continue
        mt = summarize_3wd_mechanism(q_records)
        rows.append([
            q_id,
            str(mt["n"]),
            _format_rate(mt["pos"], mt["n"]),
            _format_rate(mt["safe_pos"], mt["pos"]),
            _format_rate(mt["captured"], mt["serious"]),
            f"{mt['captured']}/{mt['serious']}",
            str(mt["bnd"]),
            _format_mean(mt["bnd_gains"]),
            f"{mt['bnd_improved']}/{mt['bnd_same']}/{mt['bnd_worsened']}",
        ])

    if len(questions) > 1:
        mt = summarize_3wd_mechanism(all_records)
        rows.append([
            "GLOBAL",
            str(mt["n"]),
            _format_rate(mt["pos"], mt["n"]),
            _format_rate(mt["safe_pos"], mt["pos"]),
            _format_rate(mt["captured"], mt["serious"]),
            f"{mt['captured']}/{mt['serious']}",
            str(mt["bnd"]),
            _format_mean(mt["bnd_gains"]),
            f"{mt['bnd_improved']}/{mt['bnd_same']}/{mt['bnd_worsened']}",
        ])

    if rows:
        print_closed_table(headers=headers, rows=rows, col_widths=[7, 5, 9, 10, 12, 9, 7, 10, 10])
        print()
        print("  Notes:")
        print("    POS cov = POS routed samples / valid samples")
        print("    Safe POS = POS samples with abs(final - teacher) <= 2 / POS samples")
        print("    Risk recall = baseline serious-error samples routed to BND or NEG / baseline serious-error samples")
        print("    BND gain = mean(abs(selected baseline - teacher) - abs(final - teacher)) on BND samples")


def print_boundary_gain_audit(questions, ts_db):
    """Audit whether final 3WD corrections improve over the selected baseline."""
    print()
    print("  3WD gain audit | final_calibrated_score vs selected_baseline_score")
    print()
    headers = ["Q", "route", "N", "improved", "worsened", "same", "mean_gain", "mean_delta"]
    rows = []
    top_worsened = []

    for q_id in questions:
        path = result_path_for(q_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except FileNotFoundError:
            continue

        grouped = {}
        for r in results:
            sid = r.get("student_id", "")
            teacher = get_teacher(ts_db, sid, q_id)
            avg = r.get("model_avg_score", None)
            baseline = get_score_value(r, "selected_baseline_score")
            final = r.get("final_calibrated_score", None)
            if teacher is None or teacher < 0 or baseline is None or final is None:
                continue
            teacher = float(teacher)
            avg = float(avg) if avg is not None else float(baseline)
            baseline = float(baseline)
            final = float(final)
            route = r.get("3wd_route", "")
            gain = abs(baseline - teacher) - abs(final - teacher)
            delta = final - baseline
            grouped.setdefault(route, []).append((sid, teacher, avg, baseline, final, gain, delta, r))
            top_worsened.append((gain, q_id, route, sid, teacher, avg, baseline, final, r))

        for route in sorted(grouped):
            items = grouped[route]
            gains = [x[5] for x in items]
            deltas = [x[6] for x in items]
            rows.append([
                q_id,
                route,
                str(len(items)),
                str(sum(1 for g in gains if g > 1e-9)),
                str(sum(1 for g in gains if g < -1e-9)),
                str(sum(1 for g in gains if abs(g) <= 1e-9)),
                f"{np.mean(gains):+.3f}",
                f"{np.mean(deltas):+.3f}",
            ])

    if rows:
        print_closed_table(headers=headers, rows=rows, col_widths=[6, 8, 5, 10, 10, 6, 11, 11])

    worsened = [x for x in sorted(top_worsened, key=lambda v: v[0]) if x[0] < -1e-9]
    if worsened:
        print()
        print("  Top worsened by 3WD correction")
        detail_rows = []
        for gain, q_id, route, sid, teacher, avg, baseline, final, r in worsened[:10]:
            gate = r.get("boundary_gate") or {}
            detail_rows.append([
                q_id,
                sid,
                route,
                f"{teacher:.1f}",
                f"{avg:.1f}",
                f"{baseline:.1f}",
                f"{final:.1f}",
                f"{gain:+.1f}",
                str(gate.get("action", ""))[:18],
            ])
        print_closed_table(
            headers=["Q", "sid", "route", "teacher", "avg", "selected", "final", "gain", "gate"],
            rows=detail_rows,
            col_widths=[4, 13, 7, 8, 7, 9, 7, 7, 18],
        )


def evaluate_question(q_id, total_score, ts_db, score_key="final_calibrated_score", show_detail=False):
    result = compute_question_metrics(q_id, total_score, ts_db, score_key)
    if result is None:
        print(f"  {q_id}: no valid data")
        return None

    metrics, details = result
    label = SCORE_KEY_LABEL.get(score_key, score_key)

    print()
    print()
    print(f"  {q_id}  |  max score {total_score}  |  N = {metrics['n']}  |  {label}")
    print()

    metric_w = [10, 10, 10, 10, 12, 12]
    print_closed_table(
        headers=["MAE", "RMSE", "QWK", "Pearson r", "TAR(2)", "SER(>2)"],
        rows=[[f"{metrics['MAE']:.4f}", f"{metrics['RMSE']:.4f}",
                f"{metrics['QWK']:.4f}", f"{metrics['Pearson']:.4f}",
                f"{metrics['TAR2']:.1%}", f"{metrics['SER2']:.1%}"]],
        col_widths=metric_w,
    )

    print()
    stat_w = [10, 10, 10, 10, 14, 12]
    print_closed_table(
        headers=["teacher", "model", "bias", "Pearson p", "abs diff >2", "over / under"],
        rows=[[f"{metrics['teacher_mean']:.2f}", f"{metrics['model_mean']:.2f}",
               f"{metrics['bias']:+.2f}", f"{metrics['Pearson_p']:.1e}",
               f"{metrics['serious']}", f"{metrics['high_over']} / {metrics['high_under']}"]],
        col_widths=stat_w,
    )

    route_counts = metrics.get("route_counts", {})
    if route_counts:
        print()
        print("  Route distribution: " + " | ".join(f"{k}={v}" for k, v in sorted(route_counts.items())))

    failed_records = load_failed_records(q_id)
    if failed_records:
        print(f"  Failed records: {len(failed_records)} (see {failed_path_for(q_id)})")

    if show_detail:
        print()
        det_w = [13, 6, 6, 7, 7, 6, 7, 7, 6]
        print_closed_table(
            headers=["sid", "teacher", "model", "avg", "diff", "route", "blank", "unwritten", "flag"],
            rows=[
                [d["sid"],
                 f"{d['teacher']:.0f}", f"{d['model']:.1f}",
                 (f"{d['avg']:.1f}" if d["avg"] is not None else "-"),
                 f"{d['diff']:+.1f}", d["route"],
                 f"{d['blank_rate']:.0%}", str(d["unwritten"]),
                 (">>>" if abs(d["diff"]) > 2 else (">>" if abs(d["diff"]) > 1 else ""))]
                for d in sorted(details, key=lambda x: abs(x["diff"]), reverse=True)
            ],
            col_widths=det_w,
        )

    return metrics


def main():
    global RESULT_SOURCE, RESULTS_DIR, RUBRIC_DIR, TEACHER_DB, DATABASE_PATH

    parser = argparse.ArgumentParser(description="RefGrader evaluation script")
    parser.add_argument("--questions", nargs="+", default=["Q4", "Q5", "Q6", "Q7"],
                        help="Questions to evaluate, e.g. Q6 Q7")
    parser.add_argument("--score-key", default="final_calibrated_score",
                        help=(
                            "Score field: final_calibrated_score / selected_baseline_score / "
                            "model_avg_score / single_first_score"
                        ))
    parser.add_argument("--detail", action="store_true", help="Show per-student details")
    parser.add_argument("--compare", action="store_true",
                        help=(
                            "Compare single_first_score, model_avg_score, "
                            "selected_baseline_score, and final_calibrated_score"
                        ))
    parser.add_argument("--compare-score-keys", nargs="+", default=None,
                        help=(
                            "Score types to include in --compare, e.g. "
                            "single avg selected 3wd. Full field names are also supported."
                        ))
    parser.add_argument("--compare-output", default=None,
                        help="Optional CSV path for per-student single/avg/selected/3WD comparison")
    parser.add_argument("--summary-output", default=None,
                        help="Optional JSON path for machine-readable evaluation metrics")
    parser.add_argument("--result-source", choices=["graded", "checkpoint"], default="graded",
                        help="graded reads *_graded_results.json; checkpoint reads full *_grading_checkpoint.json")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help="Directory containing Qx result JSON files")
    parser.add_argument("--rubric-dir", default=None,
                        help="Reserved for run directories that keep rubrics elsewhere")
    parser.add_argument("--teacher-db", default=TEACHER_DB,
                        help="Teacher score JSON")
    parser.add_argument("--database-path", default=DATABASE_PATH,
                        help="Question database JSON used to read dynamic max scores")
    args = parser.parse_args()
    RESULT_SOURCE = args.result_source
    RESULTS_DIR = args.results_dir
    RUBRIC_DIR = args.rubric_dir or RESULTS_DIR
    TEACHER_DB = args.teacher_db
    DATABASE_PATH = args.database_path
    score_map = load_score_map(DATABASE_PATH)
    try:
        compare_score_keys = normalize_score_keys(args.compare_score_keys)
    except ValueError as exc:
        parser.error(str(exc))
    score_key = SCORE_KEY_ALIASES.get(str(args.score_key).strip().lower(), args.score_key)

    ts_db = load_teacher_scores()
    label = SCORE_KEY_LABEL.get(score_key, score_key)
    print(
        f"\n  Loaded teacher scores: {len(ts_db)} students | score field: {label} | "
        f"result source: {RESULT_SOURCE} | results dir: {RESULTS_DIR}"
    )

    all_metrics = []
    for q_id in args.questions:
        total = score_map.get(q_id, 20)
        m = evaluate_question(q_id, total, ts_db, score_key=score_key, show_detail=args.detail)
        if m:
            all_metrics.append(m)

    # Summary table
    if len(all_metrics) >= 1:
        print()
        print()
        print(f"  Summary | {label}")
        print()

        sum_w = [6, 4, 8, 8, 8, 10, 8, 8, 8, 6, 6]
        print_closed_table(
            headers=["Q", "N", "MAE", "RMSE", "QWK", "Pearson r", "TAR(2)", "SER(>2)", "Bias", "Over", "Under"],
            rows=[
                [m["q_id"], str(m["n"]),
                 f"{m['MAE']:.3f}", f"{m['RMSE']:.3f}",
                 f"{m['QWK']:.4f}", f"{m['Pearson']:.4f}",
                 f"{m['TAR2']:.1%}", f"{m['SER2']:.1%}", f"{m['bias']:+.2f}",
                 str(m["high_over"]), str(m["high_under"])]
                for m in all_metrics
            ],
            col_widths=sum_w,
        )

    # Global metrics
    all_t, all_m = [], []
    for q_id in args.questions:
        path = result_path_for(q_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except FileNotFoundError:
            continue
        for r in results:
            sid = r.get("student_id", "")
            t = get_teacher(ts_db, sid, q_id)
            if t is not None and t >= 0:
                m = get_score_value(r, score_key)
                if m is not None:
                    all_t.append(t)
                    all_m.append(round(float(m), 2))

    if len(all_t) >= 3:
        all_t_a, all_m_a = np.array(all_t), np.array(all_m)
        g_mae = mean_absolute_error(all_t_a, all_m_a)
        g_rmse = np.sqrt(mean_squared_error(all_t_a, all_m_a))
        g_pr, _ = stats.pearsonr(all_t_a, all_m_a)
        g_tar2 = float(np.mean(np.abs(all_t_a - all_m_a) <= 2))

        append_row_to_table(
            sum_w,
            ["GLOBAL", str(len(all_t)),
             f"{g_mae:.3f}", f"{g_rmse:.3f}",
             "--", f"{g_pr:.4f}",
             f"{g_tar2:.1%}", f"{1.0 - g_tar2:.1%}",
             f"{np.mean(all_m_a - all_t_a):+.2f}", "", ""],
        )

    # ---- Compare summary (--compare) ----
    if args.compare:
        cmp_w = [10, 6, 4, 8, 8, 8, 10, 8, 8, 8, 6, 6]
        cmp_headers = ["score type", "Q", "N", "MAE", "RMSE", "QWK", "Pearson r", "TAR(2)", "SER(>2)", "Bias", "Over", "Under"]
        cmp_rows = []
        cmp_global = {key: ([], []) for key in compare_score_keys}

        for q_id in args.questions:
            total = score_map.get(q_id, 20)
            for key in compare_score_keys:
                res = compute_question_metrics(q_id, total, ts_db, score_key=key)
                if res is None:
                    continue
                mt, _ = res
                cmp_rows.append([
                    CMP_LABEL[key], mt["q_id"], str(mt["n"]),
                    f"{mt['MAE']:.3f}", f"{mt['RMSE']:.3f}",
                    f"{mt['QWK']:.4f}", f"{mt['Pearson']:.4f}",
                    f"{mt['TAR2']:.1%}", f"{mt['SER2']:.1%}", f"{mt['bias']:+.2f}",
                    str(mt["high_over"]), str(mt["high_under"]),
                ])
            # Collect global data.
            path = result_path_for(q_id)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except FileNotFoundError:
                continue
            for r in results:
                sid = r.get("student_id", "")
                t = get_teacher(ts_db, sid, q_id)
                if t is not None and t >= 0:
                    for key in compare_score_keys:
                        v = get_score_value(r, key)
                        if v is not None:
                            cmp_global[key][0].append(t)
                            cmp_global[key][1].append(round(float(v), 2))

        if cmp_rows:
            print()
            print()
            compare_label = " vs ".join(CMP_LABEL[key] for key in compare_score_keys)
            print(f"  Compare summary | {compare_label}")
            print()
            print_closed_table(headers=cmp_headers, rows=cmp_rows, col_widths=cmp_w)

            for key in compare_score_keys:
                gt, gm = cmp_global[key]
                if len(gt) >= 3:
                    gt_a, gm_a = np.array(gt), np.array(gm)
                    g_mae = mean_absolute_error(gt_a, gm_a)
                    g_rmse = np.sqrt(mean_squared_error(gt_a, gm_a))
                    g_pr, _ = stats.pearsonr(gt_a, gm_a)
                    g_tar2 = float(np.mean(np.abs(gt_a - gm_a) <= 2))
                    append_row_to_table(
                        cmp_w,
                        [CMP_LABEL[key], "GLOBAL", str(len(gt)),
                         f"{g_mae:.3f}", f"{g_rmse:.3f}",
                         "--", f"{g_pr:.4f}",
                         f"{g_tar2:.1%}", f"{1.0 - g_tar2:.1%}",
                         f"{np.mean(gm_a - gt_a):+.2f}", "", ""],
                    )

    if args.compare and "final_calibrated_score" in compare_score_keys:
        print_3wd_mechanism_summary(args.questions, ts_db)
        print_boundary_gain_audit(args.questions, ts_db)

    if args.compare_output:
        exported = export_single_avg_3wd_csv(args.questions, ts_db, args.compare_output)
        print(f"  Exported {exported} single/avg/selected/3WD comparison rows to {args.compare_output}")

    if args.summary_output:
        summary = build_comparison_summary(
            args.questions,
            compare_score_keys if args.compare else [score_key],
            ts_db,
            score_map,
        )
        output_dir = os.path.dirname(args.summary_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.summary_output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"  Exported evaluation summary to {args.summary_output}")

    print()


if __name__ == "__main__":
    main()
