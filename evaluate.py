"""
RefGrader 评估脚本
用法:
  python evaluate.py                                     # 默认评估 Q4-Q7, 使用 final_calibrated_score
  python evaluate.py --score-key model_avg_score         # 消融: 使用 3WD 校准前的模型均分
  python evaluate.py --questions Q5 Q7 --detail          # 指定题目 + 逐条详情
  python evaluate.py --score-key model_avg_score --detail
  python evaluate.py --compare                           # 对比: 模型均分 vs 3WD校准分
  python evaluate.py --compare --questions Q5 Q6 Q7         # 指定题目的对比
"""

import json
import argparse
import numpy as np
from collections import Counter
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score

SCORES_MAP = {"Q1": 10, "Q2": 20, "Q3": 10, "Q4": 20, "Q5": 15, "Q6": 20, "Q7": 10}
RESULTS_DIR = "./results_rrd_vlm"
TEACHER_DB = "./database/teacher_scores.json"

SCORE_KEY_LABEL = {
    "final_calibrated_score": "3WD 校准分",
    "model_avg_score": "模型原始均分 (无3WD)",
}

CMP_LABEL = {
    "model_avg_score": "模型均分",
    "final_calibrated_score": "3WD校准",
}


def load_teacher_scores():
    with open(TEACHER_DB, "r", encoding="utf-8") as f:
        return json.load(f)


def get_teacher(ts_db, student_id, q_id):
    pure_id = student_id.split("_")[0]
    return ts_db.get(pure_id, {}).get(q_id, None)


def dw(s):
    """显示宽度: 中文=2, ASCII=1"""
    return sum(2 if ord(c) > 127 else 1 for c in str(s))


def cpad(s, width):
    """居中对齐"""
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
    """打印封闭式居中对齐表格"""
    print("  " + make_line(col_widths))
    print("  " + make_row(headers, col_widths))
    print("  " + make_line(col_widths, left="+", mid="+", right="+", fill="="))
    for row in rows:
        print("  " + make_row(row, col_widths))
    print("  " + make_line(col_widths))


def append_row_to_table(col_widths, cells):
    """向已有表格追加一行 (无顶线)"""
    print("  " + make_line(col_widths, left="+", mid="+", right="+", fill="-"))
    print("  " + make_row(cells, col_widths))
    print("  " + make_line(col_widths))


def compute_question_metrics(q_id, total_score, ts_db, score_key="final_calibrated_score"):
    """计算单题评估指标 (不打印), 返回 (metrics, details) 或 None"""
    path = f"{RESULTS_DIR}/{q_id}_graded_results.json"
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

    details = []
    for r in results:
        sid = r.get("student_id", "")
        t = get_teacher(ts_db, sid, q_id)
        if t is None or t < 0:
            continue
        m = r.get(score_key, None)
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
        unwritten = sum(1 for v in facts.values() if "未" in str(v) or str(v).strip() == "")

        details.append({
            "sid": sid.split("_")[0],
            "teacher": t, "model": m, "diff": diff,
            "route": r.get("3wd_route", ""),
            "blank_rate": r.get("blank_rate", 0),
            "avg": r.get("model_avg_score", None),
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
        "teacher_mean": float(np.mean(t_arr)),
        "model_mean": float(np.mean(m_arr)),
        "bias": float(np.mean(m_arr - t_arr)),
        "serious": len(serious), "high_over": high_over, "high_under": high_under,
        "route_counts": dict(route_counts),
        "risk_counts": dict(risk_counts),
    }
    return metrics, details


def print_boundary_gain_audit(questions, ts_db):
    """Audit whether 3WD corrections improve over model_avg_score."""
    print()
    print("  3WD gain audit | final_calibrated_score vs model_avg_score")
    print()
    headers = ["Q", "route", "N", "improved", "worsened", "same", "mean_gain", "mean_delta"]
    rows = []
    top_worsened = []

    for q_id in questions:
        path = f"{RESULTS_DIR}/{q_id}_graded_results.json"
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
            final = r.get("final_calibrated_score", None)
            if teacher is None or teacher < 0 or avg is None or final is None:
                continue
            teacher = float(teacher)
            avg = float(avg)
            final = float(final)
            route = r.get("3wd_route", "")
            gain = abs(avg - teacher) - abs(final - teacher)
            delta = final - avg
            grouped.setdefault(route, []).append((sid, teacher, avg, final, gain, delta, r))
            top_worsened.append((gain, q_id, route, sid, teacher, avg, final, r))

        for route in sorted(grouped):
            items = grouped[route]
            gains = [x[4] for x in items]
            deltas = [x[5] for x in items]
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
        for gain, q_id, route, sid, teacher, avg, final, r in worsened[:10]:
            gate = r.get("boundary_gate") or {}
            detail_rows.append([
                q_id,
                sid.split("_")[0],
                route,
                f"{teacher:.1f}",
                f"{avg:.1f}",
                f"{final:.1f}",
                f"{gain:+.1f}",
                str(gate.get("action", ""))[:18],
            ])
        print_closed_table(
            headers=["Q", "sid", "route", "teacher", "avg", "final", "gain", "gate"],
            rows=detail_rows,
            col_widths=[4, 13, 7, 8, 7, 7, 7, 18],
        )


def evaluate_question(q_id, total_score, ts_db, score_key="final_calibrated_score", show_detail=False):
    result = compute_question_metrics(q_id, total_score, ts_db, score_key)
    if result is None:
        print(f"  {q_id}: 无有效数据!")
        return None

    metrics, details = result
    label = SCORE_KEY_LABEL.get(score_key, score_key)

    print()
    print()
    print(f"  {q_id}  |  满分 {total_score}  |  N = {metrics['n']}  |  {label}")
    print()

    # 核心指标表
    metric_w = [10, 10, 10, 10, 12, 10]
    print_closed_table(
        headers=["MAE", "RMSE", "QWK", "Pearson r", "TAR(2)"],
        rows=[[f"{metrics['MAE']:.4f}", f"{metrics['RMSE']:.4f}",
                f"{metrics['QWK']:.4f}", f"{metrics['Pearson']:.4f}",
                f"{metrics['TAR2']:.1%}"]],
        col_widths=metric_w,
    )

    # 补充统计表
    print()
    stat_w = [10, 10, 10, 10, 14, 12]
    print_closed_table(
        headers=["教师均分", "模型均分", "系统偏差", "Pearson p", "严重偏差(>2)", "高估 / 低估"],
        rows=[[f"{metrics['teacher_mean']:.2f}", f"{metrics['model_mean']:.2f}",
               f"{metrics['bias']:+.2f}", f"{metrics['Pearson_p']:.1e}",
               f"{metrics['serious']} 人", f"{metrics['high_over']} / {metrics['high_under']}"]],
        col_widths=stat_w,
    )

    route_counts = metrics.get("route_counts", {})
    if route_counts:
        print()
        print("  路由分布: " + " | ".join(f"{k}={v}" for k, v in sorted(route_counts.items())))

    if show_detail:
        print()
        det_w = [13, 6, 6, 7, 7, 6, 7, 7, 6]
        print_closed_table(
            headers=["学号", "教师", "模型", "均分", "差值", "路由", "留白率", "未书写", "标记"],
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
    parser = argparse.ArgumentParser(description="RefGrader 评估脚本")
    parser.add_argument("--questions", nargs="+", default=["Q4", "Q5", "Q6", "Q7"],
                        help="要评估的题目 (默认 Q4 Q5 Q6 Q7)")
    parser.add_argument("--score-key", default="final_calibrated_score",
                        help="评分字段: final_calibrated_score(默认) / model_avg_score(消融)")
    parser.add_argument("--detail", action="store_true", help="显示逐条学生详情")
    parser.add_argument("--compare", action="store_true",
                        help="显示模型均分与3WD校准分的对比汇总表")
    args = parser.parse_args()

    ts_db = load_teacher_scores()
    label = SCORE_KEY_LABEL.get(args.score_key, args.score_key)
    print(f"\n  已加载教师评分: {len(ts_db)} 名学生  |  评分字段: {label}")

    all_metrics = []
    for q_id in args.questions:
        total = SCORES_MAP.get(q_id, 20)
        m = evaluate_question(q_id, total, ts_db, score_key=args.score_key, show_detail=args.detail)
        if m:
            all_metrics.append(m)

    # 汇总表
    if len(all_metrics) >= 1:
        print()
        print()
        print(f"  汇总  |  {label}")
        print()

        sum_w = [6, 4, 8, 8, 8, 10, 8, 8, 6, 6]
        print_closed_table(
            headers=["题号", "N", "MAE", "RMSE", "QWK", "Pearson r", "TAR(2)", "偏差", "高估", "低估"],
            rows=[
                [m["q_id"], str(m["n"]),
                 f"{m['MAE']:.3f}", f"{m['RMSE']:.3f}",
                 f"{m['QWK']:.4f}", f"{m['Pearson']:.4f}",
                 f"{m['TAR2']:.1%}", f"{m['bias']:+.2f}",
                 str(m["high_over"]), str(m["high_under"])]
                for m in all_metrics
            ],
            col_widths=sum_w,
        )

    # 全局
    all_t, all_m = [], []
    for q_id in args.questions:
        path = f"{RESULTS_DIR}/{q_id}_graded_results.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except FileNotFoundError:
            continue
        for r in results:
            sid = r.get("student_id", "")
            t = get_teacher(ts_db, sid, q_id)
            if t is not None and t >= 0:
                m = r.get(args.score_key, None)
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
            ["全局", str(len(all_t)),
             f"{g_mae:.3f}", f"{g_rmse:.3f}",
             "--", f"{g_pr:.4f}",
             f"{g_tar2:.1%}", f"{np.mean(all_m_a - all_t_a):+.2f}", "", ""],
        )

    # ---- 对比汇总表 (--compare) ----
    if args.compare:
        cmp_w = [10, 6, 4, 8, 8, 8, 10, 8, 8, 6, 6]
        cmp_headers = ["评分方式", "题号", "N", "MAE", "RMSE", "QWK", "Pearson r", "TAR(2)", "偏差", "高估", "低估"]
        cmp_rows = []
        cmp_global = {"model_avg_score": ([], []), "final_calibrated_score": ([], [])}

        for q_id in args.questions:
            total = SCORES_MAP.get(q_id, 20)
            for key in ("model_avg_score", "final_calibrated_score"):
                res = compute_question_metrics(q_id, total, ts_db, score_key=key)
                if res is None:
                    continue
                mt, _ = res
                cmp_rows.append([
                    CMP_LABEL[key], mt["q_id"], str(mt["n"]),
                    f"{mt['MAE']:.3f}", f"{mt['RMSE']:.3f}",
                    f"{mt['QWK']:.4f}", f"{mt['Pearson']:.4f}",
                    f"{mt['TAR2']:.1%}", f"{mt['bias']:+.2f}",
                    str(mt["high_over"]), str(mt["high_under"]),
                ])
            # 收集全局数据
            path = f"{RESULTS_DIR}/{q_id}_graded_results.json"
            try:
                with open(path, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except FileNotFoundError:
                continue
            for r in results:
                sid = r.get("student_id", "")
                t = get_teacher(ts_db, sid, q_id)
                if t is not None and t >= 0:
                    for key in ("model_avg_score", "final_calibrated_score"):
                        v = r.get(key, None)
                        if v is not None:
                            cmp_global[key][0].append(t)
                            cmp_global[key][1].append(round(float(v), 2))

        if cmp_rows:
            print()
            print()
            print("  对比汇总  |  模型均分 (无3WD) vs 3WD校准分")
            print()
            print_closed_table(headers=cmp_headers, rows=cmp_rows, col_widths=cmp_w)

            # 全局对比行
            for key in ("model_avg_score", "final_calibrated_score"):
                gt, gm = cmp_global[key]
                if len(gt) >= 3:
                    gt_a, gm_a = np.array(gt), np.array(gm)
                    g_mae = mean_absolute_error(gt_a, gm_a)
                    g_rmse = np.sqrt(mean_squared_error(gt_a, gm_a))
                    g_pr, _ = stats.pearsonr(gt_a, gm_a)
                    g_tar2 = float(np.mean(np.abs(gt_a - gm_a) <= 2))
                    append_row_to_table(
                        cmp_w,
                        [CMP_LABEL[key], "全局", str(len(gt)),
                         f"{g_mae:.3f}", f"{g_rmse:.3f}",
                         "--", f"{g_pr:.4f}",
                         f"{g_tar2:.1%}", f"{np.mean(gm_a - gt_a):+.2f}", "", ""],
                    )

    if args.compare:
        print_boundary_gain_audit(args.questions, ts_db)

    print()


if __name__ == "__main__":
    main()
