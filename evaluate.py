"""
RefGrader 评估脚本
用法:
  python evaluate.py                                     # 默认评估 Q4-Q7, 使用 final_calibrated_score
  python evaluate.py --score-key model_avg_score         # 消融: 使用 3WD 校准前的模型均分
  python evaluate.py --questions Q5 Q7 --detail          # 指定题目 + 逐条详情
  python evaluate.py --score-key model_avg_score --detail
"""

import json
import argparse
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score

SCORES_MAP = {"Q1": 10, "Q2": 20, "Q3": 10, "Q4": 20, "Q5": 15, "Q6": 20, "Q7": 10}
RESULTS_DIR = "./results_rrd_vlm"
TEACHER_DB = "./database/teacher_scores.json"

SCORE_KEY_LABEL = {
    "final_calibrated_score": "3WD 校准分",
    "model_avg_score": "模型原始均分 (无3WD)",
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


def evaluate_question(q_id, total_score, ts_db, score_key="final_calibrated_score", show_detail=False):
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
        print(f"  {q_id}: 无有效数据!")
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

    label = SCORE_KEY_LABEL.get(score_key, score_key)

    print()
    print()
    print(f"  {q_id}  |  满分 {total_score}  |  N = {n}  |  {label}")
    print()

    # 核心指标表
    metric_w = [10, 10, 10, 10, 12, 10]
    print_closed_table(
        headers=["MAE", "RMSE", "QWK", "Pearson r", "TAR(2)"],
        rows=[[f"{mae:.4f}", f"{rmse:.4f}", f"{qwk:.4f}", f"{pearson_r:.4f}", f"{tar2:.1%}"]],
        col_widths=metric_w,
    )

    # 补充统计表
    print()
    stat_w = [10, 10, 10, 10, 14, 12]
    print_closed_table(
        headers=["教师均分", "模型均分", "系统偏差", "Pearson p", "严重偏差(>2)", "高估 / 低估"],
        rows=[[f"{np.mean(t_arr):.2f}", f"{np.mean(m_arr):.2f}",
               f"{np.mean(m_arr - t_arr):+.2f}", f"{pearson_p:.1e}",
               f"{len(serious)} 人", f"{high_over} / {high_under}"]],
        col_widths=stat_w,
    )

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

    return {
        "q_id": q_id, "n": n, "total": total_score,
        "MAE": mae, "RMSE": rmse, "QWK": qwk,
        "Pearson": pearson_r, "TAR2": tar2,
        "teacher_mean": float(np.mean(t_arr)),
        "model_mean": float(np.mean(m_arr)),
        "bias": float(np.mean(m_arr - t_arr)),
        "serious": len(serious), "high_over": high_over, "high_under": high_under,
    }


def main():
    parser = argparse.ArgumentParser(description="RefGrader 评估脚本")
    parser.add_argument("--questions", nargs="+", default=["Q4", "Q5", "Q6", "Q7"],
                        help="要评估的题目 (默认 Q4 Q5 Q6 Q7)")
    parser.add_argument("--score-key", default="final_calibrated_score",
                        help="评分字段: final_calibrated_score(默认) / model_avg_score(消融)")
    parser.add_argument("--detail", action="store_true", help="显示逐条学生详情")
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

    print()


if __name__ == "__main__":
    main()
