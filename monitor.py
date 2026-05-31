"""
RefGrader 实验进度监控脚本

用法:
  python monitor.py                              # 单次显示
  python monitor.py --watch                      # 持续刷新模式 (每5秒)
  python monitor.py --watch --interval 10        # 自定义刷新间隔
  python monitor.py --verbose                    # 显示最近完成的学生列表
  python monitor.py --progress-file path.json    # 指定进度文件路径
"""

import json
import argparse
import os
import sys
import time
from datetime import datetime


def format_duration(seconds):
    """将秒数转换为人类可读格式"""
    if seconds is None:
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m}m"


def format_timestamp(iso_str):
    """将 ISO 时间戳转换为可读格式"""
    if not iso_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return iso_str


def display_progress(data, verbose=False):
    """渲染进度数据到终端。返回 True 表示实验仍在运行。"""
    exp = data.get("experiment", {})
    questions = data.get("questions", {})

    status = exp.get("status", "unknown")
    status_icon = {
        "running": "🔄",
        "completed": "✅",
        "interrupted": "🛑",
        "error": "❌",
    }.get(status, "?")

    started = exp.get("started_at", "")
    updated = exp.get("last_updated", "")

    # 计算总运行时长
    elapsed_str = ""
    if started:
        try:
            start_dt = datetime.fromisoformat(started)
            end_dt = datetime.fromisoformat(updated) if updated else datetime.now()
            elapsed = (end_dt - start_dt).total_seconds()
            elapsed_str = format_duration(elapsed)
        except (ValueError, TypeError):
            elapsed_str = "?"

    print(f"\n{'=' * 78}")
    print(f"  RefGrader Experiment Monitor")
    print(f"{'=' * 78}")
    print(f"  Run ID:     {exp.get('run_id', 'N/A')}")
    print(f"  Mode:       {exp.get('mode', 'N/A')}")
    print(f"  Model:      {exp.get('model_provider', '')} -> {exp.get('text_model', '')}")
    print(f"  Workers:    {exp.get('max_workers_outer', '?')}")
    print(f"  Status:     {status_icon} {status}")
    print(f"  Started:    {format_timestamp(started)}")
    print(f"  Updated:    {format_timestamp(updated)}")
    print(f"  Elapsed:    {elapsed_str}")

    if not questions:
        print(f"\n  尚无题目数据，等待流水线启动...")
        return status == "running"

    # 汇总统计
    total_completed = 0
    total_failed = 0
    total_all = 0

    print(f"\n  {'题号':<5} {'进度':<16} {'完成':>4} {'失败':>4} {'剩余':>4} {'ETA':>8} {'均时':>6} {'POS':>4} {'BND':>4} {'NEG':>4}")
    print(f"  {'-' * 72}")

    # 按 Q1, Q2, ... Q7 排序
    def sort_key(q_id):
        try:
            return int(q_id[1:])
        except (ValueError, IndexError):
            return 99

    for q_id in sorted(questions.keys(), key=sort_key):
        q = questions[q_id]
        completed = q.get("completed", 0)
        failed = q.get("failed", 0)
        remaining = q.get("remaining", 0)
        total = q.get("total_students", 0)
        total_completed += completed
        total_failed += failed
        total_all += total

        # 进度条
        pct = completed / total * 100 if total > 0 else 0
        bar_len = 10
        filled = int(bar_len * completed / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)

        eta = format_duration(q.get("eta_seconds"))
        avg = format_duration(q.get("avg_seconds_per_student"))

        routes = q.get("route_distribution", {})
        pos = routes.get("POS", 0)
        bnd = routes.get("BND", 0)
        neg = routes.get("NEG", 0)

        print(
            f"  {q_id:<5} {bar} {pct:5.1f}%"
            f" {completed:>4} {failed:>4} {remaining:>4}"
            f" {eta:>8} {avg:>6}"
            f" {pos:>4} {bnd:>4} {neg:>4}"
        )

    print(f"  {'-' * 72}")
    total_remaining = total_all - total_completed - total_failed
    overall_pct = total_completed / total_all * 100 if total_all > 0 else 0
    print(f"  {'合计':<5} {'':>16} {total_completed:>4} {total_failed:>4} {total_remaining:>4}")
    print(f"  整体进度: {overall_pct:.1f}% ({total_completed}/{total_all})")

    # 错误区
    any_errors = []
    for q_id, q in questions.items():
        for err in q.get("current_errors", []):
            any_errors.append(f"  [{q_id}] {err.get('student_id', '?')}: {err.get('error', '')}")
    if any_errors:
        print(f"\n  ⚠️ 最近的错误:")
        for e in any_errors[-5:]:
            print(e)

    # 详细模式：最近完成的学生
    if verbose:
        print(f"\n  最近完成的学生:")
        has_data = False
        for q_id in sorted(questions.keys(), key=sort_key):
            for rc in questions[q_id].get("recent_completions", [])[-3:]:
                has_data = True
                teacher = rc.get("teacher_score", "?")
                final = rc.get("final_score", "?")
                diff = ""
                if isinstance(teacher, (int, float)) and isinstance(final, (int, float)):
                    diff = f" (diff={final - teacher:+.1f})"
                print(
                    f"    [{q_id}] {rc.get('student_id', '?'):>15} | "
                    f"{rc.get('route', '?'):>3} | "
                    f"score={final}{diff} | "
                    f"{format_duration(rc.get('duration_seconds'))}"
                )
        if not has_data:
            print("    (暂无数据)")

    # CLI 回显
    cli_args = exp.get("cli_args", {})
    if cli_args:
        q_list = cli_args.get("questions", [])
        if q_list:
            print(f"\n  CLI: --mode {cli_args.get('mode', '?')} --questions {' '.join(q_list)}", end="")
            if cli_args.get("force_rerun"):
                print(" --force-rerun", end="")
            il = cli_args.get("img_limit")
            if il is not None:
                if isinstance(il, list):
                    print(f" --student-ids {' '.join(il)}", end="")
                else:
                    print(f" --img-limit {il}", end="")
            print()

    print(f"{'=' * 78}\n")

    return status == "running"


def load_progress(path):
    """加载进度 JSON，文件不存在或无效时返回 None"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError, OSError):
        return None


def main():
    parser = argparse.ArgumentParser(description="RefGrader 实验进度监控")
    parser.add_argument("--watch", action="store_true", help="持续监控模式 (自动刷新)")
    parser.add_argument("--interval", type=int, default=5, help="刷新间隔秒数 (默认: 5)")
    parser.add_argument(
        "--progress-file",
        default="results_rrd_vlm/progress.json",
        help="进度 JSON 文件路径 (默认: results_rrd_vlm/progress.json)",
    )
    parser.add_argument("--verbose", action="store_true", help="显示最近完成的学生列表")
    args = parser.parse_args()

    if args.watch:
        print(f"持续监控模式 (每 {args.interval} 秒刷新，Ctrl+C 退出)")
        try:
            while True:
                data = load_progress(args.progress_file)
                if data is None:
                    print(
                        f"\r等待进度数据... ({datetime.now().strftime('%H:%M:%S')})",
                        end="",
                        flush=True,
                    )
                    time.sleep(args.interval)
                    continue

                # 清屏
                os.system("clear" if os.name != "nt" else "cls")
                still_running = display_progress(data, verbose=args.verbose)

                if not still_running:
                    print("实验已结束，退出监控。")
                    break

                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n监控已停止。")
    else:
        data = load_progress(args.progress_file)
        if data is None:
            print(f"进度文件不存在或无效: {args.progress_file}")
            print("请先启动实验:")
            print("  ./run_experiment.sh run --mode FULL --questions Q5 Q6 Q7")
            sys.exit(1)
        display_progress(data, verbose=args.verbose)


if __name__ == "__main__":
    main()
