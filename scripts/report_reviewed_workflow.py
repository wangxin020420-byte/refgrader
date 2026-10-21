"""Read-only workflow report based on verified checkpoints, not progress counters.

Uses only the standard library. Does not import model clients or write files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(root, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("Invalid run ID")
    job = Path(root) / "results_runs/reviewed_csbench45_glm52" / run_id
    state = read(job / "workflow.json")
    if state["run_id"] != run_id:
        raise ValueError("Workflow run ID mismatch")
    completed = state.get("completed", {})
    report = {
        "workflow": {key: state.get(key) for key in (
            "run_id", "status", "stage", "error", "result_class",
            "deployment_gate_passed", "updated_at",
        )},
        "completed_stages": len(completed),
        "splits": {},
    }
    test_routes, test_outcomes = Counter(), Counter()
    for split in ("validation", "test"):
        directory = Path(state["paths"][split])
        completion = read(directory / "completion_report.json")
        if set(completion["questions"]) != set(state["questions"]):
            raise ValueError(f"{split}: completion report question set mismatch")
        proof = {
            Path(path).resolve(): digest
            for path, digest in completed.get("grade_" + split, {}).get("proof", {}).items()
        }
        progress_path = directory / "progress.json"
        progress = read(progress_path).get("questions", {}) if progress_path.is_file() else {}
        total, expected, stale = 0, 0, []
        for question in state["questions"]:
            checkpoint = directory / (question + "_grading_checkpoint.json")
            expected_hash = proof.get(checkpoint.resolve())
            if not expected_hash or sha(checkpoint) != expected_hash:
                raise ValueError(f"Checkpoint proof missing or changed: {checkpoint}")
            rows = read(checkpoint)
            if not isinstance(rows, list):
                raise ValueError(f"Checkpoint must be a list: {checkpoint}")
            ids = [row.get("student_id") for row in rows]
            if any(not value for value in ids) or len(set(ids)) != len(ids):
                raise ValueError(f"Missing or duplicate answer IDs: {checkpoint}")
            item = completion["questions"][question]
            count = item["expected_count"]
            if item["status"] != "complete" or len(rows) != count or item["checkpoint_count"] != count:
                raise ValueError(f"Incomplete checkpoint/completion report: {question} {split}")
            expected += count
            total += len(rows)
            if progress.get(question, {}).get("completed") != len(rows):
                stale.append(question)
            if split == "test":
                test_routes.update(row.get("3wd_route") or "missing" for row in rows)
                test_outcomes.update(row.get("3wd_sequential_outcome") or "missing" for row in rows)
        if expected != completion["expected_total"] or total != completion["checkpoint_total"]:
            raise ValueError(f"{split}: inconsistent completion totals")
        report["splits"][split] = {
            "questions": len(state["questions"]), "expected": expected,
            "checkpoint_count": total, "checkpoint_hashes": "verified",
            "stale_progress_questions": stale,
        }

    config = read(Path(state["paths"]["config"]))
    validation = config.get("validation_summary", {})
    report["calibration"] = {
        "thresholds": config.get("thresholds"),
        "gate": config.get("deployment_gate"),
        "unsafe_pos_count": validation.get("unsafe_pos_count"),
        "unsafe_pos_rate": validation.get("unsafe_pos_rate"),
        "unsafe_pos_limit": config.get("operational_costs", {}).get("max_unsafe_pos_rate"),
        "validation_route_counts": validation.get("route_counts"),
    }
    report["test_routing"] = {"routes": dict(test_routes), "outcomes": dict(test_outcomes)}
    summary = read(Path(state["paths"]["test"]) / "evaluation/summary.json")
    metrics = []
    for row in summary["global"]:
        values = [item["QWK"] for item in summary["per_question"]
                  if item["score_key"] == row["score_key"]
                  and isinstance(item.get("QWK"), (int, float))
                  and math.isfinite(item["QWK"])]
        metrics.append({
            "score_type": row["score_type"], "n": row["n"],
            "macro_qwk": sum(values) / len(values) if values else None,
            "valid_qwk_questions": len(values),
            **{key: row.get(key) for key in ("MAE", "RMSE", "bias")},
        })
    report["test_metrics"] = metrics
    frozen = read_review_lines(job / "review_decisions.jsonl")
    live_path = state.get("decisions_source")
    live = read_review_lines(Path(live_path)) if live_path and Path(live_path).is_file() else None
    report["reviews"] = {
        "frozen_records": len(frozen),
        "live_records": len(live) if live is not None else None,
        "live_decisions": dict(Counter(row.get("decision", "unknown") for row in live)) if live is not None else None,
        "note": "Live review changes are not incorporated into this frozen experiment.",
    }
    return report


def read_review_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def print_report(report):
    state = report["workflow"]
    print("Run:", state["run_id"])
    print(f"工作流：{state['status']}；已完成阶段：{report['completed_stages']}；当前错误：{state['error']}")
    print(f"结果分类：{state['result_class']}；部署门禁通过：{state['deployment_gate_passed']}")
    for split, values in report["splits"].items():
        print(f"{split}: {values['checkpoint_count']}/{values['expected']}，{values['questions']}题，检查点哈希通过")
        if values["stale_progress_questions"]:
            print(f"  旧进度计数未同步：{len(values['stale_progress_questions'])}题；本报告以已核验检查点为准。")
    calibration = report["calibration"]
    gate = calibration["gate"] or {}
    print("未通过的门禁：", ", ".join(key for key, passed in gate.get("requirements", {}).items() if passed is False) or "无")
    rate, limit = calibration["unsafe_pos_rate"], calibration["unsafe_pos_limit"]
    if rate is not None and limit is not None:
        print(f"验证集POS不安全比例：{rate:.2%}；上限：{limit:.2%}")
    print("测试集路由：", report["test_routing"]["routes"])
    print("测试集最终动作：", report["test_routing"]["outcomes"])
    print("评分阶段       宏平均QWK       MAE       RMSE       Bias")
    for row in report["test_metrics"]:
        qwk = f"{row['macro_qwk']:.2%}" if row["macro_qwk"] is not None else "N/A"
        print(f"{row['score_type']:<12} {qwk:>9} {row['MAE']:>9.4f} {row['RMSE']:>9.4f} {row['bias']:>10.4f}")
    print(f"审核记录：本轮冻结{report['reviews']['frozen_records']}条；当前本地{report['reviews']['live_records']}条。")
    print("本命令只读；不调用模型、不改写结果、不提交或推送Git。")


def main():
    # Keep redirected output readable in Windows PowerShell as well as consoles.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        report = build_report(ROOT, args.run_id)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("核验失败：" + str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
