"""Project completed RefGrader scores into the official SAS-Bench protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from json_repair import repair_json
from openai import AuthenticationError, OpenAI, PermissionDeniedError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_runtime import (  # noqa: E402
    TEXT_MODEL_PROFILES,
    runtime_model_config,
    thinking_extra_body,
)


TASK_FILE_PATTERN = re.compile(r"^\d+_.+\.jsonl$", flags=re.IGNORECASE)
METHOD_COLUMNS = {
    "single": "single_first_score",
    "avg": "model_avg_score",
    "selected": "selected_baseline_score",
    "3wd_core": "three_way_core_score",
    "3wd": "final_calibrated_score",
}
CORRECT_NAME = "步骤正确"
PROJECTION_POLICY_VERSION = 3


class ProjectionFailure(RuntimeError):
    """Preserve the final invalid model response for offline diagnosis."""

    def __init__(self, message: str, *, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def official_integer_score(value: Any, total: Any) -> int:
    numeric = float(value)
    maximum = int(float(total))
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite score: {value!r}")
    return min(max(int(numeric), 0), maximum)


def load_compare_scores(compare_path: Path, method: str) -> dict[str, float]:
    column = METHOD_COLUMNS[method]
    result = {}
    with compare_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"student_id", column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"compare.csv is missing columns: {sorted(missing)}")
        for row in reader:
            answer_id = row["student_id"]
            if answer_id in result:
                raise ValueError(f"Duplicate answer in compare.csv: {answer_id}")
            result[answer_id] = float(row[column])
    return result


def load_error_contracts(path: Path) -> dict[str, dict[str, Any]]:
    contracts = {}
    for record in read_jsonl(path):
        task_id = str(record.get("q_id", "")).strip()
        errors = record.get("errors")
        if not task_id or not isinstance(errors, list):
            raise ValueError("Invalid error_type.jsonl record")
        names = []
        for error in errors:
            name = str(error.get("name", "")).strip()
            if not name:
                raise ValueError(f"Empty error name for task {task_id}")
            names.append(name)
        if CORRECT_NAME not in names:
            names.append(CORRECT_NAME)
        contracts[task_id] = {
            "guideline": str(record.get("guideline", "")).strip(),
            "errors": errors,
            "allowed_error_names": names,
        }
    return contracts


def safe_projection_input(record: dict[str, Any]) -> dict[str, Any]:
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Missing source steps for {record.get('id')}")
    safe_steps = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Invalid source step {index} for {record.get('id')}")
        safe_steps.append(
            {"step_index": index, "response": str(step.get("response", ""))}
        )
    return {
        "id": str(record.get("id", "")),
        "question": str(record.get("question", "")),
        "reference": str(record.get("reference", "")),
        "analysis": str(record.get("analysis", "")),
        "total": record.get("total"),
        "steps": safe_steps,
    }


def build_projection_prompt(
    safe_record: dict[str, Any],
    contract: dict[str, Any],
) -> str:
    taxonomy = [
        {"name": item.get("name"), "description": item.get("description", "")}
        for item in contract["errors"]
    ]
    taxonomy.append({"name": CORRECT_NAME, "description": "该步骤正确"})
    return (
        "你是短答案分步评分协议转换器。只根据题目、参考答案、解析、评分指南和"
        "学生作答判断每个原始步骤，不得推断或使用任何教师标签。\n\n"
        f"题目：{safe_record['question']}\n"
        f"满分：{safe_record['total']}\n"
        f"参考答案：{safe_record['reference']}\n"
        f"解析：{safe_record['analysis']}\n"
        f"评分指南：{contract['guideline']}\n"
        f"允许的错因：{json.dumps(taxonomy, ensure_ascii=False)}\n"
        f"学生分步作答：{json.dumps(safe_record['steps'], ensure_ascii=False)}\n\n"
        "输出严格 JSON：{\"steps\":[{\"step_score\":非负整数,"
        "\"errors\":[\"允许的错因名称\"]}]}。steps 数量和顺序必须与输入完全一致；"
        "正确步骤使用‘步骤正确’，错误名称只能来自允许列表；步骤分是各原始步骤的"
        "独立标签，不要求其总和等于或小于整体卷面满分，也不得将其与整体分相加。"
        "不要输出整体分、教师分、解释或 Markdown。"
    )


def parse_projection_response(
    content: str,
    *,
    expected_steps: int,
    total: Any,
    allowed_errors: set[str],
) -> list[dict[str, Any]]:
    repaired = repair_json(content, return_objects=True)
    if isinstance(repaired, list) and len(repaired) == 1:
        repaired = repaired[0]
    if not isinstance(repaired, dict) or not isinstance(repaired.get("steps"), list):
        raise ValueError("Model response does not contain a steps list")
    steps = repaired["steps"]
    if len(steps) != expected_steps:
        raise ValueError(
            f"Step count mismatch: expected={expected_steps}, actual={len(steps)}"
        )
    normalized = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Step {index} is not an object")
        score = float(step.get("step_score"))
        if not math.isfinite(score) or score < 0 or not score.is_integer():
            raise ValueError(f"Invalid step_score at step {index}: {score!r}")
        errors = step.get("errors")
        if not isinstance(errors, list) or not errors:
            raise ValueError(f"Step {index} must contain at least one error label")
        normalized_errors = [str(error).strip() for error in errors]
        unknown = set(normalized_errors) - allowed_errors
        if unknown:
            raise ValueError(f"Unknown error labels at step {index}: {sorted(unknown)}")
        normalized.append(
            {"step_score": int(score), "errors": list(dict.fromkeys(normalized_errors))}
        )
    return normalized


def _api_contract(args: argparse.Namespace) -> dict[str, Any]:
    config = runtime_model_config(
        text_provider=args.text_provider,
        thinking_mode=args.thinking_mode,
        text_temperature_override=args.temperature,
    )
    if config["text_family"] == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = config["text_base_url"]
    else:
        api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
        base_url = "https://open.bigmodel.cn/api/coding/paas/v4/"
    if not api_key:
        raise RuntimeError(
            "Missing API key for the selected provider; set DEEPSEEK_API_KEY or ZHIPUAI_API_KEY"
        )
    return {**config, "api_key": api_key, "base_url": base_url}


def projection_messages(
    prompt: str,
    repair_context: tuple[str, str] | None = None,
) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": prompt}]
    if repair_context is not None:
        previous_response, contract_error = repair_context
        messages.extend(
            [
                {"role": "assistant", "content": previous_response},
                {
                    "role": "user",
                    "content": (
                        "上一输出未通过格式与分值合同校验。"
                        f"校验错误：{contract_error}\n"
                        "请根据原始任务纠正上一输出。保持步骤数量和顺序不变，"
                        "每个步骤得分均为非负整数，"
                        "错误名称只能来自允许列表。只输出纠正后的 JSON。"
                    ),
                },
            ]
        )
    return messages


def call_projection_model(
    prompt: str,
    api: dict[str, Any],
    timeout: int,
    repair_context: tuple[str, str] | None = None,
) -> str:
    request = {
        "model": api["text_model"],
        "messages": projection_messages(prompt, repair_context),
        "temperature": float(api.get("text_temperature_override", 0.0)),
        "response_format": {"type": "json_object"},
        "timeout": timeout,
    }
    if api["text_family"] == "glm":
        request["extra_body"] = thinking_extra_body(api["text_thinking"])
    client = OpenAI(api_key=api["api_key"], base_url=api["base_url"])
    response = client.chat.completions.create(**request)
    return response.choices[0].message.content.strip()


def project_one(
    record: dict[str, Any],
    task: str,
    contract: dict[str, Any],
    api: dict[str, Any],
    model_contract_sha256: str,
    timeout: int,
) -> dict[str, Any]:
    safe_record = safe_projection_input(record)
    prompt = build_projection_prompt(safe_record, contract)
    last_error = None
    last_content = None
    repair_context = None
    for attempt in range(4):
        try:
            content = call_projection_model(
                prompt,
                api,
                timeout,
                repair_context=repair_context,
            )
            last_content = content
            try:
                pred_steps = parse_projection_response(
                    content,
                    expected_steps=len(safe_record["steps"]),
                    total=safe_record["total"],
                    allowed_errors=set(contract["allowed_error_names"]),
                )
            except Exception as exc:
                repair_context = (content, str(exc))
                raise
            return {
                "answer_id": safe_record["id"],
                "source_task": task,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "model_contract_sha256": model_contract_sha256,
                "pred_steps": pred_steps,
                "raw_response": content,
            }
        except (AuthenticationError, PermissionDeniedError):
            raise
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(5 * (attempt + 1))
    raise ProjectionFailure(
        f"Projection failed after 4 attempts: {last_error}",
        raw_response=last_content,
    )


def select_source_records(
    source_dir: Path,
    score_ids: set[str],
    limit_per_task: int | None,
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    selected = {}
    found_ids = set()
    for path in sorted(source_dir.glob("*.jsonl")):
        if not TASK_FILE_PATTERN.fullmatch(path.name):
            continue
        task_records = []
        for record in read_jsonl(path):
            answer_id = str(record.get("id", ""))
            if answer_id in score_ids:
                task_records.append(record)
                found_ids.add(answer_id)
                if limit_per_task is not None and len(task_records) >= limit_per_task:
                    break
        if task_records:
            selected[path.stem] = task_records
    return selected, found_ids


def materialize_predictions(
    records_by_task: dict[str, list[dict[str, Any]]],
    predictions: dict[str, dict[str, Any]],
    scores: dict[str, float],
    output_dir: Path,
) -> int:
    written = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for task, records in records_by_task.items():
        path = output_dir / f"{task}_prediction.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for source_record in records:
                answer_id = str(source_record["id"])
                prediction = predictions.get(answer_id)
                if prediction is None:
                    continue
                output_record = dict(source_record)
                output_record["pred_label"] = official_integer_score(
                    scores[answer_id], source_record["total"]
                )
                output_record["pred_steps"] = prediction["pred_steps"]
                handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--compare-csv", type=Path, required=True)
    parser.add_argument("--protocol-base-dir", type=Path, required=True)
    parser.add_argument("--save-type-name", required=True)
    parser.add_argument("--method", choices=METHOD_COLUMNS, default="3wd")
    parser.add_argument("--text-provider", choices=TEXT_MODEL_PROFILES, default="deepseek")
    parser.add_argument("--thinking-mode", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    compare_path = args.compare_csv.resolve()
    protocol_base = args.protocol_base_dir.resolve()
    output_dir = Path(f"{protocol_base}_{args.save_type_name}_Scored")
    error_path = source_dir / "error_type.jsonl"
    if not source_dir.is_dir() or not error_path.is_file():
        raise FileNotFoundError("SAS-Bench source directory or error_type.jsonl is missing")
    scores = load_compare_scores(compare_path, args.method)
    records_by_task, found_ids = select_source_records(
        source_dir, set(scores), args.limit_per_task
    )
    selected_count = sum(len(records) for records in records_by_task.values())
    if args.limit_per_task is None and found_ids != set(scores):
        missing = sorted(set(scores) - found_ids)
        raise ValueError(f"Source records missing for compare IDs: {missing[:10]}")
    if args.expected_records is not None and selected_count != args.expected_records:
        raise ValueError(
            f"Projection record count mismatch: expected={args.expected_records}, actual={selected_count}"
        )
    print(f"Projection records: {selected_count}")
    print(f"Tasks: {len(records_by_task)}")
    print(f"Output: {output_dir}")
    if args.dry_run:
        return 0

    checkpoint_path = output_dir / "projection_checkpoint.jsonl"
    if output_dir.exists() and any(output_dir.iterdir()) and not checkpoint_path.is_file():
        raise FileExistsError(
            "Projection output is non-empty but has no resumable checkpoint: "
            f"{output_dir}. Use a new save-type-name."
        )

    api = _api_contract(args)
    contracts = load_error_contracts(error_path)
    public_model_contract = {
        "projection_policy_version": PROJECTION_POLICY_VERSION,
        **{key: value for key, value in api.items() if key != "api_key"},
    }
    model_contract_sha256 = hashlib.sha256(
        json.dumps(
            public_model_contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    checkpoint_records = read_jsonl(checkpoint_path) if checkpoint_path.is_file() else []
    checkpoint_ids = [str(record.get("answer_id", "")) for record in checkpoint_records]
    if len(checkpoint_ids) != len(set(checkpoint_ids)):
        raise ValueError("Projection checkpoint contains duplicate answer IDs")
    unexpected_checkpoint_ids = set(checkpoint_ids) - found_ids
    if unexpected_checkpoint_ids:
        raise ValueError(
            "Projection checkpoint belongs to a different record subset: "
            f"{sorted(unexpected_checkpoint_ids)[:10]}"
        )
    predictions = {record["answer_id"]: record for record in checkpoint_records}
    pending = []
    for task, records in records_by_task.items():
        task_id = task.split("_", 1)[0]
        if task_id not in contracts:
            raise ValueError(f"Missing error taxonomy for task {task}")
        for record in records:
            answer_id = str(record["id"])
            existing = predictions.get(answer_id)
            if existing is not None:
                prompt = build_projection_prompt(
                    safe_projection_input(record), contracts[task_id]
                )
                prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if existing.get("prompt_sha256") != prompt_sha256:
                    raise ValueError(
                        f"Checkpoint prompt contract changed for {answer_id}; "
                        "use a new save-type-name"
                    )
                if existing.get("model_contract_sha256") != model_contract_sha256:
                    raise ValueError(
                        f"Checkpoint model contract changed for {answer_id}; "
                        "use a new save-type-name"
                    )
            else:
                pending.append((task, record, contracts[task_id]))
    print(f"Completed checkpoint records: {len(predictions)}")
    print(f"Pending projection records: {len(pending)}")

    lock = threading.Lock()
    failures = []
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        future_map = {
            executor.submit(
                project_one,
                record,
                task,
                contract,
                api,
                model_contract_sha256,
                args.timeout,
            ): (
                task,
                str(record["id"]),
            )
            for task, record, contract in pending
        }
        for completed, future in enumerate(as_completed(future_map), start=1):
            task, answer_id = future_map[future]
            try:
                prediction = future.result()
                predictions[answer_id] = prediction
                append_jsonl(checkpoint_path, prediction, lock)
            except Exception as exc:
                failure = {
                    "source_task": task,
                    "answer_id": answer_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                raw_response = getattr(exc, "raw_response", None)
                if raw_response is not None:
                    failure["raw_response"] = raw_response
                failures.append(failure)
            if completed % 25 == 0 or completed == len(future_map):
                print(
                    f"Projection progress: {completed}/{len(future_map)}, "
                    f"success={len(predictions)}, failed={len(failures)}",
                    flush=True,
                )

    protocol_base.mkdir(parents=True, exist_ok=True)
    shutil.copy2(error_path, protocol_base / "error_type.jsonl")
    written = materialize_predictions(
        records_by_task, predictions, scores, output_dir
    )
    failure_path = output_dir / "projection_failures.jsonl"
    if failures:
        with failure_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    elif failure_path.exists():
        failure_path.unlink()
    manifest = {
        "projection_policy_version": PROJECTION_POLICY_VERSION,
        "method": args.method,
        "record_count": selected_count,
        "prediction_count": written,
        "failure_count": len(failures),
        "task_count": len(records_by_task),
        "label_blind_projection": True,
        "overall_score_source": METHOD_COLUMNS[args.method],
        "step_projection_inputs": [
            "question",
            "reference",
            "analysis",
            "total",
            "student_step_responses",
            "official_guideline",
            "official_error_taxonomy",
        ],
        "forbidden_model_inputs": ["manual_label", "step.label", "step.errors"],
        "model_contract": public_model_contract,
        "model_contract_sha256": model_contract_sha256,
    }
    (output_dir / "protocol_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failures or written != selected_count:
        raise RuntimeError(
            f"Protocol projection incomplete: written={written}/{selected_count}, failures={len(failures)}"
        )
    print(f"Protocol projection complete: {written}/{selected_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
