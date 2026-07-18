import os
import sys
import json
import logging
import argparse
import signal
import tempfile
import hashlib
import concurrent.futures
import time
import numpy as np
from datetime import datetime
from step3_rrd_generator import generate_rrd_rubrics, refine_rubric_based_on_variance
from rubric_semantics import (
    HIGH_VALUE_SPLIT_THRESHOLD,
    RUBRIC_SEMANTIC_CONTRACT_VERSION,
    apply_hierarchical_scoring_policy,
    high_value_split_targets,
    prepare_rubric_semantic_contract,
    rubric_scoring_signature,
    rubric_structure_signature,
    validate_refined_rubric,
)
from canonicalizers import build_canonical_grading_context
from step4_vlm_grader import (
    apply_canonical_score_floor,
    grade_student_3wd_pipeline,
    generate_blind_checklist,
    stage1_extract_with_backend,
    stage2_logic_grading,
    extract_and_parse_json,
    VLM_MODEL_NAME,
    TEXT_MODEL_PROVIDER,
    GLM_MODEL_NAME,
    GLM5_MODEL_NAME,
    DEEPSEEK_MODEL_NAME,
    MAX_WORKERS_OUTER,
)
from ocr.backend import ensure_paddle_ocr_cache, ocr_json_path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================
# 全局配置
# ============================================================
DEFAULT_OUTPUT_DIR = "./results_rrd_vlm"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
RUBRIC_DIR = OUTPUT_DIR
INITIAL_RUBRIC_DIR = None
ALLOW_INITIAL_RUBRIC = False
DATABASE_PATH = "./database/exam_database.json"
TEACHER_DB_PATH = "./database/teacher_scores.json"
ANSWER_METADATA_PATH = None

# 全局变量，用于缓存加载的成绩单，避免每次都读文件
_GLOBAL_SCORES_DB = None
_GLOBAL_ANSWER_METADATA = None

# 优雅关闭标志
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    _shutdown_requested = True
    logging.warning(f"⚠️ 收到 {sig_name} 信号，等待当前学生完成后停止...")


# ============================================================
# 日志系统
# ============================================================
def setup_logging(log_dir="logs", run_id=None):
    """配置双 handler 日志：控制台 + 带时间戳的文件"""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"experiment_{run_id}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 控制台 handler —— 保持原有视觉效果（无时间戳前缀）
    if not logger.handlers:  # 避免重复添加
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

        # 文件 handler —— 带时间戳 + 日志级别
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)

    # 屏蔽第三方库的 HTTP 请求日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    return log_file, run_id


# ============================================================
# 进度追踪器
# ============================================================
class ProgressTracker:
    """原子 JSON 进度文件写入器，供 monitor.py 读取"""

    def __init__(self, progress_path, run_id, mode, model_info, cli_args):
        self.progress_path = progress_path
        self.state = {
            "experiment": {
                "run_id": run_id,
                "mode": mode,
                "pid": os.getpid(),
                "status": "running",
                "started_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "model_provider": model_info.get("provider", ""),
                "vlm_model": model_info.get("vlm", ""),
                "text_model": model_info.get("text", ""),
                "max_workers_outer": model_info.get("max_workers_outer", 2),
                "cli_args": cli_args,
            },
            "questions": {},
        }
        self._flush()

    def register_question(self, q_id, total_students):
        """开始处理某题前调用"""
        self.state["questions"][q_id] = {
            "total_students": total_students,
            "completed": 0,
            "failed": 0,
            "remaining": total_students,
            "started_at": datetime.now().isoformat(),
            "last_student_at": None,
            "eta_seconds": None,
            "avg_seconds_per_student": None,
            "route_distribution": {},
            "current_errors": [],
            "recent_completions": [],
            "_completion_times": [],  # 内部计时，不写入 JSON
        }
        self._flush()

    def record_completion(self, q_id, student_id, result, duration_seconds):
        """每个学生成功完成后调用"""
        q_state = self.state["questions"][q_id]
        q_state["completed"] += 1
        q_state["remaining"] -= 1
        q_state["last_student_at"] = datetime.now().isoformat()

        route = result.get("3wd_route", "UNKNOWN")
        q_state["route_distribution"][route] = q_state["route_distribution"].get(route, 0) + 1

        # 计时与 ETA
        q_state["_completion_times"].append(duration_seconds)
        if len(q_state["_completion_times"]) >= 2:
            avg = sum(q_state["_completion_times"]) / len(q_state["_completion_times"])
            q_state["avg_seconds_per_student"] = round(avg, 1)
            q_state["eta_seconds"] = round(q_state["remaining"] * avg)

        # 最近完成（保留最后 5 条）
        entry = {
            "student_id": student_id,
            "route": route,
            "final_score": result.get("final_calibrated_score"),
            "teacher_score": result.get("teacher_score"),
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": round(duration_seconds, 1),
        }
        q_state["recent_completions"] = (q_state["recent_completions"] + [entry])[-5:]

        self.state["experiment"]["last_updated"] = datetime.now().isoformat()
        self._flush()

    def record_error(self, q_id, student_id, error_msg):
        """学生批改失败后调用"""
        q_state = self.state["questions"][q_id]
        q_state["failed"] += 1
        q_state["remaining"] -= 1
        q_state["last_student_at"] = datetime.now().isoformat()
        error_entry = {
            "student_id": student_id,
            "error": str(error_msg)[:200],
            "at": datetime.now().isoformat(),
        }
        q_state["current_errors"] = (q_state["current_errors"] + [error_entry])[-3:]
        self.state["experiment"]["last_updated"] = datetime.now().isoformat()
        self._flush()

    def mark_question_done(self, q_id):
        """某题全部处理完后调用"""
        q_state = self.state["questions"][q_id]
        if q_state["remaining"] <= 0:
            q_state["eta_seconds"] = 0
        self._flush()

    def mark_finished(self, status="completed"):
        """整个运行结束时调用"""
        self.state["experiment"]["status"] = status
        self.state["experiment"]["last_updated"] = datetime.now().isoformat()
        self._flush()

    def _flush(self):
        """原子写入：写临时文件 → rename"""
        output = {
            "experiment": self.state["experiment"],
            "questions": {},
        }
        for q_id, q_data in self.state["questions"].items():
            q_out = {k: v for k, v in q_data.items() if not k.startswith("_")}
            output["questions"][q_id] = q_out

        dir_name = os.path.dirname(self.progress_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name or ".", suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.progress_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# ============================================================
# CLI 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="RefGrader 自动阅卷主流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main_pipeline.py --mode FULL --questions Q5 Q6 Q7
  python main_pipeline.py --mode VARIANCE_OPT --questions Q1 Q2 --sample-size 5
  python main_pipeline.py --mode FULL --questions Q4 --img-limit 10 --force-rerun
  python main_pipeline.py --mode FULL --questions Q1 --student-ids E12314093 E12214171
        """,
    )
    parser.add_argument(
        "--mode", choices=["FULL", "VARIANCE_OPT", "OCR_ONLY", "GRADE_ONLY"], default="FULL",
        help=(
            "FULL=提取并评分, VARIANCE_OPT=方差优化, "
            "OCR_ONLY=只生成提取缓存, GRADE_ONLY=只读取缓存评分"
        ),
    )
    parser.add_argument(
        "--questions", nargs="+", default=None,
        help="要处理的题号，空格分隔 (例: Q5 Q6 Q7)。不指定则处理全部题目。",
    )
    parser.add_argument(
        "--img-limit", type=int, default=None,
        help="每题最多批改的试卷数 (默认: 全量)",
    )
    parser.add_argument(
        "--student-ids", nargs="+", default=None,
        help="指定学号列表，只批改这些学生 (例: E12314093 E12214171)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=5,
        help="VARIANCE_OPT 模式的采样数量 (默认: 5)",
    )
    parser.add_argument(
        "--force-rerun", action="store_true",
        help="忽略检查点，强制从头重跑",
    )
    parser.add_argument(
        "--progress-file", default=None,
        help="进度 JSON 文件路径 (默认: results_rrd_vlm/progress.json)",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Result output directory. Default: results_rrd_vlm",
    )
    parser.add_argument(
        "--rubric-dir", default=None,
        help=(
            "Directory for optimized rubric files. VARIANCE_OPT writes here; "
            "FULL/GRADE_ONLY read here. Default: results-dir"
        ),
    )
    parser.add_argument(
        "--initial-rubric-dir", default=None,
        help=(
            "Optional directory containing immutable initial rubrics used to "
            "seed VARIANCE_OPT."
        ),
    )
    parser.add_argument(
        "--allow-initial-rubric", action="store_true",
        help=(
            "Allow OCR_ONLY/FULL/GRADE_ONLY to fall back to the initial rubric "
            "when no optimized rubric exists. Intended for smoke tests only."
        ),
    )
    parser.add_argument(
        "--log-dir", default="logs",
        help="日志文件目录 (默认: logs)",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="运行标识符，用于日志和进度文件命名 (默认: 自动生成时间戳)",
    )
    parser.add_argument(
        "--extraction-backend",
        choices=["glm_vlm", "paddle_glm5", "csbench_hybrid"],
        default="glm_vlm",
        help="Stage-1 extraction backend. Default: glm_vlm.",
    )
    parser.add_argument(
        "--ocr-cache-dir",
        default="ocr_cache",
        help="Root directory for raw OCR and mapped fact caches.",
    )
    parser.add_argument(
        "--database-path",
        default=DATABASE_PATH,
        help="Question database JSON. Default: database/exam_database.json",
    )
    parser.add_argument(
        "--teacher-db",
        default=TEACHER_DB_PATH,
        help="Teacher score JSON. Default: database/teacher_scores.json",
    )
    parser.add_argument(
        "--answer-metadata",
        default=None,
        help="Optional CSBench answer_metadata.jsonl containing raw_text.",
    )
    parser.add_argument(
        "--answer-split",
        choices=["all", "calibration", "validation", "test"],
        default="all",
        help=(
            "Optional per-question answer partition. Use test for final CSBench "
            "experiments. Default: all."
        ),
    )
    return parser.parse_args()


# ============================================================
# 辅助函数
# ============================================================
def get_text_model_display():
    provider_map = {"glm": GLM_MODEL_NAME, "glm5": GLM5_MODEL_NAME, "deepseek": DEEPSEEK_MODEL_NAME}
    actual = provider_map.get(TEXT_MODEL_PROVIDER, "unknown")
    return f"{TEXT_MODEL_PROVIDER} -> {actual}"


def question_output_paths(q_id):
    return {
        "checkpoint": os.path.join(OUTPUT_DIR, f"{q_id}_grading_checkpoint.json"),
        "graded": os.path.join(OUTPUT_DIR, f"{q_id}_graded_results.json"),
        "rejected": os.path.join(OUTPUT_DIR, f"{q_id}_rejected.json"),
        "failed": os.path.join(OUTPUT_DIR, f"{q_id}_failed.json"),
    }


def load_json_list(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        logging.warning(f"[result hygiene] failed to read JSON list: {path}")
        return []


def save_json_list(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(temporary, path)


def upsert_failed_record(records, record):
    student_id = str(record.get("student_id", ""))
    if not student_id:
        return records + [record]
    return [
        item
        for item in records
        if str(item.get("student_id", "")) != student_id
    ] + [record]


def remove_failed_record(records, student_id):
    student_id = str(student_id)
    return [
        item
        for item in records
        if str(item.get("student_id", "")) != student_id
    ]


def save_failed_records(path, records):
    if records:
        save_json_list(path, records)
    elif os.path.exists(path):
        os.remove(path)


def cleanup_question_outputs(q_id):
    paths = question_output_paths(q_id)
    removed = []
    for key in ("checkpoint", "graded", "rejected", "failed"):
        path = paths[key]
        if os.path.exists(path):
            os.remove(path)
            removed.append(os.path.basename(path))
    if removed:
        logging.info(f"[force-rerun] cleaned old result files for {q_id}: {', '.join(removed)}")


def make_failed_record(q_id, img_file, error_type, reason, attempts=2):
    file_base_name = os.path.splitext(img_file)[0]
    return {
        "question_id": q_id,
        "student_id": file_base_name,
        "image_file": img_file,
        "error_type": error_type,
        "attempts": attempts,
        "reason": str(reason),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def validate_question_outputs(q_id, expected_count):
    paths = question_output_paths(q_id)
    checkpoint = load_json_list(paths["checkpoint"])
    graded = load_json_list(paths["graded"])
    rejected = load_json_list(paths["rejected"])
    failed = load_json_list(paths["failed"])

    def ids(records):
        return {str(r.get("student_id", "")) for r in records if isinstance(r, dict) and r.get("student_id")}

    checkpoint_ids = ids(checkpoint)
    graded_ids = ids(graded)
    rejected_ids = ids(rejected)
    failed_ids = ids(failed)
    overlap = graded_ids & rejected_ids
    union = graded_ids | rejected_ids
    missing_from_split = checkpoint_ids - union
    extra_in_split = union - checkpoint_ids
    accounted = len(checkpoint_ids) + len(failed_ids)

    warnings = []
    if overlap:
        warnings.append(f"graded/rejected overlap={len(overlap)}")
    if missing_from_split or extra_in_split:
        warnings.append(
            f"checkpoint/split mismatch: missing_from_split={len(missing_from_split)}, extra_in_split={len(extra_in_split)}"
        )
    if accounted != expected_count:
        warnings.append(f"success+failed={accounted} != expected={expected_count}")

    if warnings:
        logging.warning(
            f"[RESULT CONSISTENCY WARNING] {q_id}: "
            f"checkpoint={len(checkpoint_ids)}, graded={len(graded_ids)}, "
            f"rejected={len(rejected_ids)}, failed={len(failed_ids)}, expected={expected_count}; "
            + "; ".join(warnings)
        )
    else:
        logging.info(
            f"[result consistency] {q_id}: checkpoint={len(checkpoint_ids)}, "
            f"graded={len(graded_ids)}, rejected={len(rejected_ids)}, failed={len(failed_ids)}, expected={expected_count}"
        )


def rubric_group_for(q_id, q_data=None):
    if q_data and q_data.get("rubric_group"):
        return str(q_data["rubric_group"])
    return str(q_id).split("_", 1)[0]


def rubric_candidates(base_dir, q_id, q_data=None):
    rubric_name = f"{q_id}_rubric_standard.json"
    group = rubric_group_for(q_id, q_data)
    return [
        os.path.join(base_dir, group, rubric_name),
        os.path.join(base_dir, rubric_name),
    ]


def first_existing_path(paths):
    return next((path for path in paths if path and os.path.exists(path)), None)


def initial_rubric_path_for(q_data):
    q_id = q_data["question_id"]
    explicit = q_data.get("initial_rubric_path")
    candidates = [explicit] if explicit else []
    if INITIAL_RUBRIC_DIR:
        candidates.extend(rubric_candidates(INITIAL_RUBRIC_DIR, q_id, q_data))
    return first_existing_path(candidates)


def optimized_rubric_output_path(q_data):
    q_id = q_data["question_id"]
    return rubric_candidates(RUBRIC_DIR, q_id, q_data)[0]


def optimization_manifest_path(q_data):
    q_id = q_data["question_id"]
    root = (
        os.path.join(os.path.dirname(os.path.abspath(RUBRIC_DIR)), "manifests")
        if os.path.basename(os.path.normpath(RUBRIC_DIR)).lower() == "optimized"
        else os.path.join(RUBRIC_DIR, "manifests")
    )
    return os.path.join(
        root,
        rubric_group_for(q_id, q_data),
        f"{q_id}_optimization.json",
    )


def validate_optimized_rubric_provenance(q_data, rubric_path):
    if not INITIAL_RUBRIC_DIR:
        return
    manifest_path = optimization_manifest_path(q_data)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Optimized rubric manifest is missing for "
            f"{q_data['question_id']}: {manifest_path}. Run VARIANCE_OPT first."
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        manifest.get("rubric_semantic_contract_version")
        != RUBRIC_SEMANTIC_CONTRACT_VERSION
    ):
        raise ValueError(
            f"Optimized rubric for {q_data['question_id']} uses an obsolete "
            "semantic contract. Re-run VARIANCE_OPT."
        )
    if manifest.get("semantic_policy_validated") is not True:
        raise ValueError(
            f"Optimized rubric for {q_data['question_id']} has no successful "
            "semantic-policy validation. Re-run VARIANCE_OPT."
        )
    initial_path = initial_rubric_path_for(q_data)
    expected_initial_hash = sha256_path(initial_path)
    if manifest.get("initial_sha256") != expected_initial_hash:
        raise ValueError(
            f"Optimized rubric for {q_data['question_id']} was produced from "
            "a different initial rubric. Re-run VARIANCE_OPT."
        )
    if manifest.get("optimized_sha256") != sha256_path(rubric_path):
        raise ValueError(
            f"Optimized rubric hash mismatch for {q_data['question_id']}. "
            "Re-run VARIANCE_OPT or restore the recorded file."
        )


def rubric_path_for(q_id, q_data=None):
    rubric_name = f"{q_id}_rubric_standard.json"
    primary_candidates = rubric_candidates(RUBRIC_DIR, q_id, q_data)
    existing = first_existing_path(primary_candidates)
    if existing:
        if q_data:
            validate_optimized_rubric_provenance(q_data, existing)
        return existing

    if ALLOW_INITIAL_RUBRIC and q_data:
        initial = initial_rubric_path_for(q_data)
        if initial:
            logging.warning(
                f"[rubric fallback] optimized rubric missing for {q_id}; "
                f"using initial rubric for this explicitly allowed run: {initial}"
            )
            return initial

    primary = primary_candidates[0]
    fallback = os.path.join(DEFAULT_OUTPUT_DIR, rubric_name)
    if os.path.abspath(primary) != os.path.abspath(fallback) and os.path.exists(fallback):
        logging.warning(
            f"[rubric fallback] {primary} not found; using existing rubric from {fallback}"
        )
        return fallback
    return primary


def sha256_path(path):
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rubric_total(rubric):
    return sum(float(item.get("points", 0)) for item in rubric or [])


def load_calibration_ids(q_data):
    ids = q_data.get("rubric_calibration_ids")
    if isinstance(ids, list):
        return [str(value) for value in ids]
    split_path = q_data.get("rubric_split_path")
    if split_path and os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [str(value) for value in payload.get("calibration", [])]
    return []


def answer_ids_for_split(q_data, split_name):
    if not split_name or split_name == "all":
        return None
    if split_name == "calibration":
        return set(load_calibration_ids(q_data))
    split_path = q_data.get("rubric_split_path")
    if not split_path or not os.path.exists(split_path):
        raise FileNotFoundError(
            f"{q_data['question_id']} has no split metadata for {split_name}."
        )
    with open(split_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(value) for value in payload.get(split_name, [])}


def select_question_images(image_files, img_limit, q_data, answer_split="all"):
    allowed_ids = answer_ids_for_split(q_data, answer_split)
    if allowed_ids is not None:
        image_files = [
            filename
            for filename in image_files
            if os.path.splitext(filename)[0] in allowed_ids
        ]
    return selected_image_files(image_files, img_limit)


def get_teacher_score_from_your_database(student_id, q_id):
    """数据接口：从 step0 生成的 JSON 数据库中获取教师评分"""
    global _GLOBAL_SCORES_DB

    if _GLOBAL_SCORES_DB is None:
        db_path = TEACHER_DB_PATH
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                _GLOBAL_SCORES_DB = json.load(f)
            logging.info(f"📦 成功加载教师真实成绩单数据库！共包含 {len(_GLOBAL_SCORES_DB)} 名考生的数据。")
        else:
            logging.warning(f"⚠️ 严重警告：找不到成绩单 {db_path}，请先运行 step0_extract_ground_truth.py！")
            _GLOBAL_SCORES_DB = {}

    student_record = _GLOBAL_SCORES_DB.get(student_id, {})
    score = student_record.get(q_id, 0.0)
    return float(score)


def load_answer_metadata():
    """Load optional answer metadata keyed by complete answer_id."""
    global _GLOBAL_ANSWER_METADATA
    if _GLOBAL_ANSWER_METADATA is not None:
        return _GLOBAL_ANSWER_METADATA
    _GLOBAL_ANSWER_METADATA = {}
    if not ANSWER_METADATA_PATH:
        return _GLOBAL_ANSWER_METADATA
    if not os.path.exists(ANSWER_METADATA_PATH):
        logging.warning(f"[metadata] answer metadata not found: {ANSWER_METADATA_PATH}")
        return _GLOBAL_ANSWER_METADATA
    with open(ANSWER_METADATA_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            answer_id = str(record.get("answer_id", ""))
            if answer_id:
                _GLOBAL_ANSWER_METADATA[answer_id] = record
    logging.info(
        f"[metadata] loaded {len(_GLOBAL_ANSWER_METADATA)} answer records "
        f"from {ANSWER_METADATA_PATH}"
    )
    return _GLOBAL_ANSWER_METADATA


def answer_metadata_for(answer_id):
    return load_answer_metadata().get(str(answer_id), {})


def selected_image_files(image_files, img_limit):
    """Select images by complete filename stem; preserve legacy student IDs."""
    if isinstance(img_limit, list):
        target_ids = {str(value) for value in img_limit}
        return [
            filename
            for filename in image_files
            if os.path.splitext(filename)[0] in target_ids
            or os.path.splitext(filename)[0].split("_")[0] in target_ids
        ]
    if isinstance(img_limit, int):
        return image_files[:img_limit]
    return image_files


def sample_needs_ocr(extraction_backend, q_data, image_filename):
    if extraction_backend == "paddle_glm5":
        return True
    if extraction_backend != "csbench_hybrid":
        return False
    if not q_data.get("requires_visual_evidence"):
        return False
    answer_id = os.path.splitext(image_filename)[0]
    metadata = answer_metadata_for(answer_id)
    return bool(
        metadata.get("isimagine")
        or metadata.get("visual_placeholder_detected")
    )


os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 核心功能 A：基于方差的自动优化流程 (用于测试/打磨标准)
# ============================================================
def run_variance_optimization_process(
    q_data,
    sample_size=3,
    progress_tracker=None,
    force_rerun=False,
    extraction_backend="glm_vlm",
    ocr_cache_dir="ocr_cache",
):
    q_id = q_data["question_id"]
    q_score = q_data["total_score"]
    q_text = q_data["question_text"]
    q_img = q_data.get("question_image")
    ref_text = q_data["ref_text"]
    ref_img = q_data.get("ref_image")
    images_folder = q_data["student_images_dir"]
    official_rubric = q_data.get("official_rubric", "")

    rubric_save_path = optimized_rubric_output_path(q_data)
    initial_rubric_path = initial_rubric_path_for(q_data)
    checkpoint_path = os.path.join(OUTPUT_DIR, f"{q_id}_variance_checkpoint.json")
    manifest_path = optimization_manifest_path(q_data)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(rubric_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    logging.info(f"\n{'='*60}\n🔬 [断点续传模式] 处理题目: {q_id}\n{'='*60}")

    # --- Step 1: 加载或生成初始标准 ---
    draft_rubric = None
    rubric_regenerated = False
    if force_rerun and os.path.exists(rubric_save_path):
        logging.info(
            "force-rerun enabled; restart from the immutable initial rubric "
            "and ignore the existing optimized rubric."
        )
    elif os.path.exists(rubric_save_path):
        with open(rubric_save_path, "r", encoding="utf-8") as f:
            draft_rubric = json.load(f)
        logging.info("Existing rubric found; skip initial generation.")
        if not isinstance(draft_rubric, list) or not draft_rubric:
            logging.warning("Existing rubric file is empty or invalid; regenerating it.")
            draft_rubric = None

    if draft_rubric is None and initial_rubric_path:
        with open(initial_rubric_path, "r", encoding="utf-8") as handle:
            draft_rubric = json.load(handle)
        if not isinstance(draft_rubric, list) or not draft_rubric:
            raise ValueError(f"Invalid initial rubric: {initial_rubric_path}")
        logging.info(f"Loaded immutable initial rubric: {initial_rubric_path}")
        rubric_regenerated = True

    if draft_rubric is None:
        logging.info(
            "No external initial rubric configured; generating the legacy "
            "initial rubric draft."
        )
        draft_rubric = generate_rrd_rubrics(
            q_text, ref_text, official_rubric, q_score, q_img, ref_img, None
        )
        if not isinstance(draft_rubric, list) or not draft_rubric:
            logging.error(f"Failed to generate a valid rubric for {q_id}; stop VARIANCE_OPT for this question.")
            if progress_tracker:
                progress_tracker.record_error(q_id, "__rubric__", "rubric_generation_failed")
            return
        rubric_regenerated = True
        logging.info("Initial rubric generated; sleep 2 seconds before sampling.")
        time.sleep(2)

    draft_rubric = prepare_rubric_semantic_contract(draft_rubric)
    initial_total = rubric_total(draft_rubric)
    if abs(initial_total - float(q_score)) > 1e-6:
        raise ValueError(
            f"{q_id} initial rubric total {initial_total} does not match "
            f"question total_score {q_score}."
        )
    # Keep the active optimized rubric untouched until the candidate has passed
    # every structural and semantic check.

    # --- Step 2: 加载已有的方差探测进度 ---
    hard_samples_info = []
    processed_files = set()
    if force_rerun:
        logging.info("force-rerun enabled; ignore old variance checkpoint for this question.")
    elif rubric_regenerated:
        logging.info("Rubric was regenerated; ignore old variance checkpoint for this question.")
    elif os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            hard_samples_info = json.load(f)
            processed_files = {s["file"] for s in hard_samples_info}
        logging.info(f"📈 发现进度点：已完成 {len(processed_files)}/{sample_size} 个样本。")

    # --- Step 3: 探测方差 ---
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    calibration_ids = load_calibration_ids(q_data)
    if calibration_ids:
        calibration_set = set(calibration_ids)
        image_files = [
            filename
            for filename in image_files
            if os.path.splitext(filename)[0] in calibration_set
        ]
        logging.info(
            f"[data isolation] variance optimization is restricted to "
            f"{len(image_files)} calibration answers."
        )
    else:
        logging.warning(
            f"[data isolation] {q_id} has no calibration split metadata; "
            "using the legacy sorted-image sampling behavior."
        )
    allowed_files = set(image_files)
    if processed_files - allowed_files:
        hard_samples_info = [
            sample
            for sample in hard_samples_info
            if sample.get("file") in allowed_files
        ]
        processed_files = {
            sample["file"]
            for sample in hard_samples_info
            if sample.get("file")
        }
        logging.warning(
            f"[data isolation] removed checkpoint samples outside the current "
            f"{q_id} calibration split."
        )
    effective_sample_size = min(sample_size, len(image_files))
    if effective_sample_size < sample_size:
        logging.warning(
            f"{q_id} requested {sample_size} calibration samples, but only "
            f"{effective_sample_size} are available."
        )

    if progress_tracker:
        progress_tracker.register_question(q_id, effective_sample_size)

    remaining_needed = effective_sample_size - len(processed_files)
    if remaining_needed <= 0:
        logging.info("✅ 方差采样已全部完成，直接进入修正环节。")
    else:
        targets = [f for f in image_files if f not in processed_files][:remaining_needed]

        for img_file in targets:
            if _shutdown_requested:
                break

            _sample_start = time.time()
            img_path = os.path.join(images_folder, img_file)
            student_id = os.path.splitext(img_file)[0]
            scores = []

            logging.info(f"\n👉 正在处理新样本: {img_file}")

            blind_checklist = generate_blind_checklist(json.dumps(draft_rubric, ensure_ascii=False))
            logging.info("   ⏳ [V0 保护] 脱敏清单生成完毕，休眠 2 秒...")
            time.sleep(2)

            logging.info("   [单次视觉采样] 正在看图提取事实...")
            raw_ocr_dir = os.path.join(ocr_cache_dir, q_id)
            facts_cache_dir = os.path.join(
                ocr_cache_dir, "variance_facts", q_id
            )
            os.makedirs(raw_ocr_dir, exist_ok=True)
            os.makedirs(facts_cache_dir, exist_ok=True)
            if sample_needs_ocr(extraction_backend, q_data, img_file):
                ensure_paddle_ocr_cache(
                    img_path,
                    raw_ocr_dir,
                    # Force grading/rubric outputs, but reuse deterministic OCR
                    # whenever the cached image hash is still current.
                    force=False,
                )
            metadata = answer_metadata_for(student_id)
            current_facts, extraction_evidence = stage1_extract_with_backend(
                question_text=q_text,
                student_img_path=img_path,
                blind_checklist=blind_checklist,
                rubrics_json=json.dumps(draft_rubric, ensure_ascii=False),
                q_img_path=q_img,
                extraction_backend=extraction_backend,
                ocr_json_path=str(ocr_json_path(raw_ocr_dir, img_path)),
                extraction_cache_path=os.path.join(
                    facts_cache_dir, f"{student_id}.json"
                ),
                force_extraction=True,
                student_transcription=metadata.get("raw_text"),
                answer_metadata=metadata,
            )

            if not current_facts:
                logging.warning("   ⚠️ 视觉提取失败，跳过...")
                if progress_tracker:
                    student_id = os.path.splitext(img_file)[0]
                    progress_tracker.record_error(q_id, student_id, "视觉提取失败")
                continue

            logging.info("   ⏳ [V0 保护] 视觉提取完成，准备进入打分循环，休眠 2 秒...")
            time.sleep(2)

            strict_cots = []
            canonical_context = build_canonical_grading_context(
                current_facts,
                draft_rubric,
            )
            for i in range(3):
                logging.info(f"   [第 {i+1}/3 次判决] 呼叫逻辑裁判...")
                res_text = stage2_logic_grading(current_facts, json.dumps(draft_rubric, ensure_ascii=False))

                if res_text:
                    parsed = extract_and_parse_json(res_text)
                    if parsed and 'total_score' in parsed:
                        parsed = apply_canonical_score_floor(
                            parsed,
                            canonical_context,
                        )
                        parsed = apply_hierarchical_scoring_policy(
                            parsed,
                            draft_rubric,
                            canonical_context,
                        )
                        scores.append(parsed['total_score'])
                        strict_cots.append(parsed)
                        logging.info(f"      ✅ [裁判亮分] 总得分: {parsed['total_score']}")
                        if 'details' in parsed:
                            for detail in parsed.get('details', []):
                                logging.info(f"         - [条款 {detail.get('id', '?')}] 得分: {detail.get('score_given', 0)} | 理由: {detail.get('reason', '')}")

                if i < 2:
                    logging.info("   ⏳ [V0 保护] 判决完成，休眠 2 秒...")
                    time.sleep(2)

            if len(scores) >= 2:
                item_scores_history = {}
                item_category_history = {}
                for cot in strict_cots:
                    for detail in cot.get("details", []):
                        item_id = str(detail.get("id", ""))
                        if not item_id:
                            continue
                        try:
                            item_score = float(detail.get("score_given", 0))
                        except Exception:
                            item_score = 0.0
                        item_scores_history.setdefault(item_id, []).append(item_score)
                        item_category_history.setdefault(item_id, []).append(str(detail.get("error_category", "")))
                item_variance = {
                    item_id: float(np.var(values))
                    for item_id, values in item_scores_history.items()
                    if len(values) >= 2
                }
                sample_data = {
                    "file": img_file,
                    "facts": current_facts,
                    "extraction_evidence": extraction_evidence,
                    "scores": scores,
                    "strict_cots": strict_cots,
                    "item_scores_history": item_scores_history,
                    "item_category_history": item_category_history,
                    "item_variance": item_variance,
                    "max_item_variance": max(item_variance.values()) if item_variance else 0.0,
                    "avg_item_variance": float(np.mean(list(item_variance.values()))) if item_variance else 0.0,
                }
                hard_samples_info.append(sample_data)

                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(hard_samples_info, f, indent=4, ensure_ascii=False)
                logging.info(f"💾 样本 {img_file} 进度已保存。")

                if progress_tracker:
                    student_id = os.path.splitext(img_file)[0]
                    progress_tracker.record_completion(q_id, student_id, {"3wd_route": "VARIANCE"}, time.time() - _sample_start)

                logging.info("⏳ 样本间冷却 15 秒...")
                time.sleep(15)

    # --- Step 4: 智能修正 ---
    all_scores = [s["scores"] for s in hard_samples_info]
    variances = [np.var(s) for s in all_scores if len(s) > 1]
    avg_variance = np.mean(variances) if variances else 0
    item_variances = [
        float(v)
        for sample in hard_samples_info
        for v in (sample.get("item_variance") or {}).values()
    ]
    avg_item_variance = float(np.mean(item_variances)) if item_variances else 0.0
    max_item_variance = max(item_variances) if item_variances else 0.0
    logging.info(f"\n📊 采样完成。平均方差: {avg_variance:.4f}")
    logging.info(f"Item-level variance: avg={avg_item_variance:.4f}, max={max_item_variance:.4f}")

    mandatory_split_targets = high_value_split_targets(draft_rubric)
    has_refinement_candidate = bool(mandatory_split_targets)
    final_rubric = draft_rubric
    refinement_applied = False
    structural_refinement_applied = False
    metadata_enriched = False
    refinement_attempted = False

    if avg_variance > 0.1 or avg_item_variance > 0.05 or has_refinement_candidate:
        if avg_variance > 0.1:
            logging.warning("⚠️ 方差超标！开始基于高方差样本修正标准...")
        elif avg_item_variance > 0.05:
            logging.warning("⚠️ Item-level 方差超标，开始定位不稳定评分项...")
        else:
            logging.warning(
                "⚠️ 触发语义粗粒度检查：存在允许细化的复合高分条款；"
                "原子结果项保持完整。"
            )

        for sample in hard_samples_info:
            scores = sample.get('scores', [])
            sample['variance'] = np.var(scores) if len(scores) > 1 else 0.0
            item_values = list((sample.get("item_variance") or {}).values())
            sample["max_item_variance"] = max(item_values) if item_values else sample.get("max_item_variance", 0.0)
            sample["avg_item_variance"] = float(np.mean(item_values)) if item_values else sample.get("avg_item_variance", 0.0)

        sorted_samples = sorted(
            hard_samples_info,
            key=lambda x: (x.get("max_item_variance", 0.0), x.get("variance", 0.0)),
            reverse=True,
        )
        TOP_N = 3
        bad_samples = [
            s for s in sorted_samples[:TOP_N]
            if s.get("max_item_variance", 0.0) > 0 or s.get("variance", 0.0) > 0
        ]

        if not bad_samples:
            bad_samples = sorted_samples[:2]

        logging.info(f"🔧 [规则修正] 正在基于 {len(bad_samples)} 份精选样本优化规则...")
        if mandatory_split_targets:
            logging.info(
                "[rubric structure] mandatory high-value split targets: "
                + json.dumps(mandatory_split_targets, ensure_ascii=False)
            )

        refinement_attempted = True
        refined_rubric = refine_rubric_based_on_variance(
            draft_rubric, q_text, q_score, bad_samples
        )

        if refined_rubric:
            refined_valid, refined_errors = validate_refined_rubric(
                draft_rubric,
                refined_rubric,
                q_score,
            )
            if not refined_valid:
                logging.error(
                    "Refined rubric rejected by semantic contract: "
                    + "; ".join(refined_errors)
                )
            else:
                final_rubric = prepare_rubric_semantic_contract(refined_rubric)
                refinement_applied = (
                    rubric_scoring_signature(final_rubric)
                    != rubric_scoring_signature(draft_rubric)
                )
                structural_refinement_applied = (
                    rubric_structure_signature(final_rubric)
                    != rubric_structure_signature(draft_rubric)
                )
                metadata_enriched = bool(
                    final_rubric != draft_rubric and not refinement_applied
                )
                logging.info("🎉 修正后的最终标准已通过总分与语义契约校验。")
        else:
            logging.error("❌ 修正请求失败或 JSON 解析错误，保留原草稿。")
    else:
        logging.info("✅ 标准足够稳定且粒度精细，无需进一步修正。")

    semantic_policy_validated, semantic_validation_errors = validate_refined_rubric(
        draft_rubric,
        final_rubric,
        q_score,
    )
    if not semantic_policy_validated:
        logging.error(
            "Final optimized rubric failed semantic validation; reverting to "
            "the immutable draft: " + "; ".join(semantic_validation_errors)
        )
        final_rubric = prepare_rubric_semantic_contract(draft_rubric)
        refinement_applied = False
        semantic_policy_validated, semantic_validation_errors = validate_refined_rubric(
            draft_rubric,
            final_rubric,
            q_score,
        )
    if not semantic_policy_validated:
        raise RuntimeError(
            f"Refusing to save an invalid optimized rubric for {q_id}: "
            + "; ".join(semantic_validation_errors)
        )

    save_json_list(rubric_save_path, final_rubric)
    manifest = {
        "schema_version": 1,
        "rubric_semantic_contract_version": RUBRIC_SEMANTIC_CONTRACT_VERSION,
        "question_id": q_id,
        "rubric_group": rubric_group_for(q_id, q_data),
        "source_rubric": q_data.get("source_rubric_path"),
        "initial_rubric": initial_rubric_path,
        "optimized_rubric": os.path.abspath(rubric_save_path),
        "calibration_answer_ids": [
            os.path.splitext(sample["file"])[0]
            for sample in hard_samples_info
        ],
        "requested_sample_size": sample_size,
        "completed_sample_size": len(hard_samples_info),
        "extraction_backend": extraction_backend,
        "question_total_score": float(q_score),
        "initial_total_score": initial_total,
        "optimized_total_score": rubric_total(final_rubric),
        "average_score_variance": float(avg_variance),
        "average_item_variance": avg_item_variance,
        "maximum_item_variance": max_item_variance,
        "high_value_split_threshold": HIGH_VALUE_SPLIT_THRESHOLD,
        "mandatory_split_targets": mandatory_split_targets,
        "refinement_attempted": refinement_attempted,
        "refinement_applied": refinement_applied,
        "structural_refinement_applied": structural_refinement_applied,
        "metadata_enriched": metadata_enriched,
        "semantic_policy_validated": semantic_policy_validated,
        "semantic_validation_errors": semantic_validation_errors,
        "scoring_policies": sorted(
            {
                str(item.get("scoring_policy", ""))
                for item in final_rubric
                if item.get("scoring_policy")
            }
        ),
        "source_sha256": sha256_path(q_data.get("source_rubric_path")),
        "initial_sha256": sha256_path(initial_rubric_path),
        "optimized_sha256": sha256_path(rubric_save_path),
        "optimization_results_dir": os.path.abspath(OUTPUT_DIR),
        "variance_checkpoint": (
            os.path.abspath(checkpoint_path)
            if os.path.exists(checkpoint_path)
            else None
        ),
        "variance_checkpoint_sha256": (
            sha256_path(checkpoint_path)
            if os.path.exists(checkpoint_path)
            else None
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    logging.info(f"[rubric manifest] saved: {manifest_path}")

    if progress_tracker:
        progress_tracker.mark_question_done(q_id)


# ============================================================
# 核心功能 B：标准批改流程 (保留原有的逻辑)
# ============================================================
def process_ocr_only_question(
    q_data,
    img_limit=None,
    answer_split="all",
    force_rerun=False,
    progress_tracker=None,
    extraction_backend="paddle_glm5",
    ocr_cache_dir="ocr_cache",
):
    """Generate auditable Stage-1 facts without running semantic grading."""
    q_id = q_data["question_id"]
    q_text = q_data["question_text"]
    q_img = q_data.get("question_image")
    images_folder = q_data["student_images_dir"]
    with open(rubric_path_for(q_id, q_data), "r", encoding="utf-8") as handle:
        dynamic_rubrics = json.load(handle)
    rubrics_json = json.dumps(dynamic_rubrics, ensure_ascii=False)

    all_image_files = sorted(
        f for f in os.listdir(images_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    image_files = select_question_images(
        all_image_files, img_limit, q_data, answer_split
    )

    if progress_tracker:
        progress_tracker.register_question(q_id, len(image_files))

    raw_dir = os.path.join(ocr_cache_dir, q_id)
    facts_dir = os.path.join(ocr_cache_dir, "facts", q_id)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(facts_dir, exist_ok=True)

    ocr_image_files = [
        filename
        for filename in image_files
        if sample_needs_ocr(extraction_backend, q_data, filename)
    ]
    if ocr_image_files:
        if ocr_image_files == all_image_files:
            completed = ensure_paddle_ocr_cache(
                images_folder,
                raw_dir,
                force=force_rerun,
            )
            if completed.stdout.strip():
                logging.info(completed.stdout.strip())
        else:
            for img_file in ocr_image_files:
                img_path = os.path.join(images_folder, img_file)
                completed = ensure_paddle_ocr_cache(
                    img_path,
                    raw_dir,
                    force=force_rerun,
                )
                if completed.stdout.strip():
                    logging.info(completed.stdout.strip())

    blind_checklist = generate_blind_checklist(rubrics_json)
    records = []
    for img_file in image_files:
        started = time.time()
        img_path = os.path.join(images_folder, img_file)
        student_id = os.path.splitext(img_file)[0]
        metadata = answer_metadata_for(student_id)
        try:
            facts, evidence = stage1_extract_with_backend(
                question_text=q_text,
                student_img_path=img_path,
                blind_checklist=blind_checklist,
                rubrics_json=rubrics_json,
                q_img_path=q_img,
                extraction_backend=extraction_backend,
                ocr_json_path=str(ocr_json_path(raw_dir, img_path)),
                extraction_cache_path=os.path.join(facts_dir, f"{student_id}.json"),
                force_extraction=force_rerun,
                student_transcription=metadata.get("raw_text"),
                answer_metadata=metadata,
            )
            records.append({
                "question_id": q_id,
                "student_id": student_id,
                "extraction_backend": extraction_backend,
                "facts": extract_and_parse_json(facts) or facts,
                "extraction_cache_path": os.path.join(facts_dir, f"{student_id}.json"),
                "blank_authenticity": evidence.get("ocr_summary", {}).get("blank_authenticity"),
                "diagram_parser_used": evidence.get("diagram_parser_used", False),
                "visual_placeholder_detected": evidence.get(
                    "visual_placeholder_detected", False
                ),
            })
            if progress_tracker:
                progress_tracker.record_completion(
                    q_id,
                    student_id,
                    {"3wd_route": "OCR_ONLY"},
                    time.time() - started,
                )
        except Exception as exc:
            logging.error(f"[OCR_ONLY failed] {student_id}: {exc}")
            if progress_tracker:
                progress_tracker.record_error(q_id, student_id, exc)

    output_path = os.path.join(OUTPUT_DIR, f"{q_id}_ocr_only.json")
    save_json_list(output_path, records)
    logging.info(f"[OCR_ONLY] saved {len(records)} extraction records to {output_path}")
    if progress_tracker:
        progress_tracker.mark_question_done(q_id)


def process_single_question(
    q_data,
    img_limit=None,
    answer_split="all",
    generate_only=False,
    force_rerun=False,
    progress_tracker=None,
    extraction_backend="glm_vlm",
    ocr_cache_dir="ocr_cache",
    grade_only=False,
):
    """处理单道题目的完整流水线"""
    q_id = q_data["question_id"]
    q_score = q_data["total_score"]
    q_text = q_data["question_text"]
    q_img = q_data.get("question_image")
    ref_text = q_data["ref_text"]
    ref_img = q_data.get("ref_image")
    images_folder = q_data["student_images_dir"]
    official_rubric = q_data.get("official_rubric", "")

    logging.info(f"\n🚀 [标准模式] 开始处理: {q_id}")

    # 1. 生成/读取标准
    rubric_output_path = rubric_path_for(q_id, q_data)

    try:
        with open(rubric_output_path, "r", encoding="utf-8") as f:
            dynamic_rubrics = json.load(f)
        logging.info(f"⚡ 已加载本地标准: {rubric_output_path}")
    except FileNotFoundError:
        logging.error(f"❌ 找不到标准文件 {rubric_output_path}，请先运行 VARIANCE_OPT 模式生成！")
        return

    # 2. 刹车检查
    if generate_only:
        logging.info("🛑 [仅生成标准模式] 结束。")
        return

    # 3. 批改逻辑
    if not os.path.exists(images_folder):
        logging.warning(f"⚠️ 找不到文件夹: {images_folder}")
        return

    all_image_files = [
        f for f in os.listdir(images_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    all_image_files.sort()
    image_files = select_question_images(
        all_image_files, img_limit, q_data, answer_split
    )

    # 灵活选人：支持 int(前N张)、list(完整答案ID/旧学号)、None(全量)
    if isinstance(img_limit, list):
        logging.info(f"📷 指定学号模式: 筛选出 {len(image_files)} 张试卷")
    elif isinstance(img_limit, int):
        image_files = image_files[:img_limit]
        logging.info(f"📷 极速测试模式: 仅处理前 {img_limit} 张图片")

    raw_ocr_dir = os.path.join(ocr_cache_dir, q_id)
    facts_cache_dir = os.path.join(ocr_cache_dir, "facts", q_id)
    os.makedirs(raw_ocr_dir, exist_ok=True)
    os.makedirs(facts_cache_dir, exist_ok=True)
    ocr_image_files = [
        filename
        for filename in image_files
        if sample_needs_ocr(extraction_backend, q_data, filename)
    ]
    if ocr_image_files and not grade_only:
        logging.info(
            f"[PaddleOCR] preparing {len(ocr_image_files)}/{len(image_files)} "
            "visual-answer cache entries..."
        )
        if ocr_image_files == all_image_files:
            completed = ensure_paddle_ocr_cache(
                images_folder,
                raw_ocr_dir,
                force=False,
            )
            if completed.stdout.strip():
                logging.info(completed.stdout.strip())
        else:
            for img_file in ocr_image_files:
                completed = ensure_paddle_ocr_cache(
                    os.path.join(images_folder, img_file),
                    raw_ocr_dir,
                    force=False,
                )
                if completed.stdout.strip():
                    logging.info(completed.stdout.strip())

    # 注册进度追踪
    if progress_tracker:
        progress_tracker.register_question(q_id, len(image_files))

    # 断点续传：加载已完成的学生
    paths = question_output_paths(q_id)
    checkpoint_path = paths["checkpoint"]
    failed_path = paths["failed"]
    completed_ids = set()
    if force_rerun:
        cleanup_question_outputs(q_id)
    if not force_rerun and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            completed_ids = {r.get('student_id', '') for r in existing}
        logging.info(f"📈 断点续传: 发现 {len(completed_ids)} 份已完成的结果，跳过。")
    else:
        existing = []

    # 保存完整目标列表（用于最终排序）
    all_target_files = image_files[:]

    # 过滤掉已完成的学生
    remaining_files = [f for f in image_files if os.path.splitext(f)[0] not in completed_ids]
    if len(remaining_files) < len(image_files):
        logging.info(f"⏭️ 跳过 {len(image_files) - len(remaining_files)} 份已完成试卷，剩余 {len(remaining_files)} 份待处理。")
    image_files = remaining_files

    if not image_files:
        logging.info("✅ 所有试卷已处理完毕，无需重新批改。")
        results_list = existing
        normal_results = [r for r in results_list if r.get('3wd_route') != 'NEG']
        rejected_results = [r for r in results_list if r.get('3wd_route') == 'NEG']
        save_path = paths["graded"]
        save_json_list(save_path, normal_results)
        if rejected_results:
            rejected_path = paths["rejected"]
            save_json_list(rejected_path, rejected_results)
            logging.info(f"💾 最终结果已保存: {save_path} ({len(normal_results)} 人) + {rejected_path} ({len(rejected_results)} 人)")
        else:
            rejected_path = paths["rejected"]
            if os.path.exists(rejected_path):
                os.remove(rejected_path)
            logging.info(f"💾 最终结果已保存: {save_path} ({len(normal_results)} 人)")
        stale_failures = load_json_list(failed_path)
        stale_failures = [
            item
            for item in stale_failures
            if str(item.get("student_id", "")) not in completed_ids
        ]
        save_failed_records(failed_path, stale_failures)
        validate_question_outputs(q_id, expected_count=len(all_target_files))
        if progress_tracker:
            progress_tracker.mark_question_done(q_id)
        return

    results_list = []
    failed_results = []
    for failed_record in (
        load_json_list(failed_path) if not force_rerun else []
    ):
        failed_results = upsert_failed_record(
            failed_results, failed_record
        )
    total_count = 0

    # 预生成脱敏清单（同题共用，只调一次 API）
    cached_blind_checklist = None
    if not grade_only:
        cached_blind_checklist = generate_blind_checklist(
            json.dumps(dynamic_rubrics, ensure_ascii=False)
        )
        logging.info(f"⚡ 脱敏清单已预生成，{len(image_files)} 名学生共享复用。")
    else:
        logging.info("⚡ GRADE_ONLY: directly reuse mapped fact cache.")

    logging.info(f"🏃 开始批改 {len(image_files)} 张试卷 (⚡ 开启多线程模式)...")

    def process_one_student(img_file):
        file_base_name = os.path.splitext(img_file)[0]
        student_id = file_base_name
        img_path = os.path.join(images_folder, img_file)
        real_teacher_score = get_teacher_score_from_your_database(student_id, q_id)
        metadata = answer_metadata_for(student_id)
        _start_time = time.time()

        for attempt in range(2):
            logging.info(f"\n🔍 [正在处理] {file_base_name} | 启动 3WD 流水线..." + ("(重试)" if attempt > 0 else ""))

            try:
                res = grade_student_3wd_pipeline(
                    student_img_path=img_path,
                    question_text=q_text,
                    rubrics_json=json.dumps(dynamic_rubrics, ensure_ascii=False),
                    teacher_score=real_teacher_score,
                    q_img_path=q_img,
                    blind_checklist=cached_blind_checklist,
                    extraction_backend=extraction_backend,
                    ocr_json_path=str(ocr_json_path(raw_ocr_dir, img_path)),
                    extraction_cache_path=os.path.join(
                        facts_cache_dir, f"{file_base_name}.json"
                    ),
                    force_extraction=force_rerun,
                    grade_only=grade_only,
                    student_transcription=metadata.get("raw_text"),
                    answer_metadata=metadata,
                )

                if res and res.get("_pipeline_failed"):
                    failure = make_failed_record(
                        q_id,
                        img_file,
                        res.get("error_type", "empty_result"),
                        res.get("reason", "pipeline returned a structured failure"),
                        attempts=attempt + 1,
                    )
                    if progress_tracker:
                        progress_tracker.record_error(q_id, student_id, failure["reason"])
                    return failure

                if res:
                    eq = res.get('extraction_quality', 'unknown')
                    eq_icon = "🟢" if eq == "high" else ("🟡" if eq == "low" else "🔴")
                    risk_brief = (
                        f"P{res.get('perception_risk', 0):.2f}/"
                        f"U{res.get('uncertainty_index', 0):.2f}/"
                        f"F{res.get('fatal_points_ratio', 0):.2f}/"
                        f"H{int(bool(res.get('high_blank_high_score', False)))}/"
                        f"L{int(bool(res.get('lenient_review_signal', False)))}"
                    )
                    logging.info(f"✅ [批改完成] {file_base_name} | 路由: {res['3wd_route']} | 最终分: {res['final_calibrated_score']} | 风险: {risk_brief} | 提取质量: {eq_icon}{eq} | 留白率: {res['blank_rate']:.0%} | 低质量率: {res.get('low_quality_extraction_rate', 0):.0%}")
                    if progress_tracker:
                        progress_tracker.record_completion(q_id, student_id, res, time.time() - _start_time)
                    return res
                elif attempt == 0:
                    logging.warning(f"⚠️ [流水线返回空] {file_base_name} | 等待 5 秒后重试...")
                    time.sleep(5)
            except Exception as e:
                logging.error(f"❌ [进程报错] {file_base_name} | 错误原因: {e}")
                if attempt == 0:
                    time.sleep(5)

        logging.error(f"❌ [最终失败] {file_base_name} | 两次尝试均失败，跳过。")
        failure = make_failed_record(
            q_id,
            img_file,
            "empty_result",
            "pipeline returned no result after retry",
            attempts=2,
        )
        if progress_tracker:
            progress_tracker.record_error(q_id, student_id, failure["reason"])
        return failure

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_OUTER) as executor:
        futures = {executor.submit(process_one_student, f): f for f in image_files}
        logging.info(f"⏳ [主线程] 已提交 {len(futures)} 个任务，等待结果回收...")

        for future in concurrent.futures.as_completed(futures):
            # 优雅关闭检查
            if _shutdown_requested:
                logging.warning("🛑 收到终止信号，取消剩余任务...")
                for f in futures:
                    f.cancel()
                break

            try:
                result = future.result()
            except Exception as e:
                import traceback
                logging.error(f"❌ [主线程] future.result() 异常: {e}")
                traceback.print_exc()
                failed_record = make_failed_record(
                    q_id,
                    futures.get(future, "unknown"),
                    "pipeline_exception",
                    e,
                    attempts=2,
                )
                failed_results = upsert_failed_record(
                    failed_results, failed_record
                )
                save_failed_records(failed_path, failed_results)
                continue

            if result and result.get("error_type") and not result.get("3wd_route"):
                failed_results = upsert_failed_record(
                    failed_results, result
                )
                save_failed_records(failed_path, failed_results)
                logging.error(
                    f"[failed sample recorded] {result.get('student_id')} | "
                    f"{result.get('error_type')}: {result.get('reason')}"
                )
                continue

            if result:
                failed_results = remove_failed_record(
                    failed_results, result.get("student_id", "")
                )
                save_failed_records(failed_path, failed_results)
                results_list.append(result)
                total_count += 1
                route = result.get('3wd_route', '')
                tag = " [NEG-拒绝]" if route == "NEG" else ""
                logging.info(f"📢 [总进度] {total_count}/{len(image_files)} 份试卷已归档{tag}。")
                # 断点续传：增量保存 checkpoint
                all_so_far = existing + results_list
                save_json_list(checkpoint_path, all_so_far)

    # ========================================================
    # 合并 + 排序：断点续传结果 + 本次新结果
    # ========================================================
    logging.info(f"🗂️ 所有线程交卷完毕，正在整理数据... (共收集 {len(results_list)}/{len(image_files)} 份结果)")
    all_results = existing + results_list
    order_map = {os.path.splitext(f)[0]: index for index, f in enumerate(all_target_files)}
    all_results.sort(key=lambda x: order_map.get(x.get('student_id', ''), float('inf')))

    # ========================================================
    # 分流：正常结果 vs NEG 拒绝结果
    # ========================================================
    normal_results = [r for r in all_results if r.get('3wd_route') != 'NEG']
    rejected_results = [r for r in all_results if r.get('3wd_route') == 'NEG']

    save_path = paths["graded"]
    save_json_list(save_path, normal_results)
    logging.info(f"💾 正常批改结果已保存: {save_path} ({len(normal_results)} 人)")

    if rejected_results:
        rejected_path = paths["rejected"]
        save_json_list(rejected_path, rejected_results)
        logging.info(f"🛑 拒绝域结果已单独保存: {rejected_path} ({len(rejected_results)} 人)")
    else:
        rejected_path = paths["rejected"]
        if os.path.exists(rejected_path):
            os.remove(rejected_path)
        logging.info("✅ 无拒绝域案例。")

    validate_question_outputs(q_id, expected_count=len(all_target_files))

    if progress_tracker:
        progress_tracker.mark_question_done(q_id)


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # ==========================================
    # ⚙️ 快捷配置区
    # ==========================================
    # 直接 python main_pipeline.py 就会用这里的配置。
    # 终端传参会覆盖这里的设置。
    #
    # GRADING_CONFIG: 每道题可以独立设置不同的批改数量
    #   数字  -> 批改该题的前 N 张试卷
    #   列表  -> 只批改指定学号（如 ["E12314093", "E12214171"]）
    #   None  -> 批改该题的全部试卷
    #
    # 不在字典里的题会被跳过。
    # 支持断点续传：中断后重新运行，已完成的学生会自动跳过。

    RUN_MODE = "FULL"              # "FULL" = 正式批改, "VARIANCE_OPT" = 方差优化
    FORCE_RERUN = False            # True = 忽略检查点从头重跑, False = 断点续传

    GRADING_CONFIG = {
        # "Q1": None,                          # 全量
        "Q2": None,
        "Q3": None,
        "Q4": None,
        #"Q5": None,                             # 全量批改
        #"Q6": None,                             # 全量批改
        #"Q7": None,                             # 全量批改
    }

    VARIANCE_CONFIG = {
        # "Q1": 5,
        # "Q2": 5,
        "Q5": 5,
        "Q6": 5,
        "Q7": 5,
    }

    # ==========================================
    # 解析终端参数（终端参数优先于上面配置区）
    # ==========================================
    args = parse_args()
    DATABASE_PATH = args.database_path
    TEACHER_DB_PATH = args.teacher_db
    ANSWER_METADATA_PATH = args.answer_metadata
    INITIAL_RUBRIC_DIR = args.initial_rubric_dir
    ALLOW_INITIAL_RUBRIC = args.allow_initial_rubric
    if args.results_dir:
        OUTPUT_DIR = args.results_dir
    RUBRIC_DIR = args.rubric_dir or OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 初始化日志系统
    log_file, run_id = setup_logging(log_dir=args.log_dir, run_id=args.run_id)

    # 2. 注册信号处理器
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # 3. 加载题库
    with open(DATABASE_PATH, 'r', encoding='utf-8') as f:
        exam_data = json.load(f)

    # 终端 --questions 会覆盖配置区
    if args.questions:
        question_ids = set(args.questions)
        exam_data = [q for q in exam_data if q["question_id"] in question_ids]
    else:
        configured_questions = VARIANCE_CONFIG if args.mode == "VARIANCE_OPT" else GRADING_CONFIG
        question_ids = set(configured_questions.keys())
        exam_data = [q for q in exam_data if q["question_id"] in question_ids]

    # 4. 解析 img_limit（终端参数优先）
    if args.student_ids:
        cli_img_limit = args.student_ids
    elif args.img_limit:
        cli_img_limit = args.img_limit
    else:
        cli_img_limit = None

    # 5. 构建模型信息
    provider_map = {"glm": GLM_MODEL_NAME, "glm5": GLM5_MODEL_NAME, "deepseek": DEEPSEEK_MODEL_NAME}
    model_info = {
        "provider": TEXT_MODEL_PROVIDER,
        "vlm": VLM_MODEL_NAME,
        "text": provider_map.get(TEXT_MODEL_PROVIDER, "unknown"),
        "max_workers_outer": MAX_WORKERS_OUTER,
        "extraction_backend": args.extraction_backend,
    }

    # 6. 确定实际运行模式
    effective_mode = args.mode if args.mode != "FULL" or not any(
        k not in (args.questions or []) for k in VARIANCE_CONFIG
    ) else args.mode

    # 7. 初始化进度追踪器
    progress_path = args.progress_file or os.path.join(OUTPUT_DIR, "progress.json")

    # 8. 启动
    logging.info("=" * 50)
    logging.info("📂 正在加载试卷数据库...")
    logging.info(f"   运行模式: {args.mode}")
    logging.info(f"   题目范围: {[q['question_id'] for q in exam_data]}")
    logging.info(f"   文本模型: {get_text_model_display()}")
    logging.info(f"   提取后端: {args.extraction_backend}")
    logging.info(f"   题库文件: {DATABASE_PATH}")
    logging.info(f"   教师分文件: {TEACHER_DB_PATH}")
    if ANSWER_METADATA_PATH:
        logging.info(f"   答案元数据: {ANSWER_METADATA_PATH}")
    logging.info(f"   并发数:   {MAX_WORKERS_OUTER}")
    logging.info(f"   结果目录: {OUTPUT_DIR}")
    logging.info(f"   Rubric目录: {RUBRIC_DIR}")
    if INITIAL_RUBRIC_DIR:
        logging.info(f"   初始Rubric目录: {INITIAL_RUBRIC_DIR}")
    logging.info(f"   允许回退初始Rubric: {ALLOW_INITIAL_RUBRIC}")
    logging.info(f"   进度文件: {progress_path}")
    logging.info(f"   日志文件: {log_file}")

    # 确定实际要处理的题目列表（供追踪器显示）
    if args.mode == "VARIANCE_OPT":
        actual_questions = list(VARIANCE_CONFIG.keys()) if not args.questions else [q["question_id"] for q in exam_data]
    else:
        actual_questions = list(GRADING_CONFIG.keys()) if not args.questions else [q["question_id"] for q in exam_data]

    # 初始化追踪器
    cli_args_dict = {
        "mode": args.mode,
        "questions": actual_questions,
        "force_rerun": args.force_rerun if args.force_rerun else FORCE_RERUN,
        "img_limit": cli_img_limit,
        "sample_size": args.sample_size,
        "results_dir": OUTPUT_DIR,
        "rubric_dir": RUBRIC_DIR,
        "initial_rubric_dir": INITIAL_RUBRIC_DIR,
        "allow_initial_rubric": ALLOW_INITIAL_RUBRIC,
        "extraction_backend": args.extraction_backend,
        "ocr_cache_dir": args.ocr_cache_dir,
        "database_path": DATABASE_PATH,
        "teacher_db": TEACHER_DB_PATH,
        "answer_metadata": ANSWER_METADATA_PATH,
        "answer_split": args.answer_split,
    }
    tracker = ProgressTracker(
        progress_path=progress_path,
        run_id=run_id,
        mode=args.mode,
        model_info=model_info,
        cli_args=cli_args_dict,
    )

    try:
        if args.mode == "VARIANCE_OPT":
            logging.info("🔬 当前处于【方差优化模式】(VARIANCE OPT MODE)")
            logging.info("   系统将生成标准，进行小样本测试，修正并保存最终标准。")

            # 确定要处理的题目：终端 --questions 优先，否则用 VARIANCE_CONFIG
            if args.questions:
                variance_questions = {q["question_id"]: args.sample_size for q in exam_data}
            else:
                variance_questions = VARIANCE_CONFIG

            for q_data in exam_data:
                if _shutdown_requested:
                    break
                q_id = q_data["question_id"]
                if q_id in variance_questions:
                    run_variance_optimization_process(
                        q_data,
                        sample_size=variance_questions[q_id],
                        progress_tracker=tracker,
                        force_rerun=args.force_rerun if args.force_rerun else FORCE_RERUN,
                        extraction_backend=args.extraction_backend,
                        ocr_cache_dir=args.ocr_cache_dir,
                    )

        elif args.mode == "OCR_ONLY":
            logging.info("🔎 当前处于【仅提取模式】(OCR_ONLY)")
            if args.extraction_backend not in ("paddle_glm5", "csbench_hybrid"):
                logging.warning(
                    "OCR_ONLY is most useful with --extraction-backend paddle_glm5."
                )
            for q_data in exam_data:
                if _shutdown_requested:
                    break
                process_ocr_only_question(
                    q_data,
                    img_limit=cli_img_limit,
                    answer_split=args.answer_split,
                    force_rerun=args.force_rerun,
                    progress_tracker=tracker,
                    extraction_backend=args.extraction_backend,
                    ocr_cache_dir=args.ocr_cache_dir,
                )

        elif args.mode in ("FULL", "GRADE_ONLY"):
            logging.info(
                "🚀 当前处于【精准批改模式】"
                if args.mode == "FULL"
                else "🧮 当前处于【仅评分模式】(GRADE_ONLY)"
            )
            if args.mode == "GRADE_ONLY" and args.extraction_backend not in (
                "paddle_glm5",
                "csbench_hybrid",
            ):
                raise ValueError(
                    "GRADE_ONLY currently requires --extraction-backend paddle_glm5 "
                    "and an existing mapped fact cache."
                )

            # 确定要处理的题目：终端 --questions 优先，否则用 GRADING_CONFIG
            if args.questions:
                # 终端指定了题目，用终端的 img_limit（统一限制）
                for q_data in exam_data:
                    if _shutdown_requested:
                        break
                    process_single_question(
                        q_data,
                        img_limit=cli_img_limit,
                        answer_split=args.answer_split,
                        force_rerun=args.force_rerun if args.force_rerun else FORCE_RERUN,
                        progress_tracker=tracker,
                        extraction_backend=args.extraction_backend,
                        ocr_cache_dir=args.ocr_cache_dir,
                        grade_only=args.mode == "GRADE_ONLY",
                    )
            else:
                # 用配置区的 GRADING_CONFIG（每题可以不同限制）
                for q_data in exam_data:
                    if _shutdown_requested:
                        break
                    q_id = q_data["question_id"]
                    if q_id in GRADING_CONFIG:
                        limit = GRADING_CONFIG[q_id]
                        process_single_question(
                            q_data,
                            img_limit=limit,
                            answer_split=args.answer_split,
                            force_rerun=FORCE_RERUN,
                            progress_tracker=tracker,
                            extraction_backend=args.extraction_backend,
                            ocr_cache_dir=args.ocr_cache_dir,
                            grade_only=args.mode == "GRADE_ONLY",
                        )

        tracker.mark_finished(status="interrupted" if _shutdown_requested else "completed")
        if _shutdown_requested:
            logging.info("\n🛑 任务已优雅终止。")
        else:
            logging.info("\n🏆 任务完成！")

    except Exception as e:
        tracker.mark_finished(status="error")
        logging.error(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


RUN_COMMANDS = r"""
RefGrader 常用命令。下面每条命令都可以整行直接复制到 PowerShell。

【首次安装 PaddleOCR，仅执行一次】
.\scripts\setup_paddle_ocr.ps1
作用：创建 .venv-ocr 并安装 PaddlePaddle、PaddleOCR。

【原 Q2：只提取，不评分】
.\venv\Scripts\python.exe main_pipeline.py --mode OCR_ONLY --questions Q2 --extraction-backend paddle_glm5
作用：生成 PaddleOCR 原始结果和 rubric 事实缓存，不进入 Stage2 和 3WD。

【原 Q2：读取已有事实缓存评分】
.\venv\Scripts\python.exe main_pipeline.py --mode GRADE_ONLY --questions Q2 --extraction-backend paddle_glm5 --force-rerun
作用：不重新执行 OCR 和图形提取，直接执行 Stage2 与 3WD。

【原 Q2：PaddleOCR 后端完整运行】
.\venv\Scripts\python.exe main_pipeline.py --mode FULL --questions Q2 --extraction-backend paddle_glm5 --force-rerun
作用：一次完成 PaddleOCR、事实映射、图形解析、Stage2 和 3WD。

【原 Q2：旧 glm_vlm 后端对比】
.\venv\Scripts\python.exe main_pipeline.py --mode FULL --questions Q2 --extraction-backend glm_vlm --force-rerun
作用：使用旧视觉提取器运行消融基线。

【CSBench：首次生成兼容视图】
.\venv\Scripts\python.exe scripts\prepare_csbench.py --dataset-root C:\Users\wx\Desktop\CSBench_new --output-dir data\csbench --link-mode copy --exclude-questions OS_1 OS_2 --force
作用：只读外部 CSBench，在当前项目生成 source/initial/optimized 三层准则、校准/验证/测试划分和独立图片副本。

【CSBench：离线检查兼容数据与路由逻辑】
.\venv\Scripts\python.exe scripts\check_csbench_integration.py --prepared-dir data\csbench
作用：检查三层准则、总分一致性、数据划分隔离、文字转录路由和图形事实融合。

【CSBench统一入口：优化某道题准则】
python scripts/run_csbench.py optimize CO_3
作用：只需修改题号，其他数据库、准则、缓存、结果和进度路径自动生成。

【CSBench统一入口：后台正式批改】
python scripts/run_csbench.py grade CO_3 --background --force
作用：后台批改该题全部test答案；普通答案使用raw_text，视觉答案条件式使用PaddleOCR与GLM-4.6V。

【CSBench统一入口：查看状态与日志】
python scripts/run_csbench.py status
python scripts/run_csbench.py tail

【CSBench统一入口：断点续跑】
python scripts/run_csbench.py grade CO_3 --background

【CSBench统一入口：评估并导出CSV】
python scripts/run_csbench.py evaluate CO_3 --export
"""
