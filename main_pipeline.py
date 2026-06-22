import os
import sys
import json
import logging
import argparse
import signal
import tempfile
import concurrent.futures
import time
import numpy as np
from datetime import datetime
from step3_rrd_generator import generate_rrd_rubrics, refine_rubric_based_on_variance
from step4_vlm_grader import (
    grade_student_3wd_pipeline,
    generate_blind_checklist,
    stage1_blind_extraction,
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
DATABASE_PATH = "./database/exam_database.json"

# 全局变量，用于缓存加载的成绩单，避免每次都读文件
_GLOBAL_SCORES_DB = None

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
        help="Directory used to read Qx_rubric_standard.json files. Default: results-dir",
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
        choices=["glm_vlm", "paddle_glm5"],
        default="glm_vlm",
        help="Stage-1 extraction backend. Default: glm_vlm.",
    )
    parser.add_argument(
        "--ocr-cache-dir",
        default="ocr_cache",
        help="Root directory for raw OCR and mapped fact caches.",
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


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


def rubric_path_for(q_id):
    rubric_name = f"{q_id}_rubric_standard.json"
    primary = os.path.join(RUBRIC_DIR, rubric_name)
    if os.path.exists(primary):
        return primary
    fallback = os.path.join(DEFAULT_OUTPUT_DIR, rubric_name)
    if os.path.abspath(primary) != os.path.abspath(fallback) and os.path.exists(fallback):
        logging.warning(
            f"[rubric fallback] {primary} not found; using existing rubric from {fallback}"
        )
        return fallback
    return primary


def get_teacher_score_from_your_database(student_id, q_id):
    """数据接口：从 step0 生成的 JSON 数据库中获取教师评分"""
    global _GLOBAL_SCORES_DB

    if _GLOBAL_SCORES_DB is None:
        db_path = "./database/teacher_scores.json"
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


os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 核心功能 A：基于方差的自动优化流程 (用于测试/打磨标准)
# ============================================================
def run_variance_optimization_process(q_data, sample_size=3, progress_tracker=None, force_rerun=False):
    q_id = q_data["question_id"]
    q_score = q_data["total_score"]
    q_text = q_data["question_text"]
    q_img = q_data.get("question_image")
    ref_text = q_data["ref_text"]
    ref_img = q_data.get("ref_image")
    images_folder = q_data["student_images_dir"]
    official_rubric = q_data.get("official_rubric", "")

    rubric_save_path = os.path.join(OUTPUT_DIR, f"{q_id}_rubric_standard.json")
    checkpoint_path = os.path.join(OUTPUT_DIR, f"{q_id}_variance_checkpoint.json")

    logging.info(f"\n{'='*60}\n🔬 [断点续传模式] 处理题目: {q_id}\n{'='*60}")

    # --- Step 1: 加载或生成初始标准 ---
    draft_rubric = None
    rubric_regenerated = False
    if force_rerun and os.path.exists(rubric_save_path):
        logging.info("force-rerun enabled; regenerate rubric and ignore existing rubric file.")
    elif os.path.exists(rubric_save_path):
        with open(rubric_save_path, "r", encoding="utf-8") as f:
            draft_rubric = json.load(f)
        logging.info("Existing rubric found; skip initial generation.")
        if not isinstance(draft_rubric, list) or not draft_rubric:
            logging.warning("Existing rubric file is empty or invalid; regenerating it.")
            draft_rubric = None

    if draft_rubric is None:
        logging.info("Generating initial rubric draft...")
        draft_rubric = generate_rrd_rubrics(q_text, ref_text, official_rubric, q_score, q_img, ref_img, None)
        if not isinstance(draft_rubric, list) or not draft_rubric:
            logging.error(f"Failed to generate a valid rubric for {q_id}; stop VARIANCE_OPT for this question.")
            if progress_tracker:
                progress_tracker.record_error(q_id, "__rubric__", "rubric_generation_failed")
            return
        with open(rubric_save_path, "w", encoding="utf-8") as f:
            json.dump(draft_rubric, f, indent=4, ensure_ascii=False)
        rubric_regenerated = True
        logging.info("Initial rubric generated; sleep 2 seconds before sampling.")
        time.sleep(2)

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

    if progress_tracker:
        progress_tracker.register_question(q_id, sample_size)

    remaining_needed = sample_size - len(processed_files)
    if remaining_needed <= 0:
        logging.info("✅ 方差采样已全部完成，直接进入修正环节。")
    else:
        targets = [f for f in image_files if f not in processed_files][:remaining_needed]

        for img_file in targets:
            if _shutdown_requested:
                break

            _sample_start = time.time()
            img_path = os.path.join(images_folder, img_file)
            scores = []

            logging.info(f"\n👉 正在处理新样本: {img_file}")

            blind_checklist = generate_blind_checklist(json.dumps(draft_rubric, ensure_ascii=False))
            logging.info("   ⏳ [V0 保护] 脱敏清单生成完毕，休眠 2 秒...")
            time.sleep(2)

            logging.info("   [单次视觉采样] 正在看图提取事实...")
            current_facts = stage1_blind_extraction(q_text, img_path, blind_checklist, q_img)

            if not current_facts:
                logging.warning("   ⚠️ 视觉提取失败，跳过...")
                if progress_tracker:
                    student_id = os.path.splitext(img_file)[0].split('_')[0]
                    progress_tracker.record_error(q_id, student_id, "视觉提取失败")
                continue

            logging.info("   ⏳ [V0 保护] 视觉提取完成，准备进入打分循环，休眠 2 秒...")
            time.sleep(2)

            strict_cots = []
            for i in range(3):
                logging.info(f"   [第 {i+1}/3 次判决] 呼叫逻辑裁判...")
                res_text = stage2_logic_grading(current_facts, json.dumps(draft_rubric, ensure_ascii=False))

                if res_text:
                    parsed = extract_and_parse_json(res_text)
                    if parsed and 'total_score' in parsed:
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
                    student_id = os.path.splitext(img_file)[0].split('_')[0]
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

    has_coarse_item = any(item.get('points', 0) >= 4 for item in draft_rubric)

    if avg_variance > 0.1 or avg_item_variance > 0.05 or has_coarse_item:
        if avg_variance > 0.1:
            logging.warning("⚠️ 方差超标！开始基于高方差样本修正标准...")
        elif avg_item_variance > 0.05:
            logging.warning("⚠️ Item-level 方差超标，开始定位不稳定评分项...")
        else:
            logging.warning("⚠️ 触发粗粒度警报！发现单一条款分值过高(>=4分)，强制启动向下拆解...")

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

        final_rubric = refine_rubric_based_on_variance(draft_rubric, q_text, q_score, bad_samples)

        if final_rubric:
            with open(rubric_save_path, "w", encoding="utf-8") as f:
                json.dump(final_rubric, f, indent=4, ensure_ascii=False)
            logging.info("🎉 修正后的最终标准已保存。")
        else:
            logging.error("❌ 修正请求失败或 JSON 解析错误，保留原草稿。")
    else:
        logging.info("✅ 标准足够稳定且粒度精细，无需进一步修正。")

    if progress_tracker:
        progress_tracker.mark_question_done(q_id)


# ============================================================
# 核心功能 B：标准批改流程 (保留原有的逻辑)
# ============================================================
def process_ocr_only_question(
    q_data,
    img_limit=None,
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
    with open(rubric_path_for(q_id), "r", encoding="utf-8") as handle:
        dynamic_rubrics = json.load(handle)
    rubrics_json = json.dumps(dynamic_rubrics, ensure_ascii=False)

    all_image_files = sorted(
        f for f in os.listdir(images_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    image_files = all_image_files[:]
    if isinstance(img_limit, list):
        target_ids = set(img_limit)
        image_files = [
            f for f in image_files
            if os.path.splitext(f)[0].split("_")[0] in target_ids
        ]
    elif isinstance(img_limit, int):
        image_files = image_files[:img_limit]

    if progress_tracker:
        progress_tracker.register_question(q_id, len(image_files))

    raw_dir = os.path.join(ocr_cache_dir, q_id)
    facts_dir = os.path.join(ocr_cache_dir, "facts", q_id)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(facts_dir, exist_ok=True)

    if extraction_backend == "paddle_glm5":
        if image_files == all_image_files:
            completed = ensure_paddle_ocr_cache(
                images_folder,
                raw_dir,
                force=force_rerun,
            )
            if completed.stdout.strip():
                logging.info(completed.stdout.strip())
        else:
            for img_file in image_files:
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
            )
            records.append({
                "question_id": q_id,
                "student_id": student_id,
                "extraction_backend": extraction_backend,
                "facts": extract_and_parse_json(facts) or facts,
                "extraction_cache_path": os.path.join(facts_dir, f"{student_id}.json"),
                "blank_authenticity": evidence.get("ocr_summary", {}).get("blank_authenticity"),
                "diagram_parser_used": evidence.get("diagram_parser_used", False),
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
    rubric_output_path = rubric_path_for(q_id)

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
    image_files = all_image_files[:]

    # 灵活选人：支持 int(前N张)、list(指定学号)、None(全量)
    if isinstance(img_limit, list):
        target_ids = set(img_limit)
        image_files = [f for f in image_files if os.path.splitext(f)[0].split('_')[0] in target_ids]
        logging.info(f"📷 指定学号模式: 筛选出 {len(image_files)} 张试卷")
    elif isinstance(img_limit, int):
        image_files = image_files[:img_limit]
        logging.info(f"📷 极速测试模式: 仅处理前 {img_limit} 张图片")

    raw_ocr_dir = os.path.join(ocr_cache_dir, q_id)
    facts_cache_dir = os.path.join(ocr_cache_dir, "facts", q_id)
    os.makedirs(raw_ocr_dir, exist_ok=True)
    os.makedirs(facts_cache_dir, exist_ok=True)
    if extraction_backend == "paddle_glm5" and not grade_only:
        logging.info(f"[PaddleOCR] preparing {len(image_files)} raw OCR cache entries...")
        if image_files == all_image_files:
            completed = ensure_paddle_ocr_cache(
                images_folder,
                raw_ocr_dir,
                force=force_rerun,
            )
            if completed.stdout.strip():
                logging.info(completed.stdout.strip())
        else:
            for img_file in image_files:
                completed = ensure_paddle_ocr_cache(
                    os.path.join(images_folder, img_file),
                    raw_ocr_dir,
                    force=force_rerun,
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
        validate_question_outputs(q_id, expected_count=len(all_target_files))
        if progress_tracker:
            progress_tracker.mark_question_done(q_id)
        return

    results_list = []
    failed_results = load_json_list(failed_path) if not force_rerun else []
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
        pure_student_id = file_base_name.split('_')[0]
        img_path = os.path.join(images_folder, img_file)
        real_teacher_score = get_teacher_score_from_your_database(pure_student_id, q_id)
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
                        progress_tracker.record_error(q_id, pure_student_id, failure["reason"])
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
                        progress_tracker.record_completion(q_id, pure_student_id, res, time.time() - _start_time)
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
            progress_tracker.record_error(q_id, pure_student_id, failure["reason"])
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
                failed_results.append(failed_record)
                save_json_list(failed_path, failed_results)
                continue

            if result and result.get("error_type") and not result.get("3wd_route"):
                failed_results.append(result)
                save_json_list(failed_path, failed_results)
                logging.error(
                    f"[failed sample recorded] {result.get('student_id')} | "
                    f"{result.get('error_type')}: {result.get('reason')}"
                )
                continue

            if result:
                results_list.append(result)
                total_count += 1
                route = result.get('3wd_route', '')
                tag = " [NEG-拒绝]" if route == "NEG" else ""
                logging.info(f"📢 [总进度] {total_count}/{len(image_files)} 份试卷已归档{tag}。")
                # 断点续传：增量保存 checkpoint
                all_so_far = existing + results_list
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(all_so_far, f, indent=4, ensure_ascii=False)

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
    logging.info(f"   并发数:   {MAX_WORKERS_OUTER}")
    logging.info(f"   结果目录: {OUTPUT_DIR}")
    logging.info(f"   Rubric目录: {RUBRIC_DIR}")
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
        "extraction_backend": args.extraction_backend,
        "ocr_cache_dir": args.ocr_cache_dir,
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
                    )

        elif args.mode == "OCR_ONLY":
            logging.info("🔎 当前处于【仅提取模式】(OCR_ONLY)")
            if args.extraction_backend != "paddle_glm5":
                logging.warning(
                    "OCR_ONLY is most useful with --extraction-backend paddle_glm5."
                )
            for q_data in exam_data:
                if _shutdown_requested:
                    break
                process_ocr_only_question(
                    q_data,
                    img_limit=cli_img_limit,
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
            if args.mode == "GRADE_ONLY" and args.extraction_backend != "paddle_glm5":
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


# ============================================================
# 当前流水线与常用运行命令
# ============================================================
#
# PaddleOCR 只是新增的可选提取工具。日常只需要记住下面几条命令。
# 以下命令均以“单个题目 Q2”为例；更换题目时只改 Q2。
#
# 一、首次使用 PaddleOCR 时，只安装一次
#
#    .\scripts\setup_paddle_ocr.ps1
#
# 二、只使用 PaddleOCR 提取 Q2，不评分
#
#    .\venv\Scripts\python.exe main_pipeline.py `
#      --mode OCR_ONLY --questions Q2 `
#      --extraction-backend paddle_glm5
#
# 输出：
#    ocr_cache\Q2\                  PaddleOCR 原始 JSON
#    ocr_cache\facts\Q2\            GLM-5.1 映射后的事实 JSON
#    results_rrd_vlm\Q2_ocr_only.json
#
# 三、使用已经提取好的 Q2 事实进行评分
#
#    .\venv\Scripts\python.exe main_pipeline.py `
#      --mode GRADE_ONLY --questions Q2 `
#      --extraction-backend paddle_glm5 `
#      --force-rerun
#
# GRADE_ONLY 不会重新执行 PaddleOCR、GLM-5.1 映射和图形提取。
#
# 四、一条命令完成 PaddleOCR 提取和评分
#
#    .\venv\Scripts\python.exe main_pipeline.py `
#      --mode FULL --questions Q2 `
#      --extraction-backend paddle_glm5 `
#      --force-rerun
#
# 这条命令等价于先执行 OCR_ONLY，再执行 GRADE_ONLY。
#
# 五、使用原有 glm_vlm 后端进行对比
#
#    .\venv\Scripts\python.exe main_pipeline.py `
#      --mode FULL --questions Q2 `
#      --extraction-backend glm_vlm `
#      --force-rerun
#
# 不写 --extraction-backend 时，默认也是 glm_vlm。
#
# ============================================================
# 推荐的完整命令流程（单个题目 Q2）
# ============================================================
#
# 第一次使用时安装 PaddleOCR（只执行一次，不是每次批改都执行）：
#
# Windows：
#    .\scripts\setup_paddle_ocr.ps1
#
# Linux 服务器：
#    chmod +x scripts/setup_paddle_ocr.sh
#    ./scripts/setup_paddle_ocr.sh
#
#    │          └─ 运行项目提供的 PowerShell 安装脚本
#    └─ 在当前项目目录执行脚本
#
# 作用：
# 1. 创建独立环境 .venv-ocr，避免 PaddleOCR 依赖影响主项目 venv。
# 2. 安装 PaddlePaddle、PaddleOCR 及固定版本依赖。
# 3. 输出安装版本，确认 OCR 环境可以使用。
# 4. 如果 .venv-ocr 已经安装完成，以后可以跳过这一步。
# 5. .venv-ocr 不提交到 Git；本地和服务器分别执行对应安装脚本重建。
#
# 第一步：只提取，先检查 OCR 和图形关系是否正确（可选调试步骤）：
#
#    .\venv\Scripts\python.exe main_pipeline.py `
#    │                         └─ 运行主流水线入口 main_pipeline.py
#    └─ 使用主项目 venv 中的 Python；PaddleOCR 本身仍由 .venv-ocr 执行
#
#      --mode OCR_ONLY --questions Q2 `
#      │               │
#      │               └─ 本次只处理单个题目 Q2；改成 Q3 即处理 Q3
#      └─ 只执行提取，不执行 Stage2 评分和 3WD
#
#      --extraction-backend paddle_glm5
#      └─ 使用新增后端：
#         PaddleOCR 原始识别
#         -> GLM-5.1 映射到 rubric 事实
#         -> 遇到图形条目时调用 GLM-4.6V 解析图形关系
#
# 该命令会生成：
# 1. ocr_cache\Q2\：PaddleOCR 原始文字、坐标、置信度和图片哈希。
# 2. ocr_cache\facts\Q2\：映射后的评分事实和图形关系。
# 3. results_rrd_vlm\Q2_ocr_only.json：本次提取结果汇总。
#
# OCR_ONLY 的目的只是先检查提取是否正确。例如检查 Q2 顺序图是否包含
# C->D->C。它不是正式运行必须经历的步骤。
#
# 如果已经安装 PaddleOCR，并且不需要分阶段检查，可以跳过 OCR_ONLY，
# 直接执行下面的 FULL 命令，一次完成提取和评分：
#
#    .\venv\Scripts\python.exe main_pipeline.py `
#      --mode FULL --questions Q2 `
#      --extraction-backend paddle_glm5 `
#      --force-rerun
#
# 第二步：确认事实缓存后，只评分：
#    .\venv\Scripts\python.exe main_pipeline.py `
#      --mode GRADE_ONLY --questions Q2 `
#      --extraction-backend paddle_glm5 `
#      --force-rerun
#
# 如果不需要分阶段检查，直接使用一条完整命令：
#    .\venv\Scripts\python.exe main_pipeline.py `
#      --mode FULL --questions Q2 `
#      --extraction-backend paddle_glm5 `
#      --force-rerun
#
# 注意：
# 1. --force-rerun 会覆盖该题已有的正式评分结果。
# 2. OCR_ONLY 本身不需要 --force-rerun；图片哈希未变化时会复用缓存。
# 3. 当前默认正式后端仍是 glm_vlm，paddle_glm5 用于实验和消融对比。
