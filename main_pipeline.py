import os
import json
import concurrent.futures
import time
import numpy as np
from step3_rrd_generator import generate_rrd_rubrics, refine_rubric_based_on_variance
from step4_vlm_grader import (
    grade_student_3wd_pipeline,
    generate_blind_checklist,
    stage1_blind_extraction,
    stage2_logic_grading,
    extract_and_parse_json,
    VLM_MODEL_NAME,
    TEXT_MODEL_PROVIDER,
    GLM_MODEL_NAME,
    GLM5_MODEL_NAME,
    DEEPSEEK_MODEL_NAME,
    MAX_WORKERS_OUTER,
)

def get_text_model_display():
    provider_map = {"glm": GLM_MODEL_NAME, "glm5": GLM5_MODEL_NAME, "deepseek": DEEPSEEK_MODEL_NAME}
    actual = provider_map.get(TEXT_MODEL_PROVIDER, "unknown")
    return f"{TEXT_MODEL_PROVIDER} -> {actual}"

# 📂 配置输出路径
OUTPUT_DIR = "./results_rrd_vlm"
DATABASE_PATH = "./database/exam_database.json"

# 全局变量，用于缓存加载的成绩单，避免每次都读文件
_GLOBAL_SCORES_DB = None

# ==================== 获取教师评分 ====================
def get_teacher_score_from_your_database(student_id, q_id):
    """
    数据接口：从 step0 生成的 JSON 数据库中获取教师评分
    """
    global _GLOBAL_SCORES_DB
    
    # 懒加载机制：只有第一次调用时才读取文件
    if _GLOBAL_SCORES_DB is None:
        db_path = "./database/teacher_scores.json"
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                _GLOBAL_SCORES_DB = json.load(f)
            print(f"📦 成功加载教师真实成绩单数据库！共包含 {len(_GLOBAL_SCORES_DB)} 名考生的数据。")
        else:
            print(f"⚠️ 严重警告：找不到成绩单 {db_path}，请先运行 step0_extract_ground_truth.py！")
            _GLOBAL_SCORES_DB = {} # 避免后续报错

    # 从字典中安全地获取分数
    # 如果找不到这个学生的这道题，默认返回 -1 或者 0
    student_record = _GLOBAL_SCORES_DB.get(student_id, {})
    score = student_record.get(q_id, 0.0)
    
    return float(score)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 核心功能 A：基于方差的自动优化流程 (用于测试/打磨标准)
# ============================================================
def run_variance_optimization_process(q_data, sample_size=3):
    q_id = q_data["question_id"]
    q_score = q_data["total_score"]
    q_text = q_data["question_text"]
    q_img = q_data.get("question_image")
    ref_text = q_data["ref_text"]
    ref_img = q_data.get("ref_image")
    images_folder = q_data["student_images_dir"]
    official_rubric = q_data.get("official_rubric", "")

    # 📂 定义持久化文件路径
    rubric_save_path = os.path.join(OUTPUT_DIR, f"{q_id}_rubric_standard.json")
    checkpoint_path = os.path.join(OUTPUT_DIR, f"{q_id}_variance_checkpoint.json")

    print(f"\n{'='*60}\n🔬 [断点续传模式] 处理题目: {q_id}\n{'='*60}")

    # --- Step 1: 加载或生成初始标准 ---
    draft_rubric = None
    if os.path.exists(rubric_save_path):
        with open(rubric_save_path, "r", encoding="utf-8") as f:
            draft_rubric = json.load(f)
        print("⚡ 发现已有标准草稿，跳过生成步骤。")
    else:
        print("📝 正在生成初始草稿...")
        draft_rubric = generate_rrd_rubrics(q_text, ref_text, official_rubric, q_score, q_img, ref_img, None)
        with open(rubric_save_path, "w", encoding="utf-8") as f:
            json.dump(draft_rubric, f, indent=4, ensure_ascii=False)
        # 🚨 新增：给 V0 账号的 Token 池留出回血时间！
        print("⏳ 初始标准生成完毕，强制休眠 2 秒，等待接口并发额度恢复...")
        time.sleep(2)

    # --- Step 2: 加载已有的方差探测进度 ---
    hard_samples_info = []
    processed_files = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            hard_samples_info = json.load(f)
            processed_files = {s["file"] for s in hard_samples_info}
        print(f"📈 发现进度点：已完成 {len(processed_files)}/{sample_size} 个样本。")

    # --- Step 3: 探测方差 (跳过已处理的) ---
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    # 选取尚未处理的样本，补齐到 sample_size
    remaining_needed = sample_size - len(processed_files)
    if remaining_needed <= 0:
        print("✅ 方差采样已全部完成，直接进入修正环节。")
    else:
        targets = [f for f in image_files if f not in processed_files][:remaining_needed]
        
        for img_file in targets:
            img_path = os.path.join(images_folder, img_file)
            scores = []
            
            print(f"\n👉 正在处理新样本: {img_file}")
            
            # 1. 提取事实 (只看一次图，彻底隔离视觉方差)
            blind_checklist = generate_blind_checklist(json.dumps(draft_rubric, ensure_ascii=False))
            print("   ⏳ [V0 保护] 脱敏清单生成完毕，休眠 2 秒...")
            time.sleep(2)
            
            print("   [单次视觉采样] 正在看图提取事实...")
            current_facts = stage1_blind_extraction(q_text, img_path, blind_checklist, q_img)
            
            if not current_facts:
                print("   ⚠️ 视觉提取失败，跳过...")
                continue
                
            print("   ⏳ [V0 保护] 视觉提取完成，准备进入打分循环，休眠 2 秒...")
            time.sleep(2)

            # 2. 3次逻辑裁判 (根据同一份客观事实打分)
            for i in range(3):
                print(f"   [第 {i+1}/3 次判决] 呼叫逻辑裁判...")
                res_text = stage2_logic_grading(current_facts, json.dumps(draft_rubric, ensure_ascii=False))
                
                if res_text:
                    parsed = extract_and_parse_json(res_text)
                    if parsed and 'total_score' in parsed:
                        scores.append(parsed['total_score'])
                        
                        # 🚨 掀开黑盒：把大模型的判决理由直接打印在终端！
                        print(f"      ✅ [裁判亮分] 总得分: {parsed['total_score']}")
                        if 'details' in parsed:
                            for detail in parsed.get('details', []):
                                print(f"         - [条款 {detail.get('id', '?')}] 得分: {detail.get('score_given', 0)} | 理由: {detail.get('reason', '')}")
                
                # 🚨 V0 级强制冷却：提速至 25 秒
                if i < 2:
                    print("   ⏳ [V0 保护] 判决完成，休眠 2 秒...")
                    time.sleep(2)

            if len(scores) >= 2: # 至少有两次成功才算有效
                sample_data = {
                    "file": img_file,
                    "facts": current_facts, 
                    "scores": scores
                }
                hard_samples_info.append(sample_data)
                
                # 💾 每跑完一个学生，立即存一次盘（核心断点逻辑）
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(hard_samples_info, f, indent=4, ensure_ascii=False)
                print(f"💾 样本 {img_file} 进度已保存。")
                
                print("⏳ 样本间冷却 15 秒...")
                time.sleep(15)

    # --- Step 4: 智能修正 ---
    all_scores = [s["scores"] for s in hard_samples_info]
    variances = [np.var(s) for s in all_scores if len(s) > 1]
    avg_variance = np.mean(variances) if variances else 0
    print(f"\n📊 采样完成。平均方差: {avg_variance:.4f}")
    
    # 🚨 终极防漏网机制：扫描是否存在过于粗略的高分项（>=4分）
    has_coarse_item = any(item.get('points', 0) >= 4 for item in draft_rubric)
    
    # 双重触发：方差波动大 OR 存在粗粒度高分项
    if avg_variance > 0.1 or has_coarse_item: 
        if avg_variance > 0.1:
            print("⚠️ 方差超标！开始基于高方差样本修正标准...")
        else:
            print("⚠️ 触发粗粒度警报！发现单一条款分值过高(>=4分)，强制启动向下拆解...")
            
        # 🚨 新增核心逻辑：计算单体方差，排序并提取 Top-N 刺头样本
        for sample in hard_samples_info:
            scores = sample.get('scores', [])
            sample['variance'] = np.var(scores) if len(scores) > 1 else 0.0
            
        # 按方差从大到小排序
        sorted_samples = sorted(hard_samples_info, key=lambda x: x['variance'], reverse=True)
        
        # 设定专家需要的"最坏样本"数量
        TOP_N = 3
        # 严格过滤：只取前 TOP_N 个，且必须是真正引起分歧（方差>0）的样本
        bad_samples = [s for s in sorted_samples[:TOP_N] if s['variance'] > 0]
        
        # 兜底机制：如果是因为"粗粒度警报"触发（大家的方差都是0），为了给专家模型提供参考，直接取前 2 个样本
        if not bad_samples: 
            bad_samples = sorted_samples[:2]

        print(f"🔧 [规则修正] 正在基于 {len(bad_samples)} 份精选样本优化规则...")

        # 🚨 关键修改：最后一个参数把 hard_samples_info 换成了精挑细选的 bad_samples
        final_rubric = refine_rubric_based_on_variance(draft_rubric, q_text, q_score, bad_samples)
        
        if final_rubric: # 加一层安全校验，防止生成失败导致报错
            # 覆盖保存最终结果
            with open(rubric_save_path, "w", encoding="utf-8") as f:
                json.dump(final_rubric, f, indent=4, ensure_ascii=False)
            print("🎉 修正后的最终标准已保存。")
        else:
            print("❌ 修正请求失败或 JSON 解析错误，保留原草稿。")
    else:
        print("✅ 标准足够稳定且粒度精细，无需进一步修正。")

# ============================================================
# 核心功能 B：标准批改流程 (保留原有的逻辑)
# ============================================================
def process_single_question(q_data, img_limit=None, generate_only=False, force_rerun=False):
    """处理单道题目的完整流水线"""
    q_id = q_data["question_id"]
    q_score = q_data["total_score"]
    q_text = q_data["question_text"]
    q_img = q_data.get("question_image")
    ref_text = q_data["ref_text"]
    ref_img = q_data.get("ref_image")
    images_folder = q_data["student_images_dir"]
    official_rubric = q_data.get("official_rubric", "")

    print(f"\n🚀 [标准模式] 开始处理: {q_id}")

    # 1. 生成/读取标准
    rubric_output_path = os.path.join(OUTPUT_DIR, f"{q_id}_rubric_standard.json")
    
    # 默认逻辑：读取已有的标准 (配合极速测试)
    try:
        with open(rubric_output_path, "r", encoding="utf-8") as f:
            dynamic_rubrics = json.load(f)
        print(f"⚡ 已加载本地标准: {rubric_output_path}")
    except FileNotFoundError:
        print(f"❌ 找不到标准文件 {rubric_output_path}，请先运行 VARIANCE_OPT 模式生成！")
        return

    # 2. 刹车检查
    if generate_only:
        print("🛑 [仅生成标准模式] 结束。")
        return

    # 3. 批改逻辑
    if not os.path.exists(images_folder):
        print(f"⚠️ 找不到文件夹: {images_folder}")
        return

    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    # 灵活选人：支持 int(前N张)、list(指定学号)、None(全量)
    if isinstance(img_limit, list):
        # 指定学号列表，匹配文件名前缀
        target_ids = set(img_limit)
        image_files = [f for f in image_files if os.path.splitext(f)[0].split('_')[0] in target_ids]
        print(f"📷 指定学号模式: 筛选出 {len(image_files)} 张试卷")
    elif isinstance(img_limit, int):
        image_files = image_files[:img_limit]
        print(f"📷 极速测试模式: 仅处理前 {img_limit} 张图片")

    # 断点续传：加载已完成的学生
    checkpoint_path = os.path.join(OUTPUT_DIR, f"{q_id}_grading_checkpoint.json")
    completed_ids = set()
    if force_rerun and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("🔄 [强制重跑] 已删除 checkpoint，从头开始批改。")
    if not force_rerun and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            completed_ids = {r.get('student_id', '') for r in existing}
        print(f"📈 断点续传: 发现 {len(completed_ids)} 份已完成的结果，跳过。")
    else:
        existing = []

    # 保存完整目标列表（用于最终排序）
    all_target_files = image_files[:]

    # 过滤掉已完成的学生
    remaining_files = [f for f in image_files if os.path.splitext(f)[0] not in completed_ids]
    if len(remaining_files) < len(image_files):
        print(f"⏭️ 跳过 {len(image_files) - len(remaining_files)} 份已完成试卷，剩余 {len(remaining_files)} 份待处理。")
    image_files = remaining_files

    if not image_files:
        print("✅ 所有试卷已处理完毕，无需重新批改。")
        # 合并已有结果并保存最终文件
        results_list = existing
        normal_results = [r for r in results_list if r.get('3wd_route') != 'NEG']
        rejected_results = [r for r in results_list if r.get('3wd_route') == 'NEG']
        save_path = os.path.join(OUTPUT_DIR, f"{q_id}_graded_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(normal_results, f, indent=4, ensure_ascii=False)
        if rejected_results:
            rejected_path = os.path.join(OUTPUT_DIR, f"{q_id}_rejected.json")
            with open(rejected_path, "w", encoding="utf-8") as f:
                json.dump(rejected_results, f, indent=4, ensure_ascii=False)
            print(f"💾 最终结果已保存: {save_path} ({len(normal_results)} 人) + {rejected_path} ({len(rejected_results)} 人)")
        else:
            print(f"💾 最终结果已保存: {save_path} ({len(normal_results)} 人)")
        return

    results_list = []
    total_count = 0

    # 预生成脱敏清单（同题共用，只调一次 API）
    cached_blind_checklist = generate_blind_checklist(json.dumps(dynamic_rubrics, ensure_ascii=False))
    print(f"⚡ 脱敏清单已预生成，{len(image_files)} 名学生共享复用。")

    print(f"🏃 开始批改 {len(image_files)} 张试卷 (⚡ 开启多线程模式)...", flush=True)

    def process_one_student(img_file):
        file_base_name = os.path.splitext(img_file)[0]
        pure_student_id = file_base_name.split('_')[0]
        img_path = os.path.join(images_folder, img_file)
        real_teacher_score = get_teacher_score_from_your_database(pure_student_id, q_id)

        for attempt in range(2):
            print(f"\n🔍 [正在处理] {file_base_name} | 启动 3WD 流水线..." + ("(重试)" if attempt > 0 else ""), flush=True)

            try:
                res = grade_student_3wd_pipeline(
                    student_img_path=img_path,
                    question_text=q_text,
                    rubrics_json=json.dumps(dynamic_rubrics, ensure_ascii=False),
                    teacher_score=real_teacher_score,
                    q_img_path=q_img,
                    blind_checklist=cached_blind_checklist
                )

                if res:
                    eq = res.get('extraction_quality', 'unknown')
                    eq_icon = "🟢" if eq == "high" else ("🟡" if eq == "low" else "🔴")
                    print(f"✅ [批改完成] {file_base_name} | 路由: {res['3wd_route']} | 最终分: {res['final_calibrated_score']} | 提取质量: {eq_icon}{eq} | 留白率: {res['blank_rate']:.0%} | 低质量率: {res.get('low_quality_extraction_rate', 0):.0%}", flush=True)
                    return res
                elif attempt == 0:
                    print(f"⚠️ [流水线返回空] {file_base_name} | 等待 5 秒后重试...", flush=True)
                    time.sleep(5)
            except Exception as e:
                print(f"❌ [进程报错] {file_base_name} | 错误原因: {e}", flush=True)
                if attempt == 0:
                    time.sleep(5)

        print(f"❌ [最终失败] {file_base_name} | 两次尝试均失败，跳过。", flush=True)
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_OUTER) as executor:
        futures = {executor.submit(process_one_student, f): f for f in image_files}
        print(f"⏳ [主线程] 已提交 {len(futures)} 个任务，等待结果回收...", flush=True)

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                import traceback
                print(f"❌ [主线程] future.result() 异常: {e}", flush=True)
                traceback.print_exc()
                continue

            if result:
                results_list.append(result)
                total_count += 1
                route = result.get('3wd_route', '')
                tag = " [NEG-拒绝]" if route == "NEG" else ""
                print(f"📢 [总进度] {total_count}/{len(image_files)} 份试卷已归档{tag}。", flush=True)
                # 断点续传：增量保存 checkpoint
                all_so_far = existing + results_list
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(all_so_far, f, indent=4, ensure_ascii=False)

    # ========================================================
    # 合并 + 排序：断点续传结果 + 本次新结果
    # ========================================================
    print(f"🗂️ 所有线程交卷完毕，正在整理数据... (共收集 {len(results_list)}/{len(image_files)} 份结果)", flush=True)
    all_results = existing + results_list
    order_map = {os.path.splitext(f)[0]: index for index, f in enumerate(all_target_files)}
    all_results.sort(key=lambda x: order_map.get(x.get('student_id', ''), float('inf')))

    # ========================================================
    # 分流：正常结果 vs NEG 拒绝结果
    # ========================================================
    normal_results = [r for r in all_results if r.get('3wd_route') != 'NEG']
    rejected_results = [r for r in all_results if r.get('3wd_route') == 'NEG']

    save_path = os.path.join(OUTPUT_DIR, f"{q_id}_graded_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(normal_results, f, indent=4, ensure_ascii=False)
    print(f"💾 正常批改结果已保存: {save_path} ({len(normal_results)} 人)")

    if rejected_results:
        rejected_path = os.path.join(OUTPUT_DIR, f"{q_id}_rejected.json")
        with open(rejected_path, "w", encoding="utf-8") as f:
            json.dump(rejected_results, f, indent=4, ensure_ascii=False)
        print(f"🛑 拒绝域结果已单独保存: {rejected_path} ({len(rejected_results)} 人)")
    else:
        print(f"✅ 无拒绝域案例。")


if __name__ == "__main__":
    # ==========================================
    # ⚙️ 运行模式配置
    # ==========================================
    # 1. VARIANCE_OPT: 测试标准质量。生成标准 -> 抽样测试方差 -> 自动修正 -> 保存。
    # 2. FULL: 正式批改模式 (需先有标准文件)。可精准配置批改哪道题、批改多少张。
    
    RUN_MODE = "FULL"  
    
    # ------------------------------------------
    # 🎛️ 配置区 A：方差优化模式 (VARIANCE_OPT)
    # ------------------------------------------
    # 格式 -> "题号": 寻找刺头样本的采样数量
    VARIANCE_CONFIG = {
        #"Q1": 5, 
        #"Q2": 5,  
        #"Q3": 5,
        #"Q4": 5,
        "Q5": 5,
        "Q6": 5,
        "Q7": 5,
    }
    
    # ------------------------------------------
    # 🎛️ 配置区 B：正式批改模式 (FULL)
    # ------------------------------------------
    FORCE_RERUN = False # 全量重跑Q7（标准已重写，必须重跑）
    # 🚨 精准控制批改范围，支持三种格式：
    #   数字  -> 批改该题的前 N 张试卷
    #   列表  -> 只批改指定学号的试卷（如 ["E12314093", "E12214171"]）
    #   None  -> 批改该题的全部试卷
    # 支持断点续传：中断后重新运行，已完成的学生会自动跳过。
    GRADING_CONFIG = {
        #"Q2": 10,                          # 前 10 张
        # "Q1": None,                       # 全量
        # "Q3": ["E12314093", "E12214171"],  # 指定学号
        #"Q4": None,                        # 
        #"Q5": None,                        # 
        #"Q6": None,                        #
        "Q7": None                          # 
    }
    print("=" * 50)

    print("📂 正在加载试卷数据库...")
    try:
        with open(DATABASE_PATH, 'r', encoding='utf-8') as f:
            exam_data = json.load(f)
        
        if RUN_MODE == "VARIANCE_OPT":
            print("🔬 当前处于【方差优化模式】(VARIANCE OPT MODE)")
            print("-> 系统将生成标准，进行小样本测试，修正并保存最终标准。")
            for q_data in exam_data:
                q_id = q_data["question_id"]
                if q_id in VARIANCE_CONFIG:
                    run_variance_optimization_process(q_data, sample_size=VARIANCE_CONFIG[q_id])

        elif RUN_MODE == "FULL":
             print("🚀 当前处于【精准批改模式】(FULL MODE)")
             for q_data in exam_data:
                 q_id = q_data["question_id"]
                 # 只有在 GRADING_CONFIG 里取消注释的题，才会被批改
                 if q_id in GRADING_CONFIG:
                     # 读取你配置的限制数量 (如果是 None，则全量批改)
                     limit = GRADING_CONFIG[q_id] 
                     process_single_question(q_data, img_limit=limit, force_rerun=FORCE_RERUN)

        print("\n🏆 任务完成！")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()