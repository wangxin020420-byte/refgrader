import os
import json
import base64
import random
import re
import time
from zhipuai import ZhipuAI

# 🔴 记得替换为你的真实 Key
API_KEY = "4796e7b83db0453fbd36eee18e161630.nuJkwBYVO6FyCpQe"  
client = ZhipuAI(api_key=API_KEY)

VLM_MODEL_NAME = "glm-4.6v"
LOGIC_MODEL_NAME = "glm-4.5-air"

def encode_image_to_base64(image_path):
    if not image_path or not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_mime_type(image_path):
    if image_path.lower().endswith('.png'):
        return 'image/png'
    return 'image/jpeg'

def extract_and_parse_json(text):
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        text = text.strip().strip('`').strip()
        if text.startswith('json'):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except Exception as e:
        print(f"❌ JSON 解析失败: {e}\n模型原始输出: {text}")
        return None

def call_glm_vlm(content_list, max_retries=3):
    """封装 GLM-4.6v 调用逻辑，自带超时重试装甲和防JSON崩溃"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=VLM_MODEL_NAME,
                messages=[{"role": "user", "content": content_list}], # type: ignore
                temperature=0.1, 
                timeout=120  
            )
            result_text = response.choices[0].message.content.strip()
            parsed_json = extract_and_parse_json(result_text)
            
            # 👉 修复2：强制校验，如果是 None 则引发异常去重试
            if parsed_json is None:
                raise ValueError("JSON格式不合法，触发重试")
            return parsed_json
            
        except Exception as e:
            print(f"   ⏳ [网络/格式异常] API 调用受阻 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("      -> 系统正在休眠 3 秒后自动重试...")
                time.sleep(3)  
            else:
                print("   ❌ 达到最大重试次数，大模型 API 彻底失败。")
                return None

# ✨ 注意这里新增了 official_rubric 参数
def generate_rrd_rubrics(question_text, ref_answer, official_rubric, total_score, q_img_path=None, ref_img_path=None, student_images_dir=None):
    print("🧠 [RRD 冷启动] 正在将官方参考标准转换为 JSON 基线...")
    
    init_prompt = f"""
    你是一位严谨的计算机专业阅卷专家。请将以下【官方评分标准】严格转换为结构化的 JSON 格式。

    🚨 【通用客观题/理工科踩点给分元规则】（必须绝对遵守）：
    这是一门硬核理工科考试，阅卷必须遵循“见数给分、见词给分”的绝对客观原则！在转换标准时，必须严格遵守以下铁律：
    1. 中间结果具象化：推导过程或步骤分，必须具象化为具体的客观事实（如：求出了具体的中间数值、指出了特定的对象或编号、或列出了特定公式）。
    2. 严禁任何主观形容词：标准中【严禁】出现“理由充分”、“逻辑清晰”、“步骤完整”、“解释合理”等词汇！如果官方答案有“理由”要求，请将其直接替换为对应的“核心关键词”或“最终结论”。
    3. 规则原子性与正交性：每个细则必须是“原子的”（只考察单一客观事实）。下游模型只需机械地进行“字符串/数值匹配”即可。
    
    【题目总分】：{total_score} 分
    【题目内容】：{question_text}
    【标准答案（用于提取具体数值）】：{ref_answer}
    【官方评分标准（你的绝对准绳）】：\n{official_rubric}
    
    ⚠️ 转换原则（严禁篡改）：
    1. 【忠实于官方】：不要自己发明评分点！完全根据【官方评分标准】来划分初始的检查项。官方给某一项分配了几分，JSON中对应的项就必须是几分。
    2. 【保留粗粒度】：如果官方标准某一项比较粗略（例如“综合推导过程正确 给8分”），请直接将其转换为一条分值为8分的 JSON 记录，不要在这一步试图拆解它。
    3. 【数值锚定】：如果官方标准提到了类似“特定参数求解正确得2分”，请从【标准答案】中提取具体的数值或公式填入描述中，以防后续产生幻觉。
    4. 【分数校验】：所有生成的项，分值总和必须绝对等于 {total_score} 分。
    
    请输出严格的 JSON 格式数组：
    [
        {{"id": "唯一的ID", "item": "官方标准描述（包含必要的事实数值）", "points": 分值}}
    ]
    """
    init_content = [{"type": "text", "text": init_prompt}]
    if q_img_path and os.path.exists(q_img_path):
        init_content.append({"type": "image_url", "image_url": {"url": f"data:{get_mime_type(q_img_path)};base64,{encode_image_to_base64(q_img_path)}"}})
    if ref_img_path and os.path.exists(ref_img_path):
        init_content.append({"type": "image_url", "image_url": {"url": f"data:{get_mime_type(ref_img_path)};base64,{encode_image_to_base64(ref_img_path)}"}})
    
    current_rubric = call_glm_vlm(init_content)
    if not current_rubric:
        return None

    if not student_images_dir or not os.path.exists(student_images_dir):
        return current_rubric

    all_imgs = [f for f in os.listdir(student_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not all_imgs:
        return current_rubric

    MAX_ITER = 2          
    SAMPLE_PER_ROUND = 2  
    used_samples = set()  
    iteration = 1

    print(f"\n🌀 [RRD 引擎启动] 进入多轮深度迭代模式 (最大 {MAX_ITER} 轮)...")

    while iteration <= MAX_ITER:
        available_imgs = [img for img in all_imgs if img not in used_samples]
        if not available_imgs:
            break
            
        sample_size = min(SAMPLE_PER_ROUND, len(available_imgs))
        sampled_files = random.sample(available_imgs, sample_size)
        used_samples.update(sampled_files)
        
        print(f"\n   🔄 [第 {iteration} 轮] 抽取 {sample_size} 份全新答卷进行试评探测...")
        conflicts_found = []

        for filename in sampled_files:
            print(f"      -> 正在发送试评请求: {filename} (请耐心等待...)")
            img_path = os.path.join(student_images_dir, filename)
            trial_prompt = f"""
            这里有一份学生的真实答卷截图。当前的评分标准是：{json.dumps(current_rubric, ensure_ascii=False)}
            
            请尝试使用该标准进行批改。重点找出标准中【过于粗粒度、存在判定歧义，导致需要酌情给分，但当前标准只有单一总分】的条款。
            
            请严格输出 JSON：
            {{
                "has_conflict": true或false,
                "conflicted_item_ids": ["存在歧义的ID"], 
                "conflict_reason": "简述为什么这条粗粒度标准需要被进一步拆分为细则"
            }}
            """
            trial_content = [{"type": "text", "text": trial_prompt}]
            trial_content.append({"type": "image_url", "image_url": {"url": f"data:{get_mime_type(img_path)};base64,{encode_image_to_base64(img_path)}"}})
            
            trial_result = call_glm_vlm(trial_content)
            if trial_result and trial_result.get("has_conflict"):
                conflicts_found.append({
                    "student_file": filename,
                    "conflicted_ids": trial_result.get("conflicted_item_ids", []),
                    "reason": trial_result.get("conflict_reason", "")
                })

        if not conflicts_found:
            print(f"   ✅ [第 {iteration} 轮] 现有粒度已足够支撑判定，无需进一步拆解！🎯 提前收敛！")
            break 
            
        print(f"   ⚠️ 发现 {len(conflicts_found)} 处需要细化的粗粒度标准，正在进行局部拆解...")
        
        refine_prompt = f"""
        你是一个严谨的阅卷细则优化专家。在批改时，我们发现部分粗粒度标准需要进行向下拆解。
        
        🚨 【通用客观题/理工科踩点给分元规则】：
        1. 必须将过程具象化为特定的客观事实（如具体数值、特定对象/编号、公式或核心关键词）。
        2. 绝对禁止使用“理由充分”、“逻辑清晰”等主观形容词。
        
        【当前标准】：{json.dumps(current_rubric, ensure_ascii=False)}
        【新发现的试评冲突与反馈】：{json.dumps(conflicts_found, ensure_ascii=False)}
        
        请仅对**有冲突的粗粒度条款**进行细化分解。
        
        ⚠️ 核心细化法则（杜绝空泛的废话标准）：
        1. 【将正确答案具象化为特征】：绝不能生成泛泛而谈的废话标准。你必须把具体的中间数值、状态转移路径或逻辑先后顺序明确描述出来！
        2. 【局部总分绝对守恒】：拆解出的子条款的分数之和，必须严格等于原来那条粗粒度条款的分数。
        3. 【只拆解，不篡改】：没有冲突的标准必须原样保留！
        
        请输出优化后的最终版严格 JSON 数组格式：
        [
            {{"id": "唯一的ID", "item": "包含具体客观事实/数值的明确检查点", "points": 具体分值}}
        ]
        """

        refine_content = [{"type": "text", "text": refine_prompt}]
        refined_rubric = call_glm_vlm(refine_content)
        
        if refined_rubric:
            current_rubric = refined_rubric
            iteration += 1
        else:
            print("   ❌ 拆解生成失败，保留上一轮标准并终止迭代。")
            break

    print(f"\n🎉 [RRD 引擎停机] 细化完成！")
    # 👉 修复1：严格保留 ID 字段，防止方差修正函数因为找不到 ID 而崩溃
    final_rubric = []
    for idx, r in enumerate(current_rubric):
        final_rubric.append({
            "id": r.get("id", f"auto_item_{idx}"),
            "item": r.get("item", ""),
            "points": r.get("points", 0)
        })
    return final_rubric


#[新增] 基于高方差样本对评分规则进行全局修正。

def refine_rubric_based_on_variance(original_rubric_list, question_text, total_score, hard_samples_info):
    """
    [修正版] 基于高方差样本优化规则。
    """
    print(f"🔧 [规则修正] 正在基于 {len(hard_samples_info)} 份高方差样本优化规则...")
    
    samples_desc = ""
    for idx, sample in enumerate(hard_samples_info):
        samples_desc += f"""
        === 疑难样本 {idx+1} ===
        【学生作答提取】: {sample.get('facts', '无法提取')}
        【多次试打分结果】: {sample.get('scores', [])} (波动大意味着规则判定模糊)
        ---------------------
        """

    prompt = f"""
    你是一位资深的计算机科学(CS)教育与阅卷专家。在自动批改系统试运行中，我们发现当前的评分标准存在“颗粒度过粗”或“评分摇摆（方差大）”的问题。

    🚨 【通用客观题/理工科踩点给分元规则】（必须绝对遵守）：
    这是一门硬核理工科考试，阅卷必须遵循“见数给分、见词给分”的绝对客观原则！在拆解或合并修改标准时，必须严格遵守以下三条铁律：
    1. 中间结果具象化（Data/Result-Driven）：推导过程或步骤分，必须具象化为具体的客观事实（如：求出了具体的中间数值、指出了特定的对象或编号、或列出了特定公式/连线）。
    2. 严禁任何主观形容词（Absolute Ban on Subjectivity）：标准中【严禁】出现“理由充分”、“逻辑清晰”、“步骤完整”等词汇！请将其直接替换为对应的“核心数值/关键词”或“最终结论”。
    3. 规则原子性与正交性（Atomicity & Orthogonality）：每个拆分出的细则必须是“原子的”（只考察单一独立的客观事实），且相互之间不能包含或重叠。
    
    【题目内容】: {question_text}
    【总分】: {total_score}
    
    【当前评分标准 (JSON)】:
    {json.dumps(original_rubric_list, ensure_ascii=False, indent=2)}
    
    【遇到的问题】:
    以下样本暴露了系统缺陷：存在分值过高(>=4分)的笼统条款，或存在使用了主观形容词的条款，导致系统缺乏可执行的客观判定依据。
    {samples_desc}
    
    【你的任务】:
    请重构并输出优化后的最终版评分标准（纯 JSON 数组格式）。
    
    🚨【最高重构红线】（违反将导致系统崩溃）：
    1. 【保留细则】：对于原本颗粒度已经足够精细（分值 <= 3 分）且符合客观元规则的条款，必须原样保留，严禁篡改。
    2. 🚨【基于客观节点拆解大项】🚨：对于任何分值 >= 4 分的笼统大项，必须将其彻底删除，并严格根据【理工科踩点给分元规则】，将其物理拆解为多个只考察具体数值/特定对象/公式/事实的独立 JSON 对象。
    3. 【分数严格守恒】：拆解后新增的所有独立对象的 `points` 分数之和，必须绝对等于被删除大项的原始分值！各细则的权重应根据其技术重要性合理分配，严禁机械平分。
    4. 【绝对 JSON 安全】：严禁在 JSON 的 `item` 字符串中使用真实的换行符（回车）！每个细则的描述必须是单行且明确的陈述句。
    5. 【扁平结构】：输出必须是一个一维 JSON 数组，严禁生成任何嵌套的对象或多维数组。
    
    请直接输出优化后的 JSON 数组，不要包含任何 Markdown 标记或多余的解释说明。
    """

    # 👉 修复3：加入重试装甲和 timeout 防止反思超时崩溃
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LOGIC_MODEL_NAME, 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=240
            )
            result_text = response.choices[0].message.content.strip()
            new_rubric = extract_and_parse_json(result_text)
            
            if not new_rubric:
                raise ValueError("未返回有效JSON")
                
            # 校验：确保顶层结构数量一致，且总分不变
            current_total = sum(item.get('points', 0) for item in new_rubric)
            if abs(current_total - total_score) > 0.1:
                print(f"   ⚠️ 修正失败：总分变异 ({current_total} != {total_score})，抛出异常重试...")
                raise ValueError("总分不守恒")

            print("   ✅ 规则修正成功！")
            return new_rubric

        except Exception as e:
            print(f"   ⏳ [修正异常] (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print("   ❌ 规则修正彻底失败，沿用原规则。")
                return original_rubric_list