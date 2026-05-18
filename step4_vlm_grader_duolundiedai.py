import os
import json
from json_repair import repair_json
import base64
import re
import time
from zhipuai import ZhipuAI

# 🔴 你的真实 Key
API_KEY = "132a47a6484e4a9dbfaa51fea40bbae0.LqWjKhw6WcH2sdFs"  
client = ZhipuAI(api_key=API_KEY)

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
        # 先尝试标准的 json 解析
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # 如果标准解析失败，调用 json_repair 自动修复内部的未转义双引号、缺漏的括号等
            repaired_text = repair_json(text)
            return json.loads(repaired_text)
        except Exception as e:
            print(f"❌ 判卷 JSON 彻底解析失败 (修复也无效): {e}\n模型原始输出: {text}")
            return None

# ==================== 以下为解耦架构重构部分 ====================

def generate_blind_checklist(rubrics_json_str):
    """
    [脱敏处理] 将带有标准答案的 rubric 转化为只有提取任务的盲测清单。
    通过正则表达式，强行抹除原话中附带的标准答案（如 11111 或 A->B）。
    """
    try:
        rubrics = json.loads(rubrics_json_str)
        checklist = []
        for i, item in enumerate(rubrics):
            raw_item = item.get('item', '')
            
            # 🔪 数据清洗 1：剔除二进制串 (例如 "1 1 1 1 1", "01000")
            cleaned_item = re.sub(r'[01\s]{4,}', '', raw_item)
            
            # 🔪 数据清洗 2：剔除字母时序序列 (例如 "A→D→C→E→B" 或 "A>D>C")
            cleaned_item = re.sub(r'[A-Z](?:[→>][A-Z])+', '', cleaned_item)
            
            # 🔪 数据清洗 3：清理可能残留的“为”、“是”等连接词结尾
            cleaned_item = re.sub(r'(为|是)$', '', cleaned_item.strip())
            
            # 现在的 cleaned_item 变成了纯净的提问，比如："A的屏蔽字"
            checklist.append(f"目标 {i+1}: 提取图片中关于【{cleaned_item.strip()}】的真实客观物理痕迹。")
            
        return "\n".join(checklist)
    except Exception as e:
        print(f"❌ 解析 rubric 失败，无法生成盲测清单: {e}")
        return ""

def stage1_blind_extraction(question_text, student_img_path, blind_checklist, q_img_path=None):
    """
    [Stage 1: 盲眼感知] 只给学生图片和无答案清单，逼迫模型做纯粹的 OCR 和拓扑转录。
    """
    blind_prompt = f"""
    你是一个毫无感情的客观视觉扫描仪，负责从计算机科学试卷中提取物理痕迹。
    
    【题目背景】：{question_text}
    【提取任务清单】：
    {blind_checklist}
    
    🚨 【最高提取纪律】：
    1. 像素级忠实：无论学生的作答是标准文本、计算公式，还是包含占位符、残缺符号、不规范简写，你必须像无情的复印机一样原样转录其物理形态。
    2. 智能分离与抗噪：像人类一样区分“被划掉的废弃笔迹/扫描噪点”与“学生最终提交的有效作答”。只提取最终有效部分，但必须保留其真实的不规范特征。
    3. 零推导原则：绝对禁止根据你的专业知识去猜测学生的意图！绝对禁止纠错、补全方程、推导下一步或美化格式！看不清的地方直接标记为“[无法辨认]”。
    
    请严格输出 JSON 格式，返回一个字典，键为"目标 N"，值为你肉眼看到的真实客观物理痕迹。
    """
    
    content_list = [{"type": "text", "text": blind_prompt}]
    
    # 1. 插入题目原始附图（如果有）
    if q_img_path and os.path.exists(q_img_path):
        content_list.append({"type": "text", "text": "\n【以下是题目的原始附图，仅供了解题目背景】："})
        q_b64 = encode_image_to_base64(q_img_path)
        q_mime = get_mime_type(q_img_path)
        content_list.append({"type": "image_url", "image_url": {"url": f"data:{q_mime};base64,{q_b64}"}})

    # 2. 插入学生作答图（绝不插入参考答案图 ref_img_path）
    student_b64 = encode_image_to_base64(student_img_path)
    if not student_b64:
        print(f"⚠️ 无法读取学生答卷: {student_img_path}")
        return None
    student_mime = get_mime_type(student_img_path)
    
    content_list.append({
        "type": "text", 
        "text": "\n🎯 [SYSTEM TARGET]: 下方紧跟的这张图片才是【考生的真实答卷】！你需要从中如实提取客观事实。"
    })
    content_list.append({"type": "image_url", "image_url": {"url": f"data:{student_mime};base64,{student_b64}"}})
    
    # ✅ 以下带有重试机制和长超时的护甲代码
    max_retries = 3  # 设置最大重试次数为 3 次
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="glm-4.6v", 
                messages=[{"role": "user", "content": content_list}], # type: ignore
                temperature=0.1, 
                timeout=180  # 👈 核心修改：强制拉长单次请求的超时时间至 180 秒
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"  ⚠️ Stage 1 视觉提取第 {attempt + 1} 次请求失败: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt * 5  # 指数退避：第一次等5秒，第二次等10秒
                print(f"  ⏳ 触发网络保护机制，等待 {sleep_time} 秒后自动重试...")
                time.sleep(sleep_time)
            else:
                print(f"❌ Stage 1 彻底失败，已达到最大重试次数！跳过该试卷。")
                return None

def stage2_logic_grading(student_facts_str, rubrics_json_str):
    """
    [Stage 2: 闭卷逻辑裁判] 抛弃图片，使用纯文本进行极其严密的逻辑对账。
    """
    logic_prompt = f"""
    你是一位铁面无私的计算机专业阅卷逻辑裁判。
    
    【标准参考细则 (The Ground Truth)】：
    {rubrics_json_str}
    
    【学生答卷物理事实 (已通过无先验纯视觉提取)】：
    {student_facts_str}
    
    【裁判任务】：
    请你将“标准细则”与“学生答卷事实”进行严密的逻辑文本比对。
    
    🚨 【判罚准则】：
    1. 核心语义等价宽容：允许合理的符号替换（例如表示“推导/指向/优先”的不同箭头或数学符号）、无歧义的大小写差异（若无特殊约定）以及非关键的格式/排版空格差异。只要核心逻辑等价，即判定为匹配。
    2. 事实要素零容忍：对于具体的计算数值（如二进制位、概率结果、物理量）、明确的变量名、公式核心运算符，以及图表中的关键拓扑时序（谁先谁后、谁连接谁），哪怕有极其微小的字符差异、缺失或拓扑错位，也必须判定为不匹配 (false)！
    
    🚨 【极度重要：JSON 格式与引号规范】（如果不严格遵守，系统将直接崩溃！）：
    1. 你必须且只能输出完全合法的 JSON 格式。
    2. **严禁**在 JSON 的字符串值（Value）内部直接使用未转义的双引号（"）。
    3. 如果在 `semantic_comparison` 或 `student_raw_extraction` 等字段中需要引用学生的原始答案，**强制使用单引号（'）或中文引号（“ ”）**。
       - ❌ 错误示范："student_raw_extraction": "学生答案为"A>B""
       - ✅ 正确示范："student_raw_extraction": "学生答案为'A>B'"

    请严格输出 JSON 格式，直接返回如下结构：
    {{
        "evaluations": [
            {{
                "item": "评分细则中的原话", 
                "student_raw_extraction": "照抄【学生答卷物理事实】中的对应内容",
                "standard_requirement": "提取细则中的标准要求",
                "semantic_comparison": "详细
                "satisfied": true或false
            }}
        ],
        "total_score": 计算出的总得分 (satisfied为true的分数之和),
        "feedback": "一句话总结扣分原因"
    }}
    """
    
    # ✅ 以下带有重试机制和长超时的护甲代码
    max_retries = 3  # 设置最大重试次数
    
    for attempt in range(max_retries):
        try:
            # 同样增加 timeout 到 180 秒，防止复杂逻辑对账超时
            response = client.chat.completions.create(
                model="glm-4.6v", 
                messages=[{"role": "user", "content": logic_prompt}], 
                temperature=0.1, 
                timeout=180
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"  ⚠️ Stage 2 逻辑裁判第 {attempt + 1} 次请求失败: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt * 5  # 指数退避策略
                print(f"  ⏳ 等待 {sleep_time} 秒后尝试重新进行逻辑比对...")
                time.sleep(sleep_time)
            else:
                print(f"❌ Stage 2 彻底失败！已达到最大重试次数，该试卷逻辑比对失败。")
                return None

def grade_student_answer_vlm(student_img_path, question_text, rubrics_json, q_img_path=None, ref_img_path=None):
    """
    [解耦主管线] 取代单次大黑盒调用的总控逻辑
    """
    print(f"  ➡️ [解耦架构] 启动 Stage 1：盲眼事实提取...")
    blind_checklist = generate_blind_checklist(rubrics_json)
    
    student_facts = stage1_blind_extraction(question_text, student_img_path, blind_checklist, q_img_path)
    
    if not student_facts:
        print("  ⚠️ Stage 1 提取失败，跳过该试卷。")
        return None
        
    print(f"  ➡️ [解耦架构] 启动 Stage 2：闭卷逻辑比对...")
    final_result_text = stage2_logic_grading(student_facts, rubrics_json)
    
    if final_result_text:
        return extract_and_parse_json(final_result_text)
    return None