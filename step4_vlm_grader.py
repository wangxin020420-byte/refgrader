import os
import json
import math
from json_repair import repair_json
import base64
import re
import time
import concurrent.futures
from openai import OpenAI
from PIL import Image
import io
import numpy as np

# ==================== 配置区 ====================
# 视觉模型（固定 GLM）
VLM_MODEL_NAME = "glm-4.6v"

# 文本模型切换：修改此处即可，可选 "glm" / "glm5" / "deepseek"
TEXT_MODEL_PROVIDER = "glm5"

# Coding Plan 统一配置（OpenAI 兼容接口）
CODING_PLAN_API_KEY = "132a47a6484e4a9dbfaa51fea40bbae0.LqWjKhw6WcH2sdFs"
CODING_PLAN_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4/"

# GLM-4.5-air
GLM_API_KEY = CODING_PLAN_API_KEY
GLM_BASE_URL = CODING_PLAN_BASE_URL
GLM_MODEL_NAME = "glm-4.5-air"

# GLM-5.1
GLM5_API_KEY = CODING_PLAN_API_KEY
GLM5_BASE_URL = CODING_PLAN_BASE_URL
GLM5_MODEL_NAME = "glm-5.1"

# DeepSeek 配置
DEEPSEEK_API_KEY = "sk-6lCywlyf1xwXyV8G937sOrRF7kGThWMrwFVksuwGZaAWrAzP"
DEEPSEEK_BASE_URL = "https://gpt-agent.cc/v1"
DEEPSEEK_MODEL_NAME = "deepseek-v4-flash"

# 并发配置：{provider: (外层学生并发, 内层Stage2探测并发)}
MODEL_CONCURRENCY = {
    "glm":      (3, 3),  # GLM-4.5-air 并发能力强
    "glm5":     (2, 2),  # GLM-5.1 Coding Pro 限流较严，降低并发
    "deepseek": (2, 2),  # 第三方代理，保守一点
}
MAX_WORKERS_OUTER = MODEL_CONCURRENCY.get(TEXT_MODEL_PROVIDER, (3, 3))[0]
MAX_WORKERS_STAGE2 = MODEL_CONCURRENCY.get(TEXT_MODEL_PROVIDER, (3, 3))[1]

# 全局客户端
glm_client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# ==================== 统一文本模型调用 ====================

def call_text_model(messages, temperature=0.2, timeout=120):
    """统一文本模型调用入口，根据 TEXT_MODEL_PROVIDER 自动分发"""
    if TEXT_MODEL_PROVIDER == "glm5":
        for attempt in range(4):
            try:
                client = OpenAI(api_key=GLM5_API_KEY, base_url=GLM5_BASE_URL)
                response = client.chat.completions.create(
                    model=GLM5_MODEL_NAME,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"         ⚠️ [GLM-5 重试 {attempt+1}/4] {type(e).__name__}: {str(e)[:80]}... 等待{wait}秒")
                time.sleep(wait)
        raise Exception("GLM-5 4次重试均失败")
    elif TEXT_MODEL_PROVIDER == "deepseek":
        # DeepSeek：每请求独立客户端 + 指数退避重试，解决并发限流
        for attempt in range(4):
            try:
                client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
                response = client.chat.completions.create(
                    model=DEEPSEEK_MODEL_NAME,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"         ⚠️ [DeepSeek 重试 {attempt+1}/4] {type(e).__name__}: {str(e)[:80]}... 等待{wait}秒")
                time.sleep(wait)
        raise Exception("DeepSeek 4次重试均失败")
    else:
        response = glm_client.chat.completions.create(
            model=GLM_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            timeout=timeout
        )
        return response.choices[0].message.content.strip()

# ==================== 工具函数 (不变) ====================

def compress_image_to_bytes(image_path, max_long_edge=1920):
    try:
        if not os.path.exists(image_path): return None
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        width, height = img.size
        if max(width, height) > max_long_edge:
            scale = max_long_edge / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"   🗜️ [压缩] {width}x{height} -> {new_width}x{new_height}")
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
    except Exception as e:
        print(f"   ⚠️ 压缩出错: {e}")
        try: return open(image_path, "rb").read()
        except: return None

def encode_image_to_base64(image_path):
    if not image_path or not os.path.exists(image_path): return None
    img_bytes = compress_image_to_bytes(image_path)
    return base64.b64encode(img_bytes).decode('utf-8') if img_bytes else None

def extract_and_parse_json(text):
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match: text = match.group(1)
    else:
        text = text.strip().strip('`').strip()
        if text.startswith('json'): text = text[4:].strip()
    text = text.replace('\n', ' ').replace('\r', ' ')
    try: return json.loads(text)
    except:
        try: return json.loads(repair_json(text))
        except Exception as e: 
            print(f"❌ JSON 解析失败: {e}")
            return None

# ==================== 提取后题目参数守卫 ====================

def validate_extraction_against_question(question_text, extracted_facts_str):
    """检查提取结果是否误抄了题目原文参数，返回清洗后的 facts JSON 字符串"""
    prompt = f"""
你是提取质量审核员。请检查以下【提取结果】中的值，是否直接来自【题目原文】的参数。

【题目原文】：{question_text}
【提取结果】：{extracted_facts_str}

规则：
- 如果某个提取值在题目原文中完整出现，且该条目要求学生自己计算或写出答案而非抄录题目，则标记为 suspicious。
- 如果提取值虽然与题目参数相同，但它是学生公式或计算表达式的一部分（即学生将该参数用于自己的推导过程，而非单独抄录），标记为 ok——学生在计算中使用题目参数是合法的。
- 如果提取值是学生独有的计算结果（不在题目原文中出现），标记为 ok。
- "未书写"和"字迹模糊"始终标记为 ok。

输出纯 JSON 对象，key 为 item 编号，value 为 "ok" 或 "suspicious"：
{{"1": "ok", "2": "suspicious", ...}}
"""
    try:
        raw = call_text_model([{"role": "user", "content": prompt}], temperature=0.1, timeout=60)
        verdict = extract_and_parse_json(raw)
        if not verdict or not isinstance(verdict, dict):
            return extracted_facts_str

        facts_dict = extract_and_parse_json(extracted_facts_str)
        if not facts_dict or not isinstance(facts_dict, dict):
            return extracted_facts_str

        changed = 0
        for k, v in verdict.items():
            if v == "suspicious" and k in facts_dict:
                facts_dict[k] = "未书写"
                changed += 1

        if changed > 0:
            print(f"   🛡️ [参数守卫] 检测到 {changed} 处疑似抄录题目的提取值，已替换为'未书写'")
            return json.dumps(facts_dict, ensure_ascii=False)
        return extracted_facts_str
    except Exception as e:
        print(f"   ⚠️ [参数守卫] 验证调用异常，跳过: {e}")
        return extracted_facts_str

# ==================== 核心业务逻辑 ====================

def generate_blind_checklist(rubrics_json_str):
    prompt = f"""
你是一个视觉提取指令生成器。请根据以下【评分标准】，为每个评分项生成一条精确的提取指令。

🚨【红线】：绝对不能把正确答案的具体数值或正确状态写进指令里！

要求：
1. 每条指令必须明确要求提取"学生写的具体内容"（数值、公式、文字、判断结果等），而不是"是否存在"。
2. 只描述需要观察什么类型的作答内容，不暗示答案。
3. 输出严格的 JSON 数组格式，id 与评分标准条目一一对应。

【评分标准】：{rubrics_json_str}

输出格式示例：
[
    {{"id": "1", "instruction": "提取学生关于'物理地址位数'所写的具体数值（如：xx位）"}},
    {{"id": "2", "instruction": "提取学生关于'块内偏移位数'所写的具体数值（如：xx位）"}},
    {{"id": "10", "instruction": "提取学生对于地址2D07FFH的命中/未命中判断结果文字"}}
]
"""
    for attempt in range(3):
        try:
            raw = call_text_model([{"role": "user", "content": prompt}], temperature=0.3, timeout=240)
            parsed = extract_and_parse_json(raw)
            if parsed and isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False)
        except Exception as e:
            time.sleep(3)
    return json.dumps([{"id": str(i+1), "instruction": f"提取第{i+1}个评分项对应的学生作答具体内容"} for i in range(15)])

def stage1_blind_extraction(question_text, student_img_path, blind_checklist, q_img_path=None):
    blind_prompt = f"""
# Role: 纯视觉 OCR 提取引擎
你的唯一职责：逐字转录学生答卷图片中的具体内容。严禁逻辑推理，严禁判断对错。

【题目背景】：{question_text}
【提取清单（每条对应一个评分项）】：{blind_checklist}

🚨【区域辨别铁律——防止抄录题目】：
答题卷图片中通常同时包含"打印的题目文本"和"学生手写的作答内容"两部分。
- 你只应提取学生手写作答区域的内容。
- 如果某个评分项对应的作答区域没有学生的手写内容（空白、只有印刷文字），必须输出"未书写"，绝不能从打印的题目文本中抄录参数值。
- 🚨 特别警惕：如果提取到的数值与题目背景中给出的参数完全相同（如题目说"指令数10万条"而你提取到"10⁵"），这极有可能是你误读了题目文本，请重新确认该区域是否有学生手写作答痕迹。

🚨【输出规则——三级优先级，必须严格遵循】：
对于清单中的每一项，你必须在以下三种状态中选择一种输出：
1. 【具体内容】（最优先）：尽可能逐字转录学生写的数值、公式、文字或判断结果。保留原始表达。
2. "未书写"：仅当对应区域完全空白、没有任何笔迹时使用。
3. "字迹模糊"：有笔迹但完全无法辨认（狂草、严重涂改、污损）时使用。

🚨【值嵌入识别规则】：
学生经常不会将参数或中间结果单独写成独立的一行，而是直接将其嵌入在公式、表达式或推导链中（例如把参数值写在分式里、写在等式的某一项中）。对于清单中的每一项，你不应只在答卷中寻找"独立成行的参数声明"，而必须全局扫描学生的全部手写内容。只要学生在任何公式、表达式、计算步骤或推导链中写出了该清单项所要求的具体值，你就必须从中提取该值作为该项输出，而不是报"未书写"。

🚨【数值序列提取专项规则】：
- 二进制/十六进制数字：必须逐位提取，严禁截断、增添或遗漏任何一位数字（包括前导零）。
- 如果学生写了分组格式（如 "0010 1101"），保留原始格式输出。
- 如果学生写了进制标记（如后缀 B、H、₂ 等），一并保留。
- 数值位数不正确的提取是严重错误！提取时必须仔细数清每一位数字，确保与试卷上学生书写的位数完全一致。

🚨【绝对禁止输出以下废品值】：
- "是"、"有"、"存在"、"已书写"、"有书写"、"对"、"正确" → 这些是判断，不是内容！
- "有提取标注"、"有计算过程"、"有标注" → 同样是废话！

【正例】（合格输出）：
  {{"1": "23位", "2": "11位", "3": "4位", "6": "001011010000011111111111", "10": "不能访问到", "15": "能访问到"}}
【反例】（废品输出，绝对禁止）：
  {{"1": "是", "2": "有", "3": "已书写", "6": "存在", "10": "是", "15": "有"}}

请严格按照清单中每项的 id 作为 key，输出 JSON 对象。
"""
    content_list = [{"type": "text", "text": blind_prompt}]
    if q_img_path and os.path.exists(q_img_path):
        q_b64 = encode_image_to_base64(q_img_path)
        content_list.extend([{"type": "text", "text": "\n【附图】:"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{q_b64}"}}])
    student_b64 = encode_image_to_base64(student_img_path)
    content_list.extend([{"type": "text", "text": "\n【考卷】:"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{student_b64}"}}])
    
    for attempt in range(4):
        try:
            fresh_client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)
            res = fresh_client.chat.completions.create(model=VLM_MODEL_NAME, messages=[{"role": "user", "content": content_list}], temperature=0.1, timeout=180)
            time.sleep(2)
            raw_result = res.choices[0].message.content.strip()
            return validate_extraction_against_question(question_text, raw_result)
        except Exception as e:
            time.sleep(30)
    return None

def stage1_targeted_reextraction(question_text, student_img_path, blind_checklist, initial_facts_str, q_img_path=None):
    """
    二次精准提取：对首次被标记为"未书写"的条目进行二次检查。
    仅在 blank_rate >= 0.3 且空白条目 >= 2 时触发，避免不必要的 API 调用。
    """
    facts_dict = json.loads(initial_facts_str) if isinstance(initial_facts_str, str) else initial_facts_str
    if not isinstance(facts_dict, dict):
        return initial_facts_str

    blank_items = {k: v for k, v in facts_dict.items() if str(v).strip() == "未书写"}
    if len(blank_items) < 2:
        return initial_facts_str
    blank_rate = len(blank_items) / len(facts_dict) if len(facts_dict) > 0 else 0
    if blank_rate < 0.3:
        return initial_facts_str

    # 从 blind_checklist 中筛选空白条目的提取指令
    checklist_items = json.loads(blind_checklist) if isinstance(blind_checklist, str) else blind_checklist
    focused_instructions = []
    if isinstance(checklist_items, list):
        for item in checklist_items:
            if item.get('id') in blank_items:
                focused_instructions.append(item)
    if not focused_instructions:
        return initial_facts_str
    focused_checklist = json.dumps(focused_instructions, ensure_ascii=False)

    already_extracted = {k: v for k, v in facts_dict.items() if str(v).strip() != "未书写"}

    reextraction_prompt = f"""
# Role: 二次精准提取引擎
第一轮提取中，以下条目被标记为"未书写"。请对【考卷图片】进行极其仔细的二次检查。

**特别注意**：学生经常将数值、参数或计算结果直接写在公式内部、等式的某一项中、或者计算过程的推导链中，而不是单独写成独立的一行。请仔细扫描学生手写内容的每一处，寻找这些条目所要求的具体数值。

【题目背景】：{question_text}
【需要重新检查的条目（第一轮均被标记为"未书写"）】：{focused_checklist}
【第一轮已成功提取的其他条目（供上下文参考）】：{json.dumps(already_extracted, ensure_ascii=False)}

🚨【区域辨别铁律】：
- 你只应提取学生手写作答区域的内容，绝不能从打印的题目文本中抄录参数值。
- 如果某个条目的提取值与题目背景中的参数完全相同，这极有可能是你误读了题目文本，请重新确认。

对每个条目，请输出：
- 如果发现了学生写的具体内容（包括嵌入在公式/推导过程中的值），输出该具体内容
- 如果确实完全空白、没有任何笔迹，维持输出"未书写"

输出严格的 JSON 对象，key 为条目 id：
{{"item_id": "提取到的内容或未书写"}}
"""
    content_list = [{"type": "text", "text": reextraction_prompt}]
    if q_img_path and os.path.exists(q_img_path):
        q_b64 = encode_image_to_base64(q_img_path)
        content_list.extend([
            {"type": "text", "text": "\n【附图】:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{q_b64}"}}
        ])
    student_b64 = encode_image_to_base64(student_img_path)
    content_list.extend([
        {"type": "text", "text": "\n【考卷】:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{student_b64}"}}
    ])

    for attempt in range(2):
        try:
            fresh_client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)
            res = fresh_client.chat.completions.create(
                model=VLM_MODEL_NAME,
                messages=[{"role": "user", "content": content_list}],
                temperature=0.1, timeout=180
            )
            time.sleep(2)
            raw_result = res.choices[0].message.content.strip()

            reextracted = extract_and_parse_json(raw_result)
            if reextracted and isinstance(reextracted, dict):
                recovered = 0
                for k, v in reextracted.items():
                    if k in facts_dict and str(facts_dict[k]).strip() == "未书写":
                        new_val = str(v).strip()
                        if new_val and new_val != "未书写":
                            facts_dict[k] = new_val
                            recovered += 1
                if recovered > 0:
                    print(f"   🔄 [二次提取] 恢复了 {recovered}/{len(blank_items)} 个空白条目")
                    result = json.dumps(facts_dict, ensure_ascii=False)
                    return validate_extraction_against_question(question_text, result)
            return initial_facts_str
        except Exception as e:
            time.sleep(10)
    return initial_facts_str

def stage2_logic_grading(student_facts_str, rubrics_json_str, temperature=0.35):
    """
    Stage 2：语义匹配判分。通过内容语义匹配规则判断学生事实与标准是否等价。
    """
    logic_prompt = f"""
    你是一个极其严谨的计算机科学阅卷裁判。你的职责是判断学生的作答在语义上是否与评分标准匹配，而非进行表面的字符串对比。
    你需要严格根据【客观事实文本】，对照【细粒度评分标准】，计算总分。
    
    【客观事实文本】: {student_facts_str}
    【细粒度标准】: {rubrics_json_str}
    
    🚨【最高判决红线】:
    1. 只能依靠【客观事实文本】判决！只要该碎片化步骤标为”未书写”或”字迹模糊”，绝对无情扣除对应分数！
    2. 严禁动用同理心！严禁基于最终结果正确而去脑补学生懂了！没写就是没写！
    3. 🚨【信息不足拦截】：当客观事实的值为”是”、”有”、”已书写”、”存在”、”有书写”、”对”、”正确”、”有提取标注”、”有计算过程”、”有标注”等非具体内容时，该条目必须判 0 分，reason 填写”提取信息不足，无法判定”。
    4. 🚨【下游结果回溯规则】：评分项之间往往存在推导依赖关系——上游项（如参数识别、中间计算）是下游项（如最终结果、综合结论）的必要输入。当某个上游项的客观事实为"未书写"时，不能机械地判 BLANK，而应检查其下游依赖项：
    ① 如果存在至少一个下游项的结果数值正确（在容差范围内），且该下游项的正确结果在数学/逻辑上必然依赖于这个"未书写"项所要求的参数或中间步骤，则该"未书写"项应判为 MATCH 并给满分——正确的下游结果本身就是学生掌握了上游参数的铁证。
    ② 只有当所有依赖该参数的下游项结果也全部错误时，"未书写"才维持 BLANK 判决。
    ③ reason 中应注明"下游项正确，回溯推断该参数已正确使用"。

    5. 🚨【内容语义匹配规则】：在判断”匹配/不匹配”时，必须透过格式差异识别语义等价。具体原则：
       (a) 数值类：只比较核心数值，忽略单位格式差异（如”23位”=”23 位”=”23bit”=”23”）。忽略表达中的空格、标点、后缀符号。
           🚨【数值容差规则】：当评分项的标准答案是一个数值时，如果学生的数值与标准答案的相对误差 ≤ 10%，应判为 MATCH（满分）而非 SEMANTIC_FATAL。
           判断方法：提取学生答案和标准答案中的纯数值部分，计算 |学生值 - 标准值| / 标准值。例如标准=156，学生=162，误差=3.8% ≤ 10%，判 MATCH。
           单位换算：遇到”K”/”M”等单位缩写时，先换算为统一单位再比较（如 84Kb = 86016位）。如果换算后数值匹配或误差 ≤ 10%，判 MATCH。
           注意：只有当标准答案是明确的单一数值时才适用此规则。
           🚨【链式推导一致性规则】：当评分项的数值可由其他评分项推导得出时，验证步骤为：①先从标准答案推断正确的推导公式及常数（如标准控存容量86016=标准微指令长度168×512，则公式为微指令长度×512）；②用相同公式和常数作用于学生的上游项（如学生微指令33×512=16896）；③比较计算结果与学生的推导项（16896≠5940→不一致→SEMANTIC_FATAL）。只有当学生的推导项 = 正确公式(学生上游项) 时才判 MATCH。禁止通过找到一个能凑出学生答案的任意公式来判定一致。
           🚨【错误起点链式推导恢复规则】：当评分项之间存在明确的数学推导依赖关系时（如 item_1→item_2→item_3 形成计算链），如果学生的起始项值错误（被判 SEMANTIC_FATAL），但其下游项的值能够通过该错误起始值使用正确的公式和推导步骤计算得出（即学生使用了正确的方法，只是起点不同），则：起始项维持 SEMANTIC_FATAL（0分），下游推导项改为 PARTIAL_MATCH 并给予该条目 50% 的分数。验证方法：将学生的上游值代入标准公式，如果计算结果等于学生的下游答案，则确认推导正确。此规则仅适用于分值 ≥ 2 的推导类条目，不适用于识别/抄录类条目。reason 中注明链式推导内部一致：方法正确但起始值错误。
       (b) 序列类（二进制/十六进制/矩阵等）：去除所有空格、分隔符、进制标记后，比较纯字符序列是否一致。
           序列类宽容：如果学生序列与标准序列长度一致，且差异位数 ≤ 总位数的 10%（即 24 位序列允许 2-3 位错误），应判为 FORMAT_MINOR 而非 SEMANTIC_FATAL。只有当序列长度不一致或错误位数超过 10% 时才判 SEMANTIC_FATAL。
       (c) 过程类：不要求与标准措辞一致。只要学生的描述中包含了标准所要求的关键语义要素（即：操作了什么对象、进行了什么运算/比较、得出了什么中间结果），即判为匹配。允许表述顺序不同、详略不同。
       (d) 结论类：接受同义表达（如”不能””无法””不可以”均视为等价）。
       (e) 唯一拒判条件：学生的事实内容在语义上确实与标准矛盾（数值不同、结论相反），才可判不匹配。格式差异绝不能作为拒判理由。
       (f) 格式结构描述类：如果学生用字段名称、位范围、分区描述等方式表达了与标准相同的结构框架（字段的含义和顺序一致），即使未写出具体的位数数值，也应视为语义匹配。只有当学生描述的结构本身错误（字段缺失、顺序颠倒、含义不符）时才判不匹配。
       (g) 比例给分规则：当评分项分值 ≥ 3 分且包含多个可独立验证的评分要素时，如果学生的作答匹配了部分要素但非全部，应使用 PARTIAL_MATCH 并给予比例分数。禁止对多要素项目使用全有全无的 0/满分二分法。
    
    必须输出纯 JSON，每条 detail 必须包含 error_category 字段。
    {{
        "details": [
            {{"id": "1", "score_given": 0, "error_category": "BLANK", "reason": "未书写"}},
            {{"id": "2", "score_given": 2, "error_category": "MATCH", "reason": "匹配成功"}},
            {{"id": "3", "score_given": 0, "error_category": "SEMANTIC_FATAL", "reason": "数值矛盾"}},
            {{"id": "4", "score_given": 0, "error_category": "FORMAT_MINOR", "reason": "缺少单位"}},
            {{"id": "5", "score_given": 0, "error_category": "INSUFFICIENT_INFO", "reason": "提取信息不足"}},
            {{"id": "6", "score_given": 3, "error_category": "PARTIAL_MATCH", "reason": "部分匹配：答对2/3个要素"}}
        ],
        "total_score": 2
    }}

    🚨【error_category 枚举定义】（必须严格从以下6种中选择一个）：
    - "MATCH"：该条目得分 = 满分（语义匹配成功）
    - "BLANK"：学生未书写或字迹模糊（score_given 必须为 0）
    - "SEMANTIC_FATAL"：核心知识错误、结论相反、数值矛盾（score_given 必须为 0）
    - "FORMAT_MINOR"：格式不符、缺少单位、同义词未对齐等非实质性错误（score_given 必须为 0）
    - "INSUFFICIENT_INFO"：提取信息不足，无法判定（score_given 必须为 0）
    - "PARTIAL_MATCH"：该条目部分匹配，学生完成了部分评分要素但非全部。score_given 为按完成比例计算的部分分数（≥1 且 < 满分）。例如满分 5 分含 3 个要素，答对 2 个给 3 分。只有完全未涉及任何要素时才用 BLANK 或 SEMANTIC_FATAL。
    """
    for attempt in range(3):
        try:
            return call_text_model(
                [{"role": "user", "content": logic_prompt}],
                temperature=temperature, timeout=120
            )
        except Exception as e:
            time.sleep(3)
    return None

def zero_shot_leniency_agent(student_facts_str, strict_cot_str, rubrics_json_str):
    """
    宽容复查导师：以教师宽松阅卷的视角对初审扣分进行二次审查
    """
    leniency_prompt = f"""
# Role: 高校阅卷组长（教师视角复查）
你是一位经验丰富的高校教师，正在对机器初审的结果进行复查。
真实教师在批改时往往比较宽松：只要学生展现了对核心知识点的理解，即使表述不完全规范、中间步骤有省略，教师通常也会给分。

【评分标准】: {rubrics_json_str}
【学生客观事实】: {student_facts_str}
【初审扣分记录】: {strict_cot_str}

# 复查原则（教师宽松视角）
对初审中每个 score_given 为 0 的条目，用以下标准逐一复查：
1. **结论正确即给分**：如果学生写出了正确的最终结论（如”不能访问到””能访问到”），即使没有完整展示推导过程，教师通常会给予大部分分数。
2. **实质理解优先**：如果学生的表述虽然与标准答案措辞不同，但展现了对知识点的实质性理解（如正确识别了标记匹配关系），应恢复分数。
3. **过程省略宽容**：中间步骤省略但结论正确，教师一般只扣少量分甚至不扣。只有当学生的答案明显错误（数值算错、概念搞反）时才维持扣分。
4. **不恢复的情况**：学生确实在核心结论上完全错误（概念搞反、方法错误），维持 0 分。
5. **参数未单独列出的宽容**：如果某个条目因"未书写"被扣分，但该条目所要求的参数/中间值在逻辑上是另一个已正确作答条目的必要输入（例如正确算出了最终结果，说明学生必然正确使用了公式中的参数），教师通常会恢复分数。只有当依赖该参数的下游结果也错误时，才维持扣分。
6. **比例恢复原则**：对于分值 ≥ 3 分且含多个可独立评分要素的条目，如果学生被判 0 分但实际正确完成了部分要素（如计算过程正确但最终结果错误），应恢复比例分数（约 完成要素占比 × 满分），而非全有或全无。
7. **数值类评分项分层宽容原则**：仅适用于标准答案包含明确数值的评分项（如"23位"、"140条"、"86016位"）。对于描述性/概念性评分项（标准答案为文字描述、公式、结论），本原则不适用，仍按原则 1-5 判断。适用时按以下标准恢复：
   - 相对误差 ≤ 15%：恢复满分（计算过程中的合理误差，如进位、近似）
   - 相对误差 15%-50%：如果学生给出了非空白的具体数值答案且该数值在合理数量级内（非极端异常值如 0、1、999），恢复 ≥50% 分数（过程分——学生进行了计算但结果偏差较大）
   - 相对误差 > 50%：维持 0 分，除非适用原则 7（链式一致性）或学生展现了对计算方法的理解（如正确使用了单位换算、科学记数法等）
8. **链式推导内部一致性原则**：仅适用于数值类评分项。某些评分项的值可由其他评分项的数值推导得出。复查时验证步骤：①从标准答案推断正确的推导公式及常数（如标准控存86016=标准168×512，公式为微指令长度×512）；②用相同公式作用于学生的上游项；③如果学生的推导项等于计算结果，说明学生掌握了正确的计算方法，恢复满分。验证时必须使用正确的公式和常数（如×512而非×180），禁止用任意凑出的公式来判定一致。
9. **整体质量门控原则**：在应用原则 7 和 8 之前，先统计初审中被判为 SEMANTIC_FATAL 的条目占比。如果 SEMANTIC_FATAL 占比超过 50%（即超过一半的条目存在严重错误），说明学生对知识点掌握严重不足，此时原则 7 降级为仅恢复误差 ≤15% 的条目（15%-50% 的过程分不再适用），原则 8（链式推导）不再适用。只有 SEMANTIC_FATAL 占比 ≤ 50% 时，原则 6 和 7 才完整适用。原则 5（比例恢复）不受此限制。
10. **小分值条目宽容原则**：对于分值 ≤ 2 分的条目，教师通常采用"差不多就给"的宽松标准：
   - 如果学生的答案与标准在核心结论上一致（如命中/未命中判断正确、字段名称和顺序正确），但具体数值有微小偏差（如二进制提取中个别位错误、数值的最后一位不同），应恢复满分。
   - 如果学生的答案结构框架正确但数值完全错误，恢复 50% 分数（展现了对方法的理解）。
   - 只有当学生的结论完全相反（如命中判成未命中）或完全未涉及该知识点时，才维持 0 分。

🚨【硬约束】：
- 必须对每个被扣分条目逐一给出复查结论
- analysis_cot 中简要列出每条的复查决策（恢复/维持 + 一句话理由）

输出纯 JSON：
{{
    “secondary_total_score”: 复查后的总分(数字),
    “leniency_reason”: “简述主要纠正项，20字内”,
    “analysis_cot”: “逐条复查过程”
}}
"""
    for attempt in range(3):
        try:
            return call_text_model(
                [{"role": "user", "content": leniency_prompt}],
                temperature=0.3, timeout=120
            )
        except Exception as e:
            time.sleep(3)
    return None

def generate_neg_debate_summary(strict_cots):
    """为 NEG 拒绝域生成分歧焦点摘要，辅助人工审查"""
    if not strict_cots or len(strict_cots) < 2:
        return "分歧信息不足，无法生成摘要。"
    summaries = []
    for i, cot in enumerate(strict_cots):
        total = cot.get('total_score', '?')
        details_summary = "; ".join(
            f"条目{d.get('id','?')}={d.get('score_given',0)}分({d.get('reason','')[:20]})"
            for d in cot.get('details', [])[:5]
        )
        summaries.append(f"判决{i+1}: 总分{total} | {details_summary}")

    prompt = f"""以下是3次对同一份考卷的独立评分结果，它们之间存在严重分歧：
{chr(10).join(summaries)}

请用一句话（30字以内）总结评分分歧的核心焦点。只输出这句话，不要任何额外内容。"""
    for attempt in range(2):
        try:
            return call_text_model(
                [{"role": "user", "content": prompt}],
                temperature=0.1, timeout=30
            )
        except Exception:
            time.sleep(2)
    return "摘要生成失败，请人工核查。"

# ==================== 核心管线：零样本无监督 3WD ====================

def grade_student_3wd_pipeline(student_img_path, question_text, rubrics_json, teacher_score, q_img_path=None, blind_checklist=None):
    student_id = os.path.splitext(os.path.basename(student_img_path))[0]
    print(f"\n=============================================")
    print(f"🚀 开始批改试卷: [{student_id}]")
    print(f"=============================================")

    print(f"  👁️ [Stage 1] 纯视觉物理特征提取...")
    if blind_checklist is None:
        blind_checklist = generate_blind_checklist(rubrics_json)
    student_facts = stage1_blind_extraction(question_text, student_img_path, blind_checklist, q_img_path)

    if not student_facts:
        print(f"  ❌ 视觉提取失败，终止。")
        return None

    # 二次提取：对高留白率学生进行聚焦复查
    student_facts = stage1_targeted_reextraction(
        question_text, student_img_path, blind_checklist,
        student_facts, q_img_path
    )

    print(f"  ⚖️ [Stage 2] 微小温度独立盲审 (3次并行采样)...")
    model_scores = []
    strict_cots = []

    def _single_logic_probe(idx):
        res_text = stage2_logic_grading(student_facts, rubrics_json)
        if res_text:
            parsed_json = extract_and_parse_json(res_text)
            if parsed_json and 'total_score' in parsed_json:
                return (idx, parsed_json['total_score'], parsed_json)
        return (idx, None, None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_STAGE2) as s2_pool:
        s2_futures = [s2_pool.submit(_single_logic_probe, i) for i in range(3)]
        for future in concurrent.futures.as_completed(s2_futures):
            idx, score, cot = future.result()
            if score is not None:
                print(f"      ✅ [第 {idx+1}/3 次探测完成] 得分: {score}")
                model_scores.append(score)
                strict_cots.append(cot) 
    
    if len(model_scores) == 0: return None

    # 计算均分（向上取整）与标准差 (Self-Consistency)
    avg_model_score = math.ceil(np.mean(model_scores))
    std_dev = round(float(np.std(model_scores)), 4)
    
    # 动态解析评价标准总条目数和总分
    try:
        rubrics_data = json.loads(rubrics_json) if isinstance(rubrics_json, str) else rubrics_json
        TOTAL_ITEMS = len(rubrics_data) if isinstance(rubrics_data, list) else max(len(rubrics_data.keys()), 1)
        MAX_SCORE = sum(float(item.get('points', 0)) for item in rubrics_data) if isinstance(rubrics_data, list) else 100.0
    except:
        TOTAL_ITEMS, MAX_SCORE = 10, 10.0
    
    # 解析 Stage 1 提取的 JSON 事实，逐条检查 value 值
    try:
        facts_dict = json.loads(student_facts) if isinstance(student_facts, str) else student_facts
        if not isinstance(facts_dict, dict):
            facts_dict = {}
    except:
        facts_dict = {}

    fact_values = list(facts_dict.values())
    blank_count = sum(1 for v in fact_values if str(v).strip() == "未书写")
    perception_fail_count = sum(1 for v in fact_values if str(v).strip() == "字迹模糊")
    LOW_QUALITY_VALUES = {"是", "有", "已书写", "存在", "有书写", "对", "正确", "有提取标注", "有计算过程", "有标注"}
    low_quality_count = sum(1 for v in fact_values if str(v).strip() in LOW_QUALITY_VALUES)

    blank_rate = blank_count / TOTAL_ITEMS if TOTAL_ITEMS > 0 else 0
    perception_failure_rate = perception_fail_count / TOTAL_ITEMS if TOTAL_ITEMS > 0 else 0
    low_quality_rate = low_quality_count / TOTAL_ITEMS if TOTAL_ITEMS > 0 else 0

    # 综合提取质量判定
    if low_quality_rate <= 0.1 and perception_failure_rate <= 0.1:
        extraction_quality = "high"
    elif low_quality_rate >= 0.3 or perception_failure_rate >= 0.2:
        extraction_quality = "failed"
    else:
        extraction_quality = "low"

    real_diff = round(teacher_score - avg_model_score, 2) if teacher_score is not None else 0.0
    route = "UNKNOWN"
    final_score = avg_model_score
    reason_log = ""
    arbitration_flag = False

    print(f"\n  📊 [探测雷达指标] 均分={avg_model_score}, 标准差={std_dev:.4f}, 留白率={blank_rate:.0%}, 感知失效率={perception_failure_rate:.0%}, 低质量提取率={low_quality_rate:.0%}, 提取质量={extraction_quality}")

    # ==========================================
    # 中枢神经：纯正三支决策 (Three-Way Decision)
    # NEG → 拒绝域：提取质量不合格
    # BND → 边界域：模型不确定，Agent仲裁
    # POS → 接受域：模型确信，直接采信
    # ==========================================

    # 预计算归一化指标
    normalized_std = std_dev / MAX_SCORE if MAX_SCORE > 0 else 0
    score_spread = max(model_scores) - min(model_scores) if len(model_scores) >= 2 else 0
    spread_threshold = max(2.0, MAX_SCORE * 0.35)

    # 🛑 NEG 拒绝域
    # 条件 1：提取质量不合格（Stage 1 层面）
    if extraction_quality == "failed":
        route = "NEG"
        arbitration_flag = True
        neg_reason = "感知失效率" if perception_failure_rate >= 0.2 else "低质量提取率"
        print(f"      🛑 [路由 -> NEG] 提取质量不合格({neg_reason}: {max(perception_failure_rate, low_quality_rate):.0%})，拦截幻觉，移交人工！")

    # 条件 2：探测极端分裂（Stage 2 层面）
    # 归一化极差：极差占总分比例，带绝对值兜底防止小分题误触
    elif len(model_scores) >= 2 and score_spread >= spread_threshold:
        route = "NEG"
        arbitration_flag = True
        reason_log = generate_neg_debate_summary(strict_cots)
        print(f"      🛑 [路由 -> NEG] 探测极端分裂！极差={score_spread:.1f}(>={spread_threshold:.1f})，模型无共识，移交人工！")
        print(f"         🔍 [分歧焦点] {reason_log}")

    # ⚠️ BND 边界域：模型不确定，触发宽容导师Agent仲裁
    elif normalized_std >= 0.05 or (
        blank_rate <= 0.5 and avg_model_score <= MAX_SCORE * 0.80
    ):
        route = "BND"
        cot_context = json.dumps(strict_cots[0], ensure_ascii=False) if strict_cots else ""

        trigger_reason = "高认知方差" if normalized_std >= 0.05 else "内容-分数异常"
        print(f"      ⚠️ [路由 -> BND] {trigger_reason} (σ={std_dev:.4f}, blank={blank_rate:.0%}, avg={avg_model_score})！触发宽容导师仲裁...")

        agent_res_text = zero_shot_leniency_agent(student_facts, cot_context, rubrics_json)
        if agent_res_text:
            parsed_agent = extract_and_parse_json(agent_res_text)
            if parsed_agent:
                try: raw_agent_score = float(parsed_agent.get('secondary_total_score', avg_model_score))
                except: raw_agent_score = float(avg_model_score)
                agent_cap = round(avg_model_score + MAX_SCORE * 0.30, 1)
                final_score = max(min(raw_agent_score, agent_cap, MAX_SCORE), avg_model_score)
                reason_log = parsed_agent.get('leniency_reason', '')
                print(f"         ✨ [Agent 裁决] {reason_log} | 导师原始分: {raw_agent_score} | 最终分: {final_score}")

    # 🟢 POS 接受域
    else:
        route = "POS"
        print(f"      ✅ [路由 -> POS] 模型认知自洽 (σ={std_dev:.4f})，无异常，直接采信均分。")

    # 结果封装
    ordered_result = {
        "student_id": student_id,
        "teacher_score": teacher_score,
        "model_scores_history": model_scores,
        "model_avg_score": avg_model_score,
        "std_dev": std_dev,
        "blank_rate": round(blank_rate, 2),
        "perception_failure_rate": round(perception_failure_rate, 2),
        "low_quality_extraction_rate": round(low_quality_rate, 2),
        "extraction_quality": extraction_quality,
        "real_diff": real_diff,
        "3wd_route": route,
        "final_calibrated_score": final_score,
        "requires_human_arbitration": arbitration_flag,
        "reason_log": reason_log,
        "human_review_hint": reason_log if route == "NEG" else "",
        "facts": student_facts,
        "strict_cot": strict_cots[0] if strict_cots else {}
    }
    return ordered_result