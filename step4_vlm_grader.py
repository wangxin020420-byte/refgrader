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
from calibration_utils import (
    a3wa_dynamic_bounds,
    apply_boundary_action_policy,
    build_a3wa_decision,
    build_post_grading_calibration,
    prepare_rubrics_for_calibration,
)

# ==================== 配置区 ====================
# 视觉模型切换：修改此处即可，可选 "glm4v" / "glm5v"
VLM_MODEL_PROVIDER = "glm4v"
VLM_MODELS = {
    "glm4v": "glm-4.6v",
    "glm5v": "glm-5v-turbo",
}
VLM_MODEL_NAME = VLM_MODELS.get(VLM_MODEL_PROVIDER, "glm-4.6v")

A3WA_CALIBRATION_CONFIG_PATH = os.getenv(
    "A3WA_CALIBRATION_CONFIG",
    os.path.join("results_rrd_vlm", "a3wa_calibration_config.json"),
)
_A3WA_RUNTIME_CONFIG = None
PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

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

def render_prompt_template(filename, replacements, fallback):
    """Load a UTF-8 prompt template and replace {{PLACEHOLDER}} tokens."""
    path = os.path.join(PROMPT_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            prompt = f.read()
        for key, value in replacements.items():
            prompt = prompt.replace("{{" + key + "}}", str(value))
        return prompt
    except Exception as exc:
        print(f"      [Prompt template] fallback to inline prompt: {filename} ({exc})")
        return fallback

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
    6. 🚨【核心能力锚定规则】：当评分条目呈现”参数识别→公式/方法→最终结果”的层次依赖关系时，如果学生的核心推导项（公式/方法条目）和最终结果条目**全部**被判为 SEMANTIC_FATAL 或 BLANK，说明学生仅完成了低认知负荷的参数抄录，未展现对核心知识点的掌握。此时：
       - 所有纯参数识别类条目（仅需从题干直接读取、不涉及计算或分析的条目）的得分上限降为该条目满分的 30%。
       - reason 中注明”核心推导全错，参数识别分降级”。
       - 如果核心推导项中至少有一项被判为 MATCH 或 PARTIAL_MATCH，则不触发此降级。
       (a) 数值类：只比较核心数值，忽略单位格式差异（如”23位”=”23 位”=”23bit”=”23”）。忽略表达中的空格、标点、后缀符号。
           🚨【数值容差规则】：当评分项的标准答案是一个数值时，如果学生的数值与标准答案的相对误差 ≤ 10%，应判为 MATCH（满分）而非 SEMANTIC_FATAL。
           判断方法：提取学生答案和标准答案中的纯数值部分，计算 |学生值 - 标准值| / 标准值。例如标准=156，学生=162，误差=3.8% ≤ 10%，判 MATCH。
           单位换算：遇到”K”/”M”等单位缩写时，先换算为统一单位再比较（如 84Kb = 86016位）。如果换算后数值匹配或误差 ≤ 10%，判 MATCH。
           注意：只有当标准答案是明确的单一数值时才适用此规则。
           🚨【链式推导一致性规则】：当评分项的数值可由其他评分项推导得出时，验证步骤为：①先从标准答案推断正确的推导公式及常数（如标准控存容量86016=标准微指令长度168×512，则公式为微指令长度×512）；②用相同公式和常数作用于学生的上游项（如学生微指令33×512=16896）；③比较计算结果与学生的推导项（16896≠5940→不一致→SEMANTIC_FATAL）。只有当学生的推导项 = 正确公式(学生上游项) 时才判 MATCH。禁止通过找到一个能凑出学生答案的任意公式来判定一致。
           🚨【错误起点链式推导恢复规则】：当评分项之间存在明确的数学推导依赖关系时（如 item_1→item_2→item_3 形成计算链），如果学生的起始项值错误（被判 SEMANTIC_FATAL），但其下游项的值能够通过该错误起始值使用正确的公式和推导步骤计算得出（即学生使用了正确的方法，只是起点不同），则：起始项维持 SEMANTIC_FATAL（0分），下游推导项改为 PARTIAL_MATCH 并给予该条目 50% 的分数。验证方法：将学生的上游值代入标准公式，如果计算结果等于学生的下游答案，则确认推导正确。此规则仅适用于分值 ≥ 2 的推导类条目，不适用于识别/抄录类条目。reason 中注明链式推导内部一致：方法正确但起始值错误。
           🚨【数量级感知规则】：当学生的数值与标准答案的相对误差 > 10%，但满足以下全部条件时，将 error_category 从 SEMANTIC_FATAL 降级为 PARTIAL_MATCH，给予该条目满分的 50%：
           ① 学生的数值与标准答案的比值恰好为 10^n（n为非零整数，如 0.1倍、10倍、100倍），或与另一评分项的正确值的 10^n 倍一致（说明学生混淆了两个相似参数）。
           ② 该条目存在上游依赖项，且至少一个上游依赖项已被判为 SEMANTIC_FATAL 或 PARTIAL_MATCH（说明错误是从上游传播而来，不是凭空产生）。
           ③ 学生的答案不是 0 或极端异常值（排除完全不会做的情况）。
           reason 中注明"数量级偏差但方法论可循"。
           注意：此规则不与 10% 容差规则叠加。先检查 10% 容差，不满足时再检查此规则。
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
            {{"id": "4", "score_given": 1, "error_category": "FORMAT_MINOR", "reason": "单位书写不规范（如重复标注GHz），核心数值正确，70%给分"}},
            {{"id": "5", "score_given": 0, "error_category": "INSUFFICIENT_INFO", "reason": "提取信息不足"}},
            {{"id": "6", "score_given": 3, "error_category": "PARTIAL_MATCH", "reason": "部分匹配：答对2/3个要素"}}
        ],
        "total_score": 2
    }}

    🚨【error_category 枚举定义】（必须严格从以下6种中选择一个）：
    - "MATCH"：该条目得分 = 满分（语义匹配成功）
    - "BLANK"：学生未书写或字迹模糊（score_given 必须为 0）
    - "SEMANTIC_FATAL"：核心知识错误、结论相反、数值矛盾（score_given 必须为 0）
    - "FORMAT_MINOR"：格式不符、缺少单位、同义表达未对齐等非实质性错误。如果该条目的核心数值/结论正确，仅格式有瑕疵，给予该条目满分的 70%（向上取整，最少 1 分）。如果该条目的核心数值/结论也错误，则 score_given 为 0。reason 中注明格式差异的具体内容。
    - "INSUFFICIENT_INFO"：提取信息不足，无法判定（score_given 必须为 0）
    - "PARTIAL_MATCH"：该条目部分匹配，学生完成了部分评分要素但非全部。score_given 为按完成比例计算的部分分数（≥1 且 < 满分）。例如满分 5 分含 3 个要素，答对 2 个给 3 分。只有完全未涉及任何要素时才用 BLANK 或 SEMANTIC_FATAL。
    """
    logic_prompt = render_prompt_template(
        "stage2_logic_grading.md",
        {
            "STUDENT_FACTS": student_facts_str,
            "RUBRICS_JSON": rubrics_json_str,
        },
        logic_prompt,
    )
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

def _clamp(value, lower, upper):
    return max(lower, min(upper, value))

def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default

def _sum_agent_item_points(items):
    if not isinstance(items, list):
        return 0.0
    total = 0.0
    for item in items:
        if isinstance(item, dict):
            total += max(_safe_float(item.get("points", 0.0), 0.0), 0.0)
    return total

def _agent_candidate_score(parsed_agent, avg_model_score, max_score):
    """Prefer structured missed/over credit deltas; fall back to legacy total score."""
    avg_model_score = _safe_float(avg_model_score, 0.0)
    max_score = max(_safe_float(max_score, 0.0), 1.0)
    missed = _sum_agent_item_points(parsed_agent.get("missed_credit_items"))
    over = _sum_agent_item_points(parsed_agent.get("over_credit_items"))
    if missed > 0 or over > 0:
        return _clamp(avg_model_score + missed - over, 0, max_score)
    return _safe_float(
        parsed_agent.get("calibrated_score", parsed_agent.get("secondary_total_score", avg_model_score)),
        avg_model_score
    )

def _rubric_points_map(rubrics_data):
    if not isinstance(rubrics_data, list):
        return {}, 1.0
    points_map = {}
    points_values = []
    for item in rubrics_data:
        item_id = str(item.get("id", ""))
        points = _safe_float(item.get("points", 0), 0.0)
        if item_id:
            points_map[item_id] = points
        if points > 0:
            points_values.append(points)
    fallback = float(np.mean(points_values)) if points_values else 1.0
    return points_map, fallback


def load_a3wa_runtime_config():
    global _A3WA_RUNTIME_CONFIG
    if _A3WA_RUNTIME_CONFIG is not None:
        return _A3WA_RUNTIME_CONFIG
    _A3WA_RUNTIME_CONFIG = {}
    if not A3WA_CALIBRATION_CONFIG_PATH or not os.path.exists(A3WA_CALIBRATION_CONFIG_PATH):
        return _A3WA_RUNTIME_CONFIG
    try:
        with open(A3WA_CALIBRATION_CONFIG_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            _A3WA_RUNTIME_CONFIG = loaded
            print(f"      [A3WA config] loaded {A3WA_CALIBRATION_CONFIG_PATH}")
    except Exception as exc:
        print(f"      [A3WA config] failed to load {A3WA_CALIBRATION_CONFIG_PATH}: {exc}")
        _A3WA_RUNTIME_CONFIG = {}
    return _A3WA_RUNTIME_CONFIG

def _category_points_ratio(strict_cots, rubrics_data, max_score):
    points_map, fallback_points = _rubric_points_map(rubrics_data)
    category_points = {
        "SEMANTIC_FATAL": [],
        "PARTIAL_MATCH": [],
        "FORMAT_MINOR": [],
    }

    for cot in strict_cots:
        per_cot = {k: 0.0 for k in category_points}
        for detail in cot.get("details", []):
            category = detail.get("error_category", "")
            if category not in per_cot:
                continue
            item_id = str(detail.get("id", ""))
            per_cot[category] += points_map.get(item_id, fallback_points)
        for category, points in per_cot.items():
            category_points[category].append(points)

    denom = max(max_score, 1.0)
    return {
        "fatal_points_ratio": float(np.mean(category_points["SEMANTIC_FATAL"]) / denom) if category_points["SEMANTIC_FATAL"] else 0.0,
        "partial_match_points_ratio": float(np.mean(category_points["PARTIAL_MATCH"]) / denom) if category_points["PARTIAL_MATCH"] else 0.0,
        "format_minor_points_ratio": float(np.mean(category_points["FORMAT_MINOR"]) / denom) if category_points["FORMAT_MINOR"] else 0.0,
    }

def build_risk_profile(
    question_text,
    facts_dict,
    rubrics_data,
    strict_cots,
    model_scores,
    avg_model_score,
    std_dev,
    max_score,
    total_items,
    blank_rate,
    low_quality_rate,
    perception_failure_rate,
    extraction_quality,
):
    score_spread = max(model_scores) - min(model_scores) if len(model_scores) >= 2 else 0.0
    std_ratio = std_dev / max_score if max_score > 0 else 0.0
    spread_ratio = score_spread / max_score if max_score > 0 else 0.0
    avg_ratio = avg_model_score / max_score if max_score > 0 else 0.0
    points_ratios = _category_points_ratio(strict_cots, rubrics_data, max_score)

    # Paper-friendly 3WD variables:
    # P: perception risk, U: uncertainty, F: fatal-error points ratio,
    # H: high-blank/high-score contradiction, L: lenient-review trigger.
    perception_risk = max(
        perception_failure_rate / 0.20 if 0.20 > 0 else 0.0,
        low_quality_rate / 0.30 if 0.30 > 0 else 0.0,
    )
    uncertainty_index = std_ratio
    fatal_points_ratio = points_ratios["fatal_points_ratio"]
    high_blank_high_score = blank_rate >= 0.50 and avg_ratio >= 0.60
    lenient_review_signal = avg_ratio <= 0.60 and blank_rate <= 0.35

    reject_domain = (
        perception_risk >= 1.0
        or uncertainty_index >= 0.15
        or fatal_points_ratio >= 0.70
        or (high_blank_high_score and perception_risk >= 0.50)
    )

    boundary_domain = (
        perception_risk >= 0.33
        or 0.05 <= uncertainty_index < 0.15
        or 0.30 <= fatal_points_ratio < 0.70
        or high_blank_high_score
        or lenient_review_signal
    )

    risk_features = {
        "perception_risk": round(perception_risk, 4),
        "uncertainty_index": round(uncertainty_index, 4),
        "fatal_points_ratio": round(fatal_points_ratio, 4),
        "high_blank_high_score": high_blank_high_score,
        "lenient_review_signal": lenient_review_signal,
        "reject_domain": reject_domain,
        "boundary_domain": boundary_domain,
        "std_ratio": round(std_ratio, 4),
        "spread_ratio": round(spread_ratio, 4),
        "avg_ratio": round(avg_ratio, 4),
        "score_spread": round(score_spread, 4),
        "blank_rate": round(blank_rate, 4),
        "low_quality_rate": round(low_quality_rate, 4),
        "perception_failure_rate": round(perception_failure_rate, 4),
        "partial_match_points_ratio": round(points_ratios["partial_match_points_ratio"], 4),
        "format_minor_points_ratio": round(points_ratios["format_minor_points_ratio"], 4),
    }

    return {
        "perception_risk": perception_risk,
        "uncertainty_index": uncertainty_index,
        "fatal_points_ratio": fatal_points_ratio,
        "high_blank_high_score": high_blank_high_score,
        "lenient_review_signal": lenient_review_signal,
        "reject_domain": reject_domain,
        "boundary_domain": boundary_domain,
        "risk_features": risk_features,
    }

def boundary_arbitration_agent(student_facts_str, strict_cots, rubrics_json_str, risk_profile):
    arbitration_prompt = f"""
# Role: 边界样本双向校准仲裁员
你正在复核一份自动阅卷结果。你的目标是使评分尽可能接近真实教师的评分。

【评分标准】
{rubrics_json_str}

【学生客观作答事实】
{student_facts_str}

【三次独立评分记录】
{json.dumps(strict_cots, ensure_ascii=False)}

【风险特征】
{json.dumps(risk_profile, ensure_ascii=False)}

仲裁原则：
1. 双向校准：你需要同时考虑模型可能高估和低估的情况。
   - 当学生有正确的过程推导但被判 SEMANTIC_FATAL 时，考虑上调。
   - 当学生的核心结论完全错误但被判 MATCH 时，考虑下调。
2. 过程分恢复：如果学生的计算方法论正确但初始参数有误导致结果偏差，这是典型的低估场景，应适当上调。
3. 空洞分纠正：如果学生仅正确识别了题干参数但核心推导和结果全错，大量参数识别分可能是高估，应适当下调。
4. 格式不应重罚：格式问题（单位书写习惯、变量名不规范）不应实质性影响分数。
5. 最终分数必须在 [0, 题目满分] 范围内。
6. 如果证据不足以确定方向，保持原分。

请输出纯 JSON：
{{
  "decision": "raise 或 keep 或 cautious_lower",
  "calibrated_score": 数字,
  "reason": "50字以内说明"
}}
"""
    arbitration_prompt = render_prompt_template(
        "boundary_arbitration.md",
        {
            "RUBRICS_JSON": rubrics_json_str,
            "STUDENT_FACTS": student_facts_str,
            "STRICT_COTS_JSON": json.dumps(strict_cots, ensure_ascii=False),
            "RISK_PROFILE_JSON": json.dumps(risk_profile, ensure_ascii=False),
        },
        arbitration_prompt,
    )
    for attempt in range(3):
        try:
            return call_text_model(
                [{"role": "user", "content": arbitration_prompt}],
                temperature=0.2, timeout=120
            )
        except Exception:
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

    # 计算均分与标准差 (Self-Consistency)
    avg_model_score = round(float(np.mean(model_scores)), 1)
    std_dev = round(float(np.std(model_scores)), 4)
    
    # 动态解析评价标准总条目数和总分
    try:
        rubrics_data = json.loads(rubrics_json) if isinstance(rubrics_json, str) else rubrics_json
        rubrics_data = prepare_rubrics_for_calibration(rubrics_data)
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
    # 风险驱动三支决策 (Three-Way Decision)
    # POS：低风险直接接受
    # BND：中风险宽松优先、谨慎下调
    # NEG：高风险拒判/人工复核
    # ==========================================
    risk_profile = build_risk_profile(
        question_text=question_text,
        facts_dict=facts_dict,
        rubrics_data=rubrics_data,
        strict_cots=strict_cots,
        model_scores=model_scores,
        avg_model_score=avg_model_score,
        std_dev=std_dev,
        max_score=MAX_SCORE,
        total_items=TOTAL_ITEMS,
        blank_rate=blank_rate,
        low_quality_rate=low_quality_rate,
        perception_failure_rate=perception_failure_rate,
        extraction_quality=extraction_quality,
    )
    post_calibration = build_post_grading_calibration(
        facts_dict=facts_dict,
        rubrics_data=rubrics_data,
        strict_cots=strict_cots,
        avg_model_score=avg_model_score,
        max_score=MAX_SCORE,
        blank_rate=blank_rate,
        risk_profile=risk_profile,
    )
    risk_profile["risk_features"].update({
        "unsupported_match_points_ratio": post_calibration["unsupported_match_points_ratio"],
        "method_final_verified_ratio": post_calibration["method_final_verified_ratio"],
        "direct_awarded_ratio": post_calibration["direct_awarded_ratio"],
        "metadata_coverage": post_calibration["metadata_coverage"],
        "explicit_chain_coverage": post_calibration["explicit_chain_coverage"],
        "core_anchor_failed": post_calibration["core_anchor_failed"],
        "visual_blank_review": post_calibration["visual_blank_review"],
        "calibration_rule_hits": post_calibration["rule_hits"],
    })
    if post_calibration["reject_domain"]:
        risk_profile["reject_domain"] = True
        risk_profile["boundary_domain"] = False
    elif post_calibration["boundary_domain"]:
        risk_profile["boundary_domain"] = True
    risk_profile["risk_features"]["reject_domain"] = risk_profile["reject_domain"]
    risk_profile["risk_features"]["boundary_domain"] = risk_profile["boundary_domain"]

    a3wa_config = load_a3wa_runtime_config()
    a3wa_decision = build_a3wa_decision(
        model_scores=model_scores,
        avg_model_score=avg_model_score,
        std_dev=std_dev,
        max_score=MAX_SCORE,
        blank_rate=blank_rate,
        low_quality_rate=low_quality_rate,
        perception_failure_rate=perception_failure_rate,
        extraction_quality=extraction_quality,
        fatal_points_ratio=risk_profile["fatal_points_ratio"],
        high_blank_high_score=risk_profile["high_blank_high_score"],
        post_calibration=post_calibration,
        weights=a3wa_config.get("risk_weights"),
        loss_params=a3wa_config.get("loss_params"),
    )
    risk_profile["risk_features"].update({
        "a3wa_risk": a3wa_decision["risk"],
        "a3wa_mu": a3wa_decision["mu"],
        "a3wa_alpha": a3wa_decision["alpha"],
        "a3wa_beta": a3wa_decision["beta"],
        "a3wa_m": a3wa_decision["m"],
        "a3wa_route": a3wa_decision["route"],
        "a3wa_reason": a3wa_decision["reason"],
        "a3wa_risk_components": a3wa_decision["risk_components"],
    })

    perception_risk = risk_profile["perception_risk"]
    uncertainty_index = risk_profile["uncertainty_index"]
    fatal_points_ratio = risk_profile["fatal_points_ratio"]
    high_blank_high_score = risk_profile["high_blank_high_score"]
    lenient_review_signal = risk_profile["lenient_review_signal"]
    reject_domain = a3wa_decision["route"] == "NEG"
    boundary_domain = a3wa_decision["route"] == "BND"
    risk_profile["reject_domain"] = reject_domain
    risk_profile["boundary_domain"] = boundary_domain
    risk_profile["risk_features"]["reject_domain"] = reject_domain
    risk_profile["risk_features"]["boundary_domain"] = boundary_domain
    risk_features = risk_profile["risk_features"]
    arbitration_decision = "accept"
    boundary_gate = None

    print(
        "      📡 [风险画像] "
        f"P={perception_risk:.2f}, U={uncertainty_index:.2%}, F={fatal_points_ratio:.2%}, "
        f"H={high_blank_high_score}, L={lenient_review_signal}"
    )
    if post_calibration["rule_hits"]:
        print(
            "      🧭 [通用校准] "
            f"rules={post_calibration['rule_hits']}, "
            f"UM={post_calibration['unsupported_match_points_ratio']:.2%}, "
            f"MF={post_calibration['method_final_verified_ratio']:.2%}, "
            f"cap={post_calibration['upper_bound']:.2f}"
        )
    print(
        "      🧮 [A3WA可信度] "
        f"R={a3wa_decision['risk']:.3f}, μ={a3wa_decision['mu']:.3f}, "
        f"α={a3wa_decision['alpha']:.3f}, β={a3wa_decision['beta']:.3f}, "
        f"route={a3wa_decision['route']} | {a3wa_decision['reason']}"
    )

    if reject_domain:
        route = "NEG"
        arbitration_flag = True
        if a3wa_decision["hard_neg_reasons"]:
            reason_log = "A3WA硬拒判：" + ",".join(a3wa_decision["hard_neg_reasons"])
        else:
            reason_log = f"A3WA低可信度拒判：μ={a3wa_decision['mu']:.3f} <= β={a3wa_decision['beta']:.3f}"
        print(f"      🛑 [路由 -> NEG] 高风险拒判 | P={perception_risk:.2f}, U={uncertainty_index:.2%}, F={fatal_points_ratio:.2%}, H={high_blank_high_score}")
        print(f"         🔍 [复核提示] {reason_log}")

    elif boundary_domain:
        route = "BND"
        print(
            f"      ⚠️ [路由 -> BND] 边界样本 | "
            f"P={perception_risk:.2f}, U={uncertainty_index:.2%}, F={fatal_points_ratio:.2%}, H={high_blank_high_score}, L={lenient_review_signal}"
        )

        agent_res_text = boundary_arbitration_agent(student_facts, strict_cots, rubrics_json, risk_profile)
        if agent_res_text:
            parsed_agent = extract_and_parse_json(agent_res_text)
            if parsed_agent:
                raw_agent_score = _agent_candidate_score(parsed_agent, avg_model_score, MAX_SCORE)
                arbitration_decision = parsed_agent.get("decision", "keep")
                boundary_gate = apply_boundary_action_policy(
                    avg_model_score=avg_model_score,
                    candidate_score=raw_agent_score,
                    max_score=MAX_SCORE,
                    a3wa_decision=a3wa_decision,
                    risk_profile=risk_profile,
                    post_calibration=post_calibration,
                )
                final_score = round(_clamp(boundary_gate["final_score"], 0, MAX_SCORE), 2)
                arbitration_decision = f"{arbitration_decision}|{boundary_gate['action']}"
                risk_features["boundary_gate_action"] = boundary_gate["action"]
                risk_features["boundary_gate_accepted"] = boundary_gate["accepted"]
                reason_log = parsed_agent.get("reason", parsed_agent.get("leniency_reason", ""))
                print(
                    f"         ✨ [Agent 仲裁] {arbitration_decision} | "
                    f"原始仲裁分: {raw_agent_score} | 限幅后最终分: {final_score} | {reason_log}"
                )
            else:
                arbitration_decision = "keep"
                lower_bound, upper_bound, _ = a3wa_dynamic_bounds(
                    avg_model_score=avg_model_score,
                    max_score=MAX_SCORE,
                    a3wa_decision=a3wa_decision,
                    risk_profile=risk_profile,
                    post_calibration=post_calibration,
                )
                final_score = round(_clamp(avg_model_score, lower_bound, upper_bound), 2)
        else:
            arbitration_decision = "keep"
            lower_bound, upper_bound, _ = a3wa_dynamic_bounds(
                avg_model_score=avg_model_score,
                max_score=MAX_SCORE,
                a3wa_decision=a3wa_decision,
                risk_profile=risk_profile,
                post_calibration=post_calibration,
            )
            final_score = round(_clamp(avg_model_score, lower_bound, upper_bound), 2)

    else:
        route = "POS"
        final_score = avg_model_score
        print(f"      ✅ [路由 -> POS] A3WA高可信自动接受，直接采信模型均分。")

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
        "perception_risk": round(perception_risk, 4),
        "uncertainty_index": round(uncertainty_index, 4),
        "fatal_points_ratio": round(fatal_points_ratio, 4),
        "high_blank_high_score": high_blank_high_score,
        "lenient_review_signal": lenient_review_signal,
        "risk_features": risk_features,
        "post_calibration": post_calibration,
        "a3wa_decision": a3wa_decision,
        "boundary_gate": boundary_gate,
        "arbitration_decision": arbitration_decision,
        "reason_log": reason_log,
        "human_review_hint": reason_log if route == "NEG" else "",
        "facts": student_facts,
        "strict_cot": strict_cots[0] if strict_cots else {},
        "strict_cots_all": strict_cots
    }
    return ordered_result
