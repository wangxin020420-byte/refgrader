import os
import json
import base64
import random
import re
import time
from openai import OpenAI
from calibration_utils import prepare_rubrics_for_calibration
from rubric_semantics import (
    prepare_rubric_semantic_contract,
    validate_refined_rubric,
)

# 🔴 记得替换为你的真实 Key
CODING_PLAN_API_KEY = "132a47a6484e4a9dbfaa51fea40bbae0.LqWjKhw6WcH2sdFs"
CODING_PLAN_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4/"
client = OpenAI(api_key=CODING_PLAN_API_KEY, base_url=CODING_PLAN_BASE_URL)

VLM_MODEL_NAME = "glm-4.6v"
LOGIC_MODEL_NAME = "glm-5.1"

SUPPORTED_ANSWER_TYPES = {
    "direct_numeric",
    "derived_numeric",
    "numeric",
    "base_number",
    "bit_vector",
    "sequence",
    "set",
    "relation",
    "table_entry",
    "diagram_ocr",
    "formula",
    "method",
    "judgement",
    "concept_keyword",
}

ANSWER_TYPE_ALIASES = {
    "string": "concept_keyword",
    "text": "concept_keyword",
    "numeric_or_hex": "base_number",
    "hex_string": "base_number",
    "boolean_string": "judgement",
    "boolean": "judgement",
    "graph_node": "relation",
    "graph_edge": "relation",
    "diagram_node": "relation",
    "diagram_edge": "relation",
}

SUPPORTED_EVIDENCE_SOURCES = {"text", "formula", "table", "diagram"}


def normalize_answer_type(raw_type, item_text="", canonicalization=""):
    raw = str(raw_type or "").strip()
    lowered = raw.lower()
    if lowered in ANSWER_TYPE_ALIASES:
        return ANSWER_TYPE_ALIASES[lowered]
    if lowered in SUPPORTED_ANSWER_TYPES:
        return lowered

    text = f"{item_text} {canonicalization}"
    if re.search(r"\b[0-9A-Fa-f]+\s*[Hh]\b|[01]{4,}", text):
        return "base_number"
    if any(word in text for word in ("公式", "表达式", "formula", "method")):
        return "formula"
    if any(word in text for word in ("命中", "未命中", "溢出", "未溢出", "正确", "错误", "OF")):
        return "judgement"
    return "concept_keyword"


def normalize_evidence_source(raw_source, answer_type):
    source = str(raw_source or "").strip().lower()
    if source in SUPPORTED_EVIDENCE_SOURCES:
        return source
    if answer_type in {"relation", "diagram_ocr"}:
        return "diagram"
    if answer_type == "table_entry":
        return "table"
    if answer_type in {"formula", "method"}:
        return "formula"
    return "text"


def normalize_generated_rubric(rubric):
    """Normalize model-generated rubric schema before calibration."""
    if not isinstance(rubric, list):
        return rubric
    normalized = []
    for idx, raw_item in enumerate(rubric):
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        item.setdefault("id", f"auto_item_{idx}")
        item.setdefault("item", "")
        item["item"] = " ".join(str(item.get("item", "")).split())
        try:
            item["points"] = float(item.get("points", 0))
        except Exception:
            item["points"] = 0.0
        item.setdefault("source_text", item.get("item", ""))
        item.setdefault("parent_official_item", item.get("parent_id", ""))

        answer_type = normalize_answer_type(
            item.get("answer_type"),
            item_text=item.get("item", ""),
            canonicalization=item.get("canonicalization", ""),
        )
        item["answer_type"] = answer_type
        item["evidence_source"] = normalize_evidence_source(item.get("evidence_source"), answer_type)
        if item.get("canonicalization") is None:
            item.pop("canonicalization", None)
        normalized.append(item)
    return prepare_rubrics_for_calibration(normalized)

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
    init_prompt = f"""
你是计算机专业课程的自动阅卷评分准则结构化助手。
请把【官方评分准则】转换为可执行的 JSON rubric。

输入信息：
- 题目满分：{total_score}
- 题目内容：{question_text}
- 参考答案：{ref_answer}
- 官方评分准则：{official_rubric}

基本原则：
1. 忠实于官方评分准则，不得凭空增加新的得分点。
2. 初始生成阶段不要过度拆分官方粗粒度条款；只有当官方条款已经包含明确子事实时，才拆成多个小项。
3. 每个评分项必须是可检查的客观事实，例如具体数值、公式、判断结论、序列、表格项、图中关系。
4. 禁止使用“过程完整”“逻辑清晰”“理由充分”等主观描述。
5. 所有 points 之和必须严格等于 {total_score}。
6. item 字段必须是单行文本，不要包含换行。

answer_type 只能从下列集合中选择，不得创造新类型：
direct_numeric, derived_numeric, numeric, base_number, bit_vector,
sequence, set, relation, table_entry, diagram_ocr,
formula, method, judgement, concept_keyword

evidence_source 只能从下列集合中选择：
text, formula, table, diagram

字段要求：
- id：稳定唯一 ID。
- item：评分项描述，包含标准答案中的关键事实。
- points：该项分值。
- answer_type：从允许集合中选择。
- role：parameter / intermediate / method / final / unknown。
- canonicalization：建议的等价归一化方式，例如 numeric、base_number、bit_vector、sequence、set、formula、semantic_text。
- evidence_source：text / formula / table / diagram。
- source_text：来自官方评分准则或参考答案的原始依据。
- parent_official_item：对应官方粗粒度条款。

只输出 JSON 数组，不要输出 Markdown 或解释。
示例：
[
  {{
    "id": "item_1",
    "item": "写出最终结果为 128",
    "points": 2,
    "answer_type": "derived_numeric",
    "role": "final",
    "canonicalization": "numeric",
    "evidence_source": "text",
    "source_text": "官方评分准则中的对应条款",
    "parent_official_item": "official_1"
  }}
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
    current_rubric = normalize_generated_rubric(current_rubric)

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
            trial_prompt = f"""
你正在检查当前 rubric 是否存在粗粒度或歧义问题。

当前 rubric：
{json.dumps(current_rubric, ensure_ascii=False)}

请阅读学生答卷图片，判断是否存在由于 rubric 本身粒度过粗或表述歧义导致的评分冲突。
不要因为学生没写、图片看不清、OCR 失败而报告 rubric 冲突。

只输出 JSON 对象：
{{
  "has_conflict": true 或 false,
  "conflicted_item_ids": ["存在问题的 rubric id"],
  "conflict_reason": "说明为什么这是 rubric 粒度或歧义问题，而不是提取失败或学生错误"
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
            current_rubric = normalize_generated_rubric(refined_rubric)
            iteration += 1
        else:
            print("   ❌ 拆解生成失败，保留上一轮标准并终止迭代。")
            break

    print(f"\n🎉 [RRD 引擎停机] 细化完成！")
    # 👉 修复1：严格保留 ID 字段，防止方差修正函数因为找不到 ID 而崩溃
    final_rubric = []
    for idx, r in enumerate(current_rubric):
        item = dict(r)
        item.setdefault("id", f"auto_item_{idx}")
        item.setdefault("item", "")
        item.setdefault("points", 0)
        item.setdefault("source_text", item.get("item", ""))
        item.setdefault("parent_official_item", item.get("parent_id", ""))
        final_rubric.append(item)
    return normalize_generated_rubric(final_rubric)


#[新增] 基于高方差样本对评分规则进行全局修正。

def refine_rubric_based_on_variance(original_rubric_list, question_text, total_score, hard_samples_info):
    """
    [修正版] 基于高方差样本优化规则。
    """
    original_rubric_list = prepare_rubric_semantic_contract(original_rubric_list)
    print(f"🔧 [规则修正] 正在基于 {len(hard_samples_info)} 份高方差样本优化规则...")
    
    samples_desc = ""
    for idx, sample in enumerate(hard_samples_info):
        item_var = sample.get("item_variance", {})
        top_item_var = sorted(item_var.items(), key=lambda kv: kv[1], reverse=True)[:5] if item_var else []
        samples_desc += f"""
        === 疑难样本 {idx+1} ===
        【学生作答提取】: {sample.get('facts', '无法提取')}
        【多次试打分结果】: {sample.get('scores', [])} (波动大意味着规则判定模糊)
        【条目级方差Top】: {json.dumps(top_item_var, ensure_ascii=False)}
        【条目得分历史】: {json.dumps(sample.get('item_scores_history', {}), ensure_ascii=False)}
        【条目判定历史】: {json.dumps(sample.get('item_category_history', {}), ensure_ascii=False)}
        ---------------------
        """

    prompt = f"""
你是计算机相关课程的评分准则优化专家。现在要基于高方差样本，对当前 JSON rubric 做一次通用化修正。

【题目内容】
{question_text}

【题目总分】
{total_score}

【当前 rubric】
{json.dumps(original_rubric_list, ensure_ascii=False, indent=2)}

【高方差样本诊断信息】
{samples_desc}

【优化目标】
1. 只解决 rubric 本身的问题：表述歧义、粒度过粗、等价表达没有说明、答案类型或证据来源缺失。
2. 不要把 OCR/视觉提取失败、学生错误、模型偶然误判，当成 rubric 内容去硬改。
3. 不得使用教师分数历史、学生编号、具体题号等信息做针对性规则；生成结果必须能迁移到其他计算机课程题目。
4. 所有评分项 points 之和必须严格等于 {total_score}。
5. 每个原评分项都有稳定的 parent_id、parent_points 和 split_policy。不得删除父项语义，也不得把分值转移到其他父项。
6. scoring_policy=strict_atomic 的原子结果项禁止拆分计分。可补充 canonicalization 或 diagnostic_evidence，但不得改变正确答案和得分语义。
7. scoring_policy=additive_split 的父项只在确有多个独立必要条件时拆分；拆分子项的 parent_id 必须等于原父项 id，子项分值之和必须等于 parent_points。
8. scoring_policy=final_sufficient_partial_credit 表示“正确最终答案是父项满分的充分条件，同时错误最终答案仍可依据过程证据获得部分分”。此类父项必须：
   - 拆成至少一个客观过程项和一个最终答案项；
   - 恰好一个最终答案项设置 full_credit_trigger=true，其 standard_answer_text 必须等于 full_credit_anchor；
   - 所有非触发过程项 points 之和严格等于 fallback_cap；
   - 触发项 points 等于 parent_points-fallback_cap；
   - 所有子项 points 之和仍等于 parent_points，且正确最终答案不要求过程项同时出现。
9. 中间过程若既不是满分必要条件、也不属于明确的部分分兜底政策，应放入 diagnostic_evidence，不得新增为扣分前提。
10. 官方未给出子项权重时，不得按“技术重要性”主观分配分值。additive_split 只能使用等权正交原子项；hierarchical 父项必须使用已声明的 fallback_cap，过程项在 cap 内优先等权拆分，不得由样本临时调权。

【问题类型判定】
在修改前，先在内部判断每个暴露问题属于哪一类：
- rubric_ambiguity：评分项语义不清，导致不同评分轮次解释不一致。
- rubric_granularity：官方粗粒度条目过大，无法稳定判断局部得分。
- equivalent_representation_gap：学生等价表达没有被描述，例如进制、单位、序列方向、图/文字等价。
- extraction_failure：图像或 OCR 没提取到内容。
- scoring_model_error：rubric 清楚，但评分模型执行错误。

只有 rubric_ambiguity 和 rubric_granularity 可以改写或拆分评分项。
equivalent_representation_gap 只能补充 canonicalization / answer_type / evidence_source 等元数据，不得改变得分含义。
extraction_failure 和 scoring_model_error 不允许改变评分语义，只能保留原 rubric 或补充元数据。

【允许的 answer_type】
只能从以下集合选择，不得创造新类型：
direct_numeric, derived_numeric, numeric, base_number, bit_vector,
sequence, set, relation, table_entry, diagram_ocr,
formula, method, judgement, concept_keyword

【允许的 evidence_source】
只能从以下集合选择：
text, formula, table, diagram

【字段要求】
每个评分项必须包含：
- id：稳定唯一 ID；未修改的原条目尽量保留原 id。
- item：单行、客观、可检查的评分项描述，避免“逻辑清楚”“理由充分”等主观词。
- points：该项分值。
- answer_type：从允许集合中选择。
- role：parameter / intermediate / method / final / unknown。
- canonicalization：等价归一化说明，例如 numeric、base_number、bit_vector、sequence、set、formula、semantic_text。
- evidence_source：text / formula / table / diagram。
- source_text：来自官方评分准则或参考答案的依据。
- parent_official_item：对应的官方粗粒度条目。
- parent_id：必须引用当前 rubric 中某个原始父项 id。
- parent_points：保留该父项原始分值。
- split_policy：保留父项原值，不得自行修改。
- weighting_policy：保留父项原值；equal_atomic 要求所有计分子项等权。
- scoring_policy：保留父项原值，只能是 strict_atomic / additive_split / final_sufficient_partial_credit。
- full_credit_trigger：hierarchical 父项中恰好一个最终答案子项为 true，其他项为 false。
- full_credit_anchor：hierarchical 父项的充分满分答案，必须保留父项原值。
- fallback_cap：hierarchical 父项中过程兜底分的上限，必须保留父项原值。
- diagnostic_evidence：可选的零分诊断证据数组，不计入 points 总和。

【输出要求】
只输出 JSON 数组，不输出 Markdown，不输出解释文字。
"""

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
                
            new_rubric = normalize_generated_rubric(new_rubric)
            valid, validation_errors = validate_refined_rubric(
                original_rubric_list,
                new_rubric,
                total_score,
            )
            if not valid:
                reason = "; ".join(validation_errors)
                print(f"   ⚠️ 修正失败：语义契约校验未通过：{reason}")
                raise ValueError(reason)

            print("   ✅ 规则修正成功！")
            return prepare_rubric_semantic_contract(new_rubric)

        except Exception as e:
            print(f"   ⏳ [修正异常] (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print("   ❌ 规则修正彻底失败，沿用原规则。")
                return normalize_generated_rubric(original_rubric_list)
