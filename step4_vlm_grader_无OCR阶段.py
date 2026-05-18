import os
import json
import base64
import re
from zhipuai import ZhipuAI

# 🔴 请替换为你的真实 Key
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
        return json.loads(text)
    except Exception as e:
        print(f"❌ 判卷 JSON 解析失败: {e}\n模型原始输出: {text}")
        return None

def grade_student_answer_vlm(student_img_path, question_text, rubrics_json, q_img_path=None, ref_img_path=None):
    
    vlm_prompt = f"""
    你是一位极其严谨且铁面无私的视觉阅卷取证专家。
    
    【题目内容】：{question_text}
    【评分细则 (二元检查清单)】：
    {rubrics_json}
    
    【阅卷任务】：
    我提供了一张或多张图片。最后一张图片必定是学生的答题截图。如果有其他图片，则是标准参考答案。
    
    ⚠️ 【最高阅卷纪律：思维链（CoT）对账法】
    为了防止你产生“迎合标准答案”的幻觉，你必须严格按照以下 JSON 结构，分步进行“对账”。
    
    在填写 `student_raw_extraction` 字段时，你必须扮演一个没有任何专业知识的无情 OCR 扫描仪。
    🚨 【绝对禁止行为】：绝对不允许根据“评分细则”自动补全、美化或纠正学生的真实笔迹！
    
    💡 【通用转录行为规范示例 (请学习这种“只认死理”的客观转录精神)】：
    - 例1 (防脑补)：标准是“x = 5”，学生图上写着“x = s”。你必须提取为“x = s”，绝不能脑补成 5。
    - 例2 (防过滤)：标准是“大于 0”，学生图上写着“> -”。你必须提取为“> -”，绝不能强行补全为 0。
    - 例3 (抗涂改噪点)：标准是“True”，学生图上画了一个涂黑的墨团。你必须提取为“[涂改墨团]”，绝不能判定为 True。
    
    【执行步骤】：
    1. student_raw_extraction（学生原始痕迹）：依据上述规范，如实记录学生画了什么/写了什么残缺符号。
    2. standard_requirement（细则要求）：用简短的一句话提取该条细则要求看到的状态。
    3. semantic_comparison（语义对比）：对比前两项。逻辑符号（如 > 和 ->）允许等价宽容，但具体数值（数字/字母）和图表核心拓扑顺序，必须 100% 匹配，错一个字符即视为不匹配！
    4. satisfied：仅当语义和数值/拓扑完全匹配时输出 true，否则输出 false。
    
    【输出格式要求】：
    请严格输出 JSON 格式，直接返回如下结构：
    {{
        "evaluations": [
            {{
                "item": "评分细则中的原话", 
                "student_raw_extraction": "客观记录",
                "standard_requirement": "标准要求",
                "semantic_comparison": "详细对比过程",
                "satisfied": true或false
            }}
        ],
        "total_score": 总得分,
        "feedback": "一句话总结"
    }}
    """
    
    content_list = [{"type": "text", "text": vlm_prompt}]
    
    # 1. 插入题目原始附图（如果有）并打标
    if q_img_path and os.path.exists(q_img_path):
        content_list.append({"type": "text", "text": "\n【以下是题目的原始附图，仅供了解题目背景】："})
        q_b64 = encode_image_to_base64(q_img_path)
        q_mime = get_mime_type(q_img_path)
        content_list.append({"type": "image_url", "image_url": {"url": f"data:{q_mime};base64,{q_b64}"}})

    # 2. 插入官方参考图（🔥 重点：插入严厉的物理隔离警告）
    if ref_img_path and os.path.exists(ref_img_path):
        content_list.append({
            "type": "text", 
            "text": "\n🛑 [SYSTEM WARNING]: 下方紧跟的这张图片是【官方参考答案】，仅供你对比图形特征！绝对禁止将此图里的文字或数值作为学生的作答提取出来！"
        })
        ref_b64 = encode_image_to_base64(ref_img_path)
        ref_mime = get_mime_type(ref_img_path)
        content_list.append({"type": "image_url", "image_url": {"url": f"data:{ref_mime};base64,{ref_b64}"}})
        
    # 3. 插入学生作答图（🔥 重点：插入强硬的聚焦指令）
    student_b64 = encode_image_to_base64(student_img_path)
    if not student_b64:
        print(f"⚠️ 无法读取学生答卷: {student_img_path}")
        return None
    student_mime = get_mime_type(student_img_path)
    
    content_list.append({
        "type": "text", 
        "text": "\n🎯 [SYSTEM TARGET]: 下方紧跟的这张图片才是【考生的真实答卷】！你在 student_raw_extraction 字段中提取的任何笔迹、文字和内容，必须且只能来源于这最后一张图的肉眼所见！"
    })
    content_list.append({"type": "image_url", "image_url": {"url": f"data:{student_mime};base64,{student_b64}"}})
    
    # 4. 调用大模型
    try:
        response = client.chat.completions.create(
            model="glm-4.6v", 
            messages=[{"role": "user", "content": content_list}], # type: ignore
            temperature=0.1, 
        )
        result_text = response.choices[0].message.content.strip()
        return extract_and_parse_json(result_text)
    except Exception as e:
        print(f"❌ 视觉批改失败: {e}")
        return None