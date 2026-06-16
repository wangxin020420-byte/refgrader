import os
import json
import math
from json_repair import repair_json
import base64
import re
import time
import concurrent.futures
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageOps
import io
import numpy as np
from calibration_utils import (
    a3wa_dynamic_bounds,
    apply_boundary_action_policy,
    build_a3wa_decision,
    build_post_grading_calibration,
    compute_extraction_quality_counts,
    compute_extraction_risk_features,
    is_blank_extraction,
    is_low_quality_extraction,
    is_perception_failure,
    is_structure_missing_extraction,
    prepare_rubrics_for_calibration,
    select_baseline_score,
)

# ==================== 閰嶇疆鍖?====================
# 瑙嗚妯″瀷鍒囨崲锛氫慨鏀规澶勫嵆鍙紝鍙€?"glm4v" / "glm5v"
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

# 鏂囨湰妯″瀷鍒囨崲锛氫慨鏀规澶勫嵆鍙紝鍙€?"glm" / "glm5" / "deepseek"
TEXT_MODEL_PROVIDER = "glm5"

# Coding Plan 缁熶竴閰嶇疆锛圤penAI 鍏煎鎺ュ彛锛?
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

# DeepSeek 閰嶇疆
DEEPSEEK_API_KEY = "sk-6lCywlyf1xwXyV8G937sOrRF7kGThWMrwFVksuwGZaAWrAzP"
DEEPSEEK_BASE_URL = "https://gpt-agent.cc/v1"
DEEPSEEK_MODEL_NAME = "deepseek-v4-flash"

# 骞跺彂閰嶇疆锛歿provider: (澶栧眰瀛︾敓骞跺彂, 鍐呭眰Stage2鎺㈡祴骞跺彂)}
MODEL_CONCURRENCY = {
    "glm":      (3, 3),  # GLM-4.5-air 骞跺彂鑳藉姏寮?
    "glm5":     (2, 2),  # GLM-5.1 Coding Pro 闄愭祦杈冧弗锛岄檷浣庡苟鍙?
    "deepseek": (2, 2),  # 绗笁鏂逛唬鐞嗭紝淇濆畧涓€鐐?
}
MAX_WORKERS_OUTER = MODEL_CONCURRENCY.get(TEXT_MODEL_PROVIDER, (3, 3))[0]
MAX_WORKERS_STAGE2 = MODEL_CONCURRENCY.get(TEXT_MODEL_PROVIDER, (3, 3))[1]

# 鍏ㄥ眬瀹㈡埛绔?
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

# ==================== 缁熶竴鏂囨湰妯″瀷璋冪敤 ====================

def call_text_model(messages, temperature=0.2, timeout=120):
    """缁熶竴鏂囨湰妯″瀷璋冪敤鍏ュ彛锛屾牴鎹?TEXT_MODEL_PROVIDER 鑷姩鍒嗗彂"""
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
                print(f"         [GLM-5 retry {attempt+1}/4] {type(e).__name__}: {str(e)[:80]}... wait {wait}s")
                time.sleep(wait)
        raise Exception("GLM-5 4娆￠噸璇曞潎澶辫触")
    elif TEXT_MODEL_PROVIDER == "deepseek":
        # DeepSeek锛氭瘡璇锋眰鐙珛瀹㈡埛绔?+ 鎸囨暟閫€閬块噸璇曪紝瑙ｅ喅骞跺彂闄愭祦
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
                print(f"         [DeepSeek retry {attempt+1}/4] {type(e).__name__}: {str(e)[:80]}... wait {wait}s")
                time.sleep(wait)
        raise Exception("DeepSeek 4娆￠噸璇曞潎澶辫触")
    else:
        response = glm_client.chat.completions.create(
            model=GLM_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            timeout=timeout
        )
        return response.choices[0].message.content.strip()

# ==================== 宸ュ叿鍑芥暟 (涓嶅彉) ====================

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
            print(f"   馃棞锔?[鍘嬬缉] {width}x{height} -> {new_width}x{new_height}")
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
    except Exception as e:
        print(f"   鈿狅笍 鍘嬬缉鍑洪敊: {e}")
        try: return open(image_path, "rb").read()
        except: return None

def encode_image_to_base64(image_path):
    if not image_path or not os.path.exists(image_path): return None
    img_bytes = compress_image_to_bytes(image_path)
    return base64.b64encode(img_bytes).decode('utf-8') if img_bytes else None

def preprocess_student_image_to_base64(image_path, mode="contrast"):
    """Return an enhanced student answer image for visual retry extraction."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        if mode == "contrast":
            gray = ImageOps.grayscale(img)
            gray = ImageOps.autocontrast(gray, cutoff=3)
            img = gray.point(lambda x: 0 if x < 115 else 255).convert("RGB")
        elif mode == "upscale":
            img = img.resize((int(img.width * 2), int(img.height * 2)), Image.Resampling.LANCZOS)
        elif mode == "sharpen":
            img = img.resize((int(img.width * 2), int(img.height * 2)), Image.Resampling.LANCZOS)
            img = ImageEnhance.Sharpness(img).enhance(2.0)
            img = ImageEnhance.Contrast(img).enhance(1.35)
        else:
            return encode_image_to_base64(image_path)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"   [image preprocess] {mode} failed; fallback to original image: {e}")
        return encode_image_to_base64(image_path)

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
            print(f"鉂?JSON 瑙ｆ瀽澶辫触: {e}")
            return None

# ==================== 鎻愬彇鍚庨鐩弬鏁板畧鍗?====================

def validate_extraction_against_question(question_text, extracted_facts_str):
    """Remove extracted values that are likely copied from printed question text."""
    prompt = f"""
You audit whether extracted facts were copied from the printed question text.

Question text:
{question_text}

Extracted facts JSON:
{extracted_facts_str}

Rules:
- Mark an item as "suspicious" only when the extracted value appears verbatim
  in the printed question and looks like copied question text rather than a
  student's handwritten computation or answer.
- Mark "ok" for handwritten formulas that legally reuse question parameters.
- Mark "ok" for student-specific computed results.
- Mark "ok" for blank/unreadable values.

Return strict JSON where keys are item ids and values are "ok" or "suspicious":
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
            print(f"   [parameter guard] replaced {changed} suspicious copied values with 未书写")
            return json.dumps(facts_dict, ensure_ascii=False)
        return extracted_facts_str
    except Exception as e:
        print(f"   [parameter guard] skipped due to error: {e}")
        return extracted_facts_str

# ==================== 鏍稿績涓氬姟閫昏緫 ====================

def generate_blind_checklist(rubrics_json_str):
    prompt = render_prompt_template(
        "blind_checklist.md",
        {"RUBRICS_JSON": rubrics_json_str},
        fallback=(
            "You are a visual extraction instruction generator. For each rubric item "
            "produce one precise instruction asking for the student's concrete written "
            "answer. Never put the correct answer into the instruction. Output a strict "
            "JSON array with ids matching the rubric items. Rubric: " + rubrics_json_str
        ),
    )
    for attempt in range(3):
        try:
            raw = call_text_model([{"role": "user", "content": prompt}], temperature=0.3, timeout=240)
            parsed = extract_and_parse_json(raw)
            if parsed and isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False)
        except Exception as e:
            time.sleep(3)
    return json.dumps([
        {"id": str(i + 1), "instruction": f"Extract the student's concrete answer content for rubric item {i + 1}."}
        for i in range(15)
    ], ensure_ascii=False)

def stage1_blind_extraction(question_text, student_img_path, blind_checklist, q_img_path=None):
    blind_prompt = render_prompt_template(
        "stage1_extraction.md",
        {"QUESTION_TEXT": question_text, "BLIND_CHECKLIST": blind_checklist},
        fallback=(
            "# Visual OCR extraction engine\n"
            "Transcribe verbatim ONLY the student's handwritten answer region. "
            "Do NOT copy printed question text. For each checklist item output "
            "the concrete value, or BLANK if truly blank.\n"
            "Question:\n" + question_text + "\nChecklist:\n" + blind_checklist
        ),
    )
    content_list = [{"type": "text", "text": blind_prompt}]
    if q_img_path and os.path.exists(q_img_path):
        q_b64 = encode_image_to_base64(q_img_path)
        content_list.extend([{"type": "text", "text": "附图"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{q_b64}"}}])
    student_b64 = encode_image_to_base64(student_img_path)
    content_list.extend([{"type": "text", "text": "考卷"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{student_b64}"}}])
    
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

def stage1_targeted_reextraction(question_text, student_img_path, blind_checklist, initial_facts_str, q_img_path=None, rubrics_data=None):
    """Risk-triggered second-pass extraction with multi-view image retry.

    The retry is generic: it is triggered by blank, unreadable, low-quality, or
    structurally incomplete facts. It does not depend on a question id.
    """
    facts_dict = json.loads(initial_facts_str) if isinstance(initial_facts_str, str) else initial_facts_str
    if not isinstance(facts_dict, dict):
        return initial_facts_str

    rubrics_data = rubrics_data if isinstance(rubrics_data, list) else []
    extraction_counts = compute_extraction_quality_counts(facts_dict, rubrics_data)
    extraction_risk_features = compute_extraction_risk_features(extraction_counts)
    suspicious_items = extraction_counts.get("suspicious_items", [])
    suspicious_ids = {str(item.get("id", "")) for item in suspicious_items if str(item.get("id", ""))}
    if not suspicious_ids:
        return initial_facts_str

    total_items = max(extraction_counts.get("total_items", 0), len(facts_dict), 1)
    suspicious_rate = len(suspicious_ids) / total_items
    force_visual_retry = (
        extraction_risk_features.get("blank_rate", 0.0) >= 0.50
        or extraction_risk_features.get("perception_failure_rate", 0.0) >= 0.20
        or extraction_risk_features.get("structure_missing_rate", 0.0) >= 0.20
    )
    if len(suspicious_ids) < 2 and suspicious_rate < 0.25 and not force_visual_retry:
        return initial_facts_str

    checklist_items = json.loads(blind_checklist) if isinstance(blind_checklist, str) else blind_checklist
    focused_instructions = []
    reason_map = {
        str(item.get("id", "")): item.get("reason", "suspicious")
        for item in suspicious_items
        if str(item.get("id", ""))
    }
    if isinstance(checklist_items, list):
        for item in checklist_items:
            item_id = str(item.get("id", ""))
            if item_id in suspicious_ids:
                enriched = dict(item)
                enriched["first_pass_value"] = facts_dict.get(item_id, "")
                enriched["suspicious_reason"] = reason_map.get(item_id, "suspicious")
                focused_instructions.append(enriched)
    if not focused_instructions:
        focused_instructions = [
            {
                "id": item_id,
                "instruction": "Re-check this item in the student's handwritten answer area.",
                "first_pass_value": facts_dict.get(item_id, ""),
                "suspicious_reason": reason_map.get(item_id, "suspicious"),
            }
            for item_id in sorted(suspicious_ids)
        ]

    already_extracted = {k: v for k, v in facts_dict.items() if str(k) not in suspicious_ids}
    reextraction_prompt = f"""
# Role: second-pass visual extraction engine
Only inspect the student's handwritten answer region. Do not copy printed
question text. The first extraction produced suspicious values for the items
below. Re-check the image carefully and return concrete handwritten content
when it exists.

Question context:
{question_text}

Suspicious checklist items:
{json.dumps(focused_instructions, ensure_ascii=False)}

Already extracted context:
{json.dumps(already_extracted, ensure_ascii=False)}

Rules:
1. Preserve exact handwritten numeric strings, bit vectors, sequence labels,
   formulas, arrows, units, and base suffixes.
2. Preserve leading zeros in binary or bit-vector answers.
3. If there are handwriting traces but they are unreadable, output 字迹模糊.
4. If the item area is truly blank, output 未书写.
5. Return strict JSON where keys are item ids and values are extracted strings.
"""

    base_content = [{"type": "text", "text": reextraction_prompt}]
    if q_img_path and os.path.exists(q_img_path):
        q_b64 = encode_image_to_base64(q_img_path)
        if q_b64:
            base_content.extend([
                {"type": "text", "text": "\nQuestion figure:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{q_b64}"}},
            ])

    retry_modes = ["original"]
    if force_visual_retry or suspicious_rate >= 0.25 or len(suspicious_ids) >= 2:
        retry_modes.extend(["contrast", "upscale", "sharpen"])

    recovered_by = {}
    for mode in retry_modes:
        student_b64 = (
            encode_image_to_base64(student_img_path)
            if mode == "original"
            else preprocess_student_image_to_base64(student_img_path, mode=mode)
        )
        if not student_b64:
            continue
        content = list(base_content)
        content.extend([
            {"type": "text", "text": f"\nStudent answer image view: {mode}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{student_b64}"}},
        ])
        for _ in range(2):
            try:
                fresh_client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)
                res = fresh_client.chat.completions.create(
                    model=VLM_MODEL_NAME,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.1,
                    timeout=180,
                )
                time.sleep(2)
                parsed = extract_and_parse_json(res.choices[0].message.content.strip())
                if not isinstance(parsed, dict):
                    break
                for raw_id, raw_value in parsed.items():
                    item_id = str(raw_id)
                    if item_id not in suspicious_ids or item_id not in facts_dict:
                        continue
                    rubric_item = next(
                        (item for item in rubrics_data if str(item.get("id", "")) == item_id),
                        None,
                    )
                    new_val = str(raw_value).strip()
                    usable_value = (
                        new_val
                        and not is_blank_extraction(new_val)
                        and not is_perception_failure(new_val)
                        and not is_low_quality_extraction(new_val, rubric_item)
                        and not is_structure_missing_extraction(new_val, rubric_item)
                    )
                    if usable_value:
                        facts_dict[item_id] = new_val
                        recovered_by[item_id] = mode
                break
            except Exception:
                time.sleep(10)

    if recovered_by:
        facts_dict["_extraction_recovered_by"] = recovered_by
        print(f"   [targeted re-extraction] recovered {len(recovered_by)}/{len(suspicious_ids)} suspicious items via {sorted(set(recovered_by.values()))}")
        return validate_extraction_against_question(question_text, json.dumps(facts_dict, ensure_ascii=False))
    return initial_facts_str

def stage2_logic_grading(student_facts_str, rubrics_json_str, temperature=0.35):
    """
    Stage 2锛氳涔夊尮閰嶅垽鍒嗐€傞€氳繃鍐呭璇箟鍖归厤瑙勫垯鍒ゆ柇瀛︾敓浜嬪疄涓庢爣鍑嗘槸鍚︾瓑浠枫€?
    """
    logic_prompt = f"""
    浣犳槸涓€涓瀬鍏朵弗璋ㄧ殑璁＄畻鏈虹瀛﹂槄鍗疯鍒ゃ€備綘鐨勮亴璐ｆ槸鍒ゆ柇瀛︾敓鐨勪綔绛斿湪璇箟涓婃槸鍚︿笌璇勫垎鏍囧噯鍖归厤锛岃€岄潪杩涜琛ㄩ潰鐨勫瓧绗︿覆瀵规瘮銆?
    浣犻渶瑕佷弗鏍兼牴鎹€愬瑙備簨瀹炴枃鏈€戯紝瀵圭収銆愮粏绮掑害璇勫垎鏍囧噯銆戯紝璁＄畻鎬诲垎銆?
    
    銆愬瑙備簨瀹炴枃鏈€? {student_facts_str}
    銆愮粏绮掑害鏍囧噯銆? {rubrics_json_str}
    
    馃毃銆愭渶楂樺垽鍐崇孩绾裤€?
    1. 鍙兘渚濋潬銆愬瑙備簨瀹炴枃鏈€戝垽鍐筹紒鍙璇ョ鐗囧寲姝ラ鏍囦负鈥濇湭涔﹀啓鈥濇垨鈥濆瓧杩规ā绯娾€濓紝缁濆鏃犳儏鎵ｉ櫎瀵瑰簲鍒嗘暟锛?
    2. 涓ョ鍔ㄧ敤鍚岀悊蹇冿紒涓ョ鍩轰簬鏈€缁堢粨鏋滄纭€屽幓鑴戣ˉ瀛︾敓鎳備簡锛佹病鍐欏氨鏄病鍐欙紒
    3. 馃毃銆愪俊鎭笉瓒虫嫤鎴€戯細褰撳瑙備簨瀹炵殑鍊间负鈥濇槸鈥濄€佲€濇湁鈥濄€佲€濆凡涔﹀啓鈥濄€佲€濆瓨鍦ㄢ€濄€佲€濇湁涔﹀啓鈥濄€佲€濆鈥濄€佲€濇纭€濄€佲€濇湁鎻愬彇鏍囨敞鈥濄€佲€濇湁璁＄畻杩囩▼鈥濄€佲€濇湁鏍囨敞鈥濈瓑闈炲叿浣撳唴瀹规椂锛岃鏉＄洰蹇呴』鍒?0 鍒嗭紝reason 濉啓鈥濇彁鍙栦俊鎭笉瓒筹紝鏃犳硶鍒ゅ畾鈥濄€?
    4. 馃毃銆愪笅娓哥粨鏋滃洖婧鍒欍€戯細璇勫垎椤逛箣闂村線寰€瀛樺湪鎺ㄥ渚濊禆鍏崇郴鈥斺€斾笂娓搁」锛堝鍙傛暟璇嗗埆銆佷腑闂磋绠楋級鏄笅娓搁」锛堝鏈€缁堢粨鏋溿€佺患鍚堢粨璁猴級鐨勫繀瑕佽緭鍏ャ€傚綋鏌愪釜涓婃父椤圭殑瀹㈣浜嬪疄涓?鏈功鍐?鏃讹紝涓嶈兘鏈烘鍦板垽 BLANK锛岃€屽簲妫€鏌ュ叾涓嬫父渚濊禆椤癸細
    鈶?濡傛灉瀛樺湪鑷冲皯涓€涓笅娓搁」鐨勭粨鏋滄暟鍊兼纭紙鍦ㄥ宸寖鍥村唴锛夛紝涓旇涓嬫父椤圭殑姝ｇ‘缁撴灉鍦ㄦ暟瀛?閫昏緫涓婂繀鐒朵緷璧栦簬杩欎釜"鏈功鍐?椤规墍瑕佹眰鐨勫弬鏁版垨涓棿姝ラ锛屽垯璇?鏈功鍐?椤瑰簲鍒や负 MATCH 骞剁粰婊″垎鈥斺€旀纭殑涓嬫父缁撴灉鏈韩灏辨槸瀛︾敓鎺屾彙浜嗕笂娓稿弬鏁扮殑閾佽瘉銆?
    鈶?鍙湁褰撴墍鏈変緷璧栬鍙傛暟鐨勪笅娓搁」缁撴灉涔熷叏閮ㄩ敊璇椂锛?鏈功鍐?鎵嶇淮鎸?BLANK 鍒ゅ喅銆?
    鈶?reason 涓簲娉ㄦ槑"涓嬫父椤规纭紝鍥炴函鎺ㄦ柇璇ュ弬鏁板凡姝ｇ‘浣跨敤"銆?

    5. 馃毃銆愬唴瀹硅涔夊尮閰嶈鍒欍€戯細鍦ㄥ垽鏂€濆尮閰?涓嶅尮閰嶁€濇椂锛屽繀椤婚€忚繃鏍煎紡宸紓璇嗗埆璇箟绛変环銆傚叿浣撳師鍒欙細
    6. 馃毃銆愭牳蹇冭兘鍔涢敋瀹氳鍒欍€戯細褰撹瘎鍒嗘潯鐩憟鐜扳€濆弬鏁拌瘑鍒啋鍏紡/鏂规硶鈫掓渶缁堢粨鏋溾€濈殑灞傛渚濊禆鍏崇郴鏃讹紝濡傛灉瀛︾敓鐨勬牳蹇冩帹瀵奸」锛堝叕寮?鏂规硶鏉＄洰锛夊拰鏈€缁堢粨鏋滄潯鐩?*鍏ㄩ儴**琚垽涓?SEMANTIC_FATAL 鎴?BLANK锛岃鏄庡鐢熶粎瀹屾垚浜嗕綆璁ょ煡璐熻嵎鐨勫弬鏁版妱褰曪紝鏈睍鐜板鏍稿績鐭ヨ瘑鐐圭殑鎺屾彙銆傛鏃讹細
       - 鎵€鏈夌函鍙傛暟璇嗗埆绫绘潯鐩紙浠呴渶浠庨骞茬洿鎺ヨ鍙栥€佷笉娑夊強璁＄畻鎴栧垎鏋愮殑鏉＄洰锛夌殑寰楀垎涓婇檺闄嶄负璇ユ潯鐩弧鍒嗙殑 30%銆?
       - reason 涓敞鏄庘€濇牳蹇冩帹瀵煎叏閿欙紝鍙傛暟璇嗗埆鍒嗛檷绾р€濄€?
       - 濡傛灉鏍稿績鎺ㄥ椤逛腑鑷冲皯鏈変竴椤硅鍒や负 MATCH 鎴?PARTIAL_MATCH锛屽垯涓嶈Е鍙戞闄嶇骇銆?
       (a) 鏁板€肩被锛氬彧姣旇緝鏍稿績鏁板€硷紝蹇界暐鍗曚綅鏍煎紡宸紓锛堝鈥?3浣嶁€?鈥?3 浣嶁€?鈥?3bit鈥?鈥?3鈥濓級銆傚拷鐣ヨ〃杈句腑鐨勭┖鏍笺€佹爣鐐广€佸悗缂€绗﹀彿銆?
           馃毃銆愭暟鍊煎宸鍒欍€戯細褰撹瘎鍒嗛」鐨勬爣鍑嗙瓟妗堟槸涓€涓暟鍊兼椂锛屽鏋滃鐢熺殑鏁板€间笌鏍囧噯绛旀鐨勭浉瀵硅宸?鈮?10%锛屽簲鍒や负 MATCH锛堟弧鍒嗭級鑰岄潪 SEMANTIC_FATAL銆?
           鍒ゆ柇鏂规硶锛氭彁鍙栧鐢熺瓟妗堝拰鏍囧噯绛旀涓殑绾暟鍊奸儴鍒嗭紝璁＄畻 |瀛︾敓鍊?- 鏍囧噯鍊紎 / 鏍囧噯鍊笺€備緥濡傛爣鍑?156锛屽鐢?162锛岃宸?3.8% 鈮?10%锛屽垽 MATCH銆?
           鍗曚綅鎹㈢畻锛氶亣鍒扳€滽鈥?鈥滿鈥濈瓑鍗曚綅缂╁啓鏃讹紝鍏堟崲绠椾负缁熶竴鍗曚綅鍐嶆瘮杈冿紙濡?84Kb = 86016浣嶏級銆傚鏋滄崲绠楀悗鏁板€煎尮閰嶆垨璇樊 鈮?10%锛屽垽 MATCH銆?
           娉ㄦ剰锛氬彧鏈夊綋鏍囧噯绛旀鏄槑纭殑鍗曚竴鏁板€兼椂鎵嶉€傜敤姝よ鍒欍€?
           馃毃銆愰摼寮忔帹瀵间竴鑷存€ц鍒欍€戯細褰撹瘎鍒嗛」鐨勬暟鍊煎彲鐢卞叾浠栬瘎鍒嗛」鎺ㄥ寰楀嚭鏃讹紝楠岃瘉姝ラ涓猴細鈶犲厛浠庢爣鍑嗙瓟妗堟帹鏂纭殑鎺ㄥ鍏紡鍙婂父鏁帮紙濡傛爣鍑嗘帶瀛樺閲?6016=鏍囧噯寰寚浠ら暱搴?68脳512锛屽垯鍏紡涓哄井鎸囦护闀垮害脳512锛夛紱鈶＄敤鐩稿悓鍏紡鍜屽父鏁颁綔鐢ㄤ簬瀛︾敓鐨勪笂娓搁」锛堝瀛︾敓寰寚浠?3脳512=16896锛夛紱鈶㈡瘮杈冭绠楃粨鏋滀笌瀛︾敓鐨勬帹瀵奸」锛?6896鈮?940鈫掍笉涓€鑷粹啋SEMANTIC_FATAL锛夈€傚彧鏈夊綋瀛︾敓鐨勬帹瀵奸」 = 姝ｇ‘鍏紡(瀛︾敓涓婃父椤? 鏃舵墠鍒?MATCH銆傜姝㈤€氳繃鎵惧埌涓€涓兘鍑戝嚭瀛︾敓绛旀鐨勪换鎰忓叕寮忔潵鍒ゅ畾涓€鑷淬€?
           馃毃銆愰敊璇捣鐐归摼寮忔帹瀵兼仮澶嶈鍒欍€戯細褰撹瘎鍒嗛」涔嬮棿瀛樺湪鏄庣‘鐨勬暟瀛︽帹瀵间緷璧栧叧绯绘椂锛堝 item_1鈫抜tem_2鈫抜tem_3 褰㈡垚璁＄畻閾撅級锛屽鏋滃鐢熺殑璧峰椤瑰€奸敊璇紙琚垽 SEMANTIC_FATAL锛夛紝浣嗗叾涓嬫父椤圭殑鍊艰兘澶熼€氳繃璇ラ敊璇捣濮嬪€间娇鐢ㄦ纭殑鍏紡鍜屾帹瀵兼楠よ绠楀緱鍑猴紙鍗冲鐢熶娇鐢ㄤ簡姝ｇ‘鐨勬柟娉曪紝鍙槸璧风偣涓嶅悓锛夛紝鍒欙細璧峰椤圭淮鎸?SEMANTIC_FATAL锛?鍒嗭級锛屼笅娓告帹瀵奸」鏀逛负 PARTIAL_MATCH 骞剁粰浜堣鏉＄洰 50% 鐨勫垎鏁般€傞獙璇佹柟娉曪細灏嗗鐢熺殑涓婃父鍊间唬鍏ユ爣鍑嗗叕寮忥紝濡傛灉璁＄畻缁撴灉绛変簬瀛︾敓鐨勪笅娓哥瓟妗堬紝鍒欑‘璁ゆ帹瀵兼纭€傛瑙勫垯浠呴€傜敤浜庡垎鍊?鈮?2 鐨勬帹瀵肩被鏉＄洰锛屼笉閫傜敤浜庤瘑鍒?鎶勫綍绫绘潯鐩€俽eason 涓敞鏄庨摼寮忔帹瀵煎唴閮ㄤ竴鑷达細鏂规硶姝ｇ‘浣嗚捣濮嬪€奸敊璇€?
           馃毃銆愭暟閲忕骇鎰熺煡瑙勫垯銆戯細褰撳鐢熺殑鏁板€间笌鏍囧噯绛旀鐨勭浉瀵硅宸?> 10%锛屼絾婊¤冻浠ヤ笅鍏ㄩ儴鏉′欢鏃讹紝灏?error_category 浠?SEMANTIC_FATAL 闄嶇骇涓?PARTIAL_MATCH锛岀粰浜堣鏉＄洰婊″垎鐨?50%锛?
           鈶?瀛︾敓鐨勬暟鍊间笌鏍囧噯绛旀鐨勬瘮鍊兼伆濂戒负 10^n锛坣涓洪潪闆舵暣鏁帮紝濡?0.1鍊嶃€?0鍊嶃€?00鍊嶏級锛屾垨涓庡彟涓€璇勫垎椤圭殑姝ｇ‘鍊肩殑 10^n 鍊嶄竴鑷达紙璇存槑瀛︾敓娣锋穯浜嗕袱涓浉浼煎弬鏁帮級銆?
           鈶?璇ユ潯鐩瓨鍦ㄤ笂娓镐緷璧栭」锛屼笖鑷冲皯涓€涓笂娓镐緷璧栭」宸茶鍒や负 SEMANTIC_FATAL 鎴?PARTIAL_MATCH锛堣鏄庨敊璇槸浠庝笂娓镐紶鎾€屾潵锛屼笉鏄嚟绌轰骇鐢燂級銆?
           鈶?瀛︾敓鐨勭瓟妗堜笉鏄?0 鎴栨瀬绔紓甯稿€硷紙鎺掗櫎瀹屽叏涓嶄細鍋氱殑鎯呭喌锛夈€?
           reason 涓敞鏄?鏁伴噺绾у亸宸絾鏂规硶璁哄彲寰?銆?
           娉ㄦ剰锛氭瑙勫垯涓嶄笌 10% 瀹瑰樊瑙勫垯鍙犲姞銆傚厛妫€鏌?10% 瀹瑰樊锛屼笉婊¤冻鏃跺啀妫€鏌ユ瑙勫垯銆?
       (b) 搴忓垪绫伙紙浜岃繘鍒?鍗佸叚杩涘埗/鐭╅樀绛夛級锛氬幓闄ゆ墍鏈夌┖鏍笺€佸垎闅旂銆佽繘鍒舵爣璁板悗锛屾瘮杈冪函瀛楃搴忓垪鏄惁涓€鑷淬€?
           搴忓垪绫诲瀹癸細濡傛灉瀛︾敓搴忓垪涓庢爣鍑嗗簭鍒楅暱搴︿竴鑷达紝涓斿樊寮備綅鏁?鈮?鎬讳綅鏁扮殑 10%锛堝嵆 24 浣嶅簭鍒楀厑璁?2-3 浣嶉敊璇級锛屽簲鍒や负 FORMAT_MINOR 鑰岄潪 SEMANTIC_FATAL銆傚彧鏈夊綋搴忓垪闀垮害涓嶄竴鑷存垨閿欒浣嶆暟瓒呰繃 10% 鏃舵墠鍒?SEMANTIC_FATAL銆?
       (c) 杩囩▼绫伙細涓嶈姹備笌鏍囧噯鎺緸涓€鑷淬€傚彧瑕佸鐢熺殑鎻忚堪涓寘鍚簡鏍囧噯鎵€瑕佹眰鐨勫叧閿涔夎绱狅紙鍗筹細鎿嶄綔浜嗕粈涔堝璞°€佽繘琛屼簡浠€涔堣繍绠?姣旇緝銆佸緱鍑轰簡浠€涔堜腑闂寸粨鏋滐級锛屽嵆鍒や负鍖归厤銆傚厑璁歌〃杩伴『搴忎笉鍚屻€佽鐣ヤ笉鍚屻€?
       (d) 缁撹绫伙細鎺ュ彈鍚屼箟琛ㄨ揪锛堝鈥濅笉鑳解€濃€濇棤娉曗€濃€濅笉鍙互鈥濆潎瑙嗕负绛変环锛夈€?
       (e) 鍞竴鎷掑垽鏉′欢锛氬鐢熺殑浜嬪疄鍐呭鍦ㄨ涔変笂纭疄涓庢爣鍑嗙煕鐩撅紙鏁板€间笉鍚屻€佺粨璁虹浉鍙嶏級锛屾墠鍙垽涓嶅尮閰嶃€傛牸寮忓樊寮傜粷涓嶈兘浣滀负鎷掑垽鐞嗙敱銆?
       (f) 鏍煎紡缁撴瀯鎻忚堪绫伙細濡傛灉瀛︾敓鐢ㄥ瓧娈靛悕绉般€佷綅鑼冨洿銆佸垎鍖烘弿杩扮瓑鏂瑰紡琛ㄨ揪浜嗕笌鏍囧噯鐩稿悓鐨勭粨鏋勬鏋讹紙瀛楁鐨勫惈涔夊拰椤哄簭涓€鑷达級锛屽嵆浣挎湭鍐欏嚭鍏蜂綋鐨勪綅鏁版暟鍊硷紝涔熷簲瑙嗕负璇箟鍖归厤銆傚彧鏈夊綋瀛︾敓鎻忚堪鐨勭粨鏋勬湰韬敊璇紙瀛楁缂哄け銆侀『搴忛鍊掋€佸惈涔変笉绗︼級鏃舵墠鍒や笉鍖归厤銆?
       (g) 姣斾緥缁欏垎瑙勫垯锛氬綋璇勫垎椤瑰垎鍊?鈮?3 鍒嗕笖鍖呭惈澶氫釜鍙嫭绔嬮獙璇佺殑璇勫垎瑕佺礌鏃讹紝濡傛灉瀛︾敓鐨勪綔绛斿尮閰嶄簡閮ㄥ垎瑕佺礌浣嗛潪鍏ㄩ儴锛屽簲浣跨敤 PARTIAL_MATCH 骞剁粰浜堟瘮渚嬪垎鏁般€傜姝㈠澶氳绱犻」鐩娇鐢ㄥ叏鏈夊叏鏃犵殑 0/婊″垎浜屽垎娉曘€?
    
    蹇呴』杈撳嚭绾?JSON锛屾瘡鏉?detail 蹇呴』鍖呭惈 error_category 瀛楁銆?
    {{
        "details": [
            {{"id": "1", "score_given": 0, "error_category": "BLANK", "reason": "鏈功鍐?}},
            {{"id": "2", "score_given": 2, "error_category": "MATCH", "reason": "鍖归厤鎴愬姛"}},
            {{"id": "3", "score_given": 0, "error_category": "SEMANTIC_FATAL", "reason": "鏁板€肩煕鐩?}},
            {{"id": "4", "score_given": 1, "error_category": "FORMAT_MINOR", "reason": "鍗曚綅涔﹀啓涓嶈鑼冿紙濡傞噸澶嶆爣娉℅Hz锛夛紝鏍稿績鏁板€兼纭紝70%缁欏垎"}},
            {{"id": "5", "score_given": 0, "error_category": "INSUFFICIENT_INFO", "reason": "鎻愬彇淇℃伅涓嶈冻"}},
            {{"id": "6", "score_given": 3, "error_category": "PARTIAL_MATCH", "reason": "閮ㄥ垎鍖归厤锛氱瓟瀵?/3涓绱?}}
        ],
        "total_score": 2
    }}

    馃毃銆恊rror_category 鏋氫妇瀹氫箟銆戯紙蹇呴』涓ユ牸浠庝互涓?绉嶄腑閫夋嫨涓€涓級锛?
    - "MATCH"锛氳鏉＄洰寰楀垎 = 婊″垎锛堣涔夊尮閰嶆垚鍔燂級
    - "BLANK"锛氬鐢熸湭涔﹀啓鎴栧瓧杩规ā绯婏紙score_given 蹇呴』涓?0锛?
    - "SEMANTIC_FATAL"锛氭牳蹇冪煡璇嗛敊璇€佺粨璁虹浉鍙嶃€佹暟鍊肩煕鐩撅紙score_given 蹇呴』涓?0锛?
    - "FORMAT_MINOR"锛氭牸寮忎笉绗︺€佺己灏戝崟浣嶃€佸悓涔夎〃杈炬湭瀵归綈绛夐潪瀹炶川鎬ч敊璇€傚鏋滆鏉＄洰鐨勬牳蹇冩暟鍊?缁撹姝ｇ‘锛屼粎鏍煎紡鏈夌憰鐤碉紝缁欎簣璇ユ潯鐩弧鍒嗙殑 70%锛堝悜涓婂彇鏁达紝鏈€灏?1 鍒嗭級銆傚鏋滆鏉＄洰鐨勬牳蹇冩暟鍊?缁撹涔熼敊璇紝鍒?score_given 涓?0銆俽eason 涓敞鏄庢牸寮忓樊寮傜殑鍏蜂綋鍐呭銆?
    - "INSUFFICIENT_INFO"锛氭彁鍙栦俊鎭笉瓒筹紝鏃犳硶鍒ゅ畾锛坰core_given 蹇呴』涓?0锛?
    - "PARTIAL_MATCH"锛氳鏉＄洰閮ㄥ垎鍖归厤锛屽鐢熷畬鎴愪簡閮ㄥ垎璇勫垎瑕佺礌浣嗛潪鍏ㄩ儴銆俿core_given 涓烘寜瀹屾垚姣斾緥璁＄畻鐨勯儴鍒嗗垎鏁帮紙鈮? 涓?< 婊″垎锛夈€備緥濡傛弧鍒?5 鍒嗗惈 3 涓绱狅紝绛斿 2 涓粰 3 鍒嗐€傚彧鏈夊畬鍏ㄦ湭娑夊強浠讳綍瑕佺礌鏃舵墠鐢?BLANK 鎴?SEMANTIC_FATAL銆?
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
    瀹藉澶嶆煡瀵煎笀锛氫互鏁欏笀瀹芥澗闃呭嵎鐨勮瑙掑鍒濆鎵ｅ垎杩涜浜屾瀹℃煡
    """
    leniency_prompt = f"""
# Role: 楂樻牎闃呭嵎缁勯暱锛堟暀甯堣瑙掑鏌ワ級
浣犳槸涓€浣嶇粡楠屼赴瀵岀殑楂樻牎鏁欏笀锛屾鍦ㄥ鏈哄櫒鍒濆鐨勭粨鏋滆繘琛屽鏌ャ€?
鐪熷疄鏁欏笀鍦ㄦ壒鏀规椂寰€寰€姣旇緝瀹芥澗锛氬彧瑕佸鐢熷睍鐜颁簡瀵规牳蹇冪煡璇嗙偣鐨勭悊瑙ｏ紝鍗充娇琛ㄨ堪涓嶅畬鍏ㄨ鑼冦€佷腑闂存楠ゆ湁鐪佺暐锛屾暀甯堥€氬父涔熶細缁欏垎銆?

銆愯瘎鍒嗘爣鍑嗐€? {rubrics_json_str}
銆愬鐢熷瑙備簨瀹炪€? {student_facts_str}
銆愬垵瀹℃墸鍒嗚褰曘€? {strict_cot_str}

# 澶嶆煡鍘熷垯锛堟暀甯堝鏉捐瑙掞級
瀵瑰垵瀹′腑姣忎釜 score_given 涓?0 鐨勬潯鐩紝鐢ㄤ互涓嬫爣鍑嗛€愪竴澶嶆煡锛?
1. **缁撹姝ｇ‘鍗崇粰鍒?*锛氬鏋滃鐢熷啓鍑轰簡姝ｇ‘鐨勬渶缁堢粨璁猴紙濡傗€濅笉鑳借闂埌鈥濃€濊兘璁块棶鍒扳€濓級锛屽嵆浣挎病鏈夊畬鏁村睍绀烘帹瀵艰繃绋嬶紝鏁欏笀閫氬父浼氱粰浜堝ぇ閮ㄥ垎鍒嗘暟銆?
2. **瀹炶川鐞嗚В浼樺厛**锛氬鏋滃鐢熺殑琛ㄨ堪铏界劧涓庢爣鍑嗙瓟妗堟帾杈炰笉鍚岋紝浣嗗睍鐜颁簡瀵圭煡璇嗙偣鐨勫疄璐ㄦ€х悊瑙ｏ紙濡傛纭瘑鍒簡鏍囪鍖归厤鍏崇郴锛夛紝搴旀仮澶嶅垎鏁般€?
3. **杩囩▼鐪佺暐瀹藉**锛氫腑闂存楠ょ渷鐣ヤ絾缁撹姝ｇ‘锛屾暀甯堜竴鑸彧鎵ｅ皯閲忓垎鐢氳嚦涓嶆墸銆傚彧鏈夊綋瀛︾敓鐨勭瓟妗堟槑鏄鹃敊璇紙鏁板€肩畻閿欍€佹蹇垫悶鍙嶏級鏃舵墠缁存寔鎵ｅ垎銆?
4. **涓嶆仮澶嶇殑鎯呭喌**锛氬鐢熺‘瀹炲湪鏍稿績缁撹涓婂畬鍏ㄩ敊璇紙姒傚康鎼炲弽銆佹柟娉曢敊璇級锛岀淮鎸?0 鍒嗐€?
5. **鍙傛暟鏈崟鐙垪鍑虹殑瀹藉**锛氬鏋滄煇涓潯鐩洜"鏈功鍐?琚墸鍒嗭紝浣嗚鏉＄洰鎵€瑕佹眰鐨勫弬鏁?涓棿鍊煎湪閫昏緫涓婃槸鍙︿竴涓凡姝ｇ‘浣滅瓟鏉＄洰鐨勫繀瑕佽緭鍏ワ紙渚嬪姝ｇ‘绠楀嚭浜嗘渶缁堢粨鏋滐紝璇存槑瀛︾敓蹇呯劧姝ｇ‘浣跨敤浜嗗叕寮忎腑鐨勫弬鏁帮級锛屾暀甯堥€氬父浼氭仮澶嶅垎鏁般€傚彧鏈夊綋渚濊禆璇ュ弬鏁扮殑涓嬫父缁撴灉涔熼敊璇椂锛屾墠缁存寔鎵ｅ垎銆?
6. **姣斾緥鎭㈠鍘熷垯**锛氬浜庡垎鍊?鈮?3 鍒嗕笖鍚涓彲鐙珛璇勫垎瑕佺礌鐨勬潯鐩紝濡傛灉瀛︾敓琚垽 0 鍒嗕絾瀹為檯姝ｇ‘瀹屾垚浜嗛儴鍒嗚绱狅紙濡傝绠楄繃绋嬫纭絾鏈€缁堢粨鏋滈敊璇級锛屽簲鎭㈠姣斾緥鍒嗘暟锛堢害 瀹屾垚瑕佺礌鍗犳瘮 脳 婊″垎锛夛紝鑰岄潪鍏ㄦ湁鎴栧叏鏃犮€?
7. **鏁板€肩被璇勫垎椤瑰垎灞傚瀹瑰師鍒?*锛氫粎閫傜敤浜庢爣鍑嗙瓟妗堝寘鍚槑纭暟鍊肩殑璇勫垎椤癸紙濡?23浣?銆?140鏉?銆?86016浣?锛夈€傚浜庢弿杩版€?姒傚康鎬ц瘎鍒嗛」锛堟爣鍑嗙瓟妗堜负鏂囧瓧鎻忚堪銆佸叕寮忋€佺粨璁猴級锛屾湰鍘熷垯涓嶉€傜敤锛屼粛鎸夊師鍒?1-5 鍒ゆ柇銆傞€傜敤鏃舵寜浠ヤ笅鏍囧噯鎭㈠锛?
   - 鐩稿璇樊 鈮?15%锛氭仮澶嶆弧鍒嗭紙璁＄畻杩囩▼涓殑鍚堢悊璇樊锛屽杩涗綅銆佽繎浼硷級
   - 鐩稿璇樊 15%-50%锛氬鏋滃鐢熺粰鍑轰簡闈炵┖鐧界殑鍏蜂綋鏁板€肩瓟妗堜笖璇ユ暟鍊煎湪鍚堢悊鏁伴噺绾у唴锛堥潪鏋佺寮傚父鍊煎 0銆?銆?99锛夛紝鎭㈠ 鈮?0% 鍒嗘暟锛堣繃绋嬪垎鈥斺€斿鐢熻繘琛屼簡璁＄畻浣嗙粨鏋滃亸宸緝澶э級
   - 鐩稿璇樊 > 50%锛氱淮鎸?0 鍒嗭紝闄ら潪閫傜敤鍘熷垯 7锛堥摼寮忎竴鑷存€э級鎴栧鐢熷睍鐜颁簡瀵硅绠楁柟娉曠殑鐞嗚В锛堝姝ｇ‘浣跨敤浜嗗崟浣嶆崲绠椼€佺瀛﹁鏁版硶绛夛級
8. **閾惧紡鎺ㄥ鍐呴儴涓€鑷存€у師鍒?*锛氫粎閫傜敤浜庢暟鍊肩被璇勫垎椤广€傛煇浜涜瘎鍒嗛」鐨勫€煎彲鐢卞叾浠栬瘎鍒嗛」鐨勬暟鍊兼帹瀵煎緱鍑恒€傚鏌ユ椂楠岃瘉姝ラ锛氣憼浠庢爣鍑嗙瓟妗堟帹鏂纭殑鎺ㄥ鍏紡鍙婂父鏁帮紙濡傛爣鍑嗘帶瀛?6016=鏍囧噯168脳512锛屽叕寮忎负寰寚浠ら暱搴γ?12锛夛紱鈶＄敤鐩稿悓鍏紡浣滅敤浜庡鐢熺殑涓婃父椤癸紱鈶㈠鏋滃鐢熺殑鎺ㄥ椤圭瓑浜庤绠楃粨鏋滐紝璇存槑瀛︾敓鎺屾彙浜嗘纭殑璁＄畻鏂规硶锛屾仮澶嶆弧鍒嗐€傞獙璇佹椂蹇呴』浣跨敤姝ｇ‘鐨勫叕寮忓拰甯告暟锛堝脳512鑰岄潪脳180锛夛紝绂佹鐢ㄤ换鎰忓噾鍑虹殑鍏紡鏉ュ垽瀹氫竴鑷淬€?
9. **鏁翠綋璐ㄩ噺闂ㄦ帶鍘熷垯**锛氬湪搴旂敤鍘熷垯 7 鍜?8 涔嬪墠锛屽厛缁熻鍒濆涓鍒や负 SEMANTIC_FATAL 鐨勬潯鐩崰姣斻€傚鏋?SEMANTIC_FATAL 鍗犳瘮瓒呰繃 50%锛堝嵆瓒呰繃涓€鍗婄殑鏉＄洰瀛樺湪涓ラ噸閿欒锛夛紝璇存槑瀛︾敓瀵圭煡璇嗙偣鎺屾彙涓ラ噸涓嶈冻锛屾鏃跺師鍒?7 闄嶇骇涓轰粎鎭㈠璇樊 鈮?5% 鐨勬潯鐩紙15%-50% 鐨勮繃绋嬪垎涓嶅啀閫傜敤锛夛紝鍘熷垯 8锛堥摼寮忔帹瀵硷級涓嶅啀閫傜敤銆傚彧鏈?SEMANTIC_FATAL 鍗犳瘮 鈮?50% 鏃讹紝鍘熷垯 6 鍜?7 鎵嶅畬鏁撮€傜敤銆傚師鍒?5锛堟瘮渚嬫仮澶嶏級涓嶅彈姝ら檺鍒躲€?
10. **灏忓垎鍊兼潯鐩瀹瑰師鍒?*锛氬浜庡垎鍊?鈮?2 鍒嗙殑鏉＄洰锛屾暀甯堥€氬父閲囩敤"宸笉澶氬氨缁?鐨勫鏉炬爣鍑嗭細
   - 濡傛灉瀛︾敓鐨勭瓟妗堜笌鏍囧噯鍦ㄦ牳蹇冪粨璁轰笂涓€鑷达紙濡傚懡涓?鏈懡涓垽鏂纭€佸瓧娈靛悕绉板拰椤哄簭姝ｇ‘锛夛紝浣嗗叿浣撴暟鍊兼湁寰皬鍋忓樊锛堝浜岃繘鍒舵彁鍙栦腑涓埆浣嶉敊璇€佹暟鍊肩殑鏈€鍚庝竴浣嶄笉鍚岋級锛屽簲鎭㈠婊″垎銆?
   - 濡傛灉瀛︾敓鐨勭瓟妗堢粨鏋勬鏋舵纭絾鏁板€煎畬鍏ㄩ敊璇紝鎭㈠ 50% 鍒嗘暟锛堝睍鐜颁簡瀵规柟娉曠殑鐞嗚В锛夈€?
   - 鍙湁褰撳鐢熺殑缁撹瀹屽叏鐩稿弽锛堝鍛戒腑鍒ゆ垚鏈懡涓級鎴栧畬鍏ㄦ湭娑夊強璇ョ煡璇嗙偣鏃讹紝鎵嶇淮鎸?0 鍒嗐€?

馃毃銆愮‖绾︽潫銆戯細
- 蹇呴』瀵规瘡涓鎵ｅ垎鏉＄洰閫愪竴缁欏嚭澶嶆煡缁撹
- analysis_cot 涓畝瑕佸垪鍑烘瘡鏉＄殑澶嶆煡鍐崇瓥锛堟仮澶?缁存寔 + 涓€鍙ヨ瘽鐞嗙敱锛?

杈撳嚭绾?JSON锛?
{{
    鈥渟econdary_total_score鈥? 澶嶆煡鍚庣殑鎬诲垎(鏁板瓧),
    鈥渓eniency_reason鈥? 鈥滅畝杩颁富瑕佺籂姝ｉ」锛?0瀛楀唴鈥?
    鈥渁nalysis_cot鈥? 鈥滈€愭潯澶嶆煡杩囩▼鈥?
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
    """Prefer structured missed/over credit deltas; keep baseline without item evidence."""
    avg_model_score = _safe_float(avg_model_score, 0.0)
    max_score = max(_safe_float(max_score, 0.0), 1.0)
    missed = _sum_agent_item_points(parsed_agent.get("missed_credit_items"))
    over = _sum_agent_item_points(parsed_agent.get("over_credit_items"))
    if missed > 0 or over > 0:
        return _clamp(avg_model_score + missed - over, 0, max_score)
    return avg_model_score

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
    structure_missing_rate=0.0,
    suspicious_extraction_rate=0.0,
    extraction_risk=0.0,
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
        extraction_risk / 0.50 if 0.50 > 0 else 0.0,
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
        "structure_missing_rate": round(structure_missing_rate, 4),
        "suspicious_extraction_rate": round(suspicious_extraction_rate, 4),
        "extraction_risk": round(extraction_risk, 4),
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
# Role: 杈圭晫鏍锋湰鍙屽悜鏍″噯浠茶鍛?
浣犳鍦ㄥ鏍镐竴浠借嚜鍔ㄩ槄鍗风粨鏋溿€備綘鐨勭洰鏍囨槸浣胯瘎鍒嗗敖鍙兘鎺ヨ繎鐪熷疄鏁欏笀鐨勮瘎鍒嗐€?

銆愯瘎鍒嗘爣鍑嗐€?
{rubrics_json_str}

銆愬鐢熷瑙備綔绛斾簨瀹炪€?
{student_facts_str}

銆愪笁娆＄嫭绔嬭瘎鍒嗚褰曘€?
{json.dumps(strict_cots, ensure_ascii=False)}

銆愰闄╃壒寰併€?
{json.dumps(risk_profile, ensure_ascii=False)}

浠茶鍘熷垯锛?
1. 鍙屽悜鏍″噯锛氫綘闇€瑕佸悓鏃惰€冭檻妯″瀷鍙兘楂樹及鍜屼綆浼扮殑鎯呭喌銆?
   - 褰撳鐢熸湁姝ｇ‘鐨勮繃绋嬫帹瀵间絾琚垽 SEMANTIC_FATAL 鏃讹紝鑰冭檻涓婅皟銆?
   - 褰撳鐢熺殑鏍稿績缁撹瀹屽叏閿欒浣嗚鍒?MATCH 鏃讹紝鑰冭檻涓嬭皟銆?
2. 杩囩▼鍒嗘仮澶嶏細濡傛灉瀛︾敓鐨勮绠楁柟娉曡姝ｇ‘浣嗗垵濮嬪弬鏁版湁璇鑷寸粨鏋滃亸宸紝杩欐槸鍏稿瀷鐨勪綆浼板満鏅紝搴旈€傚綋涓婅皟銆?
3. 绌烘礊鍒嗙籂姝ｏ細濡傛灉瀛︾敓浠呮纭瘑鍒簡棰樺共鍙傛暟浣嗘牳蹇冩帹瀵煎拰缁撴灉鍏ㄩ敊锛屽ぇ閲忓弬鏁拌瘑鍒垎鍙兘鏄珮浼帮紝搴旈€傚綋涓嬭皟銆?
4. 鏍煎紡涓嶅簲閲嶇綒锛氭牸寮忛棶棰橈紙鍗曚綅涔﹀啓涔犳儻銆佸彉閲忓悕涓嶈鑼冿級涓嶅簲瀹炶川鎬у奖鍝嶅垎鏁般€?
5. 鏈€缁堝垎鏁板繀椤诲湪 [0, 棰樼洰婊″垎] 鑼冨洿鍐呫€?
6. 濡傛灉璇佹嵁涓嶈冻浠ョ‘瀹氭柟鍚戯紝淇濇寔鍘熷垎銆?

璇疯緭鍑虹函 JSON锛?
{{
  "decision": "raise 鎴?keep 鎴?cautious_lower",
  "calibrated_score": 鏁板瓧,
  "reason": "50瀛椾互鍐呰鏄?
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
    """Generate a short disagreement summary for NEG/manual-review samples."""
    if not strict_cots or len(strict_cots) < 2:
        return "Insufficient grading traces for disagreement summary."
    summaries = []
    for i, cot in enumerate(strict_cots):
        total = cot.get("total_score", "?")
        details_summary = "; ".join(
            f"item {d.get('id', '?')}={d.get('score_given', 0)} ({str(d.get('reason', ''))[:40]})"
            for d in cot.get("details", [])[:5]
        )
        summaries.append(f"judgement {i + 1}: total={total} | {details_summary}")

    prompt = f"""The following are independent grading traces for the same answer.
They may disagree substantially:
{chr(10).join(summaries)}

Summarize the core disagreement in one concise sentence. Output only that sentence."""
    for attempt in range(2):
        try:
            return call_text_model(
                [{"role": "user", "content": prompt}],
                temperature=0.1, timeout=30
            )
        except Exception:
            time.sleep(2)
    return "Summary generation failed; manual review is required."

# ==================== 鏍稿績绠＄嚎锛氶浂鏍锋湰鏃犵洃鐫?3WD ====================

def grade_student_3wd_pipeline(student_img_path, question_text, rubrics_json, teacher_score, q_img_path=None, blind_checklist=None):
    student_id = os.path.splitext(os.path.basename(student_img_path))[0]
    print(f"\n=============================================")
    print(f"Start grading student [{student_id}]")
    print(f"=============================================")

    print("  [Stage 1] visual fact extraction...")
    if blind_checklist is None:
        blind_checklist = generate_blind_checklist(rubrics_json)
    student_facts = stage1_blind_extraction(question_text, student_img_path, blind_checklist, q_img_path)

    if not student_facts:
        print("  [Stage 1] visual extraction failed; stop this sample.")
        return None

    # 浜屾鎻愬彇锛氬楂樼暀鐧界巼瀛︾敓杩涜鑱氱劍澶嶆煡
    try:
        rubrics_data_for_extraction = json.loads(rubrics_json) if isinstance(rubrics_json, str) else rubrics_json
        rubrics_data_for_extraction = prepare_rubrics_for_calibration(rubrics_data_for_extraction)
    except Exception:
        rubrics_data_for_extraction = []

    student_facts = stage1_targeted_reextraction(
        question_text, student_img_path, blind_checklist,
        student_facts, q_img_path, rubrics_data_for_extraction
    )

    print("  [Stage 2] independent semantic grading probes...")
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
                print(f"      鉁?[绗?{idx+1}/3 娆℃帰娴嬪畬鎴怾 寰楀垎: {score}")
                model_scores.append(score)
                strict_cots.append(cot) 
    
    if len(model_scores) == 0: return None

    # 璁＄畻鍧囧垎涓庢爣鍑嗗樊 (Self-Consistency)
    avg_model_score = round(float(np.mean(model_scores)), 1)
    std_dev = round(float(np.std(model_scores)), 4)
    
    # 鍔ㄦ€佽В鏋愯瘎浠锋爣鍑嗘€绘潯鐩暟鍜屾€诲垎
    try:
        rubrics_data = json.loads(rubrics_json) if isinstance(rubrics_json, str) else rubrics_json
        rubrics_data = prepare_rubrics_for_calibration(rubrics_data)
        TOTAL_ITEMS = len(rubrics_data) if isinstance(rubrics_data, list) else max(len(rubrics_data.keys()), 1)
        MAX_SCORE = sum(float(item.get('points', 0)) for item in rubrics_data) if isinstance(rubrics_data, list) else 100.0
    except:
        TOTAL_ITEMS, MAX_SCORE = 10, 10.0
    
    # 瑙ｆ瀽 Stage 1 鎻愬彇鐨?JSON 浜嬪疄锛岄€愭潯妫€鏌?value 鍊?
    try:
        facts_dict = json.loads(student_facts) if isinstance(student_facts, str) else student_facts
        if not isinstance(facts_dict, dict):
            facts_dict = {}
    except:
        facts_dict = {}

    extraction_counts = compute_extraction_quality_counts(facts_dict, rubrics_data)
    extraction_risk_features = compute_extraction_risk_features(extraction_counts)
    blank_count = extraction_counts["blank_count"]
    perception_fail_count = extraction_counts["perception_fail_count"]
    low_quality_count = extraction_counts["low_quality_count"]
    structure_missing_count = extraction_counts.get("structure_missing_count", 0)

    blank_rate = extraction_risk_features["blank_rate"]
    perception_failure_rate = extraction_risk_features["perception_failure_rate"]
    low_quality_rate = extraction_risk_features["low_quality_rate"]
    structure_missing_rate = extraction_risk_features["structure_missing_rate"]
    suspicious_extraction_rate = extraction_risk_features["suspicious_extraction_rate"]
    extraction_risk = extraction_risk_features["extraction_risk"]

    # 缁煎悎鎻愬彇璐ㄩ噺鍒ゅ畾
    extraction_quality = extraction_risk_features["extraction_quality"]

    real_diff = round(teacher_score - avg_model_score, 2) if teacher_score is not None else 0.0
    route = "UNKNOWN"
    final_score = avg_model_score
    reason_log = ""
    arbitration_flag = False

    print(f"\n  馃搳 [鎺㈡祴闆疯揪鎸囨爣] 鍧囧垎={avg_model_score}, 鏍囧噯宸?{std_dev:.4f}, 鐣欑櫧鐜?{blank_rate:.0%}, 鎰熺煡澶辨晥鐜?{perception_failure_rate:.0%}, 浣庤川閲忔彁鍙栫巼={low_quality_rate:.0%}, 鎻愬彇璐ㄩ噺={extraction_quality}")
    print(f"      [extraction-risk] R_ext={extraction_risk:.2f}, structure_missing={structure_missing_rate:.0%}, suspicious={suspicious_extraction_rate:.0%}")

    # ==========================================
    # 椋庨櫓椹卞姩涓夋敮鍐崇瓥 (Three-Way Decision)
    # POS锛氫綆椋庨櫓鐩存帴鎺ュ彈
    # BND锛氫腑椋庨櫓瀹芥澗浼樺厛銆佽皑鎱庝笅璋?
    # NEG锛氶珮椋庨櫓鎷掑垽/浜哄伐澶嶆牳
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
        structure_missing_rate=structure_missing_rate,
        suspicious_extraction_rate=suspicious_extraction_rate,
        extraction_risk=extraction_risk,
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
    primary_risks = post_calibration.get("primary_risks", {})
    risk_profile["risk_features"].update({
        "U_E": primary_risks.get("U_E", None),
        "U_S": primary_risks.get("U_S", None),
        "U_R": primary_risks.get("U_R", None),
        "primary_risk": primary_risks.get("risk", None),
        "primary_mu": primary_risks.get("mu", None),
        "unsupported_match_points_ratio": post_calibration["unsupported_match_points_ratio"],
        "method_final_verified_ratio": post_calibration["method_final_verified_ratio"],
        "direct_points_ratio": post_calibration["direct_points_ratio"],
        "direct_awarded_ratio": post_calibration["direct_awarded_ratio"],
        "result_correctness_signal": post_calibration["result_correctness_signal"],
        "result_strong_signal": post_calibration["result_strong_signal"],
        "method_evidence_signal": post_calibration["method_evidence_signal"],
        "partial_or_format_points_ratio": post_calibration["partial_or_format_points_ratio"],
        "bare_answer_risk": post_calibration["bare_answer_risk"],
        "lenient_undercredit_signal": post_calibration["lenient_undercredit_signal"],
        "unsupported_high_score_risk": post_calibration["unsupported_high_score_risk"],
        "metadata_coverage": post_calibration["metadata_coverage"],
        "explicit_chain_coverage": post_calibration["explicit_chain_coverage"],
        "core_anchor_failed": post_calibration["core_anchor_failed"],
        "visual_blank_review": post_calibration["visual_blank_review"],
        "structure_missing_review": post_calibration.get("structure_missing_review", False),
        "structure_missing_rate": structure_missing_rate,
        "suspicious_extraction_rate": suspicious_extraction_rate,
        "extraction_risk": extraction_risk,
        "weak_result_high_score_review": post_calibration["weak_result_high_score_review"],
        "stable_undercredit_review": post_calibration["stable_undercredit_review"],
        "direct_only_high_score_risk": post_calibration["direct_only_high_score_risk"],
        "task_type": post_calibration.get("task_type", "mixed_or_unknown"),
        "complex_derivation_task": post_calibration.get("complex_derivation_task", False),
        "upper_consensus_eligible": post_calibration.get("upper_consensus_eligible", False),
        "rubric_task_profile": post_calibration.get("rubric_task_profile", {}),
        "calibration_rule_hits": post_calibration["rule_hits"],
    })
    baseline_selection = select_baseline_score(
        model_scores=model_scores,
        model_avg_score=avg_model_score,
        max_score=MAX_SCORE,
        post_calibration=post_calibration,
        risk_profile=risk_profile,
    )
    selected_baseline_score = round(_clamp(baseline_selection["selected_baseline_score"], 0, MAX_SCORE), 2)
    baseline_signals = baseline_selection.get("baseline_selection_signals", {})
    post_calibration.update({
        "selected_baseline_score": selected_baseline_score,
        "baseline_policy": baseline_selection.get("baseline_policy", "model_avg"),
        "baseline_score_source": baseline_selection.get("baseline_score_source", "model_avg_score"),
        "baseline_selection_signals": baseline_signals,
        "score_history_max": baseline_signals.get("score_history_max", selected_baseline_score),
        "score_history_median": baseline_signals.get("score_history_median", selected_baseline_score),
        "score_history_min": baseline_signals.get("score_history_min", selected_baseline_score),
        "high_score_safety_review": bool(baseline_signals.get("high_score_safety_review", False)),
    })
    if post_calibration["high_score_safety_review"]:
        if "high_score_safety_review" not in post_calibration["rule_hits"]:
            post_calibration["rule_hits"].append("high_score_safety_review")
        post_calibration["boundary_domain"] = True
    risk_profile["risk_features"].update({
        "model_avg_ratio": round(avg_model_score / MAX_SCORE, 4) if MAX_SCORE > 0 else 0.0,
        "avg_ratio": round(selected_baseline_score / MAX_SCORE, 4) if MAX_SCORE > 0 else 0.0,
        "selected_baseline_score": selected_baseline_score,
        "baseline_policy": post_calibration["baseline_policy"],
        "baseline_score_source": post_calibration["baseline_score_source"],
        "baseline_selection_signals": baseline_signals,
        "high_score_safety_review": post_calibration["high_score_safety_review"],
    })
    risk_profile["high_blank_high_score"] = blank_rate >= 0.50 and selected_baseline_score >= 0.60 * MAX_SCORE
    risk_profile["lenient_review_signal"] = selected_baseline_score <= 0.60 * MAX_SCORE and blank_rate <= 0.35
    risk_profile["risk_features"]["high_blank_high_score"] = risk_profile["high_blank_high_score"]
    risk_profile["risk_features"]["lenient_review_signal"] = risk_profile["lenient_review_signal"]
    final_score = selected_baseline_score
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
        avg_model_score=selected_baseline_score,
        std_dev=std_dev,
        max_score=MAX_SCORE,
        blank_rate=blank_rate,
        low_quality_rate=low_quality_rate,
        perception_failure_rate=perception_failure_rate,
        extraction_quality=extraction_quality,
        structure_missing_rate=structure_missing_rate,
        extraction_risk=extraction_risk,
        fatal_points_ratio=post_calibration.get("fatal_ratio", risk_profile["fatal_points_ratio"]),
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
        "      馃摗 [椋庨櫓鐢诲儚] "
        f"P={perception_risk:.2f}, U={uncertainty_index:.2%}, F={fatal_points_ratio:.2%}, "
        f"H={high_blank_high_score}, L={lenient_review_signal}"
    )
    if post_calibration["rule_hits"]:
        print(
            "      馃Л [閫氱敤鏍″噯] "
            f"rules={post_calibration['rule_hits']}, "
            f"UM={post_calibration['unsupported_match_points_ratio']:.2%}, "
            f"MF={post_calibration['method_final_verified_ratio']:.2%}, "
            f"cap={post_calibration['upper_bound']:.2f}"
        )
    print(
        "      馃М [A3WA鍙俊搴 "
        f"R={a3wa_decision['risk']:.3f}, 渭={a3wa_decision['mu']:.3f}, "
        f"伪={a3wa_decision['alpha']:.3f}, 尾={a3wa_decision['beta']:.3f}, "
        f"route={a3wa_decision['route']} | {a3wa_decision['reason']}"
    )

    if reject_domain:
        route = "NEG"
        arbitration_flag = True
        if a3wa_decision["hard_neg_reasons"]:
            reason_log = "A3WA hard_neg: " + ",".join(a3wa_decision["hard_neg_reasons"])
        else:
            reason_log = f"A3WA low_confidence_neg: mu={a3wa_decision['mu']:.3f} <= beta={a3wa_decision['beta']:.3f}"
        print(f"      馃洃 [璺敱 -> NEG] 楂橀闄╂嫆鍒?| P={perception_risk:.2f}, U={uncertainty_index:.2%}, F={fatal_points_ratio:.2%}, H={high_blank_high_score}")
        print(f"         馃攳 [澶嶆牳鎻愮ず] {reason_log}")

    elif boundary_domain:
        route = "BND"
        print(
            f"      鈿狅笍 [璺敱 -> BND] 杈圭晫鏍锋湰 | "
            f"P={perception_risk:.2f}, U={uncertainty_index:.2%}, F={fatal_points_ratio:.2%}, H={high_blank_high_score}, L={lenient_review_signal}"
        )

        agent_res_text = boundary_arbitration_agent(student_facts, strict_cots, rubrics_json, risk_profile)
        if agent_res_text:
            parsed_agent = extract_and_parse_json(agent_res_text)
            if parsed_agent:
                raw_agent_score = _agent_candidate_score(parsed_agent, selected_baseline_score, MAX_SCORE)
                arbitration_decision = parsed_agent.get("decision", "keep")
                boundary_gate = apply_boundary_action_policy(
                    avg_model_score=selected_baseline_score,
                    candidate_score=raw_agent_score,
                    max_score=MAX_SCORE,
                    a3wa_decision=a3wa_decision,
                    risk_profile=risk_profile,
                    post_calibration=post_calibration,
                    agent_evidence=parsed_agent,
                )
                final_score = round(_clamp(boundary_gate["final_score"], 0, MAX_SCORE), 2)
                arbitration_decision = f"{arbitration_decision}|{boundary_gate['action']}"
                risk_features["boundary_gate_action"] = boundary_gate["action"]
                risk_features["boundary_gate_accepted"] = boundary_gate["accepted"]
                reason_log = parsed_agent.get("reason", parsed_agent.get("leniency_reason", ""))
                print(
                    f"         鉁?[Agent 浠茶] {arbitration_decision} | "
                    f"鍘熷浠茶鍒? {raw_agent_score} | 闄愬箙鍚庢渶缁堝垎: {final_score} | {reason_log}"
                )
            else:
                arbitration_decision = "keep"
                lower_bound, upper_bound, _ = a3wa_dynamic_bounds(
                    avg_model_score=selected_baseline_score,
                    max_score=MAX_SCORE,
                    a3wa_decision=a3wa_decision,
                    risk_profile=risk_profile,
                    post_calibration=post_calibration,
                )
                final_score = round(_clamp(selected_baseline_score, lower_bound, upper_bound), 2)
        else:
            arbitration_decision = "keep"
            lower_bound, upper_bound, _ = a3wa_dynamic_bounds(
                avg_model_score=selected_baseline_score,
                max_score=MAX_SCORE,
                a3wa_decision=a3wa_decision,
                risk_profile=risk_profile,
                post_calibration=post_calibration,
            )
            final_score = round(_clamp(selected_baseline_score, lower_bound, upper_bound), 2)

    else:
        route = "POS"
        final_score = selected_baseline_score
        print(f"      [route -> POS] accept selected_baseline_score={selected_baseline_score}")

    # 缁撴灉灏佽
    ordered_result = {
        "student_id": student_id,
        "teacher_score": teacher_score,
        "model_scores_history": model_scores,
        "model_avg_score": avg_model_score,
        "selected_baseline_score": selected_baseline_score,
        "baseline_policy": post_calibration["baseline_policy"],
        "baseline_score_source": post_calibration["baseline_score_source"],
        "baseline_selection_signals": baseline_signals,
        "std_dev": std_dev,
        "blank_rate": round(blank_rate, 2),
        "perception_failure_rate": round(perception_failure_rate, 2),
        "low_quality_extraction_rate": round(low_quality_rate, 2),
        "structure_missing_rate": round(structure_missing_rate, 2),
        "suspicious_extraction_rate": round(suspicious_extraction_rate, 2),
        "extraction_risk": round(extraction_risk, 4),
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
