import os
import json
import math
from json_repair import repair_json
import base64
import re
import time
import concurrent.futures
from datetime import datetime
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageOps
import io
import numpy as np
from ocr.backend import load_json as load_ocr_json
from ocr.backend import sha256_file
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

# ==================== 配置区 ====================
# 视觉模型切换：可选 "glm4v" / "glm5v"
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

# 文本模型切换：可选 "glm" / "glm5" / "deepseek"
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

# 并发配置：provider -> (外层学生并发, 内层 Stage2 探测并发)
MODEL_CONCURRENCY = {
    "glm":      (3, 3),  # GLM-4.5-air 并发能力较强
    "glm5":     (2, 2),  # GLM-5.1 限流较严，保持较低并发
    "deepseek": (2, 2),  # 第三方代理，保守并发
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
    """统一文本模型调用入口，根据 TEXT_MODEL_PROVIDER 自动分发。"""
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
        raise Exception("GLM-5 连续 4 次重试均失败")
    elif TEXT_MODEL_PROVIDER == "deepseek":
        # DeepSeek：每次请求使用独立客户端并重试，降低并发限流影响。
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
        raise Exception("DeepSeek 连续 4 次重试均失败")
    else:
        response = glm_client.chat.completions.create(
            model=GLM_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            timeout=timeout
        )
        return response.choices[0].message.content.strip()


def call_glm5_text(messages, temperature=0.1, timeout=180):
    """Call GLM-5.1 explicitly for OCR-to-fact mapping."""
    for attempt in range(4):
        try:
            client = OpenAI(api_key=GLM5_API_KEY, base_url=GLM5_BASE_URL)
            response = client.chat.completions.create(
                model=GLM5_MODEL_NAME,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            wait = 5 * (attempt + 1)
            print(
                f"         [GLM-5.1 mapping retry {attempt + 1}/4] "
                f"{type(exc).__name__}: {str(exc)[:80]}... wait {wait}s"
            )
            time.sleep(wait)
    raise RuntimeError("GLM-5.1 OCR fact mapping failed after 4 attempts")

# ==================== 工具函数 ====================

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
            print(f"   [压缩] {width}x{height} -> {new_width}x{new_height}")
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
    except Exception as e:
        print(f"   [压缩出错] {e}")
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


def diagram_focus_to_base64(image_path):
    """Create an enlarged lower-page view for timing/order diagrams."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        image = Image.open(image_path).convert("RGB")
        top = int(image.height * 0.43)
        crop = image.crop((0, top, image.width, image.height))
        crop = ImageOps.autocontrast(crop, cutoff=1)
        crop = crop.resize(
            (crop.width * 2, crop.height * 2),
            Image.Resampling.LANCZOS,
        )
        buffer = io.BytesIO()
        crop.save(buffer, format="JPEG", quality=98)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as exc:
        print(f"   [diagram focus] failed: {exc}")
        return None

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
            print(f"[JSON解析失败] {e}")
            return None

# ==================== 提取后题面参数保护 ====================

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

# ==================== 核心业务逻辑 ====================

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


def _checklist_ids(blind_checklist):
    parsed = extract_and_parse_json(blind_checklist) if isinstance(blind_checklist, str) else blind_checklist
    if not isinstance(parsed, list):
        return []
    return [str(item.get("id", "")) for item in parsed if str(item.get("id", ""))]


def _diagram_checklist(blind_checklist, rubrics_json):
    checklist = extract_and_parse_json(blind_checklist) if isinstance(blind_checklist, str) else blind_checklist
    rubrics = extract_and_parse_json(rubrics_json) if isinstance(rubrics_json, str) else rubrics_json
    checklist = checklist if isinstance(checklist, list) else []
    rubrics = rubrics if isinstance(rubrics, list) else []
    diagram_ids = {
        str(item.get("id", ""))
        for item in rubrics
        if (
            "diagram" in str(item.get("evidence_source", "")).lower()
            or "diagram" in str(item.get("answer_type", "")).lower()
            or "图" in str(item.get("answer_type", ""))
        )
    }
    return [item for item in checklist if str(item.get("id", "")) in diagram_ids]


VISUAL_PLACEHOLDER_PATTERNS = (
    r"如图所示",
    r"见图",
    r"图如下",
    r"如下图",
    r"答案见图",
    r"状态图略",
    r"如下表",
)


def detect_visual_placeholder(text):
    compact = re.sub(r"\s+", "", str(text or ""))
    return any(re.search(pattern, compact) for pattern in VISUAL_PLACEHOLDER_PATTERNS)


def map_transcription_to_facts(
    question_text,
    blind_checklist,
    student_transcription,
    visual_placeholder_detected=False,
):
    """Map the human transcription to blind rubric facts with GLM-5.1."""
    checklist_ids = _checklist_ids(blind_checklist)
    prompt = f"""
# Role: student-transcription fact mapper

Map the student's existing human transcription to the blind checklist. Do not
grade, solve the question, or use the reference answer. Preserve concrete
student statements, formulas, numbers, code, and intermediate reasoning.

Question context:
{question_text}

Blind checklist:
{blind_checklist}

Student transcription:
{student_transcription}

Visual-placeholder detected: {visual_placeholder_detected}

Rules:
- Return one strict JSON object whose keys are exactly the checklist ids.
- "如图所示", "见图", "如下图" and similar phrases are placeholders, not
  diagram content. For diagram-dependent items return "需要查看图像".
- Use "未书写" only when the transcription has no relevant answer.
- Do not overwrite valid transcribed text with guesses from the question.
"""
    raw = call_glm5_text(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        timeout=180,
    )
    mapped = extract_and_parse_json(raw)
    if not isinstance(mapped, dict):
        raise ValueError("GLM-5.1 transcription mapping did not return JSON")
    return {item_id: mapped.get(item_id, "未书写") for item_id in checklist_ids}


def map_paddle_ocr_to_facts(question_text, blind_checklist, ocr_payload):
    """Map raw OCR evidence to blind checklist items with GLM-5.1."""
    checklist_ids = _checklist_ids(blind_checklist)
    blank_status = (
        ocr_payload.get("summary", {})
        .get("blank_authenticity", {})
        .get("status", "uncertain")
    )
    if blank_status == "confirmed_blank":
        return {item_id: "未书写" for item_id in checklist_ids}

    tokens = [
        {
            "text": token.get("text", ""),
            "confidence": token.get("confidence"),
            "box": token.get("box"),
        }
        for token in ocr_payload.get("tokens", [])
    ]
    prompt = f"""
# Role: OCR evidence to rubric-fact mapper

Map the raw OCR tokens to the blind checklist. Use only visible student-answer
evidence. Printed question text, axes, score marks, and OCR noise are not student
answers. Do not solve the question and do not invent missing relations.

Question context (for locating printed text only):
{question_text}

Blind checklist:
{blind_checklist}

OCR tokens with confidence and coordinates:
{json.dumps(tokens, ensure_ascii=False)}

Blank diagnostic:
{json.dumps(ocr_payload.get("summary", {}).get("blank_authenticity", {}), ensure_ascii=False)}

Return one strict JSON object. Its keys must be exactly the checklist ids.
For each value:
- transcribe the concrete observed answer;
- use "未书写" only when the area is truly blank;
- use "字迹模糊" when marks exist but cannot be read;
- for diagrams, report only labels/text visible to OCR; do not infer edges.
"""
    raw = call_glm5_text([{"role": "user", "content": prompt}], temperature=0.1, timeout=180)
    mapped = extract_and_parse_json(raw)
    if not isinstance(mapped, dict):
        raise ValueError("GLM-5.1 OCR mapping did not return a JSON object")
    return {item_id: mapped.get(item_id, "未书写") for item_id in checklist_ids}


def parse_diagram_relations_with_glm4v(
    question_text,
    student_img_path,
    diagram_checklist,
    q_img_path=None,
    ocr_payload=None,
):
    """Conditionally parse diagram topology with GLM-4.6V."""
    if not diagram_checklist:
        return {}, None
    prompt = f"""
# Role: diagram relation transcriber

Inspect only the student's drawn diagram. Do not grade and do not solve the
question from its wording. Report the topology actually visible in the answer:
segment order, nesting/return edges, labels, and transitions. If a requested
relation is not visibly supported, return "图中未明确显示"; if no diagram was
drawn, return "未书写".

Evidence rules:
- Never turn isolated labels, repeated letters, table cells, or spatially
  aligned marks into arrows or execution order unless visible lines/arrows
  connect them.
- For staircase/timing/order diagrams, output a path only when a continuous
  solid path is visibly drawn. Preserve repeated return levels, e.g. C→D→C.
- For state graphs, report only visibly connected directed edges.
- For tables, report row/column/cell relations instead of inventing a path.
- For trees, Hasse diagrams, circuits, and parse graphs, report visible nodes
  and edges with direction when present.
- Distinguish solid execution paths from dashed request markers and axes.
- If labels exist but no required connecting relation is visible, explicitly
  say "图中有标签但未形成可确认的连接关系".

Question context:
{question_text}

Diagram-only blind checklist:
{json.dumps(diagram_checklist, ensure_ascii=False)}

PaddleOCR labels and coordinates from the student image:
{json.dumps([
    {
        "text": token.get("text", ""),
        "confidence": token.get("confidence"),
        "box": token.get("box"),
    }
    for token in (ocr_payload or {}).get("tokens", [])
], ensure_ascii=False)}

Return strict JSON in this shape:
{{
  "diagram_status": "present_clear|present_uncertain|missing",
  "observed_execution_path": "only when a continuous visible path exists; otherwise null",
  "items": {{
    "checklist_id": "visible relation for this item"
  }}
}}
"""
    content = [{"type": "text", "text": prompt}]
    if q_img_path and os.path.exists(q_img_path):
        q_b64 = encode_image_to_base64(q_img_path)
        content.extend(
            [
                {"type": "text", "text": "Printed axis/template reference:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{q_b64}"}},
            ]
        )
    student_b64 = encode_image_to_base64(student_img_path)
    content.extend(
        [
            {"type": "text", "text": "Student answer image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{student_b64}"}},
        ]
    )
    diagram_focus_b64 = diagram_focus_to_base64(student_img_path)
    if diagram_focus_b64:
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "Enlarged lower-page diagram view. Use this view to trace "
                        "every rise, fall, repeated level, and return segment:"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{diagram_focus_b64}"},
                },
            ]
        )
    for attempt in range(4):
        try:
            client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)
            response = client.chat.completions.create(
                model=VLM_MODELS["glm4v"],
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
                timeout=180,
            )
            parsed = extract_and_parse_json(response.choices[0].message.content.strip())
            if isinstance(parsed, dict):
                allowed = {str(item.get("id", "")) for item in diagram_checklist}
                item_values = parsed.get("items", parsed)
                if isinstance(item_values, dict):
                    facts = {
                        str(key): value
                        for key, value in item_values.items()
                        if str(key) in allowed
                    }
                    return facts, parsed.get("observed_execution_path")
        except Exception as exc:
            print(f"         [GLM-4.6V diagram retry {attempt + 1}/4] {exc}")
            time.sleep(10 * (attempt + 1))
    return {}, None


def stage1_extract_with_backend(
    question_text,
    student_img_path,
    blind_checklist,
    rubrics_json,
    q_img_path=None,
    extraction_backend="glm_vlm",
    ocr_json_path=None,
    extraction_cache_path=None,
    force_extraction=False,
    student_transcription=None,
    answer_metadata=None,
):
    """Run or load one Stage-1 extraction backend."""
    if extraction_backend == "glm_vlm":
        facts = stage1_blind_extraction(
            question_text, student_img_path, blind_checklist, q_img_path
        )
        return facts, {"backend": "glm_vlm", "diagram_parser_used": False}

    if extraction_backend not in ("paddle_glm5", "csbench_hybrid"):
        raise ValueError(f"Unsupported extraction backend: {extraction_backend}")

    image_hash = sha256_file(student_img_path)
    answer_metadata = answer_metadata if isinstance(answer_metadata, dict) else {}
    transcription = str(student_transcription or "")
    visual_placeholder_detected = detect_visual_placeholder(transcription)
    if extraction_cache_path and not force_extraction:
        cached = load_ocr_json(extraction_cache_path)
        if (
            cached
            and cached.get("backend") == extraction_backend
            and cached.get("image_sha256") == image_hash
            and isinstance(cached.get("facts"), dict)
            and (
                extraction_backend != "csbench_hybrid"
                or cached.get("student_transcription") == transcription
            )
        ):
            return json.dumps(cached["facts"], ensure_ascii=False), cached

    diagram_items = _diagram_checklist(blind_checklist, rubrics_json)
    if extraction_backend == "csbench_hybrid" and not (
        bool(answer_metadata.get("isimagine")) or visual_placeholder_detected
    ):
        diagram_items = []
    ocr_payload = {}
    if extraction_backend == "paddle_glm5" or diagram_items:
        if not ocr_json_path:
            raise ValueError(f"{extraction_backend} requires ocr_json_path")
        ocr_payload = load_ocr_json(ocr_json_path)
        if not ocr_payload:
            raise FileNotFoundError(
                f"OCR evidence not found or invalid: {ocr_json_path}"
            )
        if ocr_payload.get("image", {}).get("sha256") != image_hash:
            raise ValueError(f"OCR cache image hash mismatch: {ocr_json_path}")

    if extraction_backend == "csbench_hybrid":
        if not transcription.strip():
            facts = {
                item_id: "未书写"
                for item_id in _checklist_ids(blind_checklist)
            }
        else:
            facts = map_transcription_to_facts(
                question_text,
                blind_checklist,
                transcription,
                visual_placeholder_detected=visual_placeholder_detected,
            )
    else:
        facts = map_paddle_ocr_to_facts(
            question_text, blind_checklist, ocr_payload
        )

    diagram_facts, observed_execution_path = parse_diagram_relations_with_glm4v(
        question_text,
        student_img_path,
        diagram_items,
        q_img_path=q_img_path,
        ocr_payload=ocr_payload,
    )
    if observed_execution_path:
        diagram_facts = {
            item_id: (
                f"{value}；完整可见路径：{observed_execution_path}"
                if observed_execution_path not in str(value)
                else value
            )
            for item_id, value in diagram_facts.items()
        }
    facts.update(diagram_facts)
    evidence = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(),
        "backend": extraction_backend,
        "image_path": os.path.abspath(student_img_path),
        "image_sha256": image_hash,
        "ocr_json_path": (
            os.path.abspath(ocr_json_path)
            if ocr_json_path and ocr_payload
            else None
        ),
        "ocr_engine": ocr_payload.get("engine", {}),
        "ocr_summary": ocr_payload.get("summary", {}),
        "student_transcription": transcription if extraction_backend == "csbench_hybrid" else None,
        "transcription_source": (
            "csbench_human_transcription"
            if extraction_backend == "csbench_hybrid"
            else None
        ),
        "visual_placeholder_detected": visual_placeholder_detected,
        "answer_metadata": {
            key: answer_metadata.get(key)
            for key in ("answer_id", "question_id", "subject", "isimagine")
            if key in answer_metadata
        },
        "diagram_parser_used": bool(diagram_items),
        "diagram_model": VLM_MODELS["glm4v"] if diagram_items else None,
        "observed_execution_path": observed_execution_path,
        "diagram_facts": diagram_facts,
        "diagram_checklist": diagram_items,
        "facts": facts,
    }
    if extraction_cache_path:
        os.makedirs(os.path.dirname(extraction_cache_path) or ".", exist_ok=True)
        with open(extraction_cache_path, "w", encoding="utf-8") as handle:
            json.dump(evidence, handle, indent=2, ensure_ascii=False)
    return json.dumps(facts, ensure_ascii=False), evidence

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
    """Stage 2：根据视觉提取事实和细粒度评分准则进行语义判分。"""
    logic_prompt = f"""
你是严格的计算机类课程阅卷助手。你的任务是依据【学生客观事实】和【细粒度评分准则】逐项给分，不做表面字符串匹配，也不能凭空补全学生没有写出的内容。

【学生客观事实】
{student_facts_str}

【细粒度评分准则】
{rubrics_json_str}

判分规则：
1. 只能依据学生客观事实判分。若事实标注为“未书写”“字迹模糊”或没有具体内容，不能脑补给分。
2. 若事实只写“有”“是”“正确”“有计算过程”等非具体内容，判为 INSUFFICIENT_INFO，给 0 分。
3. 数值题优先比较核心数值；没有显式容差时，相对误差 <= 10% 可判 MATCH。单位可换算后比较。
4. 格式、单位、空格、箭头、大小写等非实质差异不应重罚；核心值或核心结论正确时可判 FORMAT_MINOR。
5. 多要素评分项应允许 PARTIAL_MATCH，按已完成要素比例给分，避免全有全无。
6. 对链式推导题，若学生起点值错误但后续用正确公式一致推导，可将后续过程项判为 PARTIAL_MATCH；若最终结论完全相反或核心方法错误，判 SEMANTIC_FATAL。
7. 若最终结果正确且能反推必要的上游参数或中间量已被正确使用，可对未单独展开的上游项谨慎判 MATCH，并在 reason 中说明“由下游正确结果回溯确认”。
8. 若参数识别正确但核心公式、方法和最终结果均错误，不应给大量空洞参数分。

error_category 只能取以下值之一：
- MATCH：语义匹配，给该项满分。
- BLANK：未书写或字迹模糊，给 0 分。
- SEMANTIC_FATAL：核心知识、结论、数值或方法错误，给 0 分。
- FORMAT_MINOR：非实质格式问题，核心值或结论正确，通常给该项 70% 分数。
- INSUFFICIENT_INFO：提取信息不足，无法判断，给 0 分。
- PARTIAL_MATCH：部分匹配，按完成比例给部分分。

必须输出纯 JSON，格式如下：
{{
  "total_score": 数字,
  "details": [
    {{"id": "item_id", "score_given": 数字, "error_category": "MATCH", "reason": "简要原因"}}
  ]
}}
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
        except Exception:
            time.sleep(3)
    return None

def zero_shot_leniency_agent(student_facts_str, strict_cot_str, rubrics_json_str):
    """宽松复查 Agent：模拟教师宽松阅卷口径，对初审扣分项进行二次审查。"""
    leniency_prompt = f"""
# Role: 高校阅卷组长（教师视角复查）
你正在复核一份自动阅卷结果。真实教师在期末考试中通常更关注核心结论、关键公式和整体推导思路；若学生结论正确且有一定过程依据，中间展开不完整不应被过度扣分。

【评分准则】
{rubrics_json_str}

【学生客观事实】
{student_facts_str}

【初审扣分记录】
{strict_cot_str}

复查原则：
1. 若学生最终答案正确，并能看到必要公式、转换或过程痕迹，可以恢复合理分数。
2. 若学生只有裸答案且缺乏过程，最多恢复结论相关分，不应恢复完整过程分。
3. 若高分项缺少答案或核心过程依据，可以维持或小幅下调。
4. 若学生核心概念、方法或最终结论完全错误，维持 0 分。
5. 对多要素条目，若学生完成部分要素，应按比例恢复，而不是全有全无。
6. 数值类条目存在合理近似、进位或单位换算差异时，可宽松恢复；但数量级错误或结论相反不能恢复。

输出纯 JSON：
{{
  "secondary_total_score": 复查后的总分,
  "leniency_reason": "50字以内概括主要修正理由",
  "analysis_cot": "逐条复查过程"
}}
"""
    for attempt in range(3):
        try:
            return call_text_model(
                [{"role": "user", "content": leniency_prompt}],
                temperature=0.3, timeout=120
            )
        except Exception:
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
    """BND 边界样本仲裁：在证据充分时进行有限上调或下调。"""
    arbitration_prompt = f"""
# Role: 边界样本双向校准仲裁员
你正在复核一份自动阅卷结果，目标是让分数更接近真实教师评分。

【评分准则】
{rubrics_json_str}

【学生客观作答事实】
{student_facts_str}

【三次独立评分记录】
{json.dumps(strict_cots, ensure_ascii=False)}

【风险特征】
{json.dumps(risk_profile, ensure_ascii=False)}

仲裁原则：
1. 同时考虑模型可能低估和高估。
2. 学生有正确公式、转换、过程痕迹或最终答案正确时，可以考虑上调。
3. 学生只有参数抄录、裸答案或核心方法错误时，不应空洞加分，可以考虑下调。
4. 格式、单位习惯、表达顺序等非实质问题不应重罚。
5. 证据不足或方向不明确时，保持原分。
6. 输出分数必须在题目满分范围内。

输出纯 JSON：
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

# ==================== 核心流水线：零样本无监督 3WD ====================

def grade_student_3wd_pipeline(
    student_img_path,
    question_text,
    rubrics_json,
    teacher_score,
    q_img_path=None,
    blind_checklist=None,
    extraction_backend="glm_vlm",
    ocr_json_path=None,
    extraction_cache_path=None,
    force_extraction=False,
    grade_only=False,
    student_transcription=None,
    answer_metadata=None,
):
    student_id = os.path.splitext(os.path.basename(student_img_path))[0]
    def _pipeline_failure(error_type, reason):
        return {
            "_pipeline_failed": True,
            "student_id": student_id,
            "error_type": error_type,
            "reason": str(reason),
        }

    print(f"\n=============================================")
    print(f"Start grading student [{student_id}]")
    print(f"=============================================")

    print(f"  [Stage 1] fact extraction backend={extraction_backend}...")
    if blind_checklist is None and not grade_only:
        blind_checklist = generate_blind_checklist(rubrics_json)
    extraction_evidence = {"backend": extraction_backend}
    if grade_only:
        cached = load_ocr_json(extraction_cache_path) if extraction_cache_path else None
        if not cached or not isinstance(cached.get("facts"), dict):
            return _pipeline_failure(
                "extraction_cache_missing",
                f"GRADE_ONLY requires a valid extraction cache: {extraction_cache_path}",
            )
        if cached.get("image_sha256") != sha256_file(student_img_path):
            return _pipeline_failure(
                "extraction_cache_stale",
                f"GRADE_ONLY extraction cache hash mismatch: {extraction_cache_path}",
            )
        student_facts = json.dumps(cached["facts"], ensure_ascii=False)
        extraction_evidence = cached
    else:
        try:
            student_facts, extraction_evidence = stage1_extract_with_backend(
                question_text=question_text,
                student_img_path=student_img_path,
                blind_checklist=blind_checklist,
                rubrics_json=rubrics_json,
                q_img_path=q_img_path,
                extraction_backend=extraction_backend,
                ocr_json_path=ocr_json_path,
                extraction_cache_path=extraction_cache_path,
                force_extraction=force_extraction,
                student_transcription=student_transcription,
                answer_metadata=answer_metadata,
            )
        except Exception as exc:
            return _pipeline_failure("stage1_failed", exc)

    if not student_facts:
        print("  [Stage 1] extraction failed; stop this sample.")
        return _pipeline_failure("stage1_failed", "fact extraction returned empty result")

    # 二次提取：对疑似提取失败或高留白样本进行聚焦复查。
    try:
        rubrics_data_for_extraction = json.loads(rubrics_json) if isinstance(rubrics_json, str) else rubrics_json
        rubrics_data_for_extraction = prepare_rubrics_for_calibration(rubrics_data_for_extraction)
    except Exception:
        rubrics_data_for_extraction = []

    if extraction_backend == "glm_vlm":
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
                print(f"      [完成] 第 {idx+1}/3 次语义评分完成，得分: {score}")
                model_scores.append(score)
                strict_cots.append(cot) 
    
    if len(model_scores) == 0:
        return _pipeline_failure("stage2_failed", "all semantic grading probes failed or returned unparsable scores")

    # 计算三次评分均分与标准差（Self-Consistency）。
    avg_model_score = round(float(np.mean(model_scores)), 1)
    std_dev = round(float(np.std(model_scores)), 4)
    
    # 动态解析评分准则的条目数与总分。
    try:
        rubrics_data = json.loads(rubrics_json) if isinstance(rubrics_json, str) else rubrics_json
        rubrics_data = prepare_rubrics_for_calibration(rubrics_data)
        TOTAL_ITEMS = len(rubrics_data) if isinstance(rubrics_data, list) else max(len(rubrics_data.keys()), 1)
        MAX_SCORE = sum(float(item.get('points', 0)) for item in rubrics_data) if isinstance(rubrics_data, list) else 100.0
    except:
        TOTAL_ITEMS, MAX_SCORE = 10, 10.0
    
    # 解析 Stage 1 提取的 JSON 事实，逐条检查 value。
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

    # 综合判断提取质量。
    extraction_quality = extraction_risk_features["extraction_quality"]

    real_diff = round(teacher_score - avg_model_score, 2) if teacher_score is not None else 0.0
    route = "UNKNOWN"
    final_score = avg_model_score
    reason_log = ""
    arbitration_flag = False

    print(f"\n  [探测指标] 均分={avg_model_score}, 标准差={std_dev:.4f}, 留白率={blank_rate:.0%}, 感知失败率={perception_failure_rate:.0%}, 低质量提取率={low_quality_rate:.0%}, 提取质量={extraction_quality}")
    print(f"      [extraction-risk] R_ext={extraction_risk:.2f}, structure_missing={structure_missing_rate:.0%}, suspicious={suspicious_extraction_rate:.0%}")

    # ==========================================
    # 风险驱动三支决策（Three-Way Decision）
    # POS：低风险，直接采信。
    # BND：中风险，进入边界仲裁。
    # NEG：高风险，拒判或交人工复核。
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
    calibrated_fatal_ratio = post_calibration.get("fatal_ratio", risk_profile["fatal_points_ratio"])
    risk_profile["fatal_points_ratio"] = calibrated_fatal_ratio
    risk_profile["risk_features"]["fatal_points_ratio"] = calibrated_fatal_ratio
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
        "      [风险画像] "
        f"P={perception_risk:.2f}, U={uncertainty_index:.2%}, F={fatal_points_ratio:.2%}, "
        f"H={high_blank_high_score}, L={lenient_review_signal}"
    )
    if post_calibration["rule_hits"]:
        print(
            "      [通用校准] "
            f"rules={post_calibration['rule_hits']}, "
            f"UM={post_calibration['unsupported_match_points_ratio']:.2%}, "
            f"MF={post_calibration['method_final_verified_ratio']:.2%}, "
            f"cap={post_calibration['upper_bound']:.2f}"
        )
    print(
        "      [A3WA可信度] "
        f"R={a3wa_decision['risk']:.3f}, mu={a3wa_decision['mu']:.3f}, "
        f"alpha={a3wa_decision['alpha']:.3f}, beta={a3wa_decision['beta']:.3f}, "
        f"route={a3wa_decision['route']} | {a3wa_decision['reason']}"
    )

    if reject_domain:
        route = "NEG"
        arbitration_flag = True
        if a3wa_decision["hard_neg_reasons"]:
            reason_log = "A3WA hard_neg: " + ",".join(a3wa_decision["hard_neg_reasons"])
        else:
            reason_log = f"A3WA low_confidence_neg: mu={a3wa_decision['mu']:.3f} <= beta={a3wa_decision['beta']:.3f}"
        print(f"      [路由 -> NEG] 高风险拒判 | P={perception_risk:.2f}, U={uncertainty_index:.2%}, F={fatal_points_ratio:.2%}, H={high_blank_high_score}")
        print(f"         [复核提示] {reason_log}")

    elif boundary_domain:
        route = "BND"
        print(
            f"      [路由 -> BND] 边界样本 | "
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
                    f"         [Agent仲裁] {arbitration_decision} | "
                    f"原始仲裁分: {raw_agent_score} | 限幅后最终分: {final_score} | {reason_log}"
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

    # 结果封装。
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
        "extraction_backend": extraction_backend,
        "extraction_evidence": extraction_evidence,
        "answer_metadata": {
            key: (answer_metadata or {}).get(key)
            for key in ("answer_id", "question_id", "subject", "isimagine")
            if key in (answer_metadata or {})
        },
        "facts": student_facts,
        "strict_cot": strict_cots[0] if strict_cots else {},
        "strict_cots_all": strict_cots
    }
    return ordered_result
