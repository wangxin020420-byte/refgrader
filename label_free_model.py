"""Independent experimental model probes; no teacher-score input.

This is not a formal runner. It does not establish dataset provenance or
validate image contents. Loss parameters and routing are supplied separately.
"""

import base64
import hashlib
import json
import math
import mimetypes
from pathlib import Path
import re

import httpx
from openai import OpenAI

BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
TEXT_MODEL = "glm-5.2"
VISION_MODEL = "glm-4.6v"
_KINDS = {"present", "missing", "unclear"}
_CATEGORIES = {"match", "partial", "mismatch", "missing", "unclear"}


def create_client(api_key, *, http_client=None):
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Coding Plan API key is required")
    transport = http_client if http_client is not None else httpx.Client(trust_env=False)
    return OpenAI(api_key=api_key, base_url=BASE_URL, http_client=transport, max_retries=2)


def _content(response, *, stage):
    """Parse one JSON object, allowing an enclosing Markdown code fence.

    Diagnostics intentionally omit the model content and any answer text.
    """
    value = None
    finish = None
    try:
        choice = response.choices[0]
        value = choice.message.content
        finish = getattr(choice, "finish_reason", None)
        if not isinstance(value, str):
            raise TypeError("non-text content")
        text = value.strip().lstrip("\ufeff").strip()
        fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()
        result = json.loads(text)
    except (AttributeError, IndexError, TypeError, json.JSONDecodeError) as exc:
        safe_finish = finish if finish in {"stop", "length", "content_filter", "tool_calls"} else "unknown"
        chars = len(value) if isinstance(value, str) else 0
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12] if isinstance(value, str) else "none"
        raise ValueError(f"{stage}: model returned invalid JSON (finish_reason={safe_finish}, content_type={type(value).__name__}, chars={chars}, sha256={digest})") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{stage}: model response must be a JSON object")
    return result


def _rubric(rubric):
    if not isinstance(rubric, list) or not rubric:
        raise ValueError("Initial rubric must be a non-empty list")
    result, ids = [], set()
    for item in rubric:
        if not isinstance(item, dict):
            raise ValueError("Invalid rubric item")
        ident = item.get("id")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError("Rubric IDs must be unique")
        value = item.get("points")
        if isinstance(value, bool):
            raise ValueError("Invalid item points")
        try:
            points = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid item points") from exc
        if not math.isfinite(points) or points < 0:
            raise ValueError("Invalid item points")
        ids.add(ident)
        result.append({"id": ident, "item": str(item.get("item") or ""),
                       "points": points, "standard_answer_text": str(item.get("standard_answer_text") or "")})
    if sum(item["points"] for item in result) <= 0:
        raise ValueError("Rubric total must be positive")
    return result


def _image(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/png", "image/jpeg"} or not path.is_file():
        raise ValueError("Expected a PNG or JPEG image")
    return {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")}}


def _validate_facts(payload, ids):
    if set(payload) != {"items"} or not isinstance(payload["items"], list):
        raise ValueError("Invalid extraction contract")
    facts = {}
    for item in payload["items"]:
        if not isinstance(item, dict) or set(item) != {"id", "status", "evidence"}:
            raise ValueError("Invalid extraction item")
        ident, status, evidence = item["id"], item["status"], item["evidence"]
        if ident not in ids or ident in facts or status not in _KINDS or not isinstance(evidence, str):
            raise ValueError("Invalid extraction item values")
        if status == "present" and not evidence.strip():
            raise ValueError("Present evidence must be nonempty")
        facts[ident] = {"status": status, "evidence": evidence}
    if set(facts) != set(ids):
        raise ValueError("Extraction items do not match rubric")
    return facts


def _validate_probe(payload, rubric):
    if set(payload) != {"items"} or not isinstance(payload["items"], list):
        raise ValueError("Invalid grading contract")
    limits = {x["id"]: x["points"] for x in rubric}
    details = {}
    for item in payload["items"]:
        if not isinstance(item, dict) or set(item) != {"id", "score", "category"}:
            raise ValueError("Invalid grading item")
        ident = item["id"]
        if ident not in limits or ident in details or item["category"] not in _CATEGORIES or isinstance(item["score"], bool):
            raise ValueError("Invalid grading item values")
        try:
            score = float(item["score"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid grading score") from exc
        if not math.isfinite(score) or not 0 <= score <= limits[ident]:
            raise ValueError("Item score outside rubric bounds")
        details[ident] = {"score": score, "category": item["category"]}
    if set(details) != set(limits):
        raise ValueError("Grading items do not match rubric")
    return details


def grade_one(client, context, image_path, *, question_image=None, reference_image=None):
    """Extract visual evidence once and grade it three times with no labels.

    Inputs must be separately audited; evidence and model judgments can be wrong.
    The three risks are engineering signals, not calibrated error probabilities.
    """
    rubric = _rubric(context["initial_rubric"])
    total = sum(item["points"] for item in rubric)
    if not math.isfinite(float(context["total_score"])) or abs(total - float(context["total_score"])) > 1e-6:
        raise ValueError("Rubric total differs from question total")
    reference_transcription = ""
    if reference_image is not None:
        reference_response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Transcribe the reference-answer image faithfully. Do not infer missing text. Return JSON exactly as {\"reference_text\":\"...\"}."},
                _image(reference_image),
            ]}], temperature=0, response_format={"type": "json_object"})
        reference_payload = _content(reference_response, stage="reference_image")
        if set(reference_payload) != {"reference_text"} or not isinstance(reference_payload["reference_text"], str) or not reference_payload["reference_text"].strip():
            raise ValueError("reference_image: invalid visual reference transcription")
        reference_transcription = reference_payload["reference_text"]
    extract_instruction = (
        "Extract only what the student visibly wrote for each requested item. "
        "Do not grade, infer missing content or use any marks as scores. "
        "Return JSON: {\"items\":[{\"id\":\"...\",\"status\":\"present|missing|unclear\",\"evidence\":\"...\"}]}; "
        "include each item exactly once. Images are identified by accompanying text.\nQuestion: "
        + str(context.get("question_text", "")) + "\nItems: "
        + json.dumps([{"id": r["id"], "item": r["item"]} for r in rubric], ensure_ascii=False)
    )
    images = [{"type": "text", "text": extract_instruction}, {"type": "text", "text": "Student answer:"}, _image(image_path)]
    if question_image is not None:
        images += [{"type": "text", "text": "Question image:"}, _image(question_image)]
    response = client.chat.completions.create(model=VISION_MODEL, messages=[{"role": "user", "content": images}],
                                               temperature=0, response_format={"type": "json_object"})
    try:
        facts = _validate_facts(_content(response, stage="student_image"), [r["id"] for r in rubric])
    except ValueError as exc:
        if str(exc).startswith("student_image:"):
            raise
        raise ValueError(f"student_image: {exc}") from exc
    instruction = (
        "Grade each rubric item only from the supplied evidence and reference. "
        "Do not use teacher scores or infer unwritten work. Return JSON only: "
        "{\"items\":[{\"id\":\"...\",\"score\":0,\"category\":\"match|partial|mismatch|missing|unclear\"}]}. "
        "Include each item once; score must be between 0 and its points.\nQuestion: "
        + str(context.get("question_text", "")) + "\nReference: " + str(context.get("reference_text", ""))
        + "\nReference image transcription: " + reference_transcription
        + "\nRubric: " + json.dumps(rubric, ensure_ascii=False)
        + "\nEvidence: " + json.dumps(facts, ensure_ascii=False)
    )
    probes = []
    for index in range(3):
        response = client.chat.completions.create(model=TEXT_MODEL, messages=[{"role": "user", "content": instruction}],
                                                   temperature=0.35, response_format={"type": "json_object"},
                                                   extra_body={"thinking": {"type": "disabled"}})
        stage = f"text_probe_{index + 1}"
        try:
            probes.append(_validate_probe(_content(response, stage=stage), rubric))
        except ValueError as exc:
            if str(exc).startswith(stage + ":"):
                raise
            raise ValueError(f"{stage}: {exc}") from exc
    scores = [sum(item["score"] for item in probe.values()) for probe in probes]
    evidence_risk = sum(r["points"] * (1 if facts[r["id"]]["status"] != "present" else 0) for r in rubric) / total
    stability_risk = min(1.0, (max(scores) - min(scores)) / total)
    mapping_risk = sum(r["points"] * (1 - max(sum(p[r["id"]]["category"] == category for p in probes) for category in _CATEGORIES) / 3) for r in rubric) / total
    return {"baseline_score": sum(scores) / 3, "valid_probe_count": 3,
            "risks": {"U_E": evidence_risk, "U_S": stability_risk, "U_R": mapping_risk},
            "probe_scores": scores, "probes": probes, "facts": facts,
            "reference_image_transcription": reference_transcription}
