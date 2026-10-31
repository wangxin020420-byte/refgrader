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
import sys

import httpx
from openai import OpenAI

BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
TEXT_MODEL = "glm-5.2"
VISION_MODEL = "glm-4.6v"
_KINDS = {"present", "missing", "unclear"}
_CATEGORIES = {"match", "partial", "mismatch", "missing", "unclear"}
_JSON_ATTEMPTS = 3


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


def _request_json(client, *, stage, **request):
    """Retry malformed JSON only; never relax the downstream schema checks.

    An unsuccessful request may still consume quota. Transport failures retain
    the SDK's separate retry policy. No response text or credentials are logged.
    """
    for attempt in range(1, _JSON_ATTEMPTS + 1):
        response = client.chat.completions.create(**request)
        try:
            return _content(response, stage=stage)
        except ValueError as exc:
            if attempt == _JSON_ATTEMPTS:
                raise ValueError(f"{exc}; JSON attempts exhausted={_JSON_ATTEMPTS}") from exc
            print(f"{stage}: invalid JSON; retry {attempt + 1}/{_JSON_ATTEMPTS}",
                  file=sys.stderr, flush=True)


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


def _reference_transcription(client, image, item_ids=None):
    instruction = ('Transcribe the reference-answer image faithfully. Do not infer missing text. '
                   'Return JSON exactly as {"reference_text":"..."}.')
    if item_ids is not None:
        instruction = ('Transcribe the reference-answer image faithfully for the listed rubric items. '
                       'Do not infer missing text. Return JSON exactly as {"reference_text":"..."}. '
                       'Item IDs: ' + json.dumps(item_ids))
    try:
        payload = _request_json(client, stage='reference_image', model=VISION_MODEL,
            messages=[{'role': 'user', 'content': [{'type': 'text', 'text': instruction}, image]}],
            temperature=0, response_format={'type': 'json_object'})
    except ValueError as exc:
        if ('reference_image: model returned invalid JSON (finish_reason=stop' not in str(exc) or
                'JSON attempts exhausted=3' not in str(exc)):
            raise
        prompt = ('Transcribe only the visible contents of this reference-answer image faithfully, '
                  'including program symbols and quotes. Do not infer missing content. '
                  'Return only the transcription between two markers on their own lines: '
                  '<<<BEGIN_REFERENCE>>> and <<<END_REFERENCE>>>. No other text.')
        if item_ids is not None:
            prompt += ' Item IDs: ' + json.dumps(item_ids)
        response = client.chat.completions.create(model=VISION_MODEL, temperature=0,
            messages=[{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, image]}])
        if not response.choices or response.choices[0].finish_reason != 'stop':
            raise ValueError('reference_image: incomplete fallback response')
        raw = response.choices[0].message.content
        if not isinstance(raw, str): raise ValueError('reference_image: fallback response is not text')
        match = re.fullmatch(r'\s*<<<BEGIN_REFERENCE>>>[ \t]*\r?\n([\s\S]*?)\r?\n<<<END_REFERENCE>>>[ \t]*\s*', raw)
        if not match or not match.group(1).strip() or '<<<BEGIN_REFERENCE>>>' in match.group(1) or '<<<END_REFERENCE>>>' in match.group(1):
            raise ValueError('reference_image: invalid fallback markers or empty transcription')
        return match.group(1), True
    if set(payload) != {'reference_text'} or not isinstance(payload['reference_text'], str) or not payload['reference_text'].strip():
        raise ValueError('reference_image: invalid visual reference transcription')
    return payload['reference_text'], False


def grade_one(client, context, image_path, *, question_image=None, reference_image=None, reference_images=None):
    """Extract visual evidence once and grade it three times with no labels.

    Inputs must be separately audited; evidence and model judgments can be wrong.
    The three risks are engineering signals, not calibrated error probabilities.
    """
    rubric = _rubric(context["initial_rubric"])
    total = sum(item["points"] for item in rubric)
    if not math.isfinite(float(context["total_score"])) or abs(total - float(context["total_score"])) > 1e-6:
        raise ValueError("Rubric total differs from question total")
    if reference_image is not None and reference_images is not None:
        raise ValueError("Choose one reference image interface")
    multi_references = []
    reference_fallbacks = []
    if reference_images is not None:
        if not isinstance(reference_images, list):
            raise ValueError("reference_images must be a list")
        valid_ids = {item["id"] for item in rubric}
        for entry in reference_images:
            if not isinstance(entry, dict) or set(entry) != {"path", "item_ids"}:
                raise ValueError("Invalid reference image mapping")
            item_ids = entry["item_ids"]
            if not isinstance(item_ids, list) or not item_ids or any(not isinstance(i, str) or i not in valid_ids for i in item_ids) or len(set(item_ids)) != len(item_ids):
                raise ValueError("Invalid reference image item IDs")
            # Validate all paths before issuing any model request.
            _image(entry["path"])
        for entry in reference_images:
            transcription, used = _reference_transcription(client, _image(entry['path']), entry['item_ids'])
            multi_references.append({'item_ids': list(entry['item_ids']), 'reference_text': transcription})
            if used: reference_fallbacks.append({'item_ids': list(entry['item_ids'])})
    reference_transcription = ""
    if reference_image is not None:
        reference_transcription, used = _reference_transcription(client, _image(reference_image))
        if used: reference_fallbacks.append({'item_ids': None})
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
    fallback = None
    try:
        payload = _request_json(client, stage="student_image", model=VISION_MODEL,
                                messages=[{"role": "user", "content": images}],
                                temperature=0, response_format={"type": "json_object"})
    except ValueError as exc:
        if (not isinstance(context.get('question_id'), str) or not context['question_id'] or
                "student_image: model returned invalid JSON (finish_reason=stop" not in str(exc) or
                "JSON attempts exhausted=3" not in str(exc)):
            raise
        from label_free_slot_fallback import extract_fallback
        facts, fallback = extract_fallback(client, context=context, rubric=rubric,
                                           images=images, vision_model=VISION_MODEL)
    else:
        try:
            facts = _validate_facts(payload, [r["id"] for r in rubric])
        except ValueError as exc:
            raise ValueError(f"student_image: {exc}") from exc
    instruction = (
        "Grade each rubric item only from the supplied evidence and reference. "
        "Do not use teacher scores or infer unwritten work. Return JSON only: "
        "{\"items\":[{\"id\":\"...\",\"score\":0,\"category\":\"match|partial|mismatch|missing|unclear\"}]}. "
        "Include each item once; score must be between 0 and its points.\nQuestion: "
        + str(context.get("question_text", "")) + "\nReference: " + str(context.get("reference_text", ""))
        + "\nReference image transcription: " + reference_transcription
        + "\nItem-linked reference image transcriptions: " + json.dumps(multi_references, ensure_ascii=False)
        + "\nRubric: " + json.dumps(rubric, ensure_ascii=False)
        + "\nEvidence: " + json.dumps(facts, ensure_ascii=False)
    )
    probes = []
    for index in range(3):
        stage = f"text_probe_{index + 1}"
        payload = _request_json(client, stage=stage, model=TEXT_MODEL,
                                messages=[{"role": "user", "content": instruction}],
                                temperature=0.35, response_format={"type": "json_object"},
                                extra_body={"thinking": {"type": "disabled"}})
        try:
            probes.append(_validate_probe(payload, rubric))
        except ValueError as exc:
            if str(exc).startswith(stage + ":"):
                raise
            raise ValueError(f"{stage}: {exc}") from exc
    scores = [sum(item["score"] for item in probe.values()) for probe in probes]
    evidence_risk = sum(r["points"] * (1 if facts[r["id"]]["status"] != "present" else 0) for r in rubric) / total
    stability_risk = min(1.0, (max(scores) - min(scores)) / total)
    mapping_risk = sum(r["points"] * (1 - max(sum(p[r["id"]]["category"] == category for p in probes) for category in _CATEGORIES) / 3) for r in rubric) / total
    result = {"baseline_score": sum(scores) / 3, "valid_probe_count": 3,
            "risks": {"U_E": evidence_risk, "U_S": stability_risk, "U_R": mapping_risk},
            "probe_scores": scores, "probes": probes, "facts": facts,
            "reference_image_transcription": reference_transcription}
    if reference_images is not None:
        result["reference_image_transcriptions"] = multi_references
    if fallback is not None:
        result["slot_fallback"] = fallback
    if reference_fallbacks:
        result['reference_fallback'] = {'protocol': 'marked_reference_transcription_v1',
                                        'images': reference_fallbacks, 'requires_human_review': True,
                                        'score_change_authorized': False}
    return result
