"""Standalone, label-blind CSBench visual transcription; never edits source data."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import time

PROMPT = '''你是忠实转写员，不是解题或评分助手。仅根据所给图片和该图片已有转写，完整转写原有内容。
保留作者的错误、遗漏、数值、公式和推导；不纠错、不补答案、不评价正确性。
图形用可核对的文字描述节点、连线、方向、标签及表格单元格，不推断未画出的关系。
忽略能明确识别的教师分数和批改标记；若无法区分，请在issues中说明。
看不清的内容写[无法辨认]，不得猜测。原有转写仅供比对，以图片为依据。
只输出JSON对象：{"text":"完整转写", "issues":["不确定之处；无则空数组"]}。'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def write(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(encoded(obj))
    temp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def image_data(path):
    from PIL import Image
    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        fmt = im.format
        im.verify()
    mime = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'WEBP': 'image/webp'}.get(fmt)
    if not mime:
        raise ValueError('Unsupported image format: ' + str(path))
    return raw, mime


def plan(inputs, out):
    inputs, out = inputs.resolve(), out.resolve()
    if inputs == out or inputs in out.parents:
        raise ValueError('Run directory must be outside frozen inputs')
    manifest = read(inputs / 'manifest.json')
    names = ['questions.jsonl', 'nonblank_text_review_queue.jsonl'] + [s + '.inputs.jsonl' for s in ('train', 'validation', 'test')]
    data = {}
    for name in names:
        raw = (inputs / name).read_bytes()
        if sha(raw) != manifest['files_sha256'][name]:
            raise ValueError('Input hash mismatch: ' + name)
        data[name] = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    rows = {r['sample_id']: r for s in ('train', 'validation', 'test') for r in data[s + '.inputs.jsonl']}
    units, missing = [], []

    def add(uid, kind, text, path):
        raw, mime = image_data(path)
        units.append(dict(unit_id=uid, kind=kind, original_text=text, image=str(Path(path).resolve()), image_sha256=sha(raw), mime=mime))

    for r in data['nonblank_text_review_queue.jsonl']:
        source = rows[r['sample_id']]
        if source['blank_passthrough'] or not source['answer'].strip():
            raise ValueError('Blank answer unexpectedly queued')
        if source['answer'] != r['answer'] or source['student_image'] != r['student_image']:
            raise ValueError('Review queue mismatch')
        add('answer:' + r['sample_id'], 'student_answer', r['answer'], r['student_image'])
    for q in data['questions.jsonl']:
        if not q['visual_context_review_required']:
            continue
        for field, text in [('question_image', q['question']), ('reference_image', '\n'.join(q['references']))]:
            if q.get(field) and Path(q[field]).is_file():
                add(field + ':' + q['question_id'], field, text, q[field])
            else:
                missing.append({'question_id': q['question_id'], 'field': field, 'reason': 'Visual context flagged but image unavailable; review required'})
    if len({u['unit_id'] for u in units}) != len(units):
        raise ValueError('Duplicate request IDs')
    obj = dict(input_directory=str(inputs), input_manifest_sha256=sha((inputs / 'manifest.json').read_bytes()),
               model='glm-4.6v', base_url='https://open.bigmodel.cn/api/coding/paas/v4/',
               max_tokens=8192, temperature=0.1, thinking={'type': 'disabled'}, prompt=PROMPT,
               units=units, missing_visual_context=missing, labels_read=False,
               blank_answers_skipped=sum(r['blank_passthrough'] for r in rows.values()))
    out.mkdir(parents=True, exist_ok=True)
    dest = out / 'plan.json'
    if dest.exists() and dest.read_bytes() != encoded(obj):
        raise ValueError('Existing plan differs; do not mix requests')
    if not dest.exists():
        write(dest, obj)
    print('PLAN_READY', len(units), 'requests; missing context:', len(missing), flush=True)
    return obj


def parse_response(response):
    c = response['choices'][0]
    if c['finish_reason'] != 'stop':
        raise ValueError('Incomplete response: ' + str(c['finish_reason']))
    text = c['message'].get('content') or ''
    if text.strip().startswith('```'):
        text = '\n'.join(text.strip().splitlines()[1:-1])
    obj = json.loads(text)
    if not isinstance(obj.get('text'), str) or not obj['text'].strip():
        raise ValueError('Empty transcription')
    if not isinstance(obj.get('issues'), list) or any(not isinstance(x, str) for x in obj['issues']):
        raise ValueError('Invalid issues list')
    return {'text': obj['text'], 'issues': obj['issues']}


def request_args(p, u, raw):
    return dict(model=p['model'], max_tokens=p['max_tokens'], temperature=p['temperature'],
                extra_body={'thinking': p['thinking']}, messages=[
        {'role': 'system', 'content': p['prompt']},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': '图片类别：' + u['kind'] + '\n该图片已有转写：\n' + u['original_text']},
            {'type': 'image_url', 'image_url': {'url': 'data:' + u['mime'] + ';base64,' + base64.b64encode(raw).decode('ascii')}}]}])


def run(inputs, out, limit, attempts, client=None, sleep=time.sleep):
    p = plan(inputs, out)
    fingerprint = sha(encoded(p))
    if client is None:
        from openai import OpenAI
        key = os.environ.get('ZHIPUAI_API_KEY') or os.environ.get('ZHIPU_API_KEY')
        if not key:
            raise ValueError('Set ZHIPUAI_API_KEY through hidden terminal input')
        client = OpenAI(api_key=key, base_url=p['base_url'], timeout=900, max_retries=0)
    cache = out / 'responses'
    cache.mkdir(exist_ok=True)
    valid, errors, calls, processed = [], [], 0, 0
    fatal = None
    for u in p['units']:
        folder = cache / sha(u['unit_id'].encode())
        folder.mkdir(exist_ok=True)
        accepted = folder / 'accepted.json'
        if accepted.exists():
            saved = read(accepted)
            envelope = folder / saved['response_file']
            if saved['plan_sha256'] != fingerprint or saved['unit_id'] != u['unit_id'] or sha(envelope.read_bytes()) != saved['response_sha256'] or parse_response(read(envelope)) != saved['transcription']:
                raise ValueError('Cache integrity mismatch: ' + u['unit_id'])
            valid.append(saved)
            continue
        if fatal or (limit and processed >= limit):
            continue
        processed += 1
        raw, _ = image_data(u['image'])
        if sha(raw) != u['image_sha256']:
            raise ValueError('Image changed')
        for attempt in range(attempts):
            token = str(time.time_ns())
            try:
                calls += 1
                response = client.chat.completions.create(**request_args(p, u, raw)).model_dump(mode='json')
                response_file = token + '.response.json'
                write(folder / response_file, response)
                parsed = parse_response(response)
                saved = dict(unit_id=u['unit_id'], plan_sha256=fingerprint, response_file=response_file,
                             response_sha256=sha((folder / response_file).read_bytes()), transcription=parsed)
                write(accepted, saved)
                valid.append(saved)
                break
            except Exception as e:
                status = getattr(e, 'status_code', None)
                # Never persist exception bodies, which may contain credentials or request data.
                error = dict(unit_id=u['unit_id'], exception_type=type(e).__name__, http_status=status, attempt=attempt + 1)
                write(folder / (token + '.error.json'), error)
                if status in (400, 401, 403, 404, 422):
                    fatal = error
                if fatal or attempt + 1 == attempts:
                    errors.append(error)
                    break
                sleep(min(60, 5 * 2 ** attempt))
        print('TRANSCRIPTION', len(valid), '/', len(p['units']), 'NEW_CALLS', calls, flush=True)
    review = []
    by_id = {u['unit_id']: u for u in p['units']}
    for v in valid:
        u = by_id[v['unit_id']]
        review.append(dict(unit_id=u['unit_id'], kind=u['kind'], image=u['image'], original_text=u['original_text'],
                           **v['transcription'], review_status='PENDING'))
    (out / 'review.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in review), encoding='utf-8')
    remaining = len(p['units']) - len(valid)
    summary = dict(status='TRANSCRIPTION_PARTIAL' if remaining else 'TRANSCRIPTION_COMPLETE_REVIEW_REQUIRED',
                   planned_requests=len(p['units']), valid_responses=len(valid), remaining=remaining,
                   new_api_calls=calls, errors=errors, missing_visual_context=p['missing_visual_context'],
                   model=p['model'], plan_sha256=fingerprint, labels_read=False, training_started=False,
                   semantic_accuracy_verified=False, ready_for_training=False,
                   records_with_reported_issues=sum(bool(x['issues']) for x in review))
    write(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if errors or (remaining and not limit) else 0


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('command', choices=['plan', 'run'])
    a.add_argument('--inputs', required=True, type=Path)
    a.add_argument('--run-dir', required=True, type=Path)
    a.add_argument('--limit', type=int, default=0)
    a.add_argument('--attempts', type=int, default=8)
    args = a.parse_args()
    if args.attempts < 1 or args.limit < 0:
        a.error('attempts must be positive; limit must be nonnegative')
    if args.command == 'plan':
        plan(args.inputs, args.run_dir)
    else:
        raise SystemExit(run(args.inputs, args.run_dir, args.limit, args.attempts))
