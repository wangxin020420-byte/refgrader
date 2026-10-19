"""Offline coverage and review inventory. No API, labels, or source mutation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

TRIGGER = re.compile(r'如\s*(?:下\s*)?(?:图|表)|见\s*(?:下\s*)?(?:图|表)|(?:图|表)\s*(?:中|内|所示)|如下所示|[【\[]\s*(?:图片|图表|表格)')
UNCERTAIN = re.compile(r'无法辨认|无法识别|看不清|不清晰|无法确定|难以辨认|模糊')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def parsed(response):
    c = response['choices'][0]
    if c['finish_reason'] != 'stop':
        raise ValueError('Incomplete cached response')
    text = c['message'].get('content') or ''
    if text.strip().startswith('```'):
        text = '\n'.join(text.strip().splitlines()[1:-1])
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        if not exc.msg.startswith('Invalid control character'):
            raise
        obj = json.loads(text, strict=False)
    if not isinstance(obj, dict) or not isinstance(obj.get('text'), str) or not obj['text'].strip():
        raise ValueError('Invalid transcription')
    if not isinstance(obj.get('issues'), list) or any(not isinstance(v, str) for v in obj['issues']):
        raise ValueError('Invalid issues')
    return dict(text=obj['text'], issues=obj['issues'])


def flags(original, transcription):
    result = []
    if transcription['issues']:
        result.append('MODEL_REPORTED_ISSUES')
    if TRIGGER.search(transcription['text']):
        result.append('VISUAL_MARKER_REMAINS_CHECK_EXPANSION')
    if UNCERTAIN.search(transcription['text']):
        result.append('UNREADABLE_CONTENT')
    if TRIGGER.search(original) and original.strip() == transcription['text'].strip():
        result.append('UNCHANGED_VISUAL_PLACEHOLDER_TEXT')
    return result


def audit(inputs, run, out, enforce_counts=True):
    inputs, run, out = inputs.resolve(), run.resolve(), out.resolve()
    if out == inputs or inputs in out.parents or out == run or run in out.parents:
        raise ValueError('Audit output must be separate from frozen input and run directories')
    manifest = read(inputs / 'manifest.json')
    verified = {}
    for name in ('train.inputs.jsonl', 'validation.inputs.jsonl', 'test.inputs.jsonl', 'questions.jsonl'):
        raw = (inputs / name).read_bytes()
        if digest(raw) != manifest['files_sha256'][name]:
            raise ValueError('Input hash mismatch: ' + name)
        verified[name] = digest(raw)
    answer_map, counts, blanks = {}, {}, Counter()
    for split in ('train', 'validation', 'test'):
        data = rows(inputs / (split + '.inputs.jsonl'))
        counts[split] = len(data)
        for r in data:
            if r['sample_id'] in answer_map:
                raise ValueError('Duplicate answer ID')
            if bool(r['blank_passthrough']) != (not r['answer'].strip()):
                raise ValueError('Blank policy mismatch')
            blanks[split] += bool(r['blank_passthrough'])
            answer_map[r['sample_id']] = dict(r, split=split)
    if enforce_counts and (counts != dict(train=345, validation=343, test=2723) or sum(blanks.values()) != 66):
        raise ValueError('Unexpected CSBench counts')
    question_map = {q['question_id']: q for q in rows(inputs / 'questions.jsonl')}
    p = read(run / 'plan.json')
    if p['input_directory'] != str(inputs) or p['input_manifest_sha256'] != digest((inputs / 'manifest.json').read_bytes()):
        raise ValueError('Plan does not match frozen inputs')
    if p['base_url'] != 'https://open.bigmodel.cn/api/coding/paas/v4/' or p['model'] != 'glm-4.6v':
        raise ValueError('Unexpected API configuration')
    fingerprint = digest(encoded(p))
    units = {u['unit_id']: u for u in p['units']}
    if len(units) != len(p['units']):
        raise ValueError('Duplicate plan units')
    review, blocked, pending, failures = [], [], [], []
    for uid, u in units.items():
        if u['kind'] == 'student_answer':
            aid = uid.removeprefix('answer:')
            r = answer_map[aid]
            if r['blank_passthrough'] or u['original_text'] != r['answer']:
                raise ValueError('Student request/source mismatch: ' + uid)
            qid, split = r['question_id'], r['split']
        else:
            qid, split = uid.split(':', 1)[1], 'shared_question_material'
        item = dict(unit_id=uid, question_id=qid, split=split, kind=u['kind'],
                    image=u['image'], original_text=u['original_text'])
        folder = run / 'responses' / digest(uid.encode())
        cache = folder / 'accepted.json'
        if cache.exists():
            try:
                s = read(cache)
                name = s['response_file']
                if Path(name).name != name:
                    raise ValueError('Invalid response filename')
                envelope = folder / name
                if s['unit_id'] != uid or s['plan_sha256'] != fingerprint or digest(envelope.read_bytes()) != s['response_sha256']:
                    raise ValueError('Accepted cache integrity mismatch')
                t = parsed(read(envelope))
                if t != s['transcription']:
                    raise ValueError('Parsed transcription differs from cache')
                if digest(Path(u['image']).read_bytes()) != u['image_sha256']:
                    raise ValueError('Source image changed')
                review.append(dict(item, transcription=t['text'], issues=t['issues'],
                    flags=flags(u['original_text'], t), review_status='PENDING',
                    reviewed_text='', reviewer='', review_notes='',
                    accepted_sha256=digest(cache.read_bytes()), response_sha256=s['response_sha256']))
            except Exception as exc:
                failures.append(dict(item, error_type=type(exc).__name__, message=str(exc)))
        else:
            is_blocked = any(str(read(f).get('service_code')) == '1301' for f in folder.glob('*.error.json'))
            entry = dict(item, manual_transcription='', reviewer='', review_notes='', review_status='PENDING')
            (blocked if is_blocked else pending).append(entry)
    coverage, uncovered = [], []
    for aid, r in answer_map.items():
        triggered = bool(TRIGGER.search(r['answer']))
        uid = 'answer:' + aid
        coverage.append(dict(sample_id=aid, question_id=r['question_id'], split=r['split'],
                             blank=r['blank_passthrough'], text_trigger=triggered, in_plan=uid in units))
        if not r['blank_passthrough'] and uid not in units and (triggered or r.get('text_review_required')):
            uncovered.append(dict(unit_id=uid, question_id=r['question_id'], split=r['split'],
                image=r['student_image'], original_text=r['answer'],
                reason='VISUAL_CUE_WITHOUT_REQUEST', decision='PENDING', notes=''))
    missing = []
    for entry in p['missing_visual_context']:
        q = question_map[entry['question_id']]
        path = q.get(entry['field'])
        missing.append(dict(entry, configured_path=path, exists_now=bool(path and Path(path).is_file()),
            question=q['question'], references=q['references'],
            resolution='PENDING', corrected_image_path='', reviewer='', notes=''))
    priority = [r for r in review if r['flags']]
    summary = dict(status='VISUAL_TRANSCRIPTION_REVIEW_INVENTORY_READY', counts=counts,
        blank_counts=dict(blanks), planned_requests=len(units), accepted_checked=len(review),
        accepted_with_model_issues=sum(bool(r['issues']) for r in review),
        priority_review_records=len(priority), priority_flag_counts=dict(Counter(f for r in review for f in r['flags'])),
        blocked_1301=len(blocked), other_pending=len(pending), cache_errors=len(failures),
        uncovered_visual_cues=len(uncovered), missing_visual_context=len(missing),
        api_calls=0, labels_read=False, source_files_modified=False, training_started=False,
        ready_for_training=False, semantic_accuracy_verified=False,
        note='Flags are review hints, not proof of omission. No transcription is automatically approved.',
        input_sha256=verified, plan_sha256=fingerprint, builder_sha256=digest(Path(__file__).read_bytes()))
    payload = {}
    for name, data in [('coverage', coverage), ('all_transcription_review', review),
                       ('priority_review', priority), ('manual_1301', blocked), ('other_pending', pending),
                       ('uncovered_visual_cues', uncovered), ('missing_visual_context', missing), ('cache_errors', failures)]:
        payload[name + '.jsonl'] = ''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in data).encode('utf-8')
    payload['summary.json'] = encoded(summary)
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in payload.items():
        (out / name).write_bytes(raw)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('input_sha256',)}, ensure_ascii=False, indent=2))
    print('OUTPUT:', out)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.inputs, args.run_dir, args.out)
    raise SystemExit(2 if result['cache_errors'] else 0)
