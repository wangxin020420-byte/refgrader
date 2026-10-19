"""Experiment-local visual transcription. Original inputs and labels remain untouched."""
import argparse
import importlib.util
import json
import re
from pathlib import Path

BASE_SHA = '17da94d1984acc0829a09de77939c8132254f4a834eb711cfaab67d56559ce8d'
CUE = re.compile(r'如图|如表|图中|表中|见图|见表|下图|下表|上图|上表|图示|表格|\[图片\]|\[图表\]')

def load_base(path):
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('Base runner hash mismatch')
    spec = importlib.util.spec_from_file_location('visual_base', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def install_plan(m, experiment):
    original = m.plan
    def plan(inputs, out):
        # Keep the original planner's hash checks, without importing any old responses.
        p = original(inputs, out / 'source_plan')
        seen = {u['unit_id'] for u in p['units']}
        for split in ('train', 'validation', 'test'):
            for r in rows(inputs / (split + '.inputs.jsonl')):
                uid = 'answer:' + r['sample_id']
                if uid in seen or r['blank_passthrough'] or not CUE.search(r['answer']):
                    continue
                raw, mime = m.image_data(r['student_image'])
                p['units'].append(dict(unit_id=uid, kind='student_answer', original_text=r['answer'],
                    image=str(Path(r['student_image']).resolve()), image_sha256=m.sha(raw), mime=mime))
                seen.add(uid)
        p['experiment_id'] = experiment
        p['prompt'] += '\n若有图表，请在对应位置展开图表的实际内容，不可仅保留如图、如表或图表标题。若图片无该图表，保留原文并在issues说明；不得补画或推测。'
        p['cache_policy'] = 'same experiment only; no cross-experiment reuse'
        dest = out / 'plan.json'
        if dest.exists() and dest.read_bytes() != m.encoded(p):
            raise ValueError('Experiment plan changed; do not mix cached responses')
        if not dest.exists():
            m.write(dest, p)
        return p
    m.plan = plan

def export(m, inputs, out):
    p = m.read(out / 'plan.json')
    fingerprint = m.sha(m.encoded(p))
    accepted, pending, issues = {}, [], []
    for u in p['units']:
        folder = out / 'responses' / m.sha(u['unit_id'].encode())
        path = folder / 'accepted.json'
        if not path.exists():
            pending.append(u['unit_id'])
            continue
        a = m.read(path)
        response = folder / a['response_file']
        if a['unit_id'] != u['unit_id'] or a['plan_sha256'] != fingerprint or m.sha(response.read_bytes()) != a['response_sha256'] or m.parse_response(m.read(response)) != a['transcription']:
            raise ValueError('Cache integrity mismatch: ' + u['unit_id'])
        accepted[u['unit_id']] = a['transcription']['text']
        if a['transcription']['issues']:
            issues.append(dict(unit_id=u['unit_id'], issues=a['transcription']['issues']))
    view = out / 'text_view'
    view.mkdir(exist_ok=True)
    unresolved_refs = []
    counts = {}
    for split in ('train', 'validation', 'test'):
        result = rows(inputs / (split + '.inputs.jsonl'))
        for r in result:
            uid = 'answer:' + r['sample_id']
            if uid in accepted:
                r['answer'] = accepted[uid]
            qid = r['question_id']
            if 'question_image:' + qid in accepted:
                r['question'] = accepted['question_image:' + qid]
            if 'reference_image:' + qid in accepted:
                if len(r['references']) == 1:
                    r['references'] = [accepted['reference_image:' + qid]]
                else:
                    unresolved_refs.append(qid)
        counts[split] = len(result)
        target = view / (split + '.inputs.jsonl')
        tmp = target.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result), encoding='utf-8')
        tmp.replace(target)
    report = dict(experiment_id=p['experiment_id'], planned_requests=len(p['units']), accepted=len(accepted),
        pending=pending, reported_issues=issues, missing_visual_context=p['missing_visual_context'],
        ambiguous_reference_questions=sorted(set(unresolved_refs)), records=counts,
        inputs_complete=not pending and not unresolved_refs, semantic_accuracy_verified=False,
        source_files_modified=False, labels_read=False, training_started=False,
        ready_for_training=False, text_view=str(view),
        note='Separate derived view. Runner integration and unresolved visual context still require checking.',
        files_sha256={x.name:m.sha(x.read_bytes()) for x in sorted(view.glob('*.inputs.jsonl'))})
    m.write(out / 'experiment_summary.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('reported_issues','pending','missing_visual_context')}, ensure_ascii=False, indent=2))
    return 2 if pending or unresolved_refs else 0

def main():
    a = argparse.ArgumentParser()
    a.add_argument('command', choices=['plan','run'])
    a.add_argument('--base-runner', type=Path, required=True)
    a.add_argument('--inputs', type=Path, required=True)
    a.add_argument('--experiments-root', type=Path, required=True)
    a.add_argument('--experiment-id', required=True)
    a.add_argument('--attempts', type=int, default=8)
    args = a.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.experiment_id) or args.attempts < 1:
        a.error('Invalid experiment ID or attempts')
    m = load_base(args.base_runner)
    install_plan(m, args.experiment_id)
    out = args.experiments_root.resolve() / args.experiment_id / 'visual'
    inputs = args.inputs.resolve()
    if args.command == 'plan':
        p = m.plan(inputs, out)
        print('EXPERIMENT_PLAN_READY', len(p['units']), out, flush=True)
        return 0
    rc = m.run(inputs, out, 0, args.attempts)
    return max(rc, export(m, inputs, out))

if __name__ == '__main__':
    raise SystemExit(main())
