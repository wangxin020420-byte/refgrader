"""Frozen experimental raw-beta branch and resumable Pe extraction. Python 3.8+."""
import argparse
import getpass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request


EXPECTED = '4f4f1a3b98160c0b08e111f4bc904ad93e331255cfbb1aa154ec6fd483c3d9e8'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def enc(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf8')


def jl(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True,
                             allow_nan=False) + '\n' for r in rows).encode('utf8')


def read(p):
    return json.loads(p.read_bytes())


def lines(p):
    return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]


def write(p, data, frozen=False):
    if frozen and p.exists():
        if p.read_bytes() != data:
            raise ValueError('Frozen artifact differs: ' + str(p))
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    with temp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(p)


def module(path):
    spec = importlib.util.spec_from_file_location('frozen_pe_stage1', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_plan(rows, units, extractions, m):
    """Reuse exact units for originals/references; exact text only for new answers."""
    validated = m.validate(units, extractions)
    by_uid = {u['unit_id']: u for u in units}
    existing = {r['unit_id']: r for r in validated}
    by_text = {}
    refs = {}
    for u in sorted(units, key=lambda u: u['unit_id']):
        by_text.setdefault(u['text'], u['unit_id'])
        if u['role'] == 'reference':
            refs[(u['question_id'], u['text'])] = u['unit_id']
    pending = {}
    reused = {}
    bindings = []
    seen = set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError('Duplicate training record ID')
        seen.add(row['id'])
        if not math.isfinite(row['label']):
            raise ValueError('Nonfinite label')
        text = row['answer']
        if row['kind'] == 'original_training':
            uid = 'student:' + row['sample_id']
            if by_uid[uid]['text'] != text:
                raise ValueError('Original answer changed')
        elif row['kind'] == 'synthetic_candidate':
            uid = by_text.get(text, 'synthetic_text:' + sha(text.encode('utf8')))
        else:
            raise ValueError('Unexpected record kind')
        if uid in existing:
            reused[uid] = existing[uid]
        else:
            pending[uid] = {'unit_id': uid, 'text': text}
        ref_ids = []
        for text in row['references']:
            rid = refs[(row['question_id'], text)] if text.strip() else None
            ref_ids.append(rid)
            if rid is not None:
                reused[rid] = existing[rid]
        bindings.append({'id': row['id'], 'answer_unit_id': uid,
                         'reference_unit_ids': ref_ids})
    return bindings, [pending[k] for k in sorted(pending)], [reused[k] for k in sorted(reused)]


def parse(m, unit, raw):
    choice = raw['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Response not complete')
    text = choice['message']['content'].strip()
    # Remove a whole-response Markdown wrapper only; do not repair answer semantics.
    if text.startswith('```') and text.endswith('```'):
        head, body = text.split('\n', 1)
        if head.lower() not in ('```json', '```'):
            raise ValueError('Unknown response wrapper')
        text = body[:-3].strip()
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {'triples'}:
        raise ValueError('Expected triples object')
    return m.validate([unit], [{'unit_id': unit['unit_id'], 'triples': obj['triples']}])[0]


def prepare(root, out, m):
    score = root / 'paper_method_v1/runs/candidate_annotation_v1'
    branch = score / 'raw_beta_0p2/augmented_inputs.jsonl'
    if sha(branch.read_bytes()) != EXPECTED:
        raise ValueError('Not the reviewed 2212-record branch')
    summary = read(score / 'summary.json')
    if summary['status'] != 'SCORING_AND_BETA_BRANCHES_COMPLETE':
        raise ValueError('Scoring incomplete')
    for name in ('raw_beta_0p2/augmented_inputs.jsonl', 'raw_beta_0p2/selected.jsonl'):
        if sha((score / name).read_bytes()) != summary['files_sha256'][name]:
            raise ValueError('Scoring output hash mismatch: ' + name)
    trip = root / 'paper_method_v1/artifacts/triplets'
    full = root / 'paper_method_v1/artifacts/full_graph_v1'
    stage_manifest = read(trip / 'stage1_manifest.json')
    if sha((trip / 'source_units.jsonl').read_bytes()) != stage_manifest['outputs_sha256']['source_units.jsonl']:
        raise ValueError('Original Pe sources changed')
    if sha(m.PROMPT.encode('utf8')) != stage_manifest['prompt_sha256']:
        raise ValueError('Pe prompt changed')
    if sha((full / 'extractions.jsonl').read_bytes()) != read(full / 'graph.json')['extractions_sha256']:
        raise ValueError('Original extractions hash mismatch')
    cfg_path = root / 'paper_method_v1/runs/pe_glm_full_v1/config.json'
    cfg = read(cfg_path)
    if cfg['prompt_sha256'] != stage_manifest['prompt_sha256']:
        raise ValueError('Pe config prompt mismatch')
    rows = lines(branch)
    if len(rows) != 2212 or sum(r['kind'] == 'original_training' for r in rows) != 885:
        raise ValueError('Unexpected branch counts')
    bindings, pending, reused = build_plan(rows, lines(trip / 'source_units.jsonl'),
                                           lines(full / 'extractions.jsonl'), m)
    config = {
        'status': 'EXPERIMENTAL_RAW_BETA_BRANCH_FROZEN',
        'branch': 'raw_beta_0p2', 'beta': 0.2, 'beta_scale_verified': False,
        'official_paper_branch_selected': False,
        'label_policy': 'Unmodified continuous annotator scores; no clipping or rounding',
        'training_records': 2212, 'original_records': 885, 'synthetic_records': 1327,
        'negative_synthetic_labels': sum(r['kind'] == 'synthetic_candidate' and r['label'] < 0 for r in rows),
        'model': cfg['model'], 'base_url': cfg['base_url'], 'max_tokens': cfg['max_tokens'],
        'prompt_sha256': stage_manifest['prompt_sha256'],
        'source_sha256': EXPECTED, 'original_pe_config_sha256': sha(cfg_path.read_bytes()),
        'original_extractions_sha256': sha((full / 'extractions.jsonl').read_bytes()),
        'stage1_sha256': sha(Path(m.__file__).read_bytes()),
        'runner_sha256': sha(Path(__file__).read_bytes()),
        'reuse_policy': 'Original/reference unit preserved. For new answers, exact text reuse; lexicographically first unit if multiple matches.',
        'deduplication_scope': 'API requests only; all training rows and labels retained',
        'test_accessed': False, 'validation_accessed': False,
    }
    for name, data in {'config.json': enc(config), 'training_inputs.jsonl': branch.read_bytes(),
                       'bindings.jsonl': jl(bindings), 'pending_units.jsonl': jl(pending),
                       'reused_extractions.jsonl': jl(reused)}.items():
        write(out / name, data, frozen=True)
    print(json.dumps({'status': 'ENTITY_PLAN_READY', 'training_records': len(rows),
                      'reused_units': len(reused), 'unique_new_texts': len(pending),
                      'api_calls': 0}, ensure_ascii=False), flush=True)
    return rows, bindings, pending, reused, config


def execute(args, out):
    root = args.root.resolve()
    m = module(root / 'control/scale-paper-stage1/stage1.py')
    rows, bindings, pending, reused, cfg = prepare(root, out, m)
    if args.command == 'prepare':
        return 0
    key = None
    calls = 0
    results = {r['unit_id']: r for r in reused}
    errors = []
    network_stopped = False
    for index, unit in enumerate(pending, 1):
        uid = unit['unit_id']
        folder = out / 'responses' / uid.split(':', 1)[1]
        if not unit['text'].strip():
            results[uid] = {'unit_id': uid, 'triples': []}
            continue
        failures = []
        for path in sorted(folder.glob('attempt_*.response.json')):
            try:
                results[uid] = parse(m, unit, read(path))
                break
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                failures.append(str(exc))
        if uid in results:
            continue
        attempts = len(list(folder.glob('attempt_*.request.json')))
        while args.command == 'run' and not network_stopped and attempts < args.max_attempts:
            if shutil.disk_usage(out).free < 512 * 1024 * 1024:
                failures.append('DISK_GUARD: less than 512 MiB free')
                network_stopped = True
                break
            if key is None:
                key = getpass.getpass('GLM API Key (hidden): ').strip()
                if not key:
                    raise ValueError('Empty key')
            body = {'model': cfg['model'], 'max_tokens': cfg['max_tokens'], 'stream': False,
                    'messages': [{'role': 'system', 'content': m.PROMPT},
                                 {'role': 'user', 'content': json.dumps({'text': unit['text']}, ensure_ascii=False)}]}
            attempts += 1
            prefix = folder / ('attempt_%03d' % attempts)
            write(Path(str(prefix) + '.request.json'), enc({'unit_id': uid, 'body': body}), frozen=True)
            req = urllib.request.Request(cfg['base_url'].rstrip('/') + '/chat/completions',
                                         data=enc(body), headers={'Content-Type': 'application/json',
                                                                  'Authorization': 'Bearer ' + key})
            calls += 1
            try:
                with urllib.request.urlopen(req, timeout=180) as response:
                    content = response.read()
                write(Path(str(prefix) + '.response.json'), content, frozen=True)
                results[uid] = parse(m, unit, json.loads(content))
                print('SAVED_ENTITY %d/%d %s' % (index, len(pending), uid), flush=True)
                time.sleep(1)
                break
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                failures.append(str(exc))
                write(Path(str(prefix) + '.error.json'), enc({'error': str(exc), 'kind': 'SCHEMA'}))
                time.sleep(1)
            except (OSError, urllib.error.URLError) as exc:
                # Stop network calls on transport/quota/disk failure; retain all completed units.
                info = {'kind': type(exc).__name__, 'http_status': getattr(exc, 'code', None)}
                if isinstance(exc, urllib.error.HTTPError):
                    info['body'] = exc.read().decode('utf8', errors='replace')
                write(Path(str(prefix) + '.error.json'), enc(info))
                failures.append(info)
                network_stopped = True
                break
        if uid not in results:
            errors.append({'unit_id': uid, 'attempts': attempts, 'errors': failures or ['missing_response']})
    complete = not errors
    summary = {'status': 'TRAIN_ENTITY_EXTRACTIONS_COMPLETE' if complete else 'TRAIN_ENTITY_EXTRACTIONS_PARTIAL',
               'training_records': len(rows), 'unique_new_texts': len(pending),
               'valid_new_texts': sum(u['unit_id'] in results for u in pending),
               'remaining': len(errors), 'errors': errors, 'new_api_calls': calls,
               'labels_modified': False, 'test_accessed': False, 'validation_accessed': False,
               'final_scale_training_started': False, 'ready_for_scale_training': False,
               'next_step': 'Entity token span preparation and SAE architecture verification',
               'experiment_status': 'EXPERIMENTAL_REPRODUCTION_BETA_SCALE_UNVERIFIED'}
    if complete:
        exports = []
        for row, binding in zip(rows, bindings):
            exports.append(dict(row, answer_triples=results[binding['answer_unit_id']]['triples'],
                                reference_triples=[results[rid]['triples'] if rid else []
                                                   for rid in binding['reference_unit_ids']]))
        files = {'extractions.jsonl': jl([results[k] for k in sorted(results)]),
                 'training_with_entities.jsonl': jl(exports)}
        for name, data in files.items():
            write(out / name, data, frozen=True)
        summary['files_sha256'] = {name: sha(data) for name, data in files.items()}
    write(out / 'summary.json', enc(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('\aENTITY_STAGE_DONE' if complete else '\aENTITY_STAGE_PARTIAL', flush=True)
    return 0 if complete else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'run', 'offline'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--max-attempts', type=int, default=2,
                   help='Total attempt cap per new text across resumes, including failed attempts')
    args = p.parse_args()
    if args.max_attempts < 1:
        p.error('--max-attempts must be positive')
    out = args.root.resolve() / 'paper_method_v1/runs/train_entities_raw_beta_v1'
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(execute(args, out))


if __name__ == '__main__':
    main()
