"""Targeted proxy regeneration with unchanged original prompts; no label transfer."""
import argparse
import getpass
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import urllib.request


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def encode(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode('utf8')


def save(path, obj, mutable=False):
    data = encode(obj)
    if path.exists() and not mutable:
        if path.read_bytes() != data:
            raise ValueError('Existing file differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)


def load_inputs(root):
    base = root / 'paper_method_v1'
    plan = base / 'artifacts/proxy_inputs_v1'
    for name, expected in read(plan / 'summary.json')['files_sha256'].items():
        if sha((plan / name).read_bytes()) != expected:
            raise ValueError('Original plan changed: ' + name)
    cfg = read(plan / 'config.json')
    script = root / 'control/scale-paper-proxy/proxy.py'
    if sha(script.read_bytes()) != cfg['script_sha256']:
        raise ValueError('Original runner changed')
    spec = importlib.util.spec_from_file_location('proxy_original', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    artifact = base / 'artifacts/proxy_generated_v1'
    data = (artifact / 'proxy_records.jsonl').read_bytes()
    if sha(data) != read(artifact / 'manifest.json')['records_sha256']:
        raise ValueError('Generated records changed')
    records = [json.loads(line) for line in data.decode('utf8').splitlines() if line.strip()]
    if len(records) != 885 or len({r['sample_id'] for r in records}) != 885:
        raise ValueError('Expected 885 unique records')
    requests = module.rows(plan / 'requests.jsonl')
    return cfg, module, records, requests, sha(data)


def execute(root, command, review_path=None, limit=None):
    cfg, module, records, requests, records_sha = load_inputs(root)
    lookup = {r['sample_id']: r for r in records}
    by_id = {r['sample_id']: r for r in requests}
    out = root / 'paper_method_v1/runs/proxy_glm_v1/semantic_retry_v1'
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if command == 'prepare':
            review = read(review_path)
            entries = review['decisions']
            if len(entries) != 244 or len({r['sample_id'] for r in entries}) != 244:
                raise ValueError('Expected 244 unique reviewed pairs')
            selected = []
            for entry in entries:
                source = lookup[entry['sample_id']]
                for field, key in [('original_answer', 'original_text_sha256'),
                                   ('proxy_answer', 'proxy_text_sha256')]:
                    value = source[field].replace('\r\n', '\n').strip()
                    if sha(value.encode('utf8')) != entry[key]:
                        raise ValueError('Review text mismatch: ' + entry['sample_id'])
                if entry['decision'] not in ('PASS', 'FAIL', 'UNCERTAIN'):
                    raise ValueError('Unknown decision')
                if entry['decision'] != 'PASS':
                    selected.append(by_id[entry['sample_id']])
            if len(selected) != 12:
                raise ValueError('Expected 12 targeted retries')
            selected.sort(key=lambda r: r['sample_id'])
            plan = {'records_sha256': records_sha, 'config': cfg,
                    'wrapper_sha256': sha(Path(__file__).read_bytes()),
                    'review_sha256': sha(encode(review)), 'requests': selected,
                    'policy': 'Same original prompt and model; fresh response per flagged source. '
                              'No scores or review criticism sent to API. Results require review.'}
            save(out / 'review.json', review)
            save(out / 'plan.json', plan)
            print('SEMANTIC_RETRY_PLAN_READY; requests=12; api_calls=0')
            print('PLAN:', out / 'plan.json')
            return

        plan = read(out / 'plan.json')
        if plan['records_sha256'] != records_sha or plan['config'] != cfg:
            raise ValueError('Inputs changed since preparation')
        if plan['wrapper_sha256'] != sha(Path(__file__).read_bytes()):
            raise ValueError('Retry runner changed')
        if sha((out / 'review.json').read_bytes()) != plan['review_sha256']:
            raise ValueError('Review changed')
        expected_ids = {r['sample_id'] for r in read(out / 'review.json')['decisions']
                        if r['decision'] != 'PASS'}
        if len(plan['requests']) != 12 or {r['sample_id'] for r in plan['requests']} != expected_ids:
            raise ValueError('Targeted request set changed')
        for req in plan['requests']:
            if req != by_id[req['sample_id']]:
                raise ValueError('Request differs from original')
        calls = 0
        errors = []
        results = []
        key = None
        stop = False
        for req in plan['requests']:
            rid = req['request_id']
            if Path(rid).name != rid:
                raise ValueError('Invalid request ID')
            response = out / 'responses' / (rid + '.json')
            attempt = out / 'attempts' / (rid + '.json')
            if not response.exists() and command == 'run' and not stop and (limit is None or calls < limit):
                if attempt.exists():
                    errors.append({'sample_id': req['sample_id'], 'error': 'Previous attempt without response; inspect before retrying.'})
                    continue
                if shutil.disk_usage(root).free < 512 * 1024 * 1024:
                    raise ValueError('Less than 512 MiB free; no new API request sent')
                if key is None:
                    key = getpass.getpass('GLM API Key (hidden): ').strip()
                    if not key:
                        raise ValueError('Empty API key')
                body = {'model': cfg['model'], 'messages': req['messages'],
                        'max_tokens': cfg['max_tokens'], 'stream': cfg['stream']}
                save(attempt, {'time': time.time(), 'body_sha256': sha(encode(body))})
                request = urllib.request.Request(cfg['base_url'].rstrip('/') + '/chat/completions',
                    data=json.dumps(body).encode('utf8'),
                    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
                calls += 1
                try:
                    with urllib.request.urlopen(request, timeout=cfg['timeout_seconds']) as reply:
                        raw = json.load(reply)
                    save(response, raw)
                    print('SAVED_RETRY', req['sample_id'], flush=True)
                    time.sleep(cfg['pause_seconds'])
                except Exception as exc:
                    error = {'sample_id': req['sample_id'], 'exception': type(exc).__name__,
                             'detail': str(exc).replace(key, '[REDACTED]')}
                    save(out / 'errors' / (rid + '.json'), error)
                    errors.append(error)
                    stop = True
            if not response.exists():
                continue
            try:
                answer, fence = module.parse(read(response))
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                errors.append({'sample_id': req['sample_id'], 'error': str(exc)})
                continue
            source = lookup[req['sample_id']]
            results.append({'sample_id': req['sample_id'], 'question': source['question'],
                            'original_answer': source['original_answer'],
                            'previous_proxy_answer': source['proxy_answer'], 'proxy_answer': answer,
                            'raw_response_sha256': sha(response.read_bytes()),
                            'assigned_label': None, 'semantic_equivalence_verified': False})
        complete = len(results) == len(plan['requests']) and not errors
        report = {'status': 'SEMANTIC_RETRY_READY_FOR_REVIEW' if complete else 'SEMANTIC_RETRY_PARTIAL',
                  'planned_requests': len(plan['requests']), 'valid_responses': len(results),
                  'new_api_calls': calls, 'errors': errors, 'original_records_modified': False,
                  'labels_assigned': 0, 'training_started': False, 'test_accessed': False}
        save(out / 'candidates.json', results, mutable=True)
        save(out / 'summary.json', report, mutable=True)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print('REVIEW_FILE:', out / 'candidates.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'offline'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.command == 'prepare' and args.review is None:
        parser.error('prepare requires --review')
    if args.command != 'prepare' and args.review is not None:
        parser.error('--review applies only to prepare')
    if args.limit is not None and (args.command != 'run' or args.limit < 1):
        parser.error('--limit requires run and a positive integer')
    execute(args.root.resolve(), args.command, args.review, args.limit)
