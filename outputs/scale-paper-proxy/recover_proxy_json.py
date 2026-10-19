"""Offline, narrowly scoped recovery of four inspected proxy JSON responses."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time


PHRASES = {
    '62': '"若仅知道指针p指向某节点，不知道头指针"',
    '515': None,
    '552': '"一次开辟、永久使用"',
    '1081': '"一次开辟，永久使用"',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def recover_content(content, number):
    phrase = PHRASES[number]
    if phrase is not None:
        if content.count(phrase) != 1:
            raise ValueError('Expected quoted phrase exactly once: ' + number)
        # Escape only the two inspected quotation marks; preserve their text.
        content = content.replace(phrase, json.dumps(phrase, ensure_ascii=False)[1:-1], 1)
    # Permit literal control characters in strings, then serialize them legally.
    # Invalid escapes, missing text, extra fields and truncated output still fail.
    obj = json.loads(content, strict=False)
    if not isinstance(obj, dict) or set(obj) != {'proxy_answer'}:
        raise ValueError('Expected proxy_answer only')
    if not isinstance(obj['proxy_answer'], str) or not obj['proxy_answer'].strip():
        raise ValueError('Empty proxy answer')
    return json.dumps(obj, ensure_ascii=False)


def load_runner(root):
    path = root / 'control/scale-paper-proxy/proxy.py'
    spec = importlib.util.spec_from_file_location('original_proxy_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(root, apply=False):
    module = load_runner(root)
    run = root / 'paper_method_v1/runs/proxy_glm_v1'
    plan = root / 'paper_method_v1/artifacts/proxy_inputs_v1'
    config = module.read(plan / 'config.json')
    if config['script_sha256'] != module.sha(Path(module.__file__)):
        raise ValueError('Original runner changed')
    for name, expected in module.read(plan / 'summary.json')['files_sha256'].items():
        if module.sha(plan / name) != expected:
            raise ValueError('Plan changed: ' + name)
    requests = module.rows(plan / 'requests.jsonl')
    original_parse = module.parse
    overrides = {}
    audits = []
    hashes = {}
    found = set()
    for req in requests:
        path = run / 'responses' / (req['request_id'] + '.json')
        data = path.read_bytes()
        hashes[path] = digest(data)
        raw = json.loads(data)
        sid = req['sample_id']
        number = sid.rsplit(':', 1)[-1]
        target = sid == 'ADS:9e1a866d58ed:train:' + number and number in PHRASES
        if not target:
            original_parse(raw)
            continue
        found.add(number)
        choice = raw['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Incomplete response: ' + sid)
        content = choice['message']['content']
        try:
            original_parse(raw)
        except json.JSONDecodeError:
            pass
        else:
            raise ValueError('Expected original JSON syntax failure: ' + sid)
        repaired = recover_content(content, number)
        clone = copy.deepcopy(raw)
        clone['choices'][0]['message']['content'] = repaired
        parsed = original_parse(clone)
        content_hash = digest(content.encode('utf-8'))
        overrides[content_hash] = parsed
        audits.append({
            'sample_id': sid, 'request_id': req['request_id'],
            'raw_response_sha256': hashes[path],
            'original_content': content, 'normalized_content': repaired,
            'proxy_answer': parsed[0],
            'operation': 'escape_inspected_quotes_and_serialize_control_characters',
        })
        print('CHECK_OK', sid, flush=True)
    if found != set(PHRASES) or len(audits) != 4:
        raise ValueError('Expected exactly the four inspected responses')
    print('ALL_RESPONSES_PARSEABLE_WITH_RECOVERY:', len(requests), flush=True)
    if not apply:
        print('CHECK_ONLY: no files changed; no API calls.')
        return
    if shutil.disk_usage(root).free < 512 * 1024 * 1024:
        raise ValueError('Need at least 512 MiB available before writing artifacts')

    audit = {
        'version': 'proxy_json_recovery_v1', 'api_calls': 0,
        'raw_responses_modified': False, 'semantic_review_required': True,
        'wrapper_sha256': module.sha(Path(__file__)), 'repairs': audits,
    }
    # Preserve prior summaries; raw API responses are never rewritten.
    archive = run / 'json_recovery_history' / str(time.time_ns())
    archive.mkdir(parents=True, exist_ok=False)
    for name in ('summary.json', 'proxy_records.jsonl', 'review_preview.md'):
        if (run / name).is_file():
            shutil.copy2(run / name, archive / name)
    module.save(run / 'json_recovery_audit_v1.json', module.enc(audit))

    def recovered_parse(raw):
        content = raw['choices'][0]['message']['content']
        key = digest(content.encode('utf-8'))
        if key in overrides:
            if raw['choices'][0].get('finish_reason') != 'stop':
                raise ValueError('Response not complete')
            return overrides[key]
        return original_parse(raw)

    def block_network(*args, **kwargs):
        raise RuntimeError('Network is disabled in offline recovery')

    old_urlopen = module.urllib.request.urlopen
    module.parse = recovered_parse
    module.urllib.request.urlopen = block_network
    try:
        module.execute(root, offline=True, limit=None, retry_failed=False)
    finally:
        module.parse = original_parse
        module.urllib.request.urlopen = old_urlopen
    for path, expected in hashes.items():
        if module.sha(path) != expected:
            raise ValueError('Raw response changed: ' + str(path))
    report = module.read(run / 'summary.json')
    if report['status'] != 'PROXY_GENERATION_COMPLETE_REVIEW_REQUIRED':
        raise ValueError('Recovery incomplete; inspect summary')
    artifact = root / 'paper_method_v1/artifacts/proxy_generated_v1'
    module.save(artifact / 'json_recovery_audit_v1.json', module.enc(audit))
    print('PASS: offline recovery complete; raw response hashes unchanged.')
    print('AUDIT:', artifact / 'json_recovery_audit_v1.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'apply'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    execute(args.root.resolve(), apply=args.command == 'apply')
