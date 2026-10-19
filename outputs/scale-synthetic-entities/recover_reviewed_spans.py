"""Offline, reviewed two-record span alignment; immutable original API responses."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

REVIEWED = {
    'synthetic_text:4e391b15a194c85de6ee5629c03f204a82752bdba0aaccbbbdb2e63f28758071':
        '95749b0fa537f6ce23b5b3c06315a7a2ee8e5b5932c3e5c18728b0c7c44e4808',
    'synthetic_text:e7dd0862cb528c7f59ab4578c150184e222923a7b8f6c3b8faa99c2b98398bd5':
        'a0180321e48a6597c480c2cca392e0456d92cf91b927f4b87bfbe8e729d00bdd',
}


def repair(ep, m, unit, raw):
    uid = unit['unit_id']
    text = unit['text']
    content = raw['choices'][0]['message']['content']
    if uid not in REVIEWED or ep.sha(content.encode('utf8')) != REVIEWED[uid]:
        raise ValueError('Response is not the reviewed third attempt')
    if uid != 'synthetic_text:' + ep.sha(text.encode('utf8')):
        raise ValueError('Source text changed')
    if raw['choices'][0]['finish_reason'] != 'stop':
        raise ValueError('Response incomplete')
    obj = json.loads(content)
    if set(obj) != {'triples'}:
        raise ValueError('Unexpected schema')
    bad = [i for i, t in enumerate(obj['triples']) if t['t'] not in text]
    if len(bad) != 1 or obj['triples'][bad[0]]['t'] != '第二次排序':
        raise ValueError('Unreviewed mismatch')
    if text.count('第二次') != 1:
        raise ValueError('Ambiguous source span')
    corrected = copy.deepcopy(obj)
    corrected['triples'][bad[0]]['t'] = '第二次'
    record = m.validate([unit], [{'unit_id': uid, 'triples': corrected['triples']}])[0]
    start = text.index('第二次')
    audit = {'unit_id': uid, 'reviewed_content_sha256': REVIEWED[uid],
             'triple_index': bad[0], 'old_t': '第二次排序', 'new_t': '第二次',
             'source_span': [start, start + len('第二次')],
             'reason': 'Model inserted 排序; align only t to the uniquely reviewed verbatim source span.',
             'c_and_w_unchanged': True, 'source_and_labels_unchanged': True}
    return record, audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    path = root / 'control/scale-synthetic-entities/entity_prepare.py'
    spec = importlib.util.spec_from_file_location('original_entity_prepare', str(path))
    ep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ep)
    out = root / 'paper_method_v1/runs/train_entities_raw_beta_v1'
    import fcntl
    with (out / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m = ep.module(root / 'control/scale-paper-stage1/stage1.py')
        # Verify the frozen inputs and code before making derived artifacts.
        ep.prepare(root, out, m)
        units = {u['unit_id']: u for u in ep.lines(out / 'pending_units.jsonl')}
        records, audits = {}, []
        original_hashes = {}
        for uid in REVIEWED:
            response = out / 'responses' / uid.split(':', 1)[1] / 'attempt_003.response.json'
            raw = ep.read(response)
            record, audit = repair(ep, m, units[uid], raw)
            original_hashes[response] = ep.sha(response.read_bytes())
            audit['response_file'] = str(response.relative_to(root))
            audit['response_file_sha256'] = original_hashes[response]
            records[uid] = record
            audits.append(audit)
        review = out / 'reviewed_span_alignment_v1'
        ep.write(review / 'corrected_extractions.jsonl', ep.jl(list(records.values())), frozen=True)
        ep.write(review / 'audit.json', ep.enc({
            'status': 'TWO_REVIEWED_SPANS_ALIGNED', 'corrections': audits,
            'api_calls': 0, 'raw_responses_modified': False, 'labels_modified': False,
            'wrapper_sha256': ep.sha(Path(__file__).read_bytes()),
            'deviation': 'Two manually reviewed source-span alignments; reproduction quality control, not an original-paper step.'
        }), frozen=True)
        original_parse = ep.parse

        def reviewed_parse(stage, unit, raw):
            uid = unit['unit_id']
            content = raw.get('choices', [{}])[0].get('message', {}).get('content')
            if uid in REVIEWED and isinstance(content, str) and ep.sha(content.encode('utf8')) == REVIEWED[uid]:
                return repair(ep, stage, unit, raw)[0]
            return original_parse(stage, unit, raw)

        ep.parse = reviewed_parse
        status = ep.execute(argparse.Namespace(root=root, command='offline', max_attempts=3), out)
        for response, digest in original_hashes.items():
            if ep.sha(response.read_bytes()) != digest:
                raise ValueError('Original response changed')
        summary = ep.read(out / 'summary.json')
        summary['reviewed_span_alignment'] = {
            'records': 2, 'audit_file': str((review / 'audit.json').relative_to(root)),
            'audit_sha256': ep.sha((review / 'audit.json').read_bytes()),
            'corrected_extractions_sha256': ep.sha((review / 'corrected_extractions.jsonl').read_bytes())}
        ep.write(out / 'summary.json', ep.enc(summary))
        print('AUDIT_FILE:', review / 'audit.json')
        print('SUMMARY_FILE:', out / 'summary.json')
        if status == 0:
            output = out / 'training_with_entities.jsonl'
            if len(ep.lines(output)) != 2212:
                raise ValueError('Unexpected export count')
            print('PASS: 2212 records; zero remaining; zero API calls')
            print('OUTPUT_FILE:', output)
        raise SystemExit(status)


if __name__ == '__main__':
    main()
