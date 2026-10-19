"""Freeze a documented proxy/original mixture; do not train or access test data."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

EXPECTED = 'c40fa6b7f9ea20d4c2f8cb86adf8d32e51719f0ab1d8321d25f15d4ff2df6dcf'
ACCEPT = {'1095', '144', '389', '488', '535'}
FALLBACK = {'551', '620', '668', '672', '733', '739', '745'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def enc(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode('utf8')


def norm(text):
    return text.replace('\r\n', '\n').strip()


def build(records, sources, candidates, review):
    by_id = {r['sample_id']: r for r in records}
    source_by_id = {r['sample_id']: r for r in sources}
    if len(records) != 885 or len(by_id) != 885 or set(by_id) != set(source_by_id) or len(sources) != 885:
        raise ValueError('Expected the same 885 unique training records')
    prefix = 'ADS:9e1a866d58ed:train:'
    target_ids = {prefix + n for n in ACCEPT | FALLBACK}
    new = {r['sample_id']: r for r in candidates}
    if len(candidates) != 12 or set(new) != target_ids:
        raise ValueError('Retry set differs')
    decisions = {r['sample_id']: r for r in review['decisions']}
    if len(review['decisions']) != 244 or len(decisions) != 244:
        raise ValueError('Expected 244 reviewed pairs')
    if {sid for sid, r in decisions.items() if r['decision'] != 'PASS'} != target_ids:
        raise ValueError('Unresolved review IDs differ')
    for sid, decision in decisions.items():
        for field, key in [('original_answer', 'original_text_sha256'), ('proxy_answer', 'proxy_text_sha256')]:
            if sha(norm(by_id[sid][field]).encode('utf8')) != decision[key]:
                raise ValueError('Reviewed text changed: ' + sid)
    for sid, candidate in new.items():
        if candidate['original_answer'] != by_id[sid]['original_answer'] or candidate['previous_proxy_answer'] != by_id[sid]['proxy_answer']:
            raise ValueError('Retry provenance differs: ' + sid)
    aligned = []
    for original in records:
        sid = original['sample_id']
        source = source_by_id[sid]
        if original['original_answer'] != source['a'] or original['source_score'] != source['l']:
            raise ValueError('Original answer/score differs from training source: ' + sid)
        row = copy.deepcopy(original)
        row['assigned_label'] = None
        row['previous_proxy_answer'] = original['proxy_answer']
        row['alignment_review_status'] = 'PASS' if sid in decisions else 'NOT_INDIVIDUALLY_REVIEWED'
        row['semantic_equivalence_verified'] = sid in decisions
        number = sid.rsplit(':', 1)[-1]
        row['alignment_action'] = 'keep_original_proxy'
        if sid in target_ids:
            # Old response provenance must not be presented as the source of new text.
            row['previous_generation_provenance'] = {
                k: row.pop(k) for k in ('raw_response_sha256', 'returned_model', 'response_id', 'outer_fence_removed') if k in row}
            if number in ACCEPT:
                row['proxy_answer'] = new[sid]['proxy_answer']
                row['raw_response_sha256'] = new[sid]['raw_response_sha256']
                row['alignment_action'] = 'use_reviewed_retry'
                row['kind'] = 'generated_proxy_retry'
            else:
                row['proxy_answer'] = source['a']
                row['alignment_action'] = 'original_answer_fallback'
                row['kind'] = 'original_answer_fallback'
                row['alignment_review_status'] = 'EXACT_ORIGINAL_PASSTHROUGH'
                row['rejected_retry_answer'] = new[sid]['proxy_answer']
        elif original['kind'] == 'blank_passthrough':
            if source['a'].strip() or row['proxy_answer'] != source['a']:
                raise ValueError('Blank passthrough changed')
            row['alignment_action'] = 'blank_passthrough'
            row['alignment_review_status'] = 'EXACT_ORIGINAL_PASSTHROUGH'
            row['semantic_equivalence_verified'] = True
        aligned.append(row)
    counts = {name: sum(r['alignment_action'] == name for r in aligned) for name in (
        'keep_original_proxy', 'use_reviewed_retry', 'original_answer_fallback', 'blank_passthrough')}
    if counts != {'keep_original_proxy': 868, 'use_reviewed_retry': 5, 'original_answer_fallback': 7, 'blank_passthrough': 5}:
        raise ValueError('Unexpected alignment counts: ' + str(counts))
    return aligned, counts


def execute(root, policy, check_only=False):
    if policy != 'original_passthrough':
        raise ValueError('Explicit original_passthrough policy required')
    base = root / 'paper_method_v1'
    artifact = base / 'artifacts/proxy_generated_v1'
    retry = base / 'runs/proxy_glm_v1/semantic_retry_v1'
    inputs = base / 'artifacts/proxy_inputs_v1'
    raw = (artifact / 'proxy_records.jsonl').read_bytes()
    if sha(raw) != read(artifact / 'manifest.json')['records_sha256']:
        raise ValueError('Proxy manifest mismatch')
    candidate_bytes = (retry / 'candidates.json').read_bytes()
    if sha(candidate_bytes) != EXPECTED:
        raise ValueError('Candidates differ from reviewed attachment')
    plan = read(retry / 'plan.json')
    if plan['records_sha256'] != sha(raw) or plan['review_sha256'] != sha((retry / 'review.json').read_bytes()):
        raise ValueError('Retry plan provenance mismatch')
    src = inputs / 'training_sources.json'
    if sha(src.read_bytes()) != read(inputs / 'summary.json')['files_sha256']['training_sources.json']:
        raise ValueError('Training source hash mismatch')
    candidates = json.loads(candidate_bytes)
    for c in candidates:
        rid = 'proxy_' + sha(c['sample_id'].encode('utf8'))
        if sha((retry / 'responses' / (rid + '.json')).read_bytes()) != c['raw_response_sha256']:
            raise ValueError('Retry response hash mismatch')
    records = [json.loads(line) for line in raw.decode('utf8').splitlines() if line.strip()]
    aligned, counts = build(records, read(src), candidates, read(retry / 'review.json'))
    data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in aligned).encode('utf8')
    report = {'status': 'PROXY_ALIGNMENT_CHECK_OK' if check_only else 'PROXY_ALIGNMENT_FROZEN_UNLABELED',
        'training_records': 885, 'counts': counts, 'unreviewed_generated_records': 636,
        'initial_reviewed_pairs': 244, 'accepted_retry_ids': sorted(ACCEPT),
        'original_fallback_ids': sorted(FALLBACK),
        'source_records_sha256': sha(raw), 'candidates_sha256': EXPECTED,
        'training_sources_sha256': sha(src.read_bytes()), 'aligned_records_sha256': sha(data),
        'builder_sha256': sha(Path(__file__).read_bytes()),
        'labels_assigned': 0, 'api_calls': 0, 'training_started': False, 'test_accessed': False,
        'ready_for_annotator_training': False,
        'deviation': 'Seven unresolved proxies use exact original answers after one targeted regeneration. '
                     'This is a documented reproduction quality-control choice, not an original-paper step.',
        'next_step': 'Prepare annotator inputs and explicitly document original-score transfer assumption.'}
    if not check_only:
        out = base / 'artifacts/proxy_aligned_v1'
        out.mkdir(parents=True, exist_ok=True)
        payloads = {'proxy_records.jsonl': data, 'manifest.json': enc(report)}
        # Check all conflicts before writing anything; never overwrite a different version.
        for name, content in payloads.items():
            if (out / name).exists() and (out / name).read_bytes() != content:
                raise ValueError('Existing aligned artifact differs: ' + name)
        for name, content in payloads.items():
            if not (out / name).exists():
                temp = out / (name + '.tmp')
                temp.write_bytes(content)
                temp.replace(out / name)
        print('OUTPUT:', out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--unresolved-policy', choices=['original_passthrough'], required=True)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    execute(args.root.resolve(), args.unresolved_policy, args.check_only)
