"""Prepare offline annotator labels, preserving references and original score scale."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ALIGNED_SHA = 'c84bd04ca7a9755c448ccb2f4e18c8d72e860d4f7f76322e4c639ad7b69d8b33'
SOURCE_SHA = '5dcff9d092ee292d5be2bfbe9bcafed2fb6fcd1cc82e02c0e2acbff7ae64fb5d'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def convert(rows, sources):
    lookup = {x['sample_id']: x for x in sources}
    if len(rows) != 885 or len(sources) != 885 or len(lookup) != 885:
        raise ValueError('Expected 885 unique training sources')
    seen = set()
    result = []
    for r in rows:
        sid = r['sample_id']
        if sid in seen or ':train:' not in sid:
            raise ValueError('Duplicate or non-training ID: ' + sid)
        seen.add(sid)
        s = lookup[sid]
        if any(r[a] != s[b] for a, b in [('original_answer', 'a'), ('source_score', 'l'),
                  ('question', 'q'), ('question_id', 'question_id'), ('references', 'r')]):
            raise ValueError('Training provenance mismatch: ' + sid)
        score = r['source_score']
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 10:
            raise ValueError('Invalid ADS score: ' + sid)
        if not isinstance(r['proxy_answer'], str) or not isinstance(r['references'], list):
            raise ValueError('Unexpected text schema')
        if not all(isinstance(x, str) for x in r['references']) or not any(x.strip() for x in r['references']):
            raise ValueError('Missing reference: ' + sid)
        passthrough = r['alignment_action'] in ('blank_passthrough', 'original_answer_fallback')
        if passthrough and r['proxy_answer'] != s['a']:
            raise ValueError('Passthrough answer changed')
        result.append({'sample_id': sid, 'question_id': r['question_id'],
            'question': r['question'], 'references': r['references'],
            'answer': r['proxy_answer'], 'label': float(score), 'source_score': score,
            'label_origin': 'original_training_score',
            'label_transfer_basis': 'exact_original_answer' if passthrough else 'proxy_semantic_equivalence_assumption',
            'alignment_action': r['alignment_action'],
            'alignment_review_status': r['alignment_review_status'],
            'semantic_equivalence_verified': r['semantic_equivalence_verified']})
    if seen != set(lookup):
        raise ValueError('Training ID coverage differs')
    return result


def execute(root):
    base = root / 'paper_method_v1'
    folder = base / 'artifacts/proxy_aligned_v1'
    raw = (folder / 'proxy_records.jsonl').read_bytes()
    manifest = json.loads((folder / 'manifest.json').read_bytes())
    if sha(raw) != ALIGNED_SHA or manifest['aligned_records_sha256'] != ALIGNED_SHA:
        raise ValueError('Aligned data differs from reviewed version')
    source = (base / 'artifacts/proxy_inputs_v1/training_sources.json').read_bytes()
    if sha(source) != SOURCE_SHA:
        raise ValueError('Original training data changed')
    rows = [json.loads(line) for line in raw.decode('utf8').splitlines() if line.strip()]
    training = convert(rows, json.loads(source))
    data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in training).encode('utf8')
    report = {'status': 'ANNOTATOR_DATA_READY_CONFIG_PENDING', 'training_records': len(training),
        'annotator_labels_assigned': len(training), 'synthetic_candidate_labels_assigned': 0,
        'score_scale': 'ADS original 0-10; no normalization or rounding',
        'score_distribution': dict(sorted(Counter(r['label'] for r in training).items())),
        'nonempty_reference_counts': dict(Counter(sum(bool(s.strip()) for s in r['references']) for r in training)),
        'reference_policy': 'All reference slots retained. No selection, concatenation, or row expansion yet.',
        'label_assumption': 'Use original score for each proxy as in paper III-B2; this assumes semantic preservation.',
        'unreviewed_generated_records': 636, 'original_fallback_records': 7,
        'aligned_records_sha256': ALIGNED_SHA, 'training_sources_sha256': SOURCE_SHA,
        'train_jsonl_sha256': sha(data), 'builder_sha256': sha(Path(__file__).read_bytes()),
        'pending_training_choices': ['multiple-reference handling', 'local BERT path',
           'tokenization length/truncation', 'annotator optimizer and training duration',
           'validation/checkpoint selection', 'score scale for beta=0.2'],
        'api_calls': 0, 'training_started': False, 'test_accessed': False,
        'ready_for_annotator_training': False}
    out = base / 'artifacts/annotator_inputs_v1'
    files = {'train.jsonl': data, 'manifest.json': (json.dumps(report, ensure_ascii=False, indent=2)+'\n').encode('utf8')}
    for name, content in files.items():
        if (out / name).exists() and (out / name).read_bytes() != content:
            raise ValueError('Existing artifact differs: ' + str(out / name))
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        if not (out / name).exists():
            tmp = out / (name + '.tmp'); tmp.write_bytes(content); tmp.replace(out / name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('OUTPUT:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--accept-proxy-label-assumption', action='store_true', required=True)
    args = parser.parse_args()
    execute(args.root.resolve())
