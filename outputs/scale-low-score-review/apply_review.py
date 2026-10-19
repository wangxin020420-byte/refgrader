import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def apply(root):
    run = root / 'runs/deepseek_ads_low_score_pilot_v1'
    source = run / 'candidates.jsonl'
    spec = json.loads((Path(__file__).parent / 'decisions.json').read_text(encoding='utf8'))
    candidates = [json.loads(s) for s in source.read_text(encoding='utf8').splitlines() if s.strip()]
    lookup = {r['candidate_id']: r for r in candidates}
    decisions = spec['decisions']
    if len(candidates) != 48 or len(lookup) != 48 or set(lookup) != {r['candidate_id'] for r in decisions}:
        raise ValueError('Candidate IDs differ from reviewed file')
    provisional = []
    for decision in decisions:
        row = lookup[decision['candidate_id']]
        actual = hashlib.sha256(row['generated_answer'].strip().encode('utf8')).hexdigest()
        if row['sample_id'] != decision['sample_id'] or actual != decision['generated_answer_sha256_stripped']:
            raise ValueError('Candidate content differs from reviewed text: '+decision['candidate_id'])
        if decision['decision'] == 'PROVISIONAL_EXPERIMENT_ONLY':
            if row['flags']:
                raise ValueError('Unexpected automatic flag: '+decision['candidate_id'])
            provisional.append(dict(row, candidate_gold_raw=None, review_status=decision['decision']))
    audit = dict(spec, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 counts=dict(Counter(d['decision'] for d in decisions)),
                 formal_training_labels_assigned=0, training_started=False)
    if len(provisional) != 30:
        raise ValueError('Expected 30 provisional candidates')
    outputs = {'ai_review_v1.json': json.dumps(audit, ensure_ascii=False, indent=2)+'\n',
               'provisional_candidates_v1.jsonl': ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in provisional)}
    for name, content in outputs.items():
        path = run / name
        if path.exists() and path.read_text(encoding='utf8') != content:
            raise ValueError('Different existing output: '+name)
    for name, content in outputs.items():
        path = run / name
        if not path.exists():
            with path.open('x', encoding='utf8') as stream:
                stream.write(content)
    print(json.dumps({'status':'LOW_SCORE_REVIEW_SAVED', 'counts':audit['counts'],
                      'formal_training_labels_assigned':0, 'training_started':False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    apply(parser.parse_args().root)
