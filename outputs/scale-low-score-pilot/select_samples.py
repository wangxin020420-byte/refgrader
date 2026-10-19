"""Select a fixed, score-balanced diagnostic sample from training rows only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import unicodedata


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]


def build(data):
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf8'))
    hashes = {}
    for name in ['train.inputs.jsonl', 'train.labels.jsonl']:
        hashes[name] = hashlib.sha256((data / name).read_bytes()).hexdigest()
        if hashes[name] != manifest['outputs_sha256'][name]:
            raise ValueError('Source checksum changed: ' + name)
    inputs, labels = read(data / 'train.inputs.jsonl'), read(data / 'train.labels.jsonl')
    gold = {r['sample_id']: r for r in labels}
    if len(inputs) != 885 or len(gold) != len(labels) or len({r['sample_id'] for r in inputs}) != len(inputs):
        raise ValueError('Unexpected count or duplicate source ID')
    if {r['sample_id'] for r in inputs} != gold.keys():
        raise ValueError('Input/label IDs mismatch')
    all_rows = []
    for row in inputs:
        label = gold[row['sample_id']]
        if label['question_id'] != row['question_id']:
            raise ValueError('Question ID mismatch')
        all_rows.append(dict(row, gold_raw=label['gold_raw']))
    rng = random.Random(20260910)
    counts, selected, used, available = Counter(), [], set(), {}
    for score, target in [(0, 12), (1, 3), (2, 3), (3, 3), (4, 3)]:
        pool = sorted([r for r in all_rows if r['gold_raw'] == score and r['answer'].strip()], key=lambda r:r['sample_id'])
        rng.shuffle(pool)
        available[str(score)] = len(pool)
        for _ in range(target):
            candidates = [r for r in pool if (r['question_id'], ''.join(unicodedata.normalize('NFKC', r['answer']).split())) not in used]
            if not candidates:
                raise ValueError('Insufficient distinct nonblank answers for score ' + str(score))
            # Spread questions while preserving deterministic random tie-breaking.
            row = min(candidates, key=lambda r: counts[r['question_id']])
            selected.append(row)
            counts[row['question_id']] += 1
            used.add((row['question_id'], ''.join(unicodedata.normalize('NFKC', row['answer']).split())))
    content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected).encode('utf8')
    result = {'status': 'LOW_SCORE_PILOT_SELECTED', 'source_split': 'train', 'seed': 20260910,
              'count': len(selected), 'score_counts': dict(Counter(str(r['gold_raw']) for r in selected)),
              'question_counts': dict(counts), 'source_sha256': hashes,
              'nonblank_available_per_score': available,
              'blank_low_score_excluded': sum(r['gold_raw'] < 5 and not r['answer'].strip() for r in all_rows),
              'selection_sha256': hashlib.sha256(content).hexdigest(),
              'warning': 'Diagnostic score-balanced sample, not population proportions; all generated labels remain unset',
              'api_calls': 0, 'training_started': False}
    return content, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    content, report = build(args.root / 'data/prepared/ads_public_adapted_v1')
    dest = args.root / 'data/pilots/deepseek_ads_low_score_v1'
    output = {'selected_train.jsonl': content,
              'selection_manifest.json': (json.dumps(report, ensure_ascii=False, indent=2)+'\n').encode('utf8')}
    for name, value in output.items():
        if (dest/name).exists() and (dest/name).read_bytes() != value:
            raise ValueError('Existing output differs: '+name)
    dest.mkdir(parents=True, exist_ok=True)
    for name, value in output.items():
        if not (dest/name).exists():
            with (dest/name).open('xb') as stream:
                stream.write(value)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
