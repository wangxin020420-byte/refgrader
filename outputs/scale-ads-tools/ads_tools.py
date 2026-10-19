"""Python 3.8 standard-library ADS preparation and evaluation. No model training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

COMMIT = '9e1a866d58edaa0d78af374d2f3529a10d72239a'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf8')


def lines(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode('utf8')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_once(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Different existing output; refusing overwrite: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def content_key(row):
    return digest(json.dumps([row['q_id'], row['answer']], ensure_ascii=False).encode('utf8'))


def load_source(path, split):
    result = []
    with path.open(encoding='utf-8-sig') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {'q_id', 'question', 'answer', 'score', 'reference1', 'reference2', 'reference3'}
            if set(row) != required:
                raise ValueError('Unexpected source schema')
            score = float(row['score'])
            if not math.isfinite(score) or not 0 <= score <= 10 or not score.is_integer():
                raise ValueError('Invalid ADS integer score')
            for field in required - {'score'}:
                if not isinstance(row[field], str):
                    raise ValueError('Expected text: ' + field)
            result.append({'row': row, 'sample_id': 'ADS:' + COMMIT[:12] + ':' + split + ':' + str(line_no),
                           'source_file': 'ADS/' + path.name, 'source_line': line_no,
                           'content_hash': content_key(row), 'gold_raw': score})
    return result


def partition(records, seed, fraction):
    by_question = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_question[record['row']['q_id']][record['content_hash']].append(record)
    train, validation, detail = [], [], {}
    for qid, groups in sorted(by_question.items()):
        keys = sorted(groups)
        rng = random.Random(str(seed) + ':' + qid)
        rng.shuffle(keys)
        target = sum(len(v) for v in groups.values()) * fraction
        selected, size = set(), 0
        # Keep at least one content group in train; whole groups never cross partitions.
        for key in keys:
            if len(selected) >= len(keys) - 1:
                break
            if not selected or abs(size + len(groups[key]) - target) < abs(size - target):
                selected.add(key)
                size += len(groups[key])
        for key, items in groups.items():
            (validation if key in selected else train).extend(items)
        detail[qid] = {'groups': len(keys), 'validation_rows': size,
                       'warning': 'Only one group; no validation for this question' if len(keys) < 2 else None}
    return sorted(train, key=lambda r: r['source_line']), sorted(validation, key=lambda r: r['source_line']), detail


def prepare(args):
    root = args.root.expanduser().resolve()
    source = root / 'data' / ('SAS-Datasets-' + COMMIT) / 'ADS'
    audit = json.loads((root / 'reports' / 'data_audit.json').read_text(encoding='utf8'))
    if audit['commit'] != COMMIT:
        raise ValueError('Unexpected source commit')
    raw = {}
    source_hashes = {}
    for split, count in [('all', 1582), ('train', 1107), ('test', 475)]:
        path = source / (split + '.jsonl')
        h = digest(path.read_bytes())
        if h != audit['datasets']['ADS'][split]['sha256']:
            raise ValueError('Source checksum mismatch: ' + split)
        source_hashes[split] = h
        raw[split] = load_source(path, split)
        if len(raw[split]) != count:
            raise ValueError('Source count mismatch: ' + split)
    counts = {s: Counter(json.dumps(r['row'], sort_keys=True) for r in rows) for s, rows in raw.items()}
    if counts['all'] != counts['train'] + counts['test']:
        raise ValueError('Original train/test do not reconstruct all')
    train, val, detail = partition(raw['train'], 42, 0.2)
    if {r['content_hash'] for r in train} & {r['content_hash'] for r in val}:
        raise ValueError('Train/validation content leakage')
    splits = {'train': train, 'validation': val, 'test': raw['test']}
    outputs = {}
    for name, records in splits.items():
        inputs, labels = [], []
        for r in records:
            inputs.append({'sample_id': r['sample_id'], 'question_id': r['row']['q_id'],
                           'question': r['row']['question'], 'answer': r['row']['answer'],
                           'references': [r['row']['reference' + str(i)] for i in (1, 2, 3)],
                           'score_max': 10, 'source_file': r['source_file'], 'source_line': r['source_line'],
                           'content_hash': r['content_hash']})
            labels.append({'sample_id': r['sample_id'], 'question_id': r['row']['q_id'], 'gold_raw': r['gold_raw']})
        outputs[name + '.inputs.jsonl'] = lines(inputs)
        outputs[name + '.labels.jsonl'] = lines(labels)
    groups = defaultdict(lambda: defaultdict(list))
    for split, records in splits.items():
        for r in records:
            groups[r['content_hash']][split].append(r['sample_id'])
    overlaps = [{'content_hash': key, 'sample_ids': value} for key, value in sorted(groups.items())
                if 'test' in value and ('train' in value or 'validation' in value)]
    outputs['overlap_audit.json'] = encoded(overlaps)
    manifest = {'protocol': 'ADS_PUBLIC_ADAPTED_V1', 'identity': 'adapted protocol, not exact paper replication',
                'dataset_commit': COMMIT, 'script_sha256': digest(Path(__file__).read_bytes()),
                'seed': 42, 'validation_target_fraction': 0.2,
                'validation_policy': 'Per question; exact answer text groups from upstream train only',
                'test_policy': 'All 475 upstream test rows retained, no test-driven cleaning',
                'counts': {s: len(rows) for s, rows in splits.items()}, 'question_details': detail,
                'source_sha256': source_hashes, 'cross_test_overlap_groups': len(overlaps),
                'metrics_policy': {'score_range': [0, 10], 'qwk_grid': 'integers 0..10',
                                   'qwk_prediction_rounding': 'floor(x+0.5)', 'out_of_range': 'reject',
                                   'normalization': 'divide by known score_max=10; never fit on test'},
                'warnings': ['Cross-test duplicates retained for upstream comparability; see overlap_audit.json',
                             'Seed, validation split and metric policy are adaptation choices, not verified paper settings'],
                'outputs_sha256': {name: digest(data) for name, data in outputs.items()}}
    outputs['manifest.json'] = encoded(manifest)
    dest = root / 'data' / 'prepared' / 'ads_public_adapted_v1'
    # Preflight every existing output before writing any file.
    for name, data in outputs.items():
        p = dest / name
        if p.exists() and p.read_bytes() != data:
            raise ValueError('Existing protocol differs; no overwrite: ' + str(p))
    for name, data in outputs.items():
        write_once(dest / name, data)
    print(json.dumps({'prepared_dir': str(dest), 'counts': manifest['counts'],
                      'cross_test_overlap_groups': len(overlaps), 'training_started': False}, indent=2))


def metrics(gold, prediction):
    if not gold or len(gold) != len(prediction):
        raise ValueError('Empty or unequal arrays')
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 10
               for x in gold + prediction):
        raise ValueError('Scores must be finite numbers in 0..10')
    if any(int(x) != x for x in gold):
        raise ValueError('ADS QWK requires integer gold labels')
    n = len(gold)
    errors = [p - g for g, p in zip(gold, prediction)]
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    mg, mp = sum(gold) / n, sum(prediction) / n
    vg = sum((g - mg) ** 2 for g in gold)
    vp = sum((p - mp) ** 2 for p in prediction)
    pearson = sum((g - mg) * (p - mp) for g, p in zip(gold, prediction)) / math.sqrt(vg * vp) if vg and vp else None
    discrete = [math.floor(p + 0.5) for p in prediction]
    a, b = Counter(gold), Counter(discrete)
    observed = sum((g - p) ** 2 for g, p in zip(gold, discrete)) / n
    expected = sum(ng * np * (g - p) ** 2 for g, ng in a.items() for p, np in b.items()) / (n * n)
    return {'n': n, 'mae_raw': mae, 'rmse_raw': rmse, 'bias_raw': sum(errors) / n,
            'mae_normalized': mae / 10, 'rmse_normalized': rmse / 10,
            'pearson': pearson, 'qwk': 1 - observed / expected if expected else None,
            'undefined_metrics': [key for key, value in [('pearson', pearson), ('qwk', expected if expected else None)] if value is None]}


def evaluate(args):
    gold_rows, pred_rows = read_jsonl(args.labels), read_jsonl(args.predictions)
    def index(rows):
        output = {}
        for row in rows:
            key = row['sample_id']
            if not isinstance(key, str) or not key or key in output:
                raise ValueError('Invalid or duplicate sample_id')
            output[key] = row
        return output
    gold, predictions = index(gold_rows), index(pred_rows)
    if gold.keys() != predictions.keys():
        raise ValueError('Missing or extra prediction IDs')
    ids = sorted(gold)
    result = metrics([gold[i]['gold_raw'] for i in ids], [predictions[i]['prediction_raw'] for i in ids])
    questions = defaultdict(list)
    for i in ids:
        questions[gold[i]['question_id']].append(i)
    result['by_question'] = {q: metrics([gold[i]['gold_raw'] for i in items], [predictions[i]['prediction_raw'] for i in items])
                             for q, items in sorted(questions.items())}
    result['label_sha256'] = digest(args.labels.read_bytes())
    result['prediction_sha256'] = digest(args.predictions.read_bytes())
    result['policy'] = 'raw 0..10; normalized /10; QWK floor(x+0.5); invalid/out-of-range rejected; null if undefined'
    result['evaluation_label'] = args.evaluation_label
    write_once(args.output, encoded(result))
    print('Evaluation written: ' + str(args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--root', type=Path, required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser('evaluate')
    for key in ('labels', 'predictions', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--evaluation-label', required=True, help='Model/run identity, or TEST_FIXTURE for constructed scores')
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
