"""Offline-only numeric POS correction prototype; never invokes any model.

Input files must be independently prepared, audited numeric projections, not
teacher labels or raw checkpoints. Outputs are new files and never overwrite.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from label_free_pos_correction import _digest, apply_candidate, fit_pos


def read(path):
    with Path(path).open('r', encoding='utf-8-sig') as stream:
        return json.load(stream)


def write_new(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    fit = sub.add_parser('fit')
    fit.add_argument('--input', required=True)
    fit.add_argument('--output', required=True)
    fit.add_argument('--min-count', type=int, required=True)
    fit.add_argument('--shrinkage', type=float, required=True)
    fit.add_argument('--cap', type=float, required=True)
    apply = sub.add_parser('apply')
    apply.add_argument('--fit', required=True)
    apply.add_argument('--input', required=True)
    apply.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        data = read(args.input)
        if not isinstance(data, dict) or set(data) != {'records'} or not isinstance(data['records'], list):
            raise ValueError('Input must contain only a records list')
        if args.mode == 'fit':
            result = fit_pos(data['records'], min_count=args.min_count,
                             shrinkage=args.shrinkage, cap=args.cap)
            output = {'fit': result, 'sha256': _digest(result)}
        else:
            envelope = read(args.fit)
            if not isinstance(envelope, dict) or set(envelope) != {'fit', 'sha256'} or _digest(envelope['fit']) != envelope['sha256']:
                raise ValueError('Invalid or modified fit envelope')
            identities = [(r.get('question_id'), r.get('answer_id')) for r in data['records'] if isinstance(r, dict)]
            if len(identities) != len(data['records']) or len(set(identities)) != len(identities):
                raise ValueError('Duplicate or invalid target records')
            results = [apply_candidate(envelope['fit'], record) for record in data['records']]
            output = {'schema': 'label_free_pos_correction_candidates_v1',
                      'fit_sha256': envelope['sha256'], 'input_sha256': _digest(data),
                      'results': results, 'scope': 'offline_exploratory_only'}
        write_new(args.output, output)
        print(json.dumps({'output': str(Path(args.output)), 'scope': 'offline_exploratory_only',
                          'api_calls_made': 0, 'candidate_only': True}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f'Prototype failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
