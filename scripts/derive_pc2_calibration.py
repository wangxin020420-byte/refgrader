"""Import the second PC's intact original calibration into a new exploratory run.

Reads existing checkpoints only, verifies every digest and unchanged input
fingerprints, and retains per-answer parent hashes. Never overwrites a run.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from label_free_checkpoint import Checkpoint, _digest, _write_atomic
from label_free_inputs import load_unlabeled_answers
from scripts.run_label_free_development import PREPARED, RUN_ROOT, _inputs, _single_writer

SOURCE_COMMIT = '36341ae331b0e4e5150e513a6f0f106bdb097e12'
SOURCE_HASH = '7338907a8e2b4663c1fab4af36f5e7c185f00d62aae253d05ee13b997c5cb6a2'
SNAPSHOT = ROOT / 'data/label_free_snapshots/csbench_3531732_co3_independent_v1'


def verify_source(source_dir, current):
    envelope = json.loads((source_dir / 'manifest.json').read_text(encoding='utf-8'))
    old = envelope['manifest']
    if set(envelope) != {'schema', 'manifest', 'sha256'} or envelope['schema'] != 'label_free_checkpoint_v1' or envelope['sha256'] != _digest(old):
        raise ValueError('Invalid source manifest')
    if old['source']['git_commit'] != SOURCE_COMMIT or old['source']['source_sha256'] != SOURCE_HASH:
        raise ValueError('Unexpected original code fingerprint')
    if old.get('teacher_scores_used') is not False or old['run_config'].get('split') != 'calibration':
        raise ValueError('Unexpected source scope')
    for field in ('run_config', 'input_groups', 'split_manifest_sha256', 'model', 'candidate_snapshot'):
        if old.get(field) != current.get(field):
            raise ValueError('Incompatible field: ' + field)
    checkpoint = Checkpoint(source_dir, old)
    files = list((source_dir / 'answers').glob('*.json'))
    if len(files) != 164:
        raise ValueError('Expected exactly 164 source answers; re-evaluate before importing')
    allowed = {(q, a['answer_id']) for q in old['run_config']['question_ids']
               for a in load_unlabeled_answers(PREPARED, q, 'calibration')}
    records = {}
    for path in files:
        value = checkpoint.load(path.stem)
        pair = (value['question_id'], value['answer_id'])
        if pair not in allowed or path.stem != pair[0] + '_' + pair[1] or path.stem in records:
            raise ValueError('Unexpected source answer identity')
        records[path.stem] = value
    return envelope, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--new-run-id', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', args.new_run_id):
            raise ValueError('Invalid new run ID')
        source_dir = Path(args.source_dir).resolve(strict=True)
        destination = RUN_ROOT / args.new_run_id
        if destination.exists():
            raise ValueError('Destination already exists; refusing overwrite')
        if not (source_dir / 'manifest.json').is_file():
            raise ValueError('Source manifest not found')
        old = json.loads((source_dir / 'manifest.json').read_text(encoding='utf-8'))['manifest']
        current = _inputs(old['run_config']['question_ids'], 'calibration', SNAPSHOT, 'candidate_max_v1')
        with _single_writer(source_dir):
            envelope, records = verify_source(source_dir, current)
            lineage = {'schema': 'label_free_pc2_derivation_lineage_v1', 'scope': 'exploratory_not_formal',
                       'parent_run_id': 'lf45_co3ind_20261029_v1_calibration',
                       'parent_manifest_sha256': envelope['sha256'], 'parent_source': envelope['manifest']['source'],
                       'target_source': current['source'],
                       'source_answers': [{'checkpoint_id': ident, 'result_sha256': _digest(value)}
                                          for ident, value in sorted(records.items())],
                       'warning': 'Mixed source versions and extraction protocols; results require stratified reporting.'}
            if not args.check:
                destination.mkdir(parents=True, exist_ok=False)
                target = Checkpoint(destination, current)
                for ident, value in records.items(): target.save(ident, value)
                for ident, value in records.items():
                    if target.load(ident) != value: raise ValueError('Imported answer mismatch')
                _write_atomic(destination / 'lineage.json', {'lineage': lineage, 'sha256': _digest(lineage)})
        print(json.dumps({'compatible': True, 'check_only': args.check, 'new_run_id': args.new_run_id,
                          'source_count': len(records), 'source': envelope['manifest']['source'],
                          'target': current['source'], 'lineage_sha256': _digest(lineage),
                          'mixed_versions': True, 'api_calls_made': 0, 'formal': False}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('PC2 derivation failed: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
