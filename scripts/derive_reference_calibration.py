"""Create a separately identified, mixed-version exploratory calibration run.

Copies only verified completed results; never mutates the parent checkpoints.
No model calls or teacher scores. Source lineage is embedded in a new record.
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

SOURCE_ID = 'lf45_co3ind_20261029_v1_calibration_slot_v1'
SOURCE_COMMIT = 'f92501444242bda01967e6e04ed3125558a3527e'
SNAPSHOT = ROOT / 'data/label_free_snapshots/csbench_3531732_co3_independent_v1'


def verify_source(source_dir, current):
    envelope = json.loads((source_dir/'manifest.json').read_text(encoding='utf-8'))
    old = envelope['manifest']
    if set(envelope) != {'schema', 'manifest', 'sha256'} or envelope['sha256'] != _digest(old):
        raise ValueError('Source manifest invalid')
    if old['source']['git_commit'] != SOURCE_COMMIT:
        raise ValueError('Unexpected source commit')
    for field in ('run_config', 'input_groups', 'split_manifest_sha256', 'model', 'candidate_snapshot'):
        if old.get(field) != current.get(field): raise ValueError('Incompatible field: ' + field)
    lineage_envelope = json.loads((source_dir/'lineage.json').read_text(encoding='utf-8'))
    lineage = lineage_envelope['lineage']
    if lineage_envelope['sha256'] != _digest(lineage) or lineage['target_source'] != old['source']:
        raise ValueError('Source lineage invalid')
    ancestor = lineage.get('parent_lineage')
    if (not isinstance(ancestor, dict) or ancestor.get('sha256') != _digest(ancestor.get('lineage')) or
            len(ancestor['lineage'].get('source_answers', [])) != 62):
        raise ValueError('Original 62-answer lineage invalid')
    source = Checkpoint(source_dir, old)
    files = list((source_dir/'answers').glob('*.json'))
    if len(files) != 175: raise ValueError('Expected exactly 175 source answers')
    allowed = {(q, a['answer_id']) for q in old['run_config']['question_ids']
               for a in load_unlabeled_answers(PREPARED, q, 'calibration')}
    records = {}
    for path in files:
        result = source.load(path.stem)
        pair = (result['question_id'], result['answer_id'])
        if pair not in allowed or path.stem != pair[0] + '_' + pair[1] or path.stem in records:
            raise ValueError('Unexpected source answer identity')
        records[path.stem] = result
    inherited = lineage['source_answers']
    if len(inherited) != 164 or len({x['checkpoint_id'] for x in inherited}) != 164:
        raise ValueError('Unexpected parent lineage count')
    for item in inherited:
        if item['checkpoint_id'] not in records or _digest(records[item['checkpoint_id']]) != item['result_sha256']:
            raise ValueError('Original imported answer differs from lineage')
    for item in ancestor['lineage']['source_answers']:
        if item['checkpoint_id'] not in records or _digest(records[item['checkpoint_id']]) != item['result_sha256']:
            raise ValueError('Original 62-answer lineage differs from results')
    return envelope, lineage_envelope, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-run-id', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', args.new_run_id): raise ValueError('Invalid run ID')
        source_dir = RUN_ROOT / SOURCE_ID
        destination = RUN_ROOT / args.new_run_id
        if destination.exists(): raise ValueError('Destination exists; refusing overwrite')
        if not (source_dir/'manifest.json').is_file(): raise ValueError('Source run missing')
        old = json.loads((source_dir/'manifest.json').read_text(encoding='utf-8'))['manifest']
        current = _inputs(old['run_config']['question_ids'], 'calibration', SNAPSHOT, 'candidate_max_v1')
        with _single_writer(source_dir):
            envelope, parent_lineage, records = verify_source(source_dir, current)
            lineage = {'schema': 'label_free_reference_derivation_lineage_v1', 'scope': 'exploratory_not_formal',
                       'parent_run_id': SOURCE_ID, 'parent_manifest_sha256': envelope['sha256'],
                       'parent_lineage': parent_lineage, 'parent_source': envelope['manifest']['source'],
                       'target_source': current['source'],
                       'source_answers': [{'checkpoint_id': ident, 'result_sha256': _digest(result)}
                                          for ident, result in sorted(records.items())],
                       'warning': 'Mixed source versions and extraction protocols; results require stratified reporting.'}
            if not args.check:
                destination.mkdir(parents=True, exist_ok=False)
                target = Checkpoint(destination, current)
                for ident, result in records.items(): target.save(ident, result)
                for ident, result in records.items():
                    if target.load(ident) != result: raise ValueError('Imported answer mismatch')
                _write_atomic(destination/'lineage.json', {'lineage': lineage, 'sha256': _digest(lineage)})
        print(json.dumps({'compatible': True, 'check_only': args.check, 'new_run_id': args.new_run_id,
                          'source_count': len(records), 'lineage_sha256': _digest(lineage),
                          'mixed_versions': True, 'api_calls_made': 0, 'formal': False}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('Derivation failed: ' + str(exc), file=sys.stderr); return 2


if __name__ == '__main__': raise SystemExit(main())
