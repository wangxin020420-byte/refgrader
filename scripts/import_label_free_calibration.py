"""Import verified older calibration checkpoints into a new exploratory run.

Offline only. The source archive remains immutable. The new checkpoint envelope
uses the current manifest; lineage records original per-answer digests. It does
not establish that label provenance or mixed-version results are scientifically
equivalent. The regular development runner can later resume the new run ID.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from label_free_checkpoint import Checkpoint, _digest, _write_atomic
from label_free_inputs import load_unlabeled_answers
from scripts.run_label_free_development import PREPARED, RUN_ROOT, _inputs

EXPECTED_ARCHIVE = '7ea02b679c0ce6259585465c6f69396a3aba64a3be57bc2a67cb560400edfbc5'
EXPECTED_SOURCE = '36341ae331b0e4e5150e513a6f0f106bdb097e12'
SNAPSHOT = ROOT / 'data/label_free_snapshots/csbench_3531732_co3_independent_v1'


def read_archive(path, expected_hash=EXPECTED_ARCHIVE):
    path = Path(path)
    with path.open('rb') as stream:
        actual = hashlib.sha256(stream.read()).hexdigest()
    if actual != expected_hash:
        raise ValueError('Archive SHA256 mismatch')
    with ZipFile(path) as archive:
        entries = {}
        for entry in archive.infolist():
            name = entry.filename.replace('\\', '/')
            if name.startswith('/') or '..' in PurePosixPath(name).parts or name in entries:
                raise ValueError('Unsafe or duplicate archive entry')
            entries[name] = entry
        names = [n for n in entries if n == 'manifest.json' or n.endswith('/manifest.json')]
        if len(names) != 1:
            raise ValueError('Expected exactly one manifest')
        name = names[0]
        prefix = name[:-len('manifest.json')]
        envelope = json.loads(archive.read(entries[name]))
        if set(envelope) != {'schema', 'manifest', 'sha256'} or envelope['schema'] != 'label_free_checkpoint_v1' or envelope['sha256'] != _digest(envelope['manifest']):
            raise ValueError('Invalid source manifest')
        records = []
        for name, entry in entries.items():
            if not name.startswith(prefix + 'answers/') or not name.endswith('.json'):
                continue
            rec = json.loads(archive.read(entry))
            if set(rec) != {'answer_id', 'manifest_sha256', 'result', 'sha256'}:
                raise ValueError('Invalid source answer envelope')
            body = {k: rec[k] for k in ('answer_id', 'manifest_sha256', 'result')}
            if rec['sha256'] != _digest(body) or rec['manifest_sha256'] != envelope['sha256']:
                raise ValueError('Source answer checksum mismatch')
            result = rec['result']
            if not isinstance(result, dict) or rec['answer_id'] != result['question_id'] + '_' + result['answer_id'] or name != prefix + 'answers/' + rec['answer_id'] + '.json':
                raise ValueError('Source answer identity mismatch')
            records.append(rec)
    if len(records) != 62 or len({r['answer_id'] for r in records}) != len(records):
        raise ValueError('Expected 62 unique source answers')
    if envelope['manifest']['source']['git_commit'] != EXPECTED_SOURCE:
        raise ValueError('Unexpected source commit')
    return actual, envelope, records


def validate(old, current, records, allowed):
    for key in ('run_config', 'input_groups', 'split_manifest_sha256', 'model', 'candidate_snapshot'):
        if old.get(key) != current.get(key):
            raise ValueError('Incompatible field: ' + key)
    cfg = old['run_config']
    if cfg.get('split') != 'calibration' or cfg.get('risk_policy') != 'candidate_max_v1':
        raise ValueError('Unexpected source split or risk policy')
    for rec in records:
        result = rec['result']
        if (result['question_id'], result['answer_id']) not in allowed:
            raise ValueError('Source answer not in calibration split')


def build_lineage(archive_hash, envelope, records, current):
    return {'schema': 'label_free_mixed_source_lineage_v1', 'scope': 'exploratory_not_formal',
            'source_archive_sha256': archive_hash, 'source_manifest_sha256': envelope['sha256'],
            'source': envelope['manifest']['source'], 'target': current['source'],
            'source_answers': [{'checkpoint_id': r['answer_id'], 'source_sha256': r['sha256'],
                                'result_sha256': _digest(r['result'])} for r in sorted(records, key=lambda r: r['answer_id'])],
            'warning': 'Mixed source versions; no claim of distributional or formal equivalence.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--new-run-id', required=True)
    parser.add_argument('--check', action='store_true', help='Validate only; do not create a run')
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', args.new_run_id):
            raise ValueError('Invalid new run ID')
        archive_hash, envelope, records = read_archive(args.archive)
        old = envelope['manifest']
        cfg = old['run_config']
        current = _inputs(cfg['question_ids'], 'calibration', SNAPSHOT, 'candidate_max_v1')
        allowed = {(q, a['answer_id']) for q in cfg['question_ids']
                   for a in load_unlabeled_answers(PREPARED, q, 'calibration')}
        validate(old, current, records, allowed)
        lineage = build_lineage(archive_hash, envelope, records, current)
        destination = RUN_ROOT / args.new_run_id
        if destination.exists():
            raise ValueError('Destination already exists; never overwrite an existing run')
        if not args.check:
            destination.mkdir(parents=True, exist_ok=False)
            checkpoint = Checkpoint(destination, current)
            for record in records:
                checkpoint.save(record['answer_id'], record['result'])
            for record in records:
                if checkpoint.load(record['answer_id']) != record['result']:
                    raise ValueError('Imported result verification failed')
            _write_atomic(destination / 'lineage.json', {'lineage': lineage, 'sha256': _digest(lineage)})
        print(json.dumps({'compatible': True, 'check_only': args.check, 'new_run_id': args.new_run_id,
                          'imported_count': len(records), 'source': old['source'], 'target': current['source'],
                          'lineage_sha256': _digest(lineage), 'mixed_versions': True,
                          'api_calls_made': 0, 'ready_for_formal_experiment': False}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('Import failed: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
