import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from label_free_checkpoint import Checkpoint, _digest
from scripts.import_label_free_calibration import read_archive, validate, build_lineage, EXPECTED_SOURCE


class ImportTests(unittest.TestCase):
    def archive(self, path, *, corrupt=False):
        manifest = {'source': {'git_commit': EXPECTED_SOURCE},
                    'run_config': {'split': 'calibration', 'risk_policy': 'candidate_max_v1'},
                    'input_groups': {'x': 1}, 'split_manifest_sha256': 'a',
                    'model': {'text': 'test'}, 'candidate_snapshot': {'hash': 'b'}}
        envelope = {'schema': 'label_free_checkpoint_v1', 'manifest': manifest, 'sha256': _digest(manifest)}
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('old\\manifest.json', json.dumps(envelope))
            for i in range(62):
                result = {'question_id': 'Q1', 'answer_id': f'A{i}'}
                body = {'answer_id': f'Q1_A{i}', 'manifest_sha256': envelope['sha256'], 'result': result}
                z.writestr('old\\answers\\Q1_A' + str(i) + '.json',
                           json.dumps({**body, 'sha256': 'bad' if corrupt and i == 1 else _digest(body)}))
        return hashlib.sha256(path.read_bytes()).hexdigest(), manifest

    def test_windows_paths_checksums_and_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.zip'
            digest, old = self.archive(path)
            sha, env, records = read_archive(path, digest)
            validate(old, dict(old, source={'git_commit': 'new'}), records,
                     {('Q1', f'A{i}') for i in range(62)})
            lineage = build_lineage(sha, env, records, dict(old, source={'git_commit': 'new'}))
            self.assertEqual(len(lineage['source_answers']), 62)
            self.assertEqual(lineage['source']['git_commit'], EXPECTED_SOURCE)
            self.assertEqual(lineage['target']['git_commit'], 'new')
            destination = Path(tmp) / 'derived'
            checkpoint = Checkpoint(destination, dict(old, source={'git_commit': 'new'}))
            for record in records:
                checkpoint.save(record['answer_id'], record['result'])
            for record in records:
                self.assertEqual(checkpoint.load(record['answer_id']), record['result'])
            self.assertTrue(path.is_file())
            with self.assertRaisesRegex(ValueError, 'Incompatible field: model'):
                validate(old, dict(old, model={}), records, set())
            with self.assertRaisesRegex(ValueError, 'not in calibration'):
                validate(old, old, records, set())

    def test_reject_archive_or_answer_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.zip'
            digest, _ = self.archive(path, corrupt=True)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                read_archive(path, digest)
            with self.assertRaisesRegex(ValueError, 'Archive SHA256 mismatch'):
                read_archive(path, '0' * 64)


if __name__ == '__main__':
    unittest.main()
