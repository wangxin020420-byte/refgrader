import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from label_free_checkpoint import Checkpoint, _digest, _write_atomic
from scripts.derive_slot_calibration import SOURCE_COMMIT, verify_source


class DerivationTests(unittest.TestCase):
    def fixture(self, root):
        old = {'source': {'git_commit': SOURCE_COMMIT}, 'run_config': {'question_ids': ['Q'], 'split': 'calibration'},
               'input_groups': {'a': 1}, 'split_manifest_sha256': 'x', 'model': {'m': 1}, 'candidate_snapshot': {'h': 2}}
        ck = Checkpoint(root, old); records = {}
        for i in range(164):
            ident = f'Q_A{i}'; value = {'question_id': 'Q', 'answer_id': f'A{i}', 'observation': {'x': i}}
            records[ident] = value; ck.save(ident, value)
        lineage = {'target': old['source'], 'source_answers': [
            {'checkpoint_id': ident, 'result_sha256': _digest(records[ident])} for ident in list(records)[:62]]}
        _write_atomic(root/'lineage.json', {'lineage': lineage, 'sha256': _digest(lineage)})
        return old, records

    def test_lineage_and_inputs_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); old, records = self.fixture(root)
            answers = [{'answer_id': f'A{i}'} for i in range(164)]
            with patch('scripts.derive_slot_calibration.load_unlabeled_answers', return_value=answers):
                _, _, actual = verify_source(root, dict(old, source={'git_commit': 'new'}))
                self.assertEqual(actual, records)
                with self.assertRaisesRegex(ValueError, 'Incompatible field: model'):
                    verify_source(root, dict(old, model={}))
            self.assertEqual(len(list((root/'answers').glob('*.json'))), 164)

    def test_tampering_or_missing_source_answer_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); old, _ = self.fixture(root)
            (root/'answers'/'Q_A10.json').write_text('{}', encoding='utf-8')
            with patch('scripts.derive_slot_calibration.load_unlabeled_answers', return_value=[{'answer_id': f'A{i}'} for i in range(164)]):
                with self.assertRaises(ValueError): verify_source(root, old)


if __name__ == '__main__': unittest.main()
