import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from label_free_checkpoint import Checkpoint, _digest, _write_atomic
from scripts.derive_reference_calibration import SOURCE_COMMIT, verify_source


class ReferenceDerivationTests(unittest.TestCase):
    def test_two_generations_preserved_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = {'source': {'git_commit': SOURCE_COMMIT}, 'run_config': {'question_ids': ['Q'], 'split': 'calibration'},
                   'input_groups': {'a': 1}, 'split_manifest_sha256': 'x', 'model': {'m': 1}, 'candidate_snapshot': {'h': 2}}
            ck = Checkpoint(root, old); records = {}
            for i in range(175):
                ident = f'Q_A{i}'; result = {'question_id':'Q', 'answer_id':f'A{i}'}
                ck.save(ident, result); records[ident] = result
            original = {'source_answers': [{'checkpoint_id':k, 'result_sha256':_digest(records[k])} for k in list(records)[:62]]}
            parent = {'target_source':old['source'], 'parent_lineage':{'lineage':original,'sha256':_digest(original)},
                      'source_answers':[{'checkpoint_id':k,'result_sha256':_digest(records[k])} for k in list(records)[:164]]}
            _write_atomic(root/'lineage.json', {'lineage':parent,'sha256':_digest(parent)})
            with patch('scripts.derive_reference_calibration.load_unlabeled_answers', return_value=[{'answer_id':f'A{i}'} for i in range(175)]):
                _, _, got = verify_source(root, dict(old, source={'git_commit':'new'}))
                self.assertEqual(got, records)
                parent['parent_lineage']['lineage']['source_answers'][0]['result_sha256']='bad'
                parent['parent_lineage']['sha256']=_digest(parent['parent_lineage']['lineage'])
                _write_atomic(root/'lineage.json', {'lineage':parent,'sha256':_digest(parent)})
                with self.assertRaisesRegex(ValueError, 'Original 62-answer lineage differs'):
                    verify_source(root, old)


if __name__=='__main__': unittest.main()
