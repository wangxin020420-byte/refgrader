import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from label_free_checkpoint import Checkpoint
from scripts.derive_pc2_calibration import SOURCE_COMMIT, SOURCE_HASH, verify_source


class PC2DerivationTests(unittest.TestCase):
    def test_source_identity_and_164_results_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old={'source': {'git_commit':SOURCE_COMMIT,'source_sha256':SOURCE_HASH},
                 'teacher_scores_used':False, 'run_config':{'question_ids':['Q'],'split':'calibration'},
                 'input_groups':{'a':1},'split_manifest_sha256':'x','model':{'m':1},'candidate_snapshot':{'h':2}}
            ck=Checkpoint(root,old); expected={}
            for i in range(164):
                ident=f'Q_A{i}'; value={'question_id':'Q','answer_id':f'A{i}','observation':{'x':i}}
                ck.save(ident,value); expected[ident]=value
            with patch('scripts.derive_pc2_calibration.load_unlabeled_answers',return_value=[{'answer_id':f'A{i}'} for i in range(164)]):
                _, got=verify_source(root,dict(old,source={'git_commit':'new'}))
                self.assertEqual(got,expected)
                with self.assertRaisesRegex(ValueError,'Incompatible field: model'):
                    verify_source(root,dict(old,model={}))
                (root/'answers'/'Q_A3.json').write_text('{}',encoding='utf-8')
                with self.assertRaises(ValueError): verify_source(root,old)


if __name__=='__main__': unittest.main()
