import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from label_free_checkpoint import Checkpoint, _digest, _write_atomic
from scripts.derive_whole_calibration import SOURCE_COMMIT, verify_source


class WholeDerivationTests(unittest.TestCase):
    def test_three_generations_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old={'source':{'git_commit':SOURCE_COMMIT},'run_config':{'question_ids':['Q'],'split':'calibration'},
                 'input_groups':{'a':1},'split_manifest_sha256':'x','model':{'m':1},'candidate_snapshot':{'h':2}}
            ck=Checkpoint(root,old); records={}
            for i in range(175):
                key=f'Q_A{i}'; result={'question_id':'Q','answer_id':f'A{i}'}
                ck.save(key,result); records[key]=result
            def entries(n): return [{'checkpoint_id':k,'result_sha256':_digest(records[k])} for k in list(records)[:n]]
            earliest={'source_answers':entries(62)}
            middle={'source_answers':entries(164),'parent_lineage':{'lineage':earliest,'sha256':_digest(earliest)}}
            latest={'target_source':old['source'],'source_answers':entries(175),'parent_lineage':{'lineage':middle,'sha256':_digest(middle)}}
            _write_atomic(root/'lineage.json',{'lineage':latest,'sha256':_digest(latest)})
            with patch('scripts.derive_whole_calibration.load_unlabeled_answers',return_value=[{'answer_id':f'A{i}'} for i in range(175)]):
                _, _, got=verify_source(root,dict(old,source={'git_commit':'new'})); self.assertEqual(got,records)
                earliest['source_answers'][0]['result_sha256']='bad'
                middle['parent_lineage']['sha256']=_digest(earliest)
                latest['parent_lineage']['sha256']=_digest(middle)
                _write_atomic(root/'lineage.json',{'lineage':latest,'sha256':_digest(latest)})
                with self.assertRaisesRegex(ValueError,'Original 62-answer lineage differs'): verify_source(root,old)


if __name__=='__main__': unittest.main()
