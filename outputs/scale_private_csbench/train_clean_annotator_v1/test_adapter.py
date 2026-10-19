import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import csbench_inputs as ci
import bert_helpers as h
import run_annotators as runner

def fixture():
    def split(n,prefix,blanks):
        return [dict(sample_id=prefix+str(i),question_id='q',references=['ref'],answer='' if i<blanks else 'O ♣1',
            label=5,score_original=2,score_max_original=4,blank_passthrough=i<blanks) for i in range(n)]
    train=split(345,'tr',5);val=split(343,'va',8)
    jobs=[dict(variant_id=str(i),sample_id='tr'+str(i%345),references=['ref'],source_score=5,assigned_label=None,answer='generated') for i in range(5000)]
    return train,val,jobs

class Tests(unittest.TestCase):
    def test_counts_labels_identity(self):
        tr,va,j=fixture();ci.validate(tr,va,j)
        va[0]['sample_id']='tr0'
        with self.assertRaises(AssertionError):ci.validate(tr,va,j)
    def test_candidate_labels_rejected(self):
        tr,va,j=fixture();j[0]['assigned_label']=4
        with self.assertRaises(AssertionError):ci.validate(tr,va,j)
    def test_scale_rejected(self):
        tr,va,j=fixture();tr[0]['label']=2
        with self.assertRaises(AssertionError):ci.validate(tr,va,j)
    def test_batches_and_qwk(self):
        groups=h.groups(list(range(345)));self.assertEqual(len(groups),22);self.assertEqual(len(groups[-1]),9)
        self.assertEqual([i for g in groups for i in g],list(range(345)))
        self.assertEqual(runner.qwk([0.,5.,10.],[0.,5.,10.]),1)
        self.assertEqual(runner.qwk([0.,10.],[10.,0.]),-1)
        with self.assertRaises(ValueError):runner.qwk([0.,0.],[0.,0.])
    def test_input_integration_and_hash_rejection(self):
        tr,va,j=fixture()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ann=root/'runs/online_experiments'/ci.EXP/'annotator_inputs_v1';ann.mkdir(parents=True)
            expected={}
            for n,rr in zip(('train.jsonl','validation.jsonl','candidate_inputs.jsonl'),(tr,va,j)):
                (ann/n).write_text(''.join(json.dumps(r)+'\n' for r in rr),encoding='utf-8');expected[n]=h.sha(ann/n)
            (ann/'manifest.json').write_text(json.dumps(dict(experiment_id=ci.EXP,status='CSBENCH_ANNOTATOR_INPUTS_READY_CONFIG_PENDING',training_records=345,validation_records=343,candidate_records=5000,automatic_semantic_errors_retained=True,human_review_performed=False,files_sha256=expected,source_sha256={})),encoding='utf-8')
            with patch.object(ci,'EXPECTED',expected):
                a,b,c,_=ci.inputs(root);self.assertEqual(a[5]['answer'],'O ♣1');self.assertEqual((len(a),len(b),len(c)),(345,343,5000))
                with (ann/'train.jsonl').open('a') as f:f.write('\n')
                with self.assertRaises(AssertionError):ci.inputs(root)

if __name__=='__main__':unittest.main()
