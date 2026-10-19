import tempfile
import unittest
from pathlib import Path
import run_stage5 as m

class Tests(unittest.TestCase):
    def test_generation_resume_and_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            project=Path(d);exp=project/'runs/online_experiments/csbench_scale_online_v1'
            retr=exp/'scale_stage2/layout/paper_method_v2/artifacts/retrieval_v1'
            sources=[dict(sample_id=str(i),question_id='Q',q='Question',r=['Reference'],a='' if i<5 else 'wrong answer',l=2) for i in range(345)]
            m.save(retr/'training_quadruples.json',sources)
            m.save(retr/'summary.json',dict(output_sha256={'training_quadruples.json':m.sha((retr/'training_quadruples.json').read_bytes())}))
            stage4=exp/'scale_stage4';sel=stage4/'layout/paper_method_v2/artifacts/initial_filter_k5_b5000_v1'
            m.save(sel/'config.json',{})
            m.save(sel/'selected_candidates.jsonl',{})
            m.save(stage4/'summary.json',dict(status='INITIAL_FILTER_COMPLETE_UNLABELED',selected_candidates=5000,assigned_labels=0,
                config_sha256=m.sha((sel/'config.json').read_bytes()),files_sha256={'selected_candidates.jsonl':m.sha((sel/'selected_candidates.jsonl').read_bytes())}))
            class Reply:
                def model_dump(self,**kw):return {'choices':[{'finish_reason':'stop','message':{'content':'{"proxy_answer":"wrong answer rewritten"}'}}]}
            class Client:
                def __init__(self):self.chat=self;self.completions=self;self.calls=0
                def create(self,**kw):
                    self.calls+=1
                    assert kw['model']=='glm-5.3'
                    assert 'source_score' not in str(kw['messages'])
                    return Reply()
            c=Client()
            self.assertEqual(m.execute(project,'csbench_scale_online_v1',client=c,sleep=lambda _:None),0)
            self.assertEqual(c.calls,340)
            records=m.rows(exp/'scale_stage5/proxy_records.jsonl')
            self.assertEqual(len(records),345)
            self.assertTrue(all(r['assigned_label'] is None for r in records))
            self.assertEqual(sum(r['kind']=='blank_passthrough' for r in records),5)
            self.assertEqual(m.execute(project,'csbench_scale_online_v1',client=c,sleep=lambda _:None),0)
            self.assertEqual(c.calls,340)
            accepted=next((exp/'scale_stage5/responses').glob('*/accepted.json'))
            v=m.read(accepted);v['response_sha256']='broken';m.save(accepted,v)
            with self.assertRaises(ValueError):m.execute(project,'csbench_scale_online_v1',client=c,sleep=lambda _:None)
    def test_contract(self):
        c=m.contract()
        with self.assertRaises(ValueError):c.parse({'choices':[{'finish_reason':'length','message':{'content':'{}'}}]})
        with self.assertRaises(ValueError):c.parse({'choices':[{'finish_reason':'stop','message':{'content':'{"proxy_answer":""}'}}]})
if __name__=='__main__':unittest.main()
