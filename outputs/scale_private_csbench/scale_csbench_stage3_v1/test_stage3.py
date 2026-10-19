import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import run_stage3 as a

def fixture(root):
    s=root/'runs/online_experiments/csbench_scale_online_v1/scale_stage2'
    art=s/'layout/paper_method_v2/artifacts';ret=art/'retrieval_v1';rep=art/'replacement_inputs_v1'
    quads=[dict(sample_id=str(i),q='Q',r=['R'],a='' if i<5 else 'original',l=7) for i in range(345)]
    weights=[dict(sample_id=q['sample_id'],probability=1/345) for q in quads]
    replacements=[dict(sample_id=q['sample_id'],source_answer_sha256=a.sha(q['a'].encode()),record_id=q['sample_id'],modified_answer='changed',answer_changed=True) for q in quads if q['a']]
    blobs={'training_quadruples.json':a.enc(quads),'sampling_weights.json':a.enc(weights)}
    for n,b in blobs.items():a.freeze(ret/n,b)
    a.save(ret/'summary.json',dict(output_sha256={n:a.sha(b) for n,b in blobs.items()}))
    raw=a.jl(replacements);a.freeze(rep/'replacement_records.jsonl',raw)
    report=dict(status='CSBENCH_RETRIEVAL_RANKING_REPLACEMENT_COMPLETE',training_records=345,output_sha256={'replacement_records.jsonl':a.sha(raw)})
    a.save(rep/'summary.json',report);a.save(s/'summary.json',report)
    return s

class Tests(unittest.TestCase):
    def test_plan_resume_and_no_labels(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);s=fixture(root);before={str(p.relative_to(s)):p.read_bytes() for p in s.rglob('*') if p.is_file()}
            calls=[]
            def create(**kw):
                calls.append(kw)
                payload=json.loads(kw['messages'][1]['content'])
                self.assertEqual(set(payload),{'question','reference_answers','tasks'})
                result=dict(answers=[dict(variant_id=t['variant_id'],answer=t['input_answer']) for t in payload['tasks']])
                envelope={'choices':[dict(finish_reason='stop',message={'content':json.dumps(result)})]}
                return SimpleNamespace(model_dump=lambda **_:envelope)
            client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            self.assertEqual(a.execute(root,'csbench_scale_online_v1',draws=20,limit=2,client=client),2)
            self.assertEqual(a.execute(root,'csbench_scale_online_v1',draws=20,client=client),0)
            total=len(calls)
            self.assertEqual(a.execute(root,'csbench_scale_online_v1',draws=20,client=client),0)
            self.assertEqual(total,len(calls))
            out=s.parent/'scale_stage3'
            self.assertEqual(len(a.rows(out/'candidates.jsonl')),5*total)
            self.assertTrue(all(x['assigned_label'] is None for x in a.rows(out/'candidates.jsonl')))
            next(out.glob('responses/*/accepted.json')).unlink()
            self.assertEqual(a.execute(root,'csbench_scale_online_v1',draws=20,client=client),0)
            self.assertEqual(total,len(calls))
            self.assertTrue(all((s/n).read_bytes()==b for n,b in before.items()))
            with self.assertRaises(ValueError):a.prepare(root,'csbench_scale_online_v1',21,42)

    def test_missing_variant_and_blank_skip(self):
        m=a.contract()
        req,skip=m.make_plan([dict(sample_id='b',q='Q',r=['R'],a='')],[dict(sample_id='b',probability=1)],[],3,42)
        self.assertEqual(req,[]);self.assertEqual(len(skip),3)
        request=dict(sample_id='a',request_id='d',provenance=[dict(variant_id='v')])
        with self.assertRaises(ValueError):m.parse_response({'choices':[dict(finish_reason='stop',message={'content':'{"answers":[]}'})]},request)

if __name__=='__main__':unittest.main()
