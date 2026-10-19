import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import run_stage4 as a

def module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def fixture(root):
    exp=root/'runs/online_experiments/csbench_scale_online_v1'
    ret=exp/'scale_stage2/layout/paper_method_v2/artifacts/retrieval_v1';gen=exp/'scale_stage3'
    sources=[dict(sample_id=str(i),a='' if i<5 else 'answer',q='Q',r=['R'],l=5) for i in range(345)]
    raw=a.enc(sources);a.freeze(ret/'training_quadruples.json',raw)
    a.freeze(ret/'summary.json',a.enc(dict(output_sha256={'training_quadruples.json':a.sha(raw)})))
    req=[];candidates=[]
    for i in range(2):
        variants=[dict(variant_id=str(i)+'_'+str(j)) for j in range(5)]
        req.append(dict(request_id=str(i),sample_id='5',provenance=variants))
        candidates.extend(dict(v,request_id=str(i),sample_id='5',answer='candidate',assigned_label=None) for v in variants)
    jl=lambda r:''.join(json.dumps(x)+'\n' for x in r).encode()
    blobs={'requests.jsonl':jl(req),'config.json':a.enc({'fixture':True})}
    for n,b in blobs.items():a.freeze(gen/n,b)
    a.freeze(gen/'plan_summary.json',a.enc(dict(candidate_slots=10,planned_requests=2,files_sha256={n:a.sha(b) for n,b in blobs.items()})))
    raw=jl(candidates);a.freeze(gen/'candidates.jsonl',raw)
    a.freeze(gen/'summary.json',a.enc(dict(status='CSBENCH_PS_GENERATION_COMPLETE_UNLABELED',remaining=0,errors=[],candidates=10,valid_requests=2,config_sha256=a.sha(blobs['config.json']),files_sha256={'candidates.jsonl':a.sha(raw)})))
    return exp

class Tests(unittest.TestCase):
    def test_bridge_provenance_and_resume(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);exp=fixture(root)
            before={str(p.relative_to(exp)):p.read_bytes() for p in exp.rglob('*') if p.is_file()}
            out,layout,_=a.prepare(root,'csbench_scale_online_v1',5,5)
            texts,records,facts=module('filter_embeddings_restored').load_inputs(layout)
            self.assertEqual((facts['training_records'],facts['candidate_records'],len(records)),(345,10,355))
            self.assertEqual(facts['blank_original_answers'],5)
            self.assertEqual(len(texts),3)
            a.prepare(root,'csbench_scale_online_v1',5,5)
            self.assertTrue(all((exp/n).read_bytes()==raw for n,raw in before.items()))
            with self.assertRaises(ValueError):a.prepare(root,'csbench_scale_online_v1',4,5)

    def test_original_filter_resume_and_unique_selection(self):
        m=module('initial_filter_restored')
        rng=np.random.default_rng(42)
        original=rng.normal(size=(6,4));original/=np.linalg.norm(original,axis=1,keepdims=True)
        candidates=rng.normal(size=(12,4));candidates/=np.linalg.norm(candidates,axis=1,keepdims=True)
        full=m.fresh_state(6);list(m.selection_steps(original,candidates,3,10,full))
        resumed=m.fresh_state(6)
        iterator=m.selection_steps(original,candidates,3,10,resumed)
        for _ in range(4):next(iterator)
        del iterator
        list(m.selection_steps(original,candidates,3,10,resumed))
        self.assertEqual(full,resumed)
        self.assertEqual(len({x['candidate_index'] for x in full['trace']}),10)

if __name__=='__main__':unittest.main()
