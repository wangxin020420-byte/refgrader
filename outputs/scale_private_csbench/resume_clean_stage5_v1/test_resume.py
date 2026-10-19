import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import resume

path=Path(__file__).resolve().parents[1]/'scale_csbench_stage5_v1/run_stage5.py'
spec=importlib.util.spec_from_file_location('stage5',path);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
contract=r.contract()
lookup={}
def parse_response(raw,req):
    answer,fence=contract.parse(raw)
    x=lookup[req['sample_id']]
    return [dict(sample_id=x['sample_id'],proxy_answer=answer,question='q',original_answer='old',kind='generated_proxy')]
m=SimpleNamespace(parse_response=parse_response)

class Tests(unittest.TestCase):
    def test_match_strict(self):
        cfg={k:k for k in resume.KEYS};req={'messages':[{'content':'old'}],'provenance':[1]}
        self.assertTrue(resume.same_request(req,copy.deepcopy(req),cfg,cfg))
        other=copy.deepcopy(req);other['messages'][0]['content']='new'
        self.assertFalse(resume.same_request(req,other,cfg,cfg))
        self.assertFalse(resume.same_request(req,req,cfg,dict(cfg,max_tokens=1)))
        self.assertFalse(resume.same_request(req,dict(req,provenance=[2]),cfg,cfg))
    def test_integration_resume(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);old=root/'old';out=root/'new';old.mkdir();out.mkdir()
            oc={k:k for k in resume.KEYS};oc['experiment_id']=resume.OLD;cfg=dict(oc,experiment_id=resume.NEW)
            oldreq=[]
            for i in range(3):
                oldreq.append(dict(request_id=f'proxy_{i:05}',sample_id=f's{i}',messages=[{'role':'user','content':'old'+str(i)}],provenance=[{'variant_id':f'{i}_{v}'} for v in range(5)]))
            requests=copy.deepcopy(oldreq);requests[1]['messages'][0]['content']='changed1';requests[2]['messages'][0]['content']='changed2'
            def response(req):return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'proxy_answer':'automatic output'})}}]}
            for name,blob in {'config.json':r.enc(oc),'requests.jsonl':r.jl(oldreq),'training_sources.json':r.enc([dict(sample_id=f's{i}',a='old') for i in range(3)]),'blank_sources.json':b'[]','proxy_records.jsonl':b'[]'}.items():(old/name).write_bytes(blob)
            r.save(old/'plan_summary.json',{'files_sha256':{n:r.sha((old/n).read_bytes()) for n in ['config.json','requests.jsonl','training_sources.json','blank_sources.json']}})
            oh=r.sha(r.enc(oc));r.save(old/'summary.json',dict(status='CSBENCH_PROXY_COMPLETE_REVIEW_REQUIRED',remaining=0,config_sha256=oh,files_sha256={'proxy_records.jsonl':r.sha(b'[]')}))
            for req in oldreq:
                folder=old/'responses'/req['request_id'];raw=folder/'saved.response.json'
                r.save(raw,dict(config_sha256=oh,request_id=req['request_id'],response=response(req)))
                r.save(folder/'accepted.json',dict(config_sha256=oh,request_id=req['request_id'],response_file=raw.name,response_sha256=r.sha(raw.read_bytes())))
            original={str(p.relative_to(old)):p.read_bytes() for p in old.rglob('*') if p.is_file()}
            lookup.update({f's{i}':dict(sample_id=f's{i}') for i in range(3)})
            resume.install(r,m,old,out,requests,cfg)
            (out/'training_sources.json').write_bytes((old/'training_sources.json').read_bytes())
            self.assertEqual(r.read(out/'reuse_audit.json')['reused_requests'],1)
            calls=[]
            def create(**kwargs):
                calls.append(kwargs);req=next(x for x in requests if x['messages']==kwargs['messages'])
                return SimpleNamespace(model_dump=lambda **kw:response(req))
            client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            with patch.object(r,'prepare',return_value=(m,out,requests,cfg)):
                self.assertEqual(r.execute(root,resume.NEW,client=client),0)
                self.assertEqual(len(calls),2)
                resume.install(r,m,old,out,requests,cfg)
                self.assertEqual(r.execute(root,resume.NEW,client=client),0)
                self.assertEqual(len(calls),2)
            self.assertEqual(r.read(out/'summary.json')['proxy_records'],3)
            self.assertEqual(original,{str(p.relative_to(old)):p.read_bytes() for p in old.rglob('*') if p.is_file()})

if __name__=='__main__':unittest.main()
