"""Local offline contract tests; no real network requests."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import types
from unittest import mock

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    here=Path(__file__).parent
    m=load('retry_under_test',here/'retry_reviewed_proxies.py')
    builder=load('review_builder',here.parent/'build_new_proxy_review_bundle.py')
    review=m.read(here/'decisions.json');new=[]
    for aid in builder.IDS:
        file=next((Path('C:/Users/wx/.codex/attachments')/aid).glob('*.txt'))
        for block in re.split(r'(?m)^## ',file.read_text(encoding='utf8'))[1:]:
            sid,body=block.split('\n',1);q,rest=body.split('Original:\n',1)
            original,rest=rest.split('Proxy:\n',1);proxy,_=rest.rsplit('Decision: PENDING',1)
            new.append({'sample_id':sid.strip(),'question':q.strip().removeprefix('Question:\n'),
                'original_answer':original.strip(),'proxy_answer':proxy.strip(),
                'assigned_label':None,'v2_origin':'new_generation'})
    old=[{'sample_id':'fixture_old_%s'%i,'v2_origin':'frozen_v1_alignment','assigned_label':None,
          'alignment_review_status':'NOT_INDIVIDUALLY_REVIEWED' if i<636 else 'PASS'} for i in range(885)]
    assert len(m.validate_records(old+new,review)[2])==6
    altered=[dict(x) for x in new];altered[0]['proxy_answer']+='changed'
    try:m.validate_records(old+altered,review)
    except ValueError:pass
    else:raise AssertionError('Text tamper accepted')
    with tempfile.TemporaryDirectory(prefix='proxy_review_test_') as td:
        root=Path(td);ctl=root/'control/scale-paper-protocol';ctl.mkdir(parents=True)
        shutil.copyfile(here.parent/'proxy_restored.py',ctl/'proxy_restored.py')
        p=load('fixture_contract',ctl/'proxy_restored.py')
        source=root/'paper_method_v2/artifacts/proxy_review_inputs_v1/proxy_records.jsonl'
        source.parent.mkdir(parents=True);source.write_bytes(p.jl(old+new));m.SOURCE_SHA=m.sha(source)
        fixture_review=dict(review,source_records_sha256=m.SOURCE_SHA)
        rp=root/'decisions.json';m.save(rp,fixture_review)
        plan=root/'paper_method_v2/artifacts/proxy_inputs_v1';plan.mkdir(parents=True)
        cfg={'model':'glm-5.3','base_url':'https://fixture.invalid/v4','max_tokens':16384,'stream':False,
             'script_sha256':m.sha(ctl/'proxy_restored.py')}
        m.save(plan/'config.json',cfg)
        requests=p.make_plan([{'sample_id':r['sample_id'],'q':r['question'],'a':r['original_answer']} for r in new])[0]
        (plan/'requests.jsonl').write_bytes(p.jl(requests))
        m.save(plan/'summary.json',{'files_sha256':{n:m.sha(plan/n) for n in ['config.json','requests.jsonl']}})
        count=[0]
        class Reply:
            def __init__(self,body):self.body=body
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return self.body
        def urlopen(request,timeout):
            count[0]+=1
            if count[0]==1:raise TimeoutError('fixture')
            body=json.loads(request.data)
            original=json.loads(body['messages'][1]['content'])['original_answer']
            return Reply(m.enc({'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'proxy_answer':original})}}],'model':'glm-5.3'}))
        fcntl=types.SimpleNamespace(flock=lambda *args:None,LOCK_EX=1,LOCK_NB=2)
        with mock.patch.dict(sys.modules,{'fcntl':fcntl}), mock.patch.object(m.urllib.request,'urlopen',side_effect=urlopen), mock.patch.object(m.getpass,'getpass',return_value='fixture-key'), mock.patch.object(m.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
            assert m.run(root,rp,True,100,900)==0 and count[0]==0
            assert m.run(root,rp,False,100,900)==0 and count[0]==7
            assert m.run(root,rp,False,100,900)==0 and count[0]==7
        out=root/'paper_method_v2/runs/proxy_glm_v1/new_semantic_retry_v1'
        summary=m.read(out/'summary.json')
        assert summary['new_api_calls']==0 and summary['valid_retry_responses']==6
        assert not summary['ready_for_annotator_training'] and summary['labels_assigned']==0
        assert all(x['semantic_decision']=='PENDING' for x in m.read(out/'candidates.json'))
        assert m.sha(source)==m.SOURCE_SHA
        source.write_bytes(source.read_bytes()+b' ')
        try:m.prepare(root,rp)
        except ValueError:pass
        else:raise AssertionError('Source hash tamper accepted')
    print('PASS: 222-pair coverage, text/source tamper rejection, dry-run, timeout retry, six-response export, zero-call resume, no source/label mutation')

if __name__=='__main__':main()
