"""Resumable Ps generation with durable attempts and bounded retries."""
import argparse
from collections import Counter
import getpass
import importlib.util
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def load(path):
    s=importlib.util.spec_from_file_location('restored_ps',path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(data);tmp.replace(path)


def transient(exc):
    if isinstance(exc,urllib.error.HTTPError):return exc.code==429 or 500<=exc.code<=599
    return isinstance(exc,(TimeoutError,urllib.error.URLError,ConnectionError))


def execute(root,attempt_limit,timeout):
    p=load(root/'control/scale-paper-protocol/ps_restored.py')
    base=root/'paper_method_v2';folder=base/'artifacts/ps_inputs_v1'
    plan=p.read(folder/'summary.json')
    for name,h in plan['files_sha256'].items():
        if p.sha((folder/name).read_bytes())!=h:raise ValueError('Changed synthesis plan: '+name)
    cfg=p.read(folder/'config.json')
    if cfg['script_sha256']!=p.sha(Path(p.__file__).read_bytes()):raise ValueError('Plan builder changed')
    for rel,h in cfg['upstream_sha256'].items():
        if p.sha((root/rel).read_bytes())!=h:raise ValueError('Upstream artifact changed: '+rel)
    requests=p.lines(folder/'requests.jsonl')
    if len({r['request_id'] for r in requests})!=len(requests):raise ValueError('Duplicate request ID')
    if len(requests)*5<5000:raise ValueError('Candidate slots below the planned initial-filter budget')
    run=base/'runs/ps_glm_v1';run.mkdir(parents=True,exist_ok=True)
    policy={'runner_sha256':p.sha(Path(__file__).read_bytes()),'attempt_limit_per_request':attempt_limit,
            'timeout_seconds':timeout,'retry_pause_seconds':15,'prompt_and_generation_body_unchanged':True,
            'resume_uncertain_requests':True,'note':'An uncertain prior network attempt may be processed or billed again'}
    p.save(run,{'config.json':p.enc(cfg),'plan_hash.json':p.enc({'sha256':p.sha((folder/'summary.json').read_bytes())}),
                'execution_policy.json':p.enc(policy)})
    candidates=[];errors=[];calls=0;key=None;usage=Counter();stop=False
    def finalize():
        duplicates=Counter(r['answer'] for r in candidates)
        for r in candidates:r['exact_duplicate_count_in_generated_pool']=duplicates[r['answer']]
        done=len(candidates)//5;complete=done==len(requests) and not errors
        summary={'status':'PS_GENERATION_COMPLETE_UNLABELED' if complete else 'PS_GENERATION_PARTIAL',
            'planned_requests':len(requests),'valid_requests':done,'candidates':len(candidates),
            'new_api_calls_this_invocation':calls,'recorded_usage':dict(usage),
            'usage_scope':'All saved API envelopes inspected this invocation, including rejected responses; timed-out usage unknown',
            'errors':errors,'assigned_labels':0,'training_started':False,'test_accessed':False,
            'validation_accessed':False,'api_model_replacement':'glm-5.3 as previously authorized',
            'execution_policy_sha256':p.sha((run/'execution_policy.json').read_bytes())}
        atomic(run/'summary.json',p.enc(summary));atomic(run/'candidates.jsonl',p.jl(candidates))
        if complete:
            p.save(base/'artifacts/ps_generated_v1',{'candidates.jsonl':p.jl(candidates),
                'manifest.json':p.enc({'candidate_sha256':p.sha(p.jl(candidates)),
                  'plan_sha256':p.sha((folder/'summary.json').read_bytes()),'assigned_labels':0})})
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
        print('ACTUAL_NETWORK_REQUESTS_THIS_INVOCATION:',calls,flush=True)
        return 0 if complete else 2
    def count(raw):
        if not isinstance(raw,dict):return
        for k in ('prompt_tokens','completion_tokens','total_tokens'):
            usage[k]+=(raw.get('usage') or {}).get(k,0) or 0
    for position,request in enumerate(requests,1):
        rid=request['request_id']
        if not rid.startswith('draw_') or not rid[5:].isdigit():raise ValueError('Invalid request ID')
        canonical=run/'responses'/(rid+'.json');valid=None;last_error='Attempts exhausted'
        if canonical.exists():
            raw=p.read(canonical);valid=p.parse_response(raw,request);count(raw)
        elif not stop:
            history=run/'attempt_history'/rid
            for attempt in range(1,attempt_limit+1):
                response=history/('attempt_%03d.response.json'%attempt)
                intent=history/('attempt_%03d.request.json'%attempt)
                if not response.exists():
                    if intent.exists():
                        last_error='Uncertain previous attempt retained; advancing to next attempt'
                        continue
                    if key is None:
                        key=getpass.getpass('GLM API Key (hidden): ').strip()
                        if not key:raise ValueError('Empty API key')
                    body={'model':cfg['model'],'messages':request['messages'],
                          'max_tokens':cfg['max_tokens'],'stream':False}
                    p.save(history,{intent.name:p.enc({'request_id':rid,'attempt':attempt,
                        'time':time.time(),'body_sha256':p.sha(p.enc(body))})})
                    req=urllib.request.Request(cfg['base_url'].rstrip('/')+'/chat/completions',
                        data=json.dumps(body).encode('utf8'),headers={'Content-Type':'application/json',
                        'Authorization':'Bearer '+key},method='POST')
                    calls+=1
                    try:
                        with urllib.request.urlopen(req,timeout=timeout) as reply:
                            wire=reply.read()
                        p.save(history,{'attempt_%03d.body.bin'%attempt:wire})
                        raw=json.loads(wire)
                        p.save(history,{response.name:p.enc(raw)})
                    except Exception as exc:
                        last_error=type(exc).__name__
                        code=getattr(exc,'code',None)
                        p.save(history,{'attempt_%03d.error.json'%attempt:p.enc({
                            'exception':last_error,'http_status':code,'request_id':rid})})
                        print('REQUEST_RETRY',rid,attempt,last_error,code,flush=True)
                        if not transient(exc) and not isinstance(exc,(json.JSONDecodeError,UnicodeDecodeError)):
                            stop=True;break
                        time.sleep(15);continue
                raw=p.read(response);count(raw)
                try:valid=p.parse_response(raw,request)
                except (ValueError,KeyError,TypeError,IndexError) as exc:
                    last_error=str(exc)
                    print('CONTENT_RETRY',rid,attempt,last_error,flush=True)
                    time.sleep(3);continue
                p.save(canonical.parent,{canonical.name:p.enc(raw)})
                p.save(run/'accepted_provenance',{rid+'.json':p.enc({'response_sha256':p.sha(response.read_bytes()),
                    'source_response':str(response.relative_to(root)),'request_id':rid})})
                print('SAVED',rid,flush=True);time.sleep(3);break
        if valid is None:
            errors.append({'request_id':rid,'error':last_error if not stop else 'Network stopped; inspect attempt_history'})
        else:candidates.extend(valid)
        if position%10==0 or position==len(requests):
            atomic(run/'progress.json',p.enc({'processed_requests':position,'planned_requests':len(requests),
                'valid_requests':len(candidates)//5,'new_api_calls':calls,'errors':len(errors)}))
            print('PS_PROGRESS',position,'/',len(requests),'valid',len(candidates)//5,'calls',calls,flush=True)
    return finalize()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--attempts',type=int,default=20);p.add_argument('--timeout',type=int,default=900)
    a=p.parse_args()
    if not 1<=a.attempts<=100 or not 180<=a.timeout<=1800:p.error('attempts 1..100; timeout 180..1800')
    root=a.root.resolve();run=root/'paper_method_v2/runs/ps_glm_v1';run.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (run/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raise SystemExit(execute(root,a.attempts,a.timeout))
