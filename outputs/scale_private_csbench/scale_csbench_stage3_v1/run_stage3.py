"""CSBench SCALE weighted style generation. No candidate labels or test input access."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time

CONTRACT='fb12722c1535350cf5e9506ba4297f239943bacdb14877f501882ba8b1e57a4d'
URL='https://open.bigmodel.cn/api/coding/paas/v4/'
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def jl(x):return ''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in x).encode()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(enc(x));tmp.replace(p)
def freeze(p,b):
    if p.exists():
        if p.read_bytes()!=b:raise ValueError('Frozen artifact changed: '+str(p))
    else:
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(b)
def contract():
    p=Path(__file__).with_name('generation_contract.py')
    if sha(p.read_bytes())!=CONTRACT:raise ValueError('Generation contract changed')
    spec=importlib.util.spec_from_file_location('ps_contract',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def prepare(project,experiment,draws,seed):
    m=contract();exp=project/'runs/online_experiments'/experiment
    stage2=exp/'scale_stage2';art=stage2/'layout/paper_method_v2/artifacts'
    retrieval=art/'retrieval_v1';rep=art/'replacement_inputs_v1'
    complete=read(stage2/'summary.json');rmeta=read(retrieval/'summary.json');pmeta=read(rep/'summary.json')
    if complete['status']!='CSBENCH_RETRIEVAL_RANKING_REPLACEMENT_COMPLETE' or complete['training_records']!=345 or complete!=pmeta:
        raise ValueError('Wrong or incomplete stage2')
    paths=[retrieval/'training_quadruples.json',retrieval/'sampling_weights.json',rep/'replacement_records.jsonl']
    for p in paths:
        expected=(pmeta if p.parent==rep else rmeta)['output_sha256'][p.name]
        if sha(p.read_bytes())!=expected:raise ValueError('Upstream changed: '+str(p))
    quads=read(paths[0]);weights=read(paths[1]);replacement=rows(paths[2])
    if len(quads)!=345 or len(weights)!=345 or len({q['sample_id'] for q in quads})!=345:
        raise ValueError('Training source coverage mismatch')
    if any(not math.isfinite(w['probability']) or w['probability']<=0 for w in weights) or not math.isclose(sum(w['probability'] for w in weights),1,abs_tol=1e-12):
        raise ValueError('Invalid weights')
    requests,skips=m.make_plan(quads,weights,replacement,draws,seed)
    cfg=dict(experiment_id=experiment,model='glm-5.3',base_url=URL,max_tokens=16384,stream=False,
        draws=draws,seed=seed,variants_per_draw=5,contract_sha256=CONTRACT,
        runner_sha256=sha(Path(__file__).read_bytes()),
        upstream_sha256={str(p.relative_to(exp)):sha(p.read_bytes()) for p in paths},
        stage2_summary_sha256=sha((stage2/'summary.json').read_bytes()),
        choices=['Original independently reconstructed Ps prompt/rule bank retained, not author templates',
                 'Weighted draws with replacement; 1200 default is prior ADS execution budget, not paper-verified',
                 'Two original-input and three replacement-input variants per draw',
                 'Blank draws skipped without resampling; no-triplet sources use explicit original-input fallback',
                 'No score or correctness assumption for replacements or rewrites',
                 'GLM replaces paper generator; provider decoding defaults retained'],
        validation_accessed=False,test_accessed=False)
    out=exp/'scale_stage3'
    blobs={'config.json':enc(cfg),'requests.jsonl':jl(requests),'skipped_draws.json':enc(skips),'rule_bank.json':enc(m.RULES)}
    for n,b in blobs.items():
        if (out/n).exists() and (out/n).read_bytes()!=b:raise ValueError('Cannot mix generation plans: '+n)
    for n,b in blobs.items():freeze(out/n,b)
    report=dict(status='CSBENCH_GENERATION_PLAN_READY',draws=draws,planned_requests=len(requests),
        candidate_slots=5*len(requests),skipped_blank_draws=len(skips),training_records=345,
        api_calls=0,assigned_labels=0,files_sha256={n:sha(b) for n,b in blobs.items()})
    freeze(out/'plan_summary.json',enc(report))
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return m,out,requests,cfg

def execute(project,experiment,draws=1200,seed=42,attempts=8,limit=0,client=None,sleep=time.sleep):
    m,out,requests,cfg=prepare(project,experiment,draws,seed)
    fingerprint=sha(enc(cfg));candidates=[];errors=[];calls=0;processed=0;valid=0;fatal=False
    def summary():
        report=dict(status='CSBENCH_PS_GENERATION_COMPLETE_UNLABELED' if valid==len(requests) else 'PS_PARTIAL',
            planned_requests=len(requests),valid_requests=valid,remaining=len(requests)-valid,candidates=len(candidates),
            new_api_calls=calls,assigned_labels=0,errors=errors,config_sha256=fingerprint,
            validation_accessed=False,test_accessed=False,training_started=False,semantic_accuracy_verified=False,
            reproduction_status='METHOD_RECONSTRUCTION_WITH_DOCUMENTED_DEVIATIONS')
        save(out/'summary.json',report);return report
    def parse_saved(path,r):
        saved=read(path)
        if saved['config_sha256']!=fingerprint or saved['request_id']!=r['request_id']:
            raise RuntimeError('Response provenance mismatch')
        return m.parse_response(saved['response'],r)
    for r in requests:
        folder=out/'responses'/r['request_id'];folder.mkdir(parents=True,exist_ok=True)
        accepted=folder/'accepted.json'
        if accepted.exists():
            saved=read(accepted);path=folder/saved['response_file']
            if saved['config_sha256']!=fingerprint or saved['request_id']!=r['request_id'] or sha(path.read_bytes())!=saved['response_sha256']:
                raise ValueError('Accepted cache changed')
            candidates.extend(parse_saved(path,r));valid+=1;continue
        recovered=False
        for path in sorted(folder.glob('*.response.json'),reverse=True):
            try:parsed=parse_saved(path,r)
            except (ValueError,KeyError,TypeError,IndexError):continue
            save(accepted,dict(config_sha256=fingerprint,request_id=r['request_id'],response_file=path.name,response_sha256=sha(path.read_bytes())))
            candidates.extend(parsed);valid+=1;recovered=True;break
        if recovered:continue
        if fatal or (limit and processed>=limit):continue
        previous=[read(p) for p in sorted(folder.glob('*.error.json'))]
        blocked=[e for e in previous if e.get('http_status') in (400,401,403,404,422)]
        if blocked:errors.append(blocked[-1]);continue
        if client is None:
            from openai import OpenAI
            key=os.environ.get('ZHIPUAI_API_KEY')
            if not key:raise ValueError('Set ZHIPUAI_API_KEY')
            client=OpenAI(api_key=key,base_url=URL,timeout=900,max_retries=0)
        processed+=1
        for attempt in range(1,attempts+1):
            token=str(time.time_ns())
            try:
                calls+=1
                response=client.chat.completions.create(model=cfg['model'],max_tokens=cfg['max_tokens'],stream=False,messages=r['messages']).model_dump(mode='json')
                path=folder/(token+'.response.json')
                save(path,dict(config_sha256=fingerprint,request_id=r['request_id'],response=response))
                parsed=m.parse_response(response,r)
                save(accepted,dict(config_sha256=fingerprint,request_id=r['request_id'],response_file=path.name,response_sha256=sha(path.read_bytes())))
                candidates.extend(parsed);valid+=1;break
            except Exception as exc:
                status=getattr(exc,'status_code',None)
                e=dict(request_id=r['request_id'],attempt=attempt,exception_type=type(exc).__name__,http_status=status)
                if isinstance(exc,(ValueError,KeyError,IndexError,TypeError)):
                    e['validation_message']=str(exc)[:300]
                body=getattr(exc,'body',None)
                if isinstance(body,dict):
                    detail=body.get('error',body)
                    if isinstance(detail,dict) and isinstance(detail.get('code'),(int,str)):e['service_code']=str(detail['code'])[:80]
                save(folder/(token+'.error.json'),e)
                print('PS_ERROR',json.dumps(e,ensure_ascii=False),flush=True)
                if status in (401,403,404,422):fatal=True
                if status==400 or fatal or attempt==attempts:errors.append(e);break
                sleep(min(60,5*2**(attempt-1)))
        print('PS',valid,'/',len(requests),'CANDIDATES',len(candidates),'CALLS',calls,flush=True)
        summary()
    report=summary()
    if valid==len(requests):
        if len(candidates)!=5*len(requests) or len({c['variant_id'] for c in candidates})!=len(candidates):raise ValueError('Candidate coverage')
        raw=jl(candidates);freeze(out/'candidates.jsonl',raw)
        report['files_sha256']={'candidates.jsonl':sha(raw)}
        save(out/'summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return 0 if valid==len(requests) else 2

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--project',required=True,type=Path);p.add_argument('--experiment-id',default='csbench_scale_online_v1')
    p.add_argument('--draws',type=int,default=1200);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--attempts',type=int,default=8);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if min(a.draws,a.attempts)<1 or a.limit<0 or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}',a.experiment_id):p.error('Invalid arguments')
    project=a.project.resolve();out=project/'runs/online_experiments'/a.experiment_id/'scale_stage3';out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.command=='prepare':prepare(project,a.experiment_id,a.draws,a.seed);return
        raise SystemExit(execute(project,a.experiment_id,a.draws,a.seed,a.attempts,a.limit))

if __name__=='__main__':main()
