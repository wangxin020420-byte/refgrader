"""CSBench SCALE training proxy generation. No candidate labels or test input access."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time

CONTRACT='e5885c140ba0317557b4415ce718d8654363a7704c0966be4c90465fbe609b65'
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
    p=Path(__file__).with_name('proxy_contract.py')
    if sha(p.read_bytes())!=CONTRACT:raise ValueError('Generation contract changed')
    spec=importlib.util.spec_from_file_location('ps_contract',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def prepare(project,experiment,draws=1200,seed=42):
    m=contract();exp=project/'runs/online_experiments'/experiment
    stage4=exp/'scale_stage4';summary=read(stage4/'summary.json')
    selected=stage4/'layout/paper_method_v2/artifacts/initial_filter_k5_b5000_v1'
    if summary['status']!='INITIAL_FILTER_COMPLETE_UNLABELED' or summary['selected_candidates']!=5000 or summary['assigned_labels']!=0:
        raise ValueError('Initial filter incomplete or labeled')
    for name,h in summary['files_sha256'].items():
        if sha((selected/name).read_bytes())!=h:raise ValueError('Selection changed: '+name)
    if sha((selected/'config.json').read_bytes())!=summary['config_sha256']:raise ValueError('Selection config changed')
    retr=exp/'scale_stage2/layout/paper_method_v2/artifacts/retrieval_v1'
    src=retr/'training_quadruples.json';meta=read(retr/'summary.json')
    if sha(src.read_bytes())!=meta['output_sha256'][src.name]:raise ValueError('Training source changed')
    sources=read(src)
    if len(sources)!=345 or len({x['sample_id'] for x in sources})!=345:raise ValueError('Expected 345 unique training sources')
    requests,blanks=m.make_plan(sources)
    cfg=dict(experiment_id=experiment,model='glm-5.3',base_url=URL,max_tokens=16384,stream=False,
        contract_sha256=CONTRACT,runner_sha256=sha(Path(__file__).read_bytes()),
        sources_sha256=sha(src.read_bytes()),initial_filter_summary_sha256=sha((stage4/'summary.json').read_bytes()),
        label_policy='Source scores are provenance only; proxy label transfer requires semantic-equivalence assumption. No candidate labels assigned.',
        prompt_basis='Preserved ADS Pr reconstruction, not author-provided prompt',
        timeout_seconds=900,validation_accessed=False,test_accessed=False)
    out=exp/'scale_stage5'
    blobs={'config.json':enc(cfg),'requests.jsonl':jl(requests),'training_sources.json':enc(sources),'blank_sources.json':enc(blanks)}
    for n,b in blobs.items():freeze(out/n,b)
    report=dict(status='CSBENCH_PROXY_PLAN_READY',training_records=len(sources),planned_requests=len(requests),
        blank_passthrough_records=len(blanks),api_calls=0,assigned_labels=0,files_sha256={n:sha(b) for n,b in blobs.items()})
    freeze(out/'plan_summary.json',enc(report));print(json.dumps(report,ensure_ascii=False),flush=True)
    lookup={x['sample_id']:x for x in sources}
    def parse_response(raw,r):
        answer,fence=m.parse(raw);x=lookup[r['sample_id']]
        return [dict(sample_id=x['sample_id'],question_id=x['question_id'],question=x['q'],references=x['r'],
            original_answer=x['a'],proxy_answer=answer,source_score=x['l'],assigned_label=None,
            semantic_equivalence_verified=False,kind='generated_proxy',request_id=r['request_id'],
            returned_model=raw.get('model'),response_id=raw.get('id'),outer_fence_removed=fence)]
    m.parse_response=parse_response
    return m,out,requests,cfg

def execute(project,experiment,draws=1200,seed=42,attempts=8,limit=0,client=None,sleep=time.sleep):
    m,out,requests,cfg=prepare(project,experiment,draws,seed)
    fingerprint=sha(enc(cfg));candidates=[];errors=[];calls=0;processed=0;valid=0;fatal=False
    def summary():
        report=dict(status='CSBENCH_PROXY_COMPLETE_REVIEW_REQUIRED' if valid==len(requests) else 'PROXY_PARTIAL',
            planned_requests=len(requests),valid_requests=valid,remaining=len(requests)-valid,generated_proxy_records=len(candidates),
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
                print('PROXY_ERROR',json.dumps(e,ensure_ascii=False),flush=True)
                if status in (401,403,404,422):fatal=True
                if status==400 or fatal or attempt==attempts:errors.append(e);break
                sleep(min(60,5*2**(attempt-1)))
        print('PROXY',valid,'/',len(requests),'CANDIDATES',len(candidates),'CALLS',calls,flush=True)
        summary()
    report=summary()
    if valid==len(requests):
        sources=read(out/'training_sources.json')
        by_id={x['sample_id']:x for x in candidates}
        for x in sources:
            if not x['a'].strip():
                by_id[x['sample_id']]=dict(sample_id=x['sample_id'],question_id=x['question_id'],question=x['q'],
                    references=x['r'],original_answer=x['a'],proxy_answer=x['a'],source_score=x['l'],assigned_label=None,
                    semantic_equivalence_verified=True,kind='blank_passthrough')
        if len(by_id)!=len(sources) or len(candidates)!=len(requests):raise ValueError('Proxy coverage mismatch')
        ordered=[by_id[x['sample_id']] for x in sources]
        raw=jl(ordered);freeze(out/'proxy_records.jsonl',raw)
        report.update(training_records=len(sources),proxy_records=len(ordered),blank_passthrough_records=len(sources)-len(requests),
            semantic_review_required=True,ready_for_annotator_training=False,
            files_sha256={'proxy_records.jsonl':sha(raw)})
        review=[]
        for r in ordered:
            review.append(dict(sample_id=r['sample_id'],question=r['question'],original_answer=r['original_answer'],
                proxy_answer=r['proxy_answer'],status='EXACT_PASSTHROUGH' if r['kind']=='blank_passthrough' else 'UNREVIEWED'))
        freeze(out/'review_pairs.json',enc(review))
        save(out/'summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return 0 if valid==len(requests) else 2

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--project',required=True,type=Path);p.add_argument('--experiment-id',default='csbench_scale_online_v1')
    p.add_argument('--attempts',type=int,default=8);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.attempts<1 or a.limit<0 or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}',a.experiment_id):p.error('Invalid arguments')
    project=a.project.resolve();out=project/'runs/online_experiments'/a.experiment_id/'scale_stage5';out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.command=='prepare':prepare(project,a.experiment_id);return
        raise SystemExit(execute(project,a.experiment_id,attempts=a.attempts,limit=a.limit))

if __name__=='__main__':main()
