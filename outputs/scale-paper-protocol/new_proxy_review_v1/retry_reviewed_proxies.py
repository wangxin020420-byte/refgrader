"""Apply content-bound review ledger and regenerate only six held proxies.

Python 3.8+, standard library only. Never changes frozen source records or labels.
Successful new outputs require a separate semantic review before merging.
"""
import argparse
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request

SOURCE_SHA = 'ab2e840c049f6e1e8cfa818245983944c59985a0e60e415522a1d4f45a5f4d9a'
TARGETS = {'166','313','390','589','1070','1089'}

def enc(obj):
    return (json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def th(s):
    return hashlib.sha256(s.replace('\r\n','\n').strip().encode('utf8')).hexdigest()

def save(path,obj,mutable=False):
    data=enc(obj)
    if path.exists() and not mutable:
        if path.read_bytes()!=data:raise ValueError('Existing artifact differs: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(data);os.replace(str(tmp),str(path))

def validate_records(records,review):
    lookup={r['sample_id']:r for r in records}
    if len(records)!=1107 or len(lookup)!=1107:raise ValueError('Expected unique 1107 records')
    decisions=review['decisions'];ds={d['sample_id']:d for d in decisions}
    new={sid for sid,r in lookup.items() if r.get('v2_origin')=='new_generation'}
    if len(decisions)!=222 or len(ds)!=222 or set(ds)!=new:raise ValueError('Review coverage mismatch')
    if any(r.get('assigned_label') is not None for r in records):raise ValueError('Unexpected assigned label')
    held=[]
    for sid,d in ds.items():
        r=lookup[sid]
        for field,key in [('question','question_sha256'),('original_answer','original_sha256'),('proxy_answer','proxy_sha256')]:
            if th(r[field])!=d[key]:raise ValueError('Reviewed text mismatch: '+sid+' '+field)
        if d['decision'] not in ('PASS','FAIL','REVIEW_REQUIRED'):raise ValueError('Unknown decision')
        if d['decision']!='PASS':held.append(r)
    if {r['sample_id'].rsplit(':',1)[1] for r in held}!=TARGETS:raise ValueError('Retry targets mismatch')
    reused=[r for r in records if r.get('v2_origin')=='frozen_v1_alignment']
    if len(reused)!=885:raise ValueError('Expected 885 reused records')
    if sum(r.get('alignment_review_status')=='NOT_INDIVIDUALLY_REVIEWED' for r in reused)!=636:
        raise ValueError('Old unreviewed count changed')
    return lookup,ds,sorted(held,key=lambda r:int(r['sample_id'].rsplit(':',1)[1]))

def prepare(root,review_path):
    base=root/'paper_method_v2'
    source=base/'artifacts/proxy_review_inputs_v1/proxy_records.jsonl'
    review=read(review_path)
    if sha(source)!=SOURCE_SHA or review['source_records_sha256']!=SOURCE_SHA:
        raise ValueError('Frozen proxy records SHA mismatch; do not bypass')
    records=[json.loads(s) for s in source.read_text(encoding='utf8').splitlines() if s.strip()]
    lookup,ds,held=validate_records(records,review)
    plan=base/'artifacts/proxy_inputs_v1';meta=read(plan/'summary.json')
    for name,h in meta['files_sha256'].items():
        p=(plan/name).resolve()
        if plan.resolve() not in p.parents or sha(p)!=h:raise ValueError('Proxy plan changed: '+name)
    contract=root/'control/scale-paper-protocol/proxy_restored.py'
    cfg=read(plan/'config.json')
    if sha(contract)!=cfg['script_sha256']:raise ValueError('Proxy contract changed')
    spec=importlib.util.spec_from_file_location('review_proxy_contract',contract)
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    if cfg['model']!='glm-5.3' or cfg.get('stream') is not False:
        raise ValueError('Unexpected model configuration')
    reqs={r['sample_id']:r for r in p.rows(plan/'requests.jsonl')}
    requests=[]
    for row in held:
        sid=row['sample_id'];req=reqs[sid]
        # Keep original question and answer; no reference answer or score in prompt.
        original_messages=p.make_plan([{'sample_id':sid,'q':row['question'],'a':row['original_answer']}])[0][0]['messages']
        if req['messages']!=original_messages:raise ValueError('Original request mismatch: '+sid)
        caution=('本条进行保守的表达润色。不要推断作者本意，不要修正疑似笔误或技术术语，'
                 '不要补齐残句。宁可保留原句，也不要改变判断、否定范围、变量、公式、代码或操作。'
                 '本条前次润色需要注意的问题：'+ds[sid]['reason'])
        messages=[dict(x) for x in req['messages']]
        messages[0]['content']+='\n'+caution
        requests.append({'sample_id':sid,'body':{'model':cfg['model'],'messages':messages,
                         'max_tokens':cfg['max_tokens'],'stream':False}})
    return source,cfg,p,lookup,review,requests

def run(root,review_path,dry_run,attempts,timeout):
    source,cfg,p,lookup,review,requests=prepare(root,review_path)
    run_dir=root/'paper_method_v2/runs/proxy_glm_v1/new_semantic_retry_v1'
    if dry_run:
        print(json.dumps({'status':'NEW_PROXY_REVIEW_PREFLIGHT_OK','reviewed_new':222,'passed':216,
            'retry':6,'retry_ids':[x['sample_id'] for x in requests],
            'model':cfg['model'],'max_tokens':cfg['max_tokens'],'api_calls':0},ensure_ascii=False,indent=2))
        return 0
    run_dir.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (run_dir/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        policy={'source_records_sha256':SOURCE_SHA,'review_sha256':sha(review_path),
            'script_sha256':sha(Path(__file__)),'original_config_sha256':sha(root/'paper_method_v2/artifacts/proxy_inputs_v1/config.json'),
            'attempt_limit_per_record':attempts,'timeout_seconds':timeout,
            'policy':'Targeted semantic regeneration. Preserve original prompt plus recorded conservative caution; no automatic acceptance or fallback.',
            'method_note':'Quality-control retry is a documented implementation choice; GLM is an authorized replacement, not the original paper generator.'}
        save(run_dir/'policy.json',policy);save(run_dir/'review_decisions.json',review)
        save(run_dir/'requests.json',requests)
        key=None;calls=0;results=[];errors=[];halt=False
        for req in requests:
            sid=req['sample_id'];row=lookup[sid]
            folder=run_dir/'responses'/('proxy_'+hashlib.sha256(sid.encode('utf8')).hexdigest())
            canonical=folder/'accepted_schema_response.json';answer=None;raw=None;last='Attempt limit reached'
            if canonical.exists():
                raw=read(canonical);answer,_=p.parse(raw)
            elif not halt:
                for n in range(1,attempts+1):
                    prefix=folder/('attempt_%03d'%n)
                    intent=prefix.with_suffix('.request.json');response=prefix.with_suffix('.response.json')
                    if response.exists():raw=read(response)
                    elif intent.exists():continue
                    else:
                        if key is None:
                            key=getpass.getpass('GLM API Key (hidden): ').strip()
                            if not key:raise ValueError('Empty API key')
                        save(intent,{'sample_id':sid,'body_sha256':hashlib.sha256(enc(req['body'])).hexdigest(),'time':time.time()})
                        request=urllib.request.Request(cfg['base_url'].rstrip('/')+'/chat/completions',
                            data=json.dumps(req['body'],ensure_ascii=False).encode('utf8'),
                            headers={'Content-Type':'application/json','Authorization':'Bearer '+key},method='POST')
                        calls+=1
                        try:
                            with urllib.request.urlopen(request,timeout=timeout) as reply:wire=reply.read()
                            prefix.with_suffix('.body.bin').write_bytes(wire)
                            raw=json.loads(wire);save(response,raw)
                        except Exception as exc:
                            status=getattr(exc,'code',None);last=type(exc).__name__
                            save(prefix.with_suffix('.error.json'),{'exception':last,'http_status':status})
                            retry=(status==429 or 500<=status<=599) if isinstance(exc,urllib.error.HTTPError) else isinstance(exc,(TimeoutError,socket.timeout,urllib.error.URLError,ConnectionError,json.JSONDecodeError,UnicodeDecodeError))
                            print('REQUEST_FAILED',sid,n,last,status,flush=True)
                            if not retry:halt=True;break
                            time.sleep(min(60,5*n));continue
                    try:answer,_=p.parse(raw)
                    except (ValueError,KeyError,TypeError,IndexError) as exc:
                        last=str(exc);print('SCHEMA_RETRY',sid,n,last,flush=True);time.sleep(3);continue
                    save(canonical,raw);print('SAVED_SEMANTIC_RETRY',sid,flush=True);break
            if answer is None:
                errors.append({'sample_id':sid,'error':last if not halt else 'Non-retryable error; inspect attempt metadata'})
            else:
                results.append({'sample_id':sid,'question':row['question'],'original_answer':row['original_answer'],
                    'previous_proxy_answer':row['proxy_answer'],'retry_proxy_answer':answer,
                    'review_reason':next(d['reason'] for d in review['decisions'] if d['sample_id']==sid),
                    'raw_response_sha256':sha(canonical),'returned_model':raw.get('model'),
                    'semantic_decision':'PENDING','assigned_label':None})
        save(run_dir/'candidates.json',results,mutable=True)
        lines=['# Six targeted proxy retries','Compare original and retry; do not correct or grade the original.']
        for r in results:
            lines.extend(['## '+r['sample_id'],'Review reason:\n'+r['review_reason'],
                'Original:\n'+r['original_answer'],'Previous proxy:\n'+r['previous_proxy_answer'],
                'Retry proxy:\n'+r['retry_proxy_answer'],'Decision: PENDING'])
        data=('\n\n'.join(lines)+'\n').encode('utf8')
        temp=run_dir/'retry_review.md.tmp';temp.write_bytes(data);os.replace(str(temp),str(run_dir/'retry_review.md'))
        if sha(source)!=SOURCE_SHA:raise ValueError('Source changed during run')
        report={'status':'SIX_PROXY_RETRIES_READY_FOR_REVIEW' if len(results)==6 and not errors else 'PROXY_SEMANTIC_RETRY_PARTIAL',
            'reviewed_new_pairs':222,'passed_new_pairs':216,'held_new_pairs':6,
            'valid_retry_responses':len(results),'errors':errors,'new_api_calls':calls,
            'reused_unreviewed_generated_records':636,'source_records_modified':False,
            'labels_assigned':0,'training_started':False,'validation_accessed':False,'test_accessed':False,
            'candidates_sha256':sha(run_dir/'candidates.json'),'review_file':str(run_dir/'retry_review.md'),
            'ready_for_annotator_training':False}
        save(run_dir/'summary.json',report,mutable=True)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        return 0 if len(results)==6 and not errors else 2

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True)
    a.add_argument('--review',type=Path,default=Path(__file__).with_name('decisions.json'))
    a.add_argument('--dry-run',action='store_true');a.add_argument('--attempts',type=int,default=100)
    a.add_argument('--timeout',type=int,default=900);args=a.parse_args()
    if not 1<=args.attempts<=100 or not 180<=args.timeout<=1800:a.error('attempts 1..100; timeout 180..1800')
    raise SystemExit(run(args.root.resolve(),args.review.resolve(),args.dry_run,args.attempts,args.timeout))

if __name__=='__main__':main()
