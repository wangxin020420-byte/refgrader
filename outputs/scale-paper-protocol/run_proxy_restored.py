"""Reuse frozen proxy alignment and generate only missing training proxies."""
import argparse
import copy
import getpass
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

OLD_SHA='c84bd04ca7a9755c448ccb2f4e18c8d72e860d4f7f76322e4c639ad7b69d8b33'


def load(path):
    spec=importlib.util.spec_from_file_location('proxy_contract',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def checked_reuse(old,sources):
    lookup={s['sample_id']:s for s in sources};result={}
    for r in old:
        sid=r['sample_id']
        if sid not in lookup or sid in result:raise ValueError('Old proxy identity mismatch')
        s=lookup[sid]
        for key,field in [('original_answer','a'),('question','q'),('question_id','question_id'),('references','r'),('source_score','l')]:
            if r[key]!=s[field]:raise ValueError('Reused source differs: '+sid+' '+key)
        if r.get('assigned_label') is not None:raise ValueError('Aligned proxy unexpectedly labeled')
        row=copy.deepcopy(r)
        row['v2_origin']='frozen_v1_alignment'
        result[sid]=row
    return result


def execute(root,attempts,timeout):
    p=load(root/'control/scale-paper-protocol/proxy_restored.py')
    plan=root/'paper_method_v2/artifacts/proxy_inputs_v1';meta=p.read(plan/'summary.json')
    for name,h in meta['files_sha256'].items():
        if p.sha(plan/name)!=h:raise ValueError('Plan changed: '+name)
    cfg=p.read(plan/'config.json')
    if cfg['script_sha256']!=p.sha(Path(p.__file__)):raise ValueError('Plan builder changed')
    sources=p.read(plan/'training_sources.json');requests=p.rows(plan/'requests.jsonl')
    if len(sources)!=1107 or len({s['sample_id'] for s in sources})!=1107:raise ValueError('Expected 1107 sources')
    lookup={s['sample_id']:s for s in sources}
    oldroot=root/'paper_method_v1/artifacts';oldpath=oldroot/'proxy_aligned_v1/proxy_records.jsonl'
    oldmeta=p.read(oldpath.parent/'manifest.json')
    if p.sha(oldpath)!=OLD_SHA or oldmeta['aligned_records_sha256']!=OLD_SHA:raise ValueError('Frozen alignment changed')
    oldplan=oldroot/'proxy_inputs_v1';oldplanmeta=p.read(oldplan/'summary.json')
    for name in ('config.json','requests.jsonl'):
        if p.sha(oldplan/name)!=oldplanmeta['files_sha256'][name]:raise ValueError('Old proxy configuration changed')
    oldcfg=p.read(oldplan/'config.json')
    for key in ('model','base_url','max_tokens','stream'):
        if cfg[key]!=oldcfg[key]:raise ValueError('Proxy generation configuration differs: '+key)
    oldreq={r['sample_id']:r for r in p.rows(oldplan/'requests.jsonl')}
    current={r['sample_id']:r for r in requests}
    for sid,r in oldreq.items():
        if sid not in current or r['messages']!=current[sid]['messages']:raise ValueError('Proxy prompt or input changed: '+sid)
    reused=checked_reuse(p.rows(oldpath),sources)
    if len(reused)!=885:raise ValueError('Expected 885 frozen aligned records')
    run=root/'paper_method_v2/runs/proxy_glm_v1';run.mkdir(parents=True,exist_ok=True)
    policy={'old_alignment_sha256':OLD_SHA,'old_manifest_sha256':p.sha(oldpath.parent/'manifest.json'),
       'plan_sha256':p.sha(plan/'summary.json'),'runner_sha256':p.sha(Path(__file__)),
       'attempts_per_missing_request':attempts,'timeout_seconds':timeout,
       'reuse_policy':'Preserve all prior review states and original-answer fallbacks; no blanket semantic approval',
       'uncertain_request_policy':'Retain request intent and advance to next attempt on resume'}
    p.save(run/'config.json',p.enc(cfg));p.save(run/'reuse_policy.json',p.enc(policy))
    records=dict(reused);calls=0;errors=[];key=None;halt=False
    print('REUSED_ALIGNED',len(reused),'MISSING_NONBLANK',sum(r['sample_id'] not in reused for r in requests),flush=True)
    for req in requests:
        sid=req['sample_id'];rid=req['request_id']
        if sid in reused:continue
        if rid!='proxy_'+hashlib.sha256(sid.encode()).hexdigest():raise ValueError('Request ID mismatch')
        source=lookup[sid];canonical=run/'responses'/(rid+'.json');answer=None;last='Attempts exhausted';fence=False
        if canonical.exists():
            raw=p.read(canonical);answer,fence=p.parse(raw)
        elif not halt:
            history=run/'attempt_history'/rid
            for number in range(1,attempts+1):
                response=history/('attempt_%03d.response.json'%number);intent=history/('attempt_%03d.request.json'%number)
                if not response.exists():
                    if intent.exists():last='Uncertain earlier request; advanced to next attempt';continue
                    if key is None:
                        key=getpass.getpass('GLM API Key (hidden): ').strip()
                        if not key:raise ValueError('Empty API key')
                    body={'model':cfg['model'],'messages':req['messages'],'max_tokens':cfg['max_tokens'],'stream':False}
                    p.save(intent,p.enc({'sample_id':sid,'time':time.time(),'body_sha256':hashlib.sha256(p.enc(body)).hexdigest()}))
                    request=urllib.request.Request(cfg['base_url'].rstrip('/')+'/chat/completions',data=json.dumps(body).encode('utf8'),
                        headers={'Content-Type':'application/json','Authorization':'Bearer '+key},method='POST')
                    calls+=1
                    try:
                        with urllib.request.urlopen(request,timeout=timeout) as reply:wire=reply.read()
                        p.save(history/('attempt_%03d.body.bin'%number),wire)
                        raw=json.loads(wire);p.save(response,p.enc(raw))
                    except Exception as exc:
                        last=type(exc).__name__;status=getattr(exc,'code',None)
                        p.save(history/('attempt_%03d.error.json'%number),p.enc({'exception':last,'http_status':status}))
                        print('REQUEST_RETRY',sid,number,last,status,flush=True)
                        retry=(status==429 or (status is not None and 500<=status<=599)) if isinstance(exc,urllib.error.HTTPError) else isinstance(exc,(TimeoutError,urllib.error.URLError,ConnectionError,json.JSONDecodeError,UnicodeDecodeError))
                        if not retry:halt=True;break
                        time.sleep(15);continue
                raw=p.read(response)
                try:answer,fence=p.parse(raw)
                except (ValueError,KeyError,TypeError,IndexError) as exc:
                    last=str(exc);print('CONTENT_RETRY',sid,number,last,flush=True);time.sleep(3);continue
                p.save(canonical,p.enc(raw));print('SAVED_PROXY',sid,flush=True);time.sleep(3);break
        if answer is None:
            errors.append({'sample_id':sid,'request_id':rid,'error':last if not halt else 'Network halted; inspect attempt history'});continue
        records[sid]={'sample_id':sid,'question_id':source['question_id'],'question':source['q'],'references':source['r'],
            'original_answer':source['a'],'proxy_answer':answer,'source_score':source['l'],'assigned_label':None,
            'semantic_equivalence_verified':False,'alignment_review_status':'PENDING_REVIEW',
            'kind':'generated_proxy','v2_origin':'new_generation','request_id':rid,
            'raw_response_sha256':p.sha(canonical),'outer_fence_removed':fence,
            'returned_model':raw.get('model'),'response_id':raw.get('id')}
        p.save(run/'progress.json',p.enc({'records_ready':len(records),'new_api_calls':calls,'errors':len(errors)}),mutable=True)
    for source in sources:
        sid=source['sample_id']
        if sid in records or source['a'].strip():continue
        records[sid]={'sample_id':sid,'question_id':source['question_id'],'question':source['q'],'references':source['r'],
            'original_answer':source['a'],'proxy_answer':source['a'],'source_score':source['l'],'assigned_label':None,
            'semantic_equivalence_verified':True,'kind':'blank_passthrough','v2_origin':'new_blank_passthrough',
            'alignment_review_status':'EXACT_ORIGINAL_PASSTHROUGH'}
    ordered=[records[s['sample_id']] for s in sources if s['sample_id'] in records]
    new=[r for r in ordered if r['v2_origin']=='new_generation']
    complete=len(ordered)==1107 and not errors
    review=run/'new_proxy_review';review.mkdir(parents=True,exist_ok=True)
    p.save(review/'pairs.json',p.enc(new),mutable=True)
    files=[]
    for offset in range(0,len(new),40):
        lines=['# New proxy semantic review','Preserve original errors, omissions, numbers, formulas and code. No scores assigned.']
        for r in new[offset:offset+40]:
            lines.extend(['\n## '+r['sample_id'],'Question:\n'+r['question'],'Original:\n'+r['original_answer'],
                          'Proxy:\n'+r['proxy_answer'],'Decision: PENDING'])
        path=review/('review_%02d.md'%(offset//40+1));p.save(path,('\n\n'.join(lines)+'\n').encode('utf8'),mutable=True);files.append(str(path))
    report={'status':'PROXY_RESTORED_COMPLETE_REVIEW_REQUIRED' if complete else 'PROXY_RESTORED_PARTIAL',
       'training_records':1107,'proxy_records':len(ordered),'reused_aligned_records':885,
       'reused_unreviewed_generated_records':sum(r.get('alignment_review_status')=='NOT_INDIVIDUALLY_REVIEWED' for r in reused.values()),
       'preserved_original_fallback_records':sum(r.get('alignment_action')=='original_answer_fallback' for r in reused.values()),
       'new_generated_records':len(new),'blank_passthrough_records':sum(r['kind']=='blank_passthrough' for r in ordered),
       'new_api_calls_this_invocation':calls,'errors':errors,'assigned_labels':0,'synthetic_labels_assigned':0,
       'semantic_review_required':True,'ready_for_annotator_training':False,'training_started':False,'test_accessed':False,
       'validation_accessed':False,'review_files':files,'records_sha256':hashlib.sha256(p.jl(ordered)).hexdigest(),
       'reuse_policy_sha256':p.sha(run/'reuse_policy.json')}
    p.save(run/'proxy_records.jsonl',p.jl(ordered),mutable=True);p.save(run/'summary.json',p.enc(report),mutable=True)
    if complete:
        out=root/'paper_method_v2/artifacts/proxy_review_inputs_v1'
        p.save(out/'proxy_records.jsonl',p.jl(ordered));p.save(out/'manifest.json',p.enc({
            'records_sha256':report['records_sha256'],'plan_sha256':p.sha(plan/'summary.json'),
            'reuse_policy_sha256':report['reuse_policy_sha256'],'ready_for_annotator_training':False,'semantic_review_required':True}))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('PROXY_RESTORED_BATCH_DONE' if complete else 'PROXY_RESTORED_PARTIAL',flush=True)
    return 0 if complete else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--attempts',type=int,default=20);p.add_argument('--timeout',type=int,default=900)
    a=p.parse_args()
    if not 1<=a.attempts<=100 or not 180<=a.timeout<=1800:p.error('attempts 1..100; timeout 180..1800')
    root=a.root.resolve();run=root/'paper_method_v2/runs/proxy_glm_v1';run.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (run/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);raise SystemExit(execute(root,a.attempts,a.timeout))
