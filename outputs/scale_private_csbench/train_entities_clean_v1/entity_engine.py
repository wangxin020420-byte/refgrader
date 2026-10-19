"""Prepare and resume five-run training entity features; never reads evaluation data."""
import argparse
import getpass
import os
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
import urllib.parse

SEEDS=[42,43,44,45,46]
COUNTS=[653,867,904,1018,782]

def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(p.read_bytes())
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]
def enc(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf8')
def jl(x):return ''.join(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n' for r in x).encode('utf8')
def write(p,b,mutable=False):
    if p.exists() and not mutable:
        if p.read_bytes()!=b:raise ValueError('Existing artifact differs: '+str(p))
        return
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(b);tmp.replace(p)

def bind(datasets,units,extractions):
    um={u['unit_id']:u for u in units};ex={r['unit_id']:r for r in extractions}
    refs={(u['question_id'],u['text']):u['unit_id'] for u in units if u['role']=='reference'}
    pending={};reused={};bindings={}
    for seed,data in datasets.items():
        bs=[]
        if len({r['id'] for r in data})!=len(data):raise ValueError('Duplicate training ID')
        for r in data:
            if not isinstance(r['answer'],str) or not math.isfinite(r['label']):raise ValueError('Invalid training row')
            if r['kind']=='original_training':
                uid='student:'+r['sample_id']
                if um[uid]['text']!=r['answer'] or um[uid]['question_id']!=r['question_id']:raise ValueError('Original unit mismatch')
                reused[uid]=ex[uid]
            elif r['kind']=='synthetic_candidate':
                if r['annotator_seed']!=seed:raise ValueError('Seed mismatch')
                uid='synthetic_text:'+digest(r['answer'].encode('utf8'))
                unit={'unit_id':uid,'text':r['answer']}
                if uid in pending and pending[uid]!=unit:raise ValueError('Hash collision')
                pending[uid]=unit
            else:raise ValueError('Unexpected row kind')
            rids=[]
            for text in r['references']:
                rid=refs[(r['question_id'],text)] if text.strip() else None
                rids.append(rid)
                if rid:reused[rid]=ex[rid]
            bs.append({'id':r['id'],'answer_unit_id':uid,'reference_unit_ids':rids})
        bindings[seed]=bs
    return bindings,[pending[k] for k in sorted(pending)],[reused[k] for k in sorted(reused)]

def parse(stage,unit,raw):
    choice=raw['choices'][0]
    if choice.get('finish_reason')!='stop':raise ValueError('Response not complete')
    content=choice['message']['content'].strip()
    if content.startswith('```') and content.endswith('```'):
        header,content=content.split('\n',1)
        if header.lower() not in ('```','```json'):raise ValueError('Unexpected fence')
        content=content[:-3].strip()
    obj=json.loads(content)
    if not isinstance(obj,dict) or set(obj)!={'triples'}:raise ValueError('Expected triples only')
    return stage.validate([unit],[{'unit_id':unit['unit_id'],'triples':obj['triples']}])[0]

def execute(root,out,command,attempts,timeout):
    stage,cfg,datasets,bindings,pending,reused=prepare(root,out)
    if command=='prepare':return 0
    write(out/'execution_policy.json',enc({'attempts_per_unit':attempts,'timeout_seconds':timeout,'schema_failures_per_unit':10}))
    results={r['unit_id']:r for r in reused};errors=[];calls=0;key=None;halt=False
    for i,unit in enumerate(pending,1):
        uid=unit['unit_id'];folder=out/'responses'/uid.split(':',1)[1];canonical=folder/'accepted.json'
        if not unit['text'].strip():results[uid]={'unit_id':uid,'triples':[]};continue
        body={'model':cfg['model'],'max_tokens':cfg['max_tokens'],'stream':False,'messages':[
            {'role':'system','content':stage.PROMPT},{'role':'user','content':json.dumps({'text':unit['text']},ensure_ascii=False)}]}
        bodyhash=digest(enc(body));failures=0;last='No valid response';lastfile=None
        if canonical.exists():
            saved=read(canonical)
            if saved['body_sha256']!=bodyhash:raise ValueError('Cache request mismatch')
            results[uid]=parse(stage,unit,saved['response']);continue
        for n in range(1,attempts+1):
            response=folder/('attempt_%03d.response.json'%n);intent=folder/('attempt_%03d.request.json'%n)
            if intent.exists() and read(intent)['body_sha256']!=bodyhash:raise ValueError('Attempt request mismatch')
            if response.exists():raw=read(response)
            elif intent.exists():continue
            elif command=='offline' or halt:break
            else:
                if shutil.disk_usage(root).free<1024**3:raise RuntimeError('Need at least 1 GiB free')
                if key is None:
                    key=os.environ.get('ZHIPUAI_API_KEY','').strip() or getpass.getpass('Coding API Key (hidden): ').strip()
                    if not key:raise ValueError('Empty key')
                write(intent,enc({'unit_id':uid,'body_sha256':bodyhash,'body':body,'time':time.time()}))
                req=urllib.request.Request(cfg['base_url']+'/chat/completions',data=enc(body),headers={'Content-Type':'application/json','Authorization':'Bearer '+key},method='POST')
                calls+=1
                try:
                    with urllib.request.urlopen(req,timeout=timeout) as reply:wire=reply.read()
                    write(folder/('attempt_%03d.body.bin'%n),wire)
                    raw=json.loads(wire);write(response,enc(raw))
                except Exception as exc:
                    status=getattr(exc,'code',None);last=type(exc).__name__
                    if isinstance(exc,urllib.error.HTTPError):write(folder/('attempt_%03d.http_error.bin'%n),exc.read())
                    write(folder/('attempt_%03d.error.json'%n),enc({'exception':last,'http_status':status}))
                    retry=(status==429 or 500<=status<=599) if isinstance(exc,urllib.error.HTTPError) else isinstance(exc,(TimeoutError,urllib.error.URLError,ConnectionError,json.JSONDecodeError,UnicodeDecodeError))
                    print('TRANSPORT_RETRY',i,n,last,status,flush=True)
                    if not retry:halt=True;break
                    time.sleep(min(60,5*n));continue
            lastfile=str(response)
            try:valid=parse(stage,unit,raw)
            except (ValueError,KeyError,TypeError,IndexError,AttributeError) as exc:
                last=str(exc);failures+=1
                if failures>=10:break
                continue
            results[uid]=valid;write(canonical,enc({'body_sha256':bodyhash,'response':raw}));break
        if uid not in results:errors.append({'unit_id':uid,'source_text':unit['text'],'error':last,'schema_failures':failures,'last_response_file':lastfile})
        print('ENTITY',i,'/',len(pending),'READY',sum(u['unit_id'] in results for u in pending),'API_CALLS',calls,flush=True)
    report={'status':'FIVE_RUN_TRAIN_ENTITIES_COMPLETE' if not errors else 'FIVE_RUN_TRAIN_ENTITIES_PARTIAL',
        'unique_synthetic_requests':len(pending),'valid_synthetic_units':sum(u['unit_id'] in results for u in pending),
        'remaining':len(errors),'new_api_calls':calls,'errors':errors,'labels_modified':False,'training_started':False,
        'test_accessed':False,'validation_accessed':False,'ready_for_sae_training':False,
        'experiment_id':'csbench_scale_online_clean_v2','automatic_semantic_errors_retained':True,'human_review_performed':False,
        'semantic_accuracy_verified':False,'next_step':'Resolve structural failures if any; otherwise prepare entity token spans and five-run SAE training. No manual semantic review.'}
    files={}
    if not errors:
        files['extractions.jsonl']=jl([results[k] for k in sorted(results)])
        for seed in SEEDS:
            export=[]
            for r,b in zip(datasets[seed],bindings[seed]):
                if r['id']!=b['id']:raise ValueError('Binding order mismatch')
                export.append(dict(r,answer_triples=results[b['answer_unit_id']]['triples'],
                    reference_triples=[results[rid]['triples'] if rid else [] for rid in b['reference_unit_ids']]))
            files['seed_%d/training_with_entities.jsonl'%seed]=jl(export)
        for name,data in files.items():write(out/name,data)
        report['files_sha256']={n:digest(b) for n,b in files.items()};report['training_records_by_seed']=dict(zip(map(str,SEEDS),COUNTS))
    write(out/'summary.json',enc(report),mutable=True);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return 0 if not errors else 2

