"""Prepare and resume five-run training entity features; never reads evaluation data."""
import argparse
import getpass
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
COUNTS=[1468,1849,1798,2366,1722]
HASHES=['733e37b9134d13f87a7ed258fb59f0031d2160fe5e9c399328b0d0ffbc973912','5d817c744a35c03b8ee44b01658019b35fb5d9df7774bc9edc407cedcc86140f','0eab00514077d8bc7da7139e73a41c11152654f10e797724cfa813d2a8005036','cdf26723017a57081613f80f36c2d3087317290863ab20383ce664c0fed9f9ff','f1432ff06a4f8f30de5cf8c9ff938b16f29fe575302d1bd313fa129aa5b3669a']
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

def prepare(root,out):
    base=root/'paper_method_v2';fine=base/'artifacts/fine_filter_raw_beta0p2_v1';summary=read(fine/'summary.json')
    if summary['status']!='FIVE_RUN_EQ14_FINE_FILTER_COMPLETE' or summary['seeds']!=SEEDS:raise ValueError('Wrong fine-filter stage')
    datasets={}
    for seed,count,hashval in zip(SEEDS,COUNTS,HASHES):
        name='seed_%d/augmented_inputs.jsonl'%seed;p=fine/name
        if sha(p)!=hashval or summary['files_sha256'][name]!=hashval:raise ValueError('Fine-filter data changed')
        data=rows(p)
        if len(data)!=count or sum(r['kind']=='original_training' for r in data)!=1107:raise ValueError('Wrong counts')
        datasets[seed]=data
    stagepath=root/'control/scale-paper-stage1/stage1.py'
    spec=importlib.util.spec_from_file_location('entity_contract',stagepath);stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
    inputs=base/'artifacts/pe_inputs_v1';meta=read(inputs/'manifest.json');unitpath=inputs/'source_units.jsonl'
    if sha(unitpath)!=meta['outputs_sha256']['source_units.jsonl'] or sha(stagepath)!=meta['stage1_code_sha256']:raise ValueError('Pe input contract changed')
    graph=base/'artifacts/full_graph_v1';gp=read(graph/'graph.json');ep=graph/'extractions.jsonl'
    cfgpath=base/'runs/pe_glm_restored_v1/config.json';old=read(cfgpath)
    if sha(ep)!=gp['extractions_sha256'] or sha(cfgpath)!=gp['config_sha256']:raise ValueError('Original entity provenance changed')
    if old['prompt_sha256']!=digest(stage.PROMPT.encode('utf8')) or old['model']!='glm-5.3':raise ValueError('Prompt/model changed')
    units=rows(unitpath);extractions=stage.validate(units,rows(ep))
    bindings,pending,reused=bind(datasets,units,extractions)
    url=old['base_url'].rstrip('/');u=urllib.parse.urlparse(url)
    if u.scheme!='https' or not u.hostname or u.username or u.password or u.query or u.fragment:raise ValueError('Invalid API URL')
    cfg={'model':old['model'],'base_url':url,'max_tokens':16384,'stream':False,'prompt_sha256':old['prompt_sha256'],
        'stage1_sha256':sha(stagepath),'runner_sha256':sha(Path(__file__)),'fine_filter_summary_sha256':sha(fine/'summary.json'),
        'original_pe_config_sha256':sha(cfgpath),'original_extractions_sha256':sha(ep),
        'reuse_policy':'Reuse fixed original/reference features by exact unit, question and text; do not treat them as new 16384-token responses. No old synthetic response caches imported.',
        'deduplication':'New synthetic requests use only text and fixed prompt/model/settings. Equal text requests are shared across seeds; training rows and labels remain separate.',
        'execution_choice':'New requests use max_tokens=16384 to reduce truncation; differs from old Pe limit and is recorded.',
        'beta_scale_verified_from_author':False,'api_calls_in_prepare':0,'test_accessed':False,'validation_accessed':False}
    payload={'config.json':enc(cfg),'pending_units.jsonl':jl(pending),'reused_extractions.jsonl':jl(reused)}
    for seed in SEEDS:payload['seed_%d/bindings.jsonl'%seed]=jl(bindings[seed])
    for name,b in payload.items():write(out/name,b)
    report={'status':'FIVE_RUN_ENTITY_PLAN_READY','unique_synthetic_requests':len(pending),'reused_original_reference_units':len(reused),
        'training_records_by_seed':dict(zip(map(str,SEEDS),COUNTS)),'api_calls':0,'test_accessed':False,'validation_accessed':False,
        'files_sha256':{n:digest(b) for n,b in payload.items()}}
    write(out/'plan.json',enc(report));print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return stage,cfg,datasets,bindings,pending,reused

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
                    key=getpass.getpass('GLM API Key (hidden): ').strip()
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
        'semantic_accuracy_verified':False,'next_step':'Review failures if any; otherwise prepare entity token spans and five-run SAE training.'}
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

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','offline']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--attempts',type=int,default=100);p.add_argument('--timeout',type=int,default=900);a=p.parse_args()
    if not 1<=a.attempts<=100 or not 180<=a.timeout<=1800:p.error('attempts 1..100; timeout 180..1800')
    out=a.root.resolve()/'paper_method_v2/runs/train_entities_five_runs_v1';out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);raise SystemExit(execute(a.root.resolve(),out,a.command,a.attempts,a.timeout))
