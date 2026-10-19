"""Fixed 50-row validation entities and SAE features, no test access."""
import argparse
import getpass
import importlib.util
import os
from pathlib import Path
import shutil
import time
import urllib.request
import urllib.error
import urllib.parse
import json
import math
import entity_io as e
import token_core as c

INPUT_SHA='608ac529a83c55b7db714e3c8f35544cf53cc7fa72b2836e13e9ab39b36fa71c'
LABEL_SHA='5dc9394d1fb8ac232cf904872af9639782ee83073580d0e19808b265e5fcb9c0'

def require(ok,msg):
    if not ok: raise ValueError(msg)

def load_inputs(root):
    split=root/'data/prepared/ads_paper_split_v1_seed42'
    ip,lp=split/'validation.inputs.jsonl',split/'validation.labels.jsonl'
    require(e.sha(ip)==INPUT_SHA and e.sha(lp)==LABEL_SHA,'Fixed validation input changed')
    rows=e.rows(ip);labels=e.rows(lp)
    gold={x['sample_id']:x for x in labels}
    require(len(rows)==len(labels)==len(gold)==50 and len({x['sample_id'] for x in rows})==50,'Validation count/ID mismatch')
    require(set(gold)=={x['sample_id'] for x in rows},'Gold ID mismatch')
    train=root/'paper_method_v2/artifacts/sae_inputs_five_runs_v1'
    report=e.read(train/'summary.json')
    require(report['status']=='FIVE_RUN_SAE_TRAIN_INPUTS_READY','Training features incomplete')
    configs=[]
    for seed in [42,43,44,45,46]:
        folder=train/('seed_%d'%seed);cfg=e.read(folder/'config.json')
        sm=e.read(folder/'summary.json')
        require(sm==report['runs'][str(seed)] and e.sha(folder/'config.json')==sm['config_sha256'],'Training summary/config mismatch')
        require(e.sha(folder/'train_features.jsonl')==sm['files_sha256']['train_features.jsonl'],'Training feature hash mismatch')
        require(not set(gold)&{x['sample_id'] for x in e.rows(folder/'train_features.jsonl')},'Validation/training ID overlap')
        configs.append({k:v for k,v in cfg.items() if k not in ('seed','entity_input_sha256')})
    require(all(x==configs[0] for x in configs),'Five training encoders differ')
    require(e.sha(Path(c.__file__))==configs[0]['token_core_sha256'],'Token encoder changed')
    result=[]
    for row in rows:
        g=gold[row['sample_id']]
        require(g.get('question_id',row['question_id'])==row['question_id'],'Gold question mismatch')
        label=float(g['gold_raw']);require(math.isfinite(label) and 0<=label<=10,'Invalid gold')
        result.append(dict(row,id=row['sample_id'],kind='validation',label=label))
    return result,configs[0],e.sha(train/'summary.json')

def make_units(rows):
    units={};bindings=[]
    def unit(text):
        uid='validation_text:'+e.digest(text.encode('utf8'))
        u={'unit_id':uid,'text':text}
        require(uid not in units or units[uid]==u,'Hash collision')
        units[uid]=u
        return uid
    for row in rows:
        bindings.append({'id':row['id'],'answer':unit(row['answer']),
                         'references':[unit(t) for t in row['references']]})
    return [units[k] for k in sorted(units)],bindings

def fetch(unit,stage,cfg,out,command,keybox):
    if not unit['text'].strip():return {'unit_id':unit['unit_id'],'triples':[]},0,None
    folder=out/'responses'/unit['unit_id'].split(':',1)[1]
    body={'model':cfg['model'],'max_tokens':16384,'stream':False,'messages':[
        {'role':'system','content':stage.PROMPT},
        {'role':'user','content':json.dumps({'text':unit['text']},ensure_ascii=False)}]}
    bh=e.digest(e.enc(body));accepted=folder/'accepted.json';calls=0;last='No accepted response'
    if accepted.exists():
        raw=e.read(accepted);require(raw['body_sha256']==bh,'Cached request differs')
        return e.parse(stage,unit,raw['response']),0,None
    failures=0
    for n in range(1,101):
        rp=folder/('attempt_%03d.response.json'%n);qp=folder/('attempt_%03d.request.json'%n)
        if qp.exists():require(e.read(qp)['body_sha256']==bh,'Attempt request differs')
        if rp.exists():raw=e.read(rp)
        elif qp.exists():continue
        elif command=='offline':break
        else:
            require(shutil.disk_usage(out).free>=1024**3,'Need 1 GiB free')
            if not keybox:
                key=getpass.getpass('GLM API Key (hidden): ').strip()
                require(bool(key),'Empty key');keybox.append(key)
            e.write(qp,e.enc({'unit_id':unit['unit_id'],'body_sha256':bh,'body':body,'time':time.time()}))
            req=urllib.request.Request(cfg['base_url']+'/chat/completions',data=e.enc(body),
                headers={'Content-Type':'application/json','Authorization':'Bearer '+keybox[0]},method='POST')
            calls+=1
            try:
                with urllib.request.urlopen(req,timeout=900) as r:wire=r.read()
                e.write(folder/('attempt_%03d.body.bin'%n),wire)
                raw=json.loads(wire);e.write(rp,e.enc(raw))
            except Exception as exc:
                status=getattr(exc,'code',None);last=type(exc).__name__
                e.write(folder/('attempt_%03d.error.json'%n),e.enc({'exception':last,'http_status':status}))
                retry=(status==429 or 500<=status<=599) if isinstance(exc,urllib.error.HTTPError) else isinstance(exc,(TimeoutError,urllib.error.URLError,ConnectionError,json.JSONDecodeError,UnicodeDecodeError))
                print('REQUEST_ERROR',unit['unit_id'],n,last,status,flush=True)
                if not retry:raise RuntimeError('Non-retryable API error: '+last+' '+str(status))
                time.sleep(min(60,5*n));continue
        try:result=e.parse(stage,unit,raw)
        except (ValueError,KeyError,TypeError,IndexError,AttributeError) as exc:
            last=str(exc);failures+=1
            if failures>=20:break
            continue
        e.write(accepted,e.enc({'body_sha256':bh,'response':raw}))
        return result,calls,None
    return None,calls,{'unit_id':unit['unit_id'],'source_text':unit['text'],'error':last,'schema_failures':failures}

def execute(root,out,command):
    rows,encoding,trainsha=load_inputs(root)
    stagepath=root/'control/scale-paper-stage1/stage1.py'
    spec=importlib.util.spec_from_file_location('stage',stagepath);stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
    old=e.read(root/'paper_method_v2/runs/train_entities_five_runs_v1/config.json')
    require(e.sha(stagepath)==old['stage1_sha256'] and e.digest(stage.PROMPT.encode('utf8'))==old['prompt_sha256'],'Extraction contract changed')
    url=old['base_url'].rstrip('/');parsed=urllib.parse.urlparse(url)
    require(parsed.scheme=='https' and parsed.hostname and not any([parsed.username,parsed.password,parsed.query,parsed.fragment]),'Invalid endpoint')
    require(old['model']=='glm-5.3' and old['max_tokens']==16384,'Extraction settings differ')
    cfg=dict(model=old['model'],base_url=url,max_tokens=16384,timeout=900,attempts=100,
        schema_failure_limit=20,prompt_sha256=old['prompt_sha256'],stage_sha256=e.sha(stagepath),
        inputs_sha256=INPUT_SHA,labels_sha256=LABEL_SHA,training_summary_sha256=trainsha,
        encoding=encoding,builder_sha256=e.sha(Path(__file__)),entity_io_sha256=e.sha(Path(e.__file__)),
        request_scope='Text only, no gold score, no question; identical validation text shared',
        reuse_policy='Only accepted exact requests within this run; no old validation cache imported',
        test_accessed=False)
    e.write(out/'config.json',e.enc(cfg))
    units,bindings=make_units(rows)
    e.write(out/'units.jsonl',e.jl(units));e.write(out/'bindings.jsonl',e.jl(bindings))
    print('VALIDATION_PLAN',len(rows),'rows',len(units),'unique text units',flush=True)
    if command=='prepare':return 0
    complete=out/'complete.json'
    if complete.exists():
        done=e.read(complete);require(done['config_sha256']==e.sha(out/'config.json'),'Complete config differs')
        for name,digest in done['files_sha256'].items():require(e.sha(out/name)==digest,'Completed output changed: '+name)
        print(json.dumps(done,ensure_ascii=False,indent=2));print('VALIDATION_FEATURES_VERIFIED');return 0
    results={};errors=[];calls=0;keybox=[]
    for i,u in enumerate(units,1):
        result,n,error=fetch(u,stage,cfg,out,command,keybox);calls+=n
        if error:errors.append(error)
        else:results[u['unit_id']]=result
        print('VALIDATION_ENTITY',i,'/',len(units),'valid',len(results),'new_api_calls',calls,flush=True)
    report=dict(status='VALIDATION_ENTITIES_PARTIAL' if errors else 'VALIDATION_ENTITIES_COMPLETE',
        remaining=len(errors),errors=errors,records=50,new_api_calls=calls,validation_accessed=True,
        test_accessed=False,training_started=False,semantic_accuracy_verified=False)
    e.write(out/'entity_summary.json',e.enc(report),mutable=True)
    if errors:print(json.dumps(report,ensure_ascii=False,indent=2));return 2
    for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[name]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import transformers
    model=root/'models/bert-base-chinese'/c.REV
    for name,digest in encoding['tokenizer_files_sha256'].items():require(e.sha(model/name)==digest,'Tokenizer file changed')
    require(transformers.__version__==encoding['transformers_version'],'Transformers version differs')
    tok=transformers.AutoTokenizer.from_pretrained(str(model),local_files_only=True,use_fast=True)
    tok.truncation_side='right';tok.padding_side='right'
    require(tok.is_fast and e.digest(tok.backend_tokenizer.to_str().encode('utf8'))==encoding['tokenizer_backend_sha256'],'Tokenizer backend differs')
    features=[];audits=[];export=[]
    for row,b in zip(rows,bindings):
        require(row['id']==b['id'],'Binding differs')
        r=dict(row,answer_triples=results[b['answer']]['triples'],reference_triples=[results[x]['triples'] for x in b['references']])
        f,a=c.encode_row(r,tok);export.append(r);features.append(f);audits.append(a)
    files={'validation_with_entities.jsonl':e.jl(export),'validation_features.jsonl':e.jl(features),
           'tokenization_audit.jsonl':e.jl(audits),'extractions.jsonl':e.jl([results[k] for k in sorted(results)])}
    for name,data in files.items():e.write(out/name,data)
    done=dict(status='SAE_FIXED_VALIDATION_INPUTS_READY',records=50,config_sha256=e.sha(out/'config.json'),
        files_sha256={n:e.digest(b) for n,b in files.items()},validation_accessed=True,test_accessed=False,
        training_started=False,labels_modified=False,semantic_accuracy_verified=False,
        text_truncated_records=sum(a['text_full_tokens']>512 for a in audits),
        knowledge_truncated_records=sum(a['knowledge_full_tokens']>512 for a in audits),
        records_without_retained_entities=sum(not f['has_entities'] for f in features),
        next_step='SAE smoke test and five seeded training runs; select each checkpoint by highest validation QWK')
    e.write(complete,e.enc(done));print(json.dumps(done,ensure_ascii=False,indent=2));return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','offline']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();out=root/'paper_method_v2/runs/sae_validation_inputs_v1';out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raise SystemExit(execute(root,out,a.command))
