"""Frozen five-model test evaluation. Gold opened only after all predictions exist."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.parse
import prepare_validation as v
import entity_io as e
import token_core as t
import sae_core as c
import run_five as train

INPUT_SHA='3fdcb8e1f00851a628ec0b3ab40e772e57933868136d10284a3d9acf00400716'
GOLD_SHA='f3c2323f088f26791078b99ac3b5b5af24713b8f4768decc6940ea9461f8e2cd'
PROTOCOL='a5ea38b92abc1fc90ca4c4e4ea0e308efbee70fdbf7ad320aae6edcfb8ea4d83'
WEIGHTS=['037ac724823cd056d9fae7d4e4bd45e42a1651fd8cedb99a20217cca86ef867d','9e4682243592d4881deeae803ed587e9faab31ca3a9f0f59578b05b8a078e61f','6ba4260bf98b90e2dec75680bfd6250655de5f096140a3a49dc6fb1acdf08962','4b8e1824dcfd3d8e1ac50ccae028290980aec0c0a4c87243f25d67d6c79be436','1da9c153d12ddb9bce703bb40773e667c06c99ce703907d5369625710eb61bd7']
require=v.require

def freeze(root,out):
    runs=root/'paper_method_v2/runs/sae_five_runs_valqwk_v1'
    require(c.sha(runs/'protocol.json')==PROTOCOL,'Training protocol changed')
    protocol=c.read(runs/'protocol.json');summary=c.read(runs/'summary.json')
    require(summary['status']=='FIVE_SAE_RUNS_TRAINED_VALIDATION_SELECTED' and summary['completed_runs']==5,'Five models incomplete')
    for seed,digest in zip(train.SEEDS,WEIGHTS):
        folder=runs/('seed_%d'%seed);done=train.completed(folder,PROTOCOL)
        require(done==summary['runs'][str(seed)] and done['weights_reload_verified'],'Model completion mismatch')
        require(c.sha(folder/'model_best.pt')==digest,'Selected model changed')
    require(c.sha(Path(c.__file__))==protocol['model_code_sha256'],'Model implementation changed')
    require(c.sha(Path(t.__file__))==protocol['token_code_sha256'],'Token implementation changed')
    require(c.sha(Path(train.__file__))==protocol['builder_sha256'],'Training helper changed')
    policy=dict(status='TEST_PROTOCOL_FROZEN',training_protocol_sha256=PROTOCOL,seeds=train.SEEDS,
        weights_sha256=dict(zip(map(str,train.SEEDS),WEIGHTS)),test_inputs_sha256=INPUT_SHA,
        test_labels_expected_sha256=GOLD_SHA,test_records=425,
        metrics=['pearson_raw','qwk_clipped_half_up','rmse_raw'],
        qwk_policy=protocol['qwk'],aggregation='Arithmetic mean of five separate metrics; sample SD ddof=1; no prediction ensemble',
        model_selection='Already selected on validation only; no test-dependent choices',
        code_sha256={p.name:c.sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        reproduction_status='METHOD_RECONSTRUCTION_WITH_DOCUMENTED_DEVIATIONS',
        unresolved=protocol['unresolved'],implementation_choices=protocol['implementation_choices'])
    e.write(out/'policy.json',e.enc(policy))
    return protocol

def inputs(root):
    split=root/'data/prepared/ads_paper_split_v1_seed42';path=split/'test.inputs.jsonl'
    require(c.sha(path)==INPUT_SHA,'Test inputs changed')
    rows=e.rows(path);require(len(rows)==425 and len({r['sample_id'] for r in rows})==425,'Test identities/count')
    used=set()
    for name in ('train.inputs.jsonl','validation.inputs.jsonl'):
        used.update(r['sample_id'] for r in e.rows(split/name))
    require(not used&{r['sample_id'] for r in rows},'Cross-split ID overlap')
    return rows

def features(root,out,protocol):
    folder=out/'inputs';folder.mkdir(parents=True,exist_ok=True)
    rows=inputs(root)
    # Placeholders satisfy the token-core schema; these are never test gold labels.
    rows=[dict(r,id=r['sample_id'],kind='test',label=0.0) for r in rows]
    units,bindings=v.make_units(rows)
    for u in units:u['unit_id']=u['unit_id'].replace('validation_text:','test_text:',1)
    for b in bindings:
        b['answer']=b['answer'].replace('validation_text:','test_text:',1)
        b['references']=[x.replace('validation_text:','test_text:',1) for x in b['references']]
    stagepath=root/'control/scale-paper-stage1/stage1.py'
    spec=importlib.util.spec_from_file_location('stage',stagepath);stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
    cfg=e.read(root/'paper_method_v2/runs/train_entities_five_runs_v1/config.json')
    require(c.sha(stagepath)==cfg['stage1_sha256'] and e.digest(stage.PROMPT.encode('utf8'))==cfg['prompt_sha256'],'Extraction contract changed')
    require(cfg['model']=='glm-5.3' and cfg['max_tokens']==16384,'Extraction configuration changed')
    url=urllib.parse.urlparse(cfg['base_url'])
    require(url.scheme=='https' and url.hostname and not any([url.username,url.password,url.query,url.fragment]),'Invalid endpoint')
    fc=dict(policy_sha256=c.sha(out/'policy.json'),encoding=protocol['encoding'],model=cfg['model'],base_url=cfg['base_url'],
        max_tokens=16384,timeout_seconds=900,attempts=100,schema_failure_limit=20,prompt_sha256=cfg['prompt_sha256'],
        test_labels_read=False,label_placeholder='0.0 only for tensor schema; never used for evaluation',
        reuse='Within-run accepted exact requests only',semantic_accuracy_verified=False)
    e.write(folder/'config.json',e.enc(fc));e.write(folder/'units.jsonl',e.jl(units));e.write(folder/'bindings.jsonl',e.jl(bindings))
    if (folder/'complete.json').exists():
        done=e.read(folder/'complete.json');require(done['config_sha256']==c.sha(folder/'config.json'),'Feature config changed')
        for n,h in done['files_sha256'].items():require(c.sha(folder/n)==h,'Feature file changed')
        return
    result={};errors=[];calls=0;key=[]
    for i,u in enumerate(units,1):
        r,n,err=v.fetch(u,stage,cfg,folder,'run',key);calls+=n
        if err:errors.append(err)
        else:result[u['unit_id']]=r
        print('TEST_ENTITY',i,'/',len(units),'VALID',len(result),'NEW_CALLS',calls,flush=True)
    e.write(folder/'entity_summary.json',e.enc(dict(remaining=len(errors),errors=errors,new_api_calls=calls,test_inputs_accessed=True,test_labels_read=False)),mutable=True)
    if errors:raise RuntimeError('Test entity extraction incomplete; inspect inputs/entity_summary.json. Saved responses retained.')
    import transformers
    model=root/'models/bert-base-chinese'/c.REV;encoding=protocol['encoding']
    require(transformers.__version__==encoding['transformers_version'],'Tokenizer library changed')
    for name,h in encoding['tokenizer_files_sha256'].items():require(c.sha(model/name)==h,'Tokenizer changed')
    tok=transformers.AutoTokenizer.from_pretrained(str(model),use_fast=True,local_files_only=True)
    tok.truncation_side='right';tok.padding_side='right'
    require(tok.is_fast and e.digest(tok.backend_tokenizer.to_str().encode())==encoding['tokenizer_backend_sha256'],'Tokenizer backend changed')
    encoded=[];audits=[]
    for r,b in zip(rows,bindings):
        require(r['id']==b['id'],'Binding mismatch')
        item=dict(r,answer_triples=result[b['answer']]['triples'],reference_triples=[result[x]['triples'] for x in b['references']])
        f,a=t.encode_row(item,tok);f['question_id']=r['question_id'];encoded.append(f);audits.append(a)
    files={'test_features.jsonl':e.jl(encoded),'tokenization_audit.jsonl':e.jl(audits),
        'extractions.jsonl':e.jl([result[k] for k in sorted(result)])}
    for name,data in files.items():e.write(folder/name,data)
    e.write(folder/'complete.json',e.enc(dict(status='TEST_FEATURES_READY',records=425,config_sha256=c.sha(folder/'config.json'),
        files_sha256={n:e.digest(b) for n,b in files.items()},test_labels_read=False,
        text_truncated_records=sum(a['text_full_tokens']>512 for a in audits),knowledge_truncated_records=sum(a['knowledge_full_tokens']>512 for a in audits))))

def load_features(out):
    f=out/'inputs';done=e.read(f/'complete.json')
    require(done['status']=='TEST_FEATURES_READY','Features incomplete')
    require(c.sha(f/'config.json')==done['config_sha256'],'Feature config mismatch')
    require(e.read(f/'config.json')['policy_sha256']==c.sha(out/'policy.json'),'Feature policy mismatch')
    for name,h in done['files_sha256'].items():require(c.sha(f/name)==h,'Feature hash mismatch')
    rows=e.rows(f/'test_features.jsonl');require(len(rows)==425 and len({r['id'] for r in rows})==425,'Feature IDs')
    for r in rows:t.validate_feature(r);require(r['label']==0.0,'Unexpected feature label')
    return rows

def pred_complete(out,seed):
    folder=out/('seed_%d'%seed);path=folder/'complete.json'
    if not path.exists():return False
    done=e.read(path)
    require(done['policy_sha256']==c.sha(out/'policy.json'),'Prediction policy mismatch')
    require(done['features_sha256']==c.sha(out/'inputs/test_features.jsonl'),'Prediction feature mismatch')
    require(c.sha(folder/'predictions.jsonl')==done['predictions_sha256'],'Prediction file changed')
    return True

def worker(args,root,out,protocol):
    rows=load_features(out);seed=args.seed;folder=out/('seed_%d'%seed);folder.mkdir(parents=True,exist_ok=True)
    if pred_complete(out,seed):print('PREDICTIONS_ALREADY_COMPLETE',seed);return
    deadline=time.monotonic()+args.wait_hours*3600
    while True:
        try:c.select_gpu(args.physical_gpu);break
        except RuntimeError:
            if time.monotonic()>=deadline:raise
            print('WAIT_GPU',flush=True);time.sleep(60)
    import torch
    import transformers
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    trained=root/'paper_method_v2/runs/sae_five_runs_valqwk_v1'/('seed_%d'%seed)
    require(c.read(trained/'runtime.json')=={'torch':torch.__version__,'transformers':transformers.__version__},'Inference runtime differs from training')
    device=torch.device('cuda:0');modelpath=root/'models/bert-base-chinese'/c.REV
    for name,h in protocol['model_files_sha256'].items():require(c.sha(modelpath/name)==h,'Local BERT changed')
    model,optimizer=c.create_model(torch,transformers,modelpath,seed,device);del optimizer
    ck=torch.load(trained/'model_best.pt',map_location='cpu',weights_only=False)
    require(ck['protocol_sha256']==PROTOCOL and ck['seed']==seed,'Checkpoint identity mismatch')
    model.load_state_dict(ck['model'],strict=True);del ck;model.eval()
    cfg=dict(policy_sha256=c.sha(out/'policy.json'),features_sha256=c.sha(out/'inputs/test_features.jsonl'),seed=seed)
    e.write(folder/'config.json',e.enc(cfg))
    cache=folder/'partial.json';pred=[]
    if cache.exists():
        saved=e.read(cache);require(saved['config']==cfg,'Partial prediction config mismatch');pred=saved['predictions']
        require(len(pred)<=len(rows),'Partial predictions exceed count')
        for a,b in zip(pred,rows):require(a['sample_id']==b['sample_id'] and a['question_id']==b['question_id'] and math.isfinite(a['prediction_raw']),'Partial prediction mismatch')
    pad=c.read(modelpath/'config.json').get('pad_token_id',0)
    with torch.inference_mode():
        for i in range(len(pred),len(rows)):
            batch=c.collate(torch,rows,[i],device,pad);score=float(model(*batch[:3]).item())
            require(math.isfinite(score),'Nonfinite test prediction')
            pred.append(dict(sample_id=rows[i]['sample_id'],question_id=rows[i]['question_id'],prediction_raw=score))
            if len(pred)%25==0 or len(pred)==len(rows):
                e.write(cache,e.enc({'config':cfg,'predictions':pred}),mutable=True);print('TEST_PREDICT',seed,len(pred),'/425',flush=True)
    e.write(folder/'predictions.jsonl',e.jl(pred))
    e.write(folder/'complete.json',e.enc(dict(cfg,status='TEST_PREDICTIONS_COMPLETE',records=425,test_labels_read=False,predictions_sha256=c.sha(folder/'predictions.jsonl'))))

def metrics(gold,pred):
    require(len(gold)==len(pred)>0 and all(math.isfinite(x) for x in gold+pred),'Invalid metric inputs')
    gm=statistics.mean(gold);pm=statistics.mean(pred)
    denom=math.sqrt(sum((x-gm)**2 for x in gold)*sum((x-pm)**2 for x in pred))
    return dict(pearson_raw=sum((a-gm)*(b-pm) for a,b in zip(gold,pred))/denom if denom else None,
        qwk_clipped_half_up=train.qwk(gold,pred),rmse_raw=math.sqrt(sum((a-b)**2 for a,b in zip(gold,pred))/len(gold)),
        out_of_range_predictions=sum(x<0 or x>10 for x in pred))

def aggregate(root,out):
    if (out/'summary.json').exists():
        done=e.read(out/'summary.json')
        require(done['policy_sha256']==c.sha(out/'policy.json'),'Completed evaluation policy changed')
        for seed in train.SEEDS:
            require(pred_complete(out,seed),'Completed predictions missing')
            require(c.sha(out/('seed_%d/predictions.jsonl'%seed))==done['predictions_sha256'][str(seed)],'Completed prediction hash differs')
        print(json.dumps(done,ensure_ascii=False,indent=2));print('TEST_ALREADY_COMPLETE_VERIFIED');return
    rows=inputs(root);ids=[r['sample_id'] for r in rows]
    predictions={}
    for seed in train.SEEDS:
        require(pred_complete(out,seed),'All five predictions required before gold access')
        pp=e.rows(out/('seed_%d/predictions.jsonl'%seed))
        require(len(pp)==425 and [r['sample_id'] for r in pp]==ids,'Prediction coverage/order mismatch')
        require(all(a['question_id']==b['question_id'] for a,b in zip(pp,rows)),'Prediction question mismatch')
        predictions[seed]=[r['prediction_raw'] for r in pp]
    gp=root/'data/prepared/ads_paper_split_v1_seed42/test.labels.jsonl'
    require(c.sha(gp)==GOLD_SHA,'Test gold changed')
    labels=e.rows(gp);goldmap={r['sample_id']:r for r in labels}
    require(len(labels)==len(goldmap)==425 and set(goldmap)==set(ids),'Test gold identities')
    gold=[]
    for r in rows:
        g=goldmap[r['sample_id']];require(g.get('question_id',r['question_id'])==r['question_id'],'Gold question mismatch')
        y=float(g['gold_raw']);require(math.isfinite(y) and 0<=y<=10,'Invalid gold');gold.append(y)
    results={str(seed):metrics(gold,predictions[seed]) for seed in train.SEEDS}
    averages={}
    for key in ('pearson_raw','qwk_clipped_half_up','rmse_raw'):
        values=[results[str(s)][key] for s in train.SEEDS]
        averages[key]={'mean':statistics.mean(values) if all(x is not None for x in values) else None,
            'sample_std':statistics.stdev(values) if all(x is not None for x in values) else None}
    report=dict(status='ADS_BERT_FIVE_RUN_TEST_COMPLETE',test_records_per_run=425,completed_runs=5,
        per_seed=results,aggregate=averages,aggregation='Mean of five metrics, not ensemble predictions; SD ddof=1',
        policy_sha256=c.sha(out/'policy.json'),test_labels_sha256=GOLD_SHA,
        predictions_sha256={str(s):c.sha(out/('seed_%d/predictions.jsonl'%s)) for s in train.SEEDS},
        test_labels_used_for_selection=False,training_modified=False,
        reproduction_status='METHOD_RECONSTRUCTION_WITH_DOCUMENTED_DEVIATIONS',full_paper_reproduction_complete=False,
        next_step='Paper baseline/ablation comparisons under identical protocol, then other datasets/backbones')
    e.write(out/'summary.json',e.enc(report));print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','all','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=train.SEEDS);p.add_argument('--physical-gpu',default='auto');p.add_argument('--wait-hours',type=float,default=48)
    a=p.parse_args()
    if a.wait_hours<0 or (a.physical_gpu!='auto' and not a.physical_gpu.isdigit()):p.error('Invalid GPU/wait')
    for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):os.environ[name]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    root=a.root.resolve();out=root/'paper_method_v2/runs/sae_five_runs_test_v1';out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lockpath=out/('worker_%s.lock'%a.seed if a.command=='worker' else 'pipeline.lock')
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=freeze(root,out)
        if a.command=='freeze':print('TEST_PROTOCOL_FROZEN; test files not read');return
        if a.command=='worker':
            if a.seed is None:p.error('worker requires --seed')
            worker(a,root,out,protocol);return
        features(root,out,protocol)
        for seed in train.SEEDS:
            if not pred_complete(out,seed):
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--root',str(root),'--seed',str(seed),
                    '--physical-gpu',a.physical_gpu,'--wait-hours',str(a.wait_hours)],check=True)
        aggregate(root,out);print('FIVE_RUN_TEST_DONE',out/'summary.json',flush=True)

if __name__=='__main__':main()
