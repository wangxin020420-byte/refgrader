"""Five SAE repetitions with validation-QWK selection and epoch-boundary resume."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import sae_core as c
import token_core as t

SEEDS=[42,43,44,45,46]
COUNTS=[1468,1849,1798,2366,1722]
HASHES=['3b084b0254bf6a2956056bfed71088cdcf848c6b2b086d2bf53bf26ee735c13a','5f9f8139413ecca4fbb326f0c4d6ee198156c1491c1ac9c0fcfa8818459937e5','c73c368d75f9defc3979734a41d194fffb9253b774e0cd0d2ed20471c1d303b0','93f9095df48b458c6882c61ffe1319c9f4f1ba462f3a2ca19fbceed38efc2347','1aac7df70656225b3a71f1b446793397886635aea4b3fd11e2a8c81d972573ad']
VAL_SHA='5133be1294dd2a86a957279f0b091b076e8ad1ea11d49aed1ea39e9b11c3b526'

def require(ok,msg):
    if not ok:raise ValueError(msg)

def qwk(gold,pred):
    require(len(gold)==len(pred)>0 and all(math.isfinite(x) for x in gold+pred),'Invalid scores')
    a=[int(math.floor(min(10,max(0,x))+.5)) for x in gold]
    b=[int(math.floor(min(10,max(0,x))+.5)) for x in pred]
    observed=sum((x-y)**2 for x,y in zip(a,b))
    ca=[a.count(i) for i in range(11)];cb=[b.count(i) for i in range(11)]
    expected=sum(ca[i]*cb[j]*(i-j)**2 for i in range(11) for j in range(11))/len(a)
    require(expected>0,'Undefined QWK')
    return 1-observed/expected

def setup(root,epochs):
    base=root/'paper_method_v2'
    train=base/'artifacts/sae_inputs_five_runs_v1'
    val=base/'runs/sae_validation_inputs_v1'
    vs=c.read(val/'complete.json')
    require(vs['status']=='SAE_FIXED_VALIDATION_INPUTS_READY' and vs['records']==50,'Validation incomplete')
    require(c.sha(val/'config.json')==vs['config_sha256'],'Validation config changed')
    require(c.sha(val/'validation_features.jsonl')==VAL_SHA==vs['files_sha256']['validation_features.jsonl'],'Validation features changed')
    validation=t.lines(val/'validation_features.jsonl');require(len(validation)==50 and len({r['id'] for r in validation})==50,'Validation IDs')
    for row in validation:t.validate_feature(row)
    datasets={};configs=[]
    for seed,count,digest in zip(SEEDS,COUNTS,HASHES):
        folder=train/('seed_%d'%seed);summary=c.read(folder/'summary.json');cfg=c.read(folder/'config.json')
        require(c.sha(folder/'config.json')==summary['config_sha256'],'Training config changed')
        require(c.sha(folder/'train_features.jsonl')==digest==summary['files_sha256']['train_features.jsonl'],'Training features changed')
        rows=t.lines(folder/'train_features.jsonl')
        require(len(rows)==count and len({r['id'] for r in rows})==count,'Training count/IDs')
        require(sum(r['kind']=='original_training' for r in rows)==1107 and sum(r['kind']=='synthetic_candidate' for r in rows)==count-1107,'Training composition')
        require(not {r['sample_id'] for r in rows}&{r['sample_id'] for r in validation},'Training/validation overlap')
        for row in rows:t.validate_feature(row)
        configs.append({k:v for k,v in cfg.items() if k not in ('seed','entity_input_sha256')});datasets[seed]=rows
    require(all(cfg==configs[0] for cfg in configs),'Encoding differs across seeds')
    require(c.read(val/'config.json')['encoding']==configs[0],'Validation encoding differs')
    model=root/'models/bert-base-chinese'/c.REV
    for name,digest in configs[0]['tokenizer_files_sha256'].items():require(c.sha(model/name)==digest,'Tokenizer/model config changed')
    weights={p.name:c.sha(p) for p in sorted(model.iterdir()) if p.suffix in ('.bin','.safetensors','.json','.txt') and p.is_file()}
    require(any(n.endswith(('.bin','.safetensors')) for n in weights),'Missing BERT weights')
    cfg=dict(protocol='SAE_V2_FIVE_RUN_VALIDATION_QWK',epochs=epochs,seeds=SEEDS,
        builder_sha256=c.sha(Path(__file__)),model_code_sha256=c.sha(Path(c.__file__)),
        token_code_sha256=c.sha(Path(t.__file__)),model_files_sha256=weights,
        training_features_sha256=dict(zip(map(str,SEEDS),HASHES)),validation_features_sha256=VAL_SHA,
        encoding=configs[0],initialization='Fresh pretrained BERT per seed; shared text/knowledge encoder',
        checkpoint_selection='Highest validation QWK; earliest epoch on tie',
        qwk='Clip prediction/gold to 0-10 then half-up rounding; fixed 11 categories; raw training labels unchanged',
        microbatch=1,original_effective_batch=16,synthetic_effective_batch=64,
        encoder_lr=5e-5,head_lr=1e-3,optimizer='AdamW foreach=False, weight_decay=0',
        precision='FP32',scheduler='constant',gradient_clip=1.0,
        resume='Atomic epoch checkpoint with model, optimizer, RNG, best model and history; interrupted epoch replayed',
        architecture='Literal equations 15-20: tuple mean, learned entity pooling, singleton cross attention, residual CLS, scalar MSE',
        implementation_choices=['10 epoch default not established from paper','Mixed homogeneous original/synthetic batches',
            'Shared encoder, tuple mean pooling, D_star=hidden size, literal singleton fusion',
            'Reference concatenation/truncation and QWK discretization','Seeds 42-46'],
        reproduction_status='METHOD_RECONSTRUCTION_WITH_DOCUMENTED_DEVIATIONS',
        unresolved=['GLM replacement','Filtering k and beta scale unverified','Incomplete manual semantic review'],
        validation_accessed=True,test_accessed=False,api_calls=0)
    out=base/'runs/sae_five_runs_valqwk_v1';out.mkdir(parents=True,exist_ok=True)
    c.save(out/'protocol.json',cfg,frozen=True)
    return datasets,validation,model,out,c.sha(out/'protocol.json')

def completed(folder,protocol):
    if not (folder/'complete.json').exists():return None
    report=c.read(folder/'complete.json');require(report['protocol_sha256']==protocol,'Completed protocol changed')
    for name,digest in report['files_sha256'].items():require(c.sha(folder/name)==digest,'Completed artifact changed: '+name)
    return report

def predict(torch,model,rows,device,pad):
    model.eval();scores=[]
    with torch.inference_mode():
        for i in range(len(rows)):
            batch=c.collate(torch,rows,[i],device,pad)
            v=float(model(*batch[:3]).item());require(math.isfinite(v),'Nonfinite prediction');scores.append(v)
    return scores

def worker(args):
    datasets,val,modelpath,out,protocol=setup(args.root.resolve(),args.epochs)
    folder=out/('seed_%d'%args.seed);folder.mkdir(parents=True,exist_ok=True)
    if completed(folder,protocol):print('ALREADY_COMPLETE',args.seed);return
    require(shutil.disk_usage(out).free>=10*1024**3,'Need 10 GiB free for checkpoints')
    deadline=time.monotonic()+args.wait_hours*3600
    while True:
        try:gpu=c.select_gpu(args.physical_gpu);break
        except RuntimeError as exc:
            if time.monotonic()>=deadline:raise
            print('WAIT_GPU',str(exc),flush=True);time.sleep(60)
    import torch
    import transformers
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require(torch.cuda.is_available(),'CUDA unavailable')
    device=torch.device('cuda:0');rows=datasets[args.seed]
    c.save(folder/'runtime.json',{'torch':torch.__version__,'transformers':transformers.__version__},frozen=True)
    if not (folder/'architecture_checks.json').exists():
        c.save(folder/'architecture_checks.json',c.architecture_checks(torch,transformers),frozen=True)
    model,optimizer=c.create_model(torch,transformers,modelpath,args.seed,device)
    pad=c.read(modelpath/'config.json').get('pad_token_id',0)
    if not (folder/'smoke.json').exists():
        c.save(folder/'smoke.json',c.smoke(torch,model,optimizer,rows,device,pad,1),frozen=True)
        del model,optimizer;torch.cuda.empty_cache()
        model,optimizer=c.create_model(torch,transformers,modelpath,args.seed,device)
    state={'epoch':0,'steps':0,'history':[],'best_qwk':-1e100,'best_epoch':None,'best_predictions':None}
    best=None;latest=folder/'latest.pt'
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        require(ck['protocol_sha256']==protocol and ck['seed']==args.seed,'Checkpoint identity changed')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        state=ck['state'];best=ck['best'];torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng']);del ck
        print('RESUME',args.seed,'completed_epochs',state['epoch'],flush=True)
    gold=[r['label'] for r in val]
    for epoch in range(state['epoch'],args.epochs):
        schedule=c.batches(rows,epoch,args.seed);sse=0;seen=0
        for j,indices in enumerate(schedule,1):
            sse+=c.train_step(torch,model,optimizer,rows,indices,device,pad,1);seen+=len(indices);state['steps']+=1
            if j%10==0 or j==len(schedule):print('SEED',args.seed,'EPOCH',epoch+1,'STEP',j,'/',len(schedule),flush=True)
        require(seen==len(rows),'Epoch coverage changed')
        pred=predict(torch,model,val,device,pad);score=qwk(gold,pred)
        if score>state['best_qwk']:
            state.update(best_qwk=score,best_epoch=epoch+1,best_predictions=pred)
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        state['history'].append({'epoch':epoch+1,'train_mse':sse/seen,'validation_qwk':score})
        state['epoch']=epoch+1
        c.torch_save(torch,latest,dict(protocol_sha256=protocol,seed=args.seed,model=model.state_dict(),optimizer=optimizer.state_dict(),
            state=state,best=best,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate()))
        c.save(folder/'history.json',state['history'])
        c.save(folder/'progress.json',dict(seed=args.seed,epoch=state['epoch'],best_epoch=state['best_epoch'],best_qwk=state['best_qwk'],gpu=gpu))
        print('EPOCH_SAVED',args.seed,epoch+1,'QWK',score,'BEST_EPOCH',state['best_epoch'],flush=True)
    require(best is not None,'No best model')
    c.torch_save(torch,folder/'model_best.pt',dict(protocol_sha256=protocol,seed=args.seed,best_epoch=state['best_epoch'],model=best))
    del model,optimizer,best;torch.cuda.empty_cache()
    model,optimizer=c.create_model(torch,transformers,modelpath,args.seed,device);del optimizer
    ck=torch.load(folder/'model_best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);del ck
    actual=predict(torch,model,val,device,pad)
    require(max(abs(a-b) for a,b in zip(actual,state['best_predictions']))<=1e-5,'Reload predictions differ')
    require(abs(qwk(gold,actual)-state['best_qwk'])<1e-12,'Reload QWK differs')
    c.save(folder/'history.json',state['history'])
    c.save(folder/'validation_predictions.json',[{'sample_id':r['sample_id'],'gold':r['label'],'prediction_raw':v} for r,v in zip(val,actual)])
    names=['model_best.pt','history.json','validation_predictions.json','smoke.json','architecture_checks.json','runtime.json']
    report=dict(status='SAE_SEED_TRAINING_COMPLETE',seed=args.seed,protocol_sha256=protocol,training_records=len(rows),
        completed_epochs=args.epochs,optimizer_steps=state['steps'],best_epoch=state['best_epoch'],best_validation_qwk=state['best_qwk'],
        weights_reload_verified=True,validation_accessed=True,test_accessed=False,api_calls=0,
        files_sha256={n:c.sha(folder/n) for n in names})
    c.save(folder/'complete.json',report);print(json.dumps(report,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','all','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--physical-gpu',default='auto');p.add_argument('--wait-hours',type=float,default=48);args=p.parse_args()
    if args.epochs<1 or args.wait_hours<0 or (args.physical_gpu!='auto' and not args.physical_gpu.isdigit()):p.error('Invalid epochs/wait-hours/GPU')
    for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):os.environ[name]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    if args.command=='worker':
        if args.seed is None:p.error('worker requires --seed')
        import fcntl
        folder=args.root.resolve()/('paper_method_v2/runs/sae_five_runs_valqwk_v1/seed_%d'%args.seed)
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            worker(args)
        return
    _,_,_,out,protocol=setup(args.root.resolve(),args.epochs)
    if args.command=='preflight':print('SAE_V2_PREFLIGHT_OK');return
    import fcntl
    with (out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reports={}
        for seed in SEEDS:
            folder=out/('seed_%d'%seed)
            if not completed(folder,protocol):
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--root',str(args.root.resolve()),
                    '--seed',str(seed),'--epochs',str(args.epochs),'--physical-gpu',args.physical_gpu,'--wait-hours',str(args.wait_hours)],check=True)
            reports[str(seed)]=completed(folder,protocol);require(reports[str(seed)] is not None,'Missing completion')
        c.save(out/'summary.json',dict(status='FIVE_SAE_RUNS_TRAINED_VALIDATION_SELECTED',completed_runs=5,seeds=SEEDS,
            best_epochs={s:r['best_epoch'] for s,r in reports.items()},runs=reports,test_accessed=False,api_calls=0,
            paper_final_model_evaluation_complete=False,next_step='Freeze five selected models; prepare reserved test inputs and evaluate each model, report five-run mean and variability'))
        print('FIVE_SAE_RUNS_COMPLETE',out/'summary.json',flush=True)

if __name__=='__main__':main()
