"""V2 vanilla BERT annotators: five seeded runs, validation QWK selection, candidate scoring."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import bert_helpers as h

REV='8f23c25b06e129b6c986331a13d8d025a92cf0ea'
SEEDS=[42,43,44,45,46]

def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]

def score_hash(values):return hashlib.sha256(json.dumps(values,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def qwk(gold,pred):
    a=[int(math.floor(min(10,max(0,x))+0.5)) for x in gold]
    b=[int(math.floor(min(10,max(0,x))+0.5)) for x in pred]
    if not a or len(a)!=len(b) or not all(math.isfinite(x) for x in gold+pred):raise ValueError('Invalid QWK inputs')
    observed=sum((x-y)**2 for x,y in zip(a,b))
    ca=[a.count(i) for i in range(11)];cb=[b.count(i) for i in range(11)]
    expected=sum(ca[i]*cb[j]*(i-j)**2 for i in range(11) for j in range(11))/len(a)
    if expected==0:raise ValueError('Undefined validation QWK')
    return 1-observed/expected

from csbench_inputs import inputs

def wait_gpu():
    while True:
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        good=[]
        for row in text.strip().splitlines():
            i,uuid,free,util=[x.strip() for x in row.split(',')]
            if int(free)>=12000 and int(util)<=10:good.append((int(free),int(i),uuid))
        if good:
            free,i,uuid=max(good);print('GPU_SELECTED',i,uuid,free,flush=True);return uuid
        print('WAIT_GPU: need >=12000 MiB free and <=10% utilization; retry in 30s',flush=True);time.sleep(30)

def setup(root,epochs):
    import torch,transformers
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    train,val,jobs,hashes=inputs(root);modelpath=Path(os.environ['SCALE_MODEL_ROOT'])/'models/bert-base-chinese'/REV
    tok=transformers.AutoTokenizer.from_pretrained(str(modelpath),local_files_only=True)
    tok.truncation_side='right';tok.padding_side='right'
    audit=[]
    def tokenize(rows,kind):
        tokens=[]
        for i,r in enumerate(rows):
            ref=h.reference_text(r);full=tok(ref,r['answer'],truncation=False,verbose=False)
            t=tok(ref,r['answer'],truncation='longest_first',max_length=512);tokens.append(t)
            if len(full['input_ids'])>512:audit.append({'kind':kind,'id':r.get('variant_id',r['sample_id']),'full_tokens':len(full['input_ids']),'kept_tokens':len(t['input_ids'])})
        return tokens
    tr=tokenize(train,'proxy_train');va=tokenize(val,'validation');ca=tokenize(jobs,'candidate')
    modelhash={str(p.relative_to(modelpath)):h.sha(p) for p in sorted(modelpath.rglob('*')) if p.is_file() and p.suffix in ('.json','.txt','.bin','.safetensors') and '.cache' not in p.parts}
    if not any(x.endswith(('.bin','.safetensors')) for x in modelhash):raise ValueError('No local model weights')
    config={'input_hashes':hashes,'model_files_sha256':modelhash,'model_revision':REV,'epochs':epochs,'seeds':SEEDS,
        'script_sha256':h.sha(Path(__file__)),'helpers_sha256':h.sha(Path(h.__file__)),
        'torch':torch.__version__,'transformers':transformers.__version__,
        'input':'references [SEP] answer; question excluded','reference_policy':'Join nonempty references in original order with newline',
        'max_length':512,'truncation':'right longest_first','microbatch':2,'effective_batch':16,
        'encoder_lr':5e-5,'head_lr':1e-3,'optimizer':'AdamW','weight_decay':0.0,'gradient_clip':1.0,'precision':'FP32','loss':'MSE CSBench normalized score_0_10',
        'checkpoint_selection':'highest validation QWK; earliest epoch wins ties',
        'qwk_policy':'clip gold and predictions to 0..10, floor(x+0.5), fixed 11 categories',
        'five_run_policy':'One annotator and candidate-score artifact per seed; do not average labels or select seed by validation',
        'implementation_choices':['10 epochs default not established from paper','seeds 42..46 not author supplied','multi-reference concatenation and truncation','QWK discretization','separate annotator per repetition is reconstruction choice; paper does not explicitly specify its repetition granularity'],
        'deviations':['GLM replaces Llama','initial k=5 unverified','CSBench per-question split: 345 train / 343 validation / 2723 test','Automatic visual transcription and proxy errors retained; no manual semantic correction'],
        'fine_filter_beta_scale':'unverified; this stage records raw predictions without selecting beta branch',
        'test_accessed':False,'api_calls':0,'human_review_performed':False,'semantic_accuracy_verified':False,'training_records':len(train),'validation_records':len(val),'candidate_records':len(jobs)}
    out=root/'runs/online_experiments/csbench_scale_online_clean_v2/annotator_bert_valqwk_v1';h.save(out/'config.json',config,immutable=True)
    h.save(out/'truncation_audit.json',audit,immutable=True)
    return torch,transformers,tok,modelpath,train,val,jobs,tr,va,ca,out,h.sha(out/'config.json')

def predict(model,tokens,tok,torch):
    model.eval();result=[]
    with torch.inference_mode():
        for group in h.groups(list(range(len(tokens))),2):
            batch=tok.pad([tokens[i] for i in group],padding=True,return_tensors='pt')
            result.extend(model(**{k:v.to('cuda:0') for k,v in batch.items()}).logits.reshape(-1).float().cpu().tolist())
    if len(result)!=len(tokens) or not all(math.isfinite(x) for x in result):raise ValueError('Invalid predictions')
    return result

def verify_complete(run,config_hash):
    if not (run/'complete.json').exists():return False
    report=h.read(run/'complete.json')
    if report['config_sha256']!=config_hash:raise ValueError('Completed config changed')
    for name,hashval in report['files_sha256'].items():
        if h.sha(run/name)!=hashval:raise ValueError('Completed output changed: '+name)
    print('ALREADY_COMPLETE',run.name,flush=True);return True

def worker(root,epochs,seed):
    out=root/'runs/online_experiments/csbench_scale_online_clean_v2/annotator_bert_valqwk_v1';run=out/('seed_%d'%seed)
    # Verify data and executable provenance even on the zero-GPU completion path.
    _,_,_,hashes=inputs(root);cfg=h.read(out/'config.json');config_hash=h.sha(out/'config.json')
    if cfg['input_hashes']!=hashes or cfg['epochs']!=epochs or cfg['script_sha256']!=h.sha(Path(__file__)) or cfg['helpers_sha256']!=h.sha(Path(h.__file__)):raise ValueError('Preflight configuration changed')
    if verify_complete(run,config_hash):return
    os.environ['CUDA_VISIBLE_DEVICES']=wait_gpu()
    torch,tf,tok,modelpath,train,val,jobs,tr,va,ca,out,ch=setup(root,epochs)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device('cuda:0');run.mkdir(parents=True,exist_ok=True)
    model,optimizer=h.make_model(modelpath,torch,tf,device,seed)
    latest=run/'latest.pt';start=0;history=[];best_q=-float('inf');best_epoch=None;best_state=None
    if latest.exists():
        cp=torch.load(str(latest),map_location='cpu',weights_only=False)
        if cp['config_sha256']!=ch or cp['seed']!=seed:raise ValueError('Resume mismatch')
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer'])
        torch.set_rng_state(cp['cpu_rng']);torch.cuda.set_rng_state(cp['cuda_rng'],device)
        start=cp['epoch'];history=cp['history'];best_q=cp['best_q'];best_epoch=cp['best_epoch'];best_state=cp['best_model'];del cp
        print('RESUME',seed,start,flush=True)
    else:
        model.train();longest=sorted(range(len(tr)),key=lambda i:-len(tr[i]['input_ids']))[:16]
        loss=h.step(model,optimizer,longest,train,tr,tok,torch,device)
        h.save(run/'smoke.json',{'status':'ANNOTATOR_SMOKE_OK','config_sha256':ch,'loss_debug_only':loss})
        del model,optimizer;torch.cuda.empty_cache()
        model,optimizer=h.make_model(modelpath,torch,tf,device,seed)
    for epoch in range(start,epochs):
        if shutil.disk_usage(root).free<8*1024**3:raise RuntimeError('Need 8 GiB free for checkpoints')
        model.train();order=list(range(len(train)));random.Random(seed+epoch).shuffle(order);total=0.0
        groups=h.groups(order)
        for n,g in enumerate(groups,1):
            loss=h.step(model,optimizer,g,train,tr,tok,torch,device);total+=loss*len(g)
            if n%10==0 or n==len(groups):print('SEED',seed,'EPOCH',epoch+1,'STEP',n,'/',len(groups),flush=True)
        pred=predict(model,va,tok,torch);score=qwk([r['label'] for r in val],pred)
        if score>best_q:
            best_q=score;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append({'epoch':epoch+1,'training_mse':total/len(train),'validation_qwk':score})
        cp={'config_sha256':ch,'seed':seed,'epoch':epoch+1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device),'history':history,'best_q':best_q,'best_epoch':best_epoch,'best_model':best_state}
        temp=run/'latest.pt.tmp'
        with temp.open('wb') as f:torch.save(cp,f);f.flush();os.fsync(f.fileno())
        temp.replace(latest);del cp
        print('EPOCH_SAVED',epoch+1,'VALIDATION_QWK',score,'BEST_EPOCH',best_epoch,flush=True)
    model.load_state_dict(best_state);del optimizer,best_state;torch.cuda.empty_cache()
    expected=predict(model,va,tok,torch)
    final=run/'model_best';model.save_pretrained(str(final),safe_serialization=True);tok.save_pretrained(str(final))
    del model;torch.cuda.empty_cache()
    model=tf.BertForSequenceClassification.from_pretrained(str(final),local_files_only=True).to(device)
    actual=predict(model,va,tok,torch)
    if max(abs(a-b) for a,b in zip(expected,actual))>1e-5:raise ValueError('Reload predictions differ')
    h.save(run/'history.json',history)
    h.save(run/'validation_predictions.json',[{'sample_id':r['sample_id'],'gold':r['label'],'score_raw':v} for r,v in zip(val,actual)])
    # Chunked, content-bound candidate scoring resumes without repeating complete chunks.
    scored=[]
    for begin in range(0,len(jobs),100):
        chunk=run/'score_chunks'/('%05d.json'%begin)
        ids=[j['variant_id'] for j in jobs[begin:begin+100]]
        if chunk.exists():
            saved=h.read(chunk)
            if saved['config_sha256']!=ch or saved['best_epoch']!=best_epoch or saved['ids']!=ids:raise ValueError('Scoring cache mismatch')
            values=saved['scores']
            if saved['scores_sha256']!=score_hash(values):raise ValueError('Scoring cache corrupted')
        else:
            values=predict(model,ca[begin:begin+100],tok,torch)
            h.save(chunk,{'config_sha256':ch,'best_epoch':best_epoch,'ids':ids,'scores':values,'scores_sha256':score_hash(values)},immutable=True)
        if len(values)!=len(ids) or not all(math.isfinite(x) for x in values):raise ValueError('Bad score chunk')
        for j,v in zip(jobs[begin:begin+100],values):
            r={k:x for k,x in j.items() if k!='references'}
            r.update(assigned_label=v,label_origin='annotator_raw_regression',annotator_seed=seed,absolute_gap_raw=abs(v-j['source_score']))
            scored.append(r)
        print('SCORED',seed,begin+len(values),'/5000',flush=True)
    path=run/'scored_candidates.jsonl';tmp=path.with_suffix('.jsonl.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored),encoding='utf8');tmp.replace(path)
    files=[p for p in final.iterdir() if p.is_file()]+[path,run/'history.json',run/'validation_predictions.json']
    report={'status':'ANNOTATOR_AND_SCORING_COMPLETE','seed':seed,'config_sha256':ch,'training_records':len(train),
        'epochs':epochs,'optimizer_steps':math.ceil(len(train)/16)*epochs,'best_epoch':best_epoch,'best_validation_qwk':best_q,
        'candidate_labels_assigned':5000,'fine_filter_applied':False,'beta_scale_verified':False,
        'out_of_range_candidate_predictions':sum(not 0<=r['assigned_label']<=10 for r in scored),
        'weights_reload_verified':True,'validation_records':len(val),'validation_accessed':True,'test_accessed':False,'api_calls':0,
        'experiment_status':'METHOD_RECONSTRUCTION_WITH_DOCUMENTED_DEVIATIONS',
        'files_sha256':{str(p.relative_to(run)):h.sha(p) for p in files}}
    h.save(run/'complete.json',report);print(json.dumps(report,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','all','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model-root',type=Path,default=Path(os.environ.get('SCALE_MODEL_ROOT','/data0/E125221219/projects/refgrader-reproductions/scale')));p.add_argument('--epochs',type=int,default=10);p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args()
    if a.epochs<1:p.error('epochs must be positive')
    if a.command=='worker' and a.seed is None:p.error('worker requires --seed')
    root=a.root.resolve()
    os.environ['SCALE_MODEL_ROOT']=str(a.model_root.resolve())
    for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[key]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    if a.command=='preflight':
        setup(root,a.epochs);print('V2_ANNOTATOR_PREFLIGHT_OK: 345 train / 343 validation / 5000 candidates; test not accessed',flush=True);return
    import fcntl
    out=root/'runs/online_experiments/csbench_scale_online_clean_v2/annotator_bert_valqwk_v1';out.mkdir(parents=True,exist_ok=True)
    lockname='pipeline.lock' if a.command=='all' else 'seed_%d.lock'%a.seed
    with (out/lockname).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.command=='worker':worker(root,a.epochs,a.seed);return
        common=['--root',str(root),'--epochs',str(a.epochs)]
        subprocess.run([sys.executable,'-u',__file__,'preflight']+common,check=True)
        for seed in SEEDS:subprocess.run([sys.executable,'-u',__file__,'worker']+common+['--seed',str(seed)],check=True)
        reports=[h.read(out/('seed_%d'%s)/'complete.json') for s in SEEDS]
        h.save(out/'summary.json',{'status':'FIVE_ANNOTATORS_AND_CANDIDATE_SCORING_COMPLETE','seeds':SEEDS,'completed_runs':5,
            'candidate_records_per_run':5000,'fine_filter_applied':False,'test_accessed':False,'api_calls':0,
            'paper_final_model_evaluation_complete':False,'best_epochs':{str(r['seed']):r['best_epoch'] for r in reports},
            'next_step':'Resolve and document beta score scale; apply Eq14 per run, prepare retained synthetic entities, then train and evaluate final SAE repetitions.'})
        print('FIVE_ANNOTATORS_AND_SCORING_DONE',flush=True)

if __name__=='__main__':main()
