"""Immutable CSBench bridge to the original answer embedding and initial filter runners."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HASHES={'filter_embeddings_restored.py':'6f9fb73174952d3375a849cf1cb5b18713036e878731f55c000e40fc1a3276c3',
        'initial_filter_restored.py':'f750909d649570fecb2a37e1af36e7590dd1f883be170c8fccf2fbd70961105b'}
REV='183bb99aa7af74355fb58d16edf8c13ae7c5433e'
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def enc(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def freeze(p,b):
    if p.exists():
        if p.read_bytes()!=b:raise ValueError('Existing artifact differs: '+str(p))
    else:
        p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(b);tmp.replace(p)

def prepare(project,experiment,neighbors,budget):
    here=Path(__file__).parent
    for n,h in HASHES.items():
        if sha((here/n).read_bytes())!=h:raise ValueError('Original algorithm changed: '+n)
    exp=project/'runs/online_experiments'/experiment
    source=exp/'scale_stage2/layout/paper_method_v2/artifacts/retrieval_v1'
    generation=exp/'scale_stage3'
    sm=read(source/'summary.json');gm=read(generation/'summary.json');plan=read(generation/'plan_summary.json')
    if gm['status']!='CSBENCH_PS_GENERATION_COMPLETE_UNLABELED' or gm['remaining']!=0 or gm['errors']:
        raise ValueError('Generation incomplete')
    if sha((generation/'config.json').read_bytes())!=gm['config_sha256']:raise ValueError('Generation config changed')
    sources=read(source/'training_quadruples.json')
    if len(sources)!=345 or len({r['sample_id'] for r in sources})!=345:raise ValueError('Wrong CSBench train set')
    if not 1<=neighbors<len(sources) or not 1<=budget<=gm['candidates']:raise ValueError('Invalid filter counts')
    if gm['candidates']!=plan['candidate_slots'] or gm['valid_requests']!=plan['planned_requests']:raise ValueError('Generation coverage mismatch')
    payload={}
    for n in ('training_quadruples.json',):
        raw=(source/n).read_bytes()
        if sha(raw)!=sm['output_sha256'][n]:raise ValueError('Training source changed')
        payload['retrieval_v1/'+n]=raw
    payload['retrieval_v1/summary.json']=(source/'summary.json').read_bytes()
    for n,h in plan['files_sha256'].items():
        if Path(n).name!=n:raise ValueError('Invalid plan filename')
        raw=(generation/n).read_bytes()
        if sha(raw)!=h:raise ValueError('Generation plan changed: '+n)
        payload['ps_inputs_v1/'+n]=raw
    plan_raw=(generation/'plan_summary.json').read_bytes()
    payload['ps_inputs_v1/summary.json']=plan_raw
    candidate_raw=(generation/'candidates.jsonl').read_bytes()
    if sha(candidate_raw)!=gm['files_sha256']['candidates.jsonl']:raise ValueError('Candidate hash mismatch')
    payload['ps_generated_v1/candidates.jsonl']=candidate_raw
    payload['ps_generated_v1/manifest.json']=enc(dict(candidate_sha256=sha(candidate_raw),plan_sha256=sha(plan_raw),assigned_labels=0))
    out=exp/'scale_stage4';layout=out/'layout';artifact=layout/'paper_method_v2/artifacts'
    cfg=dict(experiment_id=experiment,neighbors=neighbors,budget=budget,training_records=345,
        original_contracts=HASHES,adapter_sha256=sha(Path(__file__).read_bytes()),
        input_sha256={n:sha(b) for n,b in payload.items()},
        limitations=['Numeric neighbors=5 is a reconstruction choice, not author-verified',
                    'Right truncation to 512 for embedding only; complete text retained',
                    'Original Algorithm1 reconstruction, stable ties and denominator floor retained'],
        api_calls=0,validation_accessed=False,test_accessed=False,assigned_labels=0)
    freeze(out/'config.json',enc(cfg))
    for n,b in payload.items():freeze(artifact/n,b)
    # Verify coverage through the original loader, without loading model or labels.
    import importlib.util
    spec=importlib.util.spec_from_file_location('filter_embed_contract',here/'filter_embeddings_restored.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    texts,records,facts=module.load_inputs(layout)
    if facts['candidate_records']!=gm['candidates']:raise ValueError('Candidate coverage changed')
    report=dict(status='CSBENCH_INITIAL_FILTER_INPUTS_READY',**facts,neighbors=neighbors,budget=budget,api_calls=0)
    freeze(out/'prepare_summary.json',enc(report))
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return out,layout,artifact

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--project',required=True,type=Path);p.add_argument('--model-root',required=True,type=Path)
    p.add_argument('--experiment-id',default='csbench_scale_online_v1');p.add_argument('--neighbors',type=int,default=5);p.add_argument('--budget',type=int,default=5000)
    a=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}',a.experiment_id):p.error('Invalid experiment ID')
    out=a.project.resolve()/'runs/online_experiments'/a.experiment_id/'scale_stage4';out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out,layout,artifact=prepare(a.project.resolve(),a.experiment_id,a.neighbors,a.budget)
        if a.command=='prepare':return
        model=a.model_root.resolve()/'models/text2vec-base-chinese'
        if not (model/REV/'offline_manifest.json').is_file():raise ValueError('Pinned model missing')
        link=layout/'models/text2vec-base-chinese';link.parent.mkdir(parents=True,exist_ok=True)
        if link.is_symlink():
            if link.resolve()!=model:raise ValueError('Different model link')
        elif link.exists():raise ValueError('Unexpected model directory')
        else:link.symlink_to(model,target_is_directory=True)
        here=Path(__file__).parent
        subprocess.run([sys.executable,'-u',str(here/'filter_embeddings_restored.py'),'run','--root',str(layout)],check=True)
        subprocess.run([sys.executable,'-u',str(here/'initial_filter_restored.py'),'run','--root',str(layout),'--neighbors',str(a.neighbors),'--budget',str(a.budget)],check=True)
        selection=artifact/('initial_filter_k%d_b%d_v1'%(a.neighbors,a.budget))
        summary=read(selection/'summary.json')
        for n,h in summary['files_sha256'].items():
            if sha((selection/n).read_bytes())!=h:raise ValueError('Selected output changed')
        summary.update(experiment_id=a.experiment_id,validation_accessed=False,selection_directory=str(selection))
        freeze(out/'summary.json',enc(summary))
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
