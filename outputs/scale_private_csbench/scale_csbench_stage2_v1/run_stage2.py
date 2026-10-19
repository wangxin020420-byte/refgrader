"""Training-only CSBench retrieval, original Eq10 runner, and original replacement policy."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

HASHES={'retrieval_contract':'a794080f450a8be6bbe213c9a4bc5f7dbd64f30b13a2564673db4dad2b54ba12',
        'embedding_contract':'5a507281fe6d5bd70398a9d290a30d02dfe7925ee9991d5c91f2abb0bcba7848',
        'replacement_contract':'cbfc0d0b3248afb730cc5b43ab19f0de3e02491c010607055e6514754a0c2648'}
REV='183bb99aa7af74355fb58d16edf8c13ae7c5433e'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def enc(obj):return (json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def freeze(p,raw):
    if p.exists():
        if p.read_bytes()!=raw:raise ValueError('Existing artifact changed: '+str(p))
    else:
        p.parent.mkdir(parents=True,exist_ok=True)
        temp=p.with_suffix(p.suffix+'.tmp');temp.write_bytes(raw);temp.replace(p)

def load(name):
    p=Path(__file__).with_name(name+'.py')
    if sha(p.read_bytes())!=HASHES[name]:raise ValueError('Original algorithm changed')
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def prepare(project,experiment,k,seed):
    r=load('retrieval_contract')
    exp=project/'runs/online_experiments'/experiment
    previous=exp/'scale_stage1'
    summary=read(previous/'summary.json')
    if summary['status']!='CSBENCH_TRAIN_EXTRACTION_AND_GRAPH_COMPLETE' or summary['remaining']!=0:
        raise ValueError('Stage1 incomplete')
    if sha((previous/'config.json').read_bytes())!=summary['config_sha256']:
        raise ValueError('Stage1 configuration changed')
    for n in ('graph.json','extractions.jsonl','source_units.jsonl','training_sources.json'):
        if sha((previous/n).read_bytes())!=summary['files_sha256'][n]:raise ValueError('Stage1 output changed: '+n)
    quads=read(previous/'training_sources.json')
    if len(quads)!=345 or len({q['sample_id'] for q in quads})!=345:raise ValueError('Wrong CSBench training coverage')
    graph=read(previous/'graph.json')
    if graph['extractions_sha256']!=summary['files_sha256']['extractions.jsonl']:raise ValueError('Graph extraction mismatch')
    extraction={x['unit_id']:x for x in rows(previous/'extractions.jsonl')}
    units={x['unit_id']:x for x in rows(previous/'source_units.jsonl')}
    if set(extraction)!=set(units):raise ValueError('Extraction coverage mismatch')
    for q in quads:
        uid='student:'+q['sample_id'];q['extraction_unit_id']=uid
        if units[uid]['text']!=q['a']:raise ValueError('Training text changed')
    weights=r.sampling_weights([dict(sample_id=q['sample_id'],gold_raw=q['l']) for q in quads])
    if not math.isclose(sum(x['probability'] for x in weights),1,abs_tol=1e-12):raise ValueError('Invalid weights')
    index=r.build_index(graph)
    out=exp/'scale_stage2'
    # Original Eq10 expects this layout. It is wholly inside this experiment.
    layout=out/'layout'
    artifact=layout/'paper_method_v2/artifacts'
    cfg=dict(experiment_id=experiment,stage1_config_sha256=summary['config_sha256'],
        stage1_files_sha256=summary['files_sha256'],top_k=k,seed=seed,contracts=HASHES,
        adapter_sha256=sha(Path(__file__).read_bytes()),training_records=345,
        method='Existing SCALE reconstruction; not RefGrader scoring',
        deviations=['GLM model substitution','k=5 default and seed=42 are reconstruction choices',
                    'Original prose graph interpretation retained','Pinned Chinese text2vec checkpoint is not paper-verified'],
        validation_accessed=False,test_accessed=False,api_calls=0)
    freeze(out/'config.json',enc(cfg))
    for n in ('graph.json','extractions.jsonl'):
        freeze(artifact/'full_graph_v1'/n,(previous/n).read_bytes())
    blobs={'index.json':enc(index),'sampling_weights.json':enc(weights),'training_quadruples.json':enc(quads)}
    report=dict(status='CSBENCH_RETRIEVAL_READY',training_records=345,graph_sha256=sha((previous/'graph.json').read_bytes()),
        output_sha256={n:sha(b) for n,b in blobs.items()},api_calls=0,validation_accessed=False,test_accessed=False)
    for n,b in blobs.items():freeze(artifact/'retrieval_v1'/n,b)
    freeze(artifact/'retrieval_v1/summary.json',enc(report))
    print('RETRIEVAL_READY',len(quads),'training rows',flush=True)
    return out,layout,artifact,quads,extraction,index,graph

def replace(artifact,quads,extraction,index,graph,k,seed):
    import numpy as np
    m=load('replacement_contract')
    eq=artifact/'eq10_v1';summary=read(eq/'summary.json');cfg=read(eq/'config.json')
    if summary['status']!='PAPER_EQ10_RANKING_COMPLETE' or sha((eq/'config.json').read_bytes())!=summary['config_sha256']:
        raise ValueError('Ranking incomplete or changed')
    if cfg['graph_sha256']!=sha((artifact/'full_graph_v1/graph.json').read_bytes()) or cfg['index_sha256']!=sha((artifact/'retrieval_v1/index.json').read_bytes()):
        raise ValueError('Ranking input mismatch')
    ranks={}
    for item in summary['rankings']:
        path=eq/item['file']
        if sha(path.read_bytes())!=item['sha256']:raise ValueError('Ranking cache changed')
        meta=read(path.with_suffix('.json'));ids=meta['summary_ids'];c=item['type_id']
        if meta['type_id']!=c or ids!=sorted(index['summaries_by_type'][c]):raise ValueError('Ranking identities mismatch')
        with np.load(path,allow_pickle=False) as data:order=data['order'];scores=data['scores']
        if order.shape!=scores.shape or order.shape!=(len(ids),len(ids)) or not np.isfinite(scores).all():raise ValueError('Invalid ranking')
        ranks[c]=(ids,{w:i for i,w in enumerate(ids)},order,scores)
    nodes={n['id']:n for n in graph['nodes']};records=[];empty=[]
    for q in sorted(quads,key=lambda x:x['sample_id']):
        uid=q['extraction_unit_id'];triples=extraction[uid]['triples']
        if not triples:empty.append(q['sample_id'])
        for i,t in enumerate(triples):
            w='w:'+sha(t['w'].encode());c='c:'+sha(t['c'].encode())
            ids,pos,order,scores=ranks[c];row=pos[w]
            if sorted(order[row].tolist())!=list(range(len(ids))) or not np.all(scores[row][:-1]>=scores[row][1:]-1e-7):raise ValueError('Invalid ranking order')
            result=m.select(q['a'],t,[ids[int(j)] for j in order[row]],scores[row],index['phrases_by_summary'],nodes,k,'%d|%s|%d'%(seed,uid,i))
            result.update(record_id=sha(('%s|%d'%(uid,i)).encode()),sample_id=q['sample_id'],extraction_unit_id=uid,
                triplet_index=i,source_summary_id=w,entity_type_id=c,source_answer_sha256=sha(q['a'].encode()),
                assigned_label=None,status='SUBSTITUTED_INPUT_ONLY_NOT_ANNOTATED_SYNTHETIC_ANSWER')
            records.append(result)
    raw=''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in records).encode()
    folder=artifact/'replacement_inputs_v1'
    freeze(folder/'replacement_records.jsonl',raw)
    freeze(folder/'sources_without_triplets.json',enc(empty))
    report=dict(status='CSBENCH_RETRIEVAL_RANKING_REPLACEMENT_COMPLETE',training_records=len(quads),
        replacement_records=len(records),changed_answer_records=sum(x['answer_changed'] for x in records),
        source_answers_without_triplets=len(empty),top_k=k,seed=seed,assigned_labels=0,
        synthetic_answers_generated=0,api_calls=0,training_started=False,validation_accessed=False,test_accessed=False,
        reproduction_status='METHOD_RECONSTRUCTION_WITH_DOCUMENTED_DEVIATIONS',
        output_sha256={'replacement_records.jsonl':sha(raw)},
        next_step='SCALE weighted source sampling and style generation; replacements are not labeled training examples')
    freeze(folder/'summary.json',enc(report))
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--project',required=True,type=Path)
    p.add_argument('--model-root',required=True,type=Path,help='Existing SCALE root containing models/')
    p.add_argument('--experiment-id',default='csbench_scale_online_v1')
    p.add_argument('--top-k',type=int,default=5);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.top_k<1 or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}',a.experiment_id):p.error('Invalid parameters')
    out=a.project.resolve()/'runs/online_experiments'/a.experiment_id/'scale_stage2'
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out,layout,artifact,quads,extraction,index,graph=prepare(a.project.resolve(),a.experiment_id,a.top_k,a.seed)
        if a.command=='prepare':return
        model=a.model_root.resolve()/'models/text2vec-base-chinese'
        if not (model/REV/'offline_manifest.json').is_file():raise ValueError('Pinned model not found: '+str(model/REV))
        link=layout/'models/text2vec-base-chinese';link.parent.mkdir(parents=True,exist_ok=True)
        if link.is_symlink():
            if link.resolve()!=model:raise ValueError('Different model link')
        elif link.exists():raise ValueError('Expected isolated model symlink')
        else:link.symlink_to(model,target_is_directory=True)
        eq=load('embedding_contract')
        eq.execute(layout)
        report=replace(artifact,quads,extraction,index,graph,a.top_k,a.seed)
        freeze(out/'summary.json',enc(report))
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
