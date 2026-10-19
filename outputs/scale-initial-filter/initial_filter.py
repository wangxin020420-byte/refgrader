"""Offline reconstruction of SCALE Eq12-13 / Algorithm1; leaves all labels unset."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(key,'4')
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
def enc(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf8')


def write(p,data,mutable=False):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not mutable:
        if p.read_bytes()!=data:raise ValueError('Existing artifact differs: '+str(p))
        return
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_bytes(data);temp.replace(p)


def load(root):
    base=root/'paper_method_v1/artifacts'
    emb=base/'initial_filter_embeddings_v2_max512'
    report=read(emb/'summary.json')
    if report['status']!='INITIAL_FILTER_EMBEDDINGS_READY':raise ValueError('Embeddings incomplete')
    for name,key in [('config.json','config_sha256'),('records.json','records_sha256'),
                     ('texts.json','texts_sha256'),('truncation_audit.json','truncation_audit_sha256')]:
        if sha(emb/name)!=report[key]:raise ValueError('Embedding metadata changed: '+name)
    source_path=base/'retrieval_v1/training_quadruples.json'
    candidate_path=base/'ps_generated_v1/candidates.jsonl'
    if sha(source_path)!=report['sources_sha256'] or sha(candidate_path)!=report['candidates_sha256']:
        raise ValueError('Source/candidate data differ from embedding inputs')
    manifest=base/'ps_generated_v1/manifest.json'
    if sha(manifest)!=report['generation_manifest_sha256']:raise ValueError('Generation manifest changed')
    if read(manifest)['candidate_sha256']!=sha(candidate_path):raise ValueError('Candidate manifest mismatch')
    texts=read(emb/'texts.json');records=read(emb/'records.json');parts=[];start=0
    for chunk in report['chunks']:
        path=emb/chunk['file']
        if chunk['start']!=start or sha(path)!=chunk['sha256']:raise ValueError('Chunk order/hash mismatch')
        array=np.load(path,allow_pickle=False)
        if array.shape!=(chunk['count'],768) or not np.isfinite(array).all():raise ValueError('Invalid chunk')
        parts.append(array);start+=len(array)
    vectors=np.concatenate(parts).astype('float64')
    if start!=len(texts) or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):raise ValueError('Invalid embeddings')
    vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
    sources=read(source_path);candidates=sorted(rows(candidate_path),key=lambda r:r['variant_id'])
    mapping={}
    for record in records:
        key=(record['kind'],record['id'])
        if key in mapping:raise ValueError('Duplicate embedding ID')
        mapping[key]=record
    original_indices=[];candidate_indices=[]
    for kind,items,idfield,textfield,target in [
        ('original',sources,'sample_id','a',original_indices),
        ('candidate',candidates,'variant_id','answer',candidate_indices)]:
        for item in items:
            record=mapping[(kind,item[idfield])];idx=record['vector_row']
            if texts[idx]!=item[textfield] or record['sample_id']!=item['sample_id']:raise ValueError('Text/source mapping mismatch')
            target.append(idx)
    if len(mapping)!=len(sources)+len(candidates):raise ValueError('Unexpected records')
    if len(set(x['sample_id'] for x in sources))!=len(sources) or len(set(x['variant_id'] for x in candidates))!=len(candidates):raise ValueError('Duplicate IDs')
    if any(x.get('assigned_label') is not None for x in candidates):raise ValueError('Candidates already labeled')
    if any(x['sample_id'] not in {s['sample_id'] for s in sources} for x in candidates):raise ValueError('Unknown source')
    return sources,candidates,vectors[original_indices],vectors[candidate_indices],report,sha(emb/'summary.json')


def fresh_state(n):
    return {'trace':[],'round':1,'cursor':0,'round_size':n}


def selection_steps(original,candidates,k,budget,state,epsilon=1e-8):
    """Round snapshots of A; neighbors always S0; R is the global candidate pool."""
    n=len(original);m=len(candidates)
    if not 1<=k<n or not 1<=budget<=m:raise ValueError('Invalid k or budget')
    chosen=[x['candidate_index'] for x in state['trace']]
    if len(chosen)!=len(set(chosen)) or any(not 0<=i<m for i in chosen):raise ValueError('Invalid checkpoint selections')
    anchors=list(range(-1,-n-1,-1))+chosen
    if not 0<=state['cursor']<=state['round_size']<=len(anchors):raise ValueError('Invalid checkpoint cursor')
    active=np.ones(m,dtype=bool);active[chosen]=False
    cross=np.clip(candidates@original.T,-1,1)
    original_sim=np.clip(original@original.T,-1,1)
    distances=1-cross
    cache={}
    while len(state['trace'])<budget:
        if state['cursor']==state['round_size']:
            state['round']+=1;state['cursor']=0;state['round_size']=len(anchors)
        anchor=anchors[state['cursor']]
        if anchor not in cache:
            if anchor<0:
                idx=-anchor-1;sims=original_sim[idx].copy();sims[idx]=-np.inf
                numerator=distances[:,idx]
            else:
                sims=cross[anchor]
                numerator=1-np.clip(candidates@candidates[anchor],-1,1)
            neighbors=np.argsort(-sims,kind='stable')[:k]
            denominator=distances[:,neighbors].mean(axis=1)
            ratios=numerator/np.maximum(denominator,epsilon)
            cache[anchor]=(neighbors,numerator,denominator,ratios)
        neighbors,numerator,denominator,ratios=cache[anchor]
        scores=np.where(active,ratios,-np.inf)
        winner=int(np.argmax(scores))
        entry={'step':len(state['trace'])+1,'round':state['round'],'anchor_code':int(anchor),
               'candidate_index':winner,'neighbor_original_indices':[int(x) for x in neighbors],
               'distance_to_anchor':float(numerator[winner]),
               'mean_neighbor_distance':float(denominator[winner]),
               'ratio':float(ratios[winner]),'denominator_floored':bool(denominator[winner]<epsilon)}
        state['trace'].append(entry);state['cursor']+=1
        active[winner]=False;anchors.append(winner)
        yield entry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','run'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--neighbors',type=int,required=True)
    parser.add_argument('--budget',type=int,default=5000)
    args=parser.parse_args();root=args.root.resolve()
    sources,candidates,original,synthetic,upstream,emb_hash=load(root)
    if not 1<=args.neighbors<len(sources):parser.error('neighbors must be between 1 and n_original-1')
    if not 1<=args.budget<=len(candidates):parser.error('budget must be between 1 and candidate count')
    out=root/('paper_method_v1/artifacts/initial_filter_k%d_b%d_v1'%(args.neighbors,args.budget))
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config={'neighbors':args.neighbors,'budget':args.budget,'embedding_summary_sha256':emb_hash,
                'script_sha256':sha(Path(__file__)),'numpy':np.__version__,'epsilon':1e-8,
                'k_basis':'Explicit reconstruction assumption; paper prints k=a, numeric author setting unverified.',
                'scope':'All original training records and all generated candidates, following Algorithm1 global sets.',
                'iteration':'Stable input order for originals; iterate snapshot of A per round; append selected candidates for subsequent rounds.',
                'budget_rule':'Exactly budget new candidates; enforce max budget inside loop, resolving pseudocode off-by-one/overshoot.',
                'ties':'Stable original order for neighbors; ascending variant_id for candidate argmax ties.',
                'identity':'Exclude original anchor by ID from S0 neighbors, not by equal text; retain duplicate and blank source records.',
                'labels':'No labels assigned or inherited; preserve original candidate source/question ownership.',
                'selection_basis':'Paper Eq12-13 with cosine clipped [-1,1]; denominator floor=1e-8 for degenerate neighborhoods.',
                'reproduction_status':'METHOD_RECONSTRUCTION_WITH_UNVERIFIED_K',
                'training_records':len(sources),'candidate_records':len(candidates),
                'blank_original_answers':upstream['blank_original_answers'],
                'truncated_unique_texts':upstream['truncated_unique_texts']}
        write(out/'config.json',enc(config))
        config_hash=sha(out/'config.json')
        preflight={'status':'INITIAL_FILTER_PREFLIGHT_OK','training_records':len(sources),
                   'candidate_records':len(candidates),'neighbors':args.neighbors,'budget':args.budget,
                   'k_verified_from_author':False,'offline':True,'api_calls':0,'test_accessed':False}
        print(json.dumps(preflight,ensure_ascii=False,indent=2),flush=True)
        if args.command=='preflight':return
        summary_path=out/'summary.json'
        if summary_path.exists():
            done=read(summary_path)
            if done['config_sha256']!=config_hash:raise ValueError('Completed config mismatch')
            for name,h in done['files_sha256'].items():
                if sha(out/name)!=h:raise ValueError('Completed output changed: '+name)
            print('ALREADY_COMPLETE: existing selection verified; not rerun')
            print(json.dumps(done,ensure_ascii=False,indent=2));return
        checkpoint=out/'checkpoint.json'
        if checkpoint.exists():
            saved=read(checkpoint)
            if saved['config_sha256']!=config_hash:raise ValueError('Checkpoint configuration changed')
            state=saved['state']
        else:state=fresh_state(len(sources))
        for entry in selection_steps(original,synthetic,args.neighbors,args.budget,state):
            if entry['step']%100==0 or entry['step']==args.budget:
                write(checkpoint,enc({'config_sha256':config_hash,'state':state}),mutable=True)
                print('SELECTED',entry['step'],'/',args.budget,flush=True)
        selected=[candidates[t['candidate_index']] for t in state['trace']]
        audit=[]
        for t in state['trace']:
            item=dict(t);anchor=t['anchor_code']
            item['anchor_id']=sources[-anchor-1]['sample_id'] if anchor<0 else candidates[anchor]['variant_id']
            item['anchor_kind']='original' if anchor<0 else 'selected_candidate'
            item['selected_variant_id']=candidates[t['candidate_index']]['variant_id']
            item['neighbor_original_ids']=[sources[i]['sample_id'] for i in t['neighbor_original_indices']]
            audit.append(item)
        write(out/'selected_candidates.jsonl',''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in selected).encode('utf8'))
        write(out/'selection_audit.json',enc(audit))
        summary={'status':'INITIAL_FILTER_COMPLETE_UNLABELED','selected_candidates':len(selected),
                 'remaining_candidates':len(candidates)-len(selected),'training_records':len(sources),
                 'neighbors':args.neighbors,'budget':args.budget,'k_verified_from_author':False,
                 'config_sha256':config_hash,'embedding_summary_sha256':emb_hash,
                 'reproduction_status':config['reproduction_status'],
                 'zero_denominator_selections':sum(x['denominator_floored'] for x in audit),
                 'files_sha256':{n:sha(out/n) for n in ['selected_candidates.jsonl','selection_audit.json']},
                 'assigned_labels':0,'training_started':False,'test_accessed':False,'api_calls':0,'offline':True}
        write(summary_path,enc(summary));print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
