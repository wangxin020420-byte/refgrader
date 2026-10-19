"""Offline top-k / reverse-Ew replacement records for subsequent SCALE synthesis."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys


def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]
def encoded(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf8')


def select(source,triple,ranked_ids,scores,phrases,nodes,k,seed_key):
    if k<1 or not ranked_ids:raise ValueError('Empty candidate set or invalid k')
    rng=random.Random(int(sha(seed_key.encode('utf8')),16))
    width=min(k,len(ranked_ids));position=rng.randrange(width)
    chosen=ranked_ids[position]
    pool=sorted(phrases.get(chosen,[]))
    if not pool:raise ValueError('Missing reverse Ew phrases')
    chosen_phrase=rng.choice(pool);new=nodes[chosen_phrase]['text'];old=triple['t']
    start=source.find(old)
    if start<0 or not old:raise ValueError('Source phrase missing')
    answer=source[:start]+new+source[start+len(old):]
    return {'source_phrase':old,'replacement_phrase':new,'replacement_phrase_id':chosen_phrase,
            'replacement_summary_id':chosen,'replacement_summary':nodes[chosen]['text'],
            'effective_k':width,'selected_rank':position+1,'cosine':float(scores[position]),
            'source_span':[start,start+len(old)],'modified_answer':answer,
            'answer_changed':answer!=source,
            'same_summary_selected':chosen=='w:'+sha(triple['w'].encode('utf8')),
            'top_candidates':[{'summary_id':w,'cosine':float(s)} for w,s in zip(ranked_ids[:width],scores[:width])]}


def save(folder,files):
    for name,data in files.items():
        if (folder/name).exists() and (folder/name).read_bytes()!=data:raise ValueError('Existing differing artifact: '+name)
    folder.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():
        if not (folder/name).exists():
            with (folder/name).open('xb') as f:f.write(data)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--top-k',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    args=p.parse_args()
    if args.top_k<1:p.error('--top-k must be positive')
    import numpy as np
    root=args.root.resolve();base=root/'paper_method_v2';previous=base/'artifacts/retrieval_v1'
    eq=base/'artifacts/eq10_v1';eqsummary=read(eq/'summary.json');eqcfg=read(eq/'config.json')
    assert eqsummary['status']=='PAPER_EQ10_RANKING_COMPLETE'
    assert sha((eq/'config.json').read_bytes())==eqsummary['config_sha256']
    graphpath=base/'artifacts/full_graph_v1/graph.json';graph=read(graphpath)
    assert sha(graphpath.read_bytes())==eqcfg['graph_sha256']
    assert sha((previous/'index.json').read_bytes())==eqcfg['index_sha256']
    index=read(previous/'index.json');prior=read(previous/'summary.json')
    for n in ['training_quadruples.json','sampling_weights.json']:
        assert sha((previous/n).read_bytes())==prior['output_sha256'][n]
    extraction_path=base/'artifacts/full_graph_v1/extractions.jsonl'
    assert sha(extraction_path.read_bytes())==graph['extractions_sha256']
    extraction={r['unit_id']:r for r in lines(extraction_path)}
    quads=read(previous/'training_quadruples.json')
    assert len(quads)==1107 and len({q['sample_id'] for q in quads})==1107
    nodes={n['id']:n for n in graph['nodes']};ranks={}
    for item in eqsummary['rankings']:
        path=eq/item['file'];assert sha(path.read_bytes())==item['sha256']
        meta=read(path.with_suffix('.json'));c=item['type_id'];ids=meta['summary_ids']
        assert meta['type_id']==c and ids==sorted(index['summaries_by_type'][c])
        with np.load(path,allow_pickle=False) as z:order=z['order'];scores=z['scores']
        assert order.shape==scores.shape==(len(ids),len(ids))
        assert np.isfinite(scores).all()
        ranks[c]=(ids,{w:i for i,w in enumerate(ids)},order,scores)
    out=base/'artifacts/replacement_inputs_v1'
    cfg={'top_k':args.top_k,'seed':args.seed,'script_sha256':sha(Path(__file__).read_bytes()),
         'eq10_summary_sha256':sha((eq/'summary.json').read_bytes()),
         'extractions_sha256':sha(extraction_path.read_bytes()),
         'quadruples_sha256':sha((previous/'training_quadruples.json').read_bytes()),
         'sampling_weights_sha256':sha((previous/'sampling_weights.json').read_bytes()),
         'choices':['k and seed are reconstruction choices, not reported paper hyperparameters.',
                    'One independent replacement record per student-answer triplet; no final synthesis draw budget selected.',
                    'Retain self in T as in existing equations 6-9; do not resample no-op substitutions.',
                    'Uniform random choice among top min(k, count) summaries, then uniform choice among reverse Ew phrases.',
                    'Replace first exact occurrence only; other triplets are independent records, not sequential edits.',
                    'No original score is assigned to modified answers; correctness of a negative is not assumed.']}
    save(out,{'config.json':encoded(cfg)})
    records=[];empty=[]
    for q in sorted(quads,key=lambda r:r['sample_id']):
        uid=q['extraction_unit_id'];triples=extraction[uid]['triples']
        if not triples:empty.append({'sample_id':q['sample_id'],'reason':'no_extracted_triplets'});continue
        for i,triple in enumerate(triples):
            w='w:'+sha(triple['w'].encode('utf8'));c='c:'+sha(triple['c'].encode('utf8'))
            ids,positions,order,scores=ranks[c];row=positions[w];ranking=order[row]
            assert sorted(ranking.tolist())==list(range(len(ids))),'Invalid ranking permutation'
            assert np.all(scores[row][:-1]>=scores[row][1:]-1e-7),'Rank order invalid'
            selected=select(q['a'],triple,[ids[int(n)] for n in ranking],scores[row],
                            index['phrases_by_summary'],nodes,args.top_k,'%d|%s|%d'%(args.seed,uid,i))
            selected.update(record_id=sha(('%s|%d'%(uid,i)).encode('utf8')),sample_id=q['sample_id'],
                            extraction_unit_id=uid,triplet_index=i,source_summary_id=w,entity_type_id=c,
                            source_answer_sha256=sha(q['a'].encode('utf8')),assigned_label=None,
                            status='SUBSTITUTED_INPUT_ONLY_NOT_ANNOTATED_SYNTHETIC_ANSWER')
            records.append(selected)
    counts=Counter(r['sample_id'] for r in records)
    report={'status':'PAPER_TOPK_EW_REPLACEMENT_INPUTS_READY','training_records':len(quads),
            'source_answers_with_records':len(counts),'source_answers_without_triplets':len(empty),
            'replacement_records':len(records),'changed_answer_records':sum(r['answer_changed'] for r in records),
            'unchanged_answer_records':sum(not r['answer_changed'] for r in records),
            'same_summary_selected':sum(r['same_summary_selected'] for r in records),
            'top_k':args.top_k,'seed':args.seed,'style_rewrite_complete':False,'assigned_labels':0,
            'offline':True,'api_calls':0,'training_started':False,'test_accessed':False,
            'note':'All per-triplet alternatives enumerated; apply Eq5 sampling over source answers later, not over these rows.'}
    blobs={'replacement_records.jsonl':''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in records).encode('utf8'),
           'sources_without_triplets.json':encoded(empty),'preview.json':encoded(records[:5])}
    report['output_sha256']={k:sha(v) for k,v in blobs.items()};blobs['summary.json']=encoded(report)
    save(out,blobs);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
