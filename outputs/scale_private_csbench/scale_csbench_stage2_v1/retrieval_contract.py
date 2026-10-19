"""SCALE equations 5-9, plus reverse Ew index. No embedding approximation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


def sha(b):return hashlib.sha256(b).hexdigest()
def encode(x):return (json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode('utf8')
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]


def sampling_weights(labels):
    ids=[x['sample_id'] for x in labels]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate label ID')
    scores=[float(x['gold_raw']) for x in labels]
    if not scores or any(not math.isfinite(x) for x in scores):raise ValueError('Invalid scores')
    counts=Counter(scores); classes=len(counts)
    return [{'sample_id':r['sample_id'],'gold_raw':s,'score_count':counts[s],
             'unnormalized_weight':1/counts[s],'probability':1/(counts[s]*classes)} for r,s in zip(labels,scores)]


def weighted_draw(weights, count, seed):
    if count<1:raise ValueError('Positive draw count required')
    # With-replacement draws are an explicit implementation choice.
    rng=random.Random(seed)
    return rng.choices([r['sample_id'] for r in weights],weights=[r['probability'] for r in weights],k=count)


def build_index(graph):
    kinds={n['id']:n['kind'] for n in graph['nodes']}
    if len(kinds)!=len(graph['nodes']):raise ValueError('Duplicate node ID')
    groups={k:defaultdict(set) for k in ['types_by_summary','summaries_by_type','soft_neighbors','phrases_by_summary','soft_incoming']}
    expected={'Er':('w','c'),'Ew':('t','w'),'Es':('w','w')}
    if graph.get('soft_edge_mode')!='prose':raise ValueError('Expected summary co-occurrence graph')
    for edge in graph['edges']:
        k,a,b=edge['kind'],edge['source'],edge['target']
        if k not in expected or (kinds.get(a),kinds.get(b))!=expected[k]:raise ValueError('Graph edge type mismatch')
        if k=='Er':groups['types_by_summary'][a].add(b);groups['summaries_by_type'][b].add(a)
        elif k=='Ew':groups['phrases_by_summary'][b].add(a)
        else:groups['soft_neighbors'][a].add(b);groups['soft_incoming'][b].add(a)
    return {name:{k:sorted(v) for k,v in sorted(group.items())} for name,group in groups.items()}


def query(index,w,c):
    if c not in index['types_by_summary'].get(w,[]):raise ValueError('No specified Er(w,c)')
    tr=set(index['summaries_by_type'][c])
    ts=set(index['soft_neighbors'].get(w,[]))
    overlap=set()
    for wx in ts:overlap.update(index['soft_incoming'].get(wx,[]))
    tp=tr & overlap
    return {'Tr':sorted(tr),'Ts':sorted(ts),'Tp':sorted(tp),'T':sorted(tr|tp)}


def save(folder,files):
    for name,data in files.items():
        if (folder/name).exists() and (folder/name).read_bytes()!=data:raise ValueError('Existing differing artifact '+name)
    folder.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():
        if not (folder/name).exists():
            with (folder/name).open('xb') as f:f.write(data)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve()
    base=root/'paper_method_v2';graphdir=base/'artifacts/full_graph_v1'
    data=root/'data/prepared/ads_paper_split_v1_seed42'
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf8'))
    for name in ['train.inputs.jsonl','train.labels.jsonl']:
        if sha((data/name).read_bytes())!=manifest['outputs_sha256'][name]:raise ValueError('Changed training file')
    graph=json.loads((graphdir/'graph.json').read_text(encoding='utf8'))
    if sha((graphdir/'extractions.jsonl').read_bytes())!=graph['extractions_sha256']:raise ValueError('Graph/extraction mismatch')
    index=build_index(graph)
    inputs=rows(data/'train.inputs.jsonl');labels=rows(data/'train.labels.jsonl')
    lookup={r['sample_id']:r for r in inputs}
    if len(lookup)!=1107 or len(inputs)!=1107:raise ValueError('Expected frozen 1107 training IDs')
    if {r['sample_id'] for r in labels}!=set(lookup):raise ValueError('Label ID mismatch')
    weights=sampling_weights(labels);extractions={r['unit_id']:r for r in rows(graphdir/'extractions.jsonl')}
    if len(extractions)!=1125:raise ValueError('Expected 1125 completed extraction units')
    quads=[];preview=[]
    for row in labels:
        inp=lookup[row['sample_id']];uid='student:'+row['sample_id']
        if uid not in extractions:raise ValueError('Missing student extraction')
        if row['question_id']!=inp['question_id']:raise ValueError('Question ID mismatch')
        if not 0<=float(row['gold_raw'])<=float(inp['score_max']):raise ValueError('Score outside range')
        quads.append({'sample_id':row['sample_id'],'question_id':inp['question_id'],'q':inp['question'],
                      'a':inp['answer'],'r':inp['references'],'l':row['gold_raw'],'extraction_unit_id':uid})
        for triple in extractions[uid]['triples']:
            w='w:'+sha(triple['w'].encode('utf8'));c='c:'+sha(triple['c'].encode('utf8'))
            if len(preview)<5:
                result=query(index,w,c)
                preview.append({'sample_id':row['sample_id'],'phrase':triple['t'],'summary':triple['w'],
                                'entity_type':triple['c'],'counts':{k:len(v) for k,v in result.items()},
                                'T_equals_Tr':result['T']==result['Tr'],'self_in_T':w in result['T']})
    totals=defaultdict(float)
    for r in weights:totals[str(r['gold_raw'])]+=r['probability']
    assert math.isclose(sum(totals.values()),1.0,abs_tol=1e-12)
    results={'index.json':encode(index),'sampling_weights.json':encode(weights),
             'training_quadruples.json':encode(quads),'retrieval_preview.json':encode(preview)}
    report={'status':'PAPER_EQ5_TO_EQ9_READY','training_records':len(inputs),'extraction_units':len(extractions),
            'summary_nodes':sum(n['kind']=='w' for n in graph['nodes']),
            'score_classes':len(totals),'probability_sum':sum(totals.values()),'probability_mass_by_score':dict(totals),
            'graph_sha256':sha((graphdir/'graph.json').read_bytes()),'script_sha256':sha(Path(__file__).read_bytes()),
            'train_inputs_sha256':sha((data/'train.inputs.jsonl').read_bytes()),
            'train_labels_sha256':sha((data/'train.labels.jsonl').read_bytes()),
            'output_sha256':{k:sha(v) for k,v in results.items()},
            'implementation_notes':['Eq5 is normalized over training records.',
                'Eq8 defines Tp as subset of Tr; Eq9 therefore yields T=Tr. No extra filtering is introduced.',
                'Tr query takes the type of the current triplet, rather than merging all types of a shared summary.',
                'Self is retained because equations 6-9 do not explicitly exclude it.',
                'Reference alternatives remain a list; no unreported choice of reference is made.',
                'Weighted sampling helper uses replacement; no synthesis draw budget has been chosen or executed.'],
            'embedding_ranking_complete':False,'synthetic_answers_generated':0,'api_calls':0,
            'training_started':False,'validation_accessed':False,'test_accessed':False}
    results['summary.json']=encode(report)
    save(base/'artifacts/retrieval_v1',results)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
