"""Offline Eq14 raw-score beta=0.2 filtering, preserving separate seed datasets."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

SEEDS=[42,43,44,45,46]
SELECTED_SHA='03c8b8976105ee2bb518c2424606d038693bb06747b7eb500a243e6fe462655a'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def enc(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf8')
def jl(xs):return ''.join(json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n' for x in xs).encode('utf8')
def digest(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]

def select(scored,sources,candidates,seed):
    source={s['sample_id']:s for s in sources};original={c['variant_id']:c for c in candidates}
    if len(source)!=345 or len(sources)!=345 or len(original)!=5000 or len(candidates)!=5000:raise ValueError('Input coverage mismatch')
    if len(scored)!=5000 or len({c['variant_id'] for c in scored})!=5000 or {c['variant_id'] for c in scored}!=set(original):raise ValueError('Scored coverage mismatch')
    kept=[];audit=[]
    for c in scored:
        old=original[c['variant_id']];s=source[c['sample_id']]
        if c['sample_id']!=old['sample_id'] or c['answer']!=old['answer'] or old.get('assigned_label') is not None:raise ValueError('Candidate lineage mismatch')
        value=c['assigned_label']
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Invalid prediction')
        if c['annotator_seed']!=seed or c['source_score']!=s['l'] or c['label_origin']!='annotator_raw_regression':raise ValueError('Seed or score provenance mismatch')
        gap=abs(value-s['l'])
        if c['absolute_gap_raw']!=gap:raise ValueError('Stored gap inconsistent')
        retain=gap<=0.2
        audit.append({'variant_id':c['variant_id'],'sample_id':c['sample_id'],'source_score':s['l'],
            'prediction_raw':value,'absolute_gap_raw':gap,'retained':retain,'out_of_range':not 0<=value<=10})
        if retain:kept.append(c)
    augmented=[]
    for s in sources:
        if isinstance(s['l'],bool) or not isinstance(s['l'],(int,float)) or not math.isfinite(s['l']) or not 0<=s['l']<=10:raise ValueError('Invalid source label')
        augmented.append({'id':s['sample_id'],'sample_id':s['sample_id'],'question_id':s['question_id'],
            'question':s['q'],'references':s['r'],'answer':s['a'],'label':s['l'],'kind':'original_training','label_origin':'original_gold'})
    for c in kept:
        s=source[c['sample_id']]
        augmented.append({'id':c['variant_id'],'sample_id':c['sample_id'],'question_id':s['question_id'],
            'question':s['q'],'references':s['r'],'answer':c['answer'],'label':c['assigned_label'],
            'kind':'synthetic_candidate','label_origin':'annotator_raw_regression','annotator_seed':seed,'original_score':s['l']})
    if len({r['id'] for r in augmented})!=len(augmented):raise ValueError('Augmented ID collision')
    return kept,augmented,audit


