import math
from pathlib import Path
import bert_helpers as h

EXP='csbench_scale_online_clean_v2'
EXPECTED={'train.jsonl':'a4d837b033334b0983cff8f086f128526dcbf3708bca831b37b384da689a18fe',
    'validation.jsonl':'3326a4e2c14c6863cb60e4ea00c77f7edaa2b737fb474583fefe6fc216e7baeb',
    'candidate_inputs.jsonl':'b2f9c8b3c1c729b80691ccb95625af6d9bfaa747aef6ba4cfa2c166aba2fd4d0'}

def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)

def validate(train,val,jobs):
    ids=[]
    for rows,n in [(train,345),(val,343)]:
        assert len(rows)==len({r['sample_id'] for r in rows})==n
        ids.append({r['sample_id'] for r in rows})
        for r in rows:
            assert isinstance(r['answer'],str) and finite(r['label']) and 0<=r['label']<=10
            assert finite(r['score_original']) and finite(r['score_max_original']) and r['score_max_original']>0
            assert 0<=r['score_original']<=r['score_max_original']
            assert math.isclose(r['label'],10*r['score_original']/r['score_max_original'],abs_tol=1e-10)
            h.reference_text(r)
    assert not ids[0]&ids[1]
    assert sum(r['blank_passthrough'] for r in train)==5 and sum(r['blank_passthrough'] for r in val)==8
    assert all(not r['answer'].strip() for r in train+val if r['blank_passthrough'])
    source={r['sample_id']:r for r in train}
    assert len(jobs)==len({r['variant_id'] for r in jobs})==5000
    for j in jobs:
        s=source[j['sample_id']]
        assert j.get('assigned_label') is None and isinstance(j['answer'],str) and j['answer'].strip()
        assert j['references']==s['references'] and j['source_score']==s['label']
        h.reference_text(j)

def inputs(root):
    import json
    ann=root/'runs/online_experiments'/EXP/'annotator_inputs_v1'
    meta=h.read(ann/'manifest.json')
    assert meta['experiment_id']==EXP and meta['status']=='CSBENCH_ANNOTATOR_INPUTS_READY_CONFIG_PENDING'
    assert meta['training_records']==345 and meta['validation_records']==343 and meta['candidate_records']==5000
    assert meta['automatic_semantic_errors_retained'] is True and meta['human_review_performed'] is False
    assert meta['files_sha256']==EXPECTED
    hashes={}
    for name,expected in EXPECTED.items():
        assert h.sha(ann/name)==expected,'Changed frozen input: '+name
        hashes[name]=expected
    for source,expected in meta['source_sha256'].items():
        p=Path(source).resolve()
        assert p.is_relative_to(root.resolve()) and 'test' not in p.name,'Unexpected source provenance path'
        assert h.sha(p)==expected,'Source provenance changed: '+str(p)
    def rows(name):return [json.loads(s) for s in (ann/name).read_text(encoding='utf-8').splitlines() if s.strip()]
    train,val,jobs=rows('train.jsonl'),rows('validation.jsonl'),rows('candidate_inputs.jsonl')
    validate(train,val,jobs)
    hashes.update(manifest=h.sha(ann/'manifest.json'),input_adapter=h.sha(Path(__file__)))
    return train,val,jobs,hashes
