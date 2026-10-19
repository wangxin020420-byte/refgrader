import argparse
import hashlib
import json
import math
from pathlib import Path

EXP='csbench_scale_online_clean_v2'
PROXY='46e7fbceb09cc7c161b135338d08fca64aa2c312c66baba8b9a091333657d9c4'
SELECTED='2f36de2d49944d053583b808b901e5e5f39eadf100c28fb90cc3cd6f49b31c35'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def enc(o):return (json.dumps(o,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def jl(rr):return ''.join(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n' for r in rr).encode()
def index(rr):
    d={r['sample_id']:r for r in rr};assert len(d)==len(rr),'Duplicate ID';return d
def score(g):
    vals=[g[k] for k in ('score_original','score_max_original','score_0_10')]
    assert all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in vals)
    raw,maximum,value=vals
    assert maximum>0 and 0<=raw<=maximum and math.isclose(value,10*raw/maximum,abs_tol=1e-10)
    return value
def labeled(row,g,answer,origin):
    assert row['question_id']==g['question_id'] and isinstance(answer,str)
    assert isinstance(row['references'],list) and all(isinstance(x,str) for x in row['references'])
    return dict(sample_id=row['sample_id'],question_id=row['question_id'],question=row['question'],
        references=row['references'],answer=answer,label=score(g),label_origin=origin,
        score_original=g['score_original'],score_max_original=g['score_max_original'],
        blank_passthrough=row['blank_passthrough'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);a=ap.parse_args()
    root=a.project.resolve();exp=root/'runs/online_experiments'/EXP;stage5=exp/'scale_stage5'
    policy=read(stage5/'automatic_acceptance.json');sm=read(stage5/'summary.json')
    assert policy['status']=='PROXY_AUTOMATIC_OUTPUT_ACCEPTED' and policy['experiment_id']==EXP
    assert policy['human_review_performed'] is False and policy['automatic_semantic_errors_retained'] is True
    assert sm['remaining']==0 and sm['errors']==[] and sm['proxy_records']==345
    pp=stage5/'proxy_records.jsonl';assert sha(pp)==PROXY==policy['proxy_records_sha256']==sm['files_sha256'][pp.name]
    proxies=index(rows(pp));assert len(proxies)==345
    visual=exp/'visual';vs=read(visual/'experiment_summary.json');assert vs['inputs_complete'] and not vs['pending'] and vs['experiment_id']==EXP
    frozen=root/'data/csbench_within_question_v1';fm=read(frozen/'manifest.json')
    assert fm['counts']=={'train':345,'validation':343,'test':2723}
    files={};provenance={str(pp):sha(pp),str(stage5/'automatic_acceptance.json'):sha(stage5/'automatic_acceptance.json')};splitids={};lookup={}
    for split,n in [('train',345),('validation',343)]:
        ip=visual/'text_view'/f'{split}.inputs.jsonl';gp=frozen/f'{split}.labels.jsonl';op=frozen/f'{split}.inputs.jsonl'
        assert sha(ip)==vs['files_sha256'][ip.name]
        for path in (gp,op):assert sha(path)==fm['files_sha256'][path.name]
        rr=rows(ip);gg=index(rows(gp));original=rows(op)
        assert len(rr)==len(gg)==n and len(index(rr))==n
        assert [r['sample_id'] for r in rr]==[r['sample_id'] for r in original] and set(gg)==set(index(rr))
        splitids[split]=set(gg);result=[]
        for r,o in zip(rr,original):
            sid=r['sample_id'];assert r['question_id']==o['question_id'] and r['blank_passthrough']==o['blank_passthrough']
            if o['blank_passthrough']:assert r['answer']==o['answer']
            if split=='train':
                p=proxies[sid]
                assert p['question_id']==r['question_id'] and p['original_answer']==r['answer'] and p['references']==r['references']
                assert p['source_score']==score(gg[sid]) and p['assigned_label'] is None
                if r['blank_passthrough']:assert p['proxy_answer']==r['answer']
                result.append(labeled(r,gg[sid],p['proxy_answer'],'source_score_transfer_unverified_semantic_equivalence'))
                lookup[sid]=r
            else:result.append(labeled(r,gg[sid],r['answer'],'existing_validation_gold'))
        assert sum(r['blank_passthrough'] for r in result)=={'train':5,'validation':8}[split]
        files[split+'.jsonl']=jl(result)
        for path in (ip,gp,op):provenance[str(path)]=sha(path)
    assert not splitids['train']&splitids['validation']
    selection=exp/'scale_stage4/layout/paper_method_v2/artifacts/initial_filter_k5_b5000_v1'
    ss=read(exp/'scale_stage4/summary.json');cp=selection/'selected_candidates.jsonl'
    assert ss['status']=='INITIAL_FILTER_COMPLETE_UNLABELED' and sha(cp)==SELECTED==ss['files_sha256'][cp.name]
    cc=rows(cp);assert len(cc)==len({r['variant_id'] for r in cc})==5000
    jobs=[]
    for c in cc:
        sid=c['sample_id'];assert sid in splitids['train'] and c.get('assigned_label') is None
        assert isinstance(c['answer'],str) and c['answer'].strip()
        jobs.append(dict(c,references=lookup[sid]['references'],source_score=proxies[sid]['source_score']))
    files['candidate_inputs.jsonl']=jl(jobs);provenance[str(cp)]=sha(cp)
    report=dict(status='CSBENCH_ANNOTATOR_INPUTS_READY_CONFIG_PENDING',experiment_id=EXP,
        training_records=345,validation_records=343,candidate_records=5000,training_labels_transferred=345,
        synthetic_candidate_labels_assigned=0,blank_train_records=5,blank_validation_records=8,
        human_review_performed=False,semantic_accuracy_verified=False,automatic_semantic_errors_retained=True,
        score_scale='Existing score_0_10 = 10 * score_original / score_max_original',
        api_calls=0,training_started=False,validation_labels_read=True,test_labels_read=False,
        ready_for_annotator_training=False,source_files_modified=False,
        source_sha256=provenance,files_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
    files['manifest.json']=enc(report);out=exp/'annotator_inputs_v1'
    for name,b in files.items():
        p=out/name
        if p.exists():assert p.read_bytes()==b,'Existing output differs: '+str(p)
    out.mkdir(parents=True,exist_ok=True)
    for name,b in files.items():
        p=out/name
        if not p.exists():
            with p.open('xb') as f:f.write(b)
    for path,h in provenance.items():assert sha(Path(path))==h
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
    print('OUTPUT:',out)
if __name__=='__main__':main()
