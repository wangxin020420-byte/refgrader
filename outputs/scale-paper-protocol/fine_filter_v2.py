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
    if len(source)!=1107 or len(sources)!=1107 or len(original)!=5000 or len(candidates)!=5000:raise ValueError('Input coverage mismatch')
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

def build(root):
    base=root/'paper_method_v2';run=base/'runs/annotator_bert_valqwk_v1'
    summary=read(run/'summary.json');cfg=read(run/'config.json');ch=sha(run/'config.json')
    if summary['status']!='FIVE_ANNOTATORS_AND_CANDIDATE_SCORING_COMPLETE' or summary['seeds']!=SEEDS or summary['completed_runs']!=5:raise ValueError('Annotator runs incomplete')
    sourcepath=base/'artifacts/proxy_inputs_v1/training_sources.json'
    candidatepath=base/'artifacts/initial_filter_k5_b5000_v1/selected_candidates.jsonl'
    if sha(sourcepath)!=cfg['input_hashes']['sources'] or sha(candidatepath)!=SELECTED_SHA or cfg['input_hashes']['candidates']!=SELECTED_SHA:raise ValueError('Input hash mismatch')
    sources=read(sourcepath);candidates=rows(candidatepath)
    out=base/'artifacts/fine_filter_raw_beta0p2_v1';files={};reports={};texts={}
    policy={'beta':0.2,'score_scale':'ADS raw 0-10','rule':'abs(source_score - assigned_label) <= 0.2',
        'numeric_policy':'Python float comparison; no epsilon, clipping, rounding or normalization',
        'beta_scale_verified_from_author':False,'reproduction_status':'METHOD_RECONSTRUCTION_BETA_SCALE_UNVERIFIED',
        'implementation_note':'Interpret paper beta=0.2 on the current raw score scale; not author-verified. No scale selection using validation or test metrics.',
        'seeds':SEEDS,'annotator_config_sha256':ch,'source_sha256':sha(sourcepath),'candidates_sha256':SELECTED_SHA,
        'builder_sha256':sha(Path(__file__)),'api_calls':0,'test_accessed':False,'validation_accessed_this_stage':False}
    files[out/'policy.json']=enc(policy)
    for seed in SEEDS:
        sr=run/('seed_%d'%seed);complete=read(sr/'complete.json');scorepath=sr/'scored_candidates.jsonl'
        if complete['status']!='ANNOTATOR_AND_SCORING_COMPLETE' or complete['seed']!=seed or complete['config_sha256']!=ch or complete['training_records']!=1107 or complete['candidate_labels_assigned']!=5000 or not complete['weights_reload_verified']:raise ValueError('Incompatible annotator run')
        if sha(scorepath)!=complete['files_sha256']['scored_candidates.jsonl']:raise ValueError('Scoring file changed')
        kept,augmented,audit=select(rows(scorepath),sources,candidates,seed)
        payload={'selected.jsonl':jl(kept),'augmented_inputs.jsonl':jl(augmented),'selection_audit.jsonl':jl(audit)}
        report={'status':'EQ14_FINE_FILTER_COMPLETE','seed':seed,'beta':0.2,'score_scale':'ADS raw 0-10',
            'original_training_records':1107,'scored_candidates':5000,'retained_synthetic':len(kept),
            'discarded_synthetic':5000-len(kept),'augmented_records':len(augmented),
            'retained_out_of_range':sum(not 0<=r['assigned_label']<=10 for r in kept),
            'retained_by_source_score':dict(Counter(str(r['source_score']) for r in kept)),
            'scored_candidates_sha256':sha(scorepath),'annotator_complete_sha256':sha(sr/'complete.json'),
            'policy_sha256':digest(enc(policy)),'files_sha256':{n:digest(b) for n,b in payload.items()},
            'beta_scale_verified_from_author':False,'ready_for_sae_training':False,'api_calls':0,'training_started':False,
            'test_accessed':False,'validation_accessed_this_stage':False}
        dest=out/('seed_%d'%seed)
        for name,data in payload.items():files[dest/name]=data
        files[dest/'manifest.json']=enc(report);reports[str(seed)]=report
        for r in augmented:
            text=r['answer'];key=digest(text.encode('utf8'))
            item=texts.setdefault(key,{'text_sha256':key,'text':text,'occurrences':[]})
            if item['text']!=text:raise ValueError('Hash collision')
            item['occurrences'].append({'seed':seed,'id':r['id'],'sample_id':r['sample_id'],'kind':r['kind']})
    inventory=[texts[k] for k in sorted(texts)]
    files[out/'answer_text_inventory.jsonl']=jl(inventory)
    report={'status':'FIVE_RUN_EQ14_FINE_FILTER_COMPLETE','seeds':SEEDS,'beta':0.2,'score_scale':'ADS raw 0-10',
        'beta_scale_verified_from_author':False,'runs':{s:{k:r[k] for k in ['retained_synthetic','discarded_synthetic','augmented_records','retained_out_of_range']} for s,r in reports.items()},
        'unique_answer_texts_across_runs':len(inventory),'inventory_scope':'Exact answer texts only, including originals. Not an API request plan; reference texts, context and configuration must also be checked for reuse.',
        'api_calls':0,'training_started':False,'test_accessed':False,'validation_accessed_this_stage':False,
        'next_step':'Prepare context/config-matched entity extraction requests for retained training answers and references; preserve per-seed labels.',
        'files_sha256':{str(p.relative_to(out)):digest(b) for p,b in files.items()}}
    files[out/'summary.json']=enc(report)
    return out,files,report

def execute(root,dry_run):
    out,files,report=build(root)
    for path,data in files.items():
        if path.exists() and path.read_bytes()!=data:raise ValueError('Existing output differs; not overwritten: '+str(path))
    if not dry_run:
        for path,data in files.items():
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(data);tmp.replace(path)
            if sha(path)!=digest(data):raise ValueError('Write verification failed')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('FINE_FILTER_PREFLIGHT_OK' if dry_run else 'FIVE_RUN_FINE_FILTER_DONE',flush=True)
    print('OUTPUT:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    root=a.root.resolve()
    if a.dry_run:execute(root,True)
    else:
        import fcntl
        lockdir=root/'paper_method_v2/artifacts';lockdir.mkdir(parents=True,exist_ok=True)
        with (lockdir/'fine_filter_raw_beta0p2.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(root,False)
