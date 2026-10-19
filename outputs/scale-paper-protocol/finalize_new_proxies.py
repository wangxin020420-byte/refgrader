"""Freeze reviewed v2 proxies and prepare 1107 annotator inputs; offline only."""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

SOURCE_SHA='ab2e840c049f6e1e8cfa818245983944c59985a0e60e415522a1d4f45a5f4d9a'
TARGETS={'166','313','390','589','1070','1089'}

def enc(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf8')
def jl(xs):return ''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in xs).encode('utf8')
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(p.read_bytes())
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]
def th(s):return digest(s.replace('\r\n','\n').strip().encode('utf8'))

def merge(records,first,second,retries,sources):
    lookup={r['sample_id']:r for r in records};src={s['sample_id']:s for s in sources}
    if len(records)!=1107 or len(lookup)!=1107 or len(sources)!=1107 or set(src)!=set(lookup):
        raise ValueError('Training coverage mismatch')
    new={sid for sid,r in lookup.items() if r.get('v2_origin')=='new_generation'}
    old={sid for sid,r in lookup.items() if r.get('v2_origin')=='frozen_v1_alignment'}
    ds={d['sample_id']:d for d in first['decisions']};rs={r['sample_id']:r for r in retries}
    ss={d['sample_id']:d for d in second['decisions']}
    if len(first['decisions'])!=222 or set(ds)!=new or len(new)!=222 or len(old)!=885:
        raise ValueError('Initial review coverage mismatch')
    held={sid for sid,d in ds.items() if d['decision']!='PASS'}
    if {sid.rsplit(':',1)[1] for sid in held}!=TARGETS or set(rs)!=held or set(ss)!=held or len(retries)!=6 or len(second['decisions'])!=6:
        raise ValueError('Retry coverage mismatch')
    for sid,d in ds.items():
        r=lookup[sid]
        for key,field in [('question_sha256','question'),('original_sha256','original_answer'),('proxy_sha256','proxy_answer')]:
            if th(r[field])!=d[key]:raise ValueError('Initial reviewed text mismatch: '+sid)
    result=[];training=[]
    for r in records:
        sid=r['sample_id'];s=src[sid];out=copy.deepcopy(r)
        for field,key in [('original_answer','a'),('source_score','l'),('question','q'),('question_id','question_id'),('references','r')]:
            if r[field]!=s[key]:raise ValueError('Source provenance differs: '+sid+' '+field)
        if r.get('assigned_label') is not None:raise ValueError('Unexpected proxy label')
        if sid in held:
            retry=rs[sid];d=ss[sid]
            if d['decision']!='PASS':raise ValueError('Retry not approved')
            for key,field in [('original_sha256','original_answer'),('previous_proxy_sha256','previous_proxy_answer'),('retry_proxy_sha256','retry_proxy_answer')]:
                if th(retry[field])!=d[key]:raise ValueError('Retry reviewed text mismatch: '+sid)
            if retry['original_answer']!=r['original_answer'] or retry['previous_proxy_answer']!=r['proxy_answer'] or retry['question']!=r['question']:
                raise ValueError('Retry lineage mismatch: '+sid)
            if not isinstance(retry['retry_proxy_answer'],str) or not retry['retry_proxy_answer'].strip():raise ValueError('Empty retry')
            oldprov={k:out.pop(k) for k in ('request_id','raw_response_sha256','outer_fence_removed','returned_model','response_id') if k in out}
            out.update(previous_generation_provenance=oldprov,previous_proxy_answer=r['proxy_answer'],
                proxy_answer=retry['retry_proxy_answer'],kind='generated_proxy_retry',alignment_action='use_reviewed_retry',
                alignment_review_status='PASS',semantic_equivalence_verified=True,
                semantic_review_basis='Assistant comparison of submitted six-pair retry review; not human certification',
                retry_raw_response_sha256=retry['raw_response_sha256'],retry_returned_model=retry.get('returned_model'))
        elif sid in new:
            out.update(alignment_action='keep_original_proxy',alignment_review_status='PASS',semantic_equivalence_verified=True,
                semantic_review_basis='Assistant comparison of submitted 222-pair review; not human certification')
        else:
            if out!=r:raise ValueError('Reused record modified')
        score=out['source_score']
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=10:raise ValueError('Invalid score')
        if not isinstance(out['references'],list) or not all(isinstance(x,str) for x in out['references']) or not any(x.strip() for x in out['references']):raise ValueError('Missing reference')
        passthrough=out['alignment_action'] in ('blank_passthrough','original_answer_fallback')
        if passthrough and out['proxy_answer']!=s['a']:raise ValueError('Passthrough changed')
        training.append({'sample_id':sid,'question_id':out['question_id'],'question':out['question'],
            'references':out['references'],'answer':out['proxy_answer'],'label':float(score),'source_score':score,
            'label_origin':'original_training_score','label_transfer_basis':'exact_original_answer' if passthrough else 'proxy_semantic_equivalence_assumption',
            'alignment_action':out['alignment_action'],'alignment_review_status':out['alignment_review_status'],
            'semantic_equivalence_verified':out['semantic_equivalence_verified']})
        result.append(out)
    counts=Counter(r['alignment_action'] for r in result)
    if dict(counts)!={'keep_original_proxy':1084,'use_reviewed_retry':11,'original_answer_fallback':7,'blank_passthrough':5}:raise ValueError('Alignment counts differ: '+str(counts))
    if sum(r['alignment_review_status']=='NOT_INDIVIDUALLY_REVIEWED' for r in result)!=636:raise ValueError('Unreviewed status changed')
    return result,training,dict(counts)

def execute(root,review_path,dry_run):
    base=root/'paper_method_v2';folder=base/'artifacts/proxy_review_inputs_v1';source=folder/'proxy_records.jsonl'
    if sha(source)!=SOURCE_SHA or read(folder/'manifest.json')['records_sha256']!=SOURCE_SHA:raise ValueError('Proxy input hash mismatch')
    retrydir=base/'runs/proxy_glm_v1/new_semantic_retry_v1';retryfile=retrydir/'candidates.json'
    summary=read(retrydir/'summary.json');policy=read(retrydir/'policy.json')
    if summary['status']!='SIX_PROXY_RETRIES_READY_FOR_REVIEW' or summary['valid_retry_responses']!=6 or summary['errors']:raise ValueError('Retry incomplete')
    if sha(retryfile)!=summary['candidates_sha256']:raise ValueError('Retry candidates changed')
    first_path=retrydir/'review_decisions.json';first=read(first_path);second=read(review_path)
    original_review=root/'control/scale-paper-protocol/new_proxy_review_v1/decisions.json'
    if policy['source_records_sha256']!=SOURCE_SHA or sha(original_review)!=policy['review_sha256'] or first!=read(original_review) or first['source_records_sha256']!=SOURCE_SHA:raise ValueError('Review lineage mismatch')
    plan=base/'artifacts/proxy_inputs_v1';meta=read(plan/'summary.json')
    sourcesfile=plan/'training_sources.json'
    if sha(sourcesfile)!=meta['files_sha256']['training_sources.json']:raise ValueError('Training sources changed')
    aligned,training,counts=merge(rows(source),first,second,read(retryfile),read(sourcesfile))
    data=jl(aligned);train=jl(training)
    common={'training_records':1107,'api_calls':0,'training_started':False,'validation_accessed':False,'test_accessed':False,
        'unreviewed_generated_records':636,'original_fallback_records':7,
        'builder_sha256':sha(Path(__file__)),'training_sources_sha256':sha(sourcesfile),
        'aligned_records_sha256':digest(data),'reproduction_status':'METHOD_RECONSTRUCTION_WITH_DOCUMENTED_DEVIATIONS'}
    alignment=dict(common,status='PROXY_ALIGNMENT_FROZEN_UNLABELED',counts=counts,labels_assigned=0,
        source_records_sha256=SOURCE_SHA,initial_review_sha256=sha(first_path),retry_candidates_sha256=sha(retryfile),
        retry_review_sha256=sha(review_path),new_reviewed_pairs=222,new_accepted_retry_records=6,
        deviation='Retain seven previously documented original-answer fallbacks; 636 reused generated proxies remain not individually reviewed.')
    inputs=dict(common,status='ANNOTATOR_DATA_READY_CONFIG_PENDING',annotator_labels_assigned=1107,
        synthetic_candidate_labels_assigned=0,train_jsonl_sha256=digest(train),
        score_scale='ADS original 0-10; no clipping, normalization or rounding',
        score_distribution=dict(Counter(str(r['label']) for r in training)),
        reference_policy='All reference slots retained; no concatenation or row expansion yet.',
        label_assumption='Original training score transferred to proxy under semantic-preservation assumption; individual review incomplete for 636 reused records.',
        ready_for_annotator_training=False,
        next_step='Prepare v2 annotator training configuration and runner for 1107 rows; do not reuse the old 885-row trained annotator.')
    out=base/'artifacts/proxy_aligned_v1';ann=base/'artifacts/annotator_inputs_v1'
    files={out/'proxy_records.jsonl':data,out/'manifest.json':enc(alignment),out/'new_retry_review.json':enc(second),
        ann/'train.jsonl':train,ann/'manifest.json':enc(inputs)}
    # Check all outputs before any write; exact reruns are idempotent.
    for path,content in files.items():
        if path.exists() and path.read_bytes()!=content:raise ValueError('Existing output differs: '+str(path))
    if dry_run:
        print('FINALIZE_PROXY_PREFLIGHT_OK: 1107 records; 216 accepted new + 6 reviewed retries; 636 old unreviewed preserved')
        return
    for path,content in files.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(content);tmp.replace(path)
        if sha(path)!=digest(content):raise ValueError('Output verification failed')
    print(json.dumps(alignment,ensure_ascii=False,indent=2));print(json.dumps(inputs,ensure_ascii=False,indent=2))
    print('ANNOTATOR_INPUTS:',ann/'train.jsonl');print('PROXY_ALIGNMENT_AND_ANNOTATOR_INPUTS_DONE')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--review',type=Path,default=Path(__file__).with_name('six_retry_decisions.json'))
    parser.add_argument('--accept-proxy-label-assumption',action='store_true',required=True)
    parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    root=args.root.resolve()
    if args.dry_run:execute(root,args.review.resolve(),True)
    else:
        import fcntl
        lockdir=root/'paper_method_v2/artifacts';lockdir.mkdir(parents=True,exist_ok=True)
        with (lockdir/'proxy_finalize.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(root,args.review.resolve(),False)
