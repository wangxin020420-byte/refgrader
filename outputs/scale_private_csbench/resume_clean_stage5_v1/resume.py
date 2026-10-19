import argparse
import hashlib
import importlib.util
from pathlib import Path
import json

OLD='csbench_scale_online_v1'
NEW='csbench_scale_online_clean_v2'
RUNNER_SHA='9b0cc2ec7992baeaf31723d80dc3253210ed79b248bd3e2f9afbbbb863cbf84b'
KEYS=('model','base_url','max_tokens','stream','contract_sha256','runner_sha256')

def same_request(a,b,oldcfg,newcfg):
    return all(oldcfg[k]==newcfg[k] for k in KEYS) and a==b

def install(r,m,old,out,requests,cfg):
    oc=r.read(old/'config.json');os=r.read(old/'summary.json');op=r.read(old/'plan_summary.json')
    oh=r.sha(r.enc(oc));nh=r.sha(r.enc(cfg))
    assert os['status']=='CSBENCH_PROXY_COMPLETE_REVIEW_REQUIRED' and os['remaining']==0 and os['config_sha256']==oh
    for name in ('config.json','requests.jsonl','training_sources.json','blank_sources.json'):
        assert r.sha((old/name).read_bytes())==op['files_sha256'][name], 'Old plan hash mismatch: '+name
    assert r.sha((old/'proxy_records.jsonl').read_bytes())==os['files_sha256']['proxy_records.jsonl']
    oldrequests=r.rows(old/'requests.jsonl');lookup={x['request_id']:x for x in oldrequests}
    assert len(lookup)==len(oldrequests) and len({x['request_id'] for x in requests})==len(requests)
    writes={};reused=[];pending=[]
    for req in requests:
        rid=req['request_id'];assert Path(rid).name==rid and rid.startswith('proxy_')
        prior=lookup.get(rid)
        if prior is None or not same_request(prior,req,oc,cfg):
            pending.append(rid);continue
        folder=old/'responses'/rid;ap=folder/'accepted.json'
        if not ap.exists():pending.append(rid);continue
        accepted=r.read(ap);name=accepted['response_file'];assert Path(name).name==name
        assert accepted['request_id']==rid and accepted['config_sha256']==oh
        rawpath=folder/name;rawhash=r.sha(rawpath.read_bytes());assert rawhash==accepted['response_sha256']
        saved=r.read(rawpath);assert saved['config_sha256']==oh and saved['request_id']==rid
        m.parse_response(saved['response'],req)
        provenance=dict(source_experiment=OLD,source_config_sha256=oh,source_response_file=str(rawpath),
            source_response_sha256=rawhash,request_sha256=r.sha(r.enc(req)),api_call_performed=False)
        envelope=dict(config_sha256=nh,request_id=rid,response=saved['response'],source_kind='exact_request_response_reuse',reuse_provenance=provenance)
        blob=r.enc(envelope);target=out/'responses'/rid;name='exact_reuse.response.json'
        writes[target/name]=blob
        writes[target/'accepted.json']=r.enc(dict(config_sha256=nh,request_id=rid,response_file=name,response_sha256=r.sha(blob),source_kind='exact_request_response_reuse'))
        reused.append(dict(request_id=rid,**provenance))
    report=dict(status='CLEAN_STAGE5_REUSE_PLAN_READY',experiment_id=NEW,planned_requests=len(requests),
        generated_proxy_slots=len(requests),reused_requests=len(reused),requests_not_reused=len(pending),
        api_calls=0,assigned_labels=0,source_files_modified=False,semantic_accuracy_verified=False,
        match_policy='Full request equality, including messages and provenance, plus model/endpoint/decoding/contract equality',
        reused=reused,not_reused_request_ids=pending)
    writes[out/'reuse_audit.json']=r.enc(report)
    for p,b in writes.items():
        if p.exists() and p.read_bytes()!=b:raise ValueError('Existing artifact differs; not overwritten: '+str(p))
    for p,b in writes.items():r.freeze(p,b)
    print(json.dumps({k:v for k,v in report.items() if k not in ('reused','not_reused_request_ids')},ensure_ascii=False,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--project',type=Path,required=True);a=p.parse_args()
    root=a.project.resolve();path=root/'control/scale_csbench_stage5_v1/run_stage5.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==RUNNER_SHA,'Existing Stage5 runner changed'
    spec=importlib.util.spec_from_file_location('stage5',path);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    out=root/'runs/online_experiments'/NEW/'scale_stage5';out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m,out,requests,cfg=r.prepare(root,NEW)
        install(r,m,root/'runs/online_experiments'/OLD/'scale_stage5',out,requests,cfg)
        if a.command=='run':
            rc=r.execute(root,NEW,attempts=8)
            if rc==0:
                summary=r.read(out/'summary.json')
                assert summary['proxy_records']==345 and summary['blank_passthrough_records']==5
                policy=dict(status='PROXY_AUTOMATIC_OUTPUT_ACCEPTED',experiment_id=NEW,
                    proxy_records_sha256=summary['files_sha256']['proxy_records.jsonl'],
                    records=345,human_review_performed=False,semantic_accuracy_verified=False,
                    manual_review_required=False,automatic_semantic_errors_retained=True,
                    ready_for_annotator_input_preparation=True,ready_for_annotator_training=False,
                    label_transfer_policy='Use source scores in subsequent annotator-input preparation under the unverified semantic-equivalence assumption; no labels changed here',
                    synthetic_candidate_labels_assigned=0,training_started=False)
                r.freeze(out/'automatic_acceptance.json',r.enc(policy))
                print(json.dumps(policy,ensure_ascii=False,indent=2))
            raise SystemExit(rc)

if __name__=='__main__':main()
