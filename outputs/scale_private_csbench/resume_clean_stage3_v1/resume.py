import argparse
import hashlib
import importlib.util
from pathlib import Path
import json

OLD='csbench_scale_online_v1'
NEW='csbench_scale_online_clean_v2'
RUNNER_SHA='5b8df9cdd8f40fb28eab1cc509569eca0bb64c666db99eb30f4451933b229eae'
KEYS=('model','base_url','max_tokens','stream','variants_per_draw','contract_sha256','runner_sha256')

def same_request(a,b,oldcfg,newcfg):
    return all(oldcfg[k]==newcfg[k] for k in KEYS) and a==b

def install(r,m,old,out,requests,cfg):
    oc=r.read(old/'config.json');os=r.read(old/'summary.json');op=r.read(old/'plan_summary.json')
    oh=r.sha(r.enc(oc));nh=r.sha(r.enc(cfg))
    assert os['status']=='CSBENCH_PS_GENERATION_COMPLETE_UNLABELED' and os['remaining']==0 and os['config_sha256']==oh
    for name in ('config.json','requests.jsonl','rule_bank.json','skipped_draws.json'):
        assert r.sha((old/name).read_bytes())==op['files_sha256'][name], 'Old plan hash mismatch: '+name
    assert r.sha((old/'candidates.jsonl').read_bytes())==os['files_sha256']['candidates.jsonl']
    oldrequests=r.rows(old/'requests.jsonl');lookup={x['request_id']:x for x in oldrequests}
    assert len(lookup)==len(oldrequests) and len({x['request_id'] for x in requests})==len(requests)
    writes={};reused=[];pending=[]
    for req in requests:
        rid=req['request_id'];assert Path(rid).name==rid and rid.startswith('draw_')
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
    report=dict(status='CLEAN_STAGE3_REUSE_PLAN_READY',experiment_id=NEW,planned_requests=len(requests),
        candidate_slots=5*len(requests),reused_requests=len(reused),requests_not_reused=len(pending),
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
    root=a.project.resolve();path=root/'control/scale_csbench_stage3_v1/run_stage3.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==RUNNER_SHA,'Existing Stage3 runner changed'
    spec=importlib.util.spec_from_file_location('stage3',path);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    out=root/'runs/online_experiments'/NEW/'scale_stage3';out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m,out,requests,cfg=r.prepare(root,NEW,1200,42)
        install(r,m,root/'runs/online_experiments'/OLD/'scale_stage3',out,requests,cfg)
        if a.command=='run':raise SystemExit(r.execute(root,NEW,draws=1200,seed=42,attempts=8))

if __name__=='__main__':main()
