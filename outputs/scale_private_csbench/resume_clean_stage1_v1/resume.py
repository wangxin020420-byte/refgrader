"""Explicit validated-extraction reuse, not a claim of new API responses."""
import argparse
import importlib.util
import json
from pathlib import Path

OLD='csbench_scale_online_v1'
NEW='csbench_scale_online_clean_v2'
RUNNER_SHA='8b1aa375f6abec779012902d4462704d49782a5715b691826282a8c16d1ff7b4'
CHANGED={'student:ANS_ML_'+i for i in ('09','25','28','31','37','43')}
KEYS=('model','base_url','max_tokens','prompt_sha256','contract_sha256','runner_sha256','request_policy','parser_policy')

def eligible(oldcfg,newcfg,oldunits,newunits):
    for k in KEYS:
        if oldcfg[k]!=newcfg[k]:raise ValueError('Request contract changed: '+k)
    old={u['unit_id']:u for u in oldunits};new={u['unit_id']:u for u in newunits}
    assert len(old)==len(oldunits)==len(new)==len(newunits)==390 and old.keys()==new.keys()
    changed={uid for uid in new if old[uid]!=new[uid]}
    assert changed==CHANGED, 'Unexpected changed units: '+str(sorted(changed))
    return [u for u in newunits if u['unit_id'] not in changed and u['text'].strip()]

def install(runner,contract,old,out,units,cfg):
    oldcfg=runner.read(old/'config.json');oldsummary=runner.read(old/'summary.json')
    assert oldsummary['status']=='CSBENCH_TRAIN_EXTRACTION_AND_GRAPH_COMPLETE' and oldsummary['remaining']==0
    oldhash=runner.sha(runner.enc(oldcfg));newhash=runner.sha(runner.enc(cfg))
    assert oldsummary['config_sha256']==oldhash
    for name in ('source_units.jsonl','extractions.jsonl','training_sources.json'):
        assert runner.sha((old/name).read_bytes())==oldsummary['files_sha256'][name], 'Old artifact mismatch: '+name
    oldunits=runner.lines(old/'source_units.jsonl');extracted=runner.lines(old/'extractions.jsonl')
    extracted=contract.validate(oldunits,extracted)
    records={r['unit_id']:r for r in extracted};reuse=eligible(oldcfg,cfg,oldunits,units)
    pending_writes={};audit=[]
    for unit in reuse:
        uid=unit['unit_id'];record=contract.validate([unit],[records[uid]])[0]
        provenance=dict(source_experiment=OLD,source_config_sha256=oldhash,
            source_extractions_sha256=oldsummary['files_sha256']['extractions.jsonl'],
            source_record_sha256=runner.sha(runner.enc(record)),input_sha256=runner.sha(unit['text'].encode()),
            request_contract_keys=list(KEYS),api_call_performed=False)
        # Compatibility wrapper around validated triples; explicitly not a provider envelope.
        envelope=dict(unit_id=uid,config_sha256=newhash,source_kind='validated_extraction_reuse',
            reuse_provenance=provenance,response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'triples':record['triples']},ensure_ascii=False)}}]})
        assert runner.parse(contract,unit,envelope['response'])==record
        raw=runner.enc(envelope);folder=out/'responses'/runner.sha(uid.encode())
        name='validated_reuse.response.json'
        accepted=dict(unit_id=uid,config_sha256=newhash,response_file=name,response_sha256=runner.sha(raw),wrapped=True,source_kind='validated_extraction_reuse')
        pending_writes[folder/name]=raw;pending_writes[folder/'accepted.json']=runner.enc(accepted)
        audit.append(dict(unit_id=uid,**provenance))
    report=dict(status='EXACT_INPUT_EXTRACTION_REUSE_READY',experiment_id=NEW,source_experiment=OLD,
        reused_nonblank_units=len(reuse),blank_units=sum(not u['text'].strip() for u in units),
        changed_units=sorted(CHANGED),changed_unit_count=len(CHANGED),units_total=len(units),
        api_calls=0,source_files_modified=False,semantic_accuracy_verified=False,records=audit)
    pending_writes[out/'reuse_audit.json']=runner.enc(report)
    for path,raw in pending_writes.items():
        if path.exists() and path.read_bytes()!=raw:raise ValueError('Destination differs; no overwrite: '+str(path))
    for path,raw in pending_writes.items():runner.immutable(path,raw)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','run']);ap.add_argument('--project',type=Path,required=True);a=ap.parse_args()
    root=a.project.resolve();path=root/'control/scale_csbench_stage1_v1/run_stage1.py'
    import hashlib
    assert hashlib.sha256(path.read_bytes()).hexdigest()==RUNNER_SHA, 'Stage1 runner differs; do not apply blind reuse'
    spec=importlib.util.spec_from_file_location('existing_stage1',path);runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    out=root/'runs/online_experiments'/NEW/'scale_stage1';out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        contract,out,units,cfg=runner.prepare(root,NEW)
        install(runner,contract,root/'runs/online_experiments'/OLD/'scale_stage1',out,units,cfg)
        if a.command=='run':raise SystemExit(runner.execute(root,NEW,attempts=8))

if __name__=='__main__':main()
