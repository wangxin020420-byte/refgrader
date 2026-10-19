"""Restore only uniquely alignable CR/LF in two saved PE responses; no network."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

TARGETS = {
    'student:ANS_CO_1546': '1791559739924620854.response.json',
    'student:ANS_CO_987': '1791562870288635200.response.json',
}

def restore(source, phrase):
    if phrase in source:
        return phrase
    # Only missing line endings can be restored. Spaces, bars, symbols and
    # letters are never normalized, substituted or inferred.
    if '\n' in phrase or '\r' in phrase or not phrase:
        raise ValueError('Not a missing-linebreak-only case: '+repr(phrase))
    positions=[i for i,c in enumerate(source) if c not in '\r\n']
    compact=''.join(source[i] for i in positions)
    matches=[]
    start=0
    while True:
        at=compact.find(phrase,start)
        if at<0:break
        span=source[positions[at]:positions[at+len(phrase)-1]+1]
        matches.append(span)
        start=at+1
    if len(matches)!=1:
        raise ValueError('Expected one exact match ignoring only CR/LF; got '+str(len(matches))+': '+repr(phrase))
    return matches[0]

def repaired_response(a,m,unit,envelope):
    result=copy.deepcopy(envelope)
    response=result['response']
    choice=response['choices'][0]
    if choice['finish_reason']!='stop':raise ValueError('Incomplete API response')
    obj=json.loads(choice['message']['content'])
    if not isinstance(obj,dict) or set(obj)!={'triples'}:raise ValueError('Wrong schema')
    changes=[]
    for i,triple in enumerate(obj['triples']):
        old=triple['t']
        new=restore(unit['text'],old)
        if new!=old:
            changes.append(dict(index=i,before=old,after=new))
            triple['t']=new
    choice['message']['content']=json.dumps(obj,ensure_ascii=False)
    a.parse(m,unit,response)
    return result,changes

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    args=p.parse_args()
    project=args.project.resolve()
    path=project/'control/scale_csbench_stage1_v1/run_stage1.py'
    spec=importlib.util.spec_from_file_location('csbench_pe',path)
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    experiment='csbench_scale_online_v1'
    out=project/'runs/online_experiments'/experiment/'scale_stage1'
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Existing contract checks ensure the experiment/config has not changed.
        m,_,units,cfg=a.prepare(project,experiment)
        config_sha=a.sha(a.enc(cfg))
        lookup={u['unit_id']:u for u in units}
        planned=[]
        for uid,filename in TARGETS.items():
            folder=out/'responses'/a.sha(uid.encode())
            if (folder/'accepted.json').exists():
                print('ALREADY_ACCEPTED',uid,flush=True)
                continue
            source=folder/filename
            raw=source.read_bytes()
            envelope=json.loads(raw)
            if envelope['unit_id']!=uid or envelope['config_sha256']!=config_sha:
                raise ValueError('Saved response identity/config mismatch')
            fixed,changes=repaired_response(a,m,lookup[uid],envelope)
            if not changes:raise ValueError('No linebreak repair needed: '+uid)
            audit=dict(unit_id=uid,source_file=filename,source_sha256=a.sha(raw),
                policy='unique exact matching after removing CR/LF from source only',
                changes=changes,api_calls=0,semantic_accuracy_verified=False,
                note='Locally derived response; only t linebreaks restored. c and w unchanged.')
            fixed['local_linebreak_repair']=audit
            planned.append((source,raw,fixed,audit))
            print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
        if not args.apply:
            print('DRY_RUN_PASS: all proposed responses pass the original validator; no cache writes.')
            return
        # Both proposals have passed before either accepted cache is created.
        for source,raw,fixed,audit in planned:
            dest=source.parent/(str(time.time_ns())+'.response.json')
            a.immutable(dest,a.enc(fixed))
            audit=dict(audit,derived_response_file=dest.name,derived_sha256=a.sha(dest.read_bytes()))
            a.immutable(source.parent/(dest.stem+'.audit.json'),a.enc(audit))
            if source.read_bytes()!=raw:raise ValueError('Original response changed')
        def forbidden(**kwargs):
            raise RuntimeError('Offline repair: API requests forbidden')
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=forbidden)))
        rc=a.execute(project,experiment,attempts=1,client=client,sleep=lambda _:None)
        report=a.read(out/'summary.json')
        if report['new_api_calls']!=0:
            raise RuntimeError('Unexpected unresolved units; no actual network calls were made')
        print('OFFLINE_LINEBREAK_RECOVERY_DONE',flush=True)
        raise SystemExit(rc)

if __name__=='__main__':main()
