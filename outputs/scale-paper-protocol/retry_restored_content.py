"""Bounded content retries preserving every failed response and strict Pe validation."""
import argparse
import getpass
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8');tmp.replace(path)


def inspect_failures(root,summary):
    run=root/'paper_method_v2/runs/pe_glm_restored_v1'
    units_path=root/'paper_method_v2/artifacts/pe_inputs_v1/source_units.jsonl'
    units={r['unit_id']:r for r in (json.loads(s) for s in units_path.read_text(encoding='utf8').splitlines() if s.strip())}
    folders=[];details=[];seen=set()
    for error in summary.get('errors',[]):
        uid=error['unit_id']
        if uid in seen:raise ValueError('Duplicate failure identity')
        seen.add(uid)
        if uid not in units:raise ValueError('Unknown failed unit')
        stem=hashlib.sha256(uid.encode('utf8')).hexdigest()
        folder=run/'responses'/stem
        if folder.is_symlink() or folder.resolve().parent!=(run/'responses').resolve():
            raise ValueError('Unexpected response folder')
        if (run/'accepted'/(stem+'.json')).exists():raise ValueError('Refuse to reset accepted unit')
        attempts=[]
        for i in range(1,4):
            req=folder/('attempt_%03d.request.json'%i)
            resp=folder/('attempt_%03d.response.json'%i)
            if not req.is_file() or not resp.is_file():
                raise ValueError('Missing request/response; not a completed content failure: '+uid)
            if read(req)['unit_id']!=uid:raise ValueError('Request identity mismatch')
            raw=read(resp);choice=(raw.get('choices') or [{}])[0]
            attempts.append({'attempt':i,'finish_reason':choice.get('finish_reason'),
                'raw_content':choice.get('message',{}).get('content'),
                'response_sha256':sha(resp),'request_sha256':sha(req)})
        folders.append(folder)
        details.append({'unit_id':uid,'source_text':units[uid]['text'],'error':error,'attempts':attempts})
    return folders,details


def execute(root,rounds,timeout,timeout_retries):
    run=root/'paper_method_v2/runs/pe_glm_restored_v1'
    recovery_path=root/'control/scale-paper-protocol/resume_restored_pe.py'
    recovery=load(recovery_path,'transport_recovery')
    audit=run/'content_retries'/str(time.time_ns());audit.mkdir(parents=True)
    save(audit/'policy.json',{'additional_content_rounds':rounds,'attempts_per_round':3,
         'timeout_seconds':timeout,'timeout_reissues_per_transport_invocation':timeout_retries,
         'prompt_changed':False,'validation_relaxed':False,'model_parameters_changed':False,
         'wrapper_sha256':sha(Path(__file__)),'transport_wrapper_sha256':sha(recovery_path),
         'selection_rule':'First response that passes the existing strict extraction validator; not semantic certification'})
    original_getpass=getpass.getpass;key=[]
    def remembered_key(prompt):
        if not key:key.append(original_getpass(prompt))
        return key[0]
    getpass.getpass=remembered_key
    try:
        for cycle in range(rounds+1):
            # Refresh the summary and validate all cached responses first.
            code=recovery.execute(root,timeout,timeout_retries)
            summary=read(run/'summary.json')
            if code==0:
                save(audit/'result.json',{'status':'CONTENT_RETRY_COMPLETE','additional_rounds_started':cycle,
                     'final_summary_sha256':sha(run/'summary.json')})
                print('RESTORED_CONTENT_RETRY_AND_GRAPH_DONE',flush=True);return 0
            if summary.get('status')!='PE_PARTIAL':
                print('Transport or other failure; preserve state and inspect summary',flush=True);return code
            folders,details=inspect_failures(root,summary)
            save(audit/('diagnostics_%02d.json'%cycle),{'failed_units':len(details),'details':details})
            print('FAILED_CONTENT_UNITS',len(details),'ROUND',cycle,flush=True)
            if not folders:raise ValueError('Partial state without failure units')
            if cycle==rounds:
                print('CONTENT_RETRY_LIMIT_REACHED; diagnostics:',audit/('diagnostics_%02d.json'%cycle),flush=True)
                return 2
            archive=audit/('round_%02d'%(cycle+1));archive.mkdir()
            save(archive/'summary_before_retry.json',summary)
            # All paths and response pairs were checked before any move.
            for folder in folders:
                folder.rename(archive/folder.name)
            print('CONTENT_RETRY_START',cycle+1,'units',len(folders),flush=True)
        return 2
    finally:
        getpass.getpass=original_getpass
        print('CONTENT_RETRY_AUDIT:',audit,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--content-rounds',type=int,default=5)
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--timeout-retries',type=int,default=100)
    a=p.parse_args()
    if not 1<=a.content_rounds<=20 or not 180<=a.timeout<=1800 or not 1<=a.timeout_retries<=1000:
        p.error('content-rounds 1..20; timeout 180..1800; timeout-retries 1..1000')
    root=a.root.resolve();run=root/'paper_method_v2/runs/pe_glm_restored_v1'
    import fcntl
    with (run/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raise SystemExit(execute(root,a.content_rounds,a.timeout,a.timeout_retries))
