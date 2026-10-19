"""Audited, bounded timeout recovery without changing the frozen Pe runner."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import urllib.error


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    tmp.replace(p)


def recover(run,audit):
    summary=read(run/'summary.json')
    if summary.get('status')!='PE_REQUEST_INTERRUPTED':return False
    errors=summary.get('errors',[])
    # Earlier schema errors may coexist; only the final interrupted request is retried.
    if not errors:return False
    error=errors[-1]
    if error.get('exception_type') not in ('TimeoutError','timeout') or error.get('response_saved') is not False:
        return False
    uid=error['unit_id'];attempt=error['attempt']
    if not isinstance(attempt,int) or not 1<=attempt<=3:raise ValueError('Invalid attempt')
    stem=hashlib.sha256(uid.encode('utf8')).hexdigest()
    folder=run/'responses'/stem
    request=folder/('attempt_%03d.request.json'%attempt)
    response=folder/('attempt_%03d.response.json'%attempt)
    if response.exists():return False
    if not request.is_file():return False
    if read(request)['unit_id']!=uid:raise ValueError('Pending request identity mismatch')
    archive=audit/('timeout_'+str(time.time_ns()))
    archive.mkdir(parents=True)
    save(archive/'summary_before_retry.json',summary)
    save(archive/'recovery.json',{'unit_id':uid,'attempt':attempt,'request_sha256':sha(request),
         'request_path':str(request),'action':'Archive uncertain timeout intent, permit one reissue',
         'possible_duplicate_server_processing':True})
    request.rename(archive/request.name)
    print('TIMEOUT_RETRY_PREPARED',uid,flush=True)
    return True


def execute(root,timeout,retries):
    run=root/'paper_method_v2/runs/pe_glm_restored_v1'
    path=root/'control/scale-paper-protocol/extract_restored.py'
    config=read(run/'config.json')
    if sha(path)!=config['runner_sha256']:
        raise ValueError('Frozen extraction runner changed; restore original file before resuming')
    if (run/'summary.json').exists():
        prior=read(run/'summary.json')
        if prior.get('config_sha256')!=sha(run/'config.json'):raise ValueError('Config checksum mismatch')
        if prior.get('status')=='RESTORED_PE_AND_GRAPH_COMPLETE':
            print('ALREADY_COMPLETE: no requests sent');return 0
    spec=importlib.util.spec_from_file_location('restored_original',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    audit=run/'transport_recovery'/str(time.time_ns());audit.mkdir(parents=True)
    original_urlopen=module.urllib.request.urlopen
    original_getpass=module.getpass.getpass
    key=[];network_calls=0;recoveries=0
    def getkey(prompt):
        if not key:key.append(original_getpass(prompt))
        return key[0]
    def urlopen(req,**kwargs):
        nonlocal network_calls
        network_calls+=1
        kwargs['timeout']=timeout
        try:
            return original_urlopen(req,**kwargs)
        except urllib.error.URLError as exc:
            # urllib may wrap a socket timeout. Normalize only proven timeouts,
            # not authentication, DNS, or other transport failures.
            if isinstance(exc.reason,TimeoutError):
                raise TimeoutError('Wrapped socket timeout') from exc
            raise
    module.getpass.getpass=getkey;module.urllib.request.urlopen=urlopen
    result=2
    def record():
        save(audit/'transport_summary.json',{'socket_timeout_seconds':timeout,
             'timeout_reissues':recoveries,'max_timeout_reissues':retries,
             'actual_network_requests_this_wrapper':network_calls,'exit_code':result,
             'frozen_runner_sha256':sha(path),'config_sha256':sha(run/'config.json'),
             'wrapper_sha256':sha(Path(__file__)),
             'note':'Client transport change only; generation body unchanged. Timed-out requests may have been billed.'})
    try:
        if (run/'summary.json').exists() and recover(run,audit):recoveries+=1
        record()
        while True:
            result=module.execute(root)
            record()
            if result==0 or recoveries>=retries:break
            if not recover(run,audit):break
            recoveries+=1;record()
            print('Waiting 15 seconds before timeout resume',flush=True);time.sleep(15)
        return result
    finally:
        record()
        module.urllib.request.urlopen=original_urlopen;module.getpass.getpass=original_getpass
        print('ACTUAL_NETWORK_REQUESTS_THIS_WRAPPER:',network_calls,flush=True)
        print('TRANSPORT_AUDIT:',audit/'transport_summary.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=900)
    p.add_argument('--timeout-retries',type=int,default=100)
    a=p.parse_args()
    if not 180<=a.timeout<=1800 or not 1<=a.timeout_retries<=1000:p.error('timeout 180..1800; retries 1..1000')
    root=a.root.resolve();run=root/'paper_method_v2/runs/pe_glm_restored_v1'
    import fcntl
    with (run/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raise SystemExit(execute(root,a.timeout,a.timeout_retries))
