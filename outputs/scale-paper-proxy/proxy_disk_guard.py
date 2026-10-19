"""Recover Ps-proxy response temp files and resume with disk checks; original runner unchanged."""
import argparse
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

MIB=1024*1024
SPACE_ERRORS={errno.ENOSPC,getattr(errno,'EDQUOT',122)}


def check_space(path,minimum_mib=512):
    free=shutil.disk_usage(path).free
    stats=os.statvfs(path)
    if free<minimum_mib*MIB:
        raise RuntimeError('Need at least %d MiB free; available %.1f MiB. Free space or move to an allocated filesystem first.'%(minimum_mib,free/MIB))
    if stats.f_files and stats.f_favail<2000:
        raise RuntimeError('Fewer than 2000 free inodes; resolve inode exhaustion first.')
    return free


def recover_temps(run,requests,parse):
    result=[]
    for request in requests:
        path=run/'responses'/(request['request_id']+'.json')
        temp=path.with_suffix('.json.tmp')
        if not temp.exists():continue
        if temp.is_symlink():raise ValueError('Unexpected symlink: '+str(temp))
        data=temp.read_bytes()
        entry={'request_id':request['request_id'],'temp_sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        try:
            raw=json.loads(data.decode('utf8'))
        except (ValueError,UnicodeDecodeError):
            entry['action']='archived_incomplete_temp; missing response requires explicit retry'
            archive=run/'disk_recovery_archive'/str(time.time_ns())
            archive.mkdir(parents=True)
            temp.rename(archive/temp.name)
            entry['archive_path']=str(archive/temp.name)
        else:
            # Preserve complete raw JSON even when model content fails semantic/schema checks.
            try:parse(raw);entry['schema_valid']=True
            except (ValueError,KeyError,TypeError,IndexError):entry['schema_valid']=False
            if path.exists():
                entry['action']='kept_existing_response_and_temp'
            else:
                temp.rename(path);entry['action']='recovered_raw_response_without_api'
        result.append(entry)
    return result


class Guard:
    def __init__(self,run):
        self.run=run;self.reserve=run/'disk_guard_reserve.bin'
        self.emergency=False;self.network_calls=0

    def release(self):
        if self.reserve.is_symlink():raise ValueError('Unexpected reserve symlink')
        if self.reserve.exists():self.reserve.unlink()

    def allocate(self):
        # This dedicated file is only emergency writable capacity, not experiment data.
        self.release();check_space(self.run)
        try:
            with self.reserve.open('xb') as f:
                for _ in range(32):f.write(b'\0'*MIB)
                f.flush();os.fsync(f.fileno())
        except OSError:
            self.release();raise

    def save(self,original,path,data,mutable=False):
        try:return original(path,data,mutable=mutable)
        except OSError as exc:
            if exc.errno not in SPACE_ERRORS:raise
            self.emergency=True
            self.release()
            print('DISK_PRESSURE: emergency space released; saving received data, then stopping new API requests.',flush=True)
            return original(path,data,mutable=mutable)

    def urlopen(self,original,*args,**kwargs):
        if self.emergency:raise RuntimeError('Disk emergency occurred; no further API request sent.')
        check_space(self.run)
        self.network_calls+=1
        return original(*args,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['recover','run'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    if args.command=='recover' and args.retry_failed:parser.error('retry-failed applies only to run')
    root=args.root.resolve();runner=root/'control/scale-paper-proxy/proxy.py'
    spec=importlib.util.spec_from_file_location('original_proxy',runner)
    ps=importlib.util.module_from_spec(spec);spec.loader.exec_module(ps)
    plan=root/'paper_method_v1/artifacts/proxy_inputs_v1'
    report=ps.read(plan/'summary.json')
    for name,h in report['files_sha256'].items():
        if ps.sha(plan/name)!=h:raise ValueError('Plan changed: '+name)
    if ps.read(plan/'config.json')['script_sha256']!=ps.sha(runner):raise ValueError('Original runner changed')
    run=root/'paper_method_v1/runs/proxy_glm_v1'
    if not run.is_dir():raise ValueError('Expected existing proxy run')
    guard=Guard(run)
    import fcntl
    with (run/'disk_guard.lock').open('a') as wrapper_lock:
        fcntl.flock(wrapper_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Check that original runner is not active before touching its temp files.
        with (run/'process.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            guard.release();print('FREE_MIB:',round(check_space(run)/MIB,1),flush=True)
            recovered=recover_temps(run,ps.rows(plan/'requests.jsonl'),ps.parse)
            if recovered:
                ps.save(run/'disk_recovery_archive'/('audit_%d.json'%time.time_ns()),ps.enc(recovered))
                print(json.dumps(recovered,ensure_ascii=False,indent=2),flush=True)
            guard.allocate()
        old_save=ps.save;old_urlopen=ps.urllib.request.urlopen
        ps.save=lambda path,data,mutable=False:guard.save(old_save,path,data,mutable)
        ps.urllib.request.urlopen=lambda *a,**kw:guard.urlopen(old_urlopen,*a,**kw)
        try:
            ps.execute(root,offline=args.command=='recover',limit=None,retry_failed=args.retry_failed)
        finally:
            ps.save=old_save
            ps.urllib.request.urlopen=old_urlopen
            guard.release()
        status=ps.read(run/'summary.json')
        print('ACTUAL_NETWORK_REQUESTS:',guard.network_calls,flush=True)
        if args.command=='run' and status['status']!='PROXY_GENERATION_COMPLETE_REVIEW_REQUIRED':
            raise SystemExit('PARTIAL: results retained. Inspect summary errors; do not assume completion.')


if __name__=='__main__':main()
