"""One-unit, audited token-budget recovery; offline export of other cached units."""
import getpass
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import urllib.error
import urllib.request

UID = 'synthetic_text:9315c83f7b10445fc609198791a99b21ebc733895a8c561fba41801eb003cbc0'
EXPECTED = {
    'entity_engine.py':'32757e1fcc5bfaa5325a16c6a57152a26819ad29ddd10bde331c3a718dcc612d',
    'run_entities.py':'b3ebb0d5d073e604d047c5fb6c8b8f9c206ceb43dbc84851584d025f364a829d',
    'stage1_contract.py':'26c9fe7ee85fb7760e70bfa29da87533cd1266d7d147e2e60c1fa543355c919c',
}
BUDGETS = (16384,32768,32768)


def body_for(stage,cfg,unit,budget):
    return dict(model=cfg['model'], max_tokens=budget, stream=False, messages=[
        dict(role='system',content=stage.PROMPT),
        dict(role='user',content=json.dumps({'text':unit['text']},ensure_ascii=False))])


def supplement(e,stage,cfg,unit,folder):
    key=None; calls=0
    for n,budget in enumerate(BUDGETS,1):
        body=body_for(stage,cfg,unit,budget); fingerprint=e.digest(e.enc(body))
        prefix='attempt_%03d'%n
        intent=folder/(prefix+'.request.json'); response=folder/(prefix+'.response.json')
        if intent.exists() and e.read(intent)['body_sha256']!=fingerprint:
            raise ValueError('Supplement request mismatch')
        if response.exists():
            if not intent.exists():raise ValueError('Supplement response missing request provenance')
            raw=e.read(response)
        elif intent.exists():
            continue  # Preserve interrupted attempts; use next distinct slot.
        else:
            if key is None:
                key=os.environ.get('ZHIPUAI_API_KEY','').strip() or getpass.getpass('Coding API Key (hidden): ').strip()
                if not key:raise ValueError('Empty API key')
            e.write(intent,e.enc(dict(unit_id=UID,body_sha256=fingerprint,body=body,time=time.time())))
            req=urllib.request.Request(cfg['base_url']+'/chat/completions',data=e.enc(body),
                headers={'Content-Type':'application/json','Authorization':'Bearer '+key},method='POST')
            calls+=1
            try:
                with urllib.request.urlopen(req,timeout=900) as reply:wire=reply.read()
                e.write(folder/(prefix+'.body.bin'),wire)
                raw=json.loads(wire); e.write(response,e.enc(raw))
            except urllib.error.HTTPError as exc:
                e.write(folder/(prefix+'.http_error.bin'),exc.read())
                e.write(folder/(prefix+'.error.json'),e.enc(dict(exception=type(exc).__name__,http_status=exc.code)))
                if exc.code!=429 and not 500<=exc.code<=599:raise
                time.sleep(10); continue
            except (socket.timeout,TimeoutError,urllib.error.URLError,ConnectionError,json.JSONDecodeError,UnicodeDecodeError) as exc:
                e.write(folder/(prefix+'.error.json'),e.enc(dict(exception=type(exc).__name__,http_status=None)))
                time.sleep(10); continue
        try:
            result=e.parse(stage,unit,raw)
        except (ValueError,KeyError,TypeError,IndexError,AttributeError) as exc:
            e.write(folder/(prefix+'.validation.json'),e.enc(dict(error=str(exc))))
            print('SUPPLEMENT_REJECTED',n,'max_tokens',budget,str(exc),flush=True)
            continue
        accepted=dict(unit_id=UID,body_sha256=fingerprint,response_file=response.name,
                      max_tokens=budget,extraction=result)
        e.write(folder/'accepted.json',e.enc(accepted))
        return result,calls,accepted
    raise RuntimeError('Supplement attempts exhausted; original caches retained. Inspect supplement responses.')


def export(e,prepared,root,out,recovered,calls,accepted):
    stage,cfg,datasets,bindings,pending,reused=prepared
    other=[u for u in pending if u['unit_id']!=UID]
    previous_prepare=getattr(e,'prepare',None)
    e.prepare=lambda *_:(stage,cfg,datasets,bindings,other,reused+[recovered])
    try:
        rc=e.execute(root,out,'offline',8,900)
    finally:
        if previous_prepare is None:del e.prepare
        else:e.prepare=previous_prepare
    report=e.read(out/'summary.json')
    report['unique_synthetic_requests']=len(pending)
    report['valid_synthetic_units']+=1
    report['new_api_calls']=calls
    report['supplemental_extraction']=dict(unit_id=UID,max_tokens=accepted['max_tokens'],
        request_sha256=accepted['body_sha256'],policy_file='length_recovery_v1/policy.json',
        note='One-unit token-limit deviation; same prompt/model/validator; original response cache retained.')
    e.write(out/'summary.json',e.enc(report),mutable=True)
    print('FINAL_SUMMARY',e.enc(report).decode(),flush=True)
    return rc


def main():
    root=Path('/data0/E125221219/projects/scale-private')
    original=root/'control/train_entities_clean_v1'
    for name,expected in EXPECTED.items():
        if hashlib.sha256((original/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen source changed: '+name)
    sys.path.insert(0,str(original))
    import entity_engine as e
    import run_entities as adapter
    out=root/'runs/online_experiments'/adapter.EXP/'train_entities_five_runs_v1'
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepared=adapter.prepare(root,out)
        stage,cfg,_,_,pending,_=prepared
        target=[u for u in pending if u['unit_id']==UID]
        if len(target)!=1 or len(pending)!=1735:raise ValueError('Unexpected pending unit inventory')
        # Verify all 1,734 other cached results before spending any API tokens.
        for unit in pending:
            if unit['unit_id']==UID:continue
            p=out/'responses'/unit['unit_id'].split(':',1)[1]/'accepted.json'
            saved=e.read(p)
            if saved['body_sha256']!=e.digest(e.enc(body_for(stage,cfg,unit,cfg['max_tokens']))):
                raise ValueError('Cache request changed: '+unit['unit_id'])
            e.parse(stage,unit,saved['response'])
        print('CACHE_PREFLIGHT_OK 1734; only one unit may call API',flush=True)
        folder=out/'length_recovery_v1'
        policy=dict(unit_id=UID,original_max_tokens=cfg['max_tokens'],attempt_max_tokens=list(BUDGETS),
                    source_config_sha256=e.sha(out/'config.json'),runner_sha256=e.sha(Path(__file__)),
                    original_runner_sha256=EXPECTED,model=cfg['model'],base_url=cfg['base_url'],
                    changed_request_fields=['max_tokens'],labels_modified=False,
                    semantic_accuracy_verified=False,api_calls_to_other_units=0)
        e.write(folder/'policy.json',e.enc(policy))
        recovered,calls,accepted=supplement(e,stage,cfg,target[0],folder)
        return export(e,prepared,root,out,recovered,calls,accepted)


if __name__=='__main__':raise SystemExit(main())
