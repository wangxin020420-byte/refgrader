"""Bounded recovery of failed Pe outputs; never edit successfully validated caches."""
import argparse
import copy
import getpass
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


def restore_whitespace(source, phrase):
    if phrase in source: return phrase
    positions = [i for i,c in enumerate(source) if not c.isspace()]
    compact = ''.join(source[i] for i in positions)
    needle = ''.join(c for c in phrase if not c.isspace())
    if not needle: return None
    start = compact.find(needle)
    if start < 0 or compact.find(needle,start+1) >= 0: return None
    return source[positions[start]:positions[start+len(needle)-1]+1]


def offline_fix(raw, unit):
    result = copy.deepcopy(raw)
    choice = result['choices'][0]
    if choice.get('finish_reason') != 'stop': raise ValueError('Incomplete response')
    obj = json.loads(choice['message']['content'])
    changes = []
    for i,triple in enumerate(obj['triples']):
        if set(triple)=={'t','c','w',''} and triple['']=='':
            del triple['']; changes.append({'index':i,'operation':'remove_empty_key'})
        if isinstance(triple.get('t'),str) and triple['t'] not in unit['text']:
            restored = restore_whitespace(unit['text'],triple['t'])
            if restored is None: raise ValueError('Not a whitespace-only difference')
            changes.append({'index':i,'operation':'restore_source_whitespace','before':triple['t'],'after':restored})
            triple['t'] = restored
    choice['message']['content'] = json.dumps(obj,ensure_ascii=False)
    return result,changes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(); root=args.root.resolve()
    sys.path.insert(0,str(root/'control/scale-paper-stage1'))
    sys.path.insert(0,str(root/'control/scale-paper-full'))
    import stage1 as m
    from extract_full import parse_response
    import fcntl
    run=root/'paper_method_v1/runs/pe_glm_full_v1'
    repair=run/'recovery_v1'
    with (run/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=json.loads((run/'config.json').read_text(encoding='utf8'))
        assert m.hashed(Path(m.__file__).read_bytes())==cfg['stage1_sha256'],'Extraction module changed'
        source=root/'paper_method_v1/artifacts/triplets/source_units.jsonl'
        assert m.hashed(source.read_bytes())==cfg['source_sha256'],'Source changed'
        assert m.hashed(m.PROMPT.encode('utf8'))==cfg['prompt_sha256'],'Prompt changed'
        units=m.read(source); pending=[]; valid=0
        for u in units:
            stem=m.hashed(u['unit_id'].encode('utf8')); path=run/'cache'/(stem+'.json')
            if not u['text'].strip(): valid+=1;continue
            old=None; reason='missing_response'; fixed=None; changes=[]
            if path.exists():
                old=json.loads(path.read_text(encoding='utf8'))
                assert old['unit_id']==u['unit_id'],'Cache ID mismatch'
                try: parse_response(m,u,old['response']);valid+=1;continue
                except (ValueError,KeyError,IndexError,TypeError) as e: reason=str(e)
                try:
                    fixed,changes=offline_fix(old['response'],u)
                    parse_response(m,u,fixed)
                except (ValueError,KeyError,IndexError,TypeError): fixed=None;changes=[]
            pending.append((u,stem,path,old,reason,fixed,changes))
        print(json.dumps({'valid_untouched':valid,'pending':len(pending),
                          'offline_repairable':sum(x[5] is not None for x in pending),
                          'api_units':sum(x[5] is None for x in pending),'max_attempts_per_unit':2},ensure_ascii=False),flush=True)
        if not args.execute:return
        m.write_all(repair,{'policy.json':m.encoded({'config_sha256':m.hashed((run/'config.json').read_bytes()),
            'script_sha256':m.hashed(Path(__file__).read_bytes()),'max_attempts':2,
            'timeout_seconds':600,'length_retry_max_tokens':16384,
            'prompt':'unchanged original Pe prompt; no labels or error feedback',
            'offline_policy':'unique whitespace-only source span restoration or empty extra JSON key removal',
            'note':'Recovery policy is an implementation choice, not an author-specified method.'})})
        key=None; outcomes=[]
        def save_summary():
            tmp=repair/'summary.tmp';tmp.write_bytes(m.encoded({'outcomes':outcomes,'training_started':False}))
            tmp.replace(repair/'summary.json')
        for u,stem,path,old,reason,fixed,changes in pending:
            slot=repair/stem
            m.write_all(slot,{'input.json':m.encoded({'unit_id':u['unit_id'],'reason':reason,
                'original_cache_sha256':m.hashed(path.read_bytes()) if path.exists() else None})})
            if old is not None:m.write_all(slot,{'original_cache.json':path.read_bytes()})
            raw=fixed; selected='offline'; attempted=[]
            max_tokens=16384 if reason=='Response not complete' else cfg['max_tokens']
            if raw is None:
                for number in (1,2):
                    resp=slot/('response_%d.json'%number); attempt=slot/('attempt_%d.json'%number)
                    if not resp.exists():
                        if attempt.exists():
                            attempted.append({'attempt':number,'error':'uncertain_previous_attempt'});break
                        if key is None:
                            key=getpass.getpass('GLM API Key (hidden): ').strip()
                            if not key:raise ValueError('Empty key')
                        body={'model':cfg['model'],'max_tokens':max_tokens,'stream':False,
                            'messages':[{'role':'system','content':m.PROMPT},{'role':'user','content':json.dumps({'text':u['text']},ensure_ascii=False)}]}
                        m.write_all(slot,{attempt.name:m.encoded({'body':body,'time':time.time()})})
                        request=urllib.request.Request(cfg['base_url'].rstrip('/')+'/chat/completions',
                            data=json.dumps(body).encode('utf8'),headers={'Content-Type':'application/json','Authorization':'Bearer '+key},method='POST')
                        try:
                            with urllib.request.urlopen(request,timeout=600) as response:result=json.load(response)
                            m.write_all(slot,{resp.name:m.encoded(result)})
                        except Exception as exc:
                            detail=exc.read().decode('utf8',errors='replace') if isinstance(exc,urllib.error.HTTPError) else str(exc)
                            error={'exception':type(exc).__name__,'http_status':getattr(exc,'code',None),'detail':detail.replace(key,'[REDACTED]')}
                            m.write_all(slot,{'error_%d.json'%number:m.encoded(error)})
                            outcomes.append({'unit_id':u['unit_id'],'status':'REQUEST_FAILED','error':error})
                            save_summary();print(json.dumps(error,ensure_ascii=False),flush=True);return
                        time.sleep(3)
                    result=json.loads(resp.read_text(encoding='utf8'))
                    try:
                        parse_response(m,u,result);raw=result;changes=[]
                    except (ValueError,KeyError,IndexError,TypeError):
                        try:
                            candidate,edits=offline_fix(result,u);parse_response(m,u,candidate)
                            raw=candidate;changes=edits
                        except (ValueError,KeyError,IndexError,TypeError) as exc:
                            attempted.append({'attempt':number,'error':str(exc)})
                            if (result.get('choices') or [{}])[0].get('finish_reason')=='length':max_tokens=16384
                            continue
                    selected=resp.name;break
            if raw is not None:
                parse_response(m,u,raw)
                replacement={'unit_id':u['unit_id'],'origin':'recovery_v1/'+stem+'/'+selected,'response':raw,
                             'recovery_changes':changes,'original_reason':reason}
                m.write_all(slot,{'accepted_cache.json':m.encoded(replacement)})
                if path.exists() and old is not None:
                    assert path.read_bytes()==(slot/'original_cache.json').read_bytes(),'Concurrent cache change'
                temp=slot/'install.tmp';temp.write_bytes(m.encoded(replacement));temp.replace(path)
                outcomes.append({'unit_id':u['unit_id'],'status':'RECOVERED','via':selected,'changes':changes})
            else:outcomes.append({'unit_id':u['unit_id'],'status':'UNRESOLVED','attempts':attempted})
            save_summary();print(u['unit_id'],outcomes[-1]['status'],flush=True)
        print('Recovery pass finished. Run original full extractor in offline mode.',flush=True)


if __name__=='__main__':main()
