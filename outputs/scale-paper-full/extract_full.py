"""Training-only Pe extraction with reusable API cache and offline graph build."""
import argparse
import getpass
import json
from pathlib import Path
import sys
import time
import urllib.request
import urllib.error


def parse_response(m, unit, raw):
    choice = raw['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Response not complete')
    obj = json.loads(choice['message']['content'])
    if not isinstance(obj, dict) or set(obj) != {'triples'}:
        raise ValueError('Expected triples object')
    return m.validate([unit], [{'unit_id': unit['unit_id'], 'triples': obj['triples']}])[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--mode', choices=['prepare', 'online', 'offline'], required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root/'control/scale-paper-stage1'))
    import stage1 as m
    folder = root/'paper_method_v1/artifacts/triplets'
    pilot = root/'paper_method_v1/runs/pe_glm_pilot_v1'
    manifest = json.loads((folder/'stage1_manifest.json').read_text(encoding='utf8'))
    for name in ['source_units.jsonl', 'pilot_units.jsonl', 'pilot_requests.jsonl']:
        if m.hashed((folder/name).read_bytes()) != manifest['outputs_sha256'][name]:
            raise ValueError('Changed source: '+name)
    if m.hashed(m.PROMPT.encode('utf8')) != manifest['prompt_sha256']:
        raise ValueError('Prompt changed')
    units = m.read(folder/'source_units.jsonl')
    pilot_units = m.read(folder/'pilot_units.jsonl')
    pilot_config = json.loads((pilot/'config.json').read_text(encoding='utf8'))
    for req in m.read(folder/'pilot_requests.jsonl'):
        if req['messages'][0]['content'] != m.PROMPT:
            raise ValueError('Pilot prompt differs')
    config = {
        'source_sha256': m.hashed((folder/'source_units.jsonl').read_bytes()),
        'pilot_config_sha256': m.hashed((pilot/'config.json').read_bytes()),
        'prompt_sha256': manifest['prompt_sha256'],
        'stage1_sha256': m.hashed(Path(m.__file__).read_bytes()),
        'runner_sha256': m.hashed(Path(__file__).read_bytes()),
        'model': pilot_config['model'], 'base_url': pilot_config['base_url'],
        'max_tokens': pilot_config['max_tokens'], 'stream': False,
        'graph_mode': 'prose',
        'graph_choice': 'Summary-to-summary soft edges follow prose and equation 7; equation 4 differs.',
        'implementation': 'Independent reconstruction, GLM replacement, existing adapted training split',
        'test_accessed': False, 'validation_accessed': False
    }
    run = root/'paper_method_v1/runs/pe_glm_full_v1'
    run.mkdir(parents=True, exist_ok=True)
    # OS advisory lock releases automatically on process exit or disconnected job termination.
    import fcntl
    with (run/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m.write_all(run, {'config.json': m.encoded(config)})
        cache = run/'cache'
        for i, unit in enumerate(pilot_units, 1):
            raw_path = pilot/('response_%02d.json'%i)
            raw = json.loads(raw_path.read_text(encoding='utf8'))
            parse_response(m, unit, raw)
            name = m.hashed(unit['unit_id'].encode('utf8'))+'.json'
            m.write_all(cache, {name: m.encoded({'unit_id': unit['unit_id'], 'origin': str(raw_path.relative_to(root)), 'response': raw})})
        keys = {u['unit_id']: m.hashed(u['unit_id'].encode('utf8')) for u in units}
        missing = [u for u in units if u['text'].strip() and not (cache/(keys[u['unit_id']]+'.json')).exists()]
        print(json.dumps({'status':'FULL_EXTRACTION_PREFLIGHT', 'units':len(units),
                          'blank_units':sum(not u['text'].strip() for u in units),
                          'missing_api_responses':len(missing), 'mode':args.mode,
                          'model':config['model'], 'base_url':config['base_url']}, ensure_ascii=False), flush=True)
        if args.mode == 'prepare' or (args.mode == 'online' and not args.execute):
            return
        records = []; issues = []; usage = {'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
        key = None
        def checkpoint(status):
            summary = {'status':status,'units_total':len(units),'units_valid':len(records),
                       'triplets':sum(len(r['triples']) for r in records),'issues':issues,
                       'cached_usage_including_pilot':usage,'training_started':False,
                       'test_accessed':False,'validation_accessed':False}
            tmp = run/'summary.tmp'
            tmp.write_bytes(m.encoded(summary)); tmp.replace(run/'summary.json')
            return summary
        for unit in units:
            uid = unit['unit_id']; stem = keys[uid]
            path = cache/(stem+'.json')
            if not unit['text'].strip():
                records.append({'unit_id':uid,'triples':[]}); continue
            if not path.exists() and args.mode == 'online':
                pending = run/'attempts'/(stem+'.json')
                if pending.exists():
                    issues.append({'unit_id':uid,'error':'previous_attempt_without_response'}); continue
                if key is None:
                    key = getpass.getpass('GLM API Key (hidden): ').strip()
                    if not key: raise ValueError('Empty API key')
                body = {'model':config['model'],'max_tokens':config['max_tokens'],'stream':False,
                        'messages':[{'role':'system','content':m.PROMPT},
                                    {'role':'user','content':json.dumps({'text':unit['text']},ensure_ascii=False)}]}
                m.write_all(pending.parent, {pending.name:m.encoded({'unit_id':uid,'time':time.time(),'body_sha256':m.hashed(m.encoded(body))})})
                req = urllib.request.Request(config['base_url'].rstrip('/')+'/chat/completions',
                        data=json.dumps(body).encode('utf8'), headers={'Content-Type':'application/json','Authorization':'Bearer '+key}, method='POST')
                try:
                    with urllib.request.urlopen(req, timeout=180) as reply: raw = json.load(reply)
                    m.write_all(cache,{path.name:m.encoded({'unit_id':uid,'origin':'full_api','response':raw})})
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode('utf8',errors='replace').replace(key,'[REDACTED]')
                    error = {'unit_id':uid,'http_status':exc.code,'error_body':detail}
                    m.write_all(run/'errors',{stem+'.json':m.encoded(error)})
                    issues.append(error); checkpoint('API_BLOCKED')
                    print(json.dumps(error,ensure_ascii=False),flush=True); return
                except Exception as exc:
                    issues.append({'unit_id':uid,'error':'request_uncertain','exception_type':type(exc).__name__})
                    checkpoint('REQUEST_UNCERTAIN'); raise
                time.sleep(2)
            if not path.exists():
                issues.append({'unit_id':uid,'error':'missing_cached_response'}); continue
            try:
                saved = json.loads(path.read_text(encoding='utf8'))
                if saved['unit_id'] != uid: raise ValueError('Cache ID mismatch')
                raw = saved['response']
                for k in usage: usage[k] += (raw.get('usage') or {}).get(k,0) or 0
                records.append(parse_response(m,unit,raw))
            except (ValueError,KeyError,IndexError,TypeError) as exc:
                issues.append({'unit_id':uid,'error':str(exc)})
            checkpoint('EXTRACTING')
            print('Validated %d/%d; issues=%d'%(len(records),len(units),len(issues)),flush=True)
        if issues:
            print(json.dumps(checkpoint('INCOMPLETE_CACHE_OR_SCHEMA'),ensure_ascii=False),flush=True); return
        records = m.validate(units,records)
        data = m.jsonlines(records)
        graph = m.graph(records,'prose')
        graph.update(scope='all frozen training source units',extractions_sha256=m.hashed(data),
                     config_sha256=m.hashed(m.encoded(config)))
        m.write_all(root/'paper_method_v1/artifacts/full_graph_v1',
                    {'extractions.jsonl':data,'graph.json':m.encoded(graph)})
        print(json.dumps(checkpoint('FULL_EXTRACTION_AND_GRAPH_COMPLETE'),ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
