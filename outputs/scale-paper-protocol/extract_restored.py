"""Resume restored training Pe with the existing user-authorized GLM profile."""
import argparse
import getpass
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import urllib.request
import urllib.parse


def load(path,name):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf8')
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(raw);tmp.replace(path)


def parse(stage,unit,raw):
    choice=raw['choices'][0]
    if choice.get('finish_reason')!='stop':raise ValueError('Response not complete')
    obj=json.loads(choice['message']['content'])
    if not isinstance(obj,dict) or set(obj)!={'triples'}:raise ValueError('Expected triples object')
    return stage.validate([unit],[{'unit_id':unit['unit_id'],'triples':obj['triples']}])[0]


def execute(root):
    stage=load(root/'control/scale-paper-stage1/stage1.py','pe_stage')
    inputs=root/'paper_method_v2/artifacts/pe_inputs_v1'
    meta=read(inputs/'manifest.json')
    if sha(inputs/'source_units.jsonl')!=meta['outputs_sha256']['source_units.jsonl']:
        raise ValueError('Changed new units')
    if sha(Path(stage.__file__))!=meta['stage1_code_sha256']:
        raise ValueError('Changed extraction contract')
    units=stage.read(inputs/'source_units.jsonl')
    old=root/'paper_method_v1/runs/pe_glm_full_v1'
    profile=read(old/'config.json')
    if profile['model']!='glm-5.3':raise ValueError('Expected user-authorized glm-5.3 profile')
    if profile['prompt_sha256']!=stage.hashed(stage.PROMPT.encode('utf8')):
        raise ValueError('Old extraction prompt differs; do not reuse cache')
    url=profile['base_url'].rstrip('/')
    parsed=urllib.parse.urlparse(url)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Invalid credential-free HTTPS endpoint')
    source=root/'paper_method_v1/artifacts/triplets'
    source_meta=read(source/'stage1_manifest.json')
    if sha(source/'source_units.jsonl')!=source_meta['outputs_sha256']['source_units.jsonl']:
        raise ValueError('Old units changed')
    previous={u['unit_id']:u for u in stage.read(source/'source_units.jsonl')}
    out=root/'paper_method_v2/runs/pe_glm_restored_v1'
    config={'model':profile['model'],'base_url':url,'max_tokens':profile['max_tokens'],'stream':False,
            'prompt_sha256':profile['prompt_sha256'],'source_sha256':sha(inputs/'source_units.jsonl'),
            'old_profile_sha256':sha(old/'config.json'),'stage1_sha256':sha(Path(stage.__file__)),
            'runner_sha256':sha(Path(__file__)),'max_attempts_per_unit':3,
            'scope':'1107 training answers and their references; GLM replacement; independently reconstructed prompt'}
    stage.write_all(out,{'config.json':stage.encoded(config)})
    records=[];errors=[];calls=0;reused=0;key=None
    def summary(status):
        report={'status':status,'training_records':1107,'units_total':len(units),'units_valid':len(records),
                'remaining':len(units)-len(records),'triplets':sum(len(r['triples']) for r in records),
                'old_responses_reused':reused,'new_api_calls_this_invocation':calls,'errors':errors,
                'config_sha256':sha(out/'config.json'),'validation_accessed':False,'test_accessed':False,
                'training_started':False,'semantic_review_required':True}
        save(out/'summary.json',report);return report
    print('PROFILE',config['model'],url,'max_tokens',config['max_tokens'],flush=True)
    for i,unit in enumerate(units,1):
        uid=unit['unit_id'];stem=stage.hashed(uid.encode('utf8'))
        accepted=out/'accepted'/(stem+'.json')
        if not unit['text'].strip():records.append({'unit_id':uid,'triples':[]});continue
        if accepted.exists():
            saved=read(accepted)
            if saved['unit_id']!=uid:raise ValueError('Accepted identity mismatch')
            records.append(parse(stage,unit,saved['response']))
            reused+=int(saved['origin']=='old_cache')
            continue
        prior=previous.get(uid);path=old/'cache'/(stem+'.json')
        if prior and all(prior[k]==unit[k] for k in ('text','role','question_id')) and path.exists():
            raw=read(path)
            try:
                if raw['unit_id']!=uid:raise ValueError('Old cache identity mismatch')
                valid=parse(stage,unit,raw['response'])
            except (ValueError,KeyError,TypeError,IndexError):pass
            else:
                save(accepted,{'unit_id':uid,'origin':'old_cache','source_sha256':sha(path),'response':raw['response']})
                records.append(valid);reused+=1;continue
        attempts=out/'responses'/stem
        valid=None;last='No valid response'
        for attempt in range(1,4):
            response=attempts/('attempt_%03d.response.json'%attempt)
            intent=attempts/('attempt_%03d.request.json'%attempt)
            if not response.exists():
                if intent.exists():
                    last='Request attempted without a saved response; inspect before reissuing'
                    break
                if key is None:
                    key=getpass.getpass('GLM API Key (hidden): ').strip()
                    if not key:raise ValueError('Empty API key')
                body={'model':config['model'],'max_tokens':config['max_tokens'],'stream':False,
                      'messages':[{'role':'system','content':stage.PROMPT},
                                  {'role':'user','content':json.dumps({'text':unit['text']},ensure_ascii=False)}]}
                save(intent,{'unit_id':uid,'time':time.time(),'body':body})
                req=urllib.request.Request(url+'/chat/completions',data=json.dumps(body).encode('utf8'),
                     headers={'Content-Type':'application/json','Authorization':'Bearer '+key},method='POST')
                calls+=1
                try:
                    with urllib.request.urlopen(req,timeout=180) as reply: raw=json.load(reply)
                    save(response,raw)
                except Exception as exc:
                    errors.append({'unit_id':uid,'attempt':attempt,'exception_type':type(exc).__name__,
                                   'http_status':getattr(exc,'code',None),'response_saved':response.exists()})
                    print(json.dumps(summary('PE_REQUEST_INTERRUPTED'),ensure_ascii=False),flush=True)
                    return 2
                time.sleep(2)
            raw=read(response)
            try:valid=parse(stage,unit,raw)
            except (ValueError,KeyError,TypeError,IndexError) as exc:last=str(exc);continue
            save(accepted,{'unit_id':uid,'origin':'new_request','response':raw})
            break
        if valid is None:errors.append({'unit_id':uid,'error':last})
        else:records.append(valid)
        print('PE %d/%d valid=%d reused=%d calls=%d'%(i,len(units),len(records),reused,calls),flush=True)
        summary('PE_RUNNING')
    if errors:
        print(json.dumps(summary('PE_PARTIAL'),ensure_ascii=False,indent=2));return 2
    records=stage.validate(units,records)
    raw=stage.jsonlines(records)
    graph=stage.graph(records,'prose')
    graph.update(scope='Restored 1107 training records and references',extractions_sha256=stage.hashed(raw),
                 config_sha256=sha(out/'config.json'),graph_choice='Existing prose/equation-7 reconstruction retained; equation-4 ambiguity unresolved')
    stage.write_all(root/'paper_method_v2/artifacts/full_graph_v1',{'extractions.jsonl':raw,'graph.json':stage.encoded(graph)})
    print(json.dumps(summary('RESTORED_PE_AND_GRAPH_COMPLETE'),ensure_ascii=False,indent=2))
    print('RESTORED_PE_AND_GRAPH_DONE',flush=True);return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();out=root/'paper_method_v2/runs/pe_glm_restored_v1';out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raise SystemExit(execute(root))
