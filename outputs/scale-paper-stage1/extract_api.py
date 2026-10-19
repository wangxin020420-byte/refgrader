"""Six-unit Pe extraction using user-selected GLM; no synthetic answer generation."""
import argparse
import getpass
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import stage1 as m


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--model',default='glm-5.3')
    parser.add_argument('--base-url',default='https://open.bigmodel.cn/api/paas/v4')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    parsed=urllib.parse.urlparse(args.base_url)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use an HTTPS API base URL without credentials, query or fragment')
    folder=root/'paper_method_v1/artifacts/triplets'
    manifest=json.loads((folder/'stage1_manifest.json').read_text(encoding='utf8'))
    for name in ['pilot_units.jsonl','pilot_requests.jsonl']:
        if m.hashed((folder/name).read_bytes())!=manifest['outputs_sha256'][name]:raise ValueError('Changed Pe input')
    units=m.read(folder/'pilot_units.jsonl');requests=m.read(folder/'pilot_requests.jsonl')
    if len(units)!=6 or [r['unit_id'] for r in units]!=[r['unit_id'] for r in requests]:raise ValueError('Pilot IDs mismatch')
    config={'model':args.model,'base_url':args.base_url.rstrip('/'),'max_tokens':4096,'stream':False,
            'requests_sha256':m.hashed((folder/'pilot_requests.jsonl').read_bytes()),
            'runner_sha256':m.hashed(Path(__file__).read_bytes()),'automatic_retries':0,
            'identity':'SCALE Pe independent prompt, GLM replacement pilot'}
    if not args.execute:
        print(json.dumps(dict(config,status='PE_API_PREFLIGHT_OK',requests=6,api_calls=0),indent=2));return
    run=root/'paper_method_v1/runs/pe_glm_pilot_v1'
    m.write_all(run,{'config.json':m.encoded(config)})
    guard=run/'RUNNING.lock'
    with guard.open('x') as f:f.write(str(time.time()))
    try:
        key=None
        for index,req in enumerate(requests,1):
            response_name='response_%02d.json'%index
            if (run/response_name).exists():continue
            attempt_name='attempt_%02d.json'%index
            if (run/attempt_name).exists():
                print('Previous request has no saved response; inspect',run/attempt_name);break
            if key is None:
                key=getpass.getpass('GLM API Key (hidden): ').strip()
                if not key:raise ValueError('Empty API key')
            body={'model':args.model,'messages':req['messages'],'max_tokens':4096,'stream':False}
            m.write_all(run,{attempt_name:m.encoded({'unit_id':req['unit_id'],'body':body,'time':time.time()})})
            request=urllib.request.Request(config['base_url']+'/chat/completions',data=json.dumps(body).encode('utf8'),
                    headers={'Content-Type':'application/json','Authorization':'Bearer '+key},method='POST')
            try:
                with urllib.request.urlopen(request,timeout=180) as response:result=json.load(response)
                m.write_all(run,{response_name:m.encoded(result)})
                print('Saved Pe response %d/6'%index,flush=True)
            except Exception as exc:
                error={'exception_type':type(exc).__name__,'http_status':getattr(exc,'code',None)}
                m.write_all(run,{'error_%02d.json'%index:m.encoded(error)})
                print(json.dumps(error));break
        extractions=[];errors=[];usage={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0};metadata=[]
        for index,unit in enumerate(units,1):
            path=run/('response_%02d.json'%index)
            if not path.exists():errors.append({'unit_id':unit['unit_id'],'error':'missing_response'});continue
            result=json.loads(path.read_text(encoding='utf8'))
            for k in usage:usage[k]+=(result.get('usage') or {}).get(k,0) or 0
            metadata.append({'unit_id':unit['unit_id'],'returned_model':result.get('model'),
                             'response_id':result.get('id'),'system_fingerprint':result.get('system_fingerprint')})
            try:
                choice=result['choices'][0]
                if choice.get('finish_reason')!='stop':raise ValueError('Incomplete model response')
                obj=json.loads(choice['message']['content'])
                if not isinstance(obj,dict) or set(obj)!={'triples'}:raise ValueError('Expected triples object')
                row={'unit_id':unit['unit_id'],'triples':obj['triples']}
                m.validate([unit],[row]);extractions.append(row)
            except (ValueError,KeyError,IndexError,TypeError) as exc:
                errors.append({'unit_id':unit['unit_id'],'error':str(exc)})
        report={'status':'PE_PILOT_SCHEMA_OK' if not errors else 'PE_PILOT_NEEDS_ATTENTION',
                'units_valid':len(extractions),'triplet_count':sum(len(r['triples']) for r in extractions),
                'errors':errors,'usage':usage,'model_metadata':metadata,
                'semantic_review_required':True,'training_started':False}
        temp=run/'summary.tmp';temp.write_bytes(m.encoded(report));temp.replace(run/'summary.json')
        print(json.dumps(report,ensure_ascii=False,indent=2))
        if errors:raise SystemExit('Keep responses. Do not rerun or change parameters until errors are inspected.')
        extraction=m.validate(units,extractions)
        review=[];lookup={r['unit_id']:r for r in units}
        for row in extraction:
            unit=lookup[row['unit_id']]
            review.extend(['## '+row['unit_id'],'Role: '+unit['role'],'Source:',unit['text'],
                           'Extracted triples:',json.dumps(row['triples'],ensure_ascii=False,indent=2),''])
        m.write_all(folder,{'pilot_extractions.jsonl':m.jsonlines(extraction),
                            'pilot_extraction_review.txt':'\n'.join(review).encode('utf8')})
    finally:guard.unlink()


if __name__=='__main__':main()
