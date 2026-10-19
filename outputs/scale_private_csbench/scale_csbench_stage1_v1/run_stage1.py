"""Isolated CSBench SCALE training-only extraction and graph, with resumable envelopes."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time

CONTRACT_SHA = '26c9fe7ee85fb7760e70bfa29da87533cd1266d7d147e2e60c1fa543355c919c'
ENDPOINT = 'https://open.bigmodel.cn/api/coding/paas/v4/'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def enc(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(enc(obj))
    tmp.replace(path)

def immutable(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Frozen artifact differs: '+str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(raw)

def contract():
    path = Path(__file__).with_name('stage1_contract.py')
    if sha(path.read_bytes()) != CONTRACT_SHA:
        raise ValueError('Original extraction contract changed')
    spec = importlib.util.spec_from_file_location('csbench_pe_contract', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def prepare(project, experiment):
    m = contract()
    source = project/'data/csbench_within_question_v1'
    exp = project/'runs/online_experiments'/experiment
    visual = exp/'visual'
    out = exp/'scale_stage1'
    sm = read(source/'manifest.json')
    vs = read(visual/'experiment_summary.json')
    if vs['experiment_id'] != experiment or not vs['inputs_complete'] or vs['pending']:
        raise ValueError('Visual input incomplete or wrong experiment')
    if sm['counts'] != {'train':345,'validation':343,'test':2723}:
        raise ValueError('Unexpected frozen CSBench split')
    name = 'train.inputs.jsonl'
    original = source/name
    derived = visual/'text_view'/name
    label_path = source/'train.labels.jsonl'
    for path in (original,label_path):
        if sha(path.read_bytes()) != sm['files_sha256'][path.name]:
            raise ValueError('Frozen source changed: '+str(path))
    if sha(derived.read_bytes()) != vs['files_sha256'][name]:
        raise ValueError('Visual training view hash mismatch')
    rows, old, labels = lines(derived), lines(original), lines(label_path)
    if len(rows)!=345 or len(old)!=345 or len(labels)!=345:
        raise ValueError('Training coverage mismatch')
    ids = [r['sample_id'] for r in rows]
    gold = {r['sample_id']:r for r in labels}
    if len(set(ids))!=345 or len(gold)!=345 or set(ids)!=set(gold) or ids != [r['sample_id'] for r in old]:
        raise ValueError('Training identities changed')
    quads=[]
    for r,o in zip(rows,old):
        g=gold[r['sample_id']]
        if r['question_id']!=o['question_id'] or r['question_id']!=g['question_id']:
            raise ValueError('Question identity mismatch')
        if r['blank_passthrough']!=o['blank_passthrough'] or (o['blank_passthrough'] and r['answer']!=o['answer']):
            raise ValueError('Blank source changed')
        if not isinstance(r['answer'],str) or not isinstance(r['references'],list) or any(not isinstance(x,str) for x in r['references']):
            raise ValueError('Invalid source text schema')
        numbers=[g[k] for k in ('score_original','score_max_original','score_0_10')]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in numbers):
            raise ValueError('Nonfinite label')
        score,maximum,label=numbers
        if not maximum>0 or not 0<=score<=maximum or not math.isclose(label,10*score/maximum,abs_tol=1e-10):
            raise ValueError('Label scale mismatch')
        quads.append(dict(sample_id=r['sample_id'],question_id=r['question_id'],q=r['question'],r=r['references'],a=r['answer'],l=label,
                          score_original=score,score_max_original=maximum))
    units=m.make_units(rows)
    if sum(u['role']=='student' for u in units)!=345:
        raise ValueError('Training units mismatch')
    config=dict(experiment_id=experiment,model='glm-5.3',base_url=ENDPOINT,max_tokens=8192,
        prompt_sha256=sha(m.PROMPT.encode()),contract_sha256=CONTRACT_SHA,
        runner_sha256=sha(Path(__file__).read_bytes()),graph_mode='prose',
        source_manifest_sha256=sha((source/'manifest.json').read_bytes()),
        original_train_sha256=sha(original.read_bytes()),visual_train_sha256=sha(derived.read_bytes()),
        train_labels_sha256=sha(label_path.read_bytes()),visual_summary_sha256=sha((visual/'experiment_summary.json').read_bytes()),
        request_policy='Original prompt and text-only user payload; stream false; provider defaults otherwise',
        parser_policy='strict JSON; exact source span validation; no automatic quote or answer repair',
        score_policy='Use existing score_0_10; retain original score and maximum',
        limitations=['Independent SCALE reconstruction; GLM replaces paper generator',
                     'Existing prose graph mode retained; paper equation ambiguity not resolved',
                     'Visual transcription is model-generated and not semantically verified',
                     '8192 output token cap is an execution choice, not an author-verified paper parameter'],
        training_records=345,validation_inputs_read=False,test_inputs_read=False,
        validation_labels_read=False,test_labels_read=False)
    artifacts={'training_sources.json':enc(quads),'source_units.jsonl':m.jsonlines(units),'config.json':enc(config)}
    # Precheck all immutable artifacts before creating any of them.
    for name,raw in artifacts.items():
        if (out/name).exists() and (out/name).read_bytes()!=raw:
            raise ValueError('Resume inputs changed: '+name)
    for name,raw in artifacts.items():
        immutable(out/name,raw)
    train_visual_ids={'answer:'+r['sample_id'] for r in rows}
    train_questions={r['question_id'] for r in rows}
    issues=[x for x in vs.get('reported_issues',[]) if x['unit_id'] in train_visual_ids or any(x['unit_id']==kind+':'+q for q in train_questions for kind in ('question_image','reference_image'))]
    warnings=dict(status='EXPERIMENTAL_INPUT_QUALITY_NOT_VERIFIED',reported_issues=issues,
        missing_visual_context=[x for x in vs.get('missing_visual_context',[]) if x['question_id'] in train_questions],
        policy='Warnings retained; no automatic removal, correction, relabeling or formal-result claim')
    immutable(out/'input_quality.json',enc(warnings))
    report=dict(status='CSBENCH_TRAIN_UNITS_READY',training_records=345,unit_count=len(units),
        blank_units=sum(not u['text'].strip() for u in units),questions=len(train_questions),
        api_calls=0,training_started=False,test_labels_read=False,output_directory=str(out))
    save(out/'prepare_summary.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return m,out,units,config

def parse(m,unit,envelope):
    choice=envelope['choices'][0]
    if choice.get('finish_reason')!='stop':
        raise ValueError('Response not complete')
    obj=json.loads(choice['message']['content'])
    if not isinstance(obj,dict) or set(obj)!={'triples'}:
        raise ValueError('Expected triples object')
    return m.validate([unit],[dict(unit_id=unit['unit_id'],triples=obj['triples'])])[0]

def execute(project,experiment,attempts=8,limit=0,client=None,sleep=time.sleep):
    m,out,units,cfg=prepare(project,experiment)
    fingerprint=sha(enc(cfg))
    valid=[];errors=[];calls=0;processed=0
    def report(status):
        result=dict(status=status,experiment_id=experiment,training_records=345,units_total=len(units),
            units_valid=len(valid),remaining=len(units)-len(valid),new_api_calls=calls,errors=errors,
            config_sha256=fingerprint,semantic_accuracy_verified=False,experiment_class='experimental',
            validation_labels_read=False,test_labels_read=False,training_started=False)
        save(out/'summary.json',result)
        return result
    fatal=False
    for unit in units:
        uid=unit['unit_id']
        folder=out/'responses'/sha(uid.encode())
        folder.mkdir(parents=True,exist_ok=True)
        accepted=folder/'accepted.json'
        if not unit['text'].strip():
            valid.append(dict(unit_id=uid,triples=[]));continue
        if accepted.exists():
            saved=read(accepted)
            rawpath=folder/saved['response_file']
            if saved['unit_id']!=uid or saved['config_sha256']!=fingerprint or sha(rawpath.read_bytes())!=saved['response_sha256']:
                raise ValueError('Accepted cache mismatch: '+uid)
            envelope=read(rawpath)
            if envelope['unit_id']!=uid or envelope['config_sha256']!=fingerprint:
                raise ValueError('Raw cache provenance mismatch: '+uid)
            valid.append(parse(m,unit,envelope['response']));continue
        recovered=False
        for path in sorted(folder.glob('*.response.json'),reverse=True):
            try:
                envelope=read(path)
                if envelope['config_sha256']!=fingerprint or envelope['unit_id']!=uid:
                    raise RuntimeError('Raw cache provenance mismatch')
                parsed=parse(m,unit,envelope['response'])
            except (ValueError,KeyError,TypeError,IndexError):
                continue
            # accepted points to the original saved API envelope, never copied across experiments.
            save(accepted,dict(unit_id=uid,config_sha256=fingerprint,response_file=path.name,response_sha256=sha(path.read_bytes()),wrapped=True))
            valid.append(parsed);recovered=True;break
        if recovered:continue
        if fatal or (limit and processed>=limit):continue
        # Permanent service errors are not automatically resubmitted on resume.
        blocked=[read(p) for p in sorted(folder.glob('*.error.json'))]
        blocked=[e for e in blocked if e.get('http_status') in (400,401,403,404,422)]
        if blocked:
            errors.append(blocked[-1]);continue
        if client is None:
            from openai import OpenAI
            key=os.environ.get('ZHIPUAI_API_KEY')
            if not key:raise ValueError('ZHIPUAI_API_KEY is required')
            client=OpenAI(api_key=key,base_url=ENDPOINT,timeout=900,max_retries=0)
        processed+=1
        for attempt in range(1,attempts+1):
            token=str(time.time_ns())
            try:
                calls+=1
                response=client.chat.completions.create(model=cfg['model'],max_tokens=cfg['max_tokens'],stream=False,
                    messages=[{'role':'system','content':m.PROMPT},{'role':'user','content':json.dumps({'text':unit['text']},ensure_ascii=False)}]).model_dump(mode='json')
                path=folder/(token+'.response.json')
                save(path,dict(unit_id=uid,config_sha256=fingerprint,response=response))
                parsed=parse(m,unit,response)
                save(accepted,dict(unit_id=uid,config_sha256=fingerprint,response_file=path.name,response_sha256=sha(path.read_bytes()),wrapped=True))
                valid.append(parsed);break
            except Exception as exc:
                status=getattr(exc,'status_code',None)
                err=dict(unit_id=uid,attempt=attempt,exception_type=type(exc).__name__,http_status=status)
                body=getattr(exc,'body',None)
                if isinstance(body,dict):
                    detail=body.get('error',body)
                    if isinstance(detail,dict) and isinstance(detail.get('code'),(int,str)):
                        err['service_code']=str(detail['code'])[:80]
                save(folder/(token+'.error.json'),err)
                print('PE_ERROR',json.dumps(err),flush=True)
                if status in (401,403,404,422):fatal=True
                if status==400 or fatal or attempt==attempts:
                    errors.append(err);break
                sleep(min(60,5*2**(attempt-1)))
        print('PE',len(valid),'/',len(units),'CALLS',calls,flush=True)
        report('PE_RUNNING')
    if len(valid)!=len(units):
        print(json.dumps(report('PE_PARTIAL'),ensure_ascii=False),flush=True)
        return 2
    valid=m.validate(units,valid)
    raw=m.jsonlines(valid)
    graph=m.graph(valid,cfg['graph_mode'])
    graph.update(scope='CSBench training-only; 345 student answers and their references',config_sha256=fingerprint,extractions_sha256=sha(raw))
    immutable(out/'extractions.jsonl',raw)
    immutable(out/'graph.json',enc(graph))
    result=report('CSBENCH_TRAIN_EXTRACTION_AND_GRAPH_COMPLETE')
    result['files_sha256']={n:sha((out/n).read_bytes()) for n in ('extractions.jsonl','graph.json','training_sources.json','source_units.jsonl')}
    result['graph_nodes']=len(graph['nodes']);result['graph_edges']=len(graph['edges'])
    save(out/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 0

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--experiment-id',default='csbench_scale_online_v1')
    p.add_argument('--attempts',type=int,default=8)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}',a.experiment_id) or a.attempts<1 or a.limit<0:p.error('Invalid arguments')
    project=a.project.resolve()
    out=project/'runs/online_experiments'/a.experiment_id/'scale_stage1'
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.command=='prepare':prepare(project,a.experiment_id);return
        raise SystemExit(execute(project,a.experiment_id,a.attempts,a.limit))

if __name__=='__main__':main()
