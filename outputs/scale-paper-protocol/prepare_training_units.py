"""Prepare training-only Pe units for the restored ADS split; no API calls."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def lines(p):
    return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]


def execute(root):
    data = root/'data/prepared/ads_paper_split_v1_seed42'
    manifest = read(data/'manifest.json')
    if manifest['counts'] != {'train':1107,'validation':50,'test':425}:
        raise ValueError('Unexpected split')
    for name in ('train.inputs.jsonl','train.labels.jsonl'):
        if sha(data/name) != manifest['outputs_sha256'][name]:
            raise ValueError('Changed training input: '+name)
    rows = lines(data/'train.inputs.jsonl')
    labels = lines(data/'train.labels.jsonl')
    ids = {r['sample_id'] for r in rows}
    if len(rows)!=1107 or len(ids)!=1107 or len(labels)!=1107 or len({r['sample_id'] for r in labels})!=1107 or ids!={r['sample_id'] for r in labels}:
        raise ValueError('Training identity mismatch')
    module_path = root/'control/scale-paper-stage1/stage1.py'
    spec = importlib.util.spec_from_file_location('pe_contract',module_path)
    stage = importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
    units = stage.make_units(rows)
    if sum(u['role']=='student' for u in units)!=1107:
        raise ValueError('Unit construction count mismatch')
    old = root/'paper_method_v1/artifacts/triplets'
    old_manifest = read(old/'stage1_manifest.json')
    old_path = old/'source_units.jsonl'
    if sha(old_path)!=old_manifest['outputs_sha256']['source_units.jsonl']:
        raise ValueError('Old unit file changed')
    old_units = {u['unit_id']:u for u in lines(old_path)}
    candidates = []
    for unit in units:
        previous = old_units.get(unit['unit_id'])
        same = previous is not None and all(previous[k]==unit[k] for k in ('text','role','question_id'))
        cache = root/'paper_method_v1/runs/pe_glm_full_v1/cache'/(
            hashlib.sha256(unit['unit_id'].encode('utf8')).hexdigest()+'.json')
        candidates.append({'unit_id':unit['unit_id'],'exact_old_input_match':same,
                           'old_response_exists':cache.is_file() if same else False,
                           'reuse_approved':False,
                           'reason':'Input match alone does not verify model, request configuration, prompt or response validity'})
    payload = {'source_units.jsonl':stage.jsonlines(units),
               'cache_candidates.jsonl':stage.jsonlines(candidates)}
    report = {'status':'RESTORED_TRAIN_UNITS_READY_MODEL_CONFIG_PENDING',
              'training_records':1107,'unit_counts':dict(Counter(u['role'] for u in units)),
              'blank_student_units':sum(u['role']=='student' and not u['text'].strip() for u in units),
              'old_input_matches':sum(c['exact_old_input_match'] for c in candidates),
              'matching_input_with_old_response':sum(c['old_response_exists'] for c in candidates),
              'cache_reused':0,'api_calls':0,'training_started':False,'test_files_read':False,
              'validation_files_read':False,'model_configuration_frozen':False,
              'source_manifest_sha256':sha(data/'manifest.json'),
              'source_train_sha256':sha(data/'train.inputs.jsonl'),
              'stage1_code_sha256':sha(module_path),'builder_sha256':sha(Path(__file__)),
              'prompt_status':'Existing stage1 prompt is independently reconstructed; original-paper prompt equivalence not verified',
              'outputs_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in payload.items()},
              'next_step':'Freeze generator endpoint/model and prompt; then validate reusable caches and extract missing units'}
    payload['manifest.json'] = stage.encoded(report)
    out = root/'paper_method_v2/artifacts/pe_inputs_v1'
    stage.write_all(out,payload)
    for name,raw in payload.items():
        if (out/name).read_bytes()!=raw:
            raise ValueError('Output verification failed')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('OUTPUT_DIRECTORY:',out)
    print('RESTORED_TRAIN_UNITS_PREPARE_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    root=a.root.resolve()
    lock_dir=root/'paper_method_v2/artifacts/pe_inputs_v1'
    lock_dir.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (lock_dir/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(root)
