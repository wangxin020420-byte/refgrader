"""Frozen-model validation perturbation audit. No training or test input."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import os
import statistics


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]


def design(inputs, labels):
    indexed = {r['sample_id']: r for r in inputs}
    gold = {r['sample_id']: r for r in labels}
    if len(indexed) != len(inputs) or len(gold) != len(labels) or indexed.keys() != gold.keys():
        raise ValueError('Input/label IDs mismatch or duplicates')
    ordered = sorted(inputs, key=lambda r:r['sample_id'])
    groups = defaultdict(list)
    for row in ordered:
        if gold[row['sample_id']]['question_id'] != row['question_id']:
            raise ValueError('Question ID mismatch')
        groups[row['question_id']].append(row)
    donors = {}
    for group in groups.values():
        if len(group) < 2:
            raise ValueError('At least two answers per question needed')
        for index, row in enumerate(group):
            donors[row['sample_id']] = group[(index + 1) % len(group)]
    result = []
    for index, row in enumerate(ordered):
        cross = next((ordered[(index+offset) % len(ordered)] for offset in range(1,len(ordered))
                      if ordered[(index+offset) % len(ordered)]['question_id'] != row['question_id']), None)
        if cross is None:
            raise ValueError('Need multiple questions')
        same = donors[row['sample_id']]
        result.append({'sample_id':row['sample_id'], 'question_id':row['question_id'],
                       'question':row['question'], 'gold_original':gold[row['sample_id']]['gold_raw'],
                       'answers':{'original':row['answer'], 'empty':'', 'same_question':same['answer'],
                                  'cross_question':cross['answer']},
                       'same_donor_id':same['sample_id'], 'same_donor_gold':gold[same['sample_id']]['gold_raw'],
                       'same_exact_text':row['answer']==same['answer'], 'cross_donor_id':cross['sample_id'],
                       'cross_donor_question_id':cross['question_id']})
    return result


def statistics_for(records):
    report = {}
    for condition in ['original', 'empty', 'same_question', 'cross_question']:
        values = [r['scores'][condition]['clipped'] for r in records]
        raw = [r['scores'][condition]['unclipped'] for r in records]
        delta = [r['scores'][condition]['clipped']-r['scores']['original']['clipped'] for r in records]
        report[condition] = {'mean_score':statistics.mean(values), 'mean_change_vs_original':statistics.mean(delta),
                             'mean_absolute_change':statistics.mean(abs(x) for x in delta),
                             'score_ge_8_count':sum(x >= 8 for x in values),
                             'outside_range_count':sum(x<0 or x>10 for x in raw), 'n':len(records)}
    comparable = [r for r in records if r['same_donor_gold'] != r['gold_original'] and not r['same_exact_text']]
    correct = sum((r['same_donor_gold']-r['gold_original'])*
                  (r['scores']['same_question']['clipped']-r['scores']['original']['clipped']) > 0 for r in comparable)
    report['same_question_direction'] = {'n':len(comparable), 'matching_sign_count':correct,
                'fraction':correct/len(comparable) if comparable else None,
                'note':'Uses original dataset labels; tied predictions are not counted as matching. Not independent label verification.'}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    data = root/'data/prepared/ads_public_adapted_v1'
    paths = [data/'validation.inputs.jsonl', data/'validation.labels.jsonl']
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf8'))
    for p in paths:
        if digest(p) != manifest['outputs_sha256'][p.name]:
            raise ValueError('Changed validation input')
    cases = design(read(paths[0]), read(paths[1]))
    if len(cases) != 222:
        raise ValueError('Expected 222 validation cases')
    runs = []
    for seed in [42,43,44,45,46]:
        for group in 'ABC':
            folder = 'abc_pilot_v1' if seed==42 else 'abc_followup_v1'
            run = root/'runs'/folder/('%s_seed%d' % (group,seed))
            for name in ['config.json','complete.json','best.pt']:
                if not (run/name).is_file():
                    raise ValueError('Missing artifact: '+str(run/name))
            complete = json.loads((run/'complete.json').read_text(encoding='utf8'))
            step = complete['best']['step']
            pred_path=run/('validation_step_%03d.json'%step)
            if not pred_path.is_file():
                raise ValueError('Missing original predictions')
            runs.append((seed,group,run,step,pred_path))
    if args.check:
        print('SENSITIVITY_PREFLIGHT_OK: 15 checkpoints, 222 cases, 4 conditions; no inference/training.')
        return
    if args.device=='cpu':
        os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    import transformers
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    if args.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<2*1024**3):
        raise ValueError('CUDA unavailable or less than 2GiB free for inference')
    lock=json.loads((root/'sources/bert-base-chinese.lock.json').read_text(encoding='utf8'))
    model_dir=root/'models/bert-base-chinese'/lock['revision']
    tokenizer=AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False,
                                           clean_up_tokenization_spaces=False)
    output=root/'runs/abc_answer_sensitivity_v1';output.mkdir(parents=True,exist_ok=True)
    guard=output/'RUNNING.lock'
    with guard.open('x') as f:f.write(str(os.getpid()))
    summaries=[]
    try:
        for seed,group,run,step,pred_path in runs:
            cfg=json.loads((run/'config.json').read_text(encoding='utf8'))
            if cfg['seed']!=seed or cfg['group']!=group or cfg['model_revision']!=lock['revision']:
                raise ValueError('Model identity mismatch')
            identity={'seed':seed,'group':group,'step':step,'device':args.device,
                      'script_sha256':digest(Path(__file__)), 'best_sha256':digest(run/'best.pt'),
                      'prediction_sha256':digest(pred_path),'inputs_sha256':[digest(p) for p in paths],
                      'torch':str(torch.__version__),'transformers':transformers.__version__}
            target=output/('%s_seed%d.json'%(group,seed))
            if target.exists():
                existing=json.loads(target.read_text(encoding='utf8'))
                if existing['identity']!=identity:raise ValueError('Existing audit identity differs')
                summaries.append({'seed':seed,'group':group,'summary':existing['summary']})
                print('Already done:',group,seed,flush=True);continue
            # Own server-created checkpoints only. Contains model states and metadata, no downloaded pickle.
            saved=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
            if saved['config']!=cfg or saved['step']!=step:raise ValueError('Checkpoint/config mismatch')
            encoder=AutoModel.from_config(AutoConfig.from_pretrained(str(model_dir),local_files_only=True),trust_remote_code=False)
            head=torch.nn.Linear(encoder.config.hidden_size,1)
            encoder.load_state_dict(saved['encoder']);head.load_state_dict(saved['head']);del saved
            encoder.to(args.device).eval();head.to(args.device).eval()
            def predict(condition):
                values=[]
                with torch.inference_mode():
                    for offset in range(0,len(cases),2):
                        batch=cases[offset:offset+2]
                        tokens=tokenizer([r['question'] for r in batch],[r['answers'][condition] for r in batch],
                              return_tensors='pt',padding=True,truncation=True,max_length=256).to(args.device)
                        raw=(head(encoder(**tokens).last_hidden_state[:,0,:]).squeeze(-1)*10).cpu().tolist()
                        values.extend(raw)
                if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite prediction')
                return values
            predictions={'original':predict('original')}
            prior_rows=json.loads(pred_path.read_text(encoding='utf8'))['predictions']
            prior={r['sample_id']:r for r in prior_rows}
            if len(prior)!=len(prior_rows) or set(prior)!={r['sample_id'] for r in cases}:
                raise ValueError('Saved prediction IDs mismatch')
            difference=max(abs(v-prior[r['sample_id']]['prediction_unclipped']) for r,v in zip(cases,predictions['original']))
            if difference>0.001:
                raise ValueError('Original prediction reproduction differs by %.6f; stop audit'%difference)
            for condition in ['empty','same_question','cross_question']:
                predictions[condition]=predict(condition)
            records=[]
            for index,case in enumerate(cases):
                scores={k:{'unclipped':v[index],'clipped':min(10.,max(0.,v[index]))} for k,v in predictions.items()}
                records.append(dict(case,scores=scores))
            summary=statistics_for(records)
            result={'identity':identity,'status':'FROZEN_MODEL_SENSITIVITY_COMPLETE',
                    'original_max_abs_difference':difference,'summary':summary,'samples':records,
                    'training_started':False,'test_accessed':False,'perturbed_labels_assigned':False}
            temp=target.with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');temp.replace(target)
            summaries.append({'seed':seed,'group':group,'summary':summary})
            del encoder,head
            if args.device=='cuda':torch.cuda.empty_cache()
            print('Completed:',group,seed,flush=True)
        report={'status':'ALL_15_SENSITIVITY_AUDITS_COMPLETE','runs':summaries,
                'note':'Blank/cross-question scores are diagnostic, not accuracy with invented labels. Lexical overlap and tokenization changes can affect perturbations.'}
        path=root/'reports/abc_answer_sensitivity_summary_v1.json'
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print('SUMMARY:',path)
    finally:
        guard.unlink()


if __name__=='__main__':main()
