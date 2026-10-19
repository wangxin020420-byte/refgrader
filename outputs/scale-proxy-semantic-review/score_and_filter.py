"""Offline candidate scoring, training-only diagnostics and two provisional beta branches."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def enc(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf8')


def jl(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n' for r in rows).encode('utf8')


def read(p):
    return json.loads(p.read_bytes())


def lines(p):
    return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]


def write(p, data, frozen=False):
    if frozen and p.exists():
        if p.read_bytes() != data:
            raise ValueError('Existing artifact differs: '+str(p))
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    with temp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    temp.replace(p)


def references(row):
    refs = row['r']
    if not isinstance(refs, list) or not all(isinstance(x, str) for x in refs):
        raise ValueError('Invalid references')
    value = '\n'.join(x.strip() for x in refs if x.strip())
    if not value:
        raise ValueError('No reference text')
    return value


def stats(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return {'count': len(values), 'min': values[0], 'max': values[-1],
            'mean': sum(values)/len(values),
            'median': (values[(len(values)-1)//2]+values[len(values)//2])/2,
            'outside_0_10': sum(x < 0 or x > 10 for x in values)}


def make_jobs(candidates, sources, proxies):
    lookup = {s['sample_id']: s for s in sources}
    if len(lookup) != len(sources):
        raise ValueError('Duplicate original source')
    jobs = []
    for c in candidates:
        source = lookup[c['sample_id']]
        if c.get('assigned_label') is not None or not isinstance(c['answer'], str) or not c['answer'].strip():
            raise ValueError('Invalid unlabeled candidate')
        jobs.append({'id': c['variant_id'], 'kind': 'candidate', 'sample_id': c['sample_id'],
                     'reference': references(source), 'answer': c['answer'], 'source_score': source['l']})
    for p in proxies:
        source = lookup[p['sample_id']]
        if p['source_score'] != source['l'] or p['references'] != source['r']:
            raise ValueError('Proxy training provenance differs')
        for kind, answer in [('proxy_train', p['answer']), ('original_train', source['a'])]:
            jobs.append({'id': kind+':'+p['sample_id'], 'kind': kind, 'sample_id': p['sample_id'],
                         'reference': references(source), 'answer': answer, 'source_score': source['l']})
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate scoring ID')
    return jobs


def validate_chunk(chunk, expected_jobs, config_hash):
    if chunk['config_sha256'] != config_hash or chunk['jobs_sha256'] != digest(enc(expected_jobs)):
        raise ValueError('Cached chunk input mismatch')
    result = chunk['results']
    if chunk['results_sha256'] != digest(enc(result)) or len(result) != len(expected_jobs):
        raise ValueError('Cached chunk output mismatch')
    for row, job in zip(result, expected_jobs):
        if row['id'] != job['id'] or not math.isfinite(row['score_raw']):
            raise ValueError('Invalid cached prediction')
    return result


def branch(scored, sources, limit):
    kept = [r for r in scored if r['absolute_gap_raw'] <= limit]
    augmented = [{'id': r['sample_id'], 'sample_id': r['sample_id'], 'question_id': r['question_id'],
        'question': r['q'], 'references': r['r'], 'answer': r['a'], 'label': r['l'],
        'kind': 'original_training', 'label_origin': 'original_gold'} for r in sources]
    lookup = {r['sample_id']: r for r in sources}
    for r in kept:
        source = lookup[r['sample_id']]
        augmented.append({'id': r['variant_id'], 'sample_id': r['sample_id'],
            'question_id': source['question_id'], 'question': source['q'], 'references': source['r'],
            'answer': r['answer'], 'label': r['assigned_label'], 'kind': 'synthetic_candidate',
            'label_origin': 'annotator_raw_regression', 'original_score': source['l']})
    return kept, augmented


def run(args, out):
    root = args.root.resolve(); base = root/'paper_method_v1'
    model_run = base/'runs/annotator_bert_v1'; final = model_run/'model_final'
    complete = read(model_run/'complete.json'); trained = read(model_run/'config.json')
    if complete['status'] != 'ANNOTATOR_TRAINING_COMPLETE_FIXED_EPOCH' or complete['completed_epochs'] != 10:
        raise ValueError('Expected completed ten-epoch annotator')
    if complete['config_sha256'] != sha(model_run/'config.json'):
        raise ValueError('Training config mismatch')
    if trained['max_length'] != 512 or trained['loss'] != 'MSE on original ADS 0-10 scores; no normalization or clipping':
        raise ValueError('Unexpected annotator input/score protocol')
    if trained['reference_policy'] != 'Join all nonempty references in original slot order with newline; one row per source.':
        raise ValueError('Unexpected reference protocol')
    for name, h in complete['final_files_sha256'].items():
        if sha(final/name) != h:
            raise ValueError('Final model file changed: '+name)
    selection = base/'artifacts/initial_filter_k5_b5000_v1'; selected_report = read(selection/'summary.json')
    if selected_report['status'] != 'INITIAL_FILTER_COMPLETE_UNLABELED':
        raise ValueError('Initial selection incomplete')
    if sha(selection/'config.json') != selected_report['config_sha256']:
        raise ValueError('Selection config mismatch')
    for name, h in selected_report['files_sha256'].items():
        if sha(selection/name) != h:
            raise ValueError('Selection artifact changed: '+name)
    source_path = base/'artifacts/proxy_inputs_v1/training_sources.json'
    if sha(source_path) != '5dcff9d092ee292d5be2bfbe9bcafed2fb6fcd1cc82e02c0e2acbff7ae64fb5d':
        raise ValueError('Original training sources changed')
    proxy_path = base/'artifacts/annotator_inputs_v1/train.jsonl'
    if sha(proxy_path) != trained['data_sha256']:
        raise ValueError('Annotator input data changed')
    sources = read(source_path); candidates = lines(selection/'selected_candidates.jsonl'); proxies = lines(proxy_path)
    if len(sources) != 885 or len(candidates) != 5000 or len(proxies) != 885:
        raise ValueError('Unexpected record counts')
    if len({p['sample_id'] for p in proxies}) != 885 or {p['sample_id'] for p in proxies} != {s['sample_id'] for s in sources}:
        raise ValueError('Proxy coverage mismatch')
    jobs = make_jobs(candidates, sources, proxies)
    config = {'script_sha256': sha(Path(__file__)), 'annotator_complete_sha256': sha(model_run/'complete.json'),
        'selected_candidates_sha256': sha(selection/'selected_candidates.jsonl'),
        'original_sources_sha256': sha(source_path), 'proxy_train_sha256': sha(proxy_path),
        'jobs_sha256': digest(enc(jobs)), 'score_policy': 'raw FP32 regression; no rounding or clipping',
        'tokenization': 'same references + answer pair; right longest_first truncation 512',
        'chunk_size': 100, 'initial_batch_size': 8, 'oom_policy': 'halve inference batch; minimum one',
        'beta_branches': {'raw_beta_0p2': 0.2, 'normalized_beta_0p2': 2.0},
        'official_branch_selected': None, 'api_calls': 0, 'test_accessed': False}
    write(out/'config.json', enc(config), frozen=True); config_hash = sha(out/'config.json')
    if (out/'summary.json').exists():
        done = read(out/'summary.json')
        if done['config_sha256'] != config_hash:
            raise ValueError('Completed scoring config differs')
        for name, h in done['files_sha256'].items():
            if sha(out/name) != h:
                raise ValueError('Completed scoring artifact differs: '+name)
        print('ALREADY_COMPLETE: results verified; no inference repeated.', flush=True)
        print(json.dumps(done, ensure_ascii=False, indent=2)); return
    print('INPUTS_VERIFIED: 5000 candidates + 885 proxy-training + 885 original-training answers', flush=True)
    print('OFFLINE; no validation/test data; no new training; no API calls', flush=True)
    predictions = []; model = None; tokenizer = None; batch_size = 8; started = time.time()
    for start in range(0, len(jobs), 100):
        group = jobs[start:start+100]; path = out/'chunks'/('%05d.json'%start)
        if path.exists():
            result = validate_chunk(read(path), group, config_hash)
        else:
            if model is None:
                import torch
                import transformers
                torch.set_num_threads(4)
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA unavailable on requested GPU')
                tokenizer = transformers.AutoTokenizer.from_pretrained(str(final), local_files_only=True)
                tokenizer.truncation_side = 'right'; tokenizer.padding_side = 'right'
                model = transformers.BertForSequenceClassification.from_pretrained(str(final), local_files_only=True)
                if model.config.num_labels != 1:
                    raise ValueError('Expected regression model')
                model.to('cuda:0'); model.eval()
            result = []; cursor = 0
            while cursor < len(group):
                batch = group[cursor:cursor+batch_size]
                batch_tensors = None; output = None
                try:
                    full_lengths = [len(tokenizer(j['reference'], j['answer'], truncation=False, verbose=False)['input_ids']) for j in batch]
                    batch_tensors = tokenizer([j['reference'] for j in batch], [j['answer'] for j in batch],
                        padding=True, truncation='longest_first', max_length=512, return_tensors='pt')
                    batch_tensors = {k: v.to('cuda:0') for k, v in batch_tensors.items()}
                    with torch.inference_mode():
                        output = model(**batch_tensors).logits.reshape(-1)
                        scores = output.float().cpu().tolist()
                    if len(scores) != len(batch) or not all(math.isfinite(x) for x in scores):
                        raise ValueError('Nonfinite or incomplete prediction')
                    result.extend({'id': j['id'], 'score_raw': score, 'full_tokens': length,
                                   'truncated': length > 512} for j, score, length in zip(batch, scores, full_lengths))
                    cursor += len(batch)
                except torch.OutOfMemoryError:
                    if batch_size == 1:
                        raise
                    batch_tensors = None; output = None
                    torch.cuda.empty_cache(); batch_size = max(1, batch_size//2)
                    print('OOM: reducing inference batch to', batch_size, flush=True)
            chunk = {'config_sha256': config_hash, 'jobs_sha256': digest(enc(group)),
                     'results_sha256': digest(enc(result)), 'results': result}
            write(path, enc(chunk), frozen=True)
        predictions.extend(result)
        progress = {'status': 'SCORING', 'finished': len(predictions), 'total': len(jobs),
                    'elapsed_seconds': round(time.time()-started, 1)}
        write(out/'progress.json', enc(progress))
        print('SCORED %d/%d elapsed=%.1fs'%(len(predictions), len(jobs), time.time()-started), flush=True)
    by_id = {r['id']: r for r in predictions}; source_map = {s['sample_id']: s for s in sources}
    scored = []
    for c in candidates:
        score = by_id[c['variant_id']]['score_raw']; original = source_map[c['sample_id']]['l']
        scored.append(dict(c, assigned_label=score, annotator_score_raw=score, original_score=original,
            absolute_gap_raw=abs(score-original), absolute_gap_normalized=abs(score-original)/10.0,
            score_outside_0_10=score < 0 or score > 10, label_origin='annotator_bert_v1_model_final'))
    files = {}
    def export(name, data):
        write(out/name, data, frozen=True); files[name] = sha(out/name)
    export('scored_candidates.jsonl', jl(scored))
    diagnostics = {'candidate_predictions': stats([r['assigned_label'] for r in scored]),
                   'candidate_gaps_raw': stats([r['absolute_gap_raw'] for r in scored]),
                   'interpretation': 'Training-only diagnostics; not validation/test performance and not checkpoint selection.'}
    for kind in ('proxy_train', 'original_train'):
        subset = [j for j in jobs if j['kind'] == kind]
        diagnostic_rows = [dict(sample_id=j['sample_id'], original_score=j['source_score'],
            score_raw=by_id[j['id']]['score_raw']) for j in subset]
        mse = sum((r['score_raw']-r['original_score'])**2 for r in diagnostic_rows)/len(subset)
        diagnostics[kind] = {'records': len(subset), 'mse': mse, 'rmse': math.sqrt(mse)}
        export(kind+'_diagnostics.jsonl', jl(diagnostic_rows))
    export('diagnostics.json', enc(diagnostics))
    export('truncation_audit.json', enc([dict(job_id=p['id'], full_tokens=p['full_tokens']) for p in predictions if p['truncated']]))
    branches = {}
    for name, limit in [('raw_beta_0p2', 0.2), ('normalized_beta_0p2', 2.0)]:
        kept, augmented = branch(scored, sources, limit)
        export(name+'/selected.jsonl', jl(kept)); export(name+'/augmented_inputs.jsonl', jl(augmented))
        report = {'status': 'PROVISIONAL_BRANCH_EXPORTED_NOT_SELECTED', 'beta': 0.2,
            'equivalent_raw_gap_limit': limit, 'retained_synthetic': len(kept), 'discarded_synthetic': len(scored)-len(kept),
            'original_training_records': 885, 'augmented_records': len(augmented),
            'labels': 'original gold for real answers; raw annotator prediction for synthetics',
            'retained_out_of_range': sum(r['score_outside_0_10'] for r in kept),
            'retained_by_source_score': dict(Counter(str(r['original_score']) for r in kept)),
            'ready_for_scale_training': False,
            'reason': 'Beta scale unverified; synthetic entity features not prepared. No automatic branch selection.'}
        export(name+'/manifest.json', enc(report)); branches[name] = report
    summary = {'status': 'SCORING_AND_BETA_BRANCHES_COMPLETE', 'candidate_labels_assigned': 5000,
        'diagnostic_predictions': 1770, 'branches': branches, 'official_branch_selected': None,
        'config_sha256': config_hash, 'files_sha256': files, 'api_calls': 0,
        'new_training_started': False, 'validation_accessed': False, 'test_accessed': False,
        'reproduction_status': 'EXPERIMENTAL_REPRODUCTION_WITH_UNVERIFIED_BETA_SCALE'}
    write(out/'summary.json', enc(summary), frozen=True)
    write(out/'progress.json', enc({'status': 'COMPLETE', 'finished': len(jobs), 'total': len(jobs)}))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('\aPIPELINE_DONE: '+str(out/'summary.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--physical-gpu', type=int, default=1)
    args = p.parse_args()
    if args.physical_gpu < 0:
        p.error('physical-gpu must be nonnegative')
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.physical_gpu)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    out = args.root.resolve()/'paper_method_v1/runs/candidate_annotation_v1'
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args, out)
        except Exception as exc:
            write(out/'failure.json', enc({'status': 'FAILED', 'exception': type(exc).__name__, 'detail': str(exc)}))
            print('\aPIPELINE_FAILED: '+str(exc), flush=True)
            raise
