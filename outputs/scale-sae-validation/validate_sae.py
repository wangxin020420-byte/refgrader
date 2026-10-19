"""Frozen last-epoch SAE validation, resumable Pe/inference; never reads test rows."""
import argparse
from collections import Counter
import getpass
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(gold, predictions):
    if not gold or len(gold) != len(predictions):
        raise ValueError('Metric length mismatch')
    if any(not math.isfinite(x) for x in gold + predictions):
        raise ValueError('Nonfinite metric input')
    if any(x != int(x) or not 0 <= x <= 10 for x in gold):
        raise ValueError('Invalid ADS gold score')
    n = len(gold)
    clipped = [max(0.0, min(10.0, x)) for x in predictions]
    discrete = [math.floor(x + 0.5) for x in clipped]
    mg, mp = sum(gold)/n, sum(predictions)/n
    vg = sum((x-mg)**2 for x in gold)
    vp = sum((x-mp)**2 for x in predictions)
    pearson = sum((g-mg)*(p-mp) for g, p in zip(gold, predictions))/math.sqrt(vg*vp) if vg and vp else None
    a, b = Counter(gold), Counter(discrete)
    observed = sum((g-p)**2 for g, p in zip(gold, discrete))/n
    expected = sum(ng*np*(g-p)**2 for g, ng in a.items() for p, np in b.items())/(n*n)
    return {'records': n, 'pearson_raw': pearson,
            'rmse_raw': math.sqrt(sum((g-p)**2 for g, p in zip(gold, predictions))/n),
            'rmse_clipped': math.sqrt(sum((g-p)**2 for g, p in zip(gold, clipped))/n),
            'qwk_clipped_half_up': 1-observed/expected if expected else None,
            'out_of_range_predictions': sum(not 0 <= p <= 10 for p in predictions)}


def execute(args, ep, out):
    root = args.root.resolve()
    trained = root / 'paper_method_v1/runs/sae_raw_beta_seed42_v1'
    completed = ep.read(trained / 'complete.json')
    train_config = ep.read(trained / 'config.json')
    if completed['status'] != 'SAE_EXPERIMENTAL_TRAINING_COMPLETE':
        raise ValueError('Training not complete')
    if ep.sha((trained / 'config.json').read_bytes()) != completed['config_sha256']:
        raise ValueError('Training config changed')
    # Hash large weights using streaming helper.
    trainer_path = root / 'control/scale-sae-train/train_sae.py'
    trainer = load(trainer_path, 'frozen_sae_trainer')
    if trainer.sha(trainer_path) != train_config['script_sha256']:
        raise ValueError('Trained model implementation changed')
    for name, expected in completed['files_sha256'].items():
        if trainer.sha(trained / name) != expected:
            raise ValueError('Completed training file changed: ' + name)
    token_path = root / 'control/scale-sae-inputs/prepare_sae_inputs.py'
    prep = load(token_path, 'frozen_sae_tokenizer')
    input_folder = root / 'paper_method_v1/artifacts/sae_inputs_raw_beta_v1'
    input_config = ep.read(input_folder / 'config.json')
    if trainer.sha(input_folder / 'config.json') != train_config['input_config_sha256']:
        raise ValueError('Tokenization config changed')
    if trainer.sha(token_path) != input_config['script_sha256']:
        raise ValueError('Tokenizer implementation changed')
    source = root / 'data/prepared/ads_public_adapted_v1'
    manifest = ep.read(source / 'manifest.json')
    names = ['train.inputs.jsonl', 'validation.inputs.jsonl', 'validation.labels.jsonl']
    hashes = {n: trainer.sha(source / n) for n in names}
    if any(hashes[n] != manifest['outputs_sha256'][n] for n in names):
        raise ValueError('Prepared data hash mismatch')
    rows = ep.lines(source / 'validation.inputs.jsonl')
    original = ep.lines(source / 'train.inputs.jsonl')
    if len(rows) != 222 or len({r['sample_id'] for r in rows}) != 222:
        raise ValueError('Expected 222 unique validation records')
    if {r['sample_id'] for r in rows} & {r['sample_id'] for r in original}:
        raise ValueError('Train/validation ID overlap')
    if {(r['question_id'], r['answer']) for r in rows} & {(r['question_id'], r['answer']) for r in original}:
        raise ValueError('Original train/validation content overlap')
    pe_path = root / 'control/scale-paper-stage1/stage1.py'
    m = ep.module(pe_path)
    train_entities = root / 'paper_method_v1/runs/train_entities_raw_beta_v1'
    entity_summary = ep.read(train_entities / 'summary.json')
    entity_file = train_entities / 'training_with_entities.jsonl'
    if trainer.sha(entity_file) != input_config['entity_export_sha256']:
        raise ValueError('Training entity export changed')
    extraction_config = ep.read(train_entities / 'config.json')
    if ep.sha(m.PROMPT.encode('utf8')) != extraction_config['prompt_sha256']:
        raise ValueError('Entity prompt mismatch')
    training_rows = ep.lines(entity_file)
    reuse = {}
    for r in sorted(training_rows, key=lambda x: x['id']):
        reuse.setdefault(r['answer'], r['answer_triples'])
        for text, triples in zip(r['references'], r['reference_triples']):
            reuse.setdefault(text, triples)
    synthetic_keys = {(r['question_id'], r['answer']) for r in training_rows if r['kind'] == 'synthetic_candidate'}
    overlap_ids = [r['sample_id'] for r in rows if (r['question_id'], r['answer']) in synthetic_keys]
    protocol = {'status': 'FROZEN_VALIDATION_PROTOCOL', 'split': 'validation', 'records': 222,
        'data_sha256': hashes, 'training_config_sha256': completed['config_sha256'],
        'checkpoint_sha256': completed['files_sha256']['model_final.pt'],
        'checkpoint_selection': 'Previously fixed last epoch; no validation checkpoint selection',
        'input_config_sha256': train_config['input_config_sha256'],
        'entity_export_sha256': trainer.sha(entity_file),
        'pe_parser_sha256': trainer.sha(root / 'control/scale-synthetic-entities/entity_prepare.py'),
        'pe_stage1_sha256': trainer.sha(pe_path), 'prompt_sha256': extraction_config['prompt_sha256'],
        'model': extraction_config['model'], 'base_url': extraction_config['base_url'],
        'max_tokens': extraction_config['max_tokens'],
        'metric_policy': 'Pearson and RMSE on raw predictions; additionally clipped RMSE; QWK after clip 0..10 and floor(x+0.5)',
        'metric_policy_deviation': 'Explicit evaluation adapter: raw regression predictions may exceed prepared-data scorer range; raw outputs retained',
        'entity_reuse': 'Exact text only, first training row by ID; no validation labels in extraction',
        'synthetic_validation_exact_overlap_ids': overlap_ids,
        'test_accessed': False, 'training_modified': False,
        'experiment_status': 'EXPERIMENTAL_REPRODUCTION; adapted validation split, single seed, fixed epoch',
        'script_sha256': trainer.sha(Path(__file__))}
    ep.write(out / 'protocol.json', ep.enc(protocol), frozen=True)
    protocol_sha = trainer.sha(out / 'protocol.json')
    if (out / 'metrics.json').exists():
        result = ep.read(out / 'metrics.json')
        if result['protocol_sha256'] != protocol_sha:
            raise ValueError('Completed validation protocol mismatch')
        for n, value in result['files_sha256'].items():
            if trainer.sha(out / n) != value:
                raise ValueError('Validation artifact changed: ' + n)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print('VALIDATION_ALREADY_COMPLETE_VERIFIED')
        return 0
    units = {}
    for row in rows:
        for text in [row['answer']] + row['references']:
            uid = 'validation_text:' + ep.sha(text.encode('utf8'))
            units[uid] = {'unit_id': uid, 'text': text}
    ep.write(out / 'units.jsonl', ep.jl([units[k] for k in sorted(units)]), frozen=True)
    results, errors, calls = {}, [], 0
    key = None
    stopped = False
    for uid, unit in sorted(units.items()):
        text = unit['text']
        if text in reuse or not text.strip():
            triples = reuse.get(text, []) if text.strip() else []
            results[uid] = m.validate([unit], [{'unit_id': uid, 'triples': triples}])[0]
            continue
        folder = out / 'responses' / uid.split(':', 1)[1]
        problems = []
        for path in sorted(folder.glob('attempt_*.response.json')):
            try:
                results[uid] = ep.parse(m, unit, ep.read(path)); break
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                problems.append(str(exc))
        attempts = len(list(folder.glob('attempt_*.request.json')))
        while uid not in results and not stopped and attempts < args.max_attempts:
            if shutil.disk_usage(out).free < 1024**3:
                raise RuntimeError('Need at least 1 GiB free')
            if key is None:
                key = getpass.getpass('GLM API Key (hidden; validation entity extraction): ').strip()
                if not key:
                    raise ValueError('Empty key')
            body = {'model': protocol['model'], 'max_tokens': protocol['max_tokens'], 'stream': False,
                    'messages': [{'role': 'system', 'content': m.PROMPT},
                                 {'role': 'user', 'content': json.dumps({'text': text}, ensure_ascii=False)}]}
            attempts += 1
            prefix = folder / ('attempt_%03d' % attempts)
            ep.write(Path(str(prefix)+'.request.json'), ep.enc({'unit_id': uid, 'body': body}), frozen=True)
            request = urllib.request.Request(protocol['base_url'].rstrip('/')+'/chat/completions',
                data=ep.enc(body), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+key})
            calls += 1
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    raw = response.read()
                ep.write(Path(str(prefix)+'.response.json'), raw, frozen=True)
                results[uid] = ep.parse(m, unit, json.loads(raw))
                print('SAVED_VALIDATION_ENTITY', uid, flush=True)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                problems.append(str(exc))
            except (OSError, urllib.error.URLError) as exc:
                info = {'type': type(exc).__name__, 'http_status': getattr(exc, 'code', None)}
                if isinstance(exc, urllib.error.HTTPError):
                    info['body'] = exc.read().decode('utf8', errors='replace')
                ep.write(Path(str(prefix)+'.error.json'), ep.enc(info))
                problems.append(info)
                stopped = True
            time.sleep(1)
        if uid not in results:
            diagnostic = {'unit_id': uid, 'source_text': text, 'attempts': attempts, 'errors': problems}
            available = sorted(folder.glob('attempt_*.response.json'))
            if available:
                diagnostic['last_response_file'] = str(available[-1])
                try:
                    last = ep.read(available[-1])['choices'][0]
                    diagnostic['finish_reason'] = last.get('finish_reason')
                    diagnostic['raw_content'] = last['message']['content']
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    diagnostic['response_read_error'] = str(exc)
            errors.append(diagnostic)
    extraction_summary = {'status': 'VALIDATION_ENTITIES_COMPLETE' if not errors else 'VALIDATION_ENTITIES_PARTIAL',
                          'units': len(units), 'valid_units': len(results), 'errors': errors,
                          'new_api_calls': calls, 'test_accessed': False}
    ep.write(out / 'entity_summary.json', ep.enc(extraction_summary))
    print(json.dumps(extraction_summary, ensure_ascii=False, indent=2), flush=True)
    if errors:
        print('PARTIAL: see', out / 'entity_summary.json'); return 2
    ep.write(out / 'extractions.jsonl', ep.jl([results[k] for k in sorted(results)]), frozen=True)
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # Choose the GPU before importing torch/transformers. Wait applies only to resources.
    prediction_path = out / 'prediction_cache.json'
    saved = ep.read(prediction_path) if prediction_path.exists() else {'protocol_sha256': protocol_sha, 'predictions': []}
    predictions = saved['predictions']
    if saved['protocol_sha256'] != protocol_sha or len(predictions) > len(rows):
        raise ValueError('Prediction cache protocol/count mismatch')
    for cached, row in zip(predictions, rows):
        if cached['sample_id'] != row['sample_id'] or not math.isfinite(cached['prediction_raw']):
            raise ValueError('Prediction cache identity/score mismatch')
    if len(predictions) < len(rows):
        deadline = time.monotonic() + args.wait_hours*3600
        while True:
            try:
                trainer.select_gpu(args.physical_gpu); break
            except RuntimeError as exc:
                if not str(exc).startswith('No selected GPU has') or time.monotonic() >= deadline:
                    raise
                print('WAIT_GPU: retry in 60 seconds', flush=True); time.sleep(60)
        import torch
        import transformers
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model_path = root / 'models/bert-base-chinese' / trainer.REV
        for name, expected in train_config['model_files_sha256'].items():
            if trainer.sha(model_path / name) != expected:
                raise ValueError('Local model changed: ' + name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
        tokenizer.truncation_side = 'right'; tokenizer.padding_side = 'right'
        if ep.sha(tokenizer.backend_tokenizer.to_str().encode('utf8')) != input_config['tokenizer_backend_sha256']:
            raise ValueError('Tokenizer backend differs from training')
        features, audits = [], []
        for row in rows:
            uid = lambda text: 'validation_text:' + ep.sha(text.encode('utf8'))
            item = dict(row, id=row['sample_id'], kind='validation', label=0.0,
                answer_triples=results[uid(row['answer'])]['triples'],
                reference_triples=[results[uid(text)]['triples'] for text in row['references']])
            feature, audit = prep.encode_row(item, tokenizer)
            features.append(feature); audits.append(audit)
        # label=0 is a collator placeholder; gold labels are not model inputs.
        ep.write(out / 'features.jsonl', ep.jl(features), frozen=True)
        ep.write(out / 'tokenization_audit.jsonl', ep.jl(audits), frozen=True)
        cfg = transformers.BertConfig.from_pretrained(str(model_path), local_files_only=True)
        cfg._attn_implementation = 'eager'
        model = trainer.model_class(torch)(transformers.BertModel(cfg, add_pooling_layer=False))
        weights = torch.load(trained / 'model_final.pt', map_location='cpu', weights_only=False)
        if weights['config_sha256'] != completed['config_sha256']:
            raise ValueError('Checkpoint config mismatch')
        model.load_state_dict(weights['model'], strict=True)
        del weights
        device = torch.device('cuda:0'); model.to(device); model.eval()
        with torch.inference_mode():
            for index in range(len(predictions), len(rows)):
                text, knowledge, spans, _ = trainer.collate(torch, features, [index], device, cfg.pad_token_id or 0)
                value = float(model(text, knowledge, spans).cpu().item())
                if not math.isfinite(value):
                    raise ValueError('Nonfinite prediction')
                predictions.append({'sample_id': rows[index]['sample_id'], 'question_id': rows[index]['question_id'],
                                    'prediction_raw': value})
                ep.write(prediction_path, ep.enc({'protocol_sha256': protocol_sha, 'predictions': predictions}))
                if (index+1) % 25 == 0 or index+1 == len(rows):
                    print('VALIDATION_PREDICTED %d/%d' % (index+1, len(rows)), flush=True)
    ep.write(out / 'predictions.jsonl', ep.jl(predictions), frozen=True)
    # Read gold values only after prediction export; no fitting or checkpoint selection.
    labels = ep.lines(source / 'validation.labels.jsonl')
    lookup = {r['sample_id']: r for r in labels}
    if len(lookup) != len(labels) or set(lookup) != {r['sample_id'] for r in rows}:
        raise ValueError('Validation label IDs mismatch')
    for r in predictions:
        if lookup[r['sample_id']]['question_id'] != r['question_id']:
            raise ValueError('Question ID mismatch')
    gold = [lookup[r['sample_id']]['gold_raw'] for r in predictions]
    report = {'status': 'SAE_VALIDATION_COMPLETE', 'protocol_sha256': protocol_sha,
        'overall': metrics(gold, [r['prediction_raw'] for r in predictions]), 'by_question': {},
        'checkpoint_selection': 'fixed last epoch; not selected by validation',
        'synthetic_validation_exact_overlap_ids': overlap_ids,
        'interpretation_warning': 'Validation text matches some synthetic training answers; inspect overlap IDs before interpreting performance' if overlap_ids else None,
        'validation_accessed': True, 'test_accessed': False, 'training_modified': False,
        'experiment_status': 'EXPERIMENTAL_REPRODUCTION',
        'paper_five_run_evaluation_complete': False,
        'files_sha256': {n: trainer.sha(out / n) for n in
            ('predictions.jsonl', 'extractions.jsonl', 'features.jsonl', 'tokenization_audit.jsonl')},
        'next_step': 'Matched original-only BERT/SAE baselines; freeze final test evaluation after comparisons'}
    for q in sorted({r['question_id'] for r in predictions}):
        subset = [r for r in predictions if r['question_id'] == q]
        report['by_question'][q] = metrics([lookup[r['sample_id']]['gold_raw'] for r in subset],
                                         [r['prediction_raw'] for r in subset])
    ep.write(out / 'metrics.json', ep.enc(report), frozen=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('\aSAE_VALIDATION_PIPELINE_DONE', flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--max-attempts', type=int, default=3)
    p.add_argument('--physical-gpu', default='auto')
    p.add_argument('--wait-hours', type=float, default=24)
    args = p.parse_args()
    if args.max_attempts < 1 or args.wait_hours < 0:
        p.error('Invalid attempt/wait limit')
    root = args.root.resolve()
    ep = load(root / 'control/scale-synthetic-entities/entity_prepare.py', 'entity_helpers')
    out = root / 'paper_method_v1/runs/sae_raw_beta_seed42_validation_v1'
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(execute(args, ep, out))


if __name__ == '__main__':
    main()
