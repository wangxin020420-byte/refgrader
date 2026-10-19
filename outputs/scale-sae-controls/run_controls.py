"""Offline prediction audit and fixed-epoch original-only BERT/SAE controls."""
import argparse
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time

VAL_SHA = '497bd6e7c940acd5b591a806e4247455a77de786c54d37548e866d6e5bce4ee8'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]


def audit_predictions(predictions, labels):
    by_id = {r['sample_id']: r for r in labels}
    if len(by_id) != len(labels) or len({r['sample_id'] for r in predictions}) != len(predictions):
        raise ValueError('Duplicate validation IDs')
    if set(by_id) != {r['sample_id'] for r in predictions}:
        raise ValueError('Validation ID mismatch')
    if any(by_id[r['sample_id']]['question_id'] != r['question_id'] for r in predictions):
        raise ValueError('Prediction/gold question mismatch')
    def aggregate(rows):
        values = [r['prediction_raw'] for r in rows]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite prediction')
        gold = [by_id[r['sample_id']]['gold_raw'] for r in rows]
        return {'records': len(rows), 'below_zero': sum(x < 0 for x in values),
                'above_ten': sum(x > 10 for x in values), 'prediction_min': min(values),
                'prediction_max': max(values), 'prediction_mean': sum(values)/len(values),
                'gold_mean': sum(gold)/len(gold),
                'mean_signed_error': sum(p-g for p,g in zip(values,gold))/len(values),
                'out_of_range_ids': [r['sample_id'] for r in rows if not 0 <= r['prediction_raw'] <= 10]}
    return {'overall': aggregate(predictions), 'by_question': {
        q: aggregate([r for r in predictions if r['question_id'] == q])
        for q in sorted({r['question_id'] for r in predictions})},
        'interpretation': 'Descriptive validation audit only; no calibration, clipping changes or retraining of completed model'}


def build_model(tr, torch, transformers, path, kind, device):
    model, optimizer = tr.create_model(torch, transformers, path, 42, device)
    if kind == 'sae_original':
        return model, optimizer
    class BertControl(torch.nn.Module):
        def __init__(self, encoder, regressor):
            super().__init__()
            self.encoder = encoder
            self.regressor = regressor
        def forward(self, text, knowledge, spans):
            return self.regressor(self.encoder(**text).last_hidden_state[:, 0]).squeeze(-1)
    # Reuse initial shared encoder/head tensors for matched initialization.
    baseline = BertControl(model.encoder, model.regressor)
    del model, optimizer
    optimizer = torch.optim.AdamW([
        {'params': list(baseline.encoder.parameters()), 'lr': 5e-5},
        {'params': list(baseline.regressor.parameters()), 'lr': 1e-3}
    ], weight_decay=0.0, foreach=False)
    return baseline, optimizer


def run_one(root, kind, rows, validation, labels, common, tr, val, torch, transformers):
    out = root / ('paper_method_v1/runs/' + kind + '_seed42_v1')
    out.mkdir(parents=True, exist_ok=True)
    config = dict(common, kind=kind, training_records=885, effective_batch=16,
                  input='Reference-answer pair only' if kind == 'bert_original' else 'Same SAE text/entity streams as augmented model')
    tr.save(out / 'config.json', config, frozen=True)
    cfg_sha = tr.sha(out / 'config.json')
    if (out / 'complete.json').exists():
        result = tr.read(out / 'complete.json')
        if result['config_sha256'] != cfg_sha:
            raise ValueError('Completed baseline config changed')
        for name, expected in result['files_sha256'].items():
            if tr.sha(out / name) != expected:
                raise ValueError('Completed baseline changed: ' + name)
        print('BASELINE_ALREADY_COMPLETE', kind, flush=True)
        return tr.read(out / 'validation_metrics.json')
    if torch is None:
        raise RuntimeError('GPU runtime required for incomplete baseline')
    device = torch.device('cuda:0')
    model_path = root / 'models/bert-base-chinese' / tr.REV
    pad_id = tr.read(model_path / 'config.json').get('pad_token_id', 0)
    model, optimizer = build_model(tr, torch, transformers, model_path, kind, device)
    if not (out / 'smoke.json').exists():
        longest = max(range(len(rows)), key=lambda i: len(rows[i]['text']['input_ids']) + len(rows[i]['knowledge']['input_ids']))
        before = model.regressor.weight.detach().clone()
        error = tr.train_step(torch, model, optimizer, rows, [0, longest], device, pad_id, 1)
        if not math.isfinite(error) or torch.equal(before, model.regressor.weight.detach()):
            raise ValueError('Baseline optimizer smoke failed')
        tr.save(out / 'smoke.json', {'status': 'BASELINE_SMOKE_OK', 'config_sha256': cfg_sha,
                                     'smoke_weights_used_for_training': False}, frozen=True)
        del model, optimizer, before
        gc.collect(); torch.cuda.empty_cache()
        model, optimizer = build_model(tr, torch, transformers, model_path, kind, device)
    elif tr.read(out / 'smoke.json')['config_sha256'] != cfg_sha:
        raise ValueError('Baseline smoke configuration changed')
    state = {'epoch': 0, 'next_batch': 0, 'steps': 0, 'sse': 0.0, 'seen': 0, 'history': []}
    latest = out / 'latest.pt'
    if latest.exists():
        checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
        if checkpoint['config_sha256'] != cfg_sha:
            raise ValueError('Baseline checkpoint configuration changed')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        state = checkpoint['state']
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        random.setstate(checkpoint['python_rng'])
        del checkpoint
        print('RESUMED', kind, state['epoch'], state['next_batch'], flush=True)
    def checkpoint_save():
        tr.torch_save(torch, latest, {'config_sha256': cfg_sha, 'state': state,
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'python_rng': random.getstate()})
        tr.save(out / 'progress.json', dict(state, status='TRAINING'))
    for epoch in range(state['epoch'], 10):
        schedule = tr.batches(rows, epoch, 42)
        for step in range(state['next_batch'], len(schedule)):
            group = schedule[step]
            error = tr.train_step(torch, model, optimizer, rows, group, device, pad_id, 1)
            state['sse'] += error; state['seen'] += len(group)
            state['next_batch'] = step + 1; state['steps'] += 1
            if state['steps'] % 10 == 0:
                print(kind, 'EPOCH', epoch+1, 'STEP', step+1, '/', len(schedule), 'MSE', error/len(group), flush=True)
            if state['steps'] % 20 == 0:
                checkpoint_save()
        if state['seen'] != 885:
            raise ValueError('Baseline epoch coverage differs')
        state['history'].append({'epoch': epoch+1, 'train_mse_with_dropout': state['sse']/885})
        state.update(epoch=epoch+1, next_batch=0, sse=0.0, seen=0)
        checkpoint_save()
        print(kind, 'EPOCH_SAVED', epoch+1, flush=True)
    tr.save(out / 'history.json', state['history'])
    tr.torch_save(torch, out / 'model_final.pt', {'config_sha256': cfg_sha, 'model': model.state_dict()})
    del model, optimizer
    gc.collect(); torch.cuda.empty_cache()
    model, optimizer = build_model(tr, torch, transformers, model_path, kind, device)
    del optimizer
    checkpoint = torch.load(out / 'model_final.pt', map_location='cpu', weights_only=False)
    if checkpoint['config_sha256'] != cfg_sha:
        raise ValueError('Final baseline checkpoint mismatch')
    model.load_state_dict(checkpoint['model'], strict=True); del checkpoint
    model.eval(); predictions = []
    with torch.inference_mode():
        for i, row in enumerate(validation):
            text, knowledge, spans, _ = tr.collate(torch, validation, [i], device, pad_id)
            value = float(model(text, knowledge, spans).cpu().item())
            if not math.isfinite(value):
                raise ValueError('Nonfinite baseline prediction')
            predictions.append({'sample_id': row['sample_id'], 'question_id': labels[row['sample_id']]['question_id'],
                                'prediction_raw': value})
    tr.save(out / 'validation_predictions.json', predictions)
    gold = [labels[r['sample_id']]['gold_raw'] for r in predictions]
    metrics = {'kind': kind, 'overall': val.metrics(gold, [r['prediction_raw'] for r in predictions]),
        'by_question': {}, 'checkpoint_selection': 'fixed last epoch', 'test_accessed': False}
    for q in sorted({r['question_id'] for r in predictions}):
        subset = [r for r in predictions if r['question_id'] == q]
        metrics['by_question'][q] = val.metrics([labels[r['sample_id']]['gold_raw'] for r in subset],
                                              [r['prediction_raw'] for r in subset])
    tr.save(out / 'validation_metrics.json', metrics)
    tr.save(out / 'complete.json', {'status': 'ORIGINAL_ONLY_BASELINE_COMPLETE', 'kind': kind,
        'training_records': 885, 'epochs': 10, 'optimizer_steps': state['steps'],
        'config_sha256': cfg_sha, 'seed': 42, 'test_accessed': False, 'validation_accessed': True,
        'final_weights_reload_verified': True, 'api_calls': 0,
        'files_sha256': {n: tr.sha(out / n) for n in ('model_final.pt', 'validation_predictions.json',
                                                   'validation_metrics.json', 'history.json', 'smoke.json')}})
    tr.save(out / 'progress.json', dict(state, status='COMPLETE'))
    print('BASELINE_COMPLETE', kind, json.dumps(metrics['overall']), flush=True)
    return metrics


def execute(args, out):
    root = args.root.resolve()
    tr_path = root / 'control/scale-sae-train/train_sae.py'
    val_path = root / 'control/scale-sae-validation/validate_sae.py'
    tr = load(tr_path, 'sae_training_helpers')
    val = load(val_path, 'sae_validation_helpers')
    trained = root / 'paper_method_v1/runs/sae_raw_beta_seed42_v1'
    train_config = tr.read(trained / 'config.json')
    train_complete = tr.read(trained / 'complete.json')
    if tr.sha(tr_path) != train_config['script_sha256'] or tr.sha(trained/'config.json') != train_complete['config_sha256']:
        raise ValueError('Trained architecture/configuration changed')
    if tr.read(trained/'architecture_checks.json')['status'] != 'SAE_ARCHITECTURE_CHECKS_OK':
        raise ValueError('Original architecture checks missing')
    if tr.sha(trained/'architecture_checks.json') != train_complete['files_sha256']['architecture_checks.json']:
        raise ValueError('Original architecture checks changed')
    validation_run = root / 'paper_method_v1/runs/sae_raw_beta_seed42_validation_v1'
    augmented_metrics = tr.read(validation_run / 'metrics.json')
    protocol = tr.read(validation_run / 'protocol.json')
    if augmented_metrics['status'] != 'SAE_VALIDATION_COMPLETE' or tr.sha(validation_run/'protocol.json') != augmented_metrics['protocol_sha256']:
        raise ValueError('Validation incomplete/protocol changed')
    if tr.sha(val_path) != protocol['script_sha256']:
        raise ValueError('Metric implementation changed')
    for name, expected in augmented_metrics['files_sha256'].items():
        if tr.sha(validation_run / name) != expected:
            raise ValueError('Validation artifacts changed: ' + name)
    if tr.sha(validation_run/'features.jsonl') != VAL_SHA:
        raise ValueError('Unexpected validation feature set')
    all_rows, _ = tr.load_inputs(root)
    rows = [r for r in all_rows if r['kind'] == 'original_training']
    if len(rows) != 885 or any(not 0 <= r['label'] <= 10 for r in rows):
        raise ValueError('Original-only training inputs invalid')
    validation = lines(validation_run / 'features.jsonl')
    data = root / 'data/prepared/ads_public_adapted_v1'
    labels_path = data / 'validation.labels.jsonl'
    if tr.sha(labels_path) != protocol['data_sha256']['validation.labels.jsonl']:
        raise ValueError('Validation gold changed')
    label_rows = lines(labels_path); labels = {r['sample_id']: r for r in label_rows}
    if len(validation) != 222 or len(labels) != 222 or {r['sample_id'] for r in validation} != set(labels):
        raise ValueError('Validation identity mismatch')
    if {r['sample_id'] for r in rows} & set(labels):
        raise ValueError('Training/validation ID overlap')
    for r in validation:
        if not math.isfinite(r['label']):
            raise ValueError('Invalid collator placeholder')
    audit = audit_predictions(lines(validation_run/'predictions.jsonl'), label_rows)
    tr.save(out / 'augmented_prediction_audit.json', audit, frozen=True)
    print('OUT_OF_RANGE_AUDIT', json.dumps({k:v for k,v in audit['overall'].items()
                                          if k != 'out_of_range_ids'}), flush=True)
    model_path = root / 'models/bert-base-chinese' / tr.REV
    for name, expected in train_config['model_files_sha256'].items():
        if tr.sha(model_path/name) != expected:
            raise ValueError('Pretrained model changed')
    common = {'protocol': 'original_only_controls_fixed10_seed42_v1', 'seed': 42, 'epochs': 10,
        'microbatch': 1, 'encoder_lr': 5e-5, 'head_lr': 1e-3, 'optimizer': 'AdamW foreach=False',
        'weight_decay': 0.0, 'scheduler': 'constant', 'precision': 'float32', 'gradient_clip': 1.0,
        'checkpoint_selection': 'fixed last epoch; no validation-based selection',
        'trainer_sha256': tr.sha(tr_path), 'metric_code_sha256': tr.sha(val_path),
        'script_sha256': tr.sha(Path(__file__)), 'source_features_sha256': tr.FEATURE_SHA,
        'validation_features_sha256': VAL_SHA, 'validation_labels_sha256': tr.sha(labels_path),
        'reference_training_config_sha256': train_complete['config_sha256'],
        'metric_policy': protocol['metric_policy'], 'api_calls': 0, 'test_accessed': False,
        'experiment_status': 'EXPERIMENTAL_REPRODUCTION_SINGLE_SEED',
        'comparison_limit': 'Epoch-matched, not optimizer-update-matched: originals 560 steps, augmentation 770. Not five-run paper evaluation.'}
    tr.save(out/'protocol.json', common, frozen=True)
    kinds = ['bert_original', 'sae_original']
    pending = [k for k in kinds if not (root/('paper_method_v1/runs/'+k+'_seed42_v1/complete.json')).exists()]
    torch = transformers = None
    if pending:
        for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
            os.environ[name] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        deadline = time.monotonic() + args.wait_hours*3600
        while True:
            try:
                tr.select_gpu(args.physical_gpu); break
            except RuntimeError as exc:
                if not str(exc).startswith('No selected GPU has') or time.monotonic() >= deadline:
                    raise
                print('WAIT_GPU: retry in 60 seconds', flush=True); time.sleep(60)
        import torch
        import transformers
        if torch.__version__ != train_config['torch'] or transformers.__version__ != train_config['transformers']:
            raise ValueError('Training library versions changed')
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    results = {}
    for kind in kinds:
        results[kind] = run_one(root, kind, rows, validation, labels, common, tr, val, torch, transformers)
        if torch is not None:
            gc.collect(); torch.cuda.empty_cache()
    results['sae_augmented'] = augmented_metrics
    table = {kind: result['overall'] for kind, result in results.items()}
    comparison = {'status': 'THREE_WAY_VALIDATION_COMPARISON_COMPLETE', 'overall': table,
        'differences': {name: {metric: (table[a][metric]-table[b][metric])
            if table[a][metric] is not None and table[b][metric] is not None else None
            for metric in ('pearson_raw', 'qwk_clipped_half_up', 'rmse_raw')}
            for name,a,b in [('sae_original_minus_bert_original','sae_original','bert_original'),
                              ('sae_augmented_minus_sae_original','sae_augmented','sae_original')]},
        'comparison_limit': common['comparison_limit'], 'seed': 42, 'validation_records': 222,
        'api_calls': 0, 'test_accessed': False, 'existing_augmented_model_retrained': False,
        'provenance': {'protocol_sha256': tr.sha(out/'protocol.json'),
            'augmented_validation_sha256': tr.sha(validation_run/'metrics.json'),
            'baseline_complete_sha256': {kind:tr.sha(root/('paper_method_v1/runs/'+kind+'_seed42_v1/complete.json')) for kind in kinds}}}
    tr.save(out/'comparison.json', comparison, frozen=True)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print('\aCONTROLS_PIPELINE_DONE', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--physical-gpu', default='auto')
    p.add_argument('--wait-hours', type=float, default=24)
    args = p.parse_args()
    if args.wait_hours < 0:
        p.error('wait-hours must be nonnegative')
    out = args.root.resolve() / 'paper_method_v1/runs/sae_controls_seed42_v1'
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, out)


if __name__ == '__main__':
    main()
