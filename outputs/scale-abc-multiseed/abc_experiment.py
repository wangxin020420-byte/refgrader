"""Exploratory A/B/C BERT comparison. Validation only, never reads test rows."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random

PILOT_SHA = '50723f7de3bf5a75e79008b1d60ff7ba74dc738c763e85bffb5f1bd5b073e786'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def packed(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')


def immutable(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Existing content differs: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(packed(obj))
    temp.replace(path)


def join(inputs, labels):
    a = {r['sample_id']: r for r in inputs}
    b = {r['sample_id']: r for r in labels}
    if len(a) != len(inputs) or len(b) != len(labels) or a.keys() != b.keys():
        raise ValueError('Duplicate or mismatched source IDs')
    result = []
    for r in inputs:
        label = b[r['sample_id']]
        if r['question_id'] != label['question_id']:
            raise ValueError('Question mismatch')
        value = label['gold_raw']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 10:
            raise ValueError('Invalid score')
        result.append(dict(r, gold_raw=value, origin='original'))
    return result


def groups(train, candidates):
    lookup = {r['sample_id']: r for r in train}
    b, c = list(train), list(train)
    seen = set()
    for item in candidates:
        cid = item['candidate_id']
        if cid in seen or item['review_status'] != 'PROVISIONAL_EXPERIMENT_ONLY' or item['flags']:
            raise ValueError('Invalid reviewed candidate')
        seen.add(cid)
        original = lookup[item['sample_id']]
        if item['original_answer'] != original['answer'] or item['source_gold_raw'] != original['gold_raw']:
            raise ValueError('Candidate provenance mismatch')
        if item['question_id'] != original['question_id'] or not item['generated_answer'].strip():
            raise ValueError('Invalid candidate content')
        # No copied content_hash: it would describe the original, not the generated answer.
        extra = {k: v for k, v in original.items() if k != 'content_hash'}
        extra.update(sample_id='pilot:' + cid, source_sample_id=original['sample_id'],
                     origin='source_repeat', label_policy='source_score_provisional')
        b.append(dict(extra))
        c.append(dict(extra, answer=item['generated_answer'], origin='paraphrase'))
    return {'A': train, 'B': b, 'C': c}


def prepare(root):
    data = root / 'data/prepared/ads_public_adapted_v1'
    pilot = root / 'runs/deepseek_ads_paraphrase_pilot_v1/provisional_candidates_v1.jsonl'
    if sha(pilot) != PILOT_SHA:
        raise ValueError('Reviewed candidate checksum changed')
    source_manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    names = ['train.inputs.jsonl', 'train.labels.jsonl', 'validation.inputs.jsonl', 'validation.labels.jsonl']
    hashes = {name: sha(data / name) for name in names}
    if any(hashes[n] != source_manifest['outputs_sha256'][n] for n in names):
        raise ValueError('Prepared train/validation source changed')
    train = join(rows(data / names[0]), rows(data / names[1]))
    validation = join(rows(data / names[2]), rows(data / names[3]))
    candidates = rows(pilot)
    if (len(train), len(validation), len(candidates)) != (885, 222, 14):
        raise ValueError('Unexpected data counts')
    datasets = groups(train, candidates)
    dest = root / 'data/experiments/abc_pilot_v1'
    output = {}
    for name, records in dict(datasets, validation=validation).items():
        output[name + '.jsonl'] = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records).encode('utf-8')
    plan = {'protocol': 'ABC_PILOT_V1_EXPERIMENTAL_NOT_SCALE', 'counts': {k: len(v) for k, v in datasets.items()},
            'validation_count': 222, 'pilot_sha256': sha(pilot), 'source_hashes': hashes,
            'seed': 42, 'steps': 300, 'effective_batch': 16, 'validate_every': 60, 'max_length': 256,
            'encoder_lr': 5e-5, 'head_lr': 1e-3, 'weight_decay': 0.01,
            'sampling': 'with replacement, deterministic per step; identical indices for B and C',
            'input': 'question-answer pair, CLS plus linear regression head; references unused',
            'loss': 'MSE on score/10, linear unbounded training output',
            'selection': 'largest validation QWK; tie uses lower RMSE; undefined QWK treated as worst',
            'evaluation': 'prediction*10 clipped to [0,10]; also report unclipped RMSE and clip count',
            'warning': 'AI-reviewed provisional labels, small high-score-biased sample; exploratory only',
            'test_accessed': False, 'checkpoint_every_steps': 60,
            'outputs_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in output.items()}}
    output['plan.json'] = packed(plan)
    for name, content in output.items():
        path = dest / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError('Different existing experiment file: ' + name)
    for name, content in output.items():
        immutable(dest / name, content)
    print(json.dumps({'status': 'ABC_READY', 'counts': plan['counts'], 'validation': 222,
                      'test_accessed': False}, indent=2))


def batch_indices(seed, step, n, batch=16):
    rng = random.Random('%d:%d' % (seed, step))
    return [rng.randrange(n) for _ in range(batch)]


def selection_key(metrics):
    qwk = metrics['qwk']
    return (float('-inf') if qwk is None else qwk, -metrics['rmse_raw'])


def train(args):
    root = args.root.resolve()
    data = root / 'data/experiments/abc_pilot_v1'
    plan = json.loads((data / 'plan.json').read_text(encoding='utf-8'))
    for name, value in plan['outputs_sha256'].items():
        if sha(data / name) != value:
            raise ValueError('Experiment input changed: ' + name)
    training = rows(data / (args.group + '.jsonl'))
    validation = rows(data / 'validation.jsonl')
    if 16 % args.microbatch:
        raise ValueError('Microbatch must divide 16')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2)
    metric_path = root / 'control/scale-ads-tools/ads_tools.py'
    spec = importlib.util.spec_from_file_location('ads_metrics', metric_path)
    metric_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric_module)
    lock = json.loads((root / 'sources/bert-base-chinese.lock.json').read_text(encoding='utf-8'))
    model_dir = root / 'models/bert-base-chinese' / lock['revision']
    cfg = {'plan_sha256': sha(data / 'plan.json'), 'script_sha256': sha(Path(__file__)),
           'metrics_script_sha256': sha(metric_path), 'group': args.group, 'seed': args.seed,
           'microbatch': args.microbatch, 'device': args.device, 'model_revision': lock['revision'],
           'torch': torch.__version__, 'transformers': transformers.__version__}
    if args.check:
        print(json.dumps(dict(cfg, status='TRAIN_PREFLIGHT_OK', train_count=len(training),
                              validation_count=len(validation), training_started=False), indent=2))
        return
    run = root / 'runs/abc_followup_v1' / (args.group + '_seed%d' % args.seed)
    run.mkdir(parents=True, exist_ok=True)
    immutable(run / 'config.json', packed(cfg))
    if (run / 'complete.json').exists():
        print('Already complete; no retraining:', run)
        return
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable')
        free, total = torch.cuda.mem_get_info()
        if free < 8 * 1024 ** 3:
            raise ValueError('Less than 8 GiB free; wait for allocated GPU resources. This is a conservative check, not a memory guarantee.')
    runlock = run / 'RUNNING.lock'
    with runlock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                     trust_remote_code=False, clean_up_tokenization_spaces=False)
        encoder = AutoModel.from_pretrained(str(model_dir), local_files_only=True,
                     use_safetensors=True, trust_remote_code=False).to(args.device)
        head = torch.nn.Linear(encoder.config.hidden_size, 1).to(args.device)
        optimizer = torch.optim.AdamW([{'params': encoder.parameters(), 'lr': 5e-5},
                                      {'params': head.parameters(), 'lr': 1e-3}], weight_decay=0.01)
        def encode(items):
            return tokenizer([r['question'] for r in items], [r['answer'] for r in items],
                             return_tensors='pt', padding=True, truncation=True, max_length=256).to(args.device)
        def forward(items):
            return head(encoder(**encode(items)).last_hidden_state[:, 0, :]).squeeze(-1)
        history, best, start = [], None, 0
        checkpoint = run / 'last.pt'
        if checkpoint.exists():
            # Only load this runner's own local checkpoint, never a downloaded pickle.
            saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
            if saved['config'] != cfg:
                raise ValueError('Checkpoint configuration mismatch')
            encoder.load_state_dict(saved['encoder']); head.load_state_dict(saved['head'])
            optimizer.load_state_dict(saved['optimizer'])
            start, history, best = saved['step'], saved['history'], saved['best']
            torch.set_rng_state(saved['rng'].cpu())
            if args.device == 'cuda':
                torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
            print('Resuming step', start, flush=True)
        for step in range(start + 1, 301):
            encoder.train(); head.train(); optimizer.zero_grad(set_to_none=True)
            indices = batch_indices(args.seed, step, len(training))
            for offset in range(0, 16, args.microbatch):
                items = [training[i] for i in indices[offset:offset + args.microbatch]]
                target = torch.tensor([r['gold_raw'] / 10 for r in items], device=args.device)
                loss = torch.nn.functional.mse_loss(forward(items), target) * (args.microbatch / 16)
                if not torch.isfinite(loss).item():
                    raise ValueError('Non-finite training loss')
                loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step % 10 == 0:
                print(args.group, 'step', step, '/300', flush=True)
            if step % 60:
                continue
            encoder.eval(); head.eval(); raw = []
            with torch.no_grad():
                for offset in range(0, len(validation), args.microbatch):
                    raw.extend((forward(validation[offset:offset + args.microbatch]) * 10).cpu().tolist())
            if not all(math.isfinite(p) for p in raw):
                raise ValueError('Non-finite validation predictions')
            clipped = [min(10.0, max(0.0, p)) for p in raw]
            gold = [r['gold_raw'] for r in validation]
            scores = metric_module.metrics(gold, clipped)
            scores['unclipped_rmse_raw'] = math.sqrt(sum((p-g)**2 for p,g in zip(raw,gold))/len(gold))
            scores['clip_count'] = sum(p != q for p,q in zip(raw,clipped))
            current = {'step': step, 'metrics': scores}
            predictions = [{'sample_id': r['sample_id'], 'prediction_raw': p, 'prediction_unclipped': u}
                           for r,p,u in zip(validation,clipped,raw)]
            atomic_json(run / ('validation_step_%03d.json' % step), {'evaluation': current, 'predictions': predictions})
            if best is None or selection_key(scores) > selection_key(best['metrics']):
                best = current
                temp = run / 'best.pt.tmp'
                torch.save({'config': cfg, 'step': step, 'encoder': encoder.state_dict(), 'head': head.state_dict()}, temp)
                temp.replace(run / 'best.pt')
            history.append(current)
            temp = run / 'last.pt.tmp'
            torch.save({'config': cfg, 'step': step, 'encoder': encoder.state_dict(), 'head': head.state_dict(),
                        'optimizer': optimizer.state_dict(), 'history': history, 'best': best,
                        'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if args.device == 'cuda' else []}, temp)
            temp.replace(checkpoint)
            atomic_json(run / 'progress.json', {'step': step, 'best': best})
            print(json.dumps(current), flush=True)
        atomic_json(run / 'complete.json', {'status': 'EXPLORATORY_VALIDATION_COMPLETE', 'group': args.group,
                    'best': best, 'history': history, 'test_evaluated': False, 'provisional_labels': True})
    finally:
        runlock.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'train'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--group', choices=['A', 'B', 'C'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--microbatch', type=int, choices=[1, 2, 4, 8, 16], default=2)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--seed', type=int, choices=[43, 44, 45, 46], required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.root.resolve())
    else:
        if not args.group:
            parser.error('train requires --group')
        train(args)


if __name__ == '__main__':
    main()
