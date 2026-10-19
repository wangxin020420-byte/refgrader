"""Offline vanilla-BERT annotator, fixed-epoch experimental reproduction protocol."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

REV = '8f23c25b06e129b6c986331a13d8d025a92cf0ea'
TRAIN_SHA = '3130129658d2ed782be2bcf3de1112dcb5d10b6b5d8620d8a0bccb85a514706b'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def save(p, obj, immutable=False):
    data = (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf8')
    if immutable and p.exists():
        if p.read_bytes() != data:
            raise ValueError('Frozen configuration differs: ' + str(p))
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)


def reference_text(row):
    refs = row['references']
    if not isinstance(refs, list) or not all(isinstance(x, str) for x in refs):
        raise ValueError('Invalid references')
    text = '\n'.join(r.strip() for r in refs if r.strip())
    if not text:
        raise ValueError('Empty references')
    return text


def groups(indices, size=16):
    return [indices[i:i + size] for i in range(0, len(indices), size)]


def load_data(root):
    folder = root / 'paper_method_v1/artifacts/annotator_inputs_v1'
    path = folder / 'train.jsonl'
    if sha(path) != TRAIN_SHA or read(folder / 'manifest.json')['train_jsonl_sha256'] != TRAIN_SHA:
        raise ValueError('Training data hash mismatch')
    rows = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
    if len(rows) != 885 or len({r['sample_id'] for r in rows}) != 885:
        raise ValueError('Expected 885 unique samples')
    for r in rows:
        if ':train:' not in r['sample_id'] or r['label'] != r['source_score']:
            raise ValueError('Training identity or score mismatch')
        if not isinstance(r['answer'], str) or not math.isfinite(r['label']) or not 0 <= r['label'] <= 10:
            raise ValueError('Invalid answer or label')
        reference_text(r)
    return rows


def prepare(root, model_path, epochs, torch, transformers):
    rows = load_data(root)
    if not model_path.is_dir():
        raise ValueError('Local BERT directory missing')
    files = {str(p.relative_to(model_path)): sha(p) for p in sorted(model_path.rglob('*'))
             if p.is_file() and p.suffix in ('.json', '.txt', '.bin', '.safetensors') and '.cache' not in p.parts}
    if not any(n.endswith(('.bin', '.safetensors')) for n in files):
        raise ValueError('No local BERT weights')
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.truncation_side = 'right'
    tokenizer.padding_side = 'right'
    encoded = []
    audit = []
    for r in rows:
        ref = reference_text(r)
        full = tokenizer(ref, r['answer'], truncation=False, verbose=False)
        item = tokenizer(ref, r['answer'], max_length=512, truncation='longest_first')
        if len(full['input_ids']) > 512:
            audit.append({'sample_id': r['sample_id'], 'full_tokens': len(full['input_ids']),
                          'kept_tokens': len(item['input_ids']), 'removed_tokens': len(full['input_ids'])-len(item['input_ids'])})
        encoded.append(item)
    config = {'protocol': 'annotator_fixed10_v1', 'experiment_status': 'EXPERIMENTAL_REPRODUCTION',
        'model_path': str(model_path), 'model_files_sha256': files, 'data_sha256': TRAIN_SHA,
        'script_sha256': sha(Path(__file__)), 'torch': torch.__version__, 'transformers': transformers.__version__,
        'epochs': epochs, 'seed': 42, 'microbatch': 2, 'effective_batch': 16, 'max_length': 512,
        'reference_policy': 'Join all nonempty references in original slot order with newline; one row per source.',
        'input': '[CLS] references [SEP] proxy_answer [SEP]; question excluded',
        'truncation': 'longest_first, right; full answers remain in source data',
        'loss': 'MSE on original ADS 0-10 scores; no normalization or clipping',
        'optimizer': 'AdamW', 'encoder_lr': 5e-5, 'head_lr': 1e-3, 'weight_decay': 0.0,
        'scheduler': 'constant', 'gradient_clip': 1.0, 'precision': 'float32',
        'gradient_checkpointing': True, 'checkpoint_selection': 'last fixed epoch; no best-validation claim',
        'validation_accessed': False, 'test_accessed': False,
        'deviations': ['GLM replaces paper Llama', 'provisional initial-filter k=5',
            'seven original-answer fallbacks', '636 proxies not individually reviewed',
            'multiple-reference concatenation and truncation are implementation choices',
            'fixed training duration, optimizer details and checkpoint policy are implementation choices',
            'no validation-based checkpoint selection; not paper-equivalent final evaluation'],
        'fine_filter_beta_scale': 'UNRESOLVED; do not apply beta automatically'}
    return rows, tokenizer, encoded, audit, config


def make_model(model_path, torch, transformers, device):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model, info = transformers.BertForSequenceClassification.from_pretrained(str(model_path),
        local_files_only=True, num_labels=1, problem_type='regression', output_loading_info=True)
    missing = info.get('missing_keys', [])
    if any(not k.startswith('classifier.') for k in missing) or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('Unexpected BERT weight loading report: ' + str(info))
    model.gradient_checkpointing_enable()
    model.to(device)
    optimizer = torch.optim.AdamW([
        {'params': list(model.bert.parameters()), 'lr': 5e-5},
        {'params': list(model.classifier.parameters()), 'lr': 1e-3}], weight_decay=0.0)
    return model, optimizer


def step(model, optimizer, indices, rows, tokens, tokenizer, torch, device):
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for chunk in groups(indices, 2):
        batch = tokenizer.pad([tokens[i] for i in chunk], padding=True, return_tensors='pt')
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = torch.tensor([rows[i]['label'] for i in chunk], dtype=torch.float32, device=device)
        pred = model(**batch).logits.reshape(-1)
        loss = torch.nn.functional.mse_loss(pred, labels, reduction='sum') / len(indices)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        total += loss.detach().item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return total


def execute(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.physical_gpu)
    import torch
    import transformers
    import fcntl
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    root = args.root.resolve()
    model_path = root / 'models/bert-base-chinese' / REV
    run = root / 'paper_method_v1/runs/annotator_bert_v1'
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, tokenizer, tokens, audit, config = prepare(root, model_path, args.epochs, torch, transformers)
        save(run / 'config.json', config, immutable=True)
        config_sha = sha(run / 'config.json')
        save(run / 'truncation_audit.json', audit, immutable=True)
        if args.command == 'preflight':
            report = {'status': 'ANNOTATOR_PREFLIGHT_OK', 'records': len(rows),
                'overlength_pairs': len(audit), 'epochs': args.epochs,
                'steps_per_epoch': len(groups(list(range(len(rows))))),
                'reference_counts': {str(n): sum(sum(bool(x.strip()) for x in r['references']) == n for r in rows) for n in (1, 2)},
                'config_sha256': config_sha, 'offline': True, 'training_started': False,
                'test_accessed': False}
            save(run / 'preflight.json', report); print(json.dumps(report, indent=2)); return
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no CPU fallback for training')
        device = torch.device('cuda:0')
        if args.command == 'train':
            smoke = read(run / 'smoke.json')
            if smoke['status'] != 'ANNOTATOR_SMOKE_OK' or smoke['config_sha256'] != config_sha:
                raise ValueError('Matching smoke run required')
            if (run / 'complete.json').exists():
                report = read(run / 'complete.json')
                if report['config_sha256'] != config_sha:
                    raise ValueError('Completed configuration mismatch')
                for name, expected in report['final_files_sha256'].items():
                    if sha(run / 'model_final' / name) != expected:
                        raise ValueError('Completed model changed')
                print(json.dumps(report, indent=2)); return
        model, optimizer = make_model(model_path, torch, transformers, device)
        model.train()
        if args.command == 'smoke':
            # Exercise the longest tokenized pairs, rather than only short early rows.
            indices = sorted(range(len(rows)), key=lambda i: (-len(tokens[i]['input_ids']), i))[:32]
            losses = [step(model, optimizer, group, rows, tokens, tokenizer, torch, device) for group in groups(indices)]
            report = {'status': 'ANNOTATOR_SMOKE_OK', 'config_sha256': config_sha,
                'physical_gpu': args.physical_gpu, 'gpu_name': torch.cuda.get_device_name(0),
                'optimizer_steps': 2, 'losses_debug_only': losses,
                'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
                'formal_model_saved': False, 'offline': True, 'test_accessed': False}
            save(run / 'smoke.json', report); print(json.dumps(report, indent=2)); return
        latest = run / 'latest.pt'
        start = 0
        if latest.exists():
            checkpoint = torch.load(str(latest), map_location='cpu', weights_only=False)
            if checkpoint['config_sha256'] != config_sha:
                raise ValueError('Checkpoint configuration mismatch')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            torch.set_rng_state(checkpoint['cpu_rng'])
            torch.cuda.set_rng_state(checkpoint['cuda_rng'], device)
            start = checkpoint['completed_epochs']
            del checkpoint
            print('RESUME_FROM_COMPLETED_EPOCH', start, flush=True)
        for epoch in range(start, args.epochs):
            if shutil.disk_usage(root).free < 6 * 1024**3:
                raise RuntimeError('Need 6 GiB free for safe checkpoint replacement')
            order = list(range(len(rows)))
            random.Random(42 + epoch).shuffle(order)
            loss_sum = 0.0
            for n, group in enumerate(groups(order), 1):
                loss = step(model, optimizer, group, rows, tokens, tokenizer, torch, device)
                loss_sum += loss * len(group)
                if n % 10 == 0 or n == 56:
                    print('EPOCH', epoch+1, 'STEP', n, '/56', 'BATCH_MSE', round(loss, 6), flush=True)
            temp = run / 'latest.pt.tmp'
            with temp.open('wb') as f:
                torch.save({'config_sha256': config_sha, 'completed_epochs': epoch+1,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(device)}, f)
                f.flush(); os.fsync(f.fileno())
            temp.replace(latest)
            report = {'completed_epochs': epoch+1, 'training_mse': loss_sum / len(rows),
                      'config_sha256': config_sha}
            save(run / ('epoch_%02d.json' % (epoch+1)), report)
            print('EPOCH_SAVED', epoch+1, 'TRAIN_MSE', report['training_mse'], flush=True)
        final = run / 'model_final'
        model.save_pretrained(str(final), safe_serialization=True)
        tokenizer.save_pretrained(str(final))
        report = {'status': 'ANNOTATOR_TRAINING_COMPLETE_FIXED_EPOCH',
            'experiment_status': 'EXPERIMENTAL_REPRODUCTION', 'completed_epochs': args.epochs,
            'training_records': len(rows), 'optimizer_steps': 56 * args.epochs,
            'config_sha256': config_sha, 'checkpoint_selection': 'last_epoch',
            'validation_accessed': False, 'test_accessed': False, 'candidate_labels_assigned': 0,
            'final_files_sha256': {p.name: sha(p) for p in final.iterdir() if p.is_file()}}
        save(run / 'complete.json', report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'smoke', 'train'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1 or args.physical_gpu < 0:
        parser.error('epochs must be positive; physical-gpu must be nonnegative')
    execute(args)
