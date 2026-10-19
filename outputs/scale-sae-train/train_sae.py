"""Offline, single-seed experimental SAE pipeline; literal equations 15-20.

Fixed epochs and mixed source-specific batches are implementation choices.
No validation/test selection. Singleton cross attention is explicitly preserved.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

REV = '8f23c25b06e129b6c986331a13d8d025a92cf0ea'
FEATURE_SHA = 'eef59bdd4389196c72bbeb238c0fdc7cea118de67e01abea51473996e2730085'
INPUT_CONFIG_SHA = 'd86d5e7ebc00d603ee82ec26e24de428e416b54408b9f3ab5db1ba2cdc9a862d'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def enc(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf8')


def save(path, obj, frozen=False):
    data = enc(obj)
    if frozen and path.exists():
        if path.read_bytes() != data:
            raise ValueError('Frozen configuration differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def torch_save(torch, path, obj):
    if shutil.disk_usage(path.parent).free < 3 * 1024**3:
        raise RuntimeError('DISK_GUARD: need at least 3 GiB free for atomic checkpoint')
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def batches(rows, epoch, seed):
    rng = random.Random(seed + epoch)
    result = []
    for kind, size in [('original_training', 16), ('synthetic_candidate', 64)]:
        indices = [i for i, row in enumerate(rows) if row['kind'] == kind]
        rng.shuffle(indices)
        result.extend(indices[j:j+size] for j in range(0, len(indices), size))
    rng.shuffle(result)
    return result


def select_gpu(requested):
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
        universal_newlines=True)
    devices = []
    for line in output.strip().splitlines():
        index, uuid, free, util = [s.strip() for s in line.split(',')]
        devices.append({'physical_index': int(index), 'uuid': uuid,
                        'free_mib': int(free), 'utilization': int(util)})
    candidates = [d for d in devices if d['free_mib'] >= 12000 and d['utilization'] <= 10
                  and (requested == 'auto' or d['physical_index'] == int(requested))]
    if not candidates:
        raise RuntimeError('No selected GPU has >=12000 MiB free and <=10% utilization: ' + str(devices))
    chosen = max(candidates, key=lambda d: d['free_mib'])
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = chosen['uuid']
    print('GPU_SELECTED', json.dumps(chosen), flush=True)
    return chosen


def model_class(torch):
    nn = torch.nn

    class SAE(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            d = encoder.config.hidden_size
            self.heads = encoder.config.num_attention_heads
            if d % self.heads:
                raise ValueError('Head dimension mismatch')
            self.entity_hidden = nn.Linear(d, d)
            self.entity_score = nn.Linear(d, 1, bias=False)
            self.query = nn.Linear(d, d, bias=False)
            self.key = nn.Linear(d, d, bias=False)
            self.value = nn.Linear(d, d, bias=False)
            self.output = nn.Linear(d, d, bias=False)
            self.regressor = nn.Linear(d, 1)

        def forward(self, text, knowledge, spans, details=False):
            ht = self.encoder(**text).last_hidden_state[:, 0, :]
            hidden = self.encoder(**knowledge).last_hidden_state
            maximum = max(1, max(len(s) for s in spans))
            vectors, masks = [], []
            for i, entries in enumerate(spans):
                values = [hidden[i, a:b].mean(dim=0) for a, b in entries]
                n = len(values)
                if n:
                    entity = torch.stack(values)
                    entity = torch.cat([entity, hidden.new_zeros((maximum-n, hidden.shape[-1]))], dim=0)
                else:
                    entity = hidden.new_zeros((maximum, hidden.shape[-1]))
                vectors.append(entity)
                masks.append([True] * n + [False] * (maximum-n))
            he = torch.stack(vectors)
            mask = torch.tensor(masks, dtype=torch.bool, device=hidden.device)
            logits = self.entity_score(torch.tanh(self.entity_hidden(he))).squeeze(-1)
            alpha = torch.softmax(logits.masked_fill(~mask, -10000.0), dim=1) * mask
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
            pooled = (alpha.unsqueeze(-1) * he).sum(dim=1)
            batch, d = ht.shape
            shape = (batch, self.heads, 1, d // self.heads)
            q = self.query(ht).reshape(shape)
            k = self.key(pooled).reshape(shape)
            v = self.value(pooled).reshape(shape)
            # Equation 18 explicitly gives a 1x1 attention matrix: softmax == 1.
            attention = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d // self.heads), dim=-1)
            attended = torch.matmul(attention, v).reshape(batch, d)
            fused = ht + self.output(attended)
            # The literal CLS formulation has one fused sequence position.
            pooled_fused = fused.unsqueeze(1).max(dim=1).values
            prediction = self.regressor(pooled_fused).squeeze(-1)
            if details:
                return prediction, {'alpha': alpha, 'mask': mask, 'attention': attention,
                                    'text_cls': ht, 'fused': fused}
            return prediction
    return SAE


def collate(torch, rows, indices, device, pad_id):
    streams = []
    for name in ('text', 'knowledge'):
        length = max(len(rows[i][name]['input_ids']) for i in indices)
        batch = {}
        for key in ('input_ids', 'attention_mask', 'token_type_ids'):
            fill = pad_id if key == 'input_ids' else 0
            batch[key] = torch.tensor([rows[i][name][key] + [fill] *
                (length - len(rows[i][name][key])) for i in indices], dtype=torch.long, device=device)
        streams.append(batch)
    spans = [[e['token_span'] for e in rows[i]['entity_spans']] for i in indices]
    labels = torch.tensor([rows[i]['label'] for i in indices], dtype=torch.float32, device=device)
    return streams[0], streams[1], spans, labels


def seed_all(torch, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(torch, transformers, model_path, seed, device):
    seed_all(torch, seed)
    encoder, info = transformers.BertModel.from_pretrained(str(model_path), local_files_only=True,
        add_pooling_layer=False, output_loading_info=True, attn_implementation='eager')
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('Unexpected missing encoder weights: ' + str(info))
    encoder.gradient_checkpointing_enable()
    model = model_class(torch)(encoder).to(device)
    optimizer = torch.optim.AdamW([
        {'params': list(model.encoder.parameters()), 'lr': 5e-5},
        {'params': [p for name, p in model.named_parameters() if not name.startswith('encoder.')], 'lr': 1e-3}
    ], weight_decay=0.0, foreach=False)
    return model, optimizer


def train_step(torch, model, optimizer, rows, indices, device, pad_id, microbatch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    squared_error = 0.0
    for start in range(0, len(indices), microbatch):
        chunk = indices[start:start+microbatch]
        text, knowledge, spans, labels = collate(torch, rows, chunk, device, pad_id)
        pred = model(text, knowledge, spans)
        loss_sum = torch.nn.functional.mse_loss(pred, labels, reduction='sum')
        if not torch.isfinite(loss_sum):
            raise RuntimeError('Nonfinite MSE')
        (loss_sum / len(indices)).backward()
        squared_error += loss_sum.detach().item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return squared_error


def smoke(torch, model, optimizer, rows, device, pad_id, microbatch):
    longest = max(range(len(rows)), key=lambda i: len(rows[i]['text']['input_ids']) + len(rows[i]['knowledge']['input_ids']))
    indices = [longest]
    text, knowledge, spans, labels = collate(torch, rows, indices, device, pad_id)
    model.eval()
    with torch.no_grad():
        pred, details = model(text, knowledge, spans, details=True)
        if pred.shape != (1,) or not torch.isfinite(pred).all():
            raise ValueError('Invalid forward output')
        if not torch.allclose(details['alpha'].sum(1), torch.ones(1, device=device), atol=1e-6):
            raise ValueError('Entity attention does not sum to one')
        if not torch.equal(details['attention'], torch.ones_like(details['attention'])):
            raise ValueError('Equation 18 singleton contract violated')
        pred_empty, empty = model(text, knowledge, [[]], details=True)
        if not torch.isfinite(pred_empty).all() or not torch.equal(empty['fused'], empty['text_cls']):
            raise ValueError('Empty-entity fallback invalid')
        # Different entity counts exercise padding masks within a batch.
        duplicate_text = {k: v.repeat(2, 1) for k, v in text.items()}
        duplicate_knowledge = {k: v.repeat(2, 1) for k, v in knowledge.items()}
        _, masked = model(duplicate_text, duplicate_knowledge, [spans[0], []], details=True)
        if masked['alpha'][1].abs().sum().item() != 0:
            raise ValueError('Padded entities receive nonzero weight')
    train_step(torch, model, optimizer, rows, [longest, 0, 885], device, pad_id, microbatch)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError('Nonfinite gradients: ' + name)
    for name in ('encoder.embeddings.word_embeddings.weight', 'entity_hidden.weight',
                 'entity_score.weight', 'value.weight', 'output.weight', 'regressor.weight'):
        parameter = dict(model.named_parameters())[name]
        if parameter.grad is None or parameter.grad.abs().sum().item() == 0:
            raise ValueError('Expected gradient missing: ' + name)
    return {'status': 'SAE_SMOKE_OK', 'longest_record_id': rows[longest]['id'],
            'checks': ['forward finite', 'entity weights normalized', 'padded entities masked',
                       'empty entities return text-only representation', 'backward and optimizer step',
                       'literal singleton attention is one'],
            'query_key_gradient_note': 'Zero gradients are expected for singleton softmax, per literal equation 18',
            'smoke_weights_used_for_training': False}


def architecture_checks(torch, transformers):
    """Real PyTorch checks on a small CPU BERT before expensive training."""
    import copy
    import io
    seed_all(torch, 42)
    cfg = transformers.BertConfig(vocab_size=128, hidden_size=24, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=48, max_position_embeddings=64,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    cfg._attn_implementation = 'eager'
    model = model_class(torch)(transformers.BertModel(cfg, add_pooling_layer=False))
    def seq(ids):
        return {'input_ids': ids, 'attention_mask': [1]*len(ids), 'token_type_ids': [0]*len(ids)}
    rows = [
        {'label': -0.1, 'text': seq([1, 5, 6, 2]), 'knowledge': seq([1, 7, 8, 9, 2]),
         'entity_spans': [{'token_span': [1, 2]}, {'token_span': [2, 4]}]},
        {'label': 2.0, 'text': seq([1, 10, 11, 12, 13, 2]),
         'knowledge': seq([1, 14, 15, 16, 17, 18, 2]),
         'entity_spans': [{'token_span': [1, 3]}]},
        {'label': 0.0, 'text': seq([1, 2, 2]), 'knowledge': seq([1, 2, 2]), 'entity_spans': []}]
    device = torch.device('cpu')
    batch = collate(torch, rows, [0, 1, 2], device, 0)
    model.eval()
    with torch.no_grad():
        prediction, detail = model(*batch[:3], details=True)
        one = collate(torch, rows, [0], device, 0)
        if not torch.allclose(prediction[:1], model(*one[:3]), atol=2e-6, rtol=2e-6):
            raise ValueError('Architecture check: padding changes prediction')
        if not torch.equal(detail['fused'][2], detail['text_cls'][2]):
            raise ValueError('Architecture check: empty entity fallback')
        if detail['alpha'][1, 1:].abs().sum().item() != 0:
            raise ValueError('Architecture check: entity padding mask')
        # By the literal singleton equation, changing Q/K cannot change predictions.
        model.query.weight.add_(1.0)
        model.key.weight.sub_(1.0)
        if not torch.allclose(prediction, model(*batch[:3]), atol=1e-6, rtol=1e-6):
            raise ValueError('Architecture check: singleton attention interpretation')
    other = copy.deepcopy(model)
    model.zero_grad(set_to_none=True)
    other.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(*batch[:3]), batch[3], reduction='mean').backward()
    for i in range(len(rows)):
        part = collate(torch, rows, [i], device, 0)
        (torch.nn.functional.mse_loss(other(*part[:3]), part[3], reduction='sum')/len(rows)).backward()
    for (name, a), (_, b) in zip(model.named_parameters(), other.named_parameters()):
        if a.grad is None or b.grad is None:
            if a.grad is not b.grad:
                raise ValueError('Architecture check: gradient presence mismatch ' + name)
        elif not torch.allclose(a.grad, b.grad, atol=2e-5, rtol=2e-4):
            raise ValueError('Architecture check: accumulation differs ' + name)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    other.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=False))
    with torch.no_grad():
        if not torch.equal(model(*batch[:3]), other(*batch[:3])):
            raise ValueError('Architecture check: serialized model roundtrip')
    return {'status': 'SAE_ARCHITECTURE_CHECKS_OK',
            'checks': ['token padding invariance', 'entity padding mask', 'empty entities',
                       'singleton Q/K invariance', 'gradient accumulation equivalence',
                       'serialized state roundtrip'], 'device': 'CPU tiny BERT', 'api_calls': 0}


def load_inputs(root):
    folder = root / 'paper_method_v1/artifacts/sae_inputs_raw_beta_v1'
    summary = read(folder / 'summary.json')
    if summary['status'] != 'SAE_TRAIN_INPUTS_READY':
        raise ValueError('SAE inputs not ready')
    if sha(folder / 'config.json') != INPUT_CONFIG_SHA:
        raise ValueError('Unexpected tokenization config')
    for name, expected in summary['files_sha256'].items():
        if sha(folder / name) != expected:
            raise ValueError('Input file hash mismatch: ' + name)
    if sha(folder / 'train_features.jsonl') != FEATURE_SHA:
        raise ValueError('Unexpected feature dataset')
    rows = [json.loads(s) for s in (folder / 'train_features.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    if len(rows) != 2212 or len({r['id'] for r in rows}) != 2212:
        raise ValueError('Feature count/identity mismatch')
    if Counter(r['kind'] for r in rows) != {'original_training': 885, 'synthetic_candidate': 1327}:
        raise ValueError('Source counts mismatch')
    for r in rows:
        if not math.isfinite(r['label']) or ':train:' not in r['sample_id']:
            raise ValueError('Invalid label/source')
        for name in ('text', 'knowledge'):
            n = len(r[name]['input_ids'])
            if not 3 <= n <= 512 or any(len(r[name][k]) != n for k in ('attention_mask', 'token_type_ids')):
                raise ValueError('Invalid input lengths')
        for e in r['entity_spans']:
            a, b = e['token_span']
            if not 0 < a < b < len(r['knowledge']['input_ids']):
                raise ValueError('Invalid entity span')
    return rows, read(folder / 'config.json')


def run(args, out):
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    root = args.root.resolve()
    rows, input_config = load_inputs(root)
    # Completed runs are verified without reserving a GPU.
    if (out / 'complete.json').exists():
        report = read(out / 'complete.json')
        cfg = read(out / 'config.json')
        if cfg['script_sha256'] != sha(Path(__file__)) or cfg['epochs'] != args.epochs or cfg['seed'] != args.seed:
            raise ValueError('Completed run protocol mismatch')
        if cfg['microbatch'] != args.microbatch or report['config_sha256'] != sha(out / 'config.json'):
            raise ValueError('Completed run config mismatch')
        for name, expected in report['files_sha256'].items():
            if sha(out / name) != expected:
                raise ValueError('Completed run changed: ' + name)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print('ALREADY_COMPLETE_VERIFIED; no training repeated')
        return
    if shutil.disk_usage(out).free < 8 * 1024**3:
        raise RuntimeError('Need >=8 GiB free for checkpoints and final weights')
    gpu = select_gpu(args.physical_gpu)
    import torch
    import transformers
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device('cuda:0')
    model_path = root / 'models/bert-base-chinese' / REV
    for name, expected in input_config['tokenizer_files_sha256'].items():
        if sha(model_path / name) != expected:
            raise ValueError('BERT/tokenizer configuration changed: ' + name)
    model_files = {str(p.relative_to(model_path)): sha(p) for p in sorted(model_path.rglob('*'))
                   if p.is_file() and p.suffix in ('.json', '.txt', '.bin', '.safetensors') and '.cache' not in p.parts}
    if not any(n.endswith(('.bin', '.safetensors')) for n in model_files):
        raise ValueError('Missing local pretrained weights')
    config = {'protocol': 'sae_literal_eq15_20_raw_beta_fixed10_v1', 'epochs': args.epochs,
        'seed': args.seed, 'microbatch': args.microbatch, 'script_sha256': sha(Path(__file__)),
        'input_config_sha256': INPUT_CONFIG_SHA, 'features_sha256': FEATURE_SHA,
        'model_revision': REV, 'model_files_sha256': model_files,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'initialization': 'Original pretrained BERT, not trained annotator weights; shared text/entity encoder',
        'tuple_pooling': 'Mean over complete template-phrase token span',
        'entity_pooling': 'Equation 15-16 learned additive attention; D_star equals encoder hidden size',
        'fusion': 'Equations 17-19 literal 1x1 attention; q/k have zero gradients; value/output trainable',
        'final_pooling': 'Max across one fused CLS sequence position is identity; linear scalar regressor',
        'heads': read(model_path / 'config.json')['num_attention_heads'],
        'original_effective_batch': 16, 'synthetic_effective_batch': 64,
        'batch_schedule': 'Shuffle homogeneous batches together; each record once per epoch; no staged pretraining claim',
        'encoder_lr': 5e-5, 'other_lr': 1e-3, 'optimizer': 'AdamW foreach=False',
        'weight_decay': 0.0, 'scheduler': 'constant', 'precision': 'float32',
        'gradient_clip': 1.0, 'checkpoint_selection': 'fixed last epoch; no validation-based best claim',
        'checkpoint_every_optimizer_steps': 20, 'beta_scale_verified': False,
        'loss': 'MSE on unmodified raw labels, including 11 negative synthetic predictions',
        'experiment_status': 'EXPERIMENTAL_REPRODUCTION',
        'deviations_and_choices': ['GLM substitution and adapted ADS split', 'Unverified filtering k and beta scale',
            'Proxy review/fallback and two reviewed entity-span alignments retained upstream',
            'Multiple references concatenated and streams truncated to 512',
            'Mean tuple pooling, encoder sharing, attention size and literal singleton fusion',
            'Mixed homogeneous batches, fixed epochs, optimizer details and last-epoch selection',
            'One seed only; not paper five-run evaluation'],
        'validation_accessed': False, 'test_accessed': False, 'api_calls': 0}
    save(out / 'config.json', config, frozen=True)
    config_sha = sha(out / 'config.json')
    if not (out / 'architecture_checks.json').exists():
        checks = architecture_checks(torch, transformers)
        checks['config_sha256'] = config_sha
        save(out / 'architecture_checks.json', checks, frozen=True)
        print('SAE_ARCHITECTURE_CHECKS_OK', flush=True)
    elif read(out / 'architecture_checks.json')['config_sha256'] != config_sha:
        raise ValueError('Architecture check config mismatch')
    pad_id = read(model_path / 'config.json').get('pad_token_id', 0)
    model, optimizer = create_model(torch, transformers, model_path, args.seed, device)
    if not (out / 'smoke.json').exists():
        result = smoke(torch, model, optimizer, rows, device, pad_id, args.microbatch)
        result['config_sha256'] = config_sha
        save(out / 'smoke.json', result, frozen=True)
        print('SAE_SMOKE_OK; reset pretrained model before training', flush=True)
        del model, optimizer
        torch.cuda.empty_cache()
        model, optimizer = create_model(torch, transformers, model_path, args.seed, device)
    elif read(out / 'smoke.json')['config_sha256'] != config_sha:
        raise ValueError('Smoke config mismatch')
    latest = out / 'latest.pt'
    state = {'epoch': 0, 'next_batch': 0, 'optimizer_steps': 0, 'epoch_sse': 0.0,
             'epoch_records': 0, 'history': []}
    if latest.exists():
        # Only load our own locally generated checkpoint, never an external pickle.
        checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
        if checkpoint['config_sha256'] != config_sha:
            raise ValueError('Checkpoint config mismatch')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        state = checkpoint['state']
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        random.setstate(checkpoint['python_rng'])
        del checkpoint
        print('RESUMED', json.dumps(state, ensure_ascii=False), flush=True)

    def checkpoint_save():
        torch_save(torch, latest, {'config_sha256': config_sha, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'state': state, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate()})
        save(out / 'progress.json', dict(state, status='TRAINING', config_sha256=config_sha, gpu=gpu))

    for epoch in range(state['epoch'], args.epochs):
        schedule = batches(rows, epoch, args.seed)
        for batch_index in range(state['next_batch'], len(schedule)):
            indices = schedule[batch_index]
            error = train_step(torch, model, optimizer, rows, indices, device, pad_id, args.microbatch)
            state['epoch_sse'] += error
            state['epoch_records'] += len(indices)
            state['next_batch'] = batch_index + 1
            state['optimizer_steps'] += 1
            if state['optimizer_steps'] % 10 == 0 or batch_index + 1 == len(schedule):
                print('EPOCH %d STEP %d/%d BATCH_MSE %.6f' %
                      (epoch+1, batch_index+1, len(schedule), error/len(indices)), flush=True)
            if state['optimizer_steps'] % 20 == 0:
                checkpoint_save()
        if state['epoch_records'] != len(rows):
            raise ValueError('Epoch did not cover each record exactly once')
        history = {'epoch': epoch+1, 'train_mse_with_dropout': state['epoch_sse']/len(rows),
                   'records': state['epoch_records']}
        state['history'].append(history)
        state.update(epoch=epoch+1, next_batch=0, epoch_sse=0.0, epoch_records=0)
        checkpoint_save()
        save(out / 'history.json', state['history'])
        print('EPOCH_SAVED', json.dumps(history), flush=True)
    final = out / 'model_final.pt'
    torch_save(torch, final, {'config_sha256': config_sha, 'model': model.state_dict(),
                             'completed_epochs': args.epochs})
    # Release training objects, reload serialized final weights into a fresh model.
    del model, optimizer
    torch.cuda.empty_cache()
    model, optimizer = create_model(torch, transformers, model_path, args.seed, device)
    del optimizer
    checkpoint = torch.load(final, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.microbatch):
            indices = list(range(start, min(start+args.microbatch, len(rows))))
            text, knowledge, spans, labels = collate(torch, rows, indices, device, pad_id)
            pred = model(text, knowledge, spans)
            if not torch.isfinite(pred).all():
                raise ValueError('Nonfinite reloaded-model inference')
            predictions.extend({'id': rows[i]['id'], 'kind': rows[i]['kind'],
                                'label': rows[i]['label'], 'prediction': float(v)}
                               for i, v in zip(indices, pred.cpu().tolist()))
            if start % 200 == 0:
                print('TRAIN_DIAGNOSTICS %d/%d' % (len(predictions), len(rows)), flush=True)
    save(out / 'train_predictions.json', predictions)
    diagnostics = {'interpretation': 'Training-only fit diagnostics; not validation/test results', 'by_kind': {}}
    for kind in ('original_training', 'synthetic_candidate'):
        subset = [r for r in predictions if r['kind'] == kind]
        mse = sum((r['prediction']-r['label'])**2 for r in subset)/len(subset)
        diagnostics['by_kind'][kind] = {'records': len(subset), 'mse': mse, 'rmse': math.sqrt(mse)}
    save(out / 'diagnostics.json', diagnostics)
    report = {'status': 'SAE_EXPERIMENTAL_TRAINING_COMPLETE', 'completed_epochs': args.epochs,
        'training_records': len(rows), 'optimizer_steps': state['optimizer_steps'],
        'seed': args.seed, 'checkpoint_selection': 'last_epoch', 'config_sha256': config_sha,
        'final_weights_reload_verified': True, 'experiment_status': 'EXPERIMENTAL_REPRODUCTION',
        'validation_accessed': False, 'test_accessed': False, 'api_calls': 0,
        'paper_five_run_evaluation_complete': False,
        'files_sha256': {name: sha(out / name) for name in
            ('model_final.pt', 'history.json', 'diagnostics.json', 'train_predictions.json', 'smoke.json', 'architecture_checks.json')},
        'next_step': 'Freeze validation/test protocol and run baseline/comparison experiments before any formal result claim'}
    save(out / 'complete.json', report)
    save(out / 'progress.json', dict(state, status='COMPLETE', config_sha256=config_sha))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('\aSAE_PIPELINE_DONE', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--physical-gpu', default='auto')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--microbatch', type=int, default=1)
    args = p.parse_args()
    if args.epochs < 1 or args.microbatch < 1:
        p.error('epochs and microbatch must be positive')
    if args.physical_gpu != 'auto' and not args.physical_gpu.isdigit():
        p.error('physical-gpu must be auto or an index')
    out = args.root.resolve() / 'paper_method_v1/runs/sae_raw_beta_seed42_v1'
    if args.seed != 42:
        p.error('This frozen run ID is reserved for seed 42')
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args, out)
        except Exception as exc:
            save(out / 'failure.json', {'time': time.time(), 'type': type(exc).__name__, 'detail': str(exc)})
            raise


if __name__ == '__main__':
    main()
