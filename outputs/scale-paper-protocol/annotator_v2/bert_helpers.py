import hashlib, json, math, os
from pathlib import Path


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

def make_model(model_path, torch, transformers, device, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
