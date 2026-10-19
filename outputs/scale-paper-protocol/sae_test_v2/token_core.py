"""Offline SCALE III-C two-stream token inputs, with audited entity token spans."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

REV = '8f23c25b06e129b6c986331a13d8d025a92cf0ea'
BRANCH_SHA = '4f4f1a3b98160c0b08e111f4bc904ad93e331255cfbb1aa154ec6fd483c3d9e8'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(path.read_bytes())


def enc(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf8')


def jl(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False)
                   + '\n' for r in rows).encode('utf8')


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]


def write(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Frozen output differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def entity_text(sources):
    """Template slots t/c; ordered distinct pairs, with all source occurrences retained."""
    text = ''
    entities = []
    seen = {}
    for slot, source, triples in sources:
        if not isinstance(triples, list):
            raise ValueError('Triples must be a list')
        for index, triple in enumerate(triples):
            if set(triple) != {'t', 'c', 'w'} or any(
                    not isinstance(v, str) or not v.strip() for v in triple.values()):
                raise ValueError('Invalid extraction schema')
            t, c = triple['t'], triple['c']
            if t not in source:
                raise ValueError('Nonverbatim key phrase')
            key = (t, c)
            origin = {'slot': slot, 'triple_index': index}
            if key in seen:
                entities[seen[key]]['origins'].append(origin)
                continue
            if text:
                text += '\n'
            start = len(text)
            phrase = t + ' is a ' + c
            text += phrase
            seen[key] = len(entities)
            entities.append({'t': t, 'c': c, 'char_span': [start, len(text)],
                             'origins': [origin]})
    return text, entities


def stream(batch):
    return {key: list(batch[key]) for key in ('input_ids', 'attention_mask', 'token_type_ids')}


def align_entities(full, kept, groups):
    """Accept complete tuple spans only; offsets are local to each paired sequence."""
    full_seq, kept_seq = full.sequence_ids(), kept.sequence_ids()
    accepted, dropped = [], []
    for side, entities in enumerate(groups):
        for entity in entities:
            start, end = entity['char_span']
            def positions(batch, sides):
                return [i for i, (a, b) in enumerate(batch['offset_mapping'])
                        if sides[i] == side and b > a and a < end and b > start]
            expected = positions(full, full_seq)
            actual = positions(kept, kept_seq)
            signature = lambda batch, indices: [
                (batch['input_ids'][i], tuple(batch['offset_mapping'][i])) for i in indices]
            if not expected:
                raise ValueError('Nonempty entity has no tokens')
            detail = dict(entity, sequence_id=side)
            if signature(full, expected) != signature(kept, actual):
                dropped.append(dict(detail, full_tokens=len(expected), retained_tokens=len(actual),
                                    reason='TRUNCATED_ENTITY_NOT_USED_FOR_POOLING'))
                continue
            if actual != list(range(actual[0], actual[-1] + 1)):
                raise ValueError('Noncontiguous entity span')
            accepted.append(dict(detail, token_span=[actual[0], actual[-1] + 1]))
    return accepted, dropped


def encode_row(row, tokenizer):
    refs = row['references']
    if len(refs) != len(row['reference_triples']):
        raise ValueError('Reference slot count mismatch')
    ref_text = '\n'.join(s.strip() for s in refs if s.strip())
    if not ref_text:
        raise ValueError('Missing reference')
    ref_entities_text, ref_entities = entity_text([
        (slot, text, triples) for slot, (text, triples) in
        enumerate(zip(refs, row['reference_triples']))])
    ans_entities_text, ans_entities = entity_text([(0, row['answer'], row['answer_triples'])])
    text_full = tokenizer(ref_text, row['answer'], truncation=False, verbose=False)
    text_kept = tokenizer(ref_text, row['answer'], truncation='longest_first',
                          max_length=512, verbose=False)
    knowledge_full = tokenizer(ref_entities_text, ans_entities_text, truncation=False,
                               return_offsets_mapping=True, verbose=False)
    knowledge_kept = tokenizer(ref_entities_text, ans_entities_text, truncation='longest_first',
                               max_length=512, return_offsets_mapping=True, verbose=False)
    accepted, dropped = align_entities(knowledge_full, knowledge_kept, [ref_entities, ans_entities])
    result = {'id': row['id'], 'sample_id': row['sample_id'], 'kind': row['kind'],
              'label': row['label'], 'text': stream(text_kept), 'knowledge': stream(knowledge_kept),
              'entity_spans': accepted, 'has_entities': bool(accepted),
              'knowledge_reference_text': ref_entities_text,
              'knowledge_answer_text': ans_entities_text}
    audit = {'id': row['id'], 'text_full_tokens': len(text_full['input_ids']),
             'text_kept_tokens': len(text_kept['input_ids']),
             'knowledge_full_tokens': len(knowledge_full['input_ids']),
             'knowledge_kept_tokens': len(knowledge_kept['input_ids']),
             'entities_before_truncation': len(ref_entities) + len(ans_entities),
             'entities_kept': len(accepted), 'dropped_entities': dropped}
    validate_feature(result)
    return result, audit


def validate_feature(row):
    if not math.isfinite(row['label']):
        raise ValueError('Nonfinite label')
    for name in ('text', 'knowledge'):
        item = row[name]
        n = len(item['input_ids'])
        if not 3 <= n <= 512 or any(len(item[k]) != n for k in ('attention_mask', 'token_type_ids')):
            raise ValueError('Invalid stream shape')
        if item['attention_mask'] != [1] * n:
            raise ValueError('Unexpected padding; pad at batching time')
    for entity in row['entity_spans']:
        start, end = entity['token_span']
        if not 0 < start < end < len(row['knowledge']['input_ids']):
            raise ValueError('Entity span out of bounds')
        if any(x != entity['sequence_id'] for x in row['knowledge']['token_type_ids'][start:end]):
            raise ValueError('Entity crosses paired-sequence boundary')
    if row['has_entities'] != bool(row['entity_spans']):
        raise ValueError('Invalid entity mask')


def load_sources(root):
    run = root / 'paper_method_v1/runs/train_entities_raw_beta_v1'
    summary = read(run / 'summary.json')
    if summary['status'] != 'TRAIN_ENTITY_EXTRACTIONS_COMPLETE' or summary['remaining'] != 0:
        raise ValueError('Entity extraction incomplete')
    path = run / 'training_with_entities.jsonl'
    if sha(path) != summary['files_sha256'][path.name]:
        raise ValueError('Entity export hash mismatch')
    review = run / 'reviewed_span_alignment_v1/audit.json'
    if sha(review) != summary['reviewed_span_alignment']['audit_sha256']:
        raise ValueError('Reviewed alignment audit changed')
    original = run / 'training_inputs.jsonl'
    if sha(original) != BRANCH_SHA:
        raise ValueError('Unexpected experimental branch')
    rows, originals = lines(path), lines(original)
    if len(rows) != 2212 or len({r['id'] for r in rows}) != 2212 or len(originals) != len(rows):
        raise ValueError('Training IDs/count mismatch')
    for row, source in zip(rows, originals):
        if set(row) != set(source) | {'answer_triples', 'reference_triples'}:
            raise ValueError('Unexpected exported fields')
        if any(row[k] != v for k, v in source.items()):
            raise ValueError('Original input/label changed')
        if ':train:' not in row['sample_id']:
            raise ValueError('Non-training source')
    if Counter(r['kind'] for r in rows) != {'original_training': 885, 'synthetic_candidate': 1327}:
        raise ValueError('Training branch composition mismatch')
    return rows, {'training_inputs_sha256': sha(original), 'entity_export_sha256': sha(path),
                  'alignment_audit_sha256': sha(review)}


def verify(folder, rows, expected_config):
    config = read(folder / 'config.json')
    if config != expected_config:
        raise ValueError('Tokenization configuration changed')
    summary = read(folder / 'summary.json')
    if summary['config_sha256'] != sha(folder / 'config.json'):
        raise ValueError('Configuration hash mismatch')
    for name, value in summary['files_sha256'].items():
        if sha(folder / name) != value:
            raise ValueError('Output hash mismatch: ' + name)
    encoded = lines(folder / 'train_features.jsonl')
    if len(encoded) != len(rows):
        raise ValueError('Feature count mismatch')
    for feature, source in zip(encoded, rows):
        if any(feature[k] != source[k] for k in ('id', 'sample_id', 'kind', 'label')):
            raise ValueError('Feature identity/label mismatch')
        validate_feature(feature)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('SAE_INPUTS_VERIFIED: 2212 records; labels unchanged; no API calls')


def execute(args, folder):
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import transformers
    root = args.root.resolve()
    rows, provenance = load_sources(root)
    model = root / 'models/bert-base-chinese' / REV
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('Fast tokenizer required for offset mapping')
    tokenizer.truncation_side = 'right'
    tokenizer.padding_side = 'right'
    if read(model / 'config.json')['max_position_embeddings'] < 512:
        raise ValueError('Unexpected BERT position limit')
    config = dict(provenance, model_revision=REV, tokenizer_class=type(tokenizer).__name__,
        transformers_version=transformers.__version__, max_length=512,
        tokenizer_backend_sha256=digest(tokenizer.backend_tokenizer.to_str().encode('utf8')),
        tokenizer_files_sha256={n: sha(model / n) for n in
            ('config.json', 'vocab.txt', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')
            if (model / n).is_file()},
        script_sha256=sha(Path(__file__)),
        text_input='[CLS] nonempty references joined by newline [SEP] answer [SEP]; no question',
        entity_template='{t} is a {c}; braces denote slots, no literal brackets',
        entity_input='[CLS] reference entity phrases [SEP] answer entity phrases [SEP]',
        entity_order='Extraction order; deduplicate identical (t,c) pairs per side; preserve origins',
        entity_span='Entire template phrase, token start inclusive/end exclusive; special tokens excluded',
        truncation='Right longest_first at 512 per stream; incomplete tuples excluded from entity pooling',
        empty_entity_policy='has_entities=false; trainer must handle explicitly without fabricated entities',
        experiment_status='EXPERIMENTAL_REPRODUCTION', beta_scale_verified=False,
        implementation_choices=['Multiple references joined in slot order',
            'Template punctuation and tuple ordering', 'Per-side (t,c) deduplication',
            '512-token truncation and complete-tuple pooling mask'],
        labels='Raw scores preserved, including negative predictions', api_calls=0,
        validation_accessed=False, test_accessed=False)
    if args.command == 'verify' or (folder / 'summary.json').exists():
        verify(folder, rows, config)
        return
    write(folder / 'config.json', enc(config))
    # Small, real-tokenizer preflight followed immediately by complete construction.
    indices = sorted(set([0, 884, 885, len(rows)-1] +
                         [i for i, r in enumerate(rows) if not r['answer'].strip()][:1]))
    for i in indices:
        encode_row(rows[i], tokenizer)
    print('SAE_TOKENIZER_PREFLIGHT_OK', flush=True)
    features, audit = [], []
    for index, row in enumerate(rows, 1):
        item, report = encode_row(row, tokenizer)
        features.append(item)
        audit.append(report)
        if index % 100 == 0 or index == len(rows):
            print('TOKENIZED %d/%d' % (index, len(rows)), flush=True)
    files = {'train_features.jsonl': jl(features), 'tokenization_audit.jsonl': jl(audit),
             'preview.json': enc(features[:2])}
    for name, data in files.items():
        write(folder / name, data)
    summary = {
        'status': 'SAE_TRAIN_INPUTS_READY', 'training_records': len(rows),
        'original_records': 885, 'synthetic_records': 1327,
        'negative_labels_preserved': sum(r['label'] < 0 for r in features),
        'text_truncated_records': sum(a['text_full_tokens'] > 512 for a in audit),
        'knowledge_truncated_records': sum(a['knowledge_full_tokens'] > 512 for a in audit),
        'entities_before_truncation': sum(a['entities_before_truncation'] for a in audit),
        'entities_kept': sum(a['entities_kept'] for a in audit),
        'entities_excluded_by_truncation': sum(len(a['dropped_entities']) for a in audit),
        'records_without_retained_entities': sum(not r['has_entities'] for r in features),
        'config_sha256': sha(folder / 'config.json'),
        'files_sha256': {name: digest(data) for name, data in files.items()},
        'labels_modified': False, 'api_calls': 0, 'offline': True,
        'final_scale_training_started': False, 'validation_accessed': False, 'test_accessed': False,
        'ready_for_model_smoke': True, 'final_model_verified': False,
        'next_step': 'Implement and smoke-test SAE equations 15-20; no final training executed',
        'experiment_status': 'EXPERIMENTAL_REPRODUCTION_BETA_SCALE_UNVERIFIED'}
    write(folder / 'summary.json', enc(summary))
    verify(folder, rows, config)
    print('\aSAE_INPUT_PREPARATION_DONE', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['build', 'verify'])
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    folder = args.root.resolve() / 'paper_method_v1/artifacts/sae_inputs_raw_beta_v1'
    folder.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (folder / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, folder)


if __name__ == '__main__':
    main()
