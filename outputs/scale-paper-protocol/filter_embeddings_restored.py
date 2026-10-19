"""Offline answer embeddings with audited right truncation to 512 tokens; no labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path

REV = '183bb99aa7af74355fb58d16edf8c13ae7c5433e'
REPO = 'shibing624/text2vec-base-chinese'
FILES = ['config.json', 'model.safetensors', 'vocab.txt',
         'tokenizer_config.json', 'special_tokens_map.json']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode('utf8')


def save(path, value):
    data = json_bytes(value)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing artifact differs: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def load_inputs(root):
    base = root/'paper_method_v2'
    retrieval = base/'artifacts/retrieval_v1'
    generated = base/'artifacts/ps_generated_v1'
    plan = base/'artifacts/ps_inputs_v1'
    source_path = retrieval/'training_quadruples.json'
    candidate_path = generated/'candidates.jsonl'
    upstream = read(retrieval/'summary.json')
    manifest = read(generated/'manifest.json')
    if sha(source_path) != upstream['output_sha256'][source_path.name]:
        raise ValueError('Training source hash mismatch')
    if sha(candidate_path) != manifest['candidate_sha256']:
        raise ValueError('Candidate hash mismatch')
    if sha(plan/'summary.json') != manifest['plan_sha256']:
        raise ValueError('Generation plan hash mismatch')
    plan_summary = read(plan/'summary.json')
    for name, expected in plan_summary['files_sha256'].items():
        if sha(plan/name) != expected:
            raise ValueError('Generation plan file changed: '+name)
    reqs = [json.loads(x) for x in (plan/'requests.jsonl').read_text(encoding='utf8').splitlines() if x.strip()]
    expected = {}
    for request in reqs:
        for variant in request['provenance']:
            vid = variant['variant_id']
            if vid in expected:
                raise ValueError('Duplicate planned variant ID')
            expected[vid] = (request['request_id'], request['sample_id'])
    sources = read(source_path)
    candidates = [json.loads(x) for x in candidate_path.read_text(encoding='utf8').splitlines() if x.strip()]
    source_ids = {x['sample_id'] for x in sources}
    if len(source_ids) != len(sources):
        raise ValueError('Duplicate source IDs')
    rows = []
    for item in sources:
        if not isinstance(item['a'], str):
            raise ValueError('Invalid original answer')
        rows.append({'kind':'original', 'id':item['sample_id'],
                     'sample_id':item['sample_id'], 'text':item['a']})
    seen = set()
    for item in candidates:
        vid = item['variant_id']
        if vid in seen or expected.get(vid) != (item['request_id'], item['sample_id']):
            raise ValueError('Candidate provenance mismatch: '+vid)
        if item['sample_id'] not in source_ids or item.get('assigned_label') is not None:
            raise ValueError('Unexpected candidate source or label')
        if not isinstance(item['answer'], str) or not item['answer'].strip():
            raise ValueError('Empty candidate answer')
        seen.add(vid)
        rows.append({'kind':'candidate', 'id':vid, 'sample_id':item['sample_id'], 'text':item['answer']})
    if seen != set(expected):
        raise ValueError('Generation is incomplete')
    # Share computation only for exactly equal text; retain all sample records.
    texts = []
    index = {}
    records = []
    for row in rows:
        text = row.pop('text')
        if text not in index:
            index[text] = len(texts)
            texts.append(text)
        records.append(dict(row, vector_row=index[text]))
    return texts, records, {
        'training_records':len(sources), 'candidate_records':len(candidates),
        'unique_answer_texts':len(texts),
        'blank_original_answers':sum(not x['a'].strip() for x in sources),
        'sources_sha256':sha(source_path), 'candidates_sha256':sha(candidate_path),
        'generation_manifest_sha256':sha(generated/'manifest.json'),
        'generation_plan_sha256':sha(plan/'summary.json')}


def validate_vectors(vec, count):
    import numpy as np
    if vec.ndim != 2 or vec.shape != (count,768) or not np.isfinite(vec).all():
        raise ValueError('Invalid vector dimensions or values')
    if not np.allclose(np.linalg.norm(vec,axis=1),1,atol=1e-5):
        raise ValueError('Invalid vector norms')


def encode_batch(tokenizer, texts):
    tokenizer.truncation_side = 'right'
    return tokenizer(texts, padding=True, truncation=True,
                     max_length=512, return_tensors='pt')


def execute(root, command):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import fcntl
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModel
    texts, records, facts = load_inputs(root)
    modeldir = root/'models/text2vec-base-chinese'/REV
    manifest = read(modeldir/'offline_manifest.json')
    if manifest['repo_id'] != REPO or manifest['revision'] != REV:
        raise ValueError('Unexpected embedding model')
    for name in FILES:
        if sha(modeldir/name) != manifest['files_sha256'][name]:
            raise ValueError('Model hash mismatch: '+name)
    out = root/'paper_method_v2/artifacts/initial_filter_embeddings_v2_max512'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg = dict(facts, model=REPO, revision=REV,
                   model_manifest_sha256=sha(modeldir/'offline_manifest.json'),
                   script_sha256=sha(Path(__file__)), device='cpu', threads=4,
                   batch_size=8, chunk_size=64, truncation=True,
                   max_length=512, truncation_side='right',
                   length_policy='Retain leading answer tokens within 512 including special tokens; full source texts unchanged.',
                   length_policy_basis='Explicit reproduction implementation choice; not stated in the paper.',
                   pooling='attention-mask mean including special tokens; L2 normalization',
                   embedding_input='answer text only, original text preserved',
                   duplicate_policy='share vectors for exact text matches; retain every record',
                   blank_original_policy='retain and encode special tokens; explicit adapted-data behavior',
                   paper_filter_budget=5000, filter_k=None,
                   filter_k_note='Paper experimental settings print k=a without a numeric value; no k assumed.',
                   torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__)
        save(out/'config.json',cfg)
        save(out/'records.json',records)
        save(out/'texts.json',texts)
        tokenizer = AutoTokenizer.from_pretrained(str(modeldir),local_files_only=True,trust_remote_code=False)
        tokenizer.truncation_side = 'right'
        max_length = read(modeldir/'config.json')['max_position_embeddings']
        if max_length != 512:
            raise ValueError('Expected fixed 512-token model')
        lengths = []
        for start in range(0,len(texts),64):
            encoded = tokenizer(texts[start:start+64],truncation=False,padding=False)
            lengths.extend(len(x) for x in encoded['input_ids'])
        long_rows = [{'vector_row':i,'tokens':n,'omitted_tokens':n-max_length,
                      'affected_records':[r for r in records if r['vector_row']==i]}
                     for i,n in enumerate(lengths) if n>max_length]
        # Verify and record precisely which model input IDs are retained/omitted.
        # No text is decoded or rewritten, including literal backslash-n sequences.
        for entry in long_rows:
            original = tokenizer(texts[entry['vector_row']],add_special_tokens=False,truncation=False)['input_ids']
            actual = tokenizer(texts[entry['vector_row']],truncation=True,max_length=512)['input_ids']
            special_count = tokenizer.num_special_tokens_to_add(pair=False)
            capacity = 512-special_count
            expected_ids = tokenizer.build_inputs_with_special_tokens(original[:capacity])
            if tokenizer.truncation_side != 'right' or actual != expected_ids or len(actual)!=512:
                raise ValueError('Unexpected truncation behavior')
            entry['omitted_token_ids'] = original[capacity:]
            entry['encoded_input_ids_sha256'] = hashlib.sha256(json_bytes(actual)).hexdigest()
            if len(entry['omitted_token_ids']) != entry['omitted_tokens']:
                raise ValueError('Truncation token count mismatch')
        save(out/'truncation_audit.json',long_rows)
        preflight = dict(facts, status='FILTER_EMBEDDING_PREFLIGHT_OK' if not long_rows else 'FILTER_EMBEDDING_PREFLIGHT_OK_WITH_TRUNCATION',
                         max_answer_tokens=max(lengths), model_max_tokens=max_length,
                         overlength_count=len(long_rows), overlength=long_rows,
                         length_policy='right_truncation_512_including_special_tokens',
                         full_texts_preserved=True,
                         api_calls=0, training_started=False, test_accessed=False, offline=True)
        save(out/'preflight.json',preflight)
        print(json.dumps(preflight,ensure_ascii=False,indent=2),flush=True)
        if command == 'preflight':
            return
        torch.set_num_threads(4)
        torch.manual_seed(42)
        model = None
        chunks = out/'chunks'
        chunks.mkdir(exist_ok=True)
        outputs = []
        for start in range(0,len(texts),64):
            block = texts[start:start+64]
            path = chunks/('%06d.npy'%start)
            meta_path = path.with_suffix('.json')
            block_hash = hashlib.sha256(json_bytes(block)).hexdigest()
            if path.exists() and meta_path.exists():
                meta = read(meta_path)
                if meta['sha256'] != sha(path) or meta['text_block_sha256'] != block_hash:
                    raise ValueError('Changed embedding chunk')
                vec = np.load(path,allow_pickle=False)
            else:
                if model is None:
                    model = AutoModel.from_pretrained(str(modeldir),local_files_only=True,
                                                     trust_remote_code=False,use_safetensors=True).eval()
                parts = []
                for offset in range(0,len(block),8):
                    inp = encode_batch(tokenizer,block[offset:offset+8])
                    if inp['input_ids'].shape[1] > 512:
                        raise ValueError('Encoded batch exceeds model capacity')
                    with torch.inference_mode():
                        hidden = model(**inp).last_hidden_state
                        mask = inp['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                        pooled = (hidden*mask).sum(1)/mask.sum(1)
                    arr = pooled.cpu().numpy()
                    norm = np.linalg.norm(arr,axis=1,keepdims=True)
                    if (norm<=0).any():
                        raise ValueError('Zero vector')
                    parts.append((arr/norm).astype('float32'))
                vec = np.concatenate(parts)
                validate_vectors(vec,len(block))
                temp = path.with_suffix('.tmp')
                with temp.open('wb') as handle:
                    np.save(handle,vec,allow_pickle=False)
                temp.replace(path)
                save(meta_path,{'sha256':sha(path),'text_block_sha256':block_hash})
            validate_vectors(vec,len(block))
            outputs.append({'start':start,'count':len(block),'file':str(path.relative_to(out)), 'sha256':sha(path)})
            print('ANSWER_EMBEDDED',min(start+64,len(texts)),'/',len(texts),flush=True)
        report = dict(facts,status='INITIAL_FILTER_EMBEDDINGS_READY',embedding_dimension=768,
                      config_sha256=sha(out/'config.json'),records_sha256=sha(out/'records.json'),
                      texts_sha256=sha(out/'texts.json'),chunks=outputs,
                      length_policy='right_truncation_512_including_special_tokens',
                      truncated_unique_texts=len(long_rows),
                      truncation_audit_sha256=sha(out/'truncation_audit.json'),
                      filtering_executed=False, selected_candidates=0,assigned_labels=0,
                      api_calls=0,training_started=False,test_accessed=False,offline=True)
        save(out/'summary.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='chunks'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight','run'])
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    execute(args.root.resolve(),args.command)
