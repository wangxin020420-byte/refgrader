"""Offline five-run SAE token preparation; never invokes legacy v1 execute()."""
import argparse
from collections import Counter
import os
from pathlib import Path
import token_core as c

COUNTS = {42:1468, 43:1849, 44:1798, 45:2366, 46:1722}
HASHES = {
42:'88d072f7992a0dd88414ce23dca3b8be9dd7f19b49f4555aad79978edf29d094',
43:'fee339b9c3a496e31e1d6a9d3e52c0d56f8dfb159da2bab7b6ec2497820ecf6e',
44:'9801bc0b6c0545fafbb05b253420199d5f5ed2eb18f5bcf168b3ad3e66ecb217',
45:'1bae5408534d88f9672eac4863c79bd4568d7c0b6b1088539a4e35d897d83989',
46:'d8f01e1588ed5dbca9ca5f7ebc26d574acb91ba5c21b3c4f450e74b0f7912b60'}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def load(root):
    run = root/'paper_method_v2/runs/train_entities_five_runs_v1'
    summary = c.read(run/'summary.json')
    require(summary['status'] == 'FIVE_RUN_TRAIN_ENTITIES_COMPLETE' and
            summary['remaining'] == 0 and not summary['errors'], 'Extraction incomplete')
    rows = {}
    for seed, count in COUNTS.items():
        name = 'seed_%d/training_with_entities.jsonl' % seed
        path = run/name
        require(c.sha(path) == HASHES[seed] == summary['files_sha256'][name], 'Entity hash mismatch')
        rows[seed] = c.lines(path)
        source = c.lines(root/('paper_method_v2/artifacts/fine_filter_raw_beta0p2_v1/seed_%d/augmented_inputs.jsonl' % seed))
        require(len(rows[seed]) == len(source) == count, 'Count mismatch')
        require(len({r['id'] for r in source}) == count, 'Duplicate IDs')
        require(Counter(r['kind'] for r in source) ==
                {'original_training':1107, 'synthetic_candidate':count-1107}, 'Composition mismatch')
        for row, original in zip(rows[seed], source):
            require(set(row) == set(original) | {'answer_triples','reference_triples'} and
                    all(row[k] == v for k,v in original.items()), 'Input or label changed')
    return rows

def verify(folder, rows, config):
    require(c.read(folder/'config.json') == config, 'Frozen tokenization config changed')
    summary = c.read(folder/'summary.json')
    require(summary['config_sha256'] == c.sha(folder/'config.json'), 'Config hash mismatch')
    for name, digest in summary['files_sha256'].items():
        require(c.sha(folder/name) == digest, 'Output hash mismatch: '+name)
    features = c.lines(folder/'train_features.jsonl')
    require(len(features) == len(rows), 'Feature count mismatch')
    for f,r in zip(features,rows):
        require(all(f[k] == r[k] for k in ('id','sample_id','kind','label')), 'Feature identity/label mismatch')
        c.validate_feature(f)
    return summary

def build_seed(folder, rows, tokenizer, config, verify_only=False):
    if verify_only or (folder/'summary.json').exists():
        return verify(folder,rows,config)
    c.write(folder/'config.json', c.enc(config))
    features, audits = [], []
    for i,row in enumerate(rows,1):
        f,a = c.encode_row(row,tokenizer)
        features.append(f)
        audits.append(a)
        if i % 100 == 0 or i == len(rows):
            print('TOKENIZED',folder.name,i,'/',len(rows),flush=True)
    files = {'train_features.jsonl':c.jl(features),
             'tokenization_audit.jsonl':c.jl(audits),'preview.json':c.enc(features[:2])}
    for name,data in files.items():
        c.write(folder/name,data)
    summary = dict(status='SAE_TRAIN_INPUTS_READY', training_records=len(rows),
        original_records=1107, synthetic_records=len(rows)-1107,
        negative_labels_preserved=sum(f['label']<0 for f in features),
        above_ten_labels_preserved=sum(f['label']>10 for f in features),
        text_truncated_records=sum(a['text_full_tokens']>512 for a in audits),
        knowledge_truncated_records=sum(a['knowledge_full_tokens']>512 for a in audits),
        entities_before_truncation=sum(a['entities_before_truncation'] for a in audits),
        entities_kept=sum(a['entities_kept'] for a in audits),
        entities_excluded_by_truncation=sum(len(a['dropped_entities']) for a in audits),
        records_without_retained_entities=sum(not f['has_entities'] for f in features),
        config_sha256=c.sha(folder/'config.json'),
        files_sha256={n:c.digest(b) for n,b in files.items()}, labels_modified=False,
        api_calls=0, test_accessed=False, validation_accessed=False, training_started=False)
    c.write(folder/'summary.json',c.enc(summary))
    return verify(folder,rows,config)

def execute(args):
    for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import transformers
    root=args.root.resolve()
    rows=load(root)
    model=root/'models/bert-base-chinese'/c.REV
    tokenizer=transformers.AutoTokenizer.from_pretrained(str(model),local_files_only=True,use_fast=True)
    require(tokenizer.is_fast,'Fast tokenizer required')
    require(c.read(model/'config.json')['max_position_embeddings']>=512,'Model position limit')
    tokenizer.truncation_side='right'
    tokenizer.padding_side='right'
    config=dict(model_revision=c.REV,max_length=512,transformers_version=transformers.__version__,
        tokenizer_backend_sha256=c.digest(tokenizer.backend_tokenizer.to_str().encode('utf8')),
        tokenizer_files_sha256={p.name:c.sha(p) for p in model.iterdir() if p.name in
            ('config.json','vocab.txt','tokenizer.json','tokenizer_config.json','special_tokens_map.json')},
        builder_sha256=c.sha(Path(__file__)),token_core_sha256=c.sha(Path(c.__file__)),
        text_input='reference slots joined by newline; paired answer; no question',
        knowledge_input='reference t is a c phrases; paired answer t is a c phrases',
        entity_span='Complete template phrase; start inclusive/end exclusive; exclude special tokens',
        entity_order='Extraction order, distinct (t,c) per side, retain origins',
        truncation='right longest_first, 512 per stream; exclude incomplete entity spans',
        empty_entity_policy='Explicit has_entities=false; no fabricated entity',
        implementation_choices_not_author_verified=['reference concatenation','template punctuation and ordering',
            'per-side tuple deduplication','512 truncation and complete-entity masking'],
        reproduction_status='METHOD_RECONSTRUCTION_WITH_DOCUMENTED_DEVIATIONS',
        beta_scale_verified_from_author=False,labels='Unchanged raw labels including out-of-range predictions')
    out=root/'paper_method_v2/artifacts/sae_inputs_five_runs_v1'
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reports={}
        for seed in COUNTS:
            cfg=dict(config,seed=seed,entity_input_sha256=HASHES[seed])
            reports[str(seed)]=build_seed(out/('seed_%d'%seed),rows[seed],tokenizer,cfg,args.command=='verify')
        summary=dict(status='FIVE_RUN_SAE_TRAIN_INPUTS_READY',seeds=list(COUNTS),runs=reports,
            ready_for_model_smoke=True,ready_for_final_training=False,training_started=False,
            validation_accessed=False,test_accessed=False,api_calls=0,labels_modified=False,
            next_step='Prepare fixed 50-row validation features with identical encoding; adapt five-run SAE runner to validation-QWK checkpoint selection')
        c.write(out/'summary.json',c.enc(summary))
        print(c.enc(summary).decode('utf8'),flush=True)
        print('FIVE_RUN_SAE_INPUTS_VERIFIED',out,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['build','verify'])
    parser.add_argument('--root',type=Path,required=True)
    execute(parser.parse_args())
