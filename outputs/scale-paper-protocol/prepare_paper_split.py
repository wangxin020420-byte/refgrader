"""Prepare an isolated ADS split following the paper's 50-row validation rule.

No API, training, prediction, or metric evaluation. Existing artifacts are read-only.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf8')


def jsonl(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True,
                              allow_nan=False) + '\n' for r in rows).encode('utf8')


def construct(inputs, labels, seed):
    expected = {'train': 885, 'validation': 222, 'test': 475}
    seen = set()
    for split, count in expected.items():
        rows, scores = inputs[split], labels[split]
        ids = [r['sample_id'] for r in rows]
        score_ids = [r['sample_id'] for r in scores]
        if len(rows) != count or len(scores) != count:
            raise ValueError('Unexpected counts: ' + split)
        if len(set(ids)) != count or len(set(score_ids)) != count or set(ids) != set(score_ids):
            raise ValueError('Duplicate or mismatched identities: ' + split)
        if seen.intersection(ids):
            raise ValueError('Sample ID occurs in multiple old splits')
        seen.update(ids)
        by_id = {r['sample_id']: r for r in scores}
        source = 'ADS/test.jsonl' if split == 'test' else 'ADS/train.jsonl'
        for row in rows:
            if row['source_file'] != source:
                raise ValueError('Unexpected upstream source')
            score = by_id[row['sample_id']]
            if score['question_id'] != row['question_id']:
                raise ValueError('Question identity mismatch')
            gold = score['gold_raw']
            if not isinstance(gold, (int, float)) or not 0 <= gold <= 10:
                raise ValueError('Invalid gold score')
    train = sorted(inputs['train'] + inputs['validation'], key=lambda r:r['source_line'])
    test = sorted(inputs['test'], key=lambda r:r['source_line'])
    if [r['source_line'] for r in train] != list(range(1,1108)):
        raise ValueError('Restored upstream training lines are incomplete')
    if [r['source_line'] for r in test] != list(range(1,476)):
        raise ValueError('Upstream test lines are incomplete')
    selected = set(random.Random(seed).sample(range(475), 50))
    splits = {'train':train, 'validation':[r for i,r in enumerate(test) if i in selected],
              'test':[r for i,r in enumerate(test) if i not in selected]}
    scores = {r['sample_id']:r for group in labels.values() for r in group}
    output = {}
    for name, rows in splits.items():
        output[name+'.inputs.jsonl'] = jsonl(rows)
        output[name+'.labels.jsonl'] = jsonl([scores[r['sample_id']] for r in rows])
    overlap = defaultdict(lambda:defaultdict(list))
    for split, rows in splits.items():
        for r in rows:
            key = digest(json.dumps([r['question_id'],r['question'],r['answer']],
                                    ensure_ascii=False).encode('utf8'))
            overlap[key][split].append(r['sample_id'])
    overlaps = [{'text_sha256':key,'sample_ids':dict(groups)}
                for key,groups in sorted(overlap.items()) if len(groups)>1]
    output['overlap_audit.json'] = encode(overlaps)
    # Lists refer to source identities, not changed labels or replacement text.
    output['restored_training_ids.json'] = encode([r['sample_id'] for r in
                                                  sorted(inputs['validation'],key=lambda r:r['source_line'])])
    return output, splits, len(overlaps)


def execute(root, seed):
    source = root/'data/prepared/ads_public_adapted_v1'
    manifest_bytes = (source/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest['protocol'] != 'ADS_PUBLIC_ADAPTED_V1':
        raise ValueError('Unexpected source protocol')
    inputs, labels, hashes = {}, {}, {}
    for split in ('train','validation','test'):
        for suffix, dest in (('inputs',inputs),('labels',labels)):
            name = split+'.'+suffix+'.jsonl'
            raw = (source/name).read_bytes()
            hashes[name] = digest(raw)
            if hashes[name] != manifest['outputs_sha256'][name]:
                raise ValueError('Source checksum mismatch: '+name)
            dest[split] = [json.loads(s) for s in raw.decode('utf8').splitlines() if s.strip()]
    output, splits, overlaps = construct(inputs,labels,seed)
    plan = {
        'status':'PAPER_SPLIT_PREPARED_UPSTREAM_REBUILD_REQUIRED',
        'scope':'ADS public data; paper-style validation split, not a claim of exact full reproduction',
        'source_manifest_sha256':digest(manifest_bytes),
        'source_files_sha256':hashes,
        'dataset_commit':manifest['dataset_commit'],
        'split_seed':seed,
        'split_policy':'Uniform sample of 50 upstream test rows without replacement; remaining 425 reserved for testing',
        'training_policy':'Restore all 1107 upstream train rows; no cleaning, relabeling, or deduplication',
        'counts':{name:len(rows) for name,rows in splits.items()},
        'restored_training_records':222,
        'cross_split_exact_text_groups':overlaps,
        'overlap_policy':'Audit only; preserve upstream rows and random split, report overlap as limitation',
        'implementation_choices_not_specified_in_paper':[
            'Split seed 42; one fixed split shared by repetitions',
            'Proposed model training seeds 42,43,44,45,46; not author-provided seeds'],
        'paper_requirements_for_next_runner':[
            'Train each compared model five times',
            'Select checkpoint with highest validation QWK for each run',
            'Evaluate selected checkpoints on reserved test rows; average five results'],
        'unresolved_before_claiming_exact_reproduction':[
            'Generator currently GLM instead of Llama3.1-8B-Instruct',
            'Initial-filter k and fine-filter beta score scale not verified',
            'Training duration and exact QWK discretization policy not established from paper text',
            'Other paper datasets and backbones not completed'],
        'reuse_rules':[
            'Do not combine old 885-row manifests with new 1107-row inputs',
            'Reuse cached responses only when exact request and model/prompt configuration match',
            'Rebuild data-dependent retrieval/filtering and annotator inputs for restored training set',
            'Old validation may join restored training; never use it for new validation selection'],
        'api_calls':0,'training_started':False,'predictions_computed':False,
        'upstream_test_files_read_for_partition':True,
        'test_labels_used_for_selection':False,'test_metrics_computed':False,
        'ready_for_training':False,
        'next_step':'Adapt upstream pipeline to isolated 1107/50/425 protocol before five-run validation-selected training',
        'outputs_sha256':{name:digest(raw) for name,raw in output.items()},
        'builder_sha256':digest(Path(__file__).read_bytes())}
    output['manifest.json'] = encode(plan)
    dest = root/('data/prepared/ads_paper_split_v1_seed%d'%seed)
    # Preflight all existing files before any mutation; identical reruns are harmless.
    for name, raw in output.items():
        path = dest/name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('Existing file differs; not overwritten: '+str(path))
    dest.mkdir(parents=True,exist_ok=True)
    for name, raw in output.items():
        path = dest/name
        if not path.exists():
            tmp = path.with_suffix(path.suffix+'.tmp')
            tmp.write_bytes(raw);tmp.replace(path)
        if path.read_bytes() != raw:
            raise ValueError('Output verification failed: '+str(path))
    print(json.dumps({'status':plan['status'],'counts':plan['counts'],
                      'restored_training_records':222,'cross_split_exact_text_groups':overlaps,
                      'api_calls':0,'training_started':False,'ready_for_training':False,
                      'output_directory':str(dest)},ensure_ascii=False,indent=2))
    print('PAPER_SPLIT_PREPARE_DONE',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--split-seed',type=int,default=42,choices=[42])
    args = p.parse_args()
    execute(args.root.resolve(),args.split_seed)
