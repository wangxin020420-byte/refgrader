"""Offline clean_v2 filtering. Never reads validation/test labels or calls API."""
import argparse
import json
from pathlib import Path
import filter_core as c

EXP = 'csbench_scale_online_clean_v2'
EXPECTED = {'train.jsonl': 'a4d837b033334b0983cff8f086f128526dcbf3708bca831b37b384da689a18fe',
            'candidate_inputs.jsonl': 'b2f9c8b3c1c729b80691ccb95625af6d9bfaa747aef6ba4cfa2c166aba2fd4d0'}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def prepare_sources(train, visual):
    lookup = {r['sample_id']: r for r in visual}
    require(len(train) == len(visual) == len(lookup) == 345, 'Source coverage mismatch')
    require({r['sample_id'] for r in train} == set(lookup), 'Source IDs changed')
    sources = []
    for t in train:
        v = lookup[t['sample_id']]
        require(t['question_id'] == v['question_id'] and t['references'] == v['references'], 'Source context mismatch')
        require(isinstance(v['answer'], str), 'Invalid original answer')
        sources.append(dict(sample_id=t['sample_id'], question_id=t['question_id'],
                            q=v['question'], r=v['references'], a=v['answer'], l=t['label']))
    return sources

def build(root):
    exp = root/'runs/online_experiments'/EXP
    ann = exp/'annotator_inputs_v1'
    run = exp/'annotator_bert_valqwk_v1'
    summary = c.read(run/'summary.json')
    cfg = c.read(run/'config.json')
    ch = c.sha(run/'config.json')
    require(summary['status'] == 'FIVE_ANNOTATORS_AND_CANDIDATE_SCORING_COMPLETE'
            and summary['completed_runs'] == 5 and summary['seeds'] == c.SEEDS
            and summary['candidate_records_per_run'] == 5000, 'Training incomplete')
    meta = c.read(ann/'manifest.json')
    require(meta['experiment_id'] == EXP and meta['training_records'] == 345, 'Wrong input manifest')
    require(c.sha(ann/'manifest.json') == cfg['input_hashes']['manifest'], 'Manifest changed since training')
    for name, expected in EXPECTED.items():
        require(c.sha(ann/name) == expected == meta['files_sha256'][name] == cfg['input_hashes'][name], 'Changed input: '+name)
    visualpath = exp/'visual/text_view/train.inputs.jsonl'
    require(c.sha(visualpath) == meta['source_sha256'][str(visualpath)], 'Original text view changed')
    sources = prepare_sources(c.rows(ann/'train.jsonl'), c.rows(visualpath))
    candidates = c.rows(ann/'candidate_inputs.jsonl')
    out = exp/'fine_filter_score0_10_beta0p2_v1'
    policy = dict(experiment_id=EXP, beta=0.2, score_scale='CSBench score_0_10 = 10 * score_original / score_max_original',
                  rule='abs(source_score - assigned_label) <= 0.2', beta_scale_verified_from_author=False,
                  threshold_equivalent_on_0_1_scale=0.02,
                  choice='Continue prior ADS reconstruction threshold on the current 0-10 scale; not tuned to validation or test performance',
                  numeric_policy='Python float; no epsilon, clipping, rounding, or averaging across seeds',
                  original_training_policy='Use clean_v2 visual train text, NOT polished proxy answers',
                  automatic_semantic_errors_retained=True, human_review_performed=False,
                  annotator_config_sha256=ch, train_text_sha256=c.sha(visualpath),
                  input_manifest_sha256=c.sha(ann/'manifest.json'), seeds=c.SEEDS,
                  builder_sha256=c.sha(Path(__file__)), core_sha256=c.sha(Path(c.__file__)))
    files = {out/'policy.json': c.enc(policy)}
    reports = {}
    for seed in c.SEEDS:
        sr = run/f'seed_{seed}'
        complete = c.read(sr/'complete.json')
        scorepath = sr/'scored_candidates.jsonl'
        require(complete['status'] == 'ANNOTATOR_AND_SCORING_COMPLETE'
                and complete['seed'] == seed and complete['config_sha256'] == ch
                and complete['training_records'] == 345 and complete['candidate_labels_assigned'] == 5000
                and complete['weights_reload_verified'], 'Invalid seed completion')
        require(c.sha(scorepath) == complete['files_sha256']['scored_candidates.jsonl'], 'Scored candidates changed')
        require(complete['best_epoch'] == summary['best_epochs'][str(seed)], 'Best epoch mismatch')
        kept, augmented, audit = c.select(c.rows(scorepath), sources, candidates, seed)
        payload = {'selected.jsonl': c.jl(kept), 'augmented_inputs.jsonl': c.jl(augmented), 'selection_audit.jsonl': c.jl(audit)}
        report = dict(seed=seed, original_training_records=345, scored_candidates=5000,
                      retained_synthetic=len(kept), discarded_synthetic=5000-len(kept), augmented_records=len(augmented),
                      retained_out_of_range=sum(not 0 <= r['assigned_label'] <= 10 for r in kept),
                      scored_candidates_sha256=c.sha(scorepath), policy_sha256=c.digest(c.enc(policy)),
                      files_sha256={n:c.digest(b) for n,b in payload.items()}, ready_for_sae_training=False)
        for name, data in payload.items():
            files[out/f'seed_{seed}'/name] = data
        files[out/f'seed_{seed}'/'manifest.json'] = c.enc(report)
        reports[str(seed)] = report
    result = dict(status='FIVE_RUN_EQ14_FINE_FILTER_COMPLETE', experiment_id=EXP,
                  beta=0.2, score_scale=policy['score_scale'], beta_scale_verified_from_author=False,
                  reproduction_status='METHOD_RECONSTRUCTION_WITH_DOCUMENTED_DEVIATIONS',
                  seeds=c.SEEDS, runs=reports, api_calls=0, source_files_modified=False,
                  validation_accessed_this_stage=False, test_accessed=False, training_started=False,
                  semantic_accuracy_verified=False, human_review_performed=False, ready_for_sae_training=False,
                  next_step='Prepare context/config-matched entity extractions for retained training answers and references, preserving seed labels',
                  files_sha256={str(p.relative_to(out)):c.digest(b) for p,b in files.items()})
    files[out/'summary.json'] = c.enc(result)
    return out, files, result

def execute(root, dry_run):
    out, files, result = build(root)
    for p, data in files.items():
        require(not p.exists() or p.read_bytes() == data, 'Existing output differs; preserved: '+str(p))
    if not dry_run:
        for p, data in files.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                tmp = p.with_name(p.name+'.tmp')
                tmp.write_bytes(data)
                tmp.replace(p)
            require(c.sha(p) == c.digest(data), 'Output write mismatch')
    display = dict(result)
    if dry_run:
        display['status'] = 'FINE_FILTER_PREFLIGHT_OK'
    print(json.dumps(display, ensure_ascii=False, indent=2))
    print('OUTPUT:', out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    if args.dry_run:
        execute(root, True)
    else:
        import fcntl
        exp = root/'runs/online_experiments'/EXP
        require(exp.is_dir(), 'Experiment directory missing')
        with (exp/'fine_filter_score0_10_beta0p2.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(root, False)
