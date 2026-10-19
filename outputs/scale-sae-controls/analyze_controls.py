"""Offline paired, question-stratified answer-cluster bootstrap and data audit."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]


def save(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf8')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing analysis differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(data); tmp.replace(path)


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*fraction
    lo = int(index); hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(index-lo)


def cluster_groups(inputs, ordered_ids):
    by_id = {r['sample_id']: r for r in inputs}
    groups = defaultdict(lambda: defaultdict(list))
    for index, sid in enumerate(ordered_ids):
        r = by_id[sid]
        # Keep all identical question/answer rows together; do not normalize text.
        groups[r['question_id']][r['answer']].append(index)
    return [[members for _,members in sorted(group.items())] for _,group in sorted(groups.items())]


def resample(groups, rng):
    indices = []
    for clusters in groups:
        for _ in clusters:
            indices.extend(clusters[rng.randrange(len(clusters))])
    return indices


def distribution(rows):
    values = [r['label'] for r in rows]
    if not values:
        return {'records': 0}
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite training label')
    bins = Counter('below_0' if x < 0 else 'above_10' if x > 10 else
                   str(min(10, math.floor(x+0.5))) for x in values)
    return {'records': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values), 'raw_label_bins_half_up_for_display_only': dict(sorted(bins.items())),
            'unique_source_samples': len({r['sample_id'] for r in rows}),
            'unique_exact_answer_texts': len({r['answer'] for r in rows})}


def execute(args):
    root = args.root.resolve()
    control = root / 'paper_method_v1/runs/sae_controls_seed42_v1'
    comparison = read(control/'comparison.json')
    protocol = read(control/'protocol.json')
    if comparison['status'] != 'THREE_WAY_VALIDATION_COMPARISON_COMPLETE':
        raise ValueError('Comparison incomplete')
    if sha(control/'protocol.json') != comparison['provenance']['protocol_sha256']:
        raise ValueError('Control protocol changed')
    metric_code = root / 'control/scale-sae-validation/validate_sae.py'
    if sha(metric_code) != protocol['metric_code_sha256']:
        raise ValueError('Metric implementation changed')
    spec = importlib.util.spec_from_file_location('frozen_metrics', str(metric_code))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    metric = module.metrics
    data = root / 'data/prepared/ads_public_adapted_v1'
    manifest = read(data/'manifest.json')
    for name in ('validation.inputs.jsonl', 'validation.labels.jsonl'):
        if sha(data/name) != manifest['outputs_sha256'][name]:
            raise ValueError('Validation source changed: ' + name)
    inputs = lines(data/'validation.inputs.jsonl')
    gold_rows = lines(data/'validation.labels.jsonl')
    gold_map = {r['sample_id']: r['gold_raw'] for r in gold_rows}
    ids = [r['sample_id'] for r in inputs]
    if len(ids) != 222 or len(set(ids)) != 222 or len(gold_rows) != 222 or set(ids) != set(gold_map):
        raise ValueError('Validation identity mismatch')
    gold = [gold_map[sid] for sid in ids]
    sources = {}
    for name in ('bert_original', 'sae_original'):
        folder = root / ('paper_method_v1/runs/'+name+'_seed42_v1')
        if sha(folder/'complete.json') != comparison['provenance']['baseline_complete_sha256'][name]:
            raise ValueError('Baseline completion changed')
        complete = read(folder/'complete.json')
        path = folder/'validation_predictions.json'
        if sha(path) != complete['files_sha256'][path.name]:
            raise ValueError('Baseline predictions changed')
        sources[name] = read(path)
    val = root/'paper_method_v1/runs/sae_raw_beta_seed42_validation_v1'
    if sha(val/'metrics.json') != comparison['provenance']['augmented_validation_sha256']:
        raise ValueError('Augmented validation report changed')
    if sha(val/'predictions.jsonl') != read(val/'metrics.json')['files_sha256']['predictions.jsonl']:
        raise ValueError('Augmented predictions changed')
    sources['sae_augmented'] = lines(val/'predictions.jsonl')
    predictions = {}
    qids = {r['sample_id']: r['question_id'] for r in inputs}
    for name, rows in sources.items():
        lookup = {r['sample_id']: r for r in rows}
        if len(rows) != len(lookup) or set(lookup) != set(ids):
            raise ValueError('Prediction identity mismatch')
        if any(lookup[s]['question_id'] != qids[s] for s in ids):
            raise ValueError('Prediction question mismatch')
        predictions[name] = [lookup[s]['prediction_raw'] for s in ids]
        calculated = metric(gold, predictions[name])
        for key in ('pearson_raw', 'qwk_clipped_half_up', 'rmse_raw', 'rmse_clipped'):
            expected = comparison['overall'][name][key]
            if expected is None or calculated[key] is None:
                if expected is not calculated[key]:
                    raise ValueError('Metric null mismatch')
            elif not math.isclose(expected, calculated[key], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Published metric mismatch: '+name+' '+key)
    groups = cluster_groups(inputs, ids)
    pairs = [('sae_original_minus_bert_original','sae_original','bert_original'),
             ('sae_augmented_minus_sae_original','sae_augmented','sae_original')]
    keys = ('pearson_raw', 'qwk_clipped_half_up', 'rmse_raw')
    draws = {name: {k: [] for k in keys} for name,_,_ in pairs}
    rng = random.Random(20260927)
    for iteration in range(args.replicates):
        chosen = resample(groups, rng)
        resampled_gold = [gold[i] for i in chosen]
        scores = {name: metric(resampled_gold, [values[i] for i in chosen])
                  for name, values in predictions.items()}
        for name,a,b in pairs:
            for key in keys:
                if scores[a][key] is not None and scores[b][key] is not None:
                    draws[name][key].append(scores[a][key]-scores[b][key])
        if (iteration+1) % 500 == 0:
            print('PAIRED_BOOTSTRAP %d/%d' % (iteration+1,args.replicates), flush=True)
    intervals = {name: {key: {'point_difference': comparison['differences'][name][key],
        'percentile_95_interval': [percentile(values,0.025), percentile(values,0.975)],
        'valid_replicates': len(values)} for key,values in entry.items()} for name,entry in draws.items()}
    question_differences = []
    for q in sorted(set(qids.values())):
        ix = [i for i,s in enumerate(ids) if qids[s] == q]
        row = {'question_id':q, 'records':len(ix)}
        for name,values in predictions.items():
            row[name] = metric([gold[i] for i in ix], [values[i] for i in ix])
        row['augmentation_delta_sse'] = sum((predictions['sae_augmented'][i]-gold[i])**2
            -(predictions['sae_original'][i]-gold[i])**2 for i in ix)
        question_differences.append(row)
    question_differences.sort(key=lambda r:r['augmentation_delta_sse'], reverse=True)
    augmented = root/'paper_method_v1/runs/train_entities_raw_beta_v1/training_inputs.jsonl'
    if sha(augmented) != '4f4f1a3b98160c0b08e111f4bc904ad93e331255cfbb1aa154ec6fd483c3d9e8':
        raise ValueError('Frozen augmented training data changed')
    training = lines(augmented)
    original = [r for r in training if r['kind']=='original_training']
    synthetic = [r for r in training if r['kind']=='synthetic_candidate']
    if (len(original),len(synthetic)) != (885,1327):
        raise ValueError('Training counts differ')
    source_scores = {r['sample_id']: r['label'] for r in original}
    data_audit = {'original': distribution(original), 'synthetic': distribution(synthetic),
        'synthetic_count_by_source_gold': dict(sorted(Counter(str(source_scores[r['sample_id']]) for r in synthetic).items())),
        'per_question': {q: {'original':distribution([r for r in original if r['question_id']==q]),
                            'synthetic':distribution([r for r in synthetic if r['question_id']==q])}
                         for q in sorted({r['question_id'] for r in training})},
        'warning': 'Row frequencies are not equal to optimizer weights with source-specific batch sizes; no causal attribution from this audit'}
    report = {'status':'CONTROL_ANALYSIS_COMPLETE', 'validation_records':222, 'training_seed':42,
        'replicates':args.replicates, 'bootstrap_seed':20260927,
        'resampling':'Paired across models; within each question resample exact-answer clusters with replacement',
        'answer_clusters':sum(len(g) for g in groups), 'questions':len(groups),
        'paired_intervals':intervals,
        'limitations':['Conditional on these three fitted models and observed questions; does not measure training-seed variance',
                      'Percentile intervals are exploratory, not multiple-testing-adjusted',
                      'Epoch-matched but unequal optimizer updates; no causal proof',
                      'Validation only; not paper five-run or official test results'],
        'api_calls':0, 'new_training_started':False, 'test_accessed':False,
        'source_sha256':{'comparison':sha(control/'comparison.json'), 'training':sha(augmented),
                         'validation_inputs':sha(data/'validation.inputs.jsonl'),
                         'validation_labels':sha(data/'validation.labels.jsonl')},
        'script_sha256':sha(Path(__file__))}
    out = control/'analysis_v1'
    save(out/'paired_analysis.json',report)
    save(out/'training_distribution.json',data_audit)
    save(out/'question_comparison.json',question_differences)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('TRAINING_DISTRIBUTION',json.dumps({'original':data_audit['original'],'synthetic':data_audit['synthetic']},ensure_ascii=False))
    print('OUTPUT_DIRECTORY:',out)
    print('\aCONTROL_ANALYSIS_DONE',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--replicates',type=int,default=2000)
    args=parser.parse_args()
    if args.replicates < 100:
        parser.error('At least 100 replicates required')
    execute(args)
