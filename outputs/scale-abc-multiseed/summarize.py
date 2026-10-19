"""Read-only paired-seed validation summary; discovery seed shown separately."""
import argparse
import json
from pathlib import Path
import statistics


def describe(values):
    return {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values), 'n': len(values)}


def aggregate(reports, seeds):
    output = {'seeds': seeds, 'groups': {}, 'paired_differences': {}}
    for group in 'ABC':
        scores = [reports[(seed, group)]['best']['metrics'] for seed in seeds]
        output['groups'][group] = {metric: describe([s[metric] for s in scores])
                                  for metric in ['qwk', 'rmse_raw', 'unclipped_rmse_raw', 'clip_count']}
    for left, right in [('C', 'A'), ('C', 'B'), ('B', 'A')]:
        deltas = {}
        for metric in ['qwk', 'rmse_raw', 'unclipped_rmse_raw']:
            values = [reports[(s, left)]['best']['metrics'][metric] - reports[(s, right)]['best']['metrics'][metric]
                      for s in seeds]
            deltas[metric] = dict(describe(values), per_seed=values,
                                 wins=sum(v > 0 if metric == 'qwk' else v < 0 for v in values))
        output['paired_differences'][left + '_minus_' + right] = deltas
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    baseline_hash = json.loads((Path(__file__).parent / 'change_provenance.json').read_text(encoding='utf8'))['base_script_sha256']
    reports, missing, signatures = {}, [], []
    print('seed\tgroup\tbest_step\tQWK\tRMSE_clipped\tRMSE_unclipped\tclip_count')
    for seed in [42, 43, 44, 45, 46]:
        folder = 'abc_pilot_v1' if seed == 42 else 'abc_followup_v1'
        for group in 'ABC':
            run = args.root / 'runs' / folder / ('%s_seed%d' % (group, seed))
            path = run / 'complete.json'
            if not path.exists():
                missing.append(str(path))
                continue
            report = json.loads(path.read_text(encoding='utf8'))
            cfg = json.loads((run / 'config.json').read_text(encoding='utf8'))
            if cfg['seed'] != seed or cfg['group'] != group:
                raise ValueError('Run identity mismatch: ' + str(run))
            if seed == 42 and cfg['script_sha256'] != baseline_hash:
                raise ValueError('Seed42 baseline script differs from expected version')
            signatures.append({k: cfg[k] for k in ['plan_sha256', 'metrics_script_sha256', 'microbatch', 'device',
                                                   'model_revision', 'torch', 'transformers']})
            m = report['best']['metrics']
            if m['qwk'] is None:
                raise ValueError('Undefined QWK; inspect ' + str(run))
            reports[(seed, group)] = report
            print('%d\t%s\t%d\t%.6f\t%.6f\t%.6f\t%d' %
                  (seed, group, report['best']['step'], m['qwk'], m['rmse_raw'], m['unclipped_rmse_raw'], m['clip_count']))
    if signatures and any(s != signatures[0] for s in signatures):
        raise ValueError('Execution configurations differ; do not pool these runs')
    if missing:
        print('INCOMPLETE: %d runs missing; no pooled conclusion.' % len(missing))
        for path in missing:
            print(path)
        return
    output = {'status': 'VALIDATION_MULTI_SEED_SUMMARY',
              'warning': 'Exploratory provisional labels; seed42 motivated this follow-up. No test evaluation or significance claim.',
              'all_five_descriptive': aggregate(reports, [42, 43, 44, 45, 46]),
              'four_new_seeds': aggregate(reports, [43, 44, 45, 46])}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
