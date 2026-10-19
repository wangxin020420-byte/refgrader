"""Frozen CSBench clean_v2 training-only entity extraction adapter."""
import argparse
from pathlib import Path
import entity_engine as e
import stage1_contract as contract

EXP = 'csbench_scale_online_clean_v2'
CONTRACT_SHA = '26c9fe7ee85fb7760e70bfa29da87533cd1266d7d147e2e60c1fa543355c919c'
HASHES = [
    'b47bfe33080de432cb248f41b8709f186743844eb52753dde1d288795a1b2f8c',
    'bfb0c9d258afacb8c8368a329bdb04e807889f4da94a643ed2ca90f776d2040b',
    'e9470313f8863a78bccc804116b14748daa18c1f26503280c7b3602aa99bf9c6',
    '7be943232aebcf250f0d0d1f44fc7b6710855ee364cb13429171a810cb09188a',
    'a2192fd559039720f1c1e9b8734cc33cb9111f2531d7b17c23fcc59005aa690e',
]
STAGE_HASHES = {
    'config.json': '57f740233d47ed216d6c67f42e2bc5db1b561584e54d719670e388181a14d873',
    'extractions.jsonl': '7caa0b32430ff54ca7dd9ad589bac948ef2c4c28cd6f00fef7cb76d2ae68d5ed',
    'source_units.jsonl': 'd102ac996f2aca77409d15ab51e523543147d3cea1941fb44ae9f59b5a94ca33',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def prepare(root, out):
    exp = root/'runs/online_experiments'/EXP
    fine = exp/'fine_filter_score0_10_beta0p2_v1'
    summary = e.read(fine/'summary.json')
    require(summary['experiment_id'] == EXP and summary['seeds'] == e.SEEDS
            and summary['status'] == 'FIVE_RUN_EQ14_FINE_FILTER_COMPLETE', 'Wrong fine filter')
    require(summary['beta'] == 0.2, 'Wrong beta')
    require(e.sha(fine/'policy.json') == summary['files_sha256']['policy.json']
            == 'e10cc7b7cf9671d537f0a4641091e7f71fc8b19094f0ad4d8aa64d1f5fc49cb6', 'Policy changed')
    datasets = {}
    for seed, count, expected in zip(e.SEEDS, e.COUNTS, HASHES):
        name = 'seed_%d/augmented_inputs.jsonl' % seed
        require(e.sha(fine/name) == summary['files_sha256'][name] == expected, 'Augmented input changed: '+name)
        data = e.rows(fine/name)
        require(len(data) == count and sum(r['kind'] == 'original_training' for r in data) == 345, 'Wrong row counts')
        datasets[seed] = data
    stage = exp/'scale_stage1'
    require(e.sha(Path(contract.__file__)) == CONTRACT_SHA, 'Contract changed')
    for name, expected in STAGE_HASHES.items():
        require(e.sha(stage/name) == expected, 'Stage1 provenance mismatch: '+name)
    old = e.read(stage/'config.json')
    require(old['model'] == 'glm-5.3' and old['base_url'].rstrip('/') == 'https://open.bigmodel.cn/api/coding/paas/v4', 'Wrong model or endpoint')
    require(old['prompt_sha256'] == e.digest(contract.PROMPT.encode('utf8')), 'Prompt changed')
    units = e.rows(stage/'source_units.jsonl')
    require(len(units) == 390, 'Wrong source units')
    extractions = contract.validate(units, e.rows(stage/'extractions.jsonl'))
    bindings, pending, reused = e.bind(datasets, units, extractions)
    cfg = dict(experiment_id=EXP, model=old['model'], base_url=old['base_url'].rstrip('/'),
               max_tokens=old['max_tokens'], stream=False, prompt_sha256=old['prompt_sha256'],
               stage1_sha256=CONTRACT_SHA, runner_sha256=e.sha(Path(__file__)),
               engine_sha256=e.sha(Path(e.__file__)), source_sha256=STAGE_HASHES,
               fine_filter_summary_sha256=e.sha(fine/'summary.json'),
               reuse_policy='Original/reference units matched by question, exact text and unit ID; frozen stage1 output. No previous synthetic caches imported.',
               deduplication='Only new synthetic text requests deduplicated across seeds. Request contains text only with fixed prompt/model/config. Labels and reference slots stay separate.',
               label_policy='Preserve every per-seed label including out-of-range values. No clipping, rounding or averaging.',
               human_review_performed=False, automatic_semantic_errors_retained=True,
               semantic_accuracy_verified=False, test_accessed=False, validation_accessed=False)
    payload = {'config.json':e.enc(cfg), 'pending_units.jsonl':e.jl(pending), 'reused_extractions.jsonl':e.jl(reused)}
    for seed in e.SEEDS:
        payload['seed_%d/bindings.jsonl'%seed] = e.jl(bindings[seed])
    for name, data in payload.items():
        e.write(out/name, data)
    report = dict(status='FIVE_RUN_ENTITY_PLAN_READY', experiment_id=EXP,
                  unique_synthetic_requests=len(pending),
                  nonblank_requests=sum(bool(u['text'].strip()) for u in pending),
                  reused_original_reference_units=len(reused),
                  training_records_by_seed=dict(zip(map(str,e.SEEDS),e.COUNTS)),
                  api_calls=0, test_accessed=False, validation_accessed=False,
                  files_sha256={n:e.digest(b) for n,b in payload.items()})
    e.write(out/'plan.json', e.enc(report))
    print(e.enc(report).decode(), flush=True)
    return contract, cfg, datasets, bindings, pending, reused


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare','run','offline'])
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    require((root/'runs/online_experiments'/EXP).is_dir(), 'Existing experiment missing')
    out = root/'runs/online_experiments'/EXP/'train_entities_five_runs_v1'
    out.mkdir(exist_ok=True)
    import fcntl
    with (out/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        e.prepare = prepare
        return e.execute(root, out, a.command, attempts=8, timeout=900)


if __name__ == '__main__':
    raise SystemExit(main())
