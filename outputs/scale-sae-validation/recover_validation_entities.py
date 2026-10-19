"""Five reviewed validation extractions; offline repair then frozen SAE evaluation."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

PLAN = {'validation_text:06425730f0a15540a6da41fb22fcfa314b2681949ffac3f0366506c393ef1cdb': {'response_filename': 'attempt_001.response.json', 'content_sha256': '4280a818b59dddb8646e686c8db808ff0fa331508f8721b1c27b687733d4d8de', 'changes': [{'index': 7, 'field': 't', 'before': '第二次排序：2.1.3.4', 'after': '第二次：2.1.3.4', 'source_span': [83, 94], 'reason': 'Restore the reviewed exact source phrase, including original line breaks.'}]}, 'validation_text:31c1bfcf51b2b9a2b3fd944ffc61809767591366254959d7d64983b770f2703e': {'response_filename': 'attempt_001.response.json', 'content_sha256': 'a9411954366379d05d33f574dc7c6e7558257ad5c4cbdd093c036e5f0a2ece23', 'changes': [{'index': 1, 'field': 't', 'before': '多次将两个或两个以上的有序表合并成一个新的有序表', 'after': '多次将两个或两个以上的有序表合并成一个新的有序\n表', 'source_span': [18, 43], 'reason': 'Restore the reviewed exact source phrase, including original line breaks.'}]}, 'validation_text:428109e07d4284eb1bbb9eea4dfb1a93393a6b313a9263dbe2e0d278d9d2adb8': {'response_filename': 'attempt_001.response.json', 'content_sha256': '8301ec56e2a35b1146e37716fab4736cc0dee957bb3611a8f6bbf70c528fcaa4', 'changes': [{'index': 6, 'field': 't', 'before': '第二次排序：2.1.3.4', 'after': '第二次：2.1.3.4', 'source_span': [83, 94], 'reason': 'Restore the reviewed exact source phrase, including original line breaks.'}]}, 'validation_text:c545916d6df2d2a60157bc141da06c3d504c53c38aa6ab44d85363564787e885': {'response_filename': 'attempt_001.response.json', 'content_sha256': '7009c932f15fc1f751969a825b4d4c69c342750ba7fbf2e6500acf2d88b4b19a', 'changes': [{'index': 2, 'field': 't', 'before': '队S2执行出栈操作', 'after': '队S2执行\n出栈操作', 'source_span': [64, 74], 'reason': 'Restore the reviewed exact source phrase, including original line breaks.'}, {'index': 3, 'field': 't', 'before': '先将S1中所有元素出栈并入栈到S2中，再在S2中出栈即可实现出队操作', 'after': '先将S1中所有元素出栈并入栈到S2中，再在S2中出栈\n即可实现出队操作', 'source_span': [77, 112], 'reason': 'Restore the reviewed exact source phrase, including original line breaks.'}, {'index': 7, 'field': 't', 'before': '若S1满，必须先保证S2为空，才能将S1中的元素全部插入S2中', 'after': '若S1满，必须先保证S2为空，才能将S1中\n的元素全部插入S2中', 'source_span': [210, 242], 'reason': 'Restore the reviewed exact source phrase, including original line breaks.'}, {'index': 17, 'field': 'w', 'before': '标题为入队算法下给出的DeQueue函数：S2非空则从S2出栈，S2空且S1空则输出队列为空，否则先将S1元素全部转入S2再出栈', 'after': '标题为入队算法下给出的DeQueue函数：S2非空则从S2出栈，S2空且S1非空则输出队列为空，否则先将S1元素全部转入S2再出栈', 'reason': 'Source code says else if(!StackEmpty(S1)); preserve its nonempty condition, without correcting the student code.'}, {'index': 19, 'field': 'w', 'before': 'S2为空且S1为空时输出队列为空', 'after': 'S2为空且S1非空时输出队列为空', 'reason': 'Source code says else if(!StackEmpty(S1)); preserve its nonempty condition, without correcting the student code.'}]}, 'validation_text:e3b7ed385a92b5f90aa65a6e8baca79445955761e1d30a44d791b1ab051f4d80': {'response_filename': 'attempt_002.response.json', 'content_sha256': 'a1aeb2bbfa6bc3d0726de8b8b3ea128a54fdb09c64436e0b679f097dceb0aac0', 'changes': [{'index': 6, 'field': 't', 'before': '3第三趟移向了与最终位置相反的方向', 'after': '3第一趟移向了与最终位置相反的方向', 'source_span': [51, 68], 'reason': 'Restore the reviewed exact source phrase, including original line breaks.'}, {'index': 6, 'field': 'w', 'before': '元素3在第三趟排序中移动方向与其最终位置相反的现象', 'after': '元素3在第一趟排序中移动方向与其最终位置相反的现象', 'reason': 'The source says 第一趟, not 第三趟; retain the original claim.'}]}}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def repair(ep, m, unit, raw):
    uid = unit['unit_id']
    plan = PLAN[uid]
    if uid != 'validation_text:' + ep.sha(unit['text'].encode('utf8')):
        raise ValueError('Reviewed source changed')
    choice = raw['choices'][0]
    content = choice['message']['content']
    if choice.get('finish_reason') != 'stop' or ep.sha(content.encode('utf8')) != plan['content_sha256']:
        raise ValueError('Response is not the reviewed complete attempt')
    obj = json.loads(content)
    if set(obj) != {'triples'}:
        raise ValueError('Unexpected extraction fields')
    fixed = copy.deepcopy(obj)
    for change in plan['changes']:
        triple = fixed['triples'][change['index']]
        if triple[change['field']] != change['before']:
            raise ValueError('Reviewed field changed')
        triple[change['field']] = change['after']
        if change['field'] == 't':
            a, b = change['source_span']
            if unit['text'][a:b] != change['after']:
                raise ValueError('Reviewed source span changed')
    result = m.validate([unit], [{'unit_id': uid, 'triples': fixed['triples']}])[0]
    if len(result['triples']) != len(obj['triples']):
        raise ValueError('Unexpected triple removal')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--physical-gpu', default='auto')
    p.add_argument('--wait-hours', type=float, default=24)
    args = p.parse_args()
    root = args.root.resolve()
    out = root / 'paper_method_v1/runs/sae_raw_beta_seed42_validation_v1'
    ep = load(root / 'control/scale-synthetic-entities/entity_prepare.py', 'original_entity_helpers')
    v = load(root / 'control/scale-sae-validation/validate_sae.py', 'original_validation_pipeline')
    m = ep.module(root / 'control/scale-paper-stage1/stage1.py')
    import fcntl
    with (out / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        units = {r['unit_id']: r for r in ep.lines(out / 'units.jsonl')}
        audits, results, hashes = [], [], {}
        for uid, plan in PLAN.items():
            response = out / 'responses' / uid.split(':', 1)[1] / plan['response_filename']
            raw = ep.read(response)
            results.append(repair(ep, m, units[uid], raw))
            hashes[response] = ep.sha(response.read_bytes())
            audits.append(dict(plan, unit_id=uid, response_file=str(response.relative_to(root)),
                               response_file_sha256=hashes[response]))
        review = out / 'reviewed_entity_recovery_v1'
        ep.write(review / 'corrected_extractions.jsonl', ep.jl(results), frozen=True)
        audit = {'status': 'FIVE_REVIEWED_EXTRACTIONS_RECOVERED', 'records': len(results),
                 'changes': audits, 'api_calls': 0, 'source_answers_modified': False,
                 'raw_responses_modified': False, 'model_weights_modified': False,
                 'gold_labels_used_for_repair': False,
                 'wrapper_sha256': ep.sha(Path(__file__).read_bytes()),
                 'deviation': 'Manual source-grounded extraction quality control, not an original-paper step. '
                              'Restore verbatim spans and source-consistent summaries; do not correct student answers.'}
        ep.write(review / 'audit.json', ep.enc(audit), frozen=True)
        original_parse = ep.parse

        def reviewed_parse(stage, unit, raw):
            uid = unit['unit_id']
            content = raw.get('choices', [{}])[0].get('message', {}).get('content')
            if uid in PLAN and isinstance(content, str) and ep.sha(content.encode('utf8')) == PLAN[uid]['content_sha256']:
                return repair(ep, stage, unit, raw)
            return original_parse(stage, unit, raw)

        ep.parse = reviewed_parse
        # execute accepts an attempt budget; zero ensures cache-only entity resolution.
        args.max_attempts = 0
        print('RECOVERED_REVIEWED_UNITS:', len(results), flush=True)
        print('AUDIT_FILE:', review / 'audit.json', flush=True)
        status = v.execute(args, ep, out)
        for response, digest in hashes.items():
            if ep.sha(response.read_bytes()) != digest:
                raise ValueError('Original response changed')
        if status == 0:
            path = out / 'metrics.json'
            metrics = ep.read(path)
            metrics['reviewed_entity_recovery'] = {
                'records': 5, 'audit_file': str((review / 'audit.json').relative_to(root)),
                'audit_sha256': ep.sha((review / 'audit.json').read_bytes()),
                'corrected_extractions_sha256': ep.sha((review / 'corrected_extractions.jsonl').read_bytes())}
            ep.write(path, ep.enc(metrics))
            print('METRICS_FILE:', path, flush=True)
            print(json.dumps(metrics['overall'], ensure_ascii=False, indent=2), flush=True)
            print('\aREVIEWED_VALIDATION_COMPLETE; zero new API calls', flush=True)
        raise SystemExit(status)


if __name__ == '__main__':
    main()
