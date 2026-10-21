"""Validate all frozen reviews offline without rerunning any experiment."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[4]

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def main():
    tasks=rows(ROOT/'tasks.jsonl');results=rows(ROOT/'results.jsonl')
    manifest=read_json(ROOT/'manifest.json');summary=read_json(ROOT/'summary.json')
    tm={(x['question_id'],x['answer_id']):x for x in tasks}
    rm={(x['question_id'],x['answer_id']):x for x in results}
    assert len(tm)==len(tasks)==len(rm)==len(results)==manifest['target_count']==909
    assert set(tm)==set(rm) and digest(tasks)==manifest['tasks_sha256']
    image_hashes={};checkpoint_hashes={}
    prepared_scores=read_json(PROJECT/'data/csbench/teacher_scores.json')
    for key,t in tm.items():
        r=rm[key]
        assert digest({k:v for k,v in t.items() if k!='input_sha256'})==t['input_sha256']
        assert r['input_sha256']==t['input_sha256']
        assert r['status']=='completed' and r['version']==manifest['version']
        assert r['review_run_id']==manifest['review_run_id']
        assert r['basis']=='existing_experiment_only' and r['new_experiment'] is False
        assert r['not_a_human_decision'] is True and r['automatic_policy_change'] is False
        assert r['summary'] and r['evidence'] and r['original_image_observations']
        assert r['teacher_score']==t['teacher_score']==prepared_scores[key[1]][key[0]]
        assert r['source_checkpoint']==t['model_evidence']['source_path']
        assert r==read_json(ROOT/'answers'/key[0]/(key[1]+'.json'))
        for kind,im in t['images'].items():
            if im:
                path=Path(im['path'])
                actual=image_hashes.setdefault(str(path),sha(path))
                assert actual==im['sha256'], (key,kind)
        checkpoint=Path(r['source_checkpoint'])
        if not checkpoint.is_absolute():
            checkpoint=PROJECT/checkpoint
        if str(checkpoint) not in checkpoint_hashes:
            checkpoint_hashes[str(checkpoint)]=sha(checkpoint)
    actual_answer_keys=set()
    for path in (ROOT/'answers').glob('*/*.json'):
        r=read_json(path);actual_answer_keys.add((r['question_id'],r['answer_id']))
    assert actual_answer_keys==set(tm)
    counts=Counter(x['category'] for x in results)
    by_question=Counter(x['question_id'] for x in results)
    assert counts==summary['categories'] and by_question==summary['by_question']
    assert by_question==Counter(x['question_id'] for x in tasks)
    assert summary['target']==summary['completed']==909 and summary['pending']==0
    human=ROOT.parent/'teacher_label_audit_all43_20260727_160210_decisions.jsonl'
    assert sha(human)==sha(ROOT/'human_decisions_at_start.jsonl')==manifest['human_decisions_sha256']
    assert len(rows(human))==119
    validation=dict(validated_at=datetime.now(timezone.utc).isoformat(),
        frozen_targets=909,unique_completed=909,pending=0,questions_with_frozen_targets=len(by_question),
        categories=dict(counts),by_question=dict(by_question),
        scope_levels=dict(Counter(x['suspicion_level'] for x in tasks)),
        all_input_fingerprints_match=True,all_review_files_match_publication=True,
        all_original_cleaned_and_question_image_hashes_match=True,unique_image_files_verified=len(image_hashes),
        current_checkpoint_hashes_for_archive=checkpoint_hashes,
        prepared_teacher_scores_match_frozen_inputs=True,
        human_decisions_preserved=119,human_decisions_sha256=sha(human),
        result_sha256=sha(ROOT/'results.jsonl'),basis='existing_experiment_only',new_experiment=False,
        automatic_policy_change=False)
    (ROOT/'final_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in validation.items() if k not in ('current_checkpoint_hashes_for_archive','by_question')},ensure_ascii=False))

if __name__=='__main__':
    main()
