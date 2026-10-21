"""Offline dossiers and recorded reviews of existing experiments. No model API."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.review_teacher_labels import TeacherLabelReviewStore

REPORT_RUN = 'teacher_label_audit_all43_20260727_160210'
VERSION = 'existing-evidence-review-v1'
CATEGORIES = {'teacher_suspect', 'model_or_extraction_issue', 'label_supported', 'uncertain'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def prepare(prepared, report_run):
    directory = prepared / 'quality_control/reviews' / (report_run + '_evidence_review')
    manifest_path = directory / 'manifest.json'
    if manifest_path.is_file():
        manifest = load(manifest_path)
        tasks = [json.loads(line) for line in (directory / 'tasks.jsonl').read_text(encoding='utf-8').splitlines()]
        if manifest['tasks_sha256'] != digest(tasks):
            raise ValueError('Frozen scope modified')
        return directory, manifest, tasks
    previous = prepared / 'quality_control/reviews' / (report_run + '_assisted')
    if (previous / 'tasks.jsonl').is_file():
        # Reuse only the frozen source evidence and original scope, never new
        # model outputs from the superseded four-answer pilot.
        tasks = [json.loads(line) for line in (previous / 'tasks.jsonl').read_text(encoding='utf-8').splitlines()]
        old = load(previous / 'manifest.json')
        if digest(tasks) != old['tasks_sha256']:
            raise ValueError('Original frozen source mismatch')
        backup = (previous / 'human_decisions_at_start.jsonl').read_bytes()
    else:
        decisions = prepared / 'quality_control/reviews' / (report_run + '_decisions.jsonl')
        store = TeacherLabelReviewStore(prepared_dir=prepared, report_dir=prepared / 'quality_control/reports' / report_run,
            decisions_path=decisions, screening_path=prepared / 'quality_control/reviews' / (report_run + '_initial_screening.csv'), all_answers=True)
        tasks = []
        for row in store.candidates:
            key = (row['question_id'], row['answer_id'])
            if row['suspicion_level'] != 'low' and not (row['suspicion_level'] == 'medium' and key not in store.decisions):
                continue
            images = {}
            for name, path in [('original', store.resolve_image_path(row, original=True)), ('cleaned', store.resolve_image_path(row)), ('question', store.resolve_question_image_path(row['question_id']))]:
                images[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} if path else None
            task = {'question_id': key[0], 'answer_id': key[1], 'teacher_score': row['teacher_score'], 'max_score': row['max_score'],
                'suspicion_level': row['suspicion_level'], 'transcript': row['raw_text'], 'question': store.question_contexts[key[0]],
                'model_evidence': store.grading_evidence(*key), 'images': images, 'human_decision_at_start': store.decisions.get(key)}
            task['input_sha256'] = digest(task)
            tasks.append(task)
        backup = decisions.read_bytes()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'tasks.jsonl').write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in tasks), encoding='utf-8')
    (directory / 'human_decisions_at_start.jsonl').write_bytes(backup)
    manifest = {'schema_version': 1, 'version': VERSION, 'review_run_id': directory.name, 'report_run': report_run,
        'created_at': datetime.now(timezone.utc).isoformat(), 'target_count': len(tasks), 'tasks_sha256': digest(tasks),
        'scope': 'pending_medium_and_all_low', 'basis': 'existing_experiment_only', 'new_experiment': False,
        'levels': dict(Counter(t['suspicion_level'] for t in tasks)), 'human_decisions_sha256': hashlib.sha256(backup).hexdigest()}
    save(manifest_path, manifest)
    return directory, manifest, tasks


def publish(directory, manifest, tasks):
    results = []
    for task in tasks:
        path = directory / 'answers' / task['question_id'] / (task['answer_id'] + '.json')
        if path.is_file():
            row = load(path)
            if row['input_sha256'] != task['input_sha256']:
                raise ValueError('Review source mismatch')
            results.append(row)
    target = directory / 'results.jsonl'
    temp = target.with_suffix('.jsonl.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results), encoding='utf-8')
    os.replace(temp, target)
    summary = {'target': len(tasks), 'completed': len(results), 'pending': len(tasks) - len(results),
        'categories': dict(Counter(r['category'] for r in results)), 'by_question': dict(Counter(r['question_id'] for r in results)),
        'basis': 'existing_experiment_only', 'new_experiment': False, 'updated_at': datetime.now(timezone.utc).isoformat()}
    save(directory / 'summary.json', summary)
    return summary


def record_reviews(directory, manifest, tasks, reviews):
    source = {(t['question_id'], t['answer_id']): t for t in tasks}
    seen = set()
    for review in reviews:
        key = (review['question_id'], review['answer_id'])
        if key in seen or key not in source:
            raise ValueError('Duplicate or out of scope review')
        seen.add(key)
        task = source[key]
        if review['category'] not in CATEGORIES or not review.get('summary') or not review.get('evidence') or not review.get('original_image_observations'):
            raise ValueError('Missing individual evidence')
        if not review.get('model_issue') and not review.get('teacher_issue') and review['category'] != 'label_supported':
            raise ValueError('Missing source-of-disagreement analysis')
        # Verified image input plus explicit observation is required for every
        # completed record; failures or merely generated dossiers are not reviews.
        original = task['images'].get('original')
        if not original or hashlib.sha256(Path(original['path']).read_bytes()).hexdigest() != original['sha256']:
            raise ValueError('Original image missing or modified')
        row = {**review, 'schema_version': 1, 'version': VERSION, 'review_run_id': manifest['review_run_id'],
            'status': 'completed', 'input_sha256': task['input_sha256'], 'teacher_score': task['teacher_score'],
            'basis': 'existing_experiment_only', 'new_experiment': False, 'reviewer': 'codex_existing_evidence',
            'source_checkpoint': task['model_evidence']['source_path'], 'not_a_human_decision': True,
            'automatic_policy_change': False, 'created_at': datetime.now(timezone.utc).isoformat()}
        path = directory / 'answers' / key[0] / (key[1] + '.json')
        if path.exists():
            raise ValueError('Completed review already exists; preserve it: ' + str(path))
        save(path, row)
    return publish(directory, manifest, tasks)


def dossier(task):
    probes = task['model_evidence']['probes']
    return {'answer_id': task['answer_id'], 'question_id': task['question_id'], 'teacher': task['teacher_score'],
        'max': task['max_score'], 'transcript': task['transcript'], 'images': task['images'],
        'scores': task['model_evidence'].get('score_history'), 'probes': probes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, default=ROOT / 'data/csbench')
    parser.add_argument('--report-run', default=REPORT_RUN)
    parser.add_argument('--question')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=6)
    parser.add_argument('--import-reviews', type=Path)
    args = parser.parse_args()
    directory, manifest, tasks = prepare(args.prepared_dir.resolve(), args.report_run)
    if args.import_reviews:
        print(json.dumps(record_reviews(directory, manifest, tasks, load(args.import_reviews)), ensure_ascii=False))
    elif args.question:
        selected = [t for t in tasks if t['question_id'] == args.question]
        print(json.dumps({'question': selected[0]['question'] if selected else {}, 'total': len(selected)}, ensure_ascii=False))
        for task in selected[args.offset:args.offset + args.limit]:
            print(json.dumps(dossier(task), ensure_ascii=False))
    else:
        print(json.dumps(publish(directory, manifest, tasks), ensure_ascii=False))


if __name__ == '__main__':
    main()
