"""Isolated, resumable initial-rubric ablation; never activates shared products."""
from __future__ import annotations

import argparse
import contextlib
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_reviewed_full_workflow as common
from sample_quality import SampleQualityPolicy

SNAPSHOT = ROOT / 'experiments/initial_rubric_ablation/reviewed85_20261019'
BASE = ROOT / 'results_runs/initial_rubric_ablation'
DEFAULT_RUN = 'initial85_glm52_20261020_pc2'


def text_sha(path):
    text = Path(path).read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def source_files():
    names = ['main_pipeline.py', 'step3_rrd_generator.py', 'step4_vlm_grader.py',
             'calibration_utils.py', 'model_runtime.py', 'rubric_semantics.py',
             'sample_quality.py', 'portable_hash.py', 'evaluate.py',
             'requirements-main.lock.txt', 'scripts/calibrate_a3wa.py',
             'scripts/run_csbench.py', 'scripts/run_reviewed_full_workflow.py',
             'scripts/compile_sample_quality_policy.py', 'scripts/run_initial_rubric_ablation.py']
    for folder in ('prompts', 'canonicalizers', 'ocr'):
        names.extend(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*')
                     if p.is_file() and p.suffix in {'.py', '.txt', '.md'} and '__pycache__' not in p.parts)
    return sorted(set(names))


def freeze(source_run):
    if SNAPSHOT.exists():
        raise FileExistsError('Frozen snapshot already exists; do not overwrite: '+str(SNAPSHOT))
    job = ROOT/'results_runs/reviewed_csbench45_glm52'/source_run
    state = common.read(job/'workflow.json')
    common.assert_inputs(state)
    if state['run_id'] != source_run:
        raise ValueError('Source run mismatch')
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.snapshot-', dir=SNAPSHOT.parent))
    try:
        prepared = Path(state['prepared'])
        for name in ('teacher_scores.json', 'answer_metadata.jsonl', 'exam_database.json'):
            shutil.copy2(prepared/name, temporary/name)
        for folder in ('rubrics/initial', 'splits/by_question'):
            shutil.copytree(prepared/folder, temporary/folder)
        for name in ('review_decisions.jsonl', 'sample_policy.json', 'selection.json'):
            shutil.copy2(job/name, temporary/name)
        # Preserve raw-byte dataset hashes across Git checkout on either machine.
        (temporary/'.gitattributes').write_text('* -text\n**/* -text\n', encoding='utf-8')
        assets = {}
        for filename, expected in common.read(job/'input_sha256.json').items():
            path = Path(filename)
            if path.is_relative_to(ROOT) and path.is_relative_to(prepared):
                relative = path.relative_to(prepared)
                if relative.parts[0] in ('student_images', 'reference_images'):
                    assets[path.relative_to(ROOT).as_posix()] = expected
        files = {p.relative_to(temporary).as_posix(): common.sha(p)
                 for p in temporary.rglob('*') if p.is_file()}
        contract = {'version':1, 'source_run_id':source_run,
                    'source_selection_sha256':state['selection_sha256'],
                    'files':files, 'asset_sha256':assets,
                    'source_lf_sha256':{name:text_sha(ROOT/name) for name in source_files()},
                    'model':state['model'], 'rubric_source':'initial',
                    'note':'Stage-1 optimization removed. Stage-4/5 refitted on initial-rubric validation only.'}
        common.write(temporary/'snapshot_contract.json', contract)
        common.assert_inputs(state)
        temporary.rename(SNAPSHOT)
    except BaseException:
        # A failed preparation is retained for inspection, never recursively deleted.
        raise
    print('Frozen ablation inputs:', SNAPSHOT)
    print('No model calls, no shared active configuration changed.')


def check_snapshot():
    contract = common.read(SNAPSHOT/'snapshot_contract.json')
    model = contract['model']
    if (model.get('text_model') != 'glm-5.2' or model.get('thinking') != 'disabled'
            or model.get('vlm_model') != 'glm-4.6v'):
        raise ValueError('The frozen main experiment uses a different model contract')
    common.verify_proof({str(SNAPSHOT/name): digest for name, digest in contract['files'].items()})
    common.verify_proof({str(ROOT/name): digest for name, digest in contract['asset_sha256'].items()})
    for name, expected in contract['source_lf_sha256'].items():
        if text_sha(ROOT/name) != expected:
            raise RuntimeError('Code differs from the paired ablation contract: '+name)
    policy = SampleQualityPolicy.load(SNAPSHOT, explicit_path=SNAPSHOT/'sample_policy.json', force_raw=False)
    if policy.mode != 'active':
        raise ValueError('Frozen reviewed policy not active')
    selection = common.read(SNAPSHOT/'selection.json')
    questions = selection['questions']
    if len(questions) != 45:
        raise ValueError('Expected the same 45-question dataset')
    return contract, policy, selection


def runtime_database(job):
    rows = common.read(SNAPSHOT/'exam_database.json')
    for row in rows:
        qid = row['question_id']
        group = row.get('rubric_group', qid.split('_')[0])
        row['initial_rubric_path'] = str(SNAPSHOT/'rubrics/initial'/group/(qid+'_rubric_standard.json'))
        row['optimized_rubric_path'] = str(job/'no_optimized_rubrics'/group/(qid+'_rubric_standard.json'))
        row['rubric_split_path'] = str(SNAPSHOT/'splits/by_question'/(qid+'.json'))
    return rows


def grade_command(state, split):
    job = Path(state['job'])
    return [sys.executable, '-u', str(ROOT/'main_pipeline.py'), '--mode', 'FULL',
            '--questions', *state['questions'], '--database-path', str(job/'exam_database.json'),
            '--teacher-db', str(SNAPSHOT/'teacher_scores.json'),
            '--answer-metadata', str(SNAPSHOT/'answer_metadata.jsonl'),
            '--sample-policy', str(job/'sample_policy.json'),
            '--initial-rubric-dir', str(SNAPSHOT/'rubrics/initial'),
            '--rubric-dir', str(job/'no_optimized_rubrics'), '--allow-initial-rubric',
            '--extraction-backend', 'csbench_hybrid', '--ocr-cache-dir', str(job/'cache'),
            '--answer-split', split, '--results-dir', str(job/split),
            '--progress-file', str(job/split/'progress.json'), '--log-dir', str(job/'logs'),
            '--run-id', state['run_id']+'_'+split]


def stages(state):
    job = Path(state['job'])
    py = [sys.executable, '-u']
    files = [str(job/'validation'/(qid+'_grading_checkpoint.json')) for qid in state['questions']]
    calibration = [*py, str(ROOT/'scripts/calibrate_a3wa.py'), '--files', *files,
        '--teacher-db', str(SNAPSHOT/'teacher_scores.json'), '--database-path', str(job/'exam_database.json'),
        '--output', str(job/'a3wa_config.json'), '--score-calibration',
        '--diagnostic-report-dir', str(job/'calibration_diagnostics'),
        '--text-provider', 'glm5', '--text-model', 'glm-5.2', '--thinking-mode', 'disabled',
        '--vlm-provider', 'glm4v', '--vlm-model', 'glm-4.6v']
    evaluation = [*py, str(ROOT/'evaluate.py'), '--questions', *state['questions'],
        '--results-dir', str(job/'test'), '--result-source', 'checkpoint', '--compare',
        '--compare-score-keys', 'single', 'avg', 'selected', '3wd-core', '3wd',
        '--teacher-db', str(SNAPSHOT/'teacher_scores.json'), '--database-path', str(job/'exam_database.json'),
        '--compare-output', str(job/'reports/score_comparison.csv'),
        '--summary-output', str(job/'reports/summary.json')]
    return [
        {'name':'grade_validation', 'command':[ *py, str(Path(__file__).resolve()), '--run-id', state['run_id'], '--grade-split', 'validation'], 'attempts':36, 'proof':files},
        {'name':'calibrate', 'command':calibration, 'attempts':1, 'proof':[str(job/'a3wa_config.json')]},
        {'name':'grade_test', 'command':[*py, str(Path(__file__).resolve()), '--run-id', state['run_id'], '--grade-split', 'test'], 'attempts':36, 'proof':[str(job/'test'/(q+'_grading_checkpoint.json')) for q in state['questions']]},
        {'name':'evaluate_test', 'command':evaluation, 'attempts':2, 'proof':[str(job/'reports/summary.json'),str(job/'reports/score_comparison.csv')]},
    ]


def verify_complete(state, split):
    policy = SampleQualityPolicy.load(SNAPSHOT, explicit_path=Path(state['job'])/'sample_policy.json', force_raw=False)
    for qid in state['questions']:
        expected = set(policy.filter_ids(qid, common.read(SNAPSHOT/'splits/by_question'/(qid+'.json'))[split]))
        path = Path(state['job'])/split/(qid+'_grading_checkpoint.json')
        rows = common.read(path) if path.is_file() else []
        ids = [row.get('student_id') for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected or any(
                row.get('model_avg_score') is None or row.get('three_way_core_score') is None
                or row.get('final_calibrated_score') is None for row in rows):
            raise RuntimeError('Incomplete '+split+' scoring: '+qid+'; resume the same run')


def validate_commands(state):
    import main_pipeline
    job = Path(state['job'])
    for split in ('validation','test'):
        command = grade_command(state, split)
        with patch.object(sys, 'argv', command[2:]):
            args = main_pipeline.parse_args()
        if not args.allow_initial_rubric or Path(args.teacher_db).parent != SNAPSHOT:
            raise ValueError('Unsafe initial-rubric command')
    with (patch.object(main_pipeline, 'RUBRIC_DIR', str(job/'no_optimized_rubrics')),
          patch.object(main_pipeline, 'INITIAL_RUBRIC_DIR', str(SNAPSHOT/'rubrics/initial')),
          patch.object(main_pipeline, 'ALLOW_INITIAL_RUBRIC', True)):
        for row in runtime_database(job):
            actual = Path(main_pipeline.rubric_path_for(row['question_id'], row)).resolve()
            if actual != Path(row['initial_rubric_path']).resolve():
                raise RuntimeError('An optimized rubric would be used: '+row['question_id'])
            rubric = common.read(actual)
            if abs(main_pipeline.rubric_total(rubric)-float(row['total_score'])) > 1e-6:
                raise ValueError('Initial rubric total differs: '+row['question_id'])


def prepare(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id):
        raise ValueError('Invalid run ID')
    job = BASE/run_id
    if (job/'workflow.json').is_file():
        state = common.read(job/'workflow.json')
        common.assert_inputs(state)
        return state
    _, _, selection = check_snapshot()
    job.mkdir(parents=True, exist_ok=False)
    for name in ('sample_policy.json','review_decisions.jsonl','selection.json'):
        shutil.copy2(SNAPSHOT/name, job/name)
    common.write(job/'exam_database.json', runtime_database(job))
    inputs = {str(p.resolve()):common.sha(p) for p in SNAPSHOT.rglob('*') if p.is_file()}
    inputs.update({str(ROOT/name):common.sha(ROOT/name) for name in source_files()})
    inputs.update({str(ROOT/name):digest for name, digest in
                   common.read(SNAPSHOT/'snapshot_contract.json')['asset_sha256'].items()})
    inputs[str(job/'exam_database.json')] = common.sha(job/'exam_database.json')
    common.write(job/'input_sha256.json', inputs)
    state = {'run_id':run_id,'job':str(job),'prepared':str(SNAPSHOT),'questions':selection['questions'],
        'status':'prepared','stage':'prepared','completed':{},'attempts':{},
        'review_sha256':common.sha(job/'review_decisions.jsonl'), 'policy_sha256':common.sha(job/'sample_policy.json'),
        'selection_sha256':common.sha(job/'selection.json'), 'input_manifest_sha256':common.sha(job/'input_sha256.json'),
        'rubric_source':'initial', 'model':{'text_model':'glm-5.2','thinking':'disabled','vlm_model':'glm-4.6v'}}
    common.write(job/'workflow.json', state)
    return state


def child_environment(state, split=None):
    job = Path(state['job'])
    env = common.frozen_environment(job/'sample_policy.json')
    env['REFGRADER_ALLOW_EXPERIMENTAL_A3WA'] = '1'
    if split == 'test':
        if not (job/'a3wa_config.json').is_file():
            raise FileNotFoundError('Initial-rubric validation calibration is required before test')
        env['A3WA_CALIBRATION_CONFIG'] = str(job/'a3wa_config.json')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--freeze-from-main')
    action.add_argument('--check', action='store_true')
    action.add_argument('--background', action='store_true')
    action.add_argument('--status', action='store_true')
    action.add_argument('--grade-split', choices=['validation','test'], help=argparse.SUPPRESS)
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    args = parser.parse_args()
    if args.freeze_from_main:
        freeze(args.freeze_from_main)
        return 0
    common.BASE = BASE  # Process-local lock directory; does not edit the other workflow.
    if args.status:
        path = BASE/args.run_id/'workflow.json'
        if not path.is_file():
            print('Not started:',args.run_id)
            return 0
        state = common.read(path)
        try:
            with common.workflow_lock(): running=False
        except RuntimeError: running=True
        print(json.dumps({'running':running, **{k:state.get(k) for k in ('run_id','status','stage','pid','console_log','error','result_class')}},ensure_ascii=False,indent=2))
        return 0
    if args.grade_split:
        state = common.read(BASE/args.run_id/'workflow.json')
        common.assert_inputs(state)
        rc = subprocess.run(grade_command(state,args.grade_split),cwd=ROOT,env=child_environment(state,args.grade_split)).returncode
        if rc: return rc
        verify_complete(state,args.grade_split)
        return 0
    _, _, selection = check_snapshot()
    with contextlib.redirect_stdout(io.StringIO()):
        validate_commands({'run_id':args.run_id,'job':str(BASE/args.run_id),'questions':selection['questions']})
    if args.check:
        from ocr.backend import default_ocr_python
        if not Path(os.getenv('REFGRADER_OCR_PYTHON', str(default_ocr_python()))).is_file():
            raise FileNotFoundError('OCR venv missing; run scripts/setup_paddle_ocr.ps1')
        print(json.dumps({'offline_preflight':'passed','rubric_source':'initial','questions':len(selection['questions']),
            'splits':selection['splits'],'source_run_id':common.read(SNAPSHOT/'snapshot_contract.json')['source_run_id'],
            'api_base_url':'https://open.bigmodel.cn/api/coding/paas/v4','api_calls_made':0},ensure_ascii=False,indent=2))
        return 0
    with common.workflow_lock():
        state = prepare(args.run_id)
    if state['status']=='completed':
        print('Already completed:',args.run_id)
        return 0
    if not (os.getenv('ZHIPUAI_API_KEY') or os.getenv('ZHIPU_API_KEY')):
        if not sys.stdin.isatty(): raise RuntimeError('Set ZHIPUAI_API_KEY in the launching terminal')
        key = getpass.getpass('GLM API key (hidden): ').strip()
        if not key: raise ValueError('Empty key')
        os.environ['ZHIPUAI_API_KEY'] = key
    if args.background:
        log = Path(state['job'])/'console.log'
        common.save(state, console_log=str(log))
        flags = subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        with log.open('ab') as handle:
            process = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--run-id',args.run_id],
                cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                creationflags=flags,start_new_session=os.name!='nt')
        print(f'Background ablation launched. PID={process.pid}\nRun ID: {args.run_id}\nLog: {log}')
        return 0
    with common.workflow_lock(), common.keep_awake():
        common.save(state,pid=os.getpid(),status='running')
        try:
            for item in stages(state):
                common.execute_stage(state,item,child_environment(state))
            common.save(state,status='completed',stage='completed')
        except BaseException as exc:
            common.save(state,status='failed',error=str(exc))
            raise
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError) as exc:
        print('ERROR:',exc,file=sys.stderr)
        raise SystemExit(1)
