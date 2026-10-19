#!/usr/bin/env python3
"""Offline preparation only; never trains SCALE or changes upstream data."""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import zipfile

COMMIT = '9e1a866d58edaa0d78af374d2f3529a10d72239a'
ARCHIVE_SHA = '5194a3d89f2fbc37e4624b00b8bf946bc2907ac1a4ea59d132ad5bcb6a93bc66'
SPECS = {
    'ADS': ('all.jsonl', 'train.jsonl', 'test.jsonl', None, None),
    'ASAG': ('train_zh.csv', 's_train_zh.csv', 's_eval_zh.csv', 4, 1),
    'LE': ('train.csv', 's_train_zh.csv', 's_eval_zh.csv', 4, 3),
    'SR': ('sentences-goldstandard.csv', 's_train_zh.csv', 's_eval_zh.csv', 8, 3),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stable_write(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError('Existing file differs; left unchanged: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def load_rows(path, width, score_index):
    data = path.read_text(encoding='utf-8-sig')
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        required = {'q_id', 'question', 'answer', 'score', 'reference1', 'reference2', 'reference3'}
        if any(set(row) != required for row in rows):
            raise ValueError('Unexpected ADS schema: ' + str(path))
        scores = [float(row['score']) for row in rows]
        qids = [row['q_id'] for row in rows]
        content = [(row['q_id'], row['answer']) for row in rows]
    else:
        rows = list(csv.reader(io.StringIO(data), delimiter='\t'))
        if any(len(row) != width for row in rows):
            raise ValueError('Unexpected TSV schema: ' + str(path))
        scores = [float(row[score_index]) for row in rows]
        qids = [row[6] if path.parent.name == 'SR' else row[0] for row in rows]
        answer_index = {'ASAG': 3, 'LE': 2, 'SR': 1}[path.parent.name]
        content = list(zip(qids, [row[answer_index] for row in rows]))
    if not rows or not all(math.isfinite(score) for score in scores):
        raise ValueError('Empty data or invalid score: ' + str(path))
    return rows, scores, qids, content


def audit(root):
    report = {'commit': COMMIT, 'archive_sha256': ARCHIVE_SHA,
              'normalization': 'none; original labels retained',
              'protocol_status': 'upstream file audit, not confirmed paper protocol',
              'datasets': {}, 'warnings': [
                  'ADS README says 80/20, actual files are 1107/475 (~70/30).',
                  'Root README says ADS scores normalized 0..1; actual labels are 0..10.',
                  'ASAG README says 630; this pinned version contains 629 records.',
                  'SCALE full training implementation and final paper protocol are unverified.',
                  'SR reference identifiers are not verified original question identifiers.',
              ]}
    for name, (all_file, train_file, test_file, width, score_index) in SPECS.items():
        entries = {}
        loaded = {}
        for split, filename in [('all', all_file), ('train', train_file), ('test', test_file)]:
            path = root / name / filename
            rows, scores, qids, content = load_rows(path, width, score_index)
            loaded[split] = (rows, content)
            entries[split] = {'file': name + '/' + filename, 'sha256': sha(path.read_bytes()),
                              'rows': len(rows), 'group_ids': len(set(qids)),
                              'score_min': min(scores), 'score_max': max(scores),
                              'score_values': sorted(set(scores))}
        counters = {split: collections.Counter(json.dumps(row, sort_keys=True, ensure_ascii=True)
                    for row in loaded[split][0]) for split in loaded}
        train_content = collections.Counter(loaded['train'][1])
        test_content = collections.Counter(loaded['test'][1])
        entries['partition_matches_all'] = counters['all'] == counters['train'] + counters['test']
        entries['train_test_identical_record_overlap'] = sum((counters['train'] & counters['test']).values())
        entries['train_test_same_group_answer_overlap'] = sum((train_content & test_content).values())
        if not entries['partition_matches_all']:
            raise ValueError(name + ': train + test do not match all records')
        if entries['train_test_same_group_answer_overlap']:
            report['warnings'].append(name + ': identical group/answer appears across splits; preserve upstream and review leakage separately.')
        report['datasets'][name] = entries
    report['files'] = {str(path.relative_to(root)).replace('\\', '/'): sha(path.read_bytes())
                       for path in sorted(root.rglob('*')) if path.is_file()}
    return report


def probe(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors='replace', timeout=15)
        return {'returncode': result.returncode, 'output': (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'unavailable': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projects-root', type=Path, default=Path.home() / 'projects')
    parser.add_argument('--inventory-only', action='store_true', help='Record hardware without installing or training')
    args = parser.parse_args()
    projects = args.projects_root.expanduser().resolve()
    projects.mkdir(parents=True, exist_ok=True)
    scale = projects / 'refgrader-reproductions' / 'scale'
    for folder in ('control', 'sources', 'data', 'env', 'runs', 'reports', 'logs'):
        (scale / folder).mkdir(parents=True, exist_ok=True)
    if args.inventory_only:
        report = {'platform': platform.platform(), 'python': platform.python_version(),
                  'projects_root': str(projects), 'cpu_count': os.cpu_count(),
                  'disk_free_bytes': shutil.disk_usage(projects).free,
                  'gpu': probe(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,driver_version', '--format=csv']),
                  'scheduler': {name: shutil.which(name) for name in ('srun', 'sbatch', 'qsub', 'tmux', 'conda')},
                  'memory': probe(['free', '-h']), 'git': probe(['git', '--version'])}
        payload = json_bytes(report)
        path = scale / 'reports' / ('server_inventory_' + sha(payload)[:12] + '.json')
        stable_write(path, payload)
        print('Inventory: ' + str(path))
        return
    kit = Path(__file__).resolve().parent
    archive = kit / 'vendor' / 'SAS-Datasets-9e1a866d.zip'
    if sha(archive.read_bytes()) != ARCHIVE_SHA:
        raise ValueError('Dataset archive checksum mismatch')
    data_root = scale / 'data' / ('SAS-Datasets-' + COMMIT)
    with zipfile.ZipFile(archive) as z:
        files = []
        for info in z.infolist():
            parts = PurePosixPath(info.filename).parts
            if not parts or parts[0] != 'SAS-Datasets-' + COMMIT or '..' in parts:
                raise ValueError('Unexpected archive path')
            if info.is_dir():
                continue
            target = data_root.joinpath(*parts[1:])
            if not target.resolve().is_relative_to(data_root.resolve()):
                raise ValueError('Archive path outside destination')
            payload = z.read(info)
            if target.exists() and target.read_bytes() != payload:
                raise ValueError('Modified upstream file; no overwrite: ' + str(target))
            files.append((target, payload))
        for path, payload in files:
            stable_write(path, payload)
    report = audit(data_root)
    stable_write(scale / 'reports' / 'data_audit.json', json_bytes(report))
    workspace = {'folders': [{'name': 'projects', 'path': '.'}],
                 'settings': {'git.openRepositoryInParentFolders': 'never', 'git.autofetch': False,
                              'scm.alwaysShowRepositories': True}}
    stable_write(projects / 'Projects-Only.code-workspace', json_bytes(workspace))
    for filename in ('prepare_scale.py', 'PLAN.md', 'PROTOCOL_CHECKLIST.json', 'README.md'):
        stable_write(scale / 'control' / filename, (kit / filename).read_bytes())
    stable_write(scale / 'control' / 'vendor' / archive.name, archive.read_bytes())
    print('Prepared: ' + str(scale))
    print('Workspace: ' + str(projects / 'Projects-Only.code-workspace'))
    print('Data audit: ' + str(scale / 'reports' / 'data_audit.json'))
    for name, entries in report['datasets'].items():
        print(name + ': all=' + str(entries['all']['rows']) + ', train=' + str(entries['train']['rows']) + ', test=' + str(entries['test']['rows']))
    print('Status: PREPARATION_ONLY. Training has NOT started. Read control/PLAN.md.')


if __name__ == '__main__':
    main()
