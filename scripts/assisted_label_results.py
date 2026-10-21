"""Read assisted review results without changing the human policy layer."""
from __future__ import annotations
import json
from pathlib import Path

REVIEW_BUCKETS = (
    {'id': 'human_reviewed', 'label': '人工已审核', 'description': '保留原人工决定，辅助审核意见不能覆盖人工决定。'},
    {'id': 'teacher_suspect', 'label': '教师标签可疑', 'description': '图像、文字及原图复核支持具体教师标签问题；请优先人工确认，尚未自动修改分数。'},
    {'id': 'auto_reviewed', 'label': '证据已复核', 'description': '依据现有实验图像、转录及评分轨迹逐份分析；标签可保留或分歧来自模型、提取、规则，未重新运行实验。'},
    {'id': 'needs_review', 'label': '证据不足待确认', 'description': '已分析现有证据但无法确定归因，不能判定教师标签有错。'},
    {'id': 'pending', 'label': '待审样本', 'description': '触发原候选规则，但尚无已完成的人工决定或证据审核记录。'},
    {'id': 'unflagged', 'label': '未触发规则', 'description': '未入选本次审核；不代表标签已验证正确。'},
)


def review_bucket(row, decision, assisted):
    if decision:
        return 'human_reviewed'
    if assisted:
        category = assisted.get('category')
        if category == 'teacher_suspect':
            return 'teacher_suspect'
        if category in {'label_supported', 'model_or_extraction_issue'}:
            return 'auto_reviewed'
        return 'needs_review'
    return 'unflagged' if row['suspicion_level'] == 'unflagged' else 'pending'


def load_assisted_results(directory: Path):
    manifest_path = directory / 'manifest.json'
    if not manifest_path.is_file():
        return {}, {}, None
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    results = {}
    path = directory / 'results.jsonl'
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('status') != 'completed' or row.get('version') != manifest['version'] or row.get('review_run_id') != manifest['review_run_id']:
                raise ValueError('Assisted result provenance mismatch')
            key = (row['question_id'], row['answer_id'])
            if key in results:
                raise ValueError('Duplicate assisted result')
            results[key] = row
    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
    # results.jsonl is authoritative when publication is between file writes.
    summary.update(target=manifest['target_count'], completed=len(results), pending=manifest['target_count'] - len(results), run_id=manifest['review_run_id'])
    return results, summary, manifest


def compact_result(row):
    return {k: v for k, v in row.items() if k != 'stages'}
