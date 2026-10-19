"""Bind supplied semantic-review decisions to server records; export affected questions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def text_sha(text):
    return sha(text.replace('\r\n', '\n').strip().encode('utf-8'))


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def save(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing review artifact differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def prepare(root, decisions_path):
    folder = root / 'paper_method_v1/artifacts/proxy_generated_v1'
    data = (folder / 'proxy_records.jsonl').read_bytes()
    manifest = json.loads((folder / 'manifest.json').read_bytes())
    if sha(data) != manifest['records_sha256']:
        raise ValueError('Proxy records do not match manifest')
    records = [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    lookup = {r['sample_id']: r for r in records}
    if len(records) != 885 or len(lookup) != 885:
        raise ValueError('Expected 885 unique source records')
    if any(r['assigned_label'] is not None for r in records):
        raise ValueError('Unexpected label state')
    decisions_bytes = decisions_path.read_bytes()
    review = json.loads(decisions_bytes)
    seen = set()
    affected = set()
    for item in review['decisions']:
        sid = item['sample_id']
        if sid in seen or item['decision'] not in ('PASS', 'FAIL', 'UNCERTAIN'):
            raise ValueError('Invalid review entry: ' + sid)
        seen.add(sid)
        row = lookup[sid]
        if text_sha(row['original_answer']) != item['original_text_sha256']:
            raise ValueError('Original text differs from reviewed attachment: ' + sid)
        if text_sha(row['proxy_answer']) != item['proxy_text_sha256']:
            raise ValueError('Proxy text differs from reviewed attachment: ' + sid)
        if item['decision'] != 'PASS':
            affected.add(str(row['question_id']))
    counts = dict(Counter(x['decision'] for x in review['decisions']))
    if len(seen) != 105 or counts != {'PASS': 101, 'FAIL': 1, 'UNCERTAIN': 3}:
        raise ValueError('Unexpected review scope')
    pending = sorted((r for r in records if r['kind'] == 'generated_proxy'
                      and r['sample_id'] not in seen
                      and str(r['question_id']) in affected), key=lambda r: r['sample_id'])
    tag = sha(data)[:12] + '_' + sha(decisions_bytes)[:12]
    out = root / 'paper_method_v1/reviews' / ('proxy_followup_' + tag)
    report = {
        'status': 'PROXY_REVIEW_FOLLOWUP_REQUIRED', 'reviewed_pairs': len(seen),
        'decisions': counts, 'affected_question_count': len(affected),
        'additional_pairs_to_review': len(pending),
        'unreviewed_generated_pairs_total': sum(r['kind'] == 'generated_proxy' for r in records) - len(seen),
        'records_sha256': sha(data), 'decisions_sha256': sha(decisions_bytes),
        'labels_assigned': 0, 'api_calls': 0, 'training_started': False,
        'test_accessed': False, 'ready_for_annotator_training': False,
        'note': 'Review artifacts only. No generated answer, original score, or training record is modified.'
    }
    save(out / 'review_decisions.json', decisions_bytes)
    save(out / 'summary.json', encoded(report))
    unresolved = [dict(lookup[x['sample_id']], review_decision=x['decision'], review_note=x['note'])
                  for x in review['decisions'] if x['decision'] != 'PASS']
    save(out / 'unresolved_pairs.json', encoded(unresolved))
    for offset in range(0, len(pending), 40):
        lines = ['# Additional proxy semantic review',
                 'Only previously unreviewed pairs from affected questions. No new scores assigned.', '']
        for r in pending[offset:offset + 40]:
            lines.extend(['## ' + r['sample_id'], '', '题目：' + r['question'], '',
                          '### 原答案', r['original_answer'], '', '### 代理答案', r['proxy_answer'], '',
                          '审核结论：', '', '差异说明：', ''])
        save(out / ('additional_review_%02d.md' % (offset // 40 + 1)), '\n'.join(lines).encode('utf-8'))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('REVIEW_DIRECTORY:', out)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--decisions', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.decisions.resolve())
