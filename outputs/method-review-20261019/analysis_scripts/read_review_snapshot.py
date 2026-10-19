import json
from collections import Counter
from pathlib import Path
base=Path('data/csbench')
p=base/'quality_control/reviews/teacher_label_audit_all43_20260727_160210_decisions.jsonl'
rows=[json.loads(line) for line in p.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
print('review_file',str(p),'rows',len(rows))
print('keys',list(rows[0]) if rows else [])
print('decisions',dict(Counter(r.get('decision','missing') for r in rows)))
print('questions',len(set(r.get('question_id') for r in rows)))
print('question_counts',dict(Counter(r.get('question_id') for r in rows)))
print('last_updated',max(str(r.get('reviewed_at',r.get('updated_at',''))) for r in rows) if rows else '')
a=base/'quality_control/policies/active_sample_policy.json'
print('active_policy_exists',a.exists())
print('policy_files',[x.name for x in (base/'quality_control/policies').glob('*.json')])
d=json.loads((base/'splits/by_question/CO_1.json').read_text(encoding='utf8'))
print('splitkeys',list(d) if isinstance(d,dict) else type(d).__name__)
print('split_shapes',{k:len(v) if isinstance(v,(list,dict)) else type(v).__name__ for k,v in d.items()})
