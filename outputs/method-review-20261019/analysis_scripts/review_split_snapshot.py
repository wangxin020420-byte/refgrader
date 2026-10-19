import json,os,hashlib
from pathlib import Path
from collections import Counter,defaultdict
base=Path('data/csbench')
p=base/'quality_control/reviews/teacher_label_audit_all43_20260727_160210_decisions.jsonl'
raw=p.read_bytes();rows=[json.loads(line)for line in raw.decode('utf-8-sig').splitlines()if line.strip()]
idx={}
for f in (base/'splits/by_question').glob('*.json'):
 d=json.loads(f.read_text(encoding='utf8'));qid=d.get('question_id',f.stem)
 for split in ['calibration','validation','test']:
  for aid in d.get(split,[]):idx[(qid,str(aid))]=split
counts=defaultdict(Counter);reasons=defaultdict(Counter)
for r in rows:
 counts[idx.get((r['question_id'],r['answer_id']),'unmapped')][r['decision']]+=1
 reasons[r['decision']][r.get('reason_code','')]+=1
print('split_decisions',{k:dict(v)for k,v in counts.items()})
print('reason_codes',{k:dict(v)for k,v in reasons.items()})
print('policy_env',os.getenv('REFGRADER_SAMPLE_POLICY'),'mode_env',os.getenv('REFGRADER_SAMPLE_POLICY_MODE'))
print('snapshot_sha256',hashlib.sha256(raw).hexdigest())
print('reviewed_at_range',min(r.get('reviewed_at','')for r in rows),max(r.get('reviewed_at','')for r in rows))
print('noise_note_examples',[r.get('review_note','')[:100]for r in rows if r['decision']=='confirmed_noise'][:3])
print('all_q',len({q for q,a in idx}))
