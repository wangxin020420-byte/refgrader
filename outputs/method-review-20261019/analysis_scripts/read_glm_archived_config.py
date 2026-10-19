import json,hashlib,math
from pathlib import Path
p=Path(r'C:\Users\wx\Desktop\refgrader-artifacts\csbench\CO_1\grading_runs\csbench45_full_experimental_glm52_20260825_163432\calibration\a3wa_config.json');d=json.loads(p.read_text(encoding='utf8'));print('HASH',hashlib.sha256(p.read_bytes()).hexdigest());print('KEYS',list(d))
for k in ['loss_params','membership_model','thresholds','boundary_policy','operational_costs','deployment_gate','validation_summary']:
 print(k,json.dumps(d.get(k),ensure_ascii=False))
for k in ['candidate_diagnostics','cross_validation']:
 v=d.get(k)or{};print(k,json.dumps({a:b for a,b in v.items() if a not in ['folds','top_candidates']},ensure_ascii=False)[:2500])
print('riskgroups',{k:v.get('membership') for k,v in (d.get('risk_diagnostics')or{}).items() if isinstance(v,dict) and 'membership' in v})
