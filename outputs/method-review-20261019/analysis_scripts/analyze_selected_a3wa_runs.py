import json,math,statistics,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
root=Path.cwd(); out=root/'outputs/method-review-20261019';out.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
teacher=read(root/'data/csbench/teacher_scores.json');maxima={r['question_id']:r['total_score'] for r in read(root/'data/csbench/exam_database.json')}
cfgpaths={'glm':Path(r'C:\Users\wx\Desktop\refgrader-artifacts\csbench\CO_1\grading_runs\csbench45_full_experimental_glm52_20260825_163432\calibration\a3wa_config.json'),'deepseek':root/'results_runs/a3wa_configs/private45_validation_deepseek_official_flash_t0_20260927_200831.json'}
runs=[('glm','validation','csbench45_validation_glm52_20260822_090920'),('glm','test','csbench45_full_experimental_glm52_20260825_163432'),('deepseek','validation','private45_validation_deepseek_official_flash_t0_20260927_200831'),('deepseek','test','private45_test_deepseek_official_flash_t0_20260927_200831')]
res={'configs':{},'runs':{}};sig=lambda x:1/(1+math.exp(-x))
for name,path in cfgpaths.items():
 d=read(path);m=d['membership_model'];ws=m['coefficients'];summ=d['validation_summary'];diag=d.get('candidate_diagnostics',{})
 res['configs'][name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'loss_params':d['loss_params'],'thresholds':d['thresholds'],'membership':m,'theoretical_mu_range':[sig(m['intercept']-sum(ws.values())),sig(m['intercept'])],'boundary_policy':d['boundary_policy'],'validation_summary':summ,'feasible':diag.get('feasible_candidate_count'),'tradeoff':diag.get('feasibility_tradeoff'),'deployment_gate':d['deployment_gate'],'cv_summary':{k:v for k,v in (d.get('cross_validation')or{}).items() if k!='folds'},'risk_summary':{k:v.get('membership') for k,v in (d.get('risk_diagnostics')or{}).items() if isinstance(v,dict) and 'membership'in v},'score_uncertainty':d.get('score_uncertainty')}
for model,split,rid in runs:
 group='validation'if split=='validation'else'full';base=root/f'results_runs/csbench_q45_12039bfe25ca4168_{group}/runs/{rid}';stats=Counter();routes=Counter();outcomes=Counter();gates=Counter();conf=Counter();params=Counter();same3=0;mus=[];records=[]; evidence_oldroutes=Counter();differences=Counter();total_gain=0;matchmu=[]
 for p in sorted(base.glob('*_grading_checkpoint.json')):
  qid=p.name.removesuffix('_grading_checkpoint.json');M=maxima[qid]
  for row in read(p):
   if not isinstance(row,dict):continue
   y=teacher.get(str(row.get('student_id')),{}).get(qid)
   if y is None:stats['missing_teacher']+=1;continue
   a=row.get('a3wa_decision')or{};bg=row.get('boundary_gate')or{};r=row.get('3wd_route','missing');mu=a.get('mu'); baseval=row.get('selected_baseline_score',row.get('model_avg_score')); core=row.get('three_way_core_score',row.get('final_calibrated_score'));final=row.get('final_calibrated_score');unsafe=abs(baseval-y)>max(.5,.1*M);pr=(row.get('post_calibration')or{}).get('primary_risks')or(row.get('post_calibration')or{}).get('three_way_primary_risks')or{};actual=a.get('risk_components')or{}
   stats['n']+=1;routes[r]+=1;outcomes[row.get('3wd_sequential_outcome','missing')]+=1;gates[bg.get('gate_reason','none')]+=1;params[(a.get('alpha'),a.get('beta'))]+=1
   scores=row.get('model_scores_history')or[]
   if len(scores)>=2 and len(set(scores))==1:same3+=1
   if row.get('requires_human_arbitration'):stats['human']+=1
   if a.get('route')!=r:stats['route_changed']+=1
   if a.get('reason','').startswith('hard_neg'):stats['hard_neg']+=1
   if a.get('reason')=='membership_le_beta':stats['threshold_neg']+=1
   if 'conformal_interval_exceeds_safe_tolerance'in a.get('review_signals',[]):stats['wide_interval']+=1
   if r=='POS'and'conformal_interval_exceeds_safe_tolerance'in a.get('review_signals',[]):stats['wide_interval_POS']+=1
   if mu is not None:mus.append(mu)
   s=bg.get('agent_evidence_summary')or{}
   if s.get('has_agent_evidence'):stats['agent_evidence']+=1;evidence_oldroutes[r]+=1
   if len(scores)>=2 and len(set(scores))==1 and unsafe:stats['stable_but_unsafe']+=1
   if all(float(actual.get(k,0))==0 for k in ['U_E','U_S','U_R']):
    stats['zero_risk']+=1
    if unsafe:stats['zero_risk_unsafe']+=1
   if r=='POS':stats['POS_unsafe']+=int(unsafe)
   if row.get('3wd_sequential_outcome')=='auto_accepted_canonical_lock':
    stats['canonical_lock']+=1
    if a.get('route')!='POS':stats['canonical_nonPOS']+=1
    if unsafe:stats['canonical_unsafe_base']+=1
   pc=row.get('post_calibration')or{}
   if pc.get('primary_risks'):
    for k in ['U_E','U_S','U_R']:
     if abs(float(pr.get(k,0))-float(actual.get(k,0)))>1e-5:differences[k]+=1
   records.append({'qid':qid,'id':row.get('student_id'),'y':y,'M':M,'mean':row.get('model_avg_score'),'base':baseval,'core':core,'final':final,'mu':mu,'unsafe':unsafe,'route':r,'human':bool(row.get('requires_human_arbitration')),'evidence':bool(s.get('has_agent_evidence')),'risks':[actual.get(k,0)for k in ['U_E','U_S','U_R']]})
 def metr(key):
  v=[r[key]-r['y'] for r in records if r.get(key)is not None];return {'MAE':sum(abs(x)for x in v)/len(v),'Bias':sum(v)/len(v)}
 # Pairwise-ranking AUC for safety: positive=safe, higher mu safer.
 vals=[r for r in records if r['mu']is not None];posit=[r['mu'] for r in vals if not r['unsafe']];neg=[r['mu'] for r in vals if r['unsafe']];auc=sum((a>b)+.5*(a==b)for a in posit for b in neg)/max(1,len(posit)*len(neg))
 result={'run_id':rid,'path':str(base),'stats':dict(stats),'routes':dict(routes),'outcomes':dict(outcomes),'gate_reasons':dict(gates),'param_counts':{str(k):v for k,v in params.items()},'all_equal_score_count':same3,'mu_quantiles':dict(zip(['min','p10','p50','p90','max'],map(float,np.quantile(mus,[0,.1,.5,.9,1])))),'mu_distinct':len(set(mus)),'safety_auc':auc,'metrics':{k:metr(k)for k in ['mean','base','core','final']},'loader_vs_runtime_risk_mismatch_count':dict(differences),'agent_evidence_routes':dict(evidence_oldroutes)}
 res['runs'][f'{model}_{split}']=result
 (out/f'{model}_{split}_case_summary.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
 print(f'{model}_{split}',json.dumps(result,ensure_ascii=False))
(out/'verified_findings.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf8')
print('CONFIG_RANGES',{k:v['theoretical_mu_range']for k,v in res['configs'].items()})
