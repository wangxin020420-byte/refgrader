import sys,json,math
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path.cwd()))
from calibration_utils import build_a3wa_decision
from scripts.calibrate_a3wa import load_records,load_question_scores
root=Path.cwd();out=root/'outputs/method-review-20261019';find=json.loads((out/'verified_findings.json').read_text(encoding='utf8'));teacher=json.loads((root/'data/csbench/teacher_scores.json').read_text(encoding='utf8'));scores=load_question_scores(root/'data/csbench/exam_database.json');report={}
for model in ['glm','deepseek']:
 cfg=json.loads(Path(find['configs'][model]['path']).read_text(encoding='utf8'));base=Path(find['runs'][model+'_validation']['path']);rs=load_records(list(base.glob('*_grading_checkpoint.json')),teacher,scores);rows=[];oldE=Counter();summary={'n':len(rs),'new_BND_missing_evidence':0,'new_BND_available_evidence':0,'old_BND_available_evidence':0};confcounts=Counter();rawmiss=allowedmiss=rawover=allowedover=0
 for r in rs:
  d=build_a3wa_decision(model_scores=r['model_scores'],avg_model_score=r['avg'],std_dev=r['std_dev'],max_score=r['max_score'],blank_rate=r['blank_rate'],low_quality_rate=r['low_quality_rate'],perception_failure_rate=r['perception_failure_rate'],extraction_quality=r['extraction_quality'],structure_missing_rate=r['structure_missing_rate'],extraction_risk=r['extraction_risk'],fatal_points_ratio=r['fatal_points_ratio'],high_blank_high_score=r['high_blank_high_score'],post_calibration=r['post_calibration'],loss_params=cfg['loss_params'],membership_model=cfg['membership_model'],score_uncertainty=cfg.get('score_uncertainty'))
  ev=r.get('agent_evidence')or{};has=ev.get('has_agent_evidence',False)
  if d['route']=='BND':summary['new_BND_available_evidence'if has else'new_BND_missing_evidence']+=1
  if r['old_route']=='BND'and has:
   summary['old_BND_available_evidence']+=1
   confcounts[str(ev.get('confidence'))]+=1
   rawmiss+=ev.get('missed_points',0);allowedmiss+=ev.get('allowed_missed_points',0);rawover+=ev.get('over_points',0);allowedover+=ev.get('allowed_over_points',0)
  rows.append((d['mu'],abs(r['avg']-r['teacher'])>max(.5,.1*r['max_score'])))
 groups=[]
 for t in sorted(set(x[0]for x in rows)):
  a=[unsafe for mu,unsafe in rows if mu>=t];groups.append({'alpha':t,'n':len(a),'unsafe':sum(a),'rate':sum(a)/len(a)})
 feasible=[g for g in groups if g['n']>=1 and g['rate']<=.1]
 summary.update({'evidence_confidences':dict(confcounts),'sum_raw_missed':rawmiss,'sum_allowed_missed':allowedmiss,'sum_raw_over':rawover,'sum_allowed_over':allowedover,'nonempty_POS_empirical_safe_threshold_count':len(feasible),'max_empirically_safe_POS':max(feasible,key=lambda x:x['n'])if feasible else None,'top_score_tie':groups[-1],'minimum_nonempty_POS_unsafe_rate':min(g['rate']for g in groups)})
 # Existing predictions only. Numeric runtime membership reproduces from archived config.
 cases=json.loads((out/f'{model}_test_case_summary.json').read_text(encoding='utf8'));b=cfg['membership_model']['intercept'];ws=list(cfg['membership_model']['coefficients'].values());maxdiff=max(abs(1/(1+math.exp(-(b-sum(a*c for a,c in zip(ws,r['risks'])))))-r['mu'])for r in cases);summary['archived_cfg_vs_test_mu_max_abs_diff']=maxdiff
 auto=[r for r in cases if not r['human']];summary['all_auto_final_unsafe']={'n':len(auto),'unsafe':sum(abs(r['final']-r['y'])>max(.5,.1*r['M'])for r in auto)}
 p=cfg['membership_model']['positive_rate'];summary['constant_brier_reference']=p*(1-p);summary['fitted_in_sample_brier']=cfg['membership_model']['brier_score']
 report[model]=summary;print(model,json.dumps(summary,ensure_ascii=False))
(out/'contract_and_reachability_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
