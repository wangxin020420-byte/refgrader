import json
from pathlib import Path
for path in [Path('results_runs/csbench_q45_12039bfe25ca4168_full/runs/csbench45_full_experimental_glm52_20260825_163432/run_state.json'),Path('results_runs/csbench_q45_12039bfe25ca4168_validation/runs/csbench45_validation_glm52_20260822_090920/run_state.json'),Path('results_runs/a3wa_configs/private45_validation_deepseek_official_flash_t0_20260927_200831.json')]:
 print('\nFILE',path)
 d=json.loads(path.read_text(encoding='utf-8-sig'));print('KEYS',list(d))
 for k,v in d.items():
  if any(s in k for s in ['config','calib','a3wa','route','gate','metric','selection','loss','threshold','boundary','cross_validation','diagnostic','membership','operational']) and k!='files':
   s=json.dumps(v,ensure_ascii=False);print(k,s[:6000])
