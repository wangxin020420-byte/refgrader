"""Read-only, label-file-blind scan of named CSBench experiment artifacts."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

PATTERN = re.compile(r'评卷员|阅卷员|(?:评分|得分)\s*[:：]\s*[-+]?\d+(?:\.\d+)?')
KNOWN = {'ANS_ML_09', 'ANS_ML_25', 'ANS_ML_28', 'ANS_ML_31', 'ANS_ML_37', 'ANS_ML_43'}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def scan_record(row, fields):
    hits=[]
    def visit(value, field):
        if isinstance(value,str):
            for m in PATTERN.finditer(value):
                hits.append(dict(field=field,match=m.group(),context=value[max(0,m.start()-35):m.end()+70]))
        elif isinstance(value,list):
            for i,v in enumerate(value):visit(v,f'{field}[{i}]')
        elif isinstance(value,dict):
            for k,v in value.items():visit(v,f'{field}.{k}')
    for f in fields:
        if f in row:visit(row[f],f)
    return hits

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path('/data0/E125221219/projects/scale-private'))
    a=p.parse_args();exp=a.project/'runs/online_experiments/csbench_scale_online_v1'
    targets=[]
    for split in ('train','validation','test'):
        targets.append((f'visual_{split}',exp/f'visual/text_view/{split}.inputs.jsonl',['answer','question','references']))
    targets += [
      ('stage1_sources',exp/'scale_stage1/training_sources.json',['a','q','r']),
      ('stage1_units',exp/'scale_stage1/source_units.jsonl',['text']),
      ('stage1_extractions',exp/'scale_stage1/extractions.jsonl',['triples']),
      ('stage3_requests',exp/'scale_stage3/requests.jsonl',['messages']),
      ('stage3_candidates',exp/'scale_stage3/candidates.jsonl',['answer']),
      ('stage4_selected',exp/'scale_stage4/layout/paper_method_v2/artifacts/initial_filter_k5_b5000_v1/selected_candidates.jsonl',['answer']),
      ('stage5_proxies',exp/'scale_stage5/proxy_records.jsonl',['original_answer','proxy_answer']),
    ]
    out=a.project/'reports'/('marking_lineage_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    summaries=[];details=[];trace=[];missing=[]
    for name,path,fields in targets:
        if not path.is_file():missing.append(str(path));continue
        before=sha(path)
        if path.suffix=='.jsonl':
            rows=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
        else:rows=json.loads(path.read_text(encoding='utf8'))
        if not isinstance(rows,list):raise ValueError(f'Expected record list: {path}')
        count=0;known_count=0;fields_seen=Counter()
        for i,row in enumerate(rows):
            if not isinstance(row,dict):raise ValueError(f'Invalid row in {path}')
            for f in fields:
                if f in row:fields_seen[f]+=1
            ident={k:row[k] for k in ('sample_id','question_id','unit_id','request_id','variant_id') if k in row}
            ids=[str(row.get('sample_id','')),str(row.get('unit_id','')).removeprefix('student:')]
            is_known=any(x in KNOWN for x in ids)
            hits=scan_record(row,fields)
            if hits:
                count+=1;details.append(dict(artifact=name,index=i,identity=ident,hits=hits))
            if is_known:
                known_count+=1;trace.append(dict(artifact=name,index=i,identity=ident,contains_explicit_marking=bool(hits)))
        assert sha(path)==before, f'Input changed during audit: {path}'
        summaries.append(dict(artifact=name,path=str(path),sha256=before,records=len(rows),records_with_explicit_marking=count,
            records_from_known_sources=known_count,scanned_fields_present=dict(fields_seen)))
    report=dict(status='MARKING_LINEAGE_AUDIT_COMPLETE' if not missing else 'MARKING_LINEAGE_AUDIT_INCOMPLETE',
        experiment_id='csbench_scale_online_v1',known_training_source_ids=sorted(KNOWN),artifacts=summaries,missing_files=missing,
        api_calls=0,source_files_modified=False,label_files_read=False,validation_test_answer_texts_scanned=True,
        scope='Explicit marking text only. Negative matches do not rule out bare handwritten scores or image-level leakage. No model evaluation or training.',
        ready_for_annotator_training=False)
    for n,obj in [('summary.json',report),('marking_findings.json',details),('known_source_lineage.json',trace)]:
        (out/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    import zipfile
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.glob('*.json')):z.write(f,f.name)
    for s in summaries:print(s['artifact'],'records=',s['records'],'explicit_marking=',s['records_with_explicit_marking'],'known_source=',s['records_from_known_sources'])
    print('Missing files:',len(missing))
    print('REPORT_ZIP:',archive)
    print('No API calls; source files unchanged; label files unopened.')
    return 0 if not missing else 2
if __name__=='__main__':raise SystemExit(main())
