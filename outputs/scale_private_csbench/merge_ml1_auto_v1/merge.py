"""Freeze automatic ML1 transcription without semantic edits or label reads."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

OLD = 'csbench_scale_online_v1'
NEW = 'csbench_scale_online_clean_v2'
CONFIG = '6a1f4f582cc04919220662984eea868140bce2bcc90f0ad32be3369660c8d647'
COUNTS = {'train':345, 'validation':343, 'test':2723}
TARGETS = {'train':6, 'validation':6, 'test':44}

def sha(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def lines(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def enc(o): return (json.dumps(o,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()

def combine(rows, proposals, split):
    result = copy.deepcopy(rows)
    expected = {r['sample_id'] for r in rows if r['question_id']=='ML_1'}
    supplied = {r['sample_id'] for r in proposals.values() if r['split']==split}
    assert expected == supplied, 'Target identity mismatch'
    for r in result:
        if r['sample_id'] not in expected: continue
        p = proposals[r['sample_id']]
        assert not r['blank_passthrough'] and isinstance(p['text'],str) and p['text'].strip()
        assert not any(x in p['text'] for x in ('评卷员','阅卷员','评分：','评分:'))
        r['answer'] = p['text']
    for a,b in zip(rows,result):
        assert {k:v for k,v in a.items() if k!='answer'} == {k:v for k,v in b.items() if k!='answer'}
        if a['sample_id'] not in expected or a['blank_passthrough']: assert a==b
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args()
    root=args.project.resolve();base=root/'runs/online_experiments';old=base/OLD;new=base/NEW
    rec=new/'input_recovery';auto=new/'ml1_clean_transcription';out=new/'visual'
    rs=read(rec/'summary.json');sm=read(auto/'summary.json');plan=read(auto/'plan.json')
    assert rs['source_experiment']==OLD and rs['proposed_experiment']==NEW
    assert sm['config_sha256']==CONFIG and sm['accepted']==56 and sm['remaining']==0 and sm['errors']==[]
    assert sha(json.dumps(plan,sort_keys=True).encode())==CONFIG
    assert plan['model']=='glm-4.6v' and plan['base_url']=='https://open.bigmodel.cn/api/coding/paas/v4/'
    pp=lines(auto/'proposals.jsonl');proposals={p['sample_id']:p for p in pp}
    assert len(pp)==len(proposals)==56
    reqs={r['sample_id']:r for r in plan['requests']};assert set(reqs)==set(proposals)
    provenance={}
    for sid,p in proposals.items():
        req=reqs[sid];assert p['split']==req['split']
        image=Path(req['clean_path']);assert sha(image.read_bytes())==req['clean_sha256']
        acc=read(auto/'accepted'/f'{sid}.json');assert acc['config_sha256']==CONFIG and acc['sample_id']==sid
        name=acc['response_file'];assert Path(name).name==name
        raw=auto/'responses'/name;assert sha(raw.read_bytes())==acc['response_sha256']
        envelope=read(raw);assert envelope['config_sha256']==CONFIG and envelope['sample_id']==sid
        choice=envelope['response']['choices'][0];assert choice['finish_reason']=='stop'
        obj=json.loads(choice['message']['content']);assert obj==acc['result']=={'text':p['text'],'issues':p['issues']}
        provenance[sid]={'response_sha256':acc['response_sha256'],'clean_image_sha256':req['clean_sha256'],'text_sha256':sha(p['text'].encode())}
    oldsummary=read(old/'visual/experiment_summary.json')
    assert oldsummary['inputs_complete'] and not oldsummary['pending']
    frozen=root/'data/csbench_within_question_v1';fm=read(frozen/'manifest.json');assert fm['counts']==COUNTS
    files={};allids=set();blank_counts={};replaced={};sourcehash={}
    for split,n in COUNTS.items():
        src=old/'visual/text_view'/f'{split}.inputs.jsonl'
        h=sha(src.read_bytes());assert h==rs['source_files_sha256'][str(src)]
        original=lines(src);assert len(original)==n
        fixed=frozen/f'{split}.inputs.jsonl';assert sha(fixed.read_bytes())==fm['files_sha256'][fixed.name]
        ff=lines(fixed);assert [r['sample_id'] for r in ff]==[r['sample_id'] for r in original]
        for a,b in zip(original,ff):
            assert a['question_id']==b['question_id'] and a['blank_passthrough']==b['blank_passthrough']
            if b['blank_passthrough']: assert a['answer']==b['answer']
        merged=combine(original,proposals,split)
        ids={r['sample_id'] for r in merged};assert len(ids)==n and not allids.intersection(ids);allids.update(ids)
        replaced[split]=sum(r['question_id']=='ML_1' for r in merged);assert replaced[split]==TARGETS[split]
        blank_counts[split]=sum(not r['answer'].strip() for r in merged)
        sourcehash[str(src)]=h
        files['text_view/'+split+'.inputs.jsonl']=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in merged).encode()
    assert len(allids)==3411 and blank_counts=={'train':5,'validation':8,'test':53}
    issues=[x for x in oldsummary.get('reported_issues',[]) if x['unit_id'] not in {'answer:'+s for s in proposals}]
    issues += [{'unit_id':'answer:'+p['sample_id'],'issues':p['issues']} for p in pp if p['issues']]
    report=dict(experiment_id=NEW,status='AUTOMATIC_INPUTS_FROZEN_EXPERIMENTAL',source_experiment=OLD,
        records=COUNTS,replaced_records=replaced,blank_counts=blank_counts,inputs_complete=True,pending=[],
        human_review_performed=False,semantic_accuracy_verified=False,transcription_policy='automatic_output_as_is',
        reported_issues=issues,missing_visual_context=oldsummary.get('missing_visual_context',[]),
        automatic_quality_warnings_block_execution=False,known_ml1_markings_isolated=True,
        all_questions_leakage_audit_complete=False,ready_for_stage1=True,ready_for_training=False,
        api_calls=0,labels_read=False,training_started=False,source_files_modified=False,
        reproduction_status='METHOD_RECONSTRUCTION_WITH_DOCUMENTED_DEVIATIONS',
        reuse_policy='Derived repair of v1. Preserve other visual outputs; no cross-run SCALE cache copying.',
        source_files_sha256=sourcehash,transcription_config_sha256=CONFIG,
        files_sha256={Path(k).name:sha(v) for k,v in files.items()})
    files['ml1_merge_audit.json']=enc({'provenance':provenance,'proposals_sha256':sha((auto/'proposals.jsonl').read_bytes()),'policy':'No text edits; no issue removal; no label files read'})
    files['experiment_summary.json']=enc(report)
    # Verify all conflicts before creating outputs. The summary is written last.
    for name,b in files.items():
        path=out/name
        if path.exists(): assert path.read_bytes()==b, 'Existing frozen artifact differs: '+str(path)
    for name,b in files.items():
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            with path.open('xb') as f:f.write(b)
    for p,h in sourcehash.items():assert sha(Path(p).read_bytes())==h
    print(json.dumps({k:report[k] for k in ('status','experiment_id','records','replaced_records','blank_counts','api_calls','semantic_accuracy_verified','ready_for_stage1')},ensure_ascii=False,indent=2))
    print('SUMMARY:',out/'experiment_summary.json')

if __name__=='__main__':main()
