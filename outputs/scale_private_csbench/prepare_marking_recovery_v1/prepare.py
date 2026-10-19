"""Create reversible metadata-removal proposals and ML_1 image review bundle."""
import argparse,copy,hashlib,json,re,zipfile
from pathlib import Path

OLD='csbench_scale_online_v1'
NEW='csbench_scale_online_clean_v2'
PAT=re.compile(r'评卷员|阅卷员|(?:评分|得分)\s*[:：]\s*[-+]?\d+(?:\.\d+)?')
META=re.compile(r'^[ \t]*评卷员[：:][ \t]*[^\r\n：:]{1,50}?[ \t]*[，,]?[ \t]*评分[：:][ \t]*\d+(?:\.\d+)?[ \t]*[。.]?[ \t]*$')
def sha(b):return hashlib.sha256(b).hexdigest()
def enc(o):return (json.dumps(o,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf8')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
def freeze(p,b):
    if p.exists():
        if p.read_bytes()!=b:raise ValueError('Existing artifact differs: '+str(p))
    else:
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(b)
def clean(text):
    kept=[];removed=[]
    for line in text.splitlines(keepends=True):
        if META.fullmatch(line.rstrip('\r\n')):removed.append(line)
        else:kept.append(line)
    return ''.join(kept),removed

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path('/data0/E125221219/projects/scale-private'));a=p.parse_args()
    project=a.project.resolve();old=project/'runs/online_experiments'/OLD
    out=project/'runs/online_experiments'/NEW/'input_recovery'
    here=Path(__file__).resolve().parent;expected=read(here/'affected_ids.json');lineage=read(here/'lineage_summary.json')
    hashes={r['artifact']:r['sha256'] for r in lineage['artifacts']}
    files={};changes=[];unresolved=[];images=[];counts={};seen=set();original_hashes={}
    for split,n in [('train',345),('validation',343),('test',2723)]:
        source=old/'visual/text_view'/f'{split}.inputs.jsonl'
        data=source.read_bytes();assert sha(data)==hashes['visual_'+split], 'Input changed: '+str(source)
        original_hashes[str(source)]=sha(data);rr=rows(source);assert len(rr)==n
        changed=0;matched=[];cl=[]
        for r in rr:
            sid=r['sample_id'];assert sid not in seen;seen.add(sid)
            if PAT.search(r['answer']):matched.append(sid)
            c=copy.deepcopy(r);text,removed=clean(r['answer'])
            if removed:
                assert sid in expected[split] and text.strip(), 'Unexpected change: '+sid
                assert not r.get('blank_passthrough',False)
                c['answer']=text;changed+=1
                changes.append(dict(split=split,sample_id=sid,removed_lines=removed,original_answer_sha256=sha(r['answer'].encode()),proposed_answer_sha256=sha(text.encode())))
            if PAT.search(c['answer']):unresolved.append(dict(split=split,sample_id=sid,reason='Remaining marking needs image review; text preserved'))
            assert {k:v for k,v in c.items() if k!='answer'}=={k:v for k,v in r.items() if k!='answer'}
            cl.append(c)
            if r['question_id']=='ML_1':
                imagepath=r.get('student_image')
                image=Path(imagepath) if isinstance(imagepath,str) and imagepath else None
                if image is not None and not image.is_absolute():image=project.parent/'refgrader'/image
                ok=image is not None and image.is_file()
                item=dict(split=split,sample_id=sid,question_id='ML_1',path=str(image) if image else None,image_available=ok,
                    explicit_marking_in_text=sid in expected[split],original_answer=r['answer'],question=r['question'])
                if ok:item.update(image_sha256=sha(image.read_bytes()),archive_name='images/'+sid+image.suffix.lower())
                images.append(item)
        assert set(matched)==set(expected[split]), 'Marking IDs differ from audited IDs'
        counts[split]=dict(records=len(rr),changed_records=changed,blank_records=sum(not r['answer'].strip() for r in rr))
        files['text_view_proposal/'+split+'.inputs.jsonl']=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in cl).encode('utf8')
    assert len(seen)==3411
    report=dict(status='METADATA_REMOVAL_PROPOSALS_READY_IMAGE_REVIEW_REQUIRED',source_experiment=OLD,proposed_experiment=NEW,
        records=3411,counts=counts,changed_records=len(changes),unresolved=unresolved,ml1_image_records=len(images),
        missing_images=[x['sample_id'] for x in images if not x['image_available']],
        source_files_sha256=original_hashes,output_files_sha256={k:sha(v) for k,v in files.items()},
        source_files_modified=False,api_calls=0,label_files_read=False,validation_test_inputs_read=True,
        images_masked=False,visual_model_exposure_resolved=False,ready_for_training=False,
        reuse_policy='Do not reuse affected visual outputs. Re-extract changed units, rebuild shared graph/retrieval/replacement/selection. Reuse generation only after exact request/config match; no input-label mixing.',
        limitation='Text deletion does not undo model exposure to visible teacher markings. Proposal only, not an approved training input.')
    files.update({'summary.json':enc(report),'removed_metadata_audit.json':enc(changes),'image_review_manifest.json':enc(images)})
    for name,b in files.items():freeze(out/name,b)
    for path,h in original_hashes.items():assert sha(Path(path).read_bytes())==h
    bundle=out/'ml1_image_review.zip'
    if not bundle.exists():
        with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
            for name in ('summary.json','removed_metadata_audit.json','image_review_manifest.json'):z.write(out/name,name)
            for item in images:
                if item['image_available']:
                    b=Path(item['path']).read_bytes();assert sha(b)==item['image_sha256'];z.writestr(item['archive_name'],b)
    else:
        with zipfile.ZipFile(bundle) as z:
            assert z.read('summary.json')==files['summary.json']
            for item in images:
                if item['image_available']:assert sha(z.read(item['archive_name']))==item['image_sha256']
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('REVIEW_BUNDLE:',bundle)
    print('Existing experiment untouched. No API calls or training.')
if __name__=='__main__':main()
