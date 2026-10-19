import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import merge

class Tests(unittest.TestCase):
    def test_verbatim_and_blank(self):
        rows=[dict(sample_id='a',question_id='ML_1',answer='old',blank_passthrough=False),dict(sample_id='b',question_id='Q',answer='',blank_passthrough=True)]
        p={'a':dict(sample_id='a',split='train',text='O ( ♣1',issues=['unclear'])}
        r=merge.combine(rows,p,'train')
        self.assertEqual(r[0]['answer'],p['a']['text']);self.assertEqual(r[1],rows[1]);self.assertEqual(rows[0]['answer'],'old')
    def test_marking_rejected(self):
        with self.assertRaises(AssertionError):
            merge.combine([dict(sample_id='a',question_id='ML_1',answer='x',blank_passthrough=False)],{'a':dict(sample_id='a',split='train',text='评分：2')},'train')
    def test_full_freeze_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'runs/online_experiments';old=base/merge.OLD;new=base/merge.NEW;auto=new/'ml1_clean_transcription';frozen=root/'data/csbench_within_question_v1'
            def write(p,o):
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(merge.enc(o))
            def jl(p,rr):
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r)+'\n' for r in rr),encoding='utf-8')
            requests=[];proposals=[];sources={};fm={'counts':merge.COUNTS,'files_sha256':{}}
            for split,n in merge.COUNTS.items():
                rr=[]
                for i in range(n):
                    sid=split+str(i);target=i<merge.TARGETS[split];blank=i>=n-{'train':5,'validation':8,'test':53}[split]
                    rr.append(dict(sample_id=sid,question_id='ML_1' if target else 'Q',answer='' if blank else 'old',blank_passthrough=blank))
                    if target:
                        im=auto/'images'/f'{sid}.png';im.parent.mkdir(parents=True,exist_ok=True);im.write_bytes(b'fixture')
                        requests.append(dict(sample_id=sid,split=split,clean_path=str(im),clean_sha256=merge.sha(im.read_bytes())))
                        proposals.append(dict(sample_id=sid,split=split,text='O ( 1',issues=['unclear']))
                src=old/'visual/text_view'/f'{split}.inputs.jsonl';jl(src,rr);sources[str(src)]=merge.sha(src.read_bytes())
                fp=frozen/src.name;jl(fp,rr);fm['files_sha256'][fp.name]=merge.sha(fp.read_bytes())
            write(frozen/'manifest.json',fm)
            write(new/'input_recovery/summary.json',dict(source_experiment=merge.OLD,proposed_experiment=merge.NEW,source_files_sha256=sources))
            write(old/'visual/experiment_summary.json',dict(inputs_complete=True,pending=[],reported_issues=[],missing_visual_context=[]))
            plan=dict(model='glm-4.6v',base_url='https://open.bigmodel.cn/api/coding/paas/v4/',requests=requests)
            config=merge.sha(json.dumps(plan,sort_keys=True).encode());write(auto/'plan.json',plan)
            write(auto/'summary.json',dict(config_sha256=config,accepted=56,remaining=0,errors=[]));jl(auto/'proposals.jsonl',proposals)
            for p in proposals:
                sid=p['sample_id'];obj={'text':p['text'],'issues':p['issues']};name=sid+'.json';raw=auto/'responses'/name
                write(raw,dict(config_sha256=config,sample_id=sid,response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(obj)}}]}))
                write(auto/'accepted'/name,dict(config_sha256=config,sample_id=sid,response_file=name,response_sha256=merge.sha(raw.read_bytes()),result=obj))
            with patch.object(merge,'CONFIG',config),patch.object(sys,'argv',['merge.py','--project',str(root)]),contextlib.redirect_stdout(io.StringIO()):
                merge.main();output=new/'visual/experiment_summary.json';before=output.read_bytes();merge.main();self.assertEqual(before,output.read_bytes())
                proposals[0]['text']='tampered';jl(auto/'proposals.jsonl',proposals)
                with self.assertRaises(AssertionError):merge.main()

if __name__=='__main__':unittest.main()
