import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import resume

path=Path(__file__).resolve().parents[1]/'scale_csbench_stage1_v1/run_stage1.py'
spec=importlib.util.spec_from_file_location('runner',path);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
c=r.contract()

class Tests(unittest.TestCase):
    def fixture(self):
        old=[dict(unit_id=uid,text='old',role='student') for uid in sorted(resume.CHANGED)]
        old += [dict(unit_id=f'other:{i}',text='' if i<5 else 'same',role='student') for i in range(384)]
        new=copy.deepcopy(old)
        for u in new:
            if u['unit_id'] in resume.CHANGED:u['text']='new'
        cfg={k:k for k in resume.KEYS};cfg.update(graph_mode='prose',experiment_id=resume.OLD)
        fresh=dict(cfg,experiment_id=resume.NEW)
        return old,new,cfg,fresh
    def test_contract_mismatch(self):
        old,new,a,b=self.fixture();b['max_tokens']='different'
        with self.assertRaises(ValueError):resume.eligible(a,b,old,new)
    def test_unexpected_source_change(self):
        old,new,a,b=self.fixture();new[-1]['text']='other'
        with self.assertRaises(AssertionError):resume.eligible(a,b,old,new)
    def test_reuse_and_six_calls_and_resume(self):
        oldunits,units,oldcfg,cfg=self.fixture()
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);old=root/'old';out=root/'new';old.mkdir();out.mkdir()
            extract=[dict(unit_id=u['unit_id'],triples=[] if not u['text'] else [dict(t=u['text'],c='term',w='summary')]) for u in oldunits]
            (old/'source_units.jsonl').write_bytes(c.jsonlines(oldunits));(old/'extractions.jsonl').write_bytes(c.jsonlines(extract));(old/'training_sources.json').write_bytes(b'[]')
            r.save(old/'config.json',oldcfg)
            r.save(old/'summary.json',dict(status='CSBENCH_TRAIN_EXTRACTION_AND_GRAPH_COMPLETE',remaining=0,config_sha256=r.sha(r.enc(oldcfg)),files_sha256={n:r.sha((old/n).read_bytes()) for n in ['source_units.jsonl','extractions.jsonl','training_sources.json']}))
            baseline={p.name:p.read_bytes() for p in old.iterdir()}
            resume.install(r,c,old,out,units,cfg)
            self.assertEqual(len(list(out.glob('responses/*/accepted.json'))),379)
            (out/'source_units.jsonl').write_bytes(c.jsonlines(units));(out/'training_sources.json').write_bytes(b'[]')
            calls=[]
            def create(**kwargs):
                calls.append(kwargs)
                response={'choices':[{'finish_reason':'stop','message':{'content':'{"triples":[{"t":"new","c":"term","w":"summary"}]}'}}]}
                return SimpleNamespace(model_dump=lambda **kw:response)
            client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            with patch.object(r,'prepare',return_value=(c,out,units,cfg)):
                self.assertEqual(r.execute(root,resume.NEW,client=client,sleep=lambda _:None),0)
                self.assertEqual(len(calls),6)
                resume.install(r,c,old,out,units,cfg)
                self.assertEqual(r.execute(root,resume.NEW,client=client,sleep=lambda _:None),0)
                self.assertEqual(len(calls),6)
            self.assertEqual(baseline,{p.name:p.read_bytes() for p in old.iterdir()})
            self.assertEqual(r.read(out/'summary.json')['units_valid'],390)

if __name__=='__main__':unittest.main()
