import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import recover_length as r

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'train_entities_clean_v1'))
import entity_engine as e
import run_entities as a
from test_entities import fixture


class Tests(unittest.TestCase):
    def test_length_then_success_export_and_resume(self):
        units,ex,datasets=fixture()
        for seed in e.SEEDS:
            other=dict(datasets[seed][1],id='other',answer='另一个答案')
            datasets[seed].append(other)
        originals=copy.deepcopy(datasets)
        bindings,pending,reused=e.bind(datasets,units,ex)
        target=next(u for u in pending if u['text']=='新答案')
        other=next(u for u in pending if u['text']=='另一个答案')
        cfg=dict(model='glm-5.3',base_url='https://open.bigmodel.cn/api/coding/paas/v4',max_tokens=8192)
        good={'choices':[{'finish_reason':'stop','message':{'content':'{"triples":[]}'}}]}
        truncated={'choices':[{'finish_reason':'length','message':{'content':''}}]}
        class Reply:
            def __init__(self,obj):self.obj=obj
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return e.enc(self.obj)
        with tempfile.TemporaryDirectory() as tmp, patch.object(r,'UID',target['unit_id']), patch.dict('os.environ',{'ZHIPUAI_API_KEY':'test-only'}):
            root=Path(tmp); out=root/'out'; folder=out/'length_recovery_v1'
            cache=out/'responses'/other['unit_id'].split(':',1)[1]/'accepted.json'
            e.write(cache,e.enc(dict(body_sha256=e.digest(e.enc(r.body_for(a.contract,cfg,other,8192))),response=good)))
            original_cache=cache.read_bytes()
            with patch.object(r.urllib.request,'urlopen',side_effect=[Reply(truncated),Reply(good)]) as call:
                recovered,calls,accepted=r.supplement(e,a.contract,cfg,target,folder)
                self.assertEqual(calls,2)
                bodies=[__import__('json').loads(item.args[0].data) for item in call.call_args_list]
                self.assertEqual([b['max_tokens'] for b in bodies],[16384,32768])
                self.assertEqual(bodies[0]['messages'],bodies[1]['messages'])
            prepared=(a.contract,cfg,datasets,bindings,pending,reused)
            with patch.object(r.urllib.request,'urlopen',side_effect=AssertionError('must remain offline')):
                self.assertEqual(r.export(e,prepared,root,out,recovered,calls,accepted),0)
                _,resumed_calls,_=r.supplement(e,a.contract,cfg,target,folder)
                self.assertEqual(resumed_calls,0)
            self.assertEqual(cache.read_bytes(),original_cache)
            report=e.read(out/'summary.json')
            self.assertEqual(report['unique_synthetic_requests'],2)
            self.assertEqual(report['valid_synthetic_units'],2)
            self.assertEqual(report['remaining'],0)
            for seed in e.SEEDS:
                result=e.rows(out/('seed_%d/training_with_entities.jsonl'%seed))
                for row,source in zip(result,originals[seed]):
                    self.assertEqual({k:row[k] for k in source},source)

    def test_prompt_and_frozen_files(self):
        base=Path(a.__file__).parent
        for name,expected in r.EXPECTED.items():self.assertEqual(e.sha(base/name),expected)


if __name__=='__main__':unittest.main()
