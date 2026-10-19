import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import entity_engine as e
import run_entities as a


def fixture():
    units = [dict(unit_id='student:s', question_id='q', role='student', text='原文'),
             dict(unit_id='reference:r', question_id='q', role='reference', text='参考')]
    ex = [dict(unit_id=u['unit_id'], triples=[]) for u in units]
    data = {}
    for seed in e.SEEDS:
        data[seed] = [dict(id='s', sample_id='s', question_id='q', references=['参考',''],
                           answer='原文',label=4.0,kind='original_training'),
                      dict(id='c',sample_id='s',question_id='q',references=['参考',''],
                           answer='新答案',label=10.01+seed/1000,kind='synthetic_candidate',annotator_seed=seed)]
    return units, ex, data


class Tests(unittest.TestCase):
    def test_exact_contract(self):
        self.assertEqual(e.sha(Path(a.contract.__file__)),a.CONTRACT_SHA)

    def test_dedup_preserves_seed_bindings(self):
        u,x,d=fixture(); b,p,r=e.bind(d,u,x)
        self.assertEqual(len(p),1); self.assertEqual(len(r),2)
        self.assertEqual(b[42][1]['reference_unit_ids'],['reference:r',None])
        self.assertNotEqual(d[42][1]['label'],d[43][1]['label'])

    def test_question_mismatch_rejected(self):
        u,x,d=fixture(); d[42][0]['question_id']='other'
        with self.assertRaises(ValueError):e.bind(d,u,x)

    def test_invalid_span_rejected(self):
        raw={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'triples':[dict(t='错误',c='c',w='w')]})}}]}
        with self.assertRaises(ValueError):e.parse(a.contract,dict(unit_id='u',text='原文'),raw)

    def test_api_export_then_offline_resume(self):
        u,x,d=fixture(); original=copy.deepcopy(d); b,p,r=e.bind(d,u,x)
        cfg=dict(model='glm-5.3',base_url='https://open.bigmodel.cn/api/coding/paas/v4',max_tokens=8192)
        raw={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'triples':[dict(t='新答案',c='c',w='w')]},ensure_ascii=False)}}]}
        class Reply:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return e.enc(raw)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'out'
            with patch.object(e,'prepare',return_value=(a.contract,cfg,d,b,p,r),create=True), patch.dict('os.environ',{'ZHIPUAI_API_KEY':'test-only'}), patch.object(e.urllib.request,'urlopen',return_value=Reply()) as call:
                self.assertEqual(e.execute(root,out,'run',8,900),0)
                self.assertEqual(call.call_count,1)
                request=call.call_args.args[0]
                self.assertEqual(request.full_url,cfg['base_url']+'/chat/completions')
                for seed in e.SEEDS:
                    exported=e.rows(out/('seed_%s/training_with_entities.jsonl'%seed))
                    for row, src in zip(exported,original[seed]):
                        self.assertEqual({k:row[k] for k in src},src)
                    self.assertEqual(exported[1]['reference_triples'],[[],[]])
                call.reset_mock()
                self.assertEqual(e.execute(root,out,'offline',8,900),0)
                call.assert_not_called()
                self.assertEqual(e.read(out/'summary.json')['new_api_calls'],0)
                # Corrupting a cached request fingerprint must prevent reuse.
                cache=next((out/'responses').glob('*/accepted.json'))
                saved=e.read(cache); saved['body_sha256']='changed'
                cache.write_bytes(e.enc(saved))
                with self.assertRaises(ValueError):e.execute(root,out,'offline',8,900)


if __name__=='__main__':unittest.main()
