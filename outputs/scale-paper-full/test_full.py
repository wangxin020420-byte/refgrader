import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'scale-paper-stage1'))
import stage1 as m
import extract_full as full


class Checks(unittest.TestCase):
    def test_online_resume_offline(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d)
            data=root/'data/prepared/ads_public_adapted_v1'
            rows=[dict(sample_id=str(i),question_id=str(i%3),answer='alpha beta',references=['reference']) for i in range(885)]
            raw=m.jsonlines(rows)
            m.write_all(data,{'train.inputs.jsonl':raw,'manifest.json':m.encoded({'outputs_sha256':{'train.inputs.jsonl':m.hashed(raw)}})})
            m.prepare(root)
            folder=root/'paper_method_v1/artifacts/triplets'
            pilot=root/'paper_method_v1/runs/pe_glm_pilot_v1'
            cfg=dict(model='mock',base_url='https://mock.invalid',max_tokens=4096)
            m.write_all(pilot,{'config.json':m.encoded(cfg)})
            def response(text):
                return {'model':'mock','choices':[{'finish_reason':'stop','message':{'content':json.dumps({'triples':[dict(t=text,c='term',w=text)]})}}]}
            for i,u in enumerate(m.read(folder/'pilot_units.jsonl'),1):
                m.write_all(pilot,{'response_%02d.json'%i:m.encoded(response(u['text']))})
            fake_lock=types.SimpleNamespace(flock=lambda *args:None,LOCK_EX=1,LOCK_NB=2)
            def invoke(mode,execute=False):
                argv=['extract_full.py','--root',str(root),'--mode',mode]+(['--execute'] if execute else [])
                with patch.object(sys,'argv',argv):full.main()
            def call(req,timeout):
                text=json.loads(json.loads(req.data)['messages'][1]['content'])['text']
                return io.StringIO(json.dumps(response(text)))
            with patch.dict(sys.modules,{'fcntl':fake_lock}), patch.object(full.urllib.request,'urlopen',side_effect=call) as api, patch.object(full.getpass,'getpass',return_value='mock-key'), patch.object(full.time,'sleep'):
                invoke('offline')
                self.assertEqual(api.call_count,0)
                invoke('online',True)
                self.assertEqual(api.call_count,882)
                invoke('online',True)
                invoke('offline')
                self.assertEqual(api.call_count,882)
            report=json.loads((root/'paper_method_v1/runs/pe_glm_full_v1/summary.json').read_text())
            self.assertEqual(report['status'],'FULL_EXTRACTION_AND_GRAPH_COMPLETE')
            self.assertEqual(report['units_valid'],888)
            self.assertFalse((data/'test.inputs.jsonl').exists())

    def test_invalid_response(self):
        with self.assertRaises(ValueError):
            full.parse_response(m,{'unit_id':'a','text':'a'}, {'choices':[{'finish_reason':'length'}]})


if __name__=='__main__':unittest.main()
