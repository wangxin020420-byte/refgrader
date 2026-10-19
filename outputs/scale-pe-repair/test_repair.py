import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import repair
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'scale-paper-stage1'))
import stage1 as m


def response(t,finish='stop'):
    return {'choices':[{'finish_reason':finish,'message':{'content':json.dumps({'triples':[{'t':t,'c':'term','w':'summary'}]})}}]}


class Checks(unittest.TestCase):
    def test_whitespace_only(self):
        self.assertEqual(repair.restore_whitespace('a\n b','ab'),'a\n b')
        self.assertIsNone(repair.restore_whitespace('a b and a b','ab'))
        self.assertIsNone(repair.restore_whitespace('O(n）','O(n)'))
        self.assertIsNone(repair.restore_whitespace('出队进栈','出栈进栈'))

    def test_recovery_and_resume(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);run=root/'paper_method_v1/runs/pe_glm_full_v1'
            units=[{'unit_id':str(i),'text':'alpha'} for i in range(4)]
            source=m.jsonlines(units)
            m.write_all(root/'paper_method_v1/artifacts/triplets',{'source_units.jsonl':source})
            cfg={'source_sha256':m.hashed(source),'prompt_sha256':m.hashed(m.PROMPT.encode('utf8')),
                 'stage1_sha256':m.hashed(Path(m.__file__).read_bytes()),'model':'mock','base_url':'https://mock.invalid','max_tokens':4096}
            m.write_all(run,{'config.json':m.encoded(cfg)})
            originals=[]
            for i in range(4):
                raw=response('alpha' if i<2 else 'invented','length' if i==2 else 'stop')
                if i==1:
                    obj=json.loads(raw['choices'][0]['message']['content']);obj['triples'][0]['']=''
                    raw['choices'][0]['message']['content']=json.dumps(obj)
                content=m.encoded({'unit_id':str(i),'response':raw})
                originals.append(content)
                m.write_all(run/'cache',{m.hashed(str(i).encode())+'.json':content})
            max_tokens=[]
            def call(req,timeout):
                max_tokens.append(json.loads(req.data)['max_tokens'])
                return io.StringIO(json.dumps(response('alpha')))
            fake=types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1,LOCK_NB=2)
            sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'scale-paper-full'))
            with patch.dict(sys.modules,{'fcntl':fake}),patch.object(sys,'argv',['repair.py','--root',str(root),'--execute']),patch.object(repair.getpass,'getpass',return_value='SECRET_TEST_KEY'),patch.object(repair.time,'sleep'),patch.object(repair.urllib.request,'urlopen',side_effect=call) as api:
                repair.main();repair.main()
                self.assertEqual(api.call_count,2)
            self.assertEqual(max_tokens,[16384,4096])
            self.assertEqual((run/'cache'/(m.hashed(b'0')+'.json')).read_bytes(),originals[0])
            for i in [1,2,3]:
                self.assertEqual((run/'recovery_v1'/m.hashed(str(i).encode())/'original_cache.json').read_bytes(),originals[i])
            self.assertFalse(any('SECRET_TEST_KEY' in p.read_text(encoding='utf8') for p in run.rglob('*.json')))


if __name__=='__main__':unittest.main()
