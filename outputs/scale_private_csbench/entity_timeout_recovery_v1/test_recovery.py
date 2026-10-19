import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import recover

ORIGINAL = Path(__file__).resolve().parent.parent/'train_entities_clean_v1'
import sys
sys.path.insert(0,str(ORIGINAL))
import entity_engine as e
import run_entities as adapter
from test_entities import fixture


class LegacyTimeout(OSError):
    pass


class Tests(unittest.TestCase):
    def test_connect_timeout(self):
        def fail(*args, **kwargs):raise LegacyTimeout('timeout')
        with patch.object(recover.socket,'timeout',LegacyTimeout):
            with self.assertRaises(TimeoutError):recover.adapt_urlopen(fail)('url')

    def test_read_timeout_retries_and_resumes(self):
        u,x,d=fixture(); b,p,r=e.bind(d,u,x)
        cfg=dict(model='glm-5.3',base_url='https://open.bigmodel.cn/api/coding/paas/v4',max_tokens=8192)
        raw={'choices':[{'finish_reason':'stop','message':{'content':'{"triples":[]}'}}]}
        class Reply:
            def __init__(self, fail=False):self.fail=fail
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):
                if self.fail:raise LegacyTimeout('read timeout')
                return e.enc(raw)
        sequence=iter([Reply(True),Reply()])
        def fake(*args,**kwargs):return next(sequence)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out'
            with patch.object(recover.socket,'timeout',LegacyTimeout), patch.object(e,'prepare',return_value=(adapter.contract,cfg,d,b,p,r),create=True), patch.dict('os.environ',{'ZHIPUAI_API_KEY':'test'}), patch.object(e.time,'sleep'), patch.object(e.urllib.request,'urlopen',side_effect=recover.adapt_urlopen(fake)) as network:
                self.assertEqual(e.execute(Path(tmp),out,'run',8,900),0)
                self.assertEqual(network.call_count,2)
                folder=next((out/'responses').iterdir())
                self.assertEqual(e.read(folder/'attempt_001.error.json')['exception'],'TimeoutError')
                self.assertTrue((folder/'attempt_002.response.json').is_file())
                network.reset_mock()
                self.assertEqual(e.execute(Path(tmp),out,'run',8,900),0)
                network.assert_not_called()

    def test_unrelated_error_unchanged(self):
        def fail(*args,**kwargs):raise ValueError('parameter')
        with self.assertRaises(ValueError):recover.adapt_urlopen(fail)('url')


if __name__=='__main__':unittest.main()
