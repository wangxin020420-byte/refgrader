import errno
import json
from pathlib import Path
import tempfile
import contextlib
import io
import shutil
import sys
import types
import unittest
from unittest.mock import patch
import proxy as p
import proxy_disk_guard as g
import test_proxy as fixtures


class Checks(unittest.TestCase):
    def test_guarded_resume_pipeline(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);helper=fixtures.Checks();helper.fixture(root)
            runner=root/'control/scale-paper-proxy/proxy.py';runner.parent.mkdir(parents=True)
            shutil.copyfile(p.__file__,runner)
            locks=types.SimpleNamespace(LOCK_EX=2,LOCK_NB=4,flock=lambda *a:None)
            with patch.dict('sys.modules',{'fcntl':locks}),patch.object(g,'check_space',return_value=1024*g.MIB),patch.object(p.getpass,'getpass',return_value='secret'),patch.object(p.time,'sleep'),patch.object(p.urllib.request,'urlopen',side_effect=helper.response) as network:
                p.prepare(root)
                run=root/'paper_method_v1/runs/proxy_glm_v1';run.mkdir(parents=True)
                with patch.object(sys,'argv',['proxy_disk_guard.py','run','--root',str(root),'--retry-failed']):g.main()
                self.assertEqual(network.call_count,3)
                self.assertEqual(p.read(run/'summary.json')['status'],'PROXY_GENERATION_COMPLETE_REVIEW_REQUIRED')
                self.assertFalse((run/'disk_guard_reserve.bin').exists())

    def test_recover_complete_archive_partial_preserve_existing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'responses').mkdir()
            raw=p.enc({'choices':[{'finish_reason':'stop','message':{'content':'{"proxy_answer":"x"}'}}]})
            (root/'responses/a.json.tmp').write_bytes(raw)
            (root/'responses/b.json.tmp').write_bytes(b'{"choices":')
            (root/'responses/c.json').write_bytes(raw)
            (root/'responses/c.json.tmp').write_bytes(raw)
            result=g.recover_temps(root,[{'request_id':x} for x in 'abc'],p.parse)
            self.assertEqual((root/'responses/a.json').read_bytes(),raw)
            self.assertEqual((root/'responses/c.json').read_bytes(),raw)
            self.assertFalse((root/'responses/b.json').exists())
            self.assertEqual(Path(result[1]['archive_path']).read_bytes(),b'{"choices":')

    def test_release_reserve_retry_save_and_stop_network(self):
        with tempfile.TemporaryDirectory() as d:
            guard=g.Guard(Path(d))
            with patch.object(g,'check_space',return_value=1024*g.MIB):guard.allocate()
            self.assertEqual(guard.reserve.stat().st_size,32*g.MIB)
            calls=[]
            def save(path,data,mutable=False):
                calls.append(1)
                if len(calls)==1:raise OSError(errno.ENOSPC,'full')
                path.write_bytes(data)
            target=Path(d)/'response.json'
            guard.save(save,target,b'valid received response')
            self.assertEqual(target.read_bytes(),b'valid received response')
            self.assertFalse(guard.reserve.exists())
            self.assertTrue(guard.emergency)
            with self.assertRaisesRegex(RuntimeError,'no further API'):
                guard.urlopen(lambda: self.fail('network should not run'))
            self.assertEqual(guard.network_calls,0)

    def test_low_space_and_nonspace_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as d:
            guard=g.Guard(Path(d))
            with patch.object(g,'check_space',side_effect=RuntimeError('low space')):
                with self.assertRaises(RuntimeError):guard.urlopen(lambda: self.fail('network'))
            def bad(*a,**kw):raise OSError(errno.EACCES,'denied')
            with self.assertRaises(OSError):guard.save(bad,Path(d)/'x',b'a')
            self.assertFalse(guard.emergency)


if __name__=='__main__':unittest.main()
