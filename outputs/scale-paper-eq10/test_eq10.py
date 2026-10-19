import unittest
import numpy as np
import eq10
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
from unittest.mock import patch


class Checks(unittest.TestCase):
    def test_cosine_and_stable_ties(self):
        x=eq10.normalized(np.array([[2.,0.],[1.,0.],[0.,2.],[-1.,0.]],dtype='float32'))
        order,scores=eq10.ranking(x)
        self.assertEqual(order[0].tolist(),[0,1,2,3])
        np.testing.assert_allclose(scores[0],[1,1,0,-1],atol=1e-6)

    def test_invalid_vectors(self):
        for x in [[[0.,0.]],[[float('nan'),1.]]]:
            with self.assertRaises(ValueError):eq10.normalized(np.array(x))

    def test_cached_offline_build_and_resume(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);base=root/'paper_method_v1'
            model=root/'models/text2vec-base-chinese'/eq10.REV;model.mkdir(parents=True)
            for name in eq10.FILES:(model/name).write_text('test fixture')
            eq10.dump(model/'offline_manifest.json',{'repo_id':eq10.REPO,'revision':eq10.REV,'files_sha256':{n:eq10.digest(model/n) for n in eq10.FILES}})
            graph=base/'artifacts/full_graph_v1/graph.json'
            nodes=[{'id':'w:a','kind':'w','text':'a'},{'id':'w:b','kind':'w','text':'b'}]
            eq10.dump(graph,{'nodes':nodes})
            previous=base/'artifacts/retrieval_v1'
            eq10.dump(previous/'index.json',{'summaries_by_type':{'c:test':['w:a','w:b']}})
            eq10.dump(previous/'summary.json',{'output_sha256':{'index.json':eq10.digest(previous/'index.json')},'graph_sha256':eq10.digest(graph)})
            out=base/'artifacts/eq10_v1';chunk=out/'chunks/000000.npy';chunk.parent.mkdir(parents=True)
            np.save(chunk,np.array([[1.,0.],[0.,1.]],dtype='float32'),allow_pickle=False)
            eq10.dump(chunk.with_suffix('.json'),{'ids':['w:a','w:b'],'sha256':eq10.digest(chunk)})
            def forbidden(*a,**k):raise AssertionError('Cached run must not load a model or request network')
            fake_torch=types.SimpleNamespace(__version__='mock',set_num_threads=lambda n:None,manual_seed=lambda n:None)
            fake_transformers=types.SimpleNamespace(__version__='mock',AutoTokenizer=types.SimpleNamespace(from_pretrained=forbidden),AutoModel=types.SimpleNamespace(from_pretrained=forbidden))
            fake_fcntl=types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1,LOCK_NB=2)
            with patch.dict('sys.modules',{'torch':fake_torch,'transformers':fake_transformers,'fcntl':fake_fcntl}),patch.dict('os.environ',{}):
                eq10.execute(root);before=(out/'summary.json').read_bytes();eq10.execute(root)
            self.assertEqual(before,(out/'summary.json').read_bytes())
            self.assertEqual(json.loads(before)['status'],'PAPER_EQ10_RANKING_COMPLETE')


if __name__=='__main__':unittest.main()
