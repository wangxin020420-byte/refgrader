import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import filter_prepare as f


class Checks(unittest.TestCase):
    def fixture(self, root):
        base=root/'paper_method_v1/artifacts'
        retrieval=base/'retrieval_v1';gen=base/'ps_generated_v1';plan=base/'ps_inputs_v1'
        f.save(retrieval/'training_quadruples.json',[{'sample_id':'s1','a':'answer'},{'sample_id':'s2','a':''}])
        f.save(retrieval/'summary.json',{'output_sha256':{'training_quadruples.json':f.sha(retrieval/'training_quadruples.json')}})
        plan.mkdir()
        request={'request_id':'r1','sample_id':'s1','provenance':[{'variant_id':'v1'},{'variant_id':'v2'}]}
        (plan/'requests.jsonl').write_text(json.dumps(request)+'\n',encoding='utf8')
        f.save(plan/'summary.json',{'files_sha256':{'requests.jsonl':f.sha(plan/'requests.jsonl')}})
        gen.mkdir()
        rows=[{'variant_id':vid,'request_id':'r1','sample_id':'s1','answer':'answer','assigned_label':None} for vid in ['v1','v2']]
        (gen/'candidates.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf8')
        f.save(gen/'manifest.json',{'candidate_sha256':f.sha(gen/'candidates.jsonl'),'plan_sha256':f.sha(plan/'summary.json')})
        model=root/'models/text2vec-base-chinese'/f.REV
        model.mkdir(parents=True)
        for name in f.FILES:(model/name).write_text('{}',encoding='utf8')
        (model/'config.json').write_text(json.dumps({'max_position_embeddings':512}),encoding='utf8')
        f.save(model/'offline_manifest.json',{'repo_id':f.REPO,'revision':f.REV,'files_sha256':{n:f.sha(model/n) for n in f.FILES}})
        return gen

    def test_shared_vectors_preserve_rows_and_blank(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            texts,rows,facts=f.load_inputs(root)
            self.assertEqual(len(rows),4);self.assertEqual(len(texts),2)
            self.assertEqual(rows[0]['vector_row'],rows[2]['vector_row'])
            self.assertEqual(facts['blank_original_answers'],1)

    def test_tampering_and_invalid_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);gen=self.fixture(root)
            p=gen/'candidates.jsonl';p.write_text(p.read_text().replace('s1','unknown'))
            with self.assertRaisesRegex(ValueError,'hash'):f.load_inputs(root)
            manifest=f.read(gen/'manifest.json');manifest['candidate_sha256']=f.sha(p)
            (gen/'manifest.json').write_bytes(f.json_bytes(manifest))
            with self.assertRaisesRegex(ValueError,'provenance'):f.load_inputs(root)

    def test_vector_corruption_rejected(self):
        good=np.zeros((2,768),dtype='float32');good[:,0]=1
        f.validate_vectors(good,2)
        with self.assertRaises(ValueError):f.validate_vectors(good*0,2)
        with self.assertRaises(ValueError):f.validate_vectors(good[:,:3],2)

    def test_preflight_no_inference_and_long_input_gate(self):
        for token_count in [10,513]:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);self.fixture(root)
                class Tokenizer:
                    @staticmethod
                    def from_pretrained(path,**kw):
                        assert kw['local_files_only'] is True
                        return lambda texts,**kw:{'input_ids':[[1]*token_count for _ in texts]}
                modules={'torch':types.SimpleNamespace(__version__='fixture'),
                         'transformers':types.SimpleNamespace(__version__='fixture',AutoTokenizer=Tokenizer,AutoModel=None),
                         'fcntl':types.SimpleNamespace(LOCK_EX=2,LOCK_NB=4,flock=lambda *a:None)}
                with patch.dict('sys.modules',modules),contextlib.redirect_stdout(io.StringIO()):
                    if token_count>512:
                        with self.assertRaisesRegex(ValueError,'Overlength'):f.execute(root,'preflight')
                    else:
                        f.execute(root,'preflight');f.execute(root,'preflight')
                out=root/'paper_method_v1/artifacts/initial_filter_embeddings_v1'
                report=f.read(out/'preflight.json')
                self.assertEqual(report['api_calls'],0)
                self.assertFalse((out/'summary.json').exists())
                self.assertEqual(report['overlength_count'],2 if token_count>512 else 0)


if __name__=='__main__':unittest.main()
