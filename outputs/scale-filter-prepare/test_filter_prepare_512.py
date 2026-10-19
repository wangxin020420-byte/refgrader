import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import filter_prepare_512 as f
from test_filter_prepare import Checks


class Tokenizer:
    truncation_side = 'left'
    def __call__(self, texts, add_special_tokens=True, truncation=False,
                 max_length=None, **kwargs):
        def encode(text):
            ids=list(range(1000,1527)) if text else []
            if truncation:
                assert self.truncation_side=='right'
                ids=ids[:max_length-(2 if add_special_tokens else 0)]
            return [101]+ids+[102] if add_special_tokens else ids
        return {'input_ids': [encode(x) for x in texts] if isinstance(texts,list) else encode(texts)}
    def num_special_tokens_to_add(self,pair=False):return 2
    def build_inputs_with_special_tokens(self,ids):return [101]+ids+[102]
    @staticmethod
    def from_pretrained(path,**kwargs):
        assert kwargs['local_files_only']
        return Tokenizer()


class LengthChecks(unittest.TestCase):
    def test_batch_right_truncation_preserves_short_special_tokens(self):
        tokenizer=Tokenizer()
        encoded=f.encode_batch(tokenizer,['answer',''])['input_ids']
        self.assertEqual(len(encoded[0]),512)
        self.assertEqual(encoded[0],[101]+list(range(1000,1510))+[102])
        self.assertEqual(encoded[1],[101,102])

    def test_preflight_audit_and_repeat_preserve_original_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);Checks().fixture(root)
            initial={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            modules={'torch':types.SimpleNamespace(__version__='fixture'),
                     'transformers':types.SimpleNamespace(__version__='fixture',AutoTokenizer=Tokenizer,AutoModel=None),
                     'fcntl':types.SimpleNamespace(LOCK_EX=2,LOCK_NB=4,flock=lambda *a:None)}
            with patch.dict('sys.modules',modules),contextlib.redirect_stdout(io.StringIO()):
                f.execute(root,'preflight');f.execute(root,'preflight')
            out=root/'paper_method_v1/artifacts/initial_filter_embeddings_v2_max512'
            report=f.read(out/'preflight.json');audit=f.read(out/'truncation_audit.json')
            self.assertEqual(report['status'],'FILTER_EMBEDDING_PREFLIGHT_OK_WITH_TRUNCATION')
            self.assertEqual(report['overlength_count'],1)
            self.assertEqual(audit[0]['omitted_tokens'],17)
            self.assertEqual(len(audit[0]['affected_records']),3)
            self.assertEqual(audit[0]['omitted_token_ids'],list(range(1510,1527)))
            self.assertFalse((out/'summary.json').exists())
            self.assertTrue(all(p.read_bytes()==b for p,b in initial.items()))


if __name__=='__main__':unittest.main()
