import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import score_and_filter as s


class Tensor:
    def __init__(self, data): self.data = data
    def to(self, *args): return self
    def reshape(self, *args): return self
    def float(self): return self
    def cpu(self): return self
    def tolist(self): return self.data


class Tokenizer:
    def __call__(self, a, b, **kwargs):
        if isinstance(a, str): return {'input_ids': [1] * (len(a) + len(b) + 3)}
        return {'input_ids': Tensor([1] * len(a))}


class Model:
    config = types.SimpleNamespace(num_labels=1)
    def to(self, *args): return self
    def eval(self): return self
    def __call__(self, input_ids): return types.SimpleNamespace(logits=Tensor([5.0] * len(input_ids.data)))


class Tests(unittest.TestCase):
    def test_thresholds_and_labels(self):
        sources = [dict(sample_id='s', question_id='q', q='q', r=['r'], a='original', l=5)]
        scored = [dict(variant_id=str(i), sample_id='s', answer='a', assigned_label=5+d, absolute_gap_raw=d)
                  for i,d in enumerate([0, 0.2, 0.200001, 2, 2.00001])]
        a, augmented = s.branch(scored, sources, .2)
        b, _ = s.branch(scored, sources, 2)
        self.assertEqual([r['variant_id'] for r in a], ['0','1'])
        self.assertEqual(len(b), 4)
        self.assertEqual(augmented[-1]['label'], 5.2)
        self.assertEqual(augmented[0]['label'], 5)

    def test_full_mock_pipeline_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td);base=root/'paper_method_v1';mr=base/'runs/annotator_bert_v1';final=mr/'model_final'
            def put(path, obj): s.write(path, s.enc(obj))
            put(final/'config.json', {'fixture':True})
            sources=[dict(sample_id='ADS:fixture:train:'+str(i), question_id='q', q='q', r=['r','','r2'], a='a', l=i%11) for i in range(885)]
            source_path=base/'artifacts/proxy_inputs_v1/training_sources.json';put(source_path,sources)
            proxies=[dict(sample_id=r['sample_id'], references=r['r'], answer='a',source_score=r['l']) for r in sources]
            proxy_path=base/'artifacts/annotator_inputs_v1/train.jsonl';s.write(proxy_path,s.jl(proxies))
            cfg={'max_length':512,'loss':'MSE on original ADS 0-10 scores; no normalization or clipping',
                 'reference_policy':'Join all nonempty references in original slot order with newline; one row per source.', 'data_sha256':s.sha(proxy_path)}
            put(mr/'config.json',cfg)
            put(mr/'complete.json',{'status':'ANNOTATOR_TRAINING_COMPLETE_FIXED_EPOCH','completed_epochs':10,
                'config_sha256':s.sha(mr/'config.json'),'final_files_sha256':{'config.json':s.sha(final/'config.json')}})
            selection=base/'artifacts/initial_filter_k5_b5000_v1';put(selection/'config.json',{'fixture':True})
            candidates=[dict(variant_id='v'+str(i),sample_id=sources[i%885]['sample_id'],answer='a',assigned_label=None) for i in range(5000)]
            s.write(selection/'selected_candidates.jsonl',s.jl(candidates))
            put(selection/'summary.json',{'status':'INITIAL_FILTER_COMPLETE_UNLABELED','config_sha256':s.sha(selection/'config.json'),
                'files_sha256':{'selected_candidates.jsonl':s.sha(selection/'selected_candidates.jsonl')}})
            original_hash=s.sha; before=s.sha(selection/'selected_candidates.jsonl')
            # Substitute fixture hash only for original training input; all other hashes are real.
            def fixture_hash(path):
                return '5dcff9d092ee292d5be2bfbe9bcafed2fb6fcd1cc82e02c0e2acbff7ae64fb5d' if path==source_path else original_hash(path)
            torch=types.SimpleNamespace(set_num_threads=lambda n:None,cuda=types.SimpleNamespace(is_available=lambda:True),
                inference_mode=contextlib.nullcontext,OutOfMemoryError=MemoryError)
            loads=[]
            def load_model(*args,**kwargs): loads.append(1);self.assertTrue(kwargs['local_files_only']);return Model()
            transformers=types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer()),
                BertForSequenceClassification=types.SimpleNamespace(from_pretrained=load_model))
            out=base/'runs/candidate_annotation_v1';args=types.SimpleNamespace(root=root,physical_gpu=1)
            with patch.object(s,'sha',side_effect=fixture_hash),patch.dict(sys.modules,{'torch':torch,'transformers':transformers}),contextlib.redirect_stdout(io.StringIO()):
                s.run(args,out)
                summary=s.read(out/'summary.json');self.assertEqual(summary['candidate_labels_assigned'],5000)
                self.assertEqual(summary['diagnostic_predictions'],1770)
                # Simulate interruption after exports but before final summary; cached chunks suffice.
                (out/'summary.json').unlink();s.run(args,out);s.run(args,out)
            self.assertEqual(len(loads),1)
            self.assertEqual(original_hash(selection/'selected_candidates.jsonl'),before)
            self.assertEqual(len(list((out/'chunks').glob('*.json'))),68)
            self.assertLess(summary['branches']['raw_beta_0p2']['retained_synthetic'],summary['branches']['normalized_beta_0p2']['retained_synthetic'])
            self.assertIsNone(summary['official_branch_selected'])


if __name__=='__main__': unittest.main()
