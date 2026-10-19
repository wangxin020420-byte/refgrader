import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import entity_prepare as ep

STAGE = Path(__file__).resolve().parents[1] / 'scale-paper-stage1/stage1.py'


class EntityTests(unittest.TestCase):
    def test_parser_rejects_hallucinated_span_and_truncation(self):
        m = ep.module(STAGE)
        unit = {'unit_id': 'u', 'text': 'real phrase'}
        raw = {'choices': [{'finish_reason': 'stop', 'message': {'content':
            json.dumps({'triples': [{'t': 'invented', 'c': 'term', 'w': 'summary'}]})}}]}
        with self.assertRaisesRegex(ValueError, 'verbatim'):
            ep.parse(m, unit, raw)
        raw['choices'][0]['message']['content'] = '```json\n{"triples": []}\n```'
        self.assertEqual(ep.parse(m, unit, raw)['triples'], [])
        raw['choices'][0]['finish_reason'] = 'length'
        with self.assertRaisesRegex(ValueError, 'complete'):
            ep.parse(m, unit, raw)

    def test_end_to_end_cache_resume_and_label_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / 'control/scale-paper-stage1/stage1.py'
            stage.parent.mkdir(parents=True)
            shutil.copyfile(STAGE, stage)
            m = ep.module(stage)
            units = [{'unit_id': 'student:s%d' % i, 'role': 'student',
                      'question_id': 'q', 'text': 'original'} for i in range(885)]
            units.append({'unit_id': 'reference:r', 'role': 'reference',
                          'question_id': 'q', 'text': 'reference'})
            extractions = [{'unit_id': u['unit_id'], 'triples':
                [{'t': u['text'], 'c': 'term', 'w': u['text']}]} for u in units]
            rows = [{'id': 's%d' % i, 'sample_id': 's%d' % i, 'question_id': 'q',
                     'answer': 'original', 'references': ['reference', ''], 'label': 0.0,
                     'kind': 'original_training'} for i in range(885)]
            for i in range(1327):
                rows.append({'id': 'v%d' % i, 'sample_id': 's0', 'question_id': 'q',
                             'answer': ('new one' if i % 2 else 'new two') if i < 10 else 'original',
                             'references': ['reference', ''], 'label': -0.1,
                             'kind': 'synthetic_candidate'})
            trip = root / 'paper_method_v1/artifacts/triplets'
            ep.write(trip / 'source_units.jsonl', ep.jl(units))
            prompt_sha = ep.sha(m.PROMPT.encode('utf8'))
            ep.write(trip / 'stage1_manifest.json', ep.enc({'outputs_sha256':
                {'source_units.jsonl': ep.sha(ep.jl(units))}, 'prompt_sha256': prompt_sha}))
            full = root / 'paper_method_v1/artifacts/full_graph_v1'
            ep.write(full / 'extractions.jsonl', ep.jl(extractions))
            ep.write(full / 'graph.json', ep.enc({'extractions_sha256': ep.sha(ep.jl(extractions))}))
            ep.write(root / 'paper_method_v1/runs/pe_glm_full_v1/config.json', ep.enc({
                'model': 'test', 'base_url': 'https://example.invalid', 'max_tokens': 4096,
                'prompt_sha256': prompt_sha}))
            score = root / 'paper_method_v1/runs/candidate_annotation_v1'
            data = ep.jl(rows)
            ep.write(score / 'raw_beta_0p2/augmented_inputs.jsonl', data)
            ep.write(score / 'raw_beta_0p2/selected.jsonl', b'')
            ep.write(score / 'summary.json', ep.enc({'status': 'SCORING_AND_BETA_BRANCHES_COMPLETE',
                'files_sha256': {'raw_beta_0p2/augmented_inputs.jsonl': ep.sha(data),
                                'raw_beta_0p2/selected.jsonl': ep.sha(b'')}}))
            out = root / 'paper_method_v1/runs/train_entities_raw_beta_v1'
            out.mkdir(parents=True)
            args = argparse.Namespace(root=root, command='offline', max_attempts=2)

            class Response:
                def __init__(self, content='{"triples": []}'):
                    self.content = content
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def read(self):
                    return ep.enc({'choices': [{'finish_reason': 'stop',
                        'message': {'content': self.content}}]})

            with patch.object(ep, 'EXPECTED', ep.sha(data)), patch('builtins.print'):
                self.assertEqual(ep.execute(args, out), 2)
                self.assertEqual(ep.read(out / 'summary.json')['remaining'], 2)
                args.command = 'run'
                args.max_attempts = 1
                with patch.object(ep.getpass, 'getpass', return_value='not-real'), \
                     patch.object(ep.urllib.request, 'urlopen', side_effect=[Response('bad JSON'), Response()]) as network, \
                     patch.object(ep.time, 'sleep'):
                    self.assertEqual(ep.execute(args, out), 2)
                    self.assertEqual(network.call_count, 2)
                with patch.object(ep.urllib.request, 'urlopen', side_effect=AssertionError('Attempt cap exceeded')):
                    self.assertEqual(ep.execute(args, out), 2)
                args.max_attempts = 2
                with patch.object(ep.getpass, 'getpass', return_value='not-real'), \
                     patch.object(ep.urllib.request, 'urlopen', return_value=Response()) as network, \
                     patch.object(ep.time, 'sleep'):
                    self.assertEqual(ep.execute(args, out), 0)
                    self.assertEqual(network.call_count, 1)
                exported = ep.lines(out / 'training_with_entities.jsonl')
                self.assertEqual(len(exported), 2212)
                self.assertEqual(exported[885]['label'], -0.1)
                self.assertEqual(exported[0]['reference_triples'][1], [])
                self.assertEqual(exported[-1]['answer_triples'], extractions[0]['triples'])
                with patch.object(ep.urllib.request, 'urlopen', side_effect=AssertionError('Unexpected API')):
                    self.assertEqual(ep.execute(args, out), 0)
                self.assertEqual(ep.read(out / 'summary.json')['new_api_calls'], 0)
                self.assertEqual((score / 'raw_beta_0p2/augmented_inputs.jsonl').read_bytes(), data)
                args.command = 'offline'
                self.assertEqual(ep.execute(args, out), 0)
                # Changed frozen code/config cannot silently mix experiments.
                ep.write(out / 'config.json', b'{}')
                with self.assertRaisesRegex(ValueError, 'Frozen artifact'):
                    ep.execute(args, out)


if __name__ == '__main__':
    unittest.main()
