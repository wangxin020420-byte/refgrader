import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from PIL import Image
import transcribe_csbench_coding_v2 as m
import test_transcription as previous

previous.m = m


class RecoveryTests(unittest.TestCase):
    def test_literal_newline_and_invalid_json(self):
        def envelope(text, finish='stop'):
            return {'choices': [{'finish_reason': finish, 'message': {'content': text}}]}
        self.assertEqual(m.parse_response(envelope('{"text":"a\nb", "issues":[]}'))['text'], 'a\nb')
        with self.assertRaises(ValueError):
            m.parse_response(envelope('{"text":"a\nb", "issues":[]}}'))
        with self.assertRaises(ValueError):
            m.parse_response(envelope('{"text":"a\nb", "issues":[]}', 'length'))

    def test_recover_then_400_then_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = root / 'inputs'
            inputs.mkdir()
            img = root / 'a.png'
            Image.new('RGB', (8, 8)).save(img)
            rows = [dict(sample_id=str(i), question_id='q', answer='own', blank_passthrough=False, student_image=str(img)) for i in range(3)]
            data = {'train.inputs.jsonl': rows, 'validation.inputs.jsonl': [], 'test.inputs.jsonl': [],
                    'questions.jsonl': [], 'nonblank_text_review_queue.jsonl': rows}
            hashes = {}
            for n, values in data.items():
                b = ''.join(json.dumps(v) + '\n' for v in values).encode()
                (inputs / n).write_bytes(b)
                hashes[n] = m.sha(b)
            m.write(inputs / 'manifest.json', {'files_sha256': hashes})
            out = root / 'run'
            p = m.plan(inputs, out)
            plan_bytes = (out / 'plan.json').read_bytes()
            folder = out / 'responses' / m.sha(p['units'][0]['unit_id'].encode())
            folder.mkdir(parents=True)
            response = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"text":"a\nb", "issues":[]}'}}]}
            m.write(folder / '1.response.json', response)
            raw_before = (folder / '1.response.json').read_bytes()
            calls = []
            def create(**kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    e = RuntimeError('do not log whole exception')
                    e.status_code = 400
                    e.body = {'error': {'code': 'sample', 'message': 'test rejection'}}
                    raise e
                return SimpleNamespace(model_dump=lambda **kw: response)
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            self.assertEqual(m.run(inputs, out, 0, 1, client, lambda _: None), 2)
            s = m.read(out / 'summary.json')
            self.assertEqual((s['valid_responses'], s['remaining'], s['new_api_calls']), (2, 1, 2))
            self.assertEqual(s['errors'][0]['service_code'], 'sample')
            self.assertEqual((out / 'plan.json').read_bytes(), plan_bytes)
            self.assertEqual((folder / '1.response.json').read_bytes(), raw_before)
            accepted_before = (folder / 'accepted.json').read_bytes()
            self.assertEqual(m.run(inputs, out, 0, 1, client, lambda _: None), 0)
            self.assertEqual(len(calls), 3)
            self.assertEqual((folder / 'accepted.json').read_bytes(), accepted_before)


if __name__ == '__main__':
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(previous.Tests),
        unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests)])
    result = unittest.TextTestRunner().run(suite)
    raise SystemExit(not result.wasSuccessful())
