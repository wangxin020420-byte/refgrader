import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from PIL import Image
import transcribe_csbench as m


class Tests(unittest.TestCase):
    def test_blind_retry_resume_and_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = root / 'inputs'
            inputs.mkdir()
            image = root / 'a.png'
            Image.new('RGB', (8, 8), 'white').save(image)
            row = dict(sample_id='a', question_id='q', answer='student error', blank_passthrough=False, student_image=str(image))
            blank = dict(row, sample_id='b', answer='', blank_passthrough=True)
            payload = {'train.inputs.jsonl': [row, blank], 'validation.inputs.jsonl': [], 'test.inputs.jsonl': [],
                       'nonblank_text_review_queue.jsonl': [row],
                       'questions.jsonl': [dict(question_id='q', question='SECRET_QUESTION', references=['SECRET_REFERENCE'], visual_context_review_required=False)]}
            hashes = {}
            for name, rows in payload.items():
                raw = ''.join(json.dumps(x) + '\n' for x in rows).encode()
                (inputs / name).write_bytes(raw)
                hashes[name] = m.sha(raw)
            m.write(inputs / 'manifest.json', {'files_sha256': hashes})
            # Invalid labels are deliberately present; the transcriber must not read them.
            (inputs / 'test.labels.jsonl').write_text('NOT JSON SECRET_GOLD')
            calls = []

            def create(**kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise TimeoutError()
                response = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"text":"student error", "issues":[]}'}}]}
                return SimpleNamespace(model_dump=lambda **kw: response)

            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            out = root / 'run'
            self.assertEqual(m.run(inputs, out, 0, 2, client, lambda _: None), 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual(m.read(out / 'plan.json')['base_url'], 'https://open.bigmodel.cn/api/coding/paas/v4/')
            sent = json.dumps(calls)
            self.assertNotIn('SECRET_', sent)
            self.assertEqual(m.read(out / 'plan.json')['blank_answers_skipped'], 1)
            self.assertEqual(m.run(inputs, out, 0, 2, client, lambda _: None), 0)
            self.assertEqual(len(calls), 2)
            self.assertFalse(m.read(out / 'summary.json')['ready_for_training'])
            (inputs / 'train.inputs.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                m.run(inputs, out, 0, 2, client, lambda _: None)

    def test_incomplete_response_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            m.parse_response({'choices': [{'finish_reason': 'length', 'message': {'content': ''}}]})


if __name__ == '__main__':
    unittest.main()
